"""ivafr command-line interface.

Reproduction path (matches the plan's CLI):

    ivafr dataset-build --name toy            # create the toy dataset
    ivafr ingest --dataset toy
    ivafr preprocess --dataset toy --modality both
    ivafr splits --dataset toy --protocol P1_closed P2_disjoint --seeds 0 1 2 3 4
    ivafr run --exp E00       # extract + match + evaluate
    ivafr robustness --exp ...   # (M5)
    ivafr timing --exp ...        # (M6)
    ivafr aggregate --out results
"""

from __future__ import annotations

from pathlib import Path

import typer

from ivafr.config import ConfigResolver
from ivafr.logging_utils import get_logger, setup_logging
from ivafr.registry import list_datasets, list_features, list_matchers

app = typer.Typer(help="ivafr — 2D vs 3D facial recognition benchmark", no_args_is_help=True)
log = get_logger("cli")

DEFAULT_CONFIGS = "configs"
DEFAULT_DATA = "data"
DEFAULT_RESULTS = "results"


def _resolver(configs_root: str) -> ConfigResolver:
    return ConfigResolver(configs_root)


@app.command()
def dataset_build(
    name: str = typer.Option("toy", help="Dataset to build (toy for now)"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
    n_subjects: int = typer.Option(12, help="Toy subjects"),
    n_samples: int = typer.Option(15, help="Toy captures per subject"),
    size: int = typer.Option(160, help="Toy image size (px)"),
    seed: int = typer.Option(0, help="Toy generator seed"),
) -> None:
    """Generate the procedural toy dataset under data/raw/toy."""
    from ivafr.datasets.toy import generate_toy

    generate_toy(Path(data_root) / "raw", n_subjects=n_subjects, n_samples=n_samples, size=size, seed=seed)


@app.command()
def ingest(
    dataset: str = typer.Option(..., help=f"Dataset name ({', '.join(list_datasets())})"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
    anonymize: bool = typer.Option(False, "--anonymize", help="Hash subject IDs"),
) -> None:
    """Stage 0: adapter -> manifest.csv + audit."""
    from ivafr.pipelines.ingest import ingest as run_ingest

    run_ingest(dataset, data_root, anonymize=anonymize)


@app.command()
def preprocess(
    dataset: str = typer.Option(...),
    modality: str = typer.Option("both", help="2d | 3d | both"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
    configs_root: str = typer.Option(DEFAULT_CONFIGS, "--configs"),
    cfg2d: str | None = typer.Option(None, help="Override the dataset's 2D config"),
    cfg3d: str | None = typer.Option(None, help="Override the dataset's 3D config"),
) -> None:
    """Stage 1/2: 2D + 3D preprocessing chains (content-hash cached)."""
    from ivafr.pipelines.preprocess_run import preprocess_dataset

    resolver = _resolver(configs_root)
    dataset_cfg = resolver.dataset_config(dataset)
    preprocess_dataset(
        dataset,
        data_root,
        resolver.preprocess_config(cfg2d or dataset_cfg.get("preprocess_2d", "p2d_default")),
        resolver.preprocess_config(cfg3d or dataset_cfg.get("preprocess_3d", "p3d_default")),
        modality=modality,
    )


@app.command()
def splits(
    dataset: str = typer.Option(...),
    protocol: list[str] = typer.Option(["P1_closed", "P2_disjoint"], "--protocol"),
    seeds: list[int] = typer.Option([0, 1, 2, 3, 4], "--seeds"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
) -> None:
    """Generate splits (P1 closed, P2 disjoint) + verification pairs."""
    from ivafr.datasets.manifest import read_manifest
    from ivafr.datasets.splits import make_split, summary, write_split

    manifest = read_manifest(Path(data_root) / "processed" / dataset / "manifest.csv")
    out = Path(data_root) / "processed" / dataset / "splits"
    for p in protocol:
        for s in seeds:
            split = make_split(manifest, p, seed=s)
            path = write_split(split, out)
            print(summary(split), "->", path)


@app.command()
def run(
    exp: list[str] = typer.Option(..., "--exp", help="Experiment ids, e.g. E00"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
    results_root: str = typer.Option(DEFAULT_RESULTS, "--results-root"),
    configs_root: str = typer.Option(DEFAULT_CONFIGS, "--configs"),
    force: bool = typer.Option(False, "--force", help="Re-run existing run dirs"),
    seed: list[int] | None = typer.Option(None, "--seed"),
    protocol: list[str] | None = typer.Option(None, "--protocol"),
) -> None:
    """Run experiments: extract + match + evaluate for each configured arm."""
    from ivafr.pipelines.run_experiment import run_experiment

    resolver = _resolver(configs_root)
    for e in exp:
        cfg = resolver.experiment(e)
        log.info("Resolved experiment %s (%s) on %s", cfg.id, cfg.name, cfg.dataset)
        run_experiment(
            cfg,
            data_root=data_root,
            results_root=results_root,
            force=force,
            seeds=seed,
            protocols=protocol,
        )


@app.command()
def robustness(
    exp: list[str] = typer.Option(..., "--exp"),
    configs_root: str = typer.Option(DEFAULT_CONFIGS, "--configs"),
    results_root: str = typer.Option(DEFAULT_RESULTS, "--results-root"),
) -> None:
    """Run the trimmed illumination/occlusion robustness artifact pass."""
    from ivafr.pipelines.run_robustness import write_sweep
    import json
    import pandas as pd

    runs = []
    for p in Path(results_root).glob("runs/*/metrics.json"):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    out = Path(results_root) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for exp_id in exp:
        rows = []
        for m in runs:
            if m.get("exp_id") != exp_id:
                continue
            for condition, vals in m.get("identification", {}).get("per_condition", {}).items():
                rows.append({"arm": m.get("arm"), "protocol": m.get("protocol"), "seed": m.get("seed"), "condition": condition, "rank1": vals.get("rank1"), "n": vals.get("n")})
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{exp_id}_conditions.csv", index=False)
        write_sweep(out / f"{exp_id}_sweep.json", exp_id, rows)
    typer.echo(f"Robustness artifacts written to {out}")


@app.command()
def timing(
    exp: list[str] = typer.Option(..., "--exp"),
    data_root: str = typer.Option(DEFAULT_DATA, "--data-root"),
    configs_root: str = typer.Option(DEFAULT_CONFIGS, "--configs"),
    results_root: str = typer.Option(DEFAULT_RESULTS, "--results-root"),
) -> None:
    """Run configured timing experiments and write a timing table."""
    import json
    import pandas as pd

    resolver = _resolver(configs_root)
    rows = []
    for exp_id in exp:
        exp_cfg = resolver.experiment(exp_id)
        from ivafr.pipelines.run_experiment import run_experiment

        run_experiment(exp_cfg, data_root=data_root, results_root=results_root)
        for metrics_file in sorted(Path(results_root).glob(f"runs/*_{exp_id}_*/metrics.json")):
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if "timing" in metrics:
                rows.append({
                    "experiment": exp_id,
                    "dataset": metrics.get("dataset", {}).get("name"),
                    "data_modality": metrics.get("data_modality"),
                    "arm": metrics.get("arm"),
                    "protocol": metrics.get("protocol"),
                    "seed": metrics.get("seed"),
                    **metrics["timing"],
                })
    out = Path(results_root) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "T5_timing.csv", index=False)
    (out / "T5_timing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    typer.echo(f"Timing table written to {out / 'T5_timing.csv'}")


@app.command()
def aggregate(
    results_root: str = typer.Option(DEFAULT_RESULTS, "--results-root"),
    out_root: str = typer.Option("results", "--out"),
) -> None:
    """Glob metrics.json -> tables/*.csv|md + RESULTS.md."""
    from ivafr.pipelines.aggregate import aggregate as run_aggregate

    run_aggregate(results_root, out_root)


@app.command()
def architecture() -> None:
    """List registered datasets, features and matchers."""
    typer.echo(f"datasets: {', '.join(list_datasets())}")
    typer.echo(f"features: {', '.join(list_features())}")
    typer.echo(f"matchers: {', '.join(list_matchers())}")


def main() -> None:
    setup_logging()
    app()


if __name__ == "__main__":
    main()
