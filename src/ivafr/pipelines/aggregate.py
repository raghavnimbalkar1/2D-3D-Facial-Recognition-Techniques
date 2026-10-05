"""Aggregate all run metrics.json files into report tables + RESULTS.md.

Globs ``results/runs/*/metrics.json``, computes mean +- std across seeds per
(exp, arm, protocol), writes ``results/tables/T*.csv`` and a top-level
``results/RESULTS.md`` summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ivafr.evaluation.metrics import mean_std
from ivafr.logging_utils import get_logger
from ivafr.pipelines.provenance import validate_run

log = get_logger("pipelines.aggregate")

T1_COLUMNS = [
    "Method",
    "Accuracy",
    "Rank-1 / Chance",
    "Precision",
    "Recall",
    "F1-Score",
    "Processing Time",
    "protocol",
    "Seeds",
]


def collect_metrics(results_root: str | Path) -> pd.DataFrame:
    """Tidy frame: one row per (run_dir, exp, arm, protocol, seed)."""
    rows = []
    seen = set()
    families = {}
    for directory in Path(results_root).glob("runs/*"):
        if directory.is_dir() and not (directory / "completed.json").is_file():
            raise ValueError(f"Incomplete run directory: {directory}")
    for metrics_file in sorted(Path(results_root).glob("runs/*/metrics.json")):
        m = validate_run(metrics_file.parent)
        key = (m["exp_id"], m["arm"], m["protocol"], m["seed"])
        if key in seen:
            raise ValueError(f"Duplicate logical run: {key}")
        seen.add(key)
        family = {
            k: v
            for k, v in m["provenance"].items()
            if k not in {"arm", "protocol", "seed", "split_hash", "manifest_hash"}
        }
        if m["exp_id"] in families and families[m["exp_id"]] != family:
            raise ValueError(f"Incompatible experiment fingerprints: {m['exp_id']}")
        families[m["exp_id"]] = family
        ident = m.get("identification", {})
        rows.append(
            {
                "run_dir": str(metrics_file.parent),
                "exp_id": m["exp_id"],
                "arm": m["arm"],
                "protocol": m["protocol"],
                "seed": m["seed"],
                "dataset": m.get("dataset", {}).get("name", "unknown"),
                "data_modality": m.get(
                    "data_modality", m.get("dataset", {}).get("data_modality", "unknown")
                ),
                "rank1": ident.get("rank1", float("nan")),
                "gallery_size": ident.get("n_gallery", float("nan")),
                "rank5": ident.get("rank5", float("nan")),
                "accuracy": ident.get("accuracy", float("nan")),
                "precision_macro": ident.get("precision_macro", float("nan")),
                "recall_macro": ident.get("recall_macro", float("nan")),
                "f1_macro": ident.get("f1_macro", float("nan")),
                "mrr": ident.get("mrr", float("nan")),
                "eer": m.get("verification", {}).get("eer", float("nan")),
                "auc": m.get("verification", {}).get("auc", float("nan")),
                "ms_per_probe": m.get("timing", {}).get("ms_per_probe", float("nan")),
                "run_id": m["run_id"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["chance_rank1"] = 1.0 / df["gallery_size"]
        df["rank1_over_chance"] = df["rank1"] / df["chance_rank1"]
    if df.empty:
        log.warning("No metrics.json found under %s/runs — run an experiment first", results_root)
    return df


def summarized(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over seeds, per (exp_id, arm, protocol)."""
    group = ["dataset", "data_modality", "exp_id", "arm", "protocol"]
    out = df.groupby(group, dropna=False)[
        [
            "rank1",
            "chance_rank1",
            "rank1_over_chance",
            "rank5",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "mrr",
            "eer",
            "auc",
        ]
    ].agg(["mean", "std"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def render_t1(df: pd.DataFrame) -> pd.DataFrame:
    """Doc-mandated T1 layout: Method | Accuracy | Precision | Recall | F1 | Time.

    Processing time is reported only for measured stages.
    """
    rows = []
    eval_df = df.loc[df["rank1"].notna()]
    for (dataset, data_modality, exp_id, arm, protocol), g in eval_df.groupby(
        ["dataset", "data_modality", "exp_id", "arm", "protocol"]
    ):
        m, s = mean_std(g["rank1"].tolist()), mean_std(g["precision_macro"].tolist())
        r, f = mean_std(g["recall_macro"].tolist()), mean_std(g["f1_macro"].tolist())
        c = mean_std(g["rank1_over_chance"].dropna().tolist())
        rows.append(
            {
                "Method": f"{arm} ({exp_id})",
                "Accuracy": f"{m[0]:.4f}±{m[1]:.4f}",
                "Rank-1 / Chance": f"{c[0]:.2f}x±{c[1]:.2f}x" if c[0] == c[0] else "NA",
                "Precision": f"{s[0]:.4f}±{s[1]:.4f}",
                "Recall": f"{r[0]:.4f}±{r[1]:.4f}",
                "F1-Score": f"{f[0]:.4f}±{f[1]:.4f}",
                "Processing Time": (
                    f"{g['ms_per_probe'].mean():.4f} ms/probe"
                    if g["ms_per_probe"].notna().all()
                    else "not measured"
                ),
                "Seeds": int(g["seed"].nunique()),
                "protocol": protocol,
                "dataset": dataset,
                "data_modality": data_modality,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["dataset", "data_modality", *T1_COLUMNS],
    )


def render_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Extended table with Rank-5, EER, AUC, MRR."""
    rows = []
    eval_df = df.loc[df["rank1"].notna()]
    for (dataset, data_modality, exp_id, arm, protocol), g in eval_df.groupby(
        ["dataset", "data_modality", "exp_id", "arm", "protocol"]
    ):
        row = {
            "dataset": dataset,
            "data_modality": data_modality,
            "exp": exp_id,
            "arm": arm,
            "protocol": protocol,
        }
        for col in ("rank1", "chance_rank1", "rank1_over_chance", "rank5", "mrr", "eer", "auc"):
            m, s = mean_std(g[col].tolist())
            suffix = "x" if col == "rank1_over_chance" else ""
            row[col] = f"{m:.4f}{suffix}±{s:.4f}{suffix}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, out_root: str | Path) -> None:
    """Write T1 + extended tables as CSV and Markdown."""
    out = Path(out_root) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t1 = render_t1(df)
    t1.to_csv(out / "T1_main_comparison.csv", index=False)
    ext = render_extended(df)
    ext.to_csv(out / "T1_extended.csv", index=False)
    verification_rows = []
    for run_dir in df["run_dir"]:
        metrics = validate_run(Path(run_dir))
        ver = metrics.get("verification")
        if ver is None:
            continue
        for far, point in ver["operating_points"].items():
            verification_rows.append(
                {
                    "experiment": metrics["exp_id"],
                    "arm": metrics["arm"],
                    "protocol": metrics["protocol"],
                    "seed": metrics["seed"],
                    "eer": ver["eer"],
                    "eer_ci95_low": ver["eer_ci95"][0],
                    "eer_ci95_high": ver["eer_ci95"][1],
                    "auc": ver["auc"],
                    "ci_method": ver["ci_method"],
                    "n_genuine": ver["n_genuine"],
                    "n_impostor": ver["n_impostor"],
                    "requested_far": float(far),
                    **point,
                }
            )
    if verification_rows:
        pd.DataFrame(verification_rows).to_csv(out / "T3_verification.csv", index=False)
    md = (
        "| dataset | data_modality | "
        + " | ".join(T1_COLUMNS)
        + " |\n|---|---|"
        + "---|" * len(T1_COLUMNS)
        + "\n"
    )
    for _, r in t1.iterrows():
        md += (
            "| " + " | ".join(str(r[c]) for c in ["dataset", "data_modality", *T1_COLUMNS]) + " |\n"
        )
    (out / "T1_main_comparison.md").write_text(md, encoding="utf-8")
    log.info("Tables written to %s", out)


def write_results_md(
    df: pd.DataFrame, out_root: str | Path, preamble: str | Path | None = None
) -> None:
    """Auto-generate RESULTS.md from collected metrics."""
    out = Path(out_root)
    lines = [
        "# RESULTS",
        "",
        f"Generated from {len(df)} run directories on `{df['protocol'].nunique() if len(df) else 0}` protocols.",
        "",
    ]
    # Completion claims are derived from artifacts, never copied from a narrative.
    lines += ["## Validation status", ""]
    for exp_id, group in df.groupby("exp_id"):
        config = json.loads(
            (Path(group.iloc[0]["run_dir"]) / "config_resolved.json").read_text(encoding="utf-8")
        )
        expected = {
            (a["key"], p, seed)
            for a in config["arms"]
            for p in config["protocols"]
            for seed in config["seeds"]
        }
        present = set(zip(group.arm, group.protocol, group.seed))
        complete = present == expected
        validation_path = Path(group.iloc[0]["run_dir"]).parent.parent / f"validation_{exp_id}.json"
        audited = False
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            audited = validation.get("complete") is True and set(
                validation.get("run_ids", [])
            ) == set(group.run_id)
        lines.append(
            f"- {exp_id}: {len(present)}/{len(expected)} unique runs; matrix {'complete' if complete else 'INCOMPLETE'}; current-input audit {'recorded' if audited else 'not recorded'}."
        )
    lines += [
        "",
        "Synthetic results are methodology checks. P1 uses a fixed canonical gallery; its seeds resample verification pairs, not independent identification splits.",
        "",
        "## T1 — Main comparison (mean ± std over seeds)",
        "",
        (
            (out / "tables" / "T1_main_comparison.md").read_text(encoding="utf-8")
            if (out / "tables" / "T1_main_comparison.md").is_file()
            else ""
        ),
        "## Extended metrics (Rank-5, EER, AUC, MRR)",
        "",
    ]
    ext = render_extended(df)
    md = "| dataset | data_modality | exp | arm | protocol | rank1 | chance_rank1 | rank1_over_chance | rank5 | mrr | eer | auc |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in ext.iterrows():
        md += (
            "| "
            + " | ".join(
                str(r[c])
                for c in (
                    "dataset",
                    "data_modality",
                    "exp",
                    "arm",
                    "protocol",
                    "rank1",
                    "chance_rank1",
                    "rank1_over_chance",
                    "rank5",
                    "mrr",
                    "eer",
                    "auc",
                )
            )
            + " |\n"
        )
    lines.append(md)
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("RESULTS.md -> %s", out / "RESULTS.md")


def aggregate(
    results_root: str | Path = "results",
    out_root: str | Path = "results",
    preamble: str | Path | None = None,
) -> None:
    """Collect all run metrics and emit tables + RESULTS.md.

    Args:
        results_root: directory whose ``runs/`` subtree contains metrics.json.
        out_root: where tables/ and RESULTS.md land.
        preamble: deprecated compatibility argument, ignored. Completion
            statements are derived from the validated artifacts.
    """
    df = collect_metrics(results_root)
    if df.empty:
        raise ValueError("No completed experiment runs to aggregate")
    write_tables(df, out_root)
    write_results_md(df, out_root, preamble=preamble)
