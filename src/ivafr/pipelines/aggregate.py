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

log = get_logger("pipelines.aggregate")

T1_COLUMNS = [
    "Method",
    "Accuracy",
    "Rank-1 / Chance",
    "Precision",
    "Recall",
    "F1-Score",
    "Processing Time",
]


def collect_metrics(results_root: str | Path) -> pd.DataFrame:
    """Tidy frame: one row per (run_dir, exp, arm, protocol, seed)."""
    rows = []
    for metrics_file in sorted(Path(results_root).glob("runs/*/metrics.json")):
        with metrics_file.open("r", encoding="utf-8") as fh:
            m = json.load(fh)
        ident = m.get("identification", {})
        rows.append(
            {
                "run_dir": str(metrics_file.parent),
                "exp_id": m["exp_id"],
                "arm": m["arm"],
                "protocol": m["protocol"],
                "seed": m["seed"],
                "dataset": m.get("dataset", {}).get("name", "unknown"),
                "data_modality": m.get("data_modality", m.get("dataset", {}).get("data_modality", "unknown")),
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

    Processing Time is placeholder "NA" until E10 lands.
    """
    rows = []
    eval_df = df.loc[df["rank1"].notna()]
    for (dataset, data_modality, exp_id, arm, protocol), g in eval_df.groupby(["dataset", "data_modality", "exp_id", "arm", "protocol"]):
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
                "Processing Time": "NA",
                "protocol": protocol,
                "dataset": dataset,
                "data_modality": data_modality,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["dataset", "data_modality", *T1_COLUMNS, "protocol"],
    )


def render_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Extended table with Rank-5, EER, AUC, MRR."""
    rows = []
    eval_df = df.loc[df["rank1"].notna()]
    for (dataset, data_modality, exp_id, arm, protocol), g in eval_df.groupby(["dataset", "data_modality", "exp_id", "arm", "protocol"]):
        row = {"dataset": dataset, "data_modality": data_modality, "exp": exp_id, "arm": arm, "protocol": protocol}
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
    md = "| dataset | data_modality | " + " | ".join(T1_COLUMNS) + " |\n|---|---|" + "---|" * len(T1_COLUMNS) + "\n"
    for _, r in t1.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in ["dataset", "data_modality", *T1_COLUMNS]) + " |\n"
    (out / "T1_main_comparison.md").write_text(md, encoding="utf-8")
    log.info("Tables written to %s", out)


def write_results_md(df: pd.DataFrame, out_root: str | Path, preamble: str | Path | None = None) -> None:
    """Auto-generate RESULTS.md from collected metrics."""
    out = Path(out_root)
    lines = [
        "# RESULTS",
        "",
        f"Generated from {len(df)} run directories on `{df['protocol'].nunique() if len(df) else 0}` protocols.",
        "",
    ]
    if preamble is not None and Path(preamble).is_file():
        lines += [
            "<!-- preamble: docs/RESULTS_PREAMBLE.md — preserved verbatim on regeneration -->",
            "",
            Path(preamble).read_text(encoding="utf-8").strip(),
            "",
        ]
    lines += [
        "## Data separation and limitations",
        "",
        "Real Yale B rows and synthetic toy rows are kept separate by the `data_modality` field.",
        "The toy pseudo-3D and fusion results are synthetic methodology validation only.",
        "Real-data pseudo-3D is environment-blocked by the documented MediaPipe native runtime failure; no real 2D-versus-3D claim is made.",
        "A prior Yale B checkpoint was invalidated after a CLAHE float-to-uint8 conversion bug collapsed cached crops; the cache was rebuilt and the conversion is regression-tested.",
        "Aggregated tables report Rank-1 divided by the protocol's chance baseline (`1 / n_gallery`) as `Rank-1 / Chance`.",
        "",
        "## T1 — Main comparison (mean ± std over seeds)",
        "",
        (out / "tables" / "T1_main_comparison.md").read_text(encoding="utf-8") if (out / "tables" / "T1_main_comparison.md").is_file() else "",
        "## Extended metrics (Rank-5, EER, AUC, MRR)",
        "",
    ]
    ext = render_extended(df)
    md = "| dataset | data_modality | exp | arm | protocol | rank1 | chance_rank1 | rank1_over_chance | rank5 | mrr | eer | auc |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in ext.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in ("dataset", "data_modality", "exp", "arm", "protocol", "rank1", "chance_rank1", "rank1_over_chance", "rank5", "mrr", "eer", "auc")) + " |\n"
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
        preamble: optional markdown file injected into RESULTS.md verbatim
            (e.g. the current-status narrative), so regeneration never drops
            hand-written content.
    """
    df = collect_metrics(results_root)
    if df.empty:
        return
    write_tables(df, out_root)
    write_results_md(df, out_root, preamble=preamble)
