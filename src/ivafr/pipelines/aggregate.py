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

T1_COLUMNS = ["Method", "Accuracy", "Precision", "Recall", "F1-Score", "Processing Time"]


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
    if df.empty:
        log.warning("No metrics.json found under %s/runs — run an experiment first", results_root)
    return df


def summarized(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over seeds, per (exp_id, arm, protocol)."""
    group = ["dataset", "data_modality", "exp_id", "arm", "protocol"]
    out = df.groupby(group, dropna=False)[
        ["rank1", "rank5", "accuracy", "precision_macro", "recall_macro", "f1_macro", "mrr", "eer", "auc"]
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
        rows.append(
            {
                "Method": f"{arm} ({exp_id})",
                "Accuracy": f"{m[0]:.4f}±{m[1]:.4f}",
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
        for col in ("rank1", "rank5", "mrr", "eer", "auc"):
            m, s = mean_std(g[col].tolist())
            row[col] = f"{m:.4f}±{s:.4f}"
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


def write_results_md(df: pd.DataFrame, out_root: str | Path) -> None:
    """Auto-generate RESULTS.md from collected metrics."""
    out = Path(out_root)
    lines = [
        "# RESULTS",
        "",
        f"Generated from {len(df)} run directories on `{df['protocol'].nunique() if len(df) else 0}` protocols.",
        "",
        "## Data separation and limitations",
        "",
        "Real Yale B rows and synthetic toy rows are kept separate by the `data_modality` field.",
        "The toy pseudo-3D and fusion results are synthetic methodology validation only.",
        "Real-data pseudo-3D is environment-blocked by the documented MediaPipe native runtime failure; no real 2D-versus-3D claim is made.",
        "",
        "## T1 — Main comparison (mean ± std over seeds)",
        "",
        (out / "tables" / "T1_main_comparison.md").read_text(encoding="utf-8") if (out / "tables" / "T1_main_comparison.md").is_file() else "",
        "## Extended metrics (Rank-5, EER, AUC, MRR)",
        "",
    ]
    ext = render_extended(df)
    md = "| dataset | data_modality | exp | arm | protocol | rank1 | rank5 | mrr | eer | auc |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in ext.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in ("dataset", "data_modality", "exp", "arm", "protocol", "rank1", "rank5", "mrr", "eer", "auc")) + " |\n"
    lines.append(md)
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("RESULTS.md -> %s", out / "RESULTS.md")


def aggregate(results_root: str | Path = "results", out_root: str | Path = "results") -> None:
    """Collect all run metrics and emit tables + RESULTS.md."""
    df = collect_metrics(results_root)
    if df.empty:
        return
    write_tables(df, out_root)
    write_results_md(df, out_root)
