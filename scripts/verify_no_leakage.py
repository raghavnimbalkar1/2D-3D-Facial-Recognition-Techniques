#!/usr/bin/env python
"""CI guard: assert zero train/probe leakage across every split on disk.

Checks, for each split JSON under data/processed/<dataset>/splits/:
  * gallery ∩ probe == empty
  * P2: train_subjects ∩ eval_subjects == empty
  * verification pairs contain no self-pairs and never span the P2
    background pool.
Exit code 0 = all clean; 1 = a violation (fails CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ivafr.datasets.manifest import read_manifest
from ivafr.datasets.splits import assert_no_leakage


def verify_splits_dir(splits_dir: str | Path, manifest_path: str | Path | None = None) -> list[str]:
    """Check every split in a directory; returns list of error messages."""
    errors: list[str] = []
    d = Path(splits_dir)
    if not d.is_dir():
        return [f"Split directory missing: {d}"]
    subject_of: dict[str, str] = {}
    if manifest_path is not None:
        df = read_manifest(manifest_path)
        subject_of = dict(zip(df["sample_id"].astype(str), df["subject_id"]))

    for split_file in sorted(d.glob("*_seed*.json")):
        with split_file.open("r", encoding="utf-8") as fh:
            split = json.load(fh)
        try:
            assert_no_leakage(split)
        except AssertionError as exc:
            errors.append(f"{split_file.name}: {exc}")
        if subject_of:
            pool = set(split["gallery_ids"]) | set(split["probe_ids"])
            for pid in pool:
                if pid not in subject_of:
                    errors.append(f"{split_file.name}: unknown id {pid}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    processed = Path(args.data_root) / "processed"
    errors: list[str] = []
    checked = 0
    if processed.is_dir():
        for splits_dir in sorted(processed.glob("*/splits")):
            manifest = splits_dir.parent / "manifest.csv"
            checked += len(list(splits_dir.glob("*_seed*.json")))
            errors += verify_splits_dir(splits_dir, manifest)
    if not checked:
        errors.append("No splits found — run `ivafr splits` first")

    for err in errors:
        print(f"LEAKAGE-VIOLATION: {err}", file=sys.stderr)
    print(f"Checked {checked} split files, {len(errors)} violations")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())