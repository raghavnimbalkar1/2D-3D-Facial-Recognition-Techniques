#!/usr/bin/env python
"""Dedupe run directories: keep the newest metrics.json per combination.

Run dirs created before the idempotency fix may contain duplicate
(exp, arm, protocol, seed) combinations. This script keeps the newest
directory per combination and moves the rest to ``<runs>/../archive_dupes/``
so aggregation never double-counts a seed.

Exit 0 = clean; 1 = nothing to dedupe or an error.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_PATTERN = re.compile(
    r"^(?P<ts>\d{8}T\d{12})_(?P<exp>.*?)_(?P<protocol>P\d_\w+)_s(?P<seed>\d+)_(?P<arm>.*)$"
)


def dedupe(runs_dir: str | Path, dry_run: bool = False) -> list[Path]:
    """Move older duplicates to ``archive_dupes``; returns moved paths."""
    runs = Path(runs_dir)
    archive = runs.parent / "archive_dupes"
    groups: dict[tuple[str, str, int, str], list[Path]] = {}
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        m = _PATTERN.match(d.name)
        if not m or not (d / "metrics.json").is_file():
            continue
        key = (m.group("exp"), m.group("protocol"), int(m.group("seed")), m.group("arm"))
        groups.setdefault(key, []).append(d)

    moved: list[Path] = []
    for key, dirs in sorted(groups.items()):
        if len(dirs) < 2:
            continue
        newest = max(dirs, key=lambda d: d.name)  # timestamp-sorted names
        for dup in dirs:
            if dup is newest:
                continue
            dest = archive / dup.name
            if dry_run:
                print(f"[dry] would move {dup.name}")
                moved.append(dup)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dup), str(dest))
                moved.append(dup)
                print(f"moved {dup.name} -> archive_dupes/")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moved = dedupe(args.runs_dir, dry_run=args.dry_run)
    print(f"{'would move' if args.dry_run else 'moved'} {len(moved)} duplicate dirs")
    return 0


if __name__ == "__main__":
    sys.exit(main())