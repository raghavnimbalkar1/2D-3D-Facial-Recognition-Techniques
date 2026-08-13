#!/usr/bin/env python
"""Build the toy dataset (deterministic). Equivalent to `ivafr dataset-build --name toy`."""

from __future__ import annotations

import argparse

from ivafr.datasets.toy import generate_toy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the toy 2D+3D face dataset")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--n-subjects", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_toy(args.raw_root, n_subjects=args.n_subjects, n_samples=args.n_samples, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()