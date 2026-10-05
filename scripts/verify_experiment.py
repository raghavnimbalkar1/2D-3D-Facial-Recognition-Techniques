"""Certify a full matrix against current data, persisted splits and artifacts."""

import argparse
import json
from pathlib import Path

from ivafr.config import ConfigResolver
from ivafr.integrity import atomic_json
from ivafr.pipelines.provenance import verify_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()
    try:
        exp = ConfigResolver(args.configs).experiment(args.exp)
        report = verify_experiment(exp, Path(args.data_root), Path(args.results_root))
    except (ValueError, OSError, AssertionError, KeyError) as exc:
        parser.exit(1, f"VERIFICATION FAILED: {exc}\n")
    atomic_json(Path(args.results_root) / f"validation_{args.exp}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
