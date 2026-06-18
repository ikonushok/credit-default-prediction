"""CLI: aggregate long-format history -> per-id feature tables.

Usage:
    python scripts/02_aggregate.py [--feature-set baseline] [--nrows N]

--nrows limits how many leading rows of each parquet are read (smoke testing).
"""
import argparse

import _bootstrap  # noqa: F401

from credit_scoring import aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="baseline")
    ap.add_argument("--nrows", type=int, default=None,
                    help="read only the first N rows of each parquet (smoke test)")
    args = ap.parse_args()
    aggregate.run(feature_set=args.feature_set, nrows=args.nrows)


if __name__ == "__main__":
    main()
