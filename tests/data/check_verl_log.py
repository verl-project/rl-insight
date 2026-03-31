#!/usr/bin/env python3
"""Run RL-Insight VERL_LOG DataChecker on a single VeRL .log file.

Lives under rl-insight/tests/data/; can be run without pip install -e:

    python tests/data/check_verl_log.py /path/to/file.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_rl_insight_on_path() -> None:
    # tests/data -> tests -> rl-insight repo root (contains package rl_insight/)
    pkg_root = Path(__file__).resolve().parent.parent.parent
    s = str(pkg_root)
    if (
        pkg_root.is_dir()
        and (pkg_root / "rl_insight").is_dir()
        and s not in sys.path
    ):
        sys.path.insert(0, s)


def main() -> int:
    _ensure_rl_insight_on_path()

    from rl_insight.data import DataChecker, DataEnum
    from rl_insight.data.rules import DataValidationError

    parser = argparse.ArgumentParser(
        description="Validate one path with DataChecker(DataEnum.VERL_LOG, ...)."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to one non-empty .log file (VeRL training log)",
    )
    args = parser.parse_args()

    target = Path(args.path).expanduser()
    if not target.exists():
        print(f"ERROR: path does not exist: {target}", file=sys.stderr)
        return 2
    if not target.is_file():
        print(
            "ERROR: VERL_LOG validation expects a single .log file, not a directory.",
            file=sys.stderr,
        )
        return 2

    try:
        DataChecker(DataEnum.VERL_LOG, str(target)).run()
    except DataValidationError as e:
        print("VERL_LOG validation FAILED", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    print("VERL_LOG validation OK")
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
