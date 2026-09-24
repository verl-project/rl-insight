# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line entry point for one-shot offline degradation analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .data import DEFAULT_TSDB_DIR, OfflineInputError, WindowError
from .offline import OfflineAnalysisError, StateError, analyze_offline


def _timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise argparse.ArgumentTypeError(
                "time must be a Unix timestamp or ISO-8601 value with timezone"
            )
        return parsed.timestamp()


def _range_name(start_time: float, end_time: float) -> str:
    def stamp(value: float) -> str:
        return dt.datetime.fromtimestamp(value, dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    return f"{stamp(start_time)}_{stamp(end_time)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recipe.perf_analysis.degradation.cli",
        description="Analyze one time range from the local RL-Insight TSDB.",
    )
    parser.add_argument("command", choices=("analyze",))
    parser.add_argument(
        "--start-time",
        required=True,
        type=_timestamp,
        help="Unix seconds or an ISO-8601 timestamp with timezone.",
    )
    parser.add_argument(
        "--end-time",
        required=True,
        type=_timestamp,
        help="Unix seconds or an ISO-8601 timestamp with timezone.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_TSDB_DIR,
        help="Prometheus TSDB directory (default: ~/.rl-insight/data/prometheus).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Root directory for one report folder per selected time range.",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Load this baseline, or create it from the first 30 complete steps.",
    )
    parser.add_argument("--promtool", type=Path, help="Explicit promtool path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.start_time >= args.end_time:
        parser.error("--start-time must be earlier than --end-time")
    report_dir = args.analysis_dir.expanduser() / _range_name(
        args.start_time, args.end_time
    )
    baseline = (
        args.baseline_file.expanduser()
        if args.baseline_file is not None
        else report_dir / "standard_data.json"
    )
    output = report_dir / "abnormal_data.json"
    try:
        result = analyze_offline(
            args.data_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            baseline_path=baseline,
            output_path=output,
            promtool_path=args.promtool,
        )
    except (
        OfflineAnalysisError,
        OfflineInputError,
        StateError,
        WindowError,
        TypeError,
        ValueError,
    ) as exc:
        error_kind = (
            "offline_analysis_error"
            if isinstance(exc, OfflineAnalysisError)
            else "invalid_input_or_state"
        )
        print(
            json.dumps(
                {"status": "error", "error_kind": error_kind, "message": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    report_data_path = report_dir / "analysis.json"
    result["analysis"]["report_data_path"] = str(report_data_path.resolve())
    result["analysis"]["report_path"] = str((report_dir / "report.md").resolve())
    report_data_path.parent.mkdir(parents=True, exist_ok=True)
    report_data_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
