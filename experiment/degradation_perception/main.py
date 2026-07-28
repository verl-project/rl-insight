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

"""One-shot command-line entry for degradation perception."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .algorithm import DegradationPerception
from .perception_config import DEFAULT_METRIC, SUPPORTED_SOURCE_TYPES
from .serialization import to_json_serializable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception.main",
        description=(
            "Detect sustained inference-performance degradation from one "
            "explicit two-phase JSON dataset."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help=(
            "UTF-8 JSON file containing explicit standard and inference data; "
            "metric entries may be canonical series or Prometheus matrices."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional inclusive lower bound applied only to inference data.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional inclusive upper bound applied only to inference data.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[DEFAULT_METRIC],
        help="One or more metric names; names may contain path separators.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help='Detection task identifier; omitted values become "default".',
    )
    parser.add_argument(
        "--source-type",
        choices=sorted(SUPPORTED_SOURCE_TYPES),
        default="training_log",
        help="Timestamp display convention for the input source.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-metric YAML configuration.",
    )
    return parser


def run_detection(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the public detector and execute one detection pass."""

    detector_kwargs: dict[str, Any] = {
        "path": args.path,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "metrics": list(args.metrics),
        "task_id": args.task_id,
        "source_type": args.source_type,
    }
    # Let the detector use its bundled default when --config-dir is omitted;
    # Path(None) is invalid and would turn a valid default invocation into an
    # interface error.
    if args.config_dir is not None:
        detector_kwargs["config_dir"] = args.config_dir
    detector = DegradationPerception(**detector_kwargs)
    return detector.detect()


def _strict_json_dumps(value: Any) -> str:
    """Return deterministic standards-compliant JSON with no NaN extensions."""

    return json.dumps(
        to_json_serializable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, call the detector, and write exactly one JSON line."""

    args = _build_parser().parse_args(argv)
    try:
        output = _strict_json_dumps(run_detection(args))
        exit_code = 0
    except Exception as exc:  # CLI boundary: convert runtime errors to JSON.
        error = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        try:
            output = _strict_json_dumps(error)
        except Exception:
            # Strings and booleans are always JSON-native; this fallback keeps
            # the CLI contract even if a custom serializer regression occurs.
            output = '{"error":{"message":"serialization failed",' \
                '"type":"SerializationError"},"ok":false}'
        exit_code = 1
    sys.stdout.write(output + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
