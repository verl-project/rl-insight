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

"""Optional CLI over the installed-package Grafana dashboard renderer.

This is a thin wrapper: evaluation, serialization and the ``--check``
comparison all come from :mod:`rl_insight.grafana.renderer`. The wrapper adds
only argument parsing and exit codes, so the CLI and any in-process caller
render exactly the same bytes.

Usage::

    python tools/grafana/framework/generate.py --config <composition.jsonnet> \\
        --out-dir <dir>
    python tools/grafana/framework/generate.py --config <composition.jsonnet> \\
        --check --expected-dir <dir>

Exit codes: ``0`` success, ``1`` ``--check`` found stale files, ``2`` the
config could not be rendered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if (_REPO_ROOT / "rl_insight").is_dir() and str(_REPO_ROOT) not in sys.path:
    # Running straight from a source checkout without an installed package.
    sys.path.insert(0, str(_REPO_ROOT))

from rl_insight.grafana.renderer import (  # noqa: E402
    JsonnetRenderError,
    materialize_dashboards,
    stale_dashboards,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="composition config (.jsonnet)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="directory to write <dashboard-name>.json files into",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against expected files instead of writing (exit 1 on mismatch)",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        help="directory holding the expected files for --check (default: --out-dir)",
    )
    args = parser.parse_args()

    if not args.check and args.out_dir is None:
        parser.error("--out-dir is required unless --check is used")
    expected_dir = args.expected_dir if args.expected_dir is not None else args.out_dir

    if args.check:
        if expected_dir is None:
            print(
                "error: --check requires --expected-dir (or --out-dir)", file=sys.stderr
            )
            return 2
        try:
            stale = stale_dashboards(args.config, expected_dir)
        except JsonnetRenderError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if stale:
            print(
                "Generated dashboards do not match the expected files:", file=sys.stderr
            )
            for path in stale:
                print(f"  {path}", file=sys.stderr)
            return 1
        return 0

    assert args.out_dir is not None
    try:
        written = materialize_dashboards(args.config, args.out_dir)
    except JsonnetRenderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
