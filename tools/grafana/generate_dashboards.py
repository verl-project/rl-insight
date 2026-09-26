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

"""Render the registered Grafana dashboard compositions to committed JSON.

Thin wrapper around the generic framework generator
(``tools/grafana/framework/generate.py``): the Jsonnet engine, deterministic
serialization, and the render/check core are reused, not reimplemented. The
only production-specific parts are the defaults — the composition registry
entrypoint and the committed output directory.

``--check`` compares the committed JSON against the freshly rendered
dashboards by parsed-object equality (semantic check). The framework
generator's own ``--check`` remains byte-exact for framework-owned outputs;
production equality is defined semantically here so that serialization-only
differences never mask or fake content drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_GRAFANA_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = TOOLS_GRAFANA_DIR / "framework"
sys.path.insert(0, str(FRAMEWORK_DIR))

from generate import generated_text, render  # noqa: E402  (framework core)

DEFAULT_CONFIG = TOOLS_GRAFANA_DIR / "dashboards.jsonnet"
DEFAULT_OUT_DIR = (
    TOOLS_GRAFANA_DIR.parent.parent
    / "rl_insight"
    / "config"
    / "services"
    / "grafana"
    / "dashboards"
    / "verl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="composition entrypoint (.jsonnet, default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory holding <composition-name>.json (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare rendered dashboards against the committed files (exit 1 on drift)",
    )
    args = parser.parse_args()

    try:
        dashboards = render(args.config)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.check:
        drifted: list[str] = []
        for name, dashboard in dashboards.items():
            path = args.out_dir / f"{name}.json"
            committed = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            )
            if committed != dashboard:
                drifted.append(f"{path} (semantic drift)")
        if drifted:
            print(
                "Rendered dashboards do not match the committed files:",
                file=sys.stderr,
            )
            for path in drifted:
                print(f"  {path}", file=sys.stderr)
            return 1
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, dashboard in dashboards.items():
        path = args.out_dir / f"{name}.json"
        path.write_text(generated_text(dashboard), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
