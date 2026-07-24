#!/usr/bin/env python3

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

"""Generate PR#120 demo samples, then map them to Tempo (no Grafana Rebuild).

Does not modify ``generate_data.py``. After export, open Grafana
``agent_loop_trajectory`` and click **Rebuild Agent Loop**.

Usage::

    python rl_insight/experimental/export_to_tempo.py --samples 2 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rl_insight.experimental.builder import TrajectoryBuilder  # noqa: E402
from rl_insight.experimental.generate_data import generate  # noqa: E402
from rl_insight.experimental.agent_loop import (  # noqa: E402
    SERVICE_NAME_VALUE,
    export_samples_to_tempo,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo samples and export to Tempo (mapper only)"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:4318/v1/traces",
        help="OTLP/HTTP traces endpoint",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="Number of samples (default: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    builder = TrajectoryBuilder()
    generate(builder, args.samples, args.seed)
    result = export_samples_to_tempo(builder.samples, endpoint=args.endpoint)
    print("\nExported to Tempo (mapper).")
    print(f"  service.name={SERVICE_NAME_VALUE}")
    print(f"  run_id={result['run_id']}")
    print(f"  spans={result['spans']}")
    print(
        "\nOpen Grafana dashboard agent_loop_trajectory "
        "and click Rebuild Agent Loop."
    )


if __name__ == "__main__":
    main()
