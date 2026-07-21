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

"""Export trajectory demo data to Tempo via GrafanaRecord (BaseSample).

Does **not** modify ``generate_data.py`` / ``builder.py``. Reuses the public
``generate`` / ``stream`` helpers and only swaps the ``sample_factory``.

Usage::

    # Ensure rl-insight server (Tempo OTLP :4318) is running, then:
    python rl_insight/experimental/export_to_tempo.py --samples 8
    python rl_insight/experimental/export_to_tempo.py --stream --samples 4

Open Grafana ``quick_start_demo`` → Agent Loop row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rl_insight.experimental.builder import TrajectoryBuilder  # noqa: E402
from rl_insight.experimental.generate_data import generate, stream  # noqa: E402
from rl_insight.experimental.samples.grafana_record import (  # noqa: E402
    GrafanaRecord,
    configure_exporter,
    shutdown_exporter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feed generate_data events into GrafanaRecord → Tempo"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:4318/v1/traces",
        help="OTLP/HTTP traces endpoint (default: local Tempo)",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="Number of samples (default: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream events with sleeps (same as generate_data --stream)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Seconds between events in stream mode (default: 0.3)",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=1.0,
        help="Synthetic span duration per step in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for OTLP endpoint readiness",
    )
    args = parser.parse_args()

    configure_exporter(args.endpoint, wait=not args.no_wait)

    step_duration = args.step_duration
    builder = TrajectoryBuilder(
        sample_factory=lambda uid, si: GrafanaRecord.create(
            uid=uid,
            sample_index=si,
            step_duration_s=step_duration,
        )
    )

    try:
        if args.stream:
            stream(builder, args.samples, args.interval, args.seed)
        else:
            generate(builder, args.samples, args.seed)

        for sample in builder.samples:
            if isinstance(sample, GrafanaRecord):
                sample.flush_pending()
    finally:
        shutdown_exporter()

    print("\nExported to Tempo.")
    print("Open Grafana → RL-Insight → verl_trainer_v1_with_sglang_engine → Agent Loop Trajectory")
    print(
        '  TraceQL lane example: {span.state_lane_id = "sample=0/session=0/traj=0"'
        ' && resource.service.name="agent-loop-poc"}'
    )


if __name__ == "__main__":
    main()
