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

"""Generate PR#120 demo samples → Prometheus + Tempo for Repeat dashboard.

Does **not** rewrite dashboard JSON (panel count must stay variable-driven).
Build once::

    python -m rl_insight.experimental.build_repeat_dashboard

Then::

    python rl_insight/experimental/export_to_tempo.py --samples 2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from prometheus_client import start_http_server  # noqa: E402

from rl_insight.experimental.builder import TrajectoryBuilder  # noqa: E402
from rl_insight.experimental.generate_data import generate  # noqa: E402
from rl_insight.experimental.prom_export import publish_sample_runs  # noqa: E402
from rl_insight.experimental.tempo_export import (  # noqa: E402
    SERVICE_NAME_VALUE,
    export_samples_to_tempo,
    new_run_id,
)

logger = logging.getLogger(__name__)


def _try_register_scrape(port: int, job: str = "agent-loop") -> str | None:
    try:
        from omegaconf import OmegaConf

        from rl_insight.utils.monitor_config_loader import load_server_config_file
        from rl_insight.utils.prometheus_utils import (
            PrometheusTarget,
            PrometheusTargetStore,
        )

        conf = load_server_config_file()
        store = PrometheusTargetStore.from_config(conf)
        store.register(job, [PrometheusTarget(target=f"127.0.0.1:{port}")])
        try:
            store.reload()
        except Exception as exc:  # noqa: BLE001
            logger.warning("prometheus reload failed (config written): %s", exc)
        return str(OmegaConf.select(conf, "prometheus.config_file", default=""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not register scrape target: %s", exc)
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Generate PR#120 demo samples → Prometheus + Tempo "
            "(Repeat dashboard reads Prom *_info; does not materialize rows)"
        )
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:4318/v1/traces",
        help="OTLP/HTTP traces endpoint",
    )
    parser.add_argument(
        "--samples", type=int, default=2, help="Number of samples (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Shared run_id for Prom + Tempo (default: auto export-…)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9108,
        help="Prometheus /metrics listen port (default: 9108)",
    )
    parser.add_argument(
        "--register-scrape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Register this process in runtime prometheus.yml and reload "
            "(default: enabled; use --no-register-scrape to disable)"
        ),
    )
    parser.add_argument(
        "--no-tempo",
        action="store_true",
        help="Skip Tempo export (Prom only)",
    )
    parser.add_argument(
        "--serve-seconds",
        type=float,
        default=0.0,
        help="Keep /metrics up this many seconds (0 = forever)",
    )
    args = parser.parse_args()

    builder = TrajectoryBuilder()
    generate(builder, args.samples, args.seed)
    samples = builder.samples
    run_id = args.run_id.strip() or new_run_id()

    start_http_server(args.port)
    n_traj = publish_sample_runs([(run_id, samples)])
    logger.info(
        "serving agent_loop_* from PR#120 samples on :%s "
        "(run_id=%s samples=%s traj=%s)",
        args.port,
        run_id,
        len(samples),
        n_traj,
    )

    if args.register_scrape:
        cfg = _try_register_scrape(args.port)
        if not cfg:
            parser.error(
                "could not register the Prometheus scrape target; "
                "use --no-register-scrape only when it is configured elsewhere"
            )
        logger.info("scrape register attempted (config=%s)", cfg)

    if not args.no_tempo:
        result = export_samples_to_tempo(
            samples, endpoint=args.endpoint, run_id=run_id
        )
        print("\nExported to Tempo (mapper).")
        print(f"  service.name={SERVICE_NAME_VALUE}")
        print(f"  run_id={result['run_id']}")
        print(f"  spans={result['spans']}")
    else:
        print(f"\nProm only. run_id={run_id} traj_nodes={n_traj}")

    print(
        "\nOpen Grafana agent_loop_trajectory (hard-refresh). "
        "nested Rows Repeat soft-refreshes when node cardinality changes; "
        "export does NOT rewrite dashboard panel counts."
    )
    print(f"Prometheus scrape target should include 127.0.0.1:{args.port}")

    if args.serve_seconds and args.serve_seconds > 0:
        time.sleep(args.serve_seconds)
    else:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
