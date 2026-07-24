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

"""HTTP API and remote helpers for the RL-Insight server."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse
from omegaconf import DictConfig, OmegaConf

from ..utils.constants import MonitorEnv, MonitorServer, PrometheusScrape
from ..utils.monitor_config_loader import load_server_config_file
from ..utils.prometheus_utils import PrometheusTarget, PrometheusTargetStore
from .network import local_addresses

logger = logging.getLogger(__name__)


def _unix_seconds(raw: Any) -> Optional[int]:
    """Accept unix seconds or Grafana epoch milliseconds."""
    if raw is None or raw == "":
        return None
    try:
        value = int(float(str(raw)))
    except (TypeError, ValueError):
        return None
    if value > 10**12:
        value //= 1000
    return value


def server_url() -> str:
    """Return the configured RL-Insight server URL without a trailing slash."""
    return str(os.environ.get(MonitorEnv.SERVER_URL, "")).strip().rstrip("/")


def get_server_services() -> dict[str, Any]:
    """Fetch service endpoints from the RL-Insight server."""
    base_url = server_url()
    if not base_url:
        logger.error(
            "[rl-insight] RL-Insight server URL is required; set %s",
            MonitorEnv.SERVER_URL,
        )
        return {}

    url = f"{base_url}{MonitorServer.API_PREFIX}/services"
    last_error: requests.RequestException | ValueError | None = None
    for attempt in range(MonitorServer.SERVICE_DISCOVERY_RETRIES):
        try:
            response = requests.get(
                url,
                timeout=MonitorServer.SERVICE_DISCOVERY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(
                    f"services response must be an object, got {type(data).__name__}"
                )
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

        if attempt + 1 < MonitorServer.SERVICE_DISCOVERY_RETRIES:
            time.sleep(MonitorServer.SERVICE_DISCOVERY_RETRY_DELAY_SECONDS)

    logger.error(
        "[rl-insight] Failed to fetch RL-Insight server services at %s: %s",
        url,
        last_error,
    )
    return {}


def create_app(conf: DictConfig) -> FastAPI:
    """Create the RL-Insight server application."""
    app = FastAPI(title="RL-Insight server", version="0.1.0")
    store = PrometheusTargetStore.from_config(conf)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Return a lightweight liveness response for process checks."""
        return {"status": "ok"}

    @app.get(f"{MonitorServer.API_PREFIX}/services")
    def services() -> dict[str, Any]:
        """Expose enabled component ports for trainer-side endpoint discovery."""
        return {
            "status": "ok",
            "otlp_port": (
                int(OmegaConf.select(conf, "otel.otel_port"))
                if bool(OmegaConf.select(conf, "tempo.enable", default=True))
                else None
            ),
            "prometheus_port": (
                int(OmegaConf.select(conf, "prometheus.prometheus_port"))
                if bool(OmegaConf.select(conf, "prometheus.enable", default=True))
                else None
            ),
            "grafana_port": (
                int(OmegaConf.select(conf, "grafana.port"))
                if bool(OmegaConf.select(conf, "grafana.enable", default=True))
                else None
            ),
        }

    @app.post(f"{MonitorServer.API_PREFIX}/prometheus/targets")
    def register_prometheus_targets(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Register metric scrape targets into the runtime Prometheus config."""
        raw_targets = payload.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="targets must be a non-empty list",
            )

        default_labels = payload.get("labels") or {}
        if not isinstance(default_labels, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="labels must be an object",
            )

        targets: list[PrometheusTarget] = []
        # Apply request-level labels first, then let each target override them.
        for item in raw_targets:
            if isinstance(item, str):
                targets.append(
                    PrometheusTarget(target=item, labels=dict(default_labels))
                )
                continue
            if isinstance(item, dict):
                item_labels = item.get("labels") or {}
                if not isinstance(item_labels, dict):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="target labels must be an object",
                    )
                targets.append(
                    PrometheusTarget(
                        target=str(item.get("target")),
                        labels={**default_labels, **item_labels},
                    )
                )
                continue
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="each target must be either a string or an object",
            )

        job_name = str(payload.get("job_name") or PrometheusScrape.TRAINER_METRICS_JOB)
        try:
            result = store.register(job_name, targets)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        reloaded = False
        try:
            reloaded = store.reload()
        except requests.RequestException as exc:
            logger.warning(
                "[rl-insight] Failed to reload Prometheus after target update: %s",
                exc,
            )
        return {"status": "ok", "prometheus_reloaded": reloaded, **result}

    def _do_rebuild(
        *,
        start: Optional[int],
        end: Optional[int],
        run_id: Optional[str] = None,
        gap_s: float = 300.0,
        write_bundled: bool = False,
    ) -> dict[str, Any]:
        from rl_insight.experimental.agent_loop.constants import (  # noqa: PLC0415
            SERVICE_NAME_VALUE,
        )
        from rl_insight.experimental.agent_loop.rebuild.service import (  # noqa: PLC0415
            rebuild_from_tempo,
        )

        tempo_url = (
            f"http://127.0.0.1:"
            f"{int(OmegaConf.select(conf, 'tempo.query_port', default=3200))}"
        )
        return rebuild_from_tempo(
            tempo_url=tempo_url,
            service_name=SERVICE_NAME_VALUE,
            start_unix=start,
            end_unix=end,
            run_id=run_id,
            gap_s=gap_s,
            write_bundled=write_bundled,
        )

    @app.get(f"{MonitorServer.API_PREFIX}/agent-loop/rebuild/go")
    def rebuild_agent_loop_go(
        request: Request,
        from_ts: Optional[str] = Query(None, alias="from"),
        to_ts: Optional[str] = Query(None, alias="to"),
        return_url: Optional[str] = Query(None, alias="return"),
        run_id: Optional[str] = Query(None),
    ) -> Response:
        """Dashboard-link target: Rebuild, then bounce back to Grafana (same tab)."""
        from rl_insight.experimental.agent_loop.constants import (  # noqa: PLC0415
            DEFAULT_GRAFANA_BASE,
            GRAFANA_DASHBOARD_SLUG,
            GRAFANA_DASHBOARD_UID,
        )
        from rl_insight.experimental.agent_loop.dashboard.writer import (  # noqa: PLC0415
            write_agent_loop_from_runs,
        )

        start = _unix_seconds(from_ts)
        end = _unix_seconds(to_ts)
        try:
            result = _do_rebuild(start=start, end=end, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            # Still clear panels + bounce back; never leave the user on an error page.
            logger.exception("agent-loop rebuild/go failed: %s", exc)
            try:
                write_agent_loop_from_runs(
                    [], window_from=start, window_to=end
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to clear Agent Loop after rebuild error")
            result = {"runs": [], "empty": True}

        dest = (return_url or "").strip()
        referer = (request.headers.get("referer") or "").strip()
        if not dest and referer and "/d/" in referer:
            dest = referer.split("#", 1)[0]
            if start is not None and end is not None and "from=" not in dest:
                sep = "&" if "?" in dest else "?"
                dest = f"{dest}{sep}from={start * 1000}&to={end * 1000}"
        if not dest:
            dest = (
                f"{DEFAULT_GRAFANA_BASE}/d/{GRAFANA_DASHBOARD_UID}/"
                f"{GRAFANA_DASHBOARD_SLUG}"
            )
            if start is not None and end is not None:
                dest = f"{dest}?from={start * 1000}&to={end * 1000}"

        n_runs = len(result.get("runs") or [])
        return HTMLResponse(
            status_code=200,
            content=(
                "<!DOCTYPE html><html><head><meta charset=utf-8>"
                f'<meta http-equiv="refresh" content="0;url={dest}">'
                f"<script>location.replace({dest!r});</script>"
                "</head><body style='font-family:sans-serif;padding:1.5rem'>"
                f"<p>Rebuilt {n_runs} run(s). Returning to Grafana…</p>"
                f'<p><a href="{dest}">Continue</a></p>'
                "</body></html>"
            ),
        )

    @app.post(f"{MonitorServer.API_PREFIX}/agent-loop/rebuild")
    def rebuild_agent_loop(
        payload: dict[str, Any] = Body(default_factory=dict),
    ) -> dict[str, Any]:
        """JSON Rebuild API (scripts / automation)."""
        start = _unix_seconds(payload.get("from"))
        end = _unix_seconds(payload.get("to"))
        run_id = payload.get("run_id")
        gap_s = float(payload.get("gap_s") or 300.0)
        write_bundled = bool(payload.get("write_bundled") or False)
        try:
            return _do_rebuild(
                start=start,
                end=end,
                run_id=str(run_id) if run_id else None,
                gap_s=gap_s,
                write_bundled=write_bundled,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent-loop rebuild failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl_insight.server.http_api")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved server config YAML used by the RL-Insight server.",
    )
    args = parser.parse_args(argv)

    conf = load_server_config_file(args.config)
    port = int(OmegaConf.select(conf, "server.port", default=18080))
    uvicorn.run(create_app(conf), host=local_addresses()["bind"], port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
