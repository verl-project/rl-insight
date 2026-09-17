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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException, status
from omegaconf import DictConfig, OmegaConf

from ..utils.constants import MonitorEnv, MonitorServer, PrometheusScrape
from ..utils.experiment_targets import (
    ArchivedExperimentError,
    ExperimentIdentityError,
    ExperimentNotFoundError,
    ExperimentTargetStore,
    normalize_experiment_identity,
)
from ..utils.monitor_config_loader import load_server_config_file
from ..utils.prometheus_utils import PrometheusTarget, PrometheusTargetStore
from .network import local_addresses

logger = logging.getLogger(__name__)


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
    experiments = ExperimentTargetStore.from_config(conf)
    app.state.legacy_targets = store
    app.state.experiments = experiments

    try:
        migration = experiments.migrate_legacy_targets(store.targets_file)
    except Exception as exc:  # noqa: BLE001 - a broken legacy file must not take the server down
        logger.error(
            "[rl-insight] Legacy target migration failed (%s); keeping the global "
            "file untouched and continuing startup. The next start retries the "
            "migration, and legacy registration will surface the underlying error.",
            exc,
        )
        migration = {"changed": False, "scanned": False}
    if migration["changed"]:
        logger.warning(
            "[rl-insight] Migrated %d identified target record(s) from the legacy "
            "global file into experiment partitions; kept %d legacy record(s). "
            "Backup: %s",
            migration["migrated"],
            migration["kept_legacy"],
            migration["backup_file"],
        )
    elif migration["scanned"]:
        logger.info(
            "[rl-insight] Legacy target file holds %d unidentified record(s); "
            "they stay in the global partition until clients upgrade.",
            migration["kept_legacy"],
        )

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
            identity = normalize_experiment_identity(
                payload.get("project"), payload.get("experiment_name")
            )
            if identity is None:
                for item in targets:
                    conflicting = [
                        key
                        for key in ("project", "experiment_name")
                        if key in item.labels
                    ]
                    if conflicting:
                        raise ExperimentIdentityError(
                            "target labels "
                            f"{', '.join(sorted(conflicting))} are reserved for the "
                            "experiment identity; provide top-level project and "
                            "experiment_name instead of per-target labels"
                        )
                result = store.register(job_name, targets)
                logger.warning(
                    "[rl-insight] Registered %d target(s) for job %r without "
                    "experiment identity; kept in the legacy/global partition. "
                    "Pass project and experiment_name to scope new targets.",
                    len(targets),
                    job_name,
                )
            else:
                result = experiments.register(
                    identity[0], identity[1], job_name, targets
                )
        except ExperimentIdentityError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except ArchivedExperimentError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
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

    @app.get(f"{MonitorServer.API_PREFIX}/experiments")
    def list_experiments(project: str | None = None) -> list[dict[str, Any]]:
        """List experiment partitions with state, composite key, and target count."""
        return experiments.list_experiments(project)

    @app.get(f"{MonitorServer.API_PREFIX}/experiments/targets")
    def show_experiment_targets(
        project: str | None = None, experiment_name: str | None = None
    ) -> dict[str, Any]:
        """Show the discovery records of one experiment partition."""
        if not project or not experiment_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project and experiment_name query parameters are required",
            )
        try:
            return experiments.show_targets(project, experiment_name)
        except ExperimentIdentityError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except ExperimentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    def _lifecycle(payload: dict[str, Any], action: str) -> dict[str, Any]:
        try:
            identity = normalize_experiment_identity(
                payload.get("project"), payload.get("experiment_name")
            )
            if identity is None:
                raise ExperimentIdentityError(
                    "project and experiment_name are required in the request body"
                )
            operation = getattr(experiments, action)
            return operation(*identity)
        except ExperimentIdentityError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except ExperimentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    @app.post(f"{MonitorServer.API_PREFIX}/experiments/archive")
    def archive_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Stop target discovery for one experiment; history is kept."""
        return _lifecycle(payload, "archive")

    @app.post(f"{MonitorServer.API_PREFIX}/experiments/restore")
    def restore_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Re-enable target discovery for one archived experiment."""
        return _lifecycle(payload, "restore")

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
