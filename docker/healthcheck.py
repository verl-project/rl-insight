"""Check readiness of every bundled service without requiring curl."""

import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import yaml

from rl_insight.server.catalog import DEFAULT_STATE_ROOT, STATE_FILE
from rl_insight.server.network import format_host_port, local_addresses


def _runtime_config_path() -> Path:
    """Resolve the runtime dir from the active state written by ``server start``."""
    state_file = DEFAULT_STATE_ROOT / "run" / STATE_FILE
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        runtime_dir = state["runtime_dir"]
        if not isinstance(runtime_dir, str) or not runtime_dir.strip():
            raise ValueError("runtime_dir is empty")
        return Path(runtime_dir) / "server.yaml"
    except (OSError, KeyError, TypeError, ValueError):
        # The state file is written after the services start; before that the
        # default runtime dir is the only known location.
        return DEFAULT_STATE_ROOT / "runtime" / "server.yaml"


def _load_config() -> dict:
    """Load the runtime config; an unreadable config fails the healthcheck."""
    path = _runtime_config_path()
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Cannot read runtime config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Runtime config {path} is not a mapping")
    return config


opener = build_opener(ProxyHandler({}))
host = local_addresses()["loopback"]
config = _load_config()

# Read ports from the runtime config because auto-port may assign non-default
# values at startup.
for section, port_key, path in (
    ("server", "port", "/healthz"),
    ("prometheus", "prometheus_port", "/-/ready"),
    ("tempo", "query_port", "/ready"),
    ("grafana", "port", "/api/health"),
):
    service = config.get(section)
    if not isinstance(service, dict):
        raise SystemExit(f"Runtime config is missing the {section} section")
    if not service.get("enable", True):
        continue
    port = service.get(port_key)
    if port is None:
        raise SystemExit(f"Runtime config is missing {section}.{port_key}")
    try:
        port = int(port)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {section}.{port_key} in runtime config") from exc
    with opener.open(
        f"http://{format_host_port(host, port)}{path}", timeout=3
    ) as response:
        if response.status != 200:
            raise SystemExit(f"Service on port {port} is not ready")
