# Configuration

Configure the service stack and the training client separately. For Python function parameters and examples, see the [Python API](api.md).

| Configuration | Entry point | Purpose |
| --- | --- | --- |
| Server | `rl-insight server start --config server.yaml` | Services, listening ports, storage, dashboards |
| Training client | `RL_INSIGHT_CONFIG` or `rl_insight.init(config=...)` | Backend, server connection, metric reporting |

Framework-specific switches belong to the [verl integration guide](https://github.com/verl-project/verl/blob/main/docs/advance/rl_insight.md) and [verl-omni integration guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/rl_insight.md).

## Server settings

| Setting | Default | Effect |
| --- | --- | --- |
| `server.enable` | `true` | Enables local stack management; does not disable instrumentation in running workers |
| `server.backend` | `local` | Manages services locally; other values skip local management, without configuring an external deployment |
| `prometheus.enable` | `true` | Starts metric storage and scraping |
| `tempo.enable` | `true` | Starts trace storage and ingestion; disabling it makes newly initialized hubs skip trace export |
| `grafana.enable` | `true` | Starts the dashboard UI; collection can continue without it |
| `server.port` | `18080` | Server API and discovery |
| `prometheus.prometheus_port` | `9090` | Prometheus HTTP |
| `otel.otel_port` | `4318` | OTLP HTTP ingestion, used with Tempo |
| `tempo.query_port` | `3200` | Tempo queries |
| `grafana.port` | `3000` | Grafana HTTP |
| `prometheus.retention_time`, `tempo.retention_time` | `30d` | Metric and trace retention |
| `server.data_dir` | `~/.rl-insight/data` | Persistent data; `--log-dir` overrides it |
| `<service>.binary_path` | Empty | Explicit binary location, otherwise automatic discovery |

`<service>` is `prometheus`, `tempo`, or `grafana`. Their `min_version` and `install_version` settings control compatibility checks and downloads; see [Server Installation](server_installation.md).

Custom YAML overrides the bundled defaults. Template paths such as `tempo.config_file` use `${service_root}`, resolved relative to your YAML's directory. Copy the bundled configuration and templates together before editing:

```bash
python -c "from pathlib import Path; from shutil import copytree; from rl_insight.utils.constants import MonitorPaths; copytree(MonitorPaths.CONFIG_DIR, Path('monitor-config'), dirs_exist_ok=True)"
```

Edit `monitor-config/config.yaml`, then use the same file for installation and startup:

```bash
rl-insight server install --config monitor-config/config.yaml
rl-insight server start --config monitor-config/config.yaml --detach
```

For metrics-only collection, set `tempo.enable: false` in that file. Restart the stack and training job after changing service availability or ports so workers rediscover endpoints; a running hub does not reload these settings.

For example, edit these fields in the copied `monitor-config/config.yaml`, leaving its other settings intact:

```yaml
server:
  backend: local
  enable: true
  port: 18080
  data_dir: /path/to/rl-insight-data
tempo:
  enable: false
```

Use `--auto-port` to replace occupied ports and follow the printed URLs. Use `--extra-dashboard-dir` for [custom dashboards](custom_dashboards.md).

Server precedence is bundled defaults, then custom YAML, then applicable CLI overrides. `--log-dir` overrides `server.data_dir`; `--extra-dashboard-dir` overrides `grafana.extra_dashboard_dir`. Client environment variables do not change server listening ports.

## Training client settings

### Choose a backend

| `server.backend` | Requirements | Data path |
| --- | --- | --- |
| `ray` (default) | Initialized Ray and reachable `server.url` | Workers to Ray hub, then Prometheus metrics and Tempo traces |
| `push` | Configured push sinks | In-process client to sinks; no Ray or RL-Insight server URL required |

The server YAML uses `server.backend: local` to manage service processes. This is separate from the client backend; do not pass server YAML to `init()`.

### Configure the Ray client

| Setting | Default | Meaning |
| --- | --- | --- |
| `server.url` | Empty | Required server API URL for Ray; not the Grafana URL |
| `server.backend` | `ray` | Selects `ray` or `push` |
| `server.namespace` | `rl_insight_monitor` | Ray metric prefix and OTEL service name, not Ray actor namespace |
| `prometheus.metrics_report_port` | `9092` | Hub HTTP port scraped by Prometheus |

Save the following as `client.yaml` on each training node:

```yaml
server:
  backend: ray
  url: http://<server-ip>:18080
  namespace: rl_insight_monitor
prometheus:
  metrics_report_port: 9092
```

Set the path before starting training:

```bash
export RL_INSIGHT_CONFIG=/absolute/path/to/client.yaml
```

Alternatively, pass this structure to `init(config=...)`; see the [initialization example](api.md#initialize-and-finish).

### Override precedence

From lowest to highest priority:

1. Built-in client defaults.
2. YAML file named by `RL_INSIGHT_CONFIG`.
3. Explicit `init(config=...)` values.
4. `RL_INSIGHT_SERVER_URL` and `RL_INSIGHT_SERVER_BACKEND` environment overrides.

For example, if the YAML specifies port `18080` but `RL_INSIGHT_SERVER_URL` specifies `18081`, the client connects to `18081`. A missing or unreadable external YAML logs a warning and is ignored.

The Ray hub is shared within a job. Its first initialization establishes its settings; later workers do not reconfigure it. Restart the job after changing hub settings. Jobs whose hubs share a host need distinct available reporting ports. Keep the default namespace when using bundled dashboards.

## Connectivity and verification

For the Ray backend, verify these paths:

| Source | Destination | Default port |
| --- | --- | --- |
| Training processes | RL-Insight server API | `18080` |
| Ray hub / trace exporters | Tempo OTLP HTTP | `4318` |
| Prometheus | Ray hub metrics endpoint | `9092` |
| Prometheus | Registered engine or hardware exporter | Endpoint-specific |
| Browser | Grafana | `3000` |

Use cluster-reachable addresses. For Ray job submission, propagate settings through the job's runtime environment:

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
  RL_INSIGHT_SERVER_BACKEND: "ray"
```

If using `RL_INSIGHT_CONFIG`, propagate it too and ensure the file exists at the same path on each worker node.

Check discovery from the training host with `curl http://<server-ip>:18080/api/v1/services`, then inspect Prometheus **Status → Targets**. Check Tempo and OTLP connectivity separately for missing traces. See [Quick Start](quick_start.md) for an end-to-end demo and [Hardware Monitoring](hardware/index.md) for exporter setup.
