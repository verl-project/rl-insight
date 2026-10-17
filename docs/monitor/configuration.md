# Configuration and Switches

Starting the server makes storage and dashboards available. Training produces data only after its integration is enabled.

## Choose a configuration layer

| Layer | Configuration entry | Purpose |
| --- | --- | --- |
| Server stack | YAML passed to `rl-insight server start --config` | Services, ports, storage, dashboards |
| Training client | `rl_insight.init(config=...)` and environment variables | Worker connection and metric reporting |
| Framework | verl or verl-omni training arguments | Training metrics, engine metrics, and instrumented states |

These are separate configurations. Server-side `server.backend=local` manages service processes; client-side `server.backend=ray` selects the training transport. Do not pass server YAML to `init()`.

For a first run, start the services with [Quick Start](quick_start.md), set `RL_INSIGHT_SERVER_URL` in the training environment, and enable the framework logger. The [verl](rl_framework_example/verl/index.md) and [verl-omni](rl_framework_example/verl-omni/index.md) examples provide complete launch commands.

## Configuration precedence

| Layer | Order, from lowest to highest priority |
| --- | --- |
| Server | Bundled `config.yaml` → custom `--config` YAML → applicable CLI overrides (`--log-dir`, `--extra-dashboard-dir`); `--auto-port` may replace occupied ports at startup |
| Training client | Built-in defaults → `init(config=...)` → `RL_INSIGHT_SERVER_URL` / `RL_INSIGHT_SERVER_BACKEND` environment overrides |
| Framework example | Upstream Hydra configuration → recipe script arguments → arguments appended to the script through `"$@"` |

`RL_INSIGHT_SERVER_URL` connects clients to an existing server; it does not change the server's listening port. Use the URLs printed after startup, especially with `--auto-port`.

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

## Training client settings

| Setting in `init(config=...)` | Default | Meaning |
| --- | --- | --- |
| `server.url` | Empty; required | Cluster-reachable RL-Insight API URL |
| `server.backend` | `ray` | Built-in client backend; requires initialized Ray |
| `server.namespace` | `rl_insight_monitor` | Metric prefix and OTEL service name, not Ray actor namespace |
| `prometheus.metrics_report_port` | `9092` | Hub HTTP port scraped by Prometheus |

The following environment variables override the corresponding client settings when set to non-empty values:

| Environment variable | Overrides |
| --- | --- |
| `RL_INSIGHT_SERVER_URL` | `server.url` |
| `RL_INSIGHT_SERVER_BACKEND` | `server.backend` |

Set these before initializing workers. Preserve the default namespace for bundled dashboards, which query that prefix. Separate Ray jobs whose hubs share a host need distinct available reporting ports.

For custom instrumentation, call `ray.init()` before `rl_insight.init()`. A missing URL or uninitialized Ray leaves monitoring inactive. Repeated initialization after success is ignored. `finish()` resets instrumentation in the current process; it does not stop services or delete data.

The hub is shared within a Ray job. Its first initialization establishes the reporting port and namespace; setting a different config in a later worker does not reconfigure that hub. Restart the Ray job when changing hub settings.

## Framework switches

These are training command overrides, not server YAML. Defaults can be changed by the selected upstream recipe.

| Switch | Base behavior | Monitoring setting |
| --- | --- | --- |
| `trainer.logger` | Recipe-dependent | Include `rl_insight`; retain other loggers if needed |
| `trainer.project_name`, `trainer.experiment_name` | Recipe-dependent | Identities used to filter dashboards |
| `actor_rollout_ref.rollout.disable_log_stats` | Normally `true` | Set `False` for engine metrics; required by verl-omni when monitoring is enabled |
| `transfer_queue.metrics.enabled` | Normally `false` | Set `True` for a training mode that uses TransferQueue |

The framework manages `VERL_RL_INSIGHT_ENABLE`. Setting it alone does not replace `trainer.logger`. To disable framework reporting, remove `rl_insight` from the logger list and start a fresh job without a manually inherited enable variable. There is no package-wide `RL_INSIGHT_ENABLED` switch.

TransferQueue metrics do not enable TransferQueue itself. Hardware exporters require separate [hardware monitoring setup](hardware/index.md). Instrumented RL state traces and engine-native detailed inference traces are different features; enabling the logger does not guarantee detailed engine traces.

## Feature dependencies

| Data you want | Training or exporter requirement | Server requirement |
| --- | --- | --- |
| Trainer scalar metrics | `rl_insight` logger, client package installed in workers, reachable server URL | Prometheus scraping the Ray hub |
| Rollout engine metrics | Logger plus `disable_log_stats=False`; supported engine integration registers its endpoint | Prometheus can reach each registered rollout endpoint |
| TransferQueue metrics | A trainer using TransferQueue plus `transfer_queue.metrics.enabled=True` | Prometheus can reach the queue endpoint |
| RL state traces | Instrumented framework operations and enabled worker client | Tempo enabled and OTLP HTTP reachable from the hub or instrumented workers |
| CPU / Ascend NPU metrics | Exporters installed and registered separately | Prometheus can reach exporter endpoints |
| Dashboard views | Collected metrics or traces for the chosen run | Grafana and the corresponding Prometheus / Tempo data source |

Disabling Grafana does not stop collection. Disabling Tempo removes trace storage but leaves metric collection available. Disabling Prometheus removes metric scraping and storage even if training continues to emit values.

## Connectivity and verification

Workers must reach the server API (`18080`) and OTLP endpoint (`4318` when enabled). Prometheus must reach the hub (`9092` by default), rollout endpoints, and exporters. Opening Grafana alone does not verify these paths.

For Ray job submission, merge this into the runtime environment actually used by the job:

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
  RL_INSIGHT_SERVER_BACKEND: "ray"
```

Use a cluster-reachable address rather than worker-local `localhost`. Check discovery from the training host with `curl http://<server-ip>:18080/api/v1/services`, then inspect Prometheus **Status → Targets** for successful scrapes.

Continue with the [verl example](rl_framework_example/verl/index.md) or [verl-omni example](rl_framework_example/verl-omni/index.md).
