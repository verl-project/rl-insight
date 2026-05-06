# RL-Insight Monitor

RL-Insight Monitor provides an observability stack for RL training metrics and traces based on Prometheus, Tempo, and Grafana.

It has two parts:

- `rl-insight server ...`: manage the observability Docker stack.
- `rl_insight`: training-side Python APIs for metrics and traces.

## Quickstart

### 1. Install

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Start the observability stack

Default foreground mode:

```bash
rl-insight server start
```

This mode starts Docker Compose silently, keeps the CLI attached, and stops the whole stack when you press `Ctrl+C`.

Grafana will be provisioned automatically with Prometheus and Tempo datasources plus an empty starter dashboard. The datasources follow the configured Prometheus and Tempo published ports.

Background mode:

```bash
rl-insight server start --detach
```

Foreground mode with compose/container logs attached:

```bash
rl-insight server start --attach-logs
```

Use a custom config file:

```bash
rl-insight server start --config path/to/config.yaml
```

Stop the stack explicitly from another terminal:

```bash
rl-insight server stop
```

After startup, the CLI prints:

- Prometheus config file path
- Trainer OTLP traces URL
- Prometheus, Tempo, and Grafana access URLs

### 3. Initialize the training side

```python
import os
import ray
import rl_insight as insight

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://<server-ip>:4318/v1/traces"

ray.init(address="auto", namespace="rl-insight-monitor")
insight.init()
```

Notes:

- `ray.init(namespace="rl-insight-monitor")` is used to find the monitor hub actor.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` takes precedence over `insight.init(config)` -> `otel.traces_endpoint`.

### 4. Emit metrics and traces

```python
import rl_insight as insight

insight.metric_count("train_step_total", amount=1, worker="trainer_0")
insight.metric_value("reward_mean", value=1.23, worker="trainer_0")
insight.metric_distribution("step_latency_ms", value=42.5, worker="trainer_0")

with insight.trace_state("rollout", state_lane_id="trainer_0", step=10):
    run_rollout()

@insight.trace_op("update_model", stage="optimizer")
def update_model(batch):
    ...
```

## APIs

| API | Purpose |
|---|---|
| `init(config=None)` | Initialize training-side monitoring |
| `close()` | Reset monitor state in the current process |
| `metric_count()` | Report a counter |
| `metric_value()` | Report a gauge |
| `metric_distribution()` | Report a histogram |
| `trace_state()` | Report a state interval |
| `trace_op()` | Decorator for operation latency traces |

## CLI Reference

### `rl-insight server start`

| Argument | Default | Description |
|---|---:|---|
| `--detach` | `false` | Start in background and return immediately |
| `--attach-logs` | `false` | Run in foreground and stream compose/container logs |
| `--config` | `experimental/config/services/config.yaml` | Server config file path |
| `--log-level` | `INFO` | Python log level |

### `rl-insight server stop`

| Argument | Default | Description |
|---|---:|---|
| `--config` | `experimental/config/services/config.yaml` | Server config file path |
| `--log-level` | `INFO` | Python log level |

## Server YAML

| Key | Default | Description |
|---|---:|---|
| `server.backend` | `docker_compose` | Stack startup backend |
| `server.compose_file` | `docker-compose.yaml` | Compose file path |
| `server.project_name` | `rl-insight-monitor` | Compose project name |
| `prometheus.prometheus_port` | `9090` | Prometheus HTTP port |
| `prometheus.config_file` | `prometheus.yml` | Prometheus config file |
| `tempo.query_port` | `3200` | Tempo query port |
| `otel.traces_endpoint` | `http://127.0.0.1:4318/v1/traces` | Trainer trace export endpoint |
| `grafana.port` | `3000` | Grafana HTTP port |
| `grafana.provisioning_dir` | `provisioning` | Grafana provisioning directory mounted into the container |
| `grafana.dashboards_dir` | `dashboards` | Grafana dashboard JSON directory mounted into the container |

## `insight.init(config)`

| Key | Default | Description |
|---|---:|---|
| `namespace` | `rl_insight_monitor` | Metrics and trace namespace |
| `backend.type` | `ray` | Currently only `ray` is supported |
| `prometheus.metrics_report_port` | `9092` | Monitor hub `/metrics` port |
| `prometheus.prometheus_port` | `9090` | Prometheus HTTP port used for reload |
| `prometheus.config_file` | bundled absolute path | Prometheus config file to rewrite |
| `prometheus.reload.mode` | `ray` | `ray` or `none` |
| `otel.traces_endpoint` | `http://127.0.0.1:4318/v1/traces` | Trainer trace export endpoint |
