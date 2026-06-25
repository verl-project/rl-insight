# Quick Start

This guide starts RL-Insight Monitor from a fresh checkout, runs the local server stack, and adds the first metric and trace calls to training code.

For service version requirements and Linux platform support, see [Server Installation](./server_installation.md).

## 1. Install RL-Insight

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the CLI entry point:

```bash
rl-insight --help
```

## 2. Install Server Services

RL-Insight depends on Prometheus, Tempo, and Grafana for online monitoring. The easiest Linux path is to let RL-Insight install the supported versions into `~/.rl-insight/services`:

```bash
rl-insight server install
```

The installer uses these versions:

| Service | Installer version | Requirement |
|---|---:|---:|
| Prometheus | `2.54.1` | `>= 2.30.0` |
| Tempo | `2.6.1` | `>= 2.0.0` |
| Grafana | `13.0.0` | `>= 13.0.0` |

If your environment already provides compatible system packages, `server start` can use them directly.

## 3. Start The Stack

Start Prometheus, Tempo, and Grafana:

```bash
rl-insight server start
```

The command prints the detected server IP, Grafana URL, and trainer-facing OTLP endpoint. Foreground mode keeps logs attached and stops the services when you press `Ctrl+C`.

Common variants:

```bash
rl-insight server start --detach
rl-insight server start --attach-logs
rl-insight server start --config path/to/config.yaml
rl-insight server stop
```

Default endpoints:

| Endpoint | Default |
|---|---|
| Grafana | `http://<server-ip>:3000` |
| Prometheus | `http://<server-ip>:9090` |
| Tempo query API | `http://<server-ip>:3200` |
| OTLP HTTP traces | `http://<server-ip>:4318/v1/traces` |

## 4. Instrument Training Code

Set the RL-Insight server IP before launching or initializing training workers. Use the server IP printed by `rl-insight server start`:

```bash
export RL_INSIGHT_SERVICE_IP=<server-ip>
```

Then initialize Ray and enable RL-Insight once per process:

```python
import ray
import rl_insight as insight

ray.init(address="auto", namespace="rl-insight-monitor")
insight.init(project="verl", experiment_name="ppo-smoke-test")
```

If your RL framework already integrates RL-Insight, you can start the corresponding RL training job after the server stack is running and `RL_INSIGHT_SERVICE_IP` is set. The manual API calls below are for framework authors or custom training loops.

Record metrics:

```python
insight.metric_count("train_step_total", amount=1, worker="trainer_0")
insight.metric_value("reward_mean", value=1.23, worker="trainer_0")
insight.metric_distribution("step_latency_ms", value=42.5, worker="trainer_0")
```

Record RL state intervals:

```python
with insight.trace_state("rollout", state_lane_id="actor_0", step=10):
    run_rollout()

with insight.trace_state("update_policy", state_lane_id="actor_0", step=10):
    update_policy(batch)
```

Decorate synchronous operations when a duration span is enough:

```python
@insight.trace_op("reward_model", stage="reward")
def score_responses(batch):
    return reward_model(batch)
```

## 5. Open Grafana

Open the Grafana URL printed by `rl-insight server start`. By default, Grafana listens at:

```text
http://<server-ip>:3000
```

The default login is:

```text
username: admin
password: admin
```

The bundled provisioning config loads Prometheus and Tempo datasources and dashboard JSON files from:

```text
rl_insight/config/services/grafana/dashboards
```

Prometheus metrics and Tempo traces are persisted under `~/.rl-insight/data` by default. Stopping the server does not delete collected data.

## 6. Stop Services

Foreground mode:

```bash
Ctrl+C
```

Detached mode or another terminal:

```bash
rl-insight server stop
```

## Configuration Shortcuts

Pass overrides through `insight.init(config=...)`:

```python
insight.init(
    project="verl",
    experiment_name="ppo-smoke-test",
    config={
        "server": {
            "namespace": "rl_insight_monitor",
            "backend": "ray",
            "service_ip": "10.0.0.8",
        },
        "prometheus": {
            "metrics_report_port": 9092,
            "prometheus_port": 9090,
        },
        "otel": {
            "otel_port": 4318,
        },
    },
)
```

Environment variables take precedence for common deployment settings:

| Variable | Purpose |
|---|---|
| `RL_INSIGHT_SERVICE_IP` | Server IP used by training workers. |
| `RL_INSIGHT_OTEL_PORT` | OTLP HTTP port, default `4318`. |
| `RL_INSIGHT_PROMETHEUS_PORT` | Prometheus HTTP port, default `9090`. |
| `RL_INSIGHT_PROMETHEUS_CONFIG_FILE` | Prometheus config path used by target registration logic. |

## Troubleshooting

If `server start` reports missing or incompatible services, run:

```bash
rl-insight server install
```

If training workers emit no traces, check that `RL_INSIGHT_SERVICE_IP` points to the node running Tempo and that workers can reach `http://<server-ip>:4318/v1/traces`.

If metrics do not appear, check that the monitor hub process is reachable from Prometheus and that the Prometheus configuration points to the hub `/metrics` endpoint.


