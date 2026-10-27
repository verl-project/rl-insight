# Quick Start

This guide starts RL-Insight Monitor from a fresh checkout, runs the local server stack, and adds the first metric and trace calls to training code.

For service version requirements and Linux and Windows platform support, see [Server Installation](./server_installation.md).

## 1. Install RL-Insight

Use **Python >= 3.10**. For a released package:

```bash
python -m pip install rl-insight
```

For this checkout's latest features, run from the repository root instead:

```bash
pip install -e .
```

Verify the CLI entry point:

```bash
rl-insight --help
```

## 2. Install Server Services

RL-Insight depends on Prometheus, Tempo, and Grafana for online monitoring. This section shows the direct install path. For supported platforms, offline installation, or using existing service binaries, see [Server Installation](./server_installation.md). The easiest Linux or Windows path is to let RL-Insight install the supported versions into `~/.rl-insight/services`:

```bash
rl-insight server install
```

The installer uses these versions:

| Service | Installer version | Requirement |
|---|---:|---:|
| Prometheus | `2.54.1` | `>= 2.30.0` |
| Tempo | `2.6.1` | `>= 2.0.0` |
| Grafana | `13.0.0` | `>= 13.0.0` |

If your environment already provides compatible system packages, `server start` can use them directly. The detailed options and troubleshooting notes are covered in [Server Installation](./server_installation.md).

## 3. Start The Stack

Start the RL-Insight server stack:

```bash
rl-insight server start
```

The command prints the detected server IP, Grafana URL, and trainer-facing OTLP endpoint. Foreground mode keeps logs attached and stops the services when you press `Ctrl+C`.

Common variants:

```bash
rl-insight server start --detach
rl-insight server start --attach-logs
rl-insight server start --log-dir /path/to/rl-insight-data
rl-insight server start --config path/to/config.yaml
rl-insight server stop
```

`--log-dir` points the stack at a custom data directory, matching the intent of TensorBoard's `--logdir`. By default, RL-Insight stores data under `~/.rl-insight/data`.

Default endpoints:

| Endpoint | Default |
|---|---|
| Grafana | `http://<server-ip>:3000` |
| Prometheus | `http://<server-ip>:9090` |
| Tempo query API | `http://<server-ip>:3200` |
| OTLP HTTP traces | `http://<server-ip>:4318/v1/traces` |

## 4. Instrument Training Code

Set the RL-Insight server URL before launching or initializing training workers. Use the server address printed by `rl-insight server start`:

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080
```

In PowerShell, use `$env:RL_INSIGHT_SERVER_URL = "http://<server-ip>:18080"` instead. This is the server API address, not the Grafana URL.

Save the following as `monitor_demo.py`, then run `python monitor_demo.py` in that same environment. It uses the three metric helpers and one `trace_state` span inside a loop, so Prometheus and Grafana keep receiving representative live samples while it runs. Leave it running while checking Grafana and press `Ctrl+C` when finished:

```python
import time

import ray
import rl_insight as insight

ray.init(namespace="rl-insight-monitor")
insight.init(project="verl", experiment_name="quick_start_demo")

step = 0
labels = {"worker": "trainer_0"}
while True:
    with insight.trace_state("rollout_generate", state_lane_id="replica_0", step=step):
        time.sleep(2)

    insight.metric_count("train_step_total", amount=1, **labels)
    insight.metric_gauge("reward_mean", value=1.0 + step * 0.01, **labels)
    insight.metric_log("reward_mean", value=1.0 + step * 0.01, step=step, **labels)
    insight.metric_histogram(
        "step_latency_ms", value=200 + (step % 5) * 20, **labels
    )

    step += 1
    time.sleep(0.5)
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

After login, open **Dashboards** from the left navigation. Bundled dashboards are grouped into folders and tagged:

| Folder | Dashboards | Tags |
| --- | --- | --- |
| `verl` | `verl_trainer_v1_with_vllm_engine`, `verl_trainer_v1_with_sglang_engine` | `RL-Insight`, `verl`, `vllm` / `sglang` |
| `verl-omni` | `verl_omni_trainer_v1_with_vllm_omni_engine` | `RL-Insight`, `verl-omni`, `vllm-omni` |
| `agent_loop_trajectory` | `agent_loop_trajectory` | `RL-Insight`, `agent-loop`, `trajectory` |
| `quick_start_demo` | `quick_start_demo` | `RL-Insight`, `quick-start`, `demo` |

For the sample script in this guide, open **quick_start_demo** → `quick_start_demo` and set the time range to a recent window such as **Last 5 minutes** while the script is still running. For framework-specific runs, open the folder that matches that integration.

Bundled dashboard JSON files live in the package directory, one subdirectory per Grafana folder:

```text
rl_insight/config/services/grafana/dashboards/verl
rl_insight/config/services/grafana/dashboards/verl-omni
rl_insight/config/services/grafana/dashboards/agent_loop_trajectory
rl_insight/config/services/grafana/dashboards/quick_start_demo
```

At startup, RL-Insight copies them into the runtime dashboards directory and provisions Grafana from there. Grafana creates one folder per subdirectory (`foldersFromFilesStructure`).

```text
~/.rl-insight/runtime/dashboards
```

To add a dashboard, you can still put its JSON in an existing package subdirectory, or add a new subdirectory for a new folder, then restart the stack. For integration-owned dashboards, the recommended approach is to pass an external directory with `--extra-dashboard-dir`; this keeps the installed package unchanged. See [Custom Grafana Dashboards](./custom_dashboards.md) for details.

Prometheus metrics and Tempo traces are persisted under `~/.rl-insight/data` by default. Stopping the server does not delete collected data.

Prometheus scrape targets registered by trainers or `rl-insight server targets add` are stored separately in `~/.rl-insight/data/targets/prometheus-targets.yml`. The generated `prometheus.yml` references this persistent file through Prometheus file-based service discovery, so restarting the server stack does not clear registered targets. Target updates are written atomically under a cross-process lock. Existing registration paths continue to reload Prometheus for API compatibility, while file-based service discovery also refreshes the target file every five seconds.

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

See [Configuration](configuration.md) for server and client settings, defaults, and override precedence. The settings below belong to the training client, not the server YAML.

Pass overrides through `insight.init(config=...)`:

```python
insight.init(
    project="verl",
    experiment_name="ppo-smoke-test",
    config={
        "server": {
            "namespace": "rl_insight_monitor",
            "backend": "ray",
            "url": "http://<server-ip>:18080",
        },
        "prometheus": {
            "metrics_report_port": 9092,
        },
    },
)
```

Environment variables take precedence for common deployment settings:

| Variable | Purpose |
|---|---|
| `RL_INSIGHT_SERVER_URL` | RL-Insight server URL, for example `http://<server-ip>:18080`. |
| `RL_INSIGHT_SERVER_BACKEND` | Trainer-side monitor client backend (`server.backend`), for example `ray`. Also applies to worker processes that call `rl_insight.init()` without a config. |
| `RL_INSIGHT_HARDWARE_ENABLED` | Enable (`1`/`true`/`yes`/`on`) or disable (`0`/`false`/`no`/`off`) hardware monitoring (`hardware.enabled`, disabled by default). The environment variable takes precedence over the config. |

## Troubleshooting

If `server start` reports missing or incompatible services, run:

```bash
rl-insight server install
```


If metrics do not appear, check that the monitor hub process is reachable from Prometheus and that the Prometheus configuration points to the hub `/metrics` endpoint.

## Next Steps: Monitor a Real Training Run

For framework-specific setup and monitoring switches, follow the maintained upstream guides:

| Framework | Getting started | Upstream RL-Insight integration |
| --- | --- | --- |
| verl | [verl](integrations/verl.md) | [Official integration guide](https://github.com/verl-project/verl/blob/main/docs/advance/rl_insight.md) |
| verl-omni | [verl-omni](integrations/verl-omni.md) | [Official integration guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/rl_insight.md) |

For verl metric names, types, calculations, and descriptions, see the existing [verl metric reference](metric-name-catalog.md).

For custom instrumentation, see the [Python API](api.md). Once monitoring is working, continue with the [verl async training tuning guide](../tuning_case/async_mode_tuning_tutorial.md) or [verl-omni multimodal tuning case](../tuning_case/multimodal_model_tuning_using_rl_insight.md).
