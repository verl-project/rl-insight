# Quick Start

Start a monitoring server, send metrics and traces, and view them in Grafana.
If your training framework already integrates RL-Insight, configure that
integration with the server URL instead of adding the example instrumentation.

## 1. Install and start the server

From the repository root:

```bash
pip install -e .
rl-insight server install
rl-insight server start
```

Keep this terminal running. Note the server and Grafana URLs printed at startup.
For offline installation, Docker, custom ports, or a background server, see
[Server Installation](./server_installation.md).

## 2. Connect your training environment

Install RL-Insight in the Python environment that runs training. Set the server
URL before starting training, using the address printed in step 1:

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080
```

Make this setting available to every training worker. If training and the server
run on different machines, use a reachable server address and check the
[network requirements](./server_installation.md#connect-training-jobs).

## 3. Send metrics and traces

Save the following as `demo.py` and run `python demo.py` in a second terminal.
Leave it running while viewing the dashboard; press `Ctrl+C` to finish.

```python
import time

import ray
import rl_insight as insight

ray.init(namespace="rl-insight-monitor")
insight.init(project="verl", experiment_name="quick_start_demo")

step = 0
labels = {"worker": "trainer_0"}
try:
    while True:
        with insight.trace_state("rollout_generate", state_lane_id="replica_0", step=step):
            time.sleep(2)

        insight.metric_count("train_step_total", amount=1, **labels)
        insight.metric_gauge("reward_mean", value=1.0 + step * 0.01, **labels)
        insight.metric_histogram("step_latency_ms", value=200 + (step % 5) * 20, **labels)
        step += 1
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    insight.finish()
    ray.shutdown()
```

For training metrics plotted by step, use
`insight.metric_log("reward", 0.5, step=step)`.

In distributed training, initialize monitoring in the workers as required by
your integration. Wait for all workers to finish before calling `finish()` in
the process that first initialized monitoring for the job.

## 4. View the dashboard

Open the printed Grafana URL (default `http://<server-ip>:3000`). The default
login is `admin` / `admin`.

1. Open **Dashboards** in the left navigation.
2. For this example, open **quick_start_demo** → `quick_start_demo`.
   For framework training, use the corresponding folder, such as **verl**.
3. Set the time range to **Last 5 minutes** while the example is running.
4. Use **Project** and **Experiment** to choose the execution. Experiment choices
   include the experiment name, UTC start time, and a short ID, so repeated
   executions with the same name can be distinguished. The selection applies to
   both metrics and traces. Select multiple executions or **All** to compare jobs.

Execution choices depend on the selected project and collected metrics, not the
chart time range. Set the time range to cover the selected training execution. To add your own panels, see
[Custom Grafana Dashboards](./custom_dashboards.md).

## Monitor concurrent training jobs

1. Start the server once and use the same server URL for every job.
2. Start each job independently using your normal training launcher. Leave
   `prometheus.metrics_report_port` unset; if fixed ports are required, assign
   different ports to jobs on the same machine.
3. In Grafana, choose an execution under **Experiment**. Project and experiment
   names can be the same across jobs; each execution has its own identity.
4. Finish each job after its workers have completed. Other jobs keep collecting,
   and completed jobs remain viewable within the data retention period.

To try this with the example, run `python demo.py` in two terminals configured
with the same server URL, then switch between their executions in Grafana.

## Configure the server URL in Python

If you prefer configuration over an environment variable:

```python
insight.init(
    project="verl",
    experiment_name="ppo-training",
    config={"server": {"url": "http://<server-ip>:18080"}},
)
```

`RL_INSIGHT_SERVER_URL`, when set, takes precedence over this URL. Pass the
configuration or environment to workers too; connecting to an existing Ray
cluster does not automatically copy the driver's environment to workers.

## Stop and restart

Stop training before stopping the monitoring server. In the server terminal,
press `Ctrl+C`. For a server started with `--detach`, run:

```bash
rl-insight server stop
```

Restart with the same configuration and data directory to keep history.
Surviving training processes reconnect after the server returns to the same
address. A collection gap is possible; abnormal exits can lose the last metrics
or traces, and a failed monitoring process is not automatically recreated.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Server reports missing or incompatible services | Run `rl-insight server install`, then start again. |
| Dashboard is missing | Open **Dashboards**, then expand the appropriate folder. |
| Execution is missing from **Experiment** | Check **Project** and confirm the training job has sent metrics. Changing the chart time range does not change the execution list. |
| Metrics are missing | Verify the server URL in every worker and that Prometheus can reach the training metrics endpoints. Open `http://<server-ip>:9090/targets` to check their status. |
| Traces are missing | Verify that training processes can reach the trace port (default `4318`), select the correct execution and time range, and allow time for export. |
