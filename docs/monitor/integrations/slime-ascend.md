# slime-ascend Integration

Use [slime-ascend](https://gitcode.com/Ascend/slime-ascend) for RL training on Ascend NPUs and enable its optional RL-Insight backend to collect training metrics, RL state timelines, and SGLang engine metrics in one experiment.

The integration is available in the upstream [RL-Insight integration merge request](https://gitcode.com/Ascend/slime-ascend/merge_requests/129). It reuses Slime's existing tracking, training, and rollout-server hooks, so it can coexist with W&B and TensorBoard without changing the main training loop.

## Enable RL-Insight

Install and start the RL-Insight service stack on a host reachable from the Slime training and rollout workers:

```bash
pip install "rl-insight>=0.3.0"
rl-insight server install
rl-insight server start --detach
```

The Slime dashboard was added by [RL-Insight PR #188](https://github.com/verl-project/rl-insight/pull/188) and will be released officially with RL-Insight 0.4.0. Until that version is published, install RL-Insight from source if you need the bundled dashboard.

Install the same SDK version in the Slime training environment and set the RL-Insight server API URL:

```bash
pip install "rl-insight>=0.3.0"
export RL_INSIGHT_SERVER_URL=http://<server-host>:<server-port>
```

Then append the Slime arguments to your existing `train.py` or `train_async.py` command:

```bash
--use-rl-insight \
--rl-insight-project slime \
--rl-insight-experiment-name <experiment-name>
```

| Argument | Required | Effect |
| --- | --- | --- |
| `--use-rl-insight` | No | Enables RL-Insight; the default is disabled and does not require the SDK. |
| `--rl-insight-project` | No | Project name; the default is `slime`. |
| `--rl-insight-experiment-name` | Yes when enabled | Experiment name used to group metrics, traces, and engine targets. |
| `--rl-insight-config` | No | Optional trainer-side YAML or JSON configuration file. |

For Ray Jobs, propagate `RL_INSIGHT_SERVER_URL` through `runtime_env.env_vars` so every worker process receives it. If you use `--rl-insight-config`, ensure the file exists at the same path on every worker node. See [Configuration](../configuration.md) for server and client settings and override precedence.

## Collected Data

- **Training and rollout scalars**: Slime forwards its existing training, rollout, evaluation, performance, loss, reward, response-length, truncated-ratio, learning-rate, and phase-duration metrics. Training scalars retain Slime's existing main-rank reporting policy.
- **RL execution states**: Training ranks report `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update`, and `train_batch` intervals. The default rollout path reports the client-observed `sglang_generate` interval.
- **SGLang engine metrics**: Slime discovers each managed engine's native `/metrics` endpoint and registers it with RL-Insight. Targets carry project, experiment, model, replica, worker type, and server-group labels.
- **Replica attribution**: Supported self-managed SGLang workers add the final serving replica to successful generate responses. Slime reports a client-side `sglang_generate_replica` span for those requests.

Replica attribution supports regular node-0 HTTP workers with one tokenizer worker and HTTP/1.1. Custom rollout paths that bypass Slime's default generate call are outside the initial integration scope.

## View The Dashboard

Open Grafana and select **Slime Training with SGLang** (UID `slime-training-with-sglang`). Set the project and experiment variables to match the Slime arguments, then select a time range that covers the run.

For a dashboard-ready source installation, use the repository checkout that contains PR #188:

```bash
git clone https://github.com/verl-project/rl-insight.git
cd rl-insight
pip install -e .
```

```text
http://<grafana-host>:<grafana-port>/d/slime-training-with-sglang?var-project=slime&var-experiment_name=<experiment-name>
```

For a complete Qwen3-0.6B GRPO launch script, see the upstream [integration merge request](https://gitcode.com/Ascend/slime-ascend/merge_requests/129).
