# verl-SpeCo Integration

Use the current [verl-SpeCo repository](https://github.com/verl-project/verl-SpeCo) and its maintained guides for environment setup and training:

1. Follow the [verl-SpeCo Quickstart](https://github.com/verl-project/verl-SpeCo/blob/main/docs/usage.md#quickstart) to install verl-SpeCo and run co-training or standalone drafter training. Monitoring requires upstream verl `release/v0.9.0` or newer.
2. Follow the [RL-Insight monitoring guide](https://github.com/verl-project/verl-SpeCo/blob/main/docs/usage.md#online-monitoring-with-rl-insight) to enable monitoring for drafter co-training. Set `RL_INSIGHT_SERVER_URL` and add `rl_insight` to `trainer.logger`:

   ```bash
   export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
   python3 -m verl_speco.main \
       trainer.logger='["console","rl_insight"]' \
       actor_rollout_ref.rollout.disable_log_stats=False \
       ...
   ```

When no RL-Insight server is configured, verl-SpeCo's monitoring hooks degrade to no-ops. The upstream guide also describes the exported drafter metrics, traces, and dashboard panels.

For RL-Insight service setup, see [Quick Start](../quick_start.md); for server and client options, see [Configuration](../configuration.md).
