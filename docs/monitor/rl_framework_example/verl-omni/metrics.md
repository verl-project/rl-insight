# verl-omni Metric Reference

Use this reference with the [FlowGRPO example](index.md). The selected algorithm and rollout engine determine which metrics exist; text-generation panels need not contain data for diffusion workloads.

## Trainer and diffusion metrics

Scalar training values are exported as gauges with `project` and `experiment_name` labels. With the default namespace, replace `/` in the training key with `_` and prepend `rl_insight_monitor_`.

| Exported metric | Unit | Interpretation |
| --- | --- | --- |
| `rl_insight_monitor_training_global_step` | Steps | Latest reported training step |
| `rl_insight_monitor_timing_s_step` | Seconds | Complete training step duration |
| `rl_insight_monitor_timing_s_gen` | Seconds | Rollout generation duration |
| `rl_insight_monitor_critic_rewards_mean` | Reward units | Mean reward under the selected reward function |
| `rl_insight_monitor_critic_rewards_zero_std_ratio` | Fraction, 0–1 | Fraction of prompt groups with no reward variation; high values mean little relative-reward learning signal |
| `rl_insight_monitor_critic_rewards_std_mean` | Reward units | Mean within-group reward standard deviation |
| `rl_insight_monitor_timing_per_image_ms_gen` | Milliseconds/image | Generation cost normalized by image count |
| `rl_insight_monitor_perf_throughput` | Images/s/device for diffusion | Training throughput; do not compare directly with token throughput from text models |

Use training, timing, and diffusion panels in `verl_omni_trainer_v1_with_vllm_omni_engine`. Additional algorithm metrics may require Grafana Explore or a [custom dashboard](../../custom_dashboards.md). See the [upstream training metrics guide](https://github.com/verl-project/verl-omni/blob/main/docs/start/metrics.md) for algorithm-specific definitions.

## Other monitoring sources

| Source | Enablement | Expected data |
| --- | --- | --- |
| vLLM-Omni rollout | `disable_log_stats=False` and the RL-Insight logger | Native engine metrics across rollout replicas; names and units depend on engine version and model type |
| TransferQueue | V1 training plus `transfer_queue.metrics.enabled=True` | Queue/controller metrics, including operation-latency histograms where supported |
| RL state traces | Framework instrumentation and available Tempo endpoint | Generation, sleep/wake, and weight-update intervals on replica or rank lanes |
| CPU/NPU exporters | Separate exporter installation and registration | Hardware resource metrics |

Engine and TransferQueue metrics are scraped separately and retain their native names and labels. Inspect their `/metrics` endpoints for exact names; do not prepend the trainer namespace to them. Histogram families expose buckets/counts/sums where classic histograms are used. RL state traces are Tempo data, not scalar metrics.

## Reading an incomplete dashboard

First confirm project, experiment, and time range. Trainer gauges update as steps complete and are sampled by Prometheus; they do not preserve every intermediate training event. Queue panels require the V1/TransferQueue path, and model-specific engine panels can remain empty for diffusion runs. Missing hardware panels require [exporter setup](../../hardware/index.md), not another trainer flag.

Check the [configuration guide](../../configuration.md) for switches and network directions. Use Prometheus target health to diagnose scraping, and Tempo availability to diagnose traces independently.
