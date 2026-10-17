# verl Metric Reference

Use this reference with the [training example](index.md). Metrics depend on the trainer, algorithm, and engine version; an unused component will not produce data just because its dashboard panel exists.

## Trainer metrics

The `rl_insight` logger converts scalar training values into Prometheus gauges. Slashes in log keys become underscores, and the default namespace adds `rl_insight_monitor_`. For example, `timing_s/step` becomes `rl_insight_monitor_timing_s_step`. The `project` and `experiment_name` labels identify the run.

| Exported metric | Unit | Interpretation |
| --- | --- | --- |
| `rl_insight_monitor_training_global_step` | Steps | Latest reported training step; use it to check progress |
| `rl_insight_monitor_timing_s_step` | Seconds | Total step time; compare with individual stage timings |
| `rl_insight_monitor_timing_s_gen` | Seconds | Rollout generation time; a rise can reflect longer responses or slower inference |
| `rl_insight_monitor_timing_s_update_actor` | Seconds | Actor update time |
| `rl_insight_monitor_critic_rewards_mean` | Reward units | Batch mean reward; its scale depends on the reward function |
| `rl_insight_monitor_actor_pg_loss` | Objective-dependent | Policy loss; interpret alongside reward and clipping statistics |
| `rl_insight_monitor_perf_throughput` | Tokens/s/device | Training throughput; compare runs with comparable sequence and workload settings |
| `rl_insight_monitor_response_length_mean` | Tokens | Mean generated response length; helps explain changes in generation time |

These appear in training, timing, or performance panels when the trainer emits their source keys. Gauges are latest sampled values, not an event log of every step; Prometheus may miss intermediate updates between scrapes. Do not apply `rate()` to a loss or step-duration gauge.

For a broader list, see the [Metric Name Catalog](../../metric-name-catalog.md). That catalog also supports offline performance analysis and is not a guarantee that every listed series is emitted by every run.

## Engine and queue metrics

Engine metrics come from separate scrape endpoints and retain their native names and labels. Enable rollout statistics with `actor_rollout_ref.rollout.disable_log_stats=False`.

| Metric family | Type / unit | Interpretation |
| --- | --- | --- |
| `vllm:num_requests_running` | Gauge / requests | Requests currently running |
| `vllm:num_requests_waiting` | Gauge / requests | Queued requests; sustained growth suggests contention or insufficient serving capacity |
| `vllm:e2e_request_latency_seconds` | Histogram / seconds | Distribution of request completion latency |
| `vllm:time_to_first_token_seconds` | Histogram / seconds | Distribution of time to first token |
| `tq_controller_request_duration_seconds` | Histogram / seconds | TransferQueue controller operation latency; only when TransferQueue is used and its metrics are enabled |

For classic histograms, query `_bucket`, `_sum`, and `_count` series; use `histogram_quantile()` for percentiles. Native engine labels and metric availability vary by version. Select the SGLang dashboard for SGLang metrics rather than assuming vLLM names apply. Use the actual `/metrics` response to inspect replica/model labels.

## Traces and missing data

RL state intervals are stored in Tempo, not Prometheus. They describe instrumented operations on worker or replica lanes. They do not by themselves represent token-level engine timing. Hardware panels require separately registered exporters.

If data is absent, check in order: selected project/experiment and time range, completed training steps, Prometheus target health, the source-specific switches in [Configuration](../../configuration.md), and whether that trainer actually emits the metric. For traces, check Tempo and worker instrumentation separately.
