# Python API

Import the public API with `import rl_insight as insight`. The examples below use the default Ray backend and the Prometheus/Tempo stack started in [Quick Start](quick_start.md). Backend selection and configuration precedence are covered in [Configuration](configuration.md).

## Initialize and finish

`init(project=None, experiment_name=None, config=None)` initializes monitoring once per process. `project` and `experiment_name` become labels on metrics and attributes on traces. `config` accepts a partial mapping or OmegaConf `DictConfig`.

```python
import ray
import rl_insight as insight

ray.init()
insight.init(
    project="my_project",
    experiment_name="run_001",
    config={"server": {"url": "http://<server-ip>:18080"}},
)
try:
    insight.metric_gauge("reward_mean", 0.8, worker="trainer_0")
finally:
    insight.finish()
```

Replace the address with the server API URL. Environment overrides still take precedence over this mapping. For Ray, a missing URL or uninitialized Ray leaves monitoring inactive. Calls to metric and trace helpers are no-ops while monitoring is inactive. A repeated `init()` after successful initialization warns and is ignored.

`finish()` resets monitoring in the current process and asks the client to close its resources, including push pollers and sinks. It does not stop the server stack or the Ray hub, and it does not delete collected data. Framework integrations normally manage initialization; use their upstream guides instead of initializing a second time.

## Record metrics

| Function | Parameters | Use |
| --- | --- | --- |
| `metric_count` | `name, amount=1.0, documentation="", **labels` | Increment a counter, such as completed requests; use nonnegative increments |
| `metric_gauge` | `name, value, documentation="", **labels` | Set the latest value, such as reward or queue depth |
| `metric_histogram` | `name, value, documentation="", **labels` | Observe one sample, such as request latency |
| `metric_log` | `name, value, *, step, **labels` | Record a scalar for a specific training step through the trace path; stored in Tempo with the Ray backend |

```python
insight.metric_count("completed_requests", amount=1, worker="rollout_0")
insight.metric_gauge("queue_depth", value=8, worker="rollout_0")
insight.metric_histogram("request_latency_seconds", value=0.25, worker="rollout_0")
insight.metric_log("reward_mean", value=0.8, step=12, worker="trainer_0")
```

Use consistent label keys for each metric and stable values such as worker roles or replica IDs. Avoid per-request IDs as Prometheus labels. `documentation` supplies metric help text; omitting it uses generated text.

With the default Ray namespace, gauges such as `queue_depth` are exposed as `rl_insight_monitor_queue_depth`. Prometheus samples gauges at scrape time, so intermediate updates can be missed. Use `metric_log` for step-tagged trace records; it is not a Prometheus gauge and requires a working trace export path. Histograms expose `_bucket`, `_sum`, and `_count` series for aggregation and percentile queries. Push sinks may translate names and types differently; see [Push Metric Distributions](push-distributions.md).

## Trace a runtime state

`trace_state(state_name, *, state_lane_id=None, **labels)` is a context manager that records an interval on a logical timeline lane:

```python
with insight.trace_state("rollout_generate", state_lane_id="replica_0", step=12):
    responses = generate_responses()  # Your training code.
```

The span is emitted when the block exits. The default lane is the current process ID. Same-name overlapping states on one lane merge into one interval; a different state entered while that lane is occupied is dropped. Use distinct lanes for independently active workers. `state_name`, `state_lane_id`, and `monitor.trace_segment` are reserved trace attributes.

## Trace a function call

`trace_op(name=None, *, extra_labels=None, **static_labels)` decorates synchronous or asynchronous functions and records one duration span per call:

```python
@insight.trace_op("update_policy", component="actor")
def update_policy(batch):
    return train_on_batch(batch)  # Your training code.
```

The default name is the function's qualified name. `extra_labels`, when supplied, receives the first positional argument (often `self`) and returns a mapping merged over static labels. It is not called without positional arguments. Exceptions and cancellation still emit a span; the wrapped function's result or exception is preserved. Use this helper for individual call durations rather than state-lane occupancy.

## Report a completed span

`trace_span(*, name, start_time_ns, end_time_ns, attributes)` accepts caller-provided timestamps and attributes:

```python
import time

start_ns = time.time_ns()
perform_work()  # Your training code.
insight.trace_span(
    name="custom_operation",
    start_time_ns=start_ns,
    end_time_ns=time.time_ns(),
    attributes={"worker": "trainer_0", "step": 12},
)
```

Times are Unix epoch nanoseconds, not monotonic-clock readings. The API does not validate or normalize timestamps; supply an end time at or after the start. Attribute values must be OpenTelemetry scalars (`str`, `bool`, `int`, `float`) or homogeneous sequences. Unlike `trace_state` and `trace_op`, this helper does not add a `monitor.trace_segment` marker. Initialization labels are merged into the record.

For structured agent trajectories, see [Agent Loop Protocol](agent_loop_protocol.md).
