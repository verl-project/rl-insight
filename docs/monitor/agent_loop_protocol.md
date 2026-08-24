# Agent Loop Protocol

RL-Insight's Agent Loop Trajectory dashboard is framework-independent. Any
trainer or agent runtime can populate it by following this protocol; RL-Insight
does not depend on or sense any upper-layer agent framework such as Uni-Agent.

## Protocol ownership

| Concern | Owner | Rule |
|---|---|---|
| Lane ID, identity fields, span names, dashboard metrics | RL-Insight | Do not redefine these in an agent framework. |
| Trainer configuration and lazy initialization | Trainer integration (for example, verl) | Forward calls to RL-Insight; do not encode agent semantics. |
| Task, generation, and session lifecycle events | Agent runtime | Call the protocol at the correct business boundary. |

## Initialization

Initialize the normal RL-Insight monitor API before emitting events:

```python
import rl_insight

rl_insight.init(
    project="my-project",
    experiment_name="my-experiment",
)
```

Set `RL_INSIGHT_SERVER_URL` in every worker that emits data. A trainer-side
adapter may call `init` lazily, but it must pass the same project and experiment
identity to all workers.

## Session lifecycle

Create exactly one session object at the beginning of an agent session:

```python
from rl_insight import agent_loop_session

session = agent_loop_session(
    project="my-project",
    experiment_name="my-experiment",
    sample=17,
    session=0,
    traj=0,
    uid="request-17",
    global_steps=7,
    session_id="session-abc",
)
```

Keep `session.identity` immutable and use it for every span in that session.
When the session finishes, call `finish` exactly once:

```python
session.finish(
    trajectories=trajectories,
    status="success",
    runner_name="task",
    reward_source="reward_info",
    finished=True,
)
```

`trajectories` is a sequence of objects with these optional attributes:

| Attribute | Meaning | Dashboard use |
|---|---|---|
| `chain_id` | One-based framework chain ID | Converted to zero-based `traj` |
| `reward_score` | Final scalar reward | Trajectory title |
| `num_turns` | Number of model turns | Trajectory title |

If `chain_id` is absent, the list position is used as the zero-based trajectory
ID. RL-Insight emits all `agent_loop_*` hierarchy metrics from `finish`; agent
frameworks must not emit those metrics manually.

## Required identity fields

`agent_loop_session()` returns this identity mapping:

| Field | Requirement |
|---|---|
| `project` | String project identity; must match the dashboard variable. |
| `experiment_name` | String experiment identity; must match the dashboard variable. |
| `sample` | String sample identity. |
| `session` | String session identity within the sample. |
| `traj` | Initial zero-based trajectory ID. |
| `state_lane_id` | Canonical lane ID; derive later trajectories with `agent_loop_lane_id`. |
| `uid` | Optional request UID; use `""` when unavailable. |
| `global_steps` | Numeric trainer step, or `""` when unavailable. |
| `session_id` | Optional framework session ID; use `""` when unavailable. |

The lane format is:

```text
experiment=<experiment_name>/sample=<sample>/session=<session>/traj=<traj>
```

For a one-based `chain_id`, publish `traj = chain_id - 1`. All spans for one
session must use the same experiment, sample, session, and `global_steps`.

## Attribute typing

`trace_span()` accepts OpenTelemetry scalar values (`str`, `bool`, `int`,
`float`) or homogeneous sequences of those scalars. Complex objects must be
JSON encoded by the caller. In particular:

- `global_steps` must remain numeric when a step exists; do not convert it to a string.
- Empty optional string fields should be `""`, not `None`.
- Timestamps are Unix epoch nanoseconds.

The dashboard's step filter uses exact numeric matching. A string `global_steps`
value produces an empty result.

## Completed span contract

Use `trace_span()` for completed spans. The required names and attributes are:

| Span | Required source | Required attributes |
|---|---|---|
| `agent_session` | `session.finish()` | `monitor.trace_source="session"`, identity, `runner_name`, `status`, `num_trajectories`, `reward_source`, `finished` |
| `agent_task` | Task runner | `monitor.trace_source="task"`, identity, `task_name`, `image_ref`, `prompt_hash`, `status`, `reward`, `accuracy`, `finished`, `reward_posted`, `error` |
| `gateway_generation` | Model gateway | `monitor.trace_source="gateway"`, identity, `state_lane_id`, `traj`, `chain_id`, `turn`, `type`, `tools`, `content`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `status`, `error` |

Use these status values: `success`, `failure`, `empty`, `capacity_exhausted`,
or `error`. Keep timestamps monotonic within a span and include failures even
when the operation raises.

## Dashboard mapping

The bundled Agent Loop Trajectory dashboard queries Tempo by
`state_lane_id`, `experiment_name`, `project`, `global_steps`, and span source.
Session metadata gauges build the experiment → sample → session → trajectory
hierarchy in Grafana.

A minimal valid run therefore needs:

1. One `agent_loop_session()` call.
2. At least one `gateway_generation` span with a valid lane ID.
3. One `session.finish()` call.

Task spans are optional but enable their summary panel.

## Smoke test

The managed-stack smoke test generates one protocol-compliant Agent Loop run and
verifies that Tempo and Prometheus can query it:

```bash
pytest tests/monitor/special_e2e/test_monitor_smoke.py \
  -k agent_loop_dashboard_should_query_generated_protocol_data
```

The test requires a live RL-Insight server, Tempo, Prometheus, and Grafana.

## Related documentation

- [Use RL-Insight to monitor verl training](https://github.com/verl-project/verl/blob/main/docs/advance/rl_insight.md)
- [Uni-Agent RL-Insight instrumentation guide](https://github.com/verl-project/uni-agent/blob/rlinsight/docs/source/concepts/rl-insight-integration.md)
