# Inference Engine Tracing

RL-Insight stores inference-engine request traces from verl rollouts in Tempo and
shows them in the **vLLM request traces** row of the
`verl_trainer_v1_with_vllm_engine` dashboard. The rollout engine produces the
span; RL-Insight stores the engine's native attributes as-is and does not create a
second timing implementation.

## Data path

```text
verl rollout replica (vLLM)
  └─ OTLP/HTTP /v1/traces
       └─ Tempo (OTLP :4318, query :3200)
            └─ Grafana "vLLM request traces" row
```

The rollout replica discovers the Tempo OTLP endpoint from the RL-Insight server
(`GET /api/v1/services` → `otlp_port`), so there is no separate Tempo URL to
configure for verl.

## Enable tracing

Prerequisites:

- The RL-Insight server stack is running with Tempo enabled (`rl-insight server start`).
- `RL_INSIGHT_SERVER_URL` points at the RL-Insight server in every rollout worker.
- `trainer.logger` contains `rl_insight`.
- verl includes the `actor_rollout_ref.rollout.trace.enable_otel` switch
  ([verl PR #8158](https://github.com/verl-project/verl/pull/8158)).

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080

python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.trace.enable_otel=True \
    ...
```

When the switch is enabled, each rollout replica:

1. Reads `otlp_port` from `GET /api/v1/services` and exports to
   `http://<server-host>:<otlp_port>/v1/traces`.
2. Sets `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` because the engine
   exporter defaults to gRPC.
3. Sets `OTEL_RESOURCE_ATTRIBUTES` with `project`, `experiment_name`, and
   `replica`. The first two default to `trainer.project_name` and
   `trainer.experiment_name`.

Tracing is opt-in. If the server is disabled, the status has no OTLP port, or the
export fails, rollout continues without tracing.

## Correlation schema

### Resource attributes

These attributes are attached to every span exported by the replica.

| Attribute | Type | Meaning |
|---|---|---|
| `project` | string | Run project, from `actor_rollout_ref.rollout.trace.project_name` |
| `experiment_name` | string | Experiment, from `actor_rollout_ref.rollout.trace.experiment_name` |
| `replica` | string | Rollout replica rank, used to tell engines apart |
| `vllm.instrumenting_module_name` | string | vLLM tracer module, set by the engine |
| `vllm.process_id` | string | Engine process id, set by the engine |

### Per-request correlation attributes

The rollout adapter attaches these attributes when it has the information.
RL-Insight displays them when present and never fabricates them.

| Attribute | Type | Meaning |
|---|---|---|
| `global_steps` | int | Trainer global step; numeric, never stringified |
| `sample` | string | Sample id inside the batch |
| `session` | string | Agent session id |
| `state_lane_id` | string | `experiment=<name>/sample=<id>/session=<id>/traj=<id>` lane |
| `traj` | int | Trajectory id |
| `turn` | int | Model turn inside the trajectory, starting at 0 |
| `gen_ai.request.id` | string | Engine request id |

`sample` is the batch `index` (the dataset `extra_info.index`, or the batch
position when no index column exists). GRPO repeats each sample `rollout.n`
times; `traj` distinguishes those copies. `session` is the UUID for one
agent-loop run and is shared by every turn of a multi-turn trajectory, so it
identifies one `(step, sample, traj)` trajectory.

## Native vLLM fields

vLLM emits one `llm_request` span per finished request. RL-Insight keeps the
native attribute names and units; it does not rename or reinterpret them.

| Attribute | Type and unit | Meaning |
|---|---|---|
| `gen_ai.latency.e2e` | float, seconds | End-to-end request time |
| `gen_ai.latency.time_to_first_token` | float, seconds | Request start to the first generated token |
| `gen_ai.latency.time_in_queue` | float, seconds | Time spent waiting before execution |
| `gen_ai.latency.time_in_scheduler` | float, seconds | Scheduler time, when supplied |
| `gen_ai.latency.time_in_model_prefill` | float, seconds | Prefill time |
| `gen_ai.latency.time_in_model_forward` | float, seconds | Forward-pass time, when supplied |
| `gen_ai.latency.time_in_model_execute` | float, seconds | Forward, synchronization, and sampling time, when supplied |
| `gen_ai.latency.time_in_model_decode` | float, seconds | Decode time |
| `gen_ai.latency.time_in_model_inference` | float, seconds | Scheduling-to-last-token inference time |
| `gen_ai.usage.prompt_tokens` | int, tokens | Prompt token count |
| `gen_ai.usage.completion_tokens` | int, tokens | Generated token count |
| `gen_ai.request.max_tokens` | int, tokens | Requested maximum output tokens |
| `gen_ai.request.top_p` | float | Sampling parameter, when set |
| `gen_ai.request.temperature` | float | Sampling parameter, when set |
| `gen_ai.request.n` | int | Number of sequences, when set |
| `gen_ai.request.id` | string | Engine request id |
| `gen_ai.response.model` | string | Served model name, when supplied |
| `gen_ai.response.finish_reasons` | string | Finish reason, when supplied |

Version-dependent fields are absent when the running engine does not supply
them. The dashboard shows `-` and never derives or back-fills a value.

Rules:

- Unknown engine attributes stay in Tempo under their native names. The raw span
  is always the source of truth.
- Native timing attributes are not copied into Prometheus labels because request
  ids, per-request timings, and text would make the label cardinality unbounded.
- If an engine or version uses different attribute names, extend the dashboard
  allowlist; do not rename the stored attribute.

## Dashboard

Open Grafana → `verl_trainer_v1_with_vllm_engine` → **vLLM request traces**.
The row is collapsed by default; expand it to load the single **Inference
requests** table. It lists matching requests newest first, links each row to its
raw trace, and sorts by **E2E (s)** when that header is clicked.

The query has a fixed cap of 20000 spans. Tempo's search limit is numeric and
Grafana cannot interpolate a dashboard variable into it, so the cap lives in the
query and the table paginates the result. Tempo applies the cap before Grafana
sorts, so narrow the Step or time range when a run exceeds it; edit the query
`limit` to change the cap.

Filters:

- **training: Project**, **training: Experiment Name**, and **vllm: Rollout
  Replica** match the resource attributes.
- **Step** is single-select; **Sample** is multi-select. Both include All.
- Sample uses anchored regex matching, so selecting `100` does not also match
  `1006`.
- The dashboard time range limits the search.

Step uses `-1` for All because TraceQL does not support regex matching on the
integer `global_steps`; the query becomes
`(span.global_steps = $step || $step = -1)`. Sample All expands to `.*`, and
the query adds `|| "$sample" = ".*"` so requests written before the adapter
attached `sample` stay visible. Selecting a specific sample still requires the
attribute. The Sample dropdown lists values from the tag window
(`timeRangeForTags`, the retention window); prefer the table's **Sample** column
when it must match the current step.

The correlation filters need the rollout adapter to attach the matching
attributes. The adapter must emit `sample` as a string and `global_steps` as an
integer; other types break exact or regex matching. If a filter has no options,
the attribute is absent from the queried spans; fix the adapter or widen the
time range.

The server provisions the Tempo datasource with
`jsonData.timeRangeForTags` set to `tempo.retention_time` (30 days by default).
Grafana's default `1800` is treated as "no range", so Tempo returns only
intrinsic tags once the recent ingester data ages out and the Step/Sample
variables fail with `Scope for tag ... not found`. Keep this setting on Tempo
datasources that serve the trace dashboard.

Columns, in display order: start time, trace link, replica, step, turn, sample,
end-to-end latency, TTFT, queue wait, prefill, decode, **decode tok/s**
(completion tokens divided by decode time), prompt tokens, completion tokens,
max tokens, finish reasons, model inference, the version-dependent
scheduler/forward/execute split, and model. `Start` and `Trace` are pinned left;
step/turn/sample follow replica.

Trajectory, session, `gen_ai.request.id`, and `state_lane_id` are not table
columns. The **Trace** view shows them, and the request id can be matched
against rollout logs.

### Query cost

Tempo persists spans; Grafana caches panel data in memory only, so a new search
scans the WAL and blocks for the selected range. Reopening the row with
unchanged filters and time range reuses the cached frame. Refresh, time-range
changes, and filter changes rerun the query; Step/Sample tag-value queries are
separate.

For large steps, select a Step first, narrow the time range, and use pagination
or Sample filtering instead of raising the cap.

The Tempo block config stores `project`, `experiment_name`, `replica`, and
`sample` in dedicated Parquet columns, so TraceQL reads those columns instead of
decoding generic attributes. Dedicated columns require a vParquet3 or newer
block; Tempo 2.4+ uses vParquet3 by default. They apply to blocks written after
the change, and the server removes the setting for older versions. Restart the
server after editing the packaged `tempo.yaml`; the runtime copy is regenerated
on start.

### Empty row

If the row is empty, verify in this order:

1. `actor_rollout_ref.rollout.trace.enable_otel=True` and `rl_insight` in
   `trainer.logger`.
2. `curl $RL_INSIGHT_SERVER_URL/api/v1/services` returns a non-null
   `otlp_port`.
3. The rollout node can reach `<server-host>:<otlp_port>` and Tempo is enabled.
4. The Project, Experiment Name, Rollout Replica, Step, and Sample filters
   match the run.
5. The engine version emits `llm_request` spans; open Tempo directly and search
   for `{ name = "llm_request" }` to confirm.

## Text, sampling, and retention controls

Engine-native `llm_request` spans contain request metadata and token counts, not
prompt or response text. The emitter therefore owns the text policy; RL-Insight
stores what the exporter sends and cannot redact text that was never sent.

When a rollout adapter attaches turn or sentence text, it must apply:

- **Mode**: `off` (default) keeps metadata only; `redacted` removes configured
  secret and credential patterns; `full` keeps text subject to a byte cap.
- **Length limit**: cap text per span and set `text_truncated=true` plus a
  `text_hash` when truncated.
- **Sampling budget**: apply a per-worker and per-step span budget, and keep
  failed requests when the budget is exceeded.
- **No Prometheus labels**: text and per-request ids stay in Tempo only.
- **No fabricated sentence latency**: sentence rows use engine request or token
  timing; if token timestamps are unavailable, show turn-level timing without
  inventing sentence latency.

Detailed timings add measurable overhead, so tracing stays opt-in. Tempo
retention defaults to 30 days and is controlled by `tempo.retention_time` in the
server config.

## Engine compatibility

| Engine | Status | Notes |
|---|---|---|
| vLLM 0.32 (the release pinned by verl) | Supported | `llm_request` with e2e, queue, TTFT, prefill, decode, inference, tokens, and request id |
| Newer vLLM releases | Best effort | Additional `gen_ai.latency.*` fields appear automatically in the raw trace |
| SGLang | Not yet mapped | Add an allowlist entry and dashboard query when the engine exposes equivalent spans |

## See also

- [Quick Start](./quick_start.md)
- [Agent Loop Protocol](./agent_loop_protocol.md)
- [verl PR #8158](https://github.com/verl-project/verl/pull/8158)
