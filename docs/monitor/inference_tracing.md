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
(`GET /api/v1/services` → `otlp_port`, with the SDK version header), so the
default setup has no separate Tempo URL to configure for verl. Set
`actor_rollout_ref.rollout.engine_kwargs.vllm.otlp_traces_endpoint` to override
the discovered endpoint.

## Enable tracing

Prerequisites:

- The RL-Insight server stack is running with Tempo enabled (`rl-insight server start`).
- The RL-Insight SDK is 0.4.0 or newer on the driver and every rollout worker;
  the Monitor Hub and the services API reject older clients.
- `RL_INSIGHT_SERVER_URL` points at the RL-Insight server in every rollout worker.
- `trainer.logger` contains `rl_insight`.
- The verl build includes the `actor_rollout_ref.rollout.trace.enable_otel`
  switch. The switch is implemented for the vLLM rollout engine only.

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080

python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.trace.enable_otel=True \
    ...
```

With endpoint discovery, each rollout replica:

1. Reads `otlp_port` from `GET /api/v1/services` (sending the SDK version
   header) and exports to `http://<server-host>:<otlp_port>/v1/traces`.
2. Sets `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` because the engine
   exporter defaults to gRPC.
3. Sets `OTEL_RESOURCE_ATTRIBUTES` with `project`, `experiment_name`, and
   `replica`. The first two come from the job's Run, so they carry the same
   execution suffix the Hub stamps on training telemetry; when the Run is not
   available they fall back to `actor_rollout_ref.rollout.trace.project_name`
   and `actor_rollout_ref.rollout.trace.experiment_name`, which default to
   `trainer.project_name` and `trainer.experiment_name`.

RL-Insight resolves the Run through the SDK's `get_run_context()`; see the
[Python API](api.md) for the process-level behavior.

With endpoint discovery, tracing is opt-in. If the server is disabled, the
status has no OTLP port, or the export fails, rollout continues without tracing.

## Correlation schema

### Resource attributes

These attributes are attached to every span exported by the replica.

| Attribute | Type | Meaning |
|---|---|---|
| `project` | string | Run project; falls back to `actor_rollout_ref.rollout.trace.project_name` |
| `experiment_name` | string | Run experiment name including the Hub's execution suffix; falls back to `actor_rollout_ref.rollout.trace.experiment_name` |
| `replica` | string | Rollout replica rank, used to tell engines apart |
| `vllm.instrumenting_module_name` | string | vLLM tracer module, set by the engine |
| `vllm.process_id` | string | Engine process id, set by the engine |

### Per-request correlation attributes

These attributes are not part of vLLM's native span schema. They are optional:
RL-Insight displays them when the tracing integration supplies them and never
fabricates them; when they are absent, the Step, Sample, and Turn columns show
`-` and the Step filter offers only All.

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
them. The dashboard shows `-` for missing native fields; its only derived column
is **decode tok/s**.

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
raw trace, and sorts by **E2E (s)** when that header is clicked. **Step**
defaults to the newest step when its correlation attribute is present, so
expanding the row does not scan every request of the run; select **All** only
when the full history is needed.

The query sets Tempo's search limit to 20000 results. Tempo's search limit is
numeric and Grafana cannot interpolate a dashboard variable into it, so the
limit lives in the query and the table paginates the result. Tempo applies the
limit before Grafana sorts, so narrow the Step or time range when a run exceeds
it; edit the query `limit` to change the cap.

Filters:

- **training: Project** and **training: Experiment Name** match the resource
  attributes. Experiment values include the Run's execution suffix; spans
  written before engine tracing adopted the Run name keep the un-suffixed
  value, so select that entry to inspect historical requests.
- **Step**, **Replica**, and **Sample** are row-scoped filters. Step is
  single-select and defaults to the newest step when `global_steps` is present;
  its **All** entry sits at the end of the list. Replica and Sample are
  multi-select and default to All.
- Replica and Sample use anchored regex matching, so selecting `100` does not
  also match `1006`.
- The dashboard time range limits the search.

Step uses `-1` for All because TraceQL does not support regex matching on the
integer `global_steps`; the query becomes
`(span.global_steps = $step || $step = -1)`. All is a static option rather than
`includeAll`, because Grafana preselects the built-in All option whenever it is
enabled and the variable has no saved value. The static entry keeps the newest
step as the default while All stays one click away. Sample All expands to `.*`,
and the query adds `|| "$sample" = ".*"` so requests that predate `sample` stay
visible. Selecting a specific sample still requires the attribute. Replica All
expands to `.*` and uses the same fallback, so spans without the resource
attribute stay visible. The Step, Sample, and Replica dropdowns list values from
the selected project and experiment inside the tag window (`timeRangeForTags`,
the retention window); prefer the table's **Sample** column when it must match
the current step.

The correlation filters require the tracing integration to attach the matching
attributes. It must emit `sample` as a string and `global_steps` as an integer;
other types break exact or regex matching. If a filter has no options, the
attribute is absent from the queried spans; check the integration or widen the
time range.

The server provisions the Tempo datasource with
`jsonData.timeRangeForTags` set to `tempo.retention_time` (30 days by default).
Grafana's default `1800` is treated as "no range", so Tempo returns only
intrinsic tags once the recent ingester data ages out and the Step/Sample/Replica
variables fail with `Scope for tag ... not found`. Keep this setting on Tempo
datasources that serve the trace dashboard. The server also sets Tempo's
`query_frontend.search.max_duration` to the same retention window, so the table
can query the full retention period instead of Tempo's 168-hour default.

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
changes, and filter changes rerun the query; Step/Replica/Sample tag-value
queries are separate.

When `global_steps` is present, the table starts on the newest step. Selecting
Step All scans the whole run, so narrow the time range and use pagination or
Replica/Sample filtering instead of raising the cap.

The Tempo block config stores `project`, `experiment_name`, `replica`, and
`sample` in dedicated Parquet columns, so TraceQL reads those columns instead of
decoding generic attributes. Dedicated columns require a vParquet3 or newer
block; Tempo 2.4+ writes vParquet3 or newer blocks by default. They apply to
blocks written after the change, and the server removes the setting for older
versions. Restart the server after editing the packaged `tempo.yaml`; the
runtime copy is regenerated on start.

### Empty row

If the row is empty, verify in this order:

1. `actor_rollout_ref.rollout.trace.enable_otel=True` and `rl_insight` in
   `trainer.logger`.
2. `curl -H "X-RL-Insight-Client-Version: <sdk-version>" $RL_INSIGHT_SERVER_URL/api/v1/services`
   returns a non-null `otlp_port`; the services API rejects requests without
   the version header with HTTP 409.
3. The rollout node can reach `<server-host>:<otlp_port>` and Tempo is enabled.
4. The Project, Experiment Name, Replica, Step, and Sample filters match the
   run.
5. The engine version emits `llm_request` spans; open Tempo directly and search
   for `{ name = "llm_request" }` to confirm.
6. If requests appear but Step, Sample, or Turn is empty, the tracing integration
   is not attaching the correlation attributes.

## Text and retention

Engine-native `llm_request` spans contain request metadata and token counts, not
prompt or response text. RL-Insight stores exactly what the exporter sends and
does not add, infer, or redact text.

Detailed timings add measurable overhead, so tracing stays opt-in. Tempo
retention defaults to 30 days and is controlled by `tempo.retention_time` in the
server config.

## Engine compatibility

| Engine | Status | Notes |
|---|---|---|
| vLLM 0.32 (the version pinned by verl) | Supported | `llm_request` with e2e, queue, TTFT, prefill, decode, inference, tokens, and request id |
| Newer vLLM releases | Best effort | Additional `gen_ai.latency.*` fields appear automatically in the raw trace |
| SGLang | Not yet mapped | Add an allowlist entry and dashboard query when the engine exposes equivalent spans |

The vLLM exporter emits the native request attributes; the verl launcher adds
the `project`, `experiment_name`, and `replica` resource attributes. Step,
sample, turn, and trajectory correlation attributes come from the tracing
integration, not from vLLM.

## Related documentation

- [Quick Start](./quick_start.md)
- [Agent Loop Protocol](./agent_loop_protocol.md)
