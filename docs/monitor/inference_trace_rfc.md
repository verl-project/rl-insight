# RFC: Sentence-Level Inference Traces and Dashboard for verl

- **Status:** Draft
- **Owner:** RL-Insight maintainers
- **Target release:** RL-Insight 0.3
- **Related change:** [verl #8158](https://github.com/verl-project/verl/pull/8158)

## Summary

verl #8158 adds the `actor_rollout_ref.rollout.trace.enable_otel` switch. When enabled, a rollout worker discovers the OTLP/HTTP endpoint from RL-Insight `GET /services` and exports vLLM (and, later, SGLang) request traces to Tempo. This answers which replica handled a request and how long it took, but it does not show what the model reasoned about in each sentence or which sentence caused a bad result.

This RFC keeps the #8158 request-level trace compatible and adds **sentence-level inference traces**. Each generation is represented by a request span, with child spans for displayable sentences. RL-Insight defines the common attributes, sampling and redaction rules, and a Grafana dashboard that filters by project, experiment, step, replica, sample, session, trajectory, and turn. A user can drill down from a training step to one trajectory and expand every turn into reasoning, tool calls, and answer text.

## Motivation and goals

The current integration cannot reliably locate an anomalous sentence when reward, latency, length, or failure rate changes. Reasoning, tool calls, and the final answer are often captured as one text field, so they cannot be compared by turn or sentence.

This RFC aims to:

- preserve the #8158 `enable_otel`, OTLP/HTTP, and resource-attribute behavior;
- show sentence index, text, reasoning/answer type, timing, token counts, finish reason, and errors;
- support drill-down from training step → sample/session → trajectory → turn → sentence;
- use the same schema for streaming and non-streaming generation and for completions with or without `</think>`;
- keep trace collection safe by default through text limits, redaction, and sampling.

Token-level events, tokenizer implementation, reward calculation, and changes to the existing Prometheus or Agent Loop protocols are out of scope.

## Compatibility with verl #8158

The existing path is:

```text
rollout worker
  └─ RLInsightLogger.otlp_traces_endpoint()
       └─ GET /services → otlp_port
            └─ OTLP/HTTP /v1/traces → Tempo
```

The worker sets `OTEL_RESOURCE_ATTRIBUTES=project=...,experiment_name=...,replica=...` and `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`. This RFC does not change that path. Sentence spans use the same tracer, endpoint, and resource attributes and are children of the request span.

## Trace data protocol

### Span hierarchy

```text
inference_request                  # engine request; compatible with #8158
└── inference_turn                 # one prompt → completion (optional)
    ├── inference_sentence        # reasoning sentence
    ├── inference_sentence        # tool call or result (optional)
    └── inference_sentence        # answer sentence
```

If an engine cannot propagate a parent span, it must still emit request and sentence spans with the same correlation attributes.

### Required attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `monitor.trace_source` | string | `inference_request`, `inference_turn`, or `inference_sentence` |
| `project`, `experiment_name`, `replica` | string | Same values as the #8158 resource attributes |
| `global_steps` | int | Trainer step; omit when unavailable, never stringify it |
| `sample`, `session`, `traj`, `state_lane_id` | string/int | Correlation keys aligned with `agent_loop_session` |
| `turn` | int | Model turn, starting at 0 |
| `sentence_index` | int | Sentence index within a turn, starting at 0 |
| `sentence_type` | string | `reasoning`, `tool_call`, `tool_result`, or `answer` |
| `status` | string | `success`, `failure`, `empty`, or `error` |

### Optional attributes

`text`, `text_hash`, `prompt_tokens`, `completion_tokens`, `token_start`, `token_end`, `finish_reason`, `tool_name`, `tool_args`, `error`, `request_id`, `model`, and `temperature` are optional. Text attributes are capped by configuration. Reasoning and answer must be split before truncation.

The runtime owns sentence boundaries. A streaming runtime may commit a sentence when punctuation or a completion event arrives; a non-streaming runtime splits the completed output. For templates that prefill `<think>`, split at `</think>` before sentence segmentation. If the tag is missing, the entire completion is reasoning. An empty tool call must not be represented by a bare `</think>`.

Example:

```json
{
  "name": "inference_sentence",
  "parent": "inference_turn",
  "attributes": {
    "monitor.trace_source": "inference_sentence",
    "project": "verl",
    "experiment_name": "ppo_math",
    "replica": "3",
    "global_steps": 128,
    "sample": "42",
    "session": "0",
    "traj": 1,
    "state_lane_id": "experiment=ppo_math/sample=42/session=0/traj=1",
    "turn": 2,
    "sentence_index": 0,
    "sentence_type": "reasoning",
    "text": "First check the constraints.",
    "completion_tokens": 8,
    "status": "success"
  }
}
```

## Dashboard design

Add `rl_insight/config/services/grafana/dashboards/inference_trace/inference_trace.json` in a dedicated `inference_trace` folder. This avoids changing the existing verl dashboard UIDs and layouts.

Dashboard variables are `Project`, `Experiment`, `Global Step`, `Replica`, `Sample`, `Session`, `Traj`, and `Turn`. The default time range is the last 15 minutes; step and lane filters use exact matching.

Panels:

1. **Step Overview:** request count, sentence count, success rate, P50/P95 request latency, and average completion tokens.
2. **Latency by sentence type:** P50/P95 latency and token distributions for reasoning, tool calls, and answers.
3. **Trajectory table:** reward, turn count, failed sentence count, and total duration per trajectory; clicking a row sets `state_lane_id`.
4. **Turn timeline:** sentences ordered by `turn` and `sentence_index`, colored by `sentence_type`, with failed spans highlighted.
5. **Sentence detail:** text, duration, tokens, finish reason, tool arguments, and error, with a link to the Tempo trace.
6. **Raw trace:** Grafana Tempo panel showing the request and sentence parent/child relationship.

Recommended TraceQL shape:

```traceql
{ span.monitor.trace_source = "inference_sentence"
  && span.project = "${Project}"
  && span.experiment_name = "${Experiment}"
  && span.global_steps = ${GlobalStep}
  && span.state_lane_id =~ "${Lane}" }
| select(span.turn, span.sentence_index, span.sentence_type,
         span.text, span.status, span.completion_tokens)
```

## Implementation plan

### RL-Insight

1. Keep this protocol in `docs/monitor` and provide the `inference_trace` dashboard JSON.
2. Add validation, lane generation, and text truncation helpers for `inference_sentence` in `rl_insight.agent_loop`; reuse `trace_span` rather than adding a transport protocol.
3. Add `RL_INSIGHT_TRACE_TEXT`: `off` (default, statistics and hash only), `redacted` (redacted text), and `full` (text within the configured limit).
4. Add `RL_INSIGHT_TRACE_SAMPLE_RATE` and per-step/per-replica caps. Preserve failed sentences and random samples after the cap, and emit `trace.sampled_out=true` for dropped data.
5. Add unit tests for attributes and Tempo queries, plus a monitor smoke test that emits one request and two sentence spans and verifies the parent relationship, numeric step, and TraceQL queryability.

### verl

1. Keep `TraceConfig.enable_otel` defaulting to `false`.
2. Create request and turn spans in the rollout engine adapter and pass sample/session/trajectory/turn identity into the common attributes.
3. Split `</think>` and segment sentences in the completion aggregation layer. Streaming engines may send completed sentences early; non-streaming engines send them when the request finishes.
4. Use OTEL span context for parent/child relationships; if context cannot be propagated, copy the correlation attributes.
5. Document the switch, text policy, capacity estimate, and troubleshooting. Missing RL-Insight or an unavailable endpoint must never block rollout.

### Failure and degradation

- Export failures only produce a rate-limited warning and never fail generation.
- A sentence export is not retried as a business request; `BatchSpanProcessor` flushes during process shutdown.
- Missing sample/session/trajectory fields do not prevent request spans; the dashboard groups them as `unattributed`.
- When text exceeds the limit, retain `text_hash`, `text_truncated=true`, token counts, and timing.

## Performance and capacity

The approximate span count is:

```text
steps × samples_per_step × trajectories × turns × sentences_per_turn
```

The initial recommendation is at most 32 sentence spans per turn, 2 KiB per text attribute, and 256 sentence spans per worker per step. Test exporter CPU, network bandwidth, Tempo WAL, and dashboard latency at 1k, 10k, and 100k spans per minute. When the budget is exceeded, lower sampling or disable `text` while retaining structured attributes.

## Security and privacy

Prompts, reasoning, tool arguments, and answers may contain user data, secrets, or training-set content. The default is `off`; the dashboard must still show timing and token statistics without body text. `redacted` must at least filter common token/key/password/header fields. Grafana and Tempo access continues to use deployment authentication and network isolation. Raw text must never be placed in Prometheus labels because of cardinality and data-spread risks.

## Rollout and acceptance

### Phase 1: protocol and minimal dashboard

- Merge this RFC, schema helpers, and the dashboard skeleton.
- Use static `trace_span` fixtures to verify dashboard variables, trajectory drill-down, and Tempo detail.

### Phase 2: vLLM integration

- Build request/turn/sentence spans on top of #8158.
- Cover streaming, non-streaming, tool calls, missing `</think>`, and failed requests.

### Phase 3: productionization

- Add SGLang and other engines, then complete sampling, redaction, capacity testing, and operational documentation.

Acceptance criteria: with `enable_otel=true`, one training step appears in the dashboard within 30 seconds; a user can locate a turn and sentence from a trajectory; Tempo shows the sentence under its request parent; training completes when OTEL is disabled or Tempo is unavailable; and the default configuration does not upload body text.

## Open questions

1. Should the default move from `off` to `redacted` after data-governance review?
2. Should sentence splitting use each engine's streaming punctuation rules or a replaceable splitter interface supplied by verl?
3. When sample/session/trajectory fields are missing, should the rollout worker generate a request-scoped fallback ID?

