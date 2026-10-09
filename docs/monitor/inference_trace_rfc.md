# RFC: Sentence-Level Inference Tracing for RL Workloads

- Status: Draft
- Scope: RL-Insight and its verl integration
- Related implementation: https://github.com/verl-project/verl/pull/8158

## Why this is needed

RL training dashboards show aggregate reward, throughput, token counts, and engine latency. Those aggregates are insufficient when a trajectory receives a bad reward or becomes unusually slow: an engineer cannot identify the turn, reasoning step, tool call, or answer sentence that caused the outcome.

RL-Insight already stores OpenTelemetry traces and has an Agent Loop protocol, but the current integration does not provide a consistent sentence-level view for rollout inference. The missing capability is a shared identity and trace contract, a way to carry sentence text and timing safely, and a dashboard that connects a training step to the exact sentence.

This is an RL-Insight observability RFC. verl #8158 is one implementation work item in the rollout-engine integration, not the motivation for the RFC.

## Goals

- Trace every displayable inference sentence under its request and turn.
- Correlate a sentence with project, experiment, trainer step, replica, sample, session, trajectory, and turn.
- Distinguish reasoning, tool call, tool result, and answer sentences.
- Let users move from step to trajectory to turn to sentence in Grafana and Tempo.
- Preserve training behavior when tracing is disabled, sampled, or unavailable.
- Make text collection explicitly configurable and safe by default.

## Non-goals

- Token-level tracing or replacing engine profilers.
- Reimplementing tokenization, reward computation, or agent semantics in RL-Insight.
- Putting trace text into Prometheus labels.
- Requiring all rollout engines to expose the same streaming API.

## Proposed design

### Trace hierarchy

Each generation has one request span and may have one turn span. The runtime emits one child span for each sentence or tool event:

    inference_request
    +-- inference_turn
        +-- inference_sentence (reasoning)
        +-- inference_sentence (tool_call/tool_result)
        +-- inference_sentence (answer)

The spans use the existing OTLP/HTTP path and trace_span/OpenTelemetry plumbing. If parent context cannot cross an engine boundary, the same correlation attributes are copied to each span.

### Identity and attributes

Required attributes on sentence spans:

| Attribute | Meaning |
| --- | --- |
| monitor.trace_source | inference_sentence |
| project, experiment_name, replica | Run and rollout identity |
| global_steps | Numeric trainer step |
| sample, session, traj, state_lane_id | Agent-loop correlation |
| turn, sentence_index | Position within the trajectory |
| sentence_type | reasoning, tool_call, tool_result, or answer |
| status | success, failure, empty, or error |

Optional attributes include text, text_hash, token counts and offsets, finish_reason, tool_name, tool_args, request_id, and error. global_steps remains numeric so exact dashboard filters work.

For models using <think>, split at </think> before truncating or sentence segmentation. Without a closing tag, the completion is reasoning. The publisher must not emit a bare closing tag for an empty tool call. Sentence boundaries are supplied by the runtime: streaming runtimes can emit completed sentences, while non-streaming runtimes segment the completed output.

### Text policy

Add a process-level policy with three modes:

- off (default): keep timing, identity, status, and token metadata; optionally keep a hash.
- redacted: collect text after removing configured secret and credential patterns.
- full: collect text subject to a byte limit.

A per-worker and per-step span budget controls volume. Failed spans and a representative sample remain when the budget is exceeded. Truncated text carries text_truncated=true and text_hash. Export failure is a warning and never fails rollout.

### Grafana dashboard

Add an inference_trace dashboard folder with variables for project, experiment, global step, replica, sample, session, trajectory, and turn. The dashboard contains:

1. Step overview: request/sentence count, success rate, P50/P95 latency, and completion tokens.
2. Sentence-type latency: reasoning, tool, and answer latency and token distributions.
3. Trajectory table: reward, turn count, failed sentence count, and total duration.
4. Turn timeline: sentence order and duration, colored by sentence type.
5. Sentence details: text (when allowed), token counts, finish reason, tool arguments, error, and a Tempo link.
6. Raw trace: the request/turn/sentence parent-child view in Tempo.

The queries use exact numeric step matching and state_lane_id for drill-down. Text is never a Prometheus label.

## Work breakdown

### 1. Protocol and API

- Define span names, required attributes, sentence splitting rules, text policy, and limits in monitor documentation.
- Add small helpers or validation around the existing trace_span path.
- Define how missing identity fields and engine context are represented.

### 2. verl integration

- Use the rollout trace switch and OTLP endpoint discovery from verl #8158.
- Create request/turn spans and sentence child spans in the rollout aggregation path.
- Pass sample/session/trajectory/turn identity from the agent runtime.
- Cover vLLM first, then SGLang and other engines.
- Ensure tracing remains opt-in and never blocks generation.

### 3. RL-Insight backend and dashboard

- Ship the Grafana dashboard JSON and provisioning entry.
- Add Tempo query fixtures and dashboard variables.
- Add text redaction, truncation, sampling, and rate-limit configuration.
- Document storage impact and operational controls.

### 4. Testing and validation

- Unit-test think-tag splitting, sentence indexing, missing identity, truncation, redaction, sampling, and numeric step typing.
- Add an OTLP/Tempo integration test with one request, multiple sentence spans, and a verified parent relationship.
- Add dashboard smoke coverage for step, lane, trajectory, and sentence queries.
- Test streaming and non-streaming output, tool calls, empty output, missing closing tags, engine errors, OTLP failure, and process shutdown flush.
- Run a volume test at 1k, 10k, and 100k spans per minute and record exporter, network, Tempo WAL, and query latency.

## Rollout and acceptance

1. Merge the protocol and a fixture-backed dashboard.
2. Integrate vLLM through the #8158 rollout trace path.
3. Add production sampling/redaction and then other engines.

The feature is accepted when a trace-enabled step appears in the dashboard within 30 seconds, a user can drill from trajectory to sentence, Tempo shows the correct parent chain, sentence text follows the selected policy, and training completes when tracing is disabled or the endpoint is unavailable.

## Open questions

1. Should redacted become the default after a data-governance review?
2. Should sentence segmentation be an engine-provided callback or a replaceable verl interface?
3. Should missing sample/session/trajectory identity receive a request-scoped fallback ID?

