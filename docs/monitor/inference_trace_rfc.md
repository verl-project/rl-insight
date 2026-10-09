# RFC: Inference-Engine Tracing for verl Rollouts

- Status: Draft
- Scope: RL-Insight dashboards and the verl integration
- Related implementation: verl PR #8158

## Why

RL-Insight currently shows rollout throughput and aggregate training metrics, but an engineer cannot inspect the latency of one inference in a trajectory. For reasoning workloads, the useful unit is the model turn/sentence: which request waited in the scheduler, how long prefill took, when the first token arrived, how expensive decode was, and whether the request failed.

The rollout engine already measures most of this. We should export and visualize the engine trace instead of adding a second tracing implementation in RL-Insight.

## What the engine can trace

With vLLM OpenTelemetry tracing, a request can expose the following timings and request fields:

| Timing | Meaning |
| --- | --- |
| request duration | End-to-end request time |
| time in scheduler | Time waiting/being scheduled |
| time to first token (TTFT) | Request start to first generated token |
| time in model forward | Model forward-pass time while the request is batched |
| time in model execute | Forward, worker synchronization, CPU/GPU synchronization, and sampling |
| decode time / time per output token | Generation/decode phase and its per-token cost |
| queue or waiting time | Time before execution begins, when provided by the engine version |

The trace also carries request id, model, prompt/completion token counts, finish reason, and replica resource attributes. Detailed model/worker timings are enabled by the engine's detailed-trace option and have measurable overhead. The exact fields vary by vLLM/SGLang version; RL-Insight must preserve unknown attributes rather than rename or reinterpret them.

These are request/turn timings. The engine does not natively create one OpenTelemetry span for every natural-language sentence. Sentence-level display therefore uses the rollout runtime's turn/sentence metadata to group an engine request, while the timing remains the engine's request or token timing. If streaming token timestamps are available, RL-Insight can derive sentence start/end at punctuation boundaries; otherwise the dashboard shows turn-level engine timing and sentence text without inventing sentence latency.

## Proposal

1. **Enable the engine trace.** Keep the existing opt-in switch in verl, discover RL-Insight's OTLP/HTTP endpoint from GET /services, and pass the endpoint and resource attributes to each rollout replica. This is the integration work in PR #8158.
2. **Preserve engine data.** RL-Insight accepts the engine span and stores its native timing attributes in Tempo. Common correlation attributes are added by the rollout adapter: project, experiment, global step, replica, sample, session, trajectory, turn, and request id.
3. **Attach turn/sentence context.** The rollout adapter attaches the generated turn and sentence index/type as attributes or span events. It does not create a synthetic timing span. Text collection is configurable and capped; the default can keep metadata and token counts without storing text.
4. **Add a dashboard.** Grafana filters by project, experiment, step, replica, sample, session, trajectory, and turn. It shows request duration, scheduler wait, TTFT, prefill/model-forward, model-execute, decode, tokens, finish reason, and error. A trajectory table links each turn/sentence row to its Tempo trace.
5. **Version compatibility.** Use an allowlist only for dashboard calculations; display other native attributes in the raw trace. Add adapters when vLLM/SGLang names or units differ.

## Work items

### verl integration

- Land the OTLP endpoint discovery and enable_otel configuration from PR #8158.
- Pass RL identity and turn/sentence context into the engine request.
- Enable detailed engine timing only through an explicit configuration because of overhead.
- Verify vLLM first, then add the equivalent SGLang mapping.

### RL-Insight

- Define the minimal correlation schema and unit conventions.
- Keep native engine attributes in Tempo and avoid copying them into Prometheus labels.
- Add the inference-trace Grafana dashboard and Tempo links.
- Add text policy, length limits, sampling, and redaction controls.
- Document which fields are available for each engine/version.

### Testing

- Unit-test correlation attributes, numeric global step, missing identity, text limits, and unit conversion.
- Send a fixture request span containing scheduler, TTFT, forward, execute, and decode timings; verify Tempo search and raw attribute preservation.
- Add an end-to-end test with a real or mocked vLLM OTLP exporter and one rollout trajectory; verify replica/step/turn filters and the Tempo link.
- Test detailed tracing on/off, streaming and non-streaming generation, tool calls, failed requests, missing timing fields, exporter failure, and shutdown flush.
- Run an overhead test comparing throughput and latency with detailed tracing disabled and enabled.

## Acceptance criteria

- With enable_otel enabled, an inference request appears in Tempo with its native timing fields and RL correlation attributes.
- The dashboard can locate a turn from step, replica, and trajectory filters and links to the raw trace.
- Scheduler wait, TTFT, prefill/forward, execute, decode, end-to-end latency, token counts, and finish reason are visible when the engine supplies them.
- Missing fields are shown as unavailable; no value is fabricated as a sentence latency.
- Tracing is opt-in and a missing or unavailable OTLP endpoint never blocks rollout.

