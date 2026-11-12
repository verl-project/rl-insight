# Inference Engine Tracing

RL-Insight stores inference-engine request traces from verl rollouts in Tempo and shows them in the **vLLM request traces** row of the `verl_trainer_v1_with_vllm_engine` dashboard. The rollout engine produces the spans; RL-Insight stores them as-is and does not create a second timing implementation.

Span production, enablement options, and the correlation attribute schema are documented in the verl guide: [RL-Insight integration](https://github.com/verl-project/verl/blob/main/docs/advance/rl_insight.md).

## Enable tracing

Prerequisites:

- The RL-Insight server stack is running with Tempo enabled (`rl-insight server start`).
- The RL-Insight SDK is 0.4.0 or newer on the driver and every rollout worker.
- `RL_INSIGHT_SERVER_URL` points at the RL-Insight server in every rollout worker.
- `trainer.logger` contains `rl_insight`, so the job's Run exists before the rollout engines start.
- The verl build includes the `actor_rollout_ref.rollout.trace.enable_otel` switch. The switch is implemented for the vLLM rollout engine only.

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080

python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.trace.enable_otel=True \
    ...
```

Each rollout replica discovers the Tempo OTLP/HTTP endpoint from the RL-Insight server and stamps `project`, `experiment_name`, and `replica` on its spans. The first two come from the job's Run through `rl_insight.get_context()`, so they match training telemetry; see the [Python API](api.md). Tracing is opt-in: if the server is disabled, discovery fails, or export fails, rollout continues without tracing.

## Dashboard

Open Grafana → `verl_trainer_v1_with_vllm_engine` → **vLLM request traces**. The row is collapsed by default; expand it to load the **Inference requests** table. It lists matching requests newest first and links each row to its raw trace.

Filters:

- **training: Project** and **training: Experiment Name** match the span resource attributes. Experiment values include the Run's execution suffix; spans written before engine tracing adopted the Run name keep the un-suffixed value.
- **Step** is single-select, defaults to the newest step when `global_steps` is present, and puts **All** at the end of the list. All uses `-1` because TraceQL cannot regex-match the integer `global_steps`.
- **Replica** and **Sample** are multi-select, default to All, and use anchored regex so selecting `100` does not also match `1006`. All includes spans that lack the attribute.
- The dashboard time range limits the search.

The query caps Tempo's search at 20000 results, and the table paginates them. Tempo applies the limit before Grafana sorts, so narrow Step, Replica, Sample, or the time range for very large runs; edit the panel query `limit` to change the cap.

## Troubleshooting

If the row is empty, verify in this order:

1. `actor_rollout_ref.rollout.trace.enable_otel=True` and `rl_insight` in `trainer.logger`.
2. `curl -H "X-RL-Insight-Client-Version: <sdk-version>" $RL_INSIGHT_SERVER_URL/api/v1/services` returns a non-null `otlp_port`.
3. The rollout node can reach `<server-host>:<otlp_port>` and Tempo is enabled.
4. The Project, Experiment Name, Replica, Step, and Sample filters match the run.
5. `{ name = "llm_request" }` returns spans in Tempo. If requests appear but Step, Sample, or Turn is empty, the verl tracing integration is not attaching the correlation attributes.
