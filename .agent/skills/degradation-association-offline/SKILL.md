---
name: degradation-association-offline
description: "Analyze a user-selected time range from the local RL-Insight Prometheus TSDB, then save grouped Top-25 degradation associations and concise root-cause hypotheses under analysis/. Use for one-shot offline analysis, not live monitoring."
---

# Offline degradation association

Use the deterministic code in `recipe/perf_analysis/degradation`. Do not reimplement or
change its KDE baseline, eight default targets, 3-of-5 event lifecycle,
association ranking, or metric categories.

Read [algorithm-contract.md](references/algorithm-contract.md) before running the
analysis. Read [diagnostic-experience.md](references/diagnostic-experience.md)
only when at least one final event contains association entries.
Use the exact Chinese metric meanings in
[metric-name-catalog.md](../../../docs/monitor/metric-name-catalog.md) when
rendering association tables.

## 1. Environment check

Locate the checkout containing `recipe/perf_analysis/degradation/cli.py` and run commands
from its repository root. If imports are missing, report them and ask before
running `pip install -e ".[recipe]"`.

Use RL-Insight's default Prometheus TSDB at
`~/.rl-insight/data/prometheus`. Use `analysis` as the report root unless the
user supplies another path. Without `--baseline-file`, a run uses:

```text
analysis/<start>_<end>/standard_data.json
analysis/<start>_<end>/abnormal_data.json
analysis/<start>_<end>/analysis.json
analysis/<start>_<end>/report.md
```

With `--baseline-file`, the baseline remains at the supplied path; the other
three outputs remain in the report directory.

## 2. Offline data input

Require the user to supply `--start-time` and `--end-time` as Unix seconds or
ISO-8601 timestamps with timezone. Read that range directly from RL-Insight's
local Prometheus TSDB with `promtool tsdb dump`; do not query a Prometheus HTTP
port and do not ask the model to parse TSDB block files itself. Use
`rl_insight_monitor_training_global_step` as the step ruler.
Skip the global-step value already visible immediately after `--start-time` and
begin with the next observed step transition, so the first analyzed step did not
begin before the selected range.

Keep all discovered configured metrics. Never restrict input to trainer metrics
or discard non-trainer candidates. Exactly the eight `timing_s_*` metrics in the
algorithm contract are event targets; the other configured scalar metrics are
candidate evidence.

## 3. One-shot anomaly and association analysis

Run the selected range once:

```bash
python -m recipe.perf_analysis.degradation.cli analyze \
  --start-time <ISO-8601-or-Unix-seconds> \
  --end-time <ISO-8601-or-Unix-seconds>
```

By default the command trains a baseline from the first 30 complete steps in the
selected range and saves it in that run's report directory. Use
`--baseline-file` to load an existing baseline instead.

Wait for the command to finish. Do not stop after baseline training and do not
split the range into repeated calls. The one command performs point detection,
confirmed/closed event tracking, and one whole-event Top-25 association analysis
when each event closes. If a confirmed event is still open at the selected range
end, the same command analyzes its accumulated context through the final complete
step and marks it `open_at_range_end`. Offline analysis has no `latest` phase and
does not run association at confirmation.

Continue only when the command exits successfully and returns `status=ok`. On a
nonzero exit, report stderr and do not produce a no-anomaly conclusion.

For every final event with association entries, present the complete returned
Top-25, or all entries when fewer are available. Never render or diagnose its
confirmed phase. A final event phase is either `closed` or `open_at_range_end`.
Group by English metric category in best-score order, then sort metrics within
each group by descending score. Keep every category in one contiguous block.
Write the category name only in the first row of that block and leave the
category cell blank in its
remaining rows, even when the category contributes many Top-25 metrics. Do not
repeat the category name and do not insert separator rows or horizontal rules
between metrics or category blocks. The following table is mandatory: do not
replace it with JSON, bullets, prose, or a summary; do not add, remove, rename,
or reorder columns; and do not omit returned Top-K rows. Copy each Chinese
meaning verbatim from `docs/monitor/metric-name-catalog.md`; do not
translate, shorten, or infer it. Copy the stored `association_percent` number directly and
append `%`; never multiply, divide, normalize, or recalculate it. Reproduce this
structure exactly:

| Metric category | Metric name | Metric meaning | Association score |
|---|---|---|---:|
| transfer_queue | tq_partition_consumption_progress | TransferQueue 各分区各任务消费进度（0–1）。 | 94.80% |
|  | tq_storage_utilization_ratio | TransferQueue 存储当前有效键数相对容量上限的比例。 | 91.25% |
|  | tq_storage_request_latency_p99 | TransferQueue 存储请求时延 P99（s）。 | 89.10% |
| latency | rl_insight_monitor_perf_throughput | 按设备数归一化的训练吞吐量（token/s/device）。 | 88.60% |
|  | rl_insight_monitor_perf_time_per_step | 当前训练 step 耗时（s）。 | 84.30% |

Precede the table with:

```text
Abnormal target metric: <target>
Target labels: <labels>
Event steps: <start_step> -> <closed_at_step|range end>
Event phase: <closed|open_at_range_end>
Saved result: <absolute abnormal_data.json path>
```

Association score is relative evidence, not fault probability or causality. If
`processed_step_count` is zero, state that the batch trained or loaded a baseline
but did not run detection. Otherwise, if there are no final events, state that
the analyzed steps produced no confirmed abnormal target event. If a final event
has no association entries, report its stored status and do not invent a
diagnosis.

Save the complete Markdown result to the returned `report_path` under
`analysis/<start>_<end>/report.md`, including the no-final-event result when no
table or root-cause analysis is produced.

## 4. Root-cause analysis

For each final event with evidence, use the target, event phase, metric names,
categories, global ranks, final `association_percent` values, and the experience
reference. Use only the returned final association scores for evidence strength.
Do not reopen raw series, write another analysis script, compare step values, or
independently infer increases, decreases, trends, or change magnitude.

Do not use candidate direction, point state (`NORMAL`, `UP`, `DOWN`, or
`BETWEEN_MODES`), matched mode, or candidate abnormality as a diagnosis gate.
Combine association ranking with metric semantics and technical knowledge. When
length or sequence metrics occupy a substantial part of the returned Top-K,
`Sequence-length anomaly` MUST appear among the likely causes. When they also
form the leading high-scoring evidence group, it MUST be cause 1. Apply this rule
regardless of candidate direction or point state; do not infer whether sequence
length increased or decreased. Many weak scores alone do not justify high
confidence. This length-family rule is the explicit exception to the general
rule that category size alone is not a vote. The experience reference remains a
fallible prior.

Use the fault domains `Compute`, `Network`, `Host CPU`, and `HBM`. Rank exactly
two distinct domains and three to five plausible causes in confidence order.
Low confidence is acceptable. Keep the reasoning concise and do not claim
unobserved hardware, network, profiler, or operating-system signals.
Select every cause from the exact diagnostic vocabulary in
`diagnostic-experience.md`. Causes must be distinct. Never use a metric category,
component name, metric name, or symptom restatement such as `TransferQueue
delay` as a cause.

When multiple labeled series of `tq_storage_request_latency_p50` and/or
`tq_storage_request_latency_p99` occupy a substantial part of the returned
Top-K, make `Compute` the primary fault domain and include both `AI Core
overload` and `NPU frequency throttling` among the likely causes. When that
family is also the leading high-scoring evidence, rank them as the highest
remaining causes after applying the sequence-length priority rule above.
`Sequence-length anomaly` is a workload/data condition rather than one of the
four fault domains. When the sequence-length rule applies, the two fault domains
are only infrastructure alternatives and must not displace that diagnosis; keep
them low-confidence unless stronger non-length association evidence exists.

```text
Primary fault domain: <domain> (confidence: High|Medium|Low)
Secondary fault domain: <domain> (confidence: High|Medium|Low) — <brief basis>

Likely fault causes:
1. <cause> (primary; confidence: High|Medium|Low) — <brief basis>
2. <cause> (confidence: High|Medium|Low) — <brief basis>
3. <cause> (confidence: High|Medium|Low) — <brief basis>
[4-5 when supported]

Reasoning basis: <two or three concise professional sentences>
```

This Skill analyzes supplied observations only. Do not provide fault-injection
commands or procedures.
