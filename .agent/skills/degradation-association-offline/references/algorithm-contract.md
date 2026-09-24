# Offline Degradation Algorithm Contract

## Workflow

```text
RL-Insight local Prometheus TSDB + user time range
  -> promtool tsdb dump (no HTTP query)
  -> align samples to global-step intervals
  -> load a frozen baseline or train it from the first 30 complete steps
  -> classify later steps and track target events
  -> calculate one whole-event correlation/random-forest association at closure
     or at the selected range end for a confirmed event that remains open
  -> save final evidence and report under analysis/<start>_<end>/
```

## Input and alignment

- Require exactly one concrete
  `rl_insight_monitor_training_global_step` series.
- Treat the first global-step value visible after `start-time` as potentially
  partial and begin at the next observed step transition.
- Define step `k` as `[timestamp(k), timestamp(k + 1))` and use the last finite
  sample in that interval.
- Preserve missing metric values and distinct label sets.
- Require consecutive global-step boundaries; never infer a missing step.

## Metric roles

The eight scalar `UP` event targets are:

1. `rl_insight_monitor_timing_s_step`
2. `rl_insight_monitor_timing_s_gen`
3. `rl_insight_monitor_timing_s_ref`
4. `rl_insight_monitor_timing_s_adv`
5. `rl_insight_monitor_timing_s_old_log_prob`
6. `rl_insight_monitor_timing_s_update_actor`
7. `rl_insight_monitor_timing_s_update_weights`
8. `rl_insight_monitor_timing_s_testing`

The 102 configured scalar candidates use a `BOTH` policy and are grouped as
`latency`, `training_quality`, `rollout_quality`, `data_characteristics`,
`hardware_resources`, `vllm_engine`, and `transfer_queue`. Candidate anomalies
support association but never trigger target events.

## Baseline and detection

- Train a Gaussian KDE baseline from the first 30 complete steps when no
  baseline file exists; require at least 20 finite samples per fitted series.
- Load an existing schema-compatible `standard_data.json` without retraining.
- Require at least one fitted target and one fitted candidate.
- Confirm a target when at least three of five consecutive valid target points
  are above its fitted normal range.
- Close the active event when a later complete five-point window has fewer than
  three abnormal points.
- Do not reuse the closed event's five-point evidence window to confirm the next
  event.
- Missing target points delay lifecycle transitions.

## Association

- Analyze each event once, after it closes or at the selected range end when a
  confirmed event remains open, over its complete observed context.
- Retain 30 steps of pre-event context.
- Require at least 10 aligned target/candidate points for correlation.
- Random forest requires at least 30 common samples and both normal/abnormal
  classes in its chronological 70/30 train/validation split.
- Combine normalized evidence with correlation weight `0.85` and random-forest
  weight `0.15`; use the only valid evidence path at full effective weight.
- Return at most 25 candidates for each final event.
- Treat association score only as relative evidence within that event window.

## Output

`standard_data.json` stores fitted baseline ranges, not the raw first 30 steps.
`abnormal_data.json` stores final event views and one flat deterministic Top-25
record per event. A final phase is `closed` or `open_at_range_end`; offline
analysis has no `latest` or confirmed association phase. CLI stdout presents
these final events grouped by metric category for model interpretation.
`analysis.json` stores the complete CLI result, and the model saves its final
Markdown table and diagnosis as `report.md`. Each table row includes the exact
Chinese metric meaning from
[metric-name-catalog.md](../../../../docs/monitor/metric-name-catalog.md)
in the docs directory.
