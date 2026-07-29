# Association Analysis

Association analysis is optional post-processing for the existing KDE
detector. It ranks lower-level anomalous metrics for each confirmed anomaly
event of an explicitly configured target metric. It does not change KDE
thresholds, point classification, interval validation, history confirmation,
or the existing `states`, `results`, and `abnormalTimeRange` fields.

## Data Flow

```text
strict standard/inference input
-> existing preprocessing and per-metric KDE detection
-> confirmed target anomaly events
-> anomalous lower-metric filtering
-> bounded time alignment
-> Pearson/Spearman evidence
-> random-forest evidence
-> independently normalized weighted score
-> per-event Top K ranking
```

The numeric values used for correlation come from each metric's cleaned and
time-windowed `inference` `TimeSeries`. The anomaly labels come from the same
KDE run's `results.<metric>.pointDiagnostics[].abnormal` field, matched back to
the cleaned timestamps. Association analysis never reimplements the KDE
threshold decision.

Only metrics included in the detector's `metrics` list are available. A lower
metric is eligible when its KDE state is `0`, it has inference data and point
labels, it is not a constant inference series, and it has at least one
anomalous point in the target event's analysis window. Constant candidates are
reported as `constant_candidate_series` before the anomaly-window filter.
Other configured target metrics are excluded from the lower-metric set.

## Events and Analysis Windows

Each history-confirmed target interval is analyzed independently. The public
`startTime` and `endTime` may include the detector's display padding, so they
are not used as raw event boundaries. The analyzer instead uses the
unformatted candidate boundaries retained alongside the current KDE result:

```text
duration = raw_end - raw_start
analysis_start = raw_start - duration * context_ratio
analysis_end = raw_end + duration * context_ratio
```

The window is clipped to the target inference series. When the raw duration is
zero, one median positive target sample interval is used as minimum context;
`analysisWindow.usedMinimumContext` records this case.

Raw event records are process-local and belong to the current inference batch.
History may confirm and publish an older interval without its current raw
context. Such an interval is not reconstructed from padded display boundaries;
the target returns `insufficient_data`.

## Time Alignment

For `training_log` data where all target and candidate timestamps are below
`10000`, timestamps are treated as steps and must match exactly.

Other data uses nearest-timestamp matching with a finite tolerance:

- a positive `alignment_tolerance` explicitly sets the maximum distance;
- when it is `null`, the tolerance is half the larger median positive sampling
  interval of the target and candidate;
- a nearest point outside the tolerance is not matched;
- one candidate observation can match at most one target row, preventing sparse
  samples from inflating coverage or random-forest sample counts;
- matching is limited to candidate points inside the analysis window.

`coverageRatio` is the number of matched target rows divided by all valid
target rows in the analysis window. Candidates below `min_coverage_ratio` or
`min_aligned_points` are excluded. Preprocessing has already applied the
module's stable sorting, invalid-pair filtering, and keep-last duplicate
timestamp rule.

## Correlation Evidence

Pearson and Spearman coefficients are computed from aligned raw values. The
coefficient with the greater absolute value is selected:

```text
correlationStrength = max(abs(pearson), abs(spearman))
```

`selectedCorrelation` retains its sign and
`selectedCorrelationMethod` identifies `pearson` or `spearman`. An absolute
tie deterministically selects Pearson. Constant or otherwise undefined series
produce no correlation evidence rather than `NaN`.

## Random-Forest Evidence

The classifier uses aligned KDE labels:

```text
features: lower-metric abnormal labels
target:   target-metric abnormal label
```

All active features first share a common set of target rows. Candidates may be
removed when needed to reach `min_rf_samples`; constant feature columns are
not modeled. The target labels must contain both normal and abnormal classes.

Rows remain in time order and are split 70%/30%:

- when both training and validation partitions contain both classes, a
  `RandomForestClassifier` is fitted on the first 70%, and permutation
  importance is computed on the last 30% with balanced accuracy;
- when the chronological split lacks both classes, the model is fitted on all
  rows and uses impurity importance as an explicit
  `impurity_fallback`; the event is `partial_success`.

Permutation importance uses 10 repeats and the configured `random_state`.
The classifier also uses the configured `n_estimators`,
`class_weight="balanced"`, and one worker for reproducibility. If
scikit-learn is unavailable or the model lacks usable samples, correlation
evidence can still produce a partial result.

## Scoring

Correlation strengths and non-negative random-forest importances are
normalized independently across all valid candidates. When both sources are
valid:

```text
score =
  weights.correlation * normalized_correlation
  + weights.random_forest * normalized_random_forest_importance

abnormalContribution = score * 100
```

If only one source is valid, it receives the full effective weight and the
event is `partial_success`. If neither is valid, the event is
`insufficient_data`. Rankings sort by descending contribution and then metric
name. `allAssociations` contains the complete ranking; `topAssociations`
contains at most `top_k` entries. Contributions across `allAssociations` sum
to approximately 100%, but the truncated Top K need not.

## Configuration

Association settings live in the existing per-metric YAML. Old metric files
without this section are deep-merged with the disabled defaults.

```yaml
association:
  enabled: false
  target_metrics:
    - timing_s/step
  candidate_mode: abnormal_lower_metrics
  weights:
    correlation: 0.5
    random_forest: 0.5
  top_k: 5
  context_ratio: 1.0
  min_aligned_points: 10
  min_rf_samples: 30
  min_coverage_ratio: 0.6
  alignment_tolerance: null
  random_forest:
    n_estimators: 200
    class_weight: balanced
    random_state: 42
    importance_method: permutation
```

`alignment_tolerance` is an optional advanced override for non-step alignment;
it must be positive when set. The only current `candidate_mode` is
`abnormal_lower_metrics`. Weights must be non-negative and sum to `1`.

`--association-target` overrides the YAML target list and enables analysis for
that invocation. Without a CLI target or an enabled YAML section, the detector
does not add `associationAnalysis`, preserving the original response shape.

## Input and Output

No additional input root is introduced. Target and lower metrics use the same
strict `standard` and `inference` sections:

```json
{
  "standard": {
    "timing_s/step": {"timestamps": [1, 2, 3], "values": [1.0, 1.01, 0.99]},
    "gpu_utilization": {"timestamps": [1, 2, 3], "values": [40, 41, 39]}
  },
  "inference": {
    "timing_s/step": {"timestamps": [10, 11, 12], "values": [1.0, 1.5, 1.6]},
    "gpu_utilization": {"timestamps": [10, 11, 12], "values": [42, 88, 92]}
  }
}
```

An enabled response adds one top-level section. Each confirmed target event
has its own ranking:

```json
{
  "associationAnalysis": {
    "enabled": true,
    "status": "success",
    "weights": {"correlation": 0.5, "randomForest": 0.5},
    "targets": {
      "timing_s/step": {
        "status": "success",
        "events": [{
          "status": "success",
          "rawTargetAbnormalRange": {"startTime": 104.0, "endTime": 109.0},
          "analysisWindow": {
            "startTime": 99.0,
            "endTime": 114.0,
            "usedMinimumContext": false
          },
          "topAssociations": [{
            "rank": 1,
            "metric": "gpu_utilization",
            "abnormalContribution": 64.2,
            "pearson": 0.94,
            "spearman": 0.91,
            "selectedCorrelation": 0.94,
            "selectedCorrelationMethod": "pearson",
            "randomForestImportance": 0.62,
            "randomForestImportanceMethod": "permutation",
            "coverageRatio": 1.0,
            "alignedSampleCount": 30
          }]
        }]
      }
    }
  }
}
```

Events also expose `targetAbnormalRange`, `allAssociations`, candidate
diagnostics, exclusion reasons, and random-forest diagnostics.

## Status and Degradation Rules

Common target or event statuses are:

- `success`: both correlation and full permutation-importance evidence exist;
- `partial_success`: one evidence source is usable, or random forest used the
  impurity fallback;
- `target_metric_missing`: the requested target was not supplied and selected;
- `target_detection_failed`: the target KDE state is not `0`;
- `target_not_abnormal`: no confirmed target interval exists;
- `no_candidate_metrics`: every lower metric was filtered out;
- `insufficient_data`: raw event context or both evidence sources are
  insufficient;
- `analysis_error`: unexpected association post-processing failure.

Random-forest diagnostics may additionally report `dependency_unavailable` or
an `insufficient_data` reason. These are association outcomes, not KDE states.
Association failure never rewrites completed KDE results.

## Limitations

- `abnormalContribution` is an association-candidate ranking, not a causal
  probability or a confirmed root cause.
- Historical intervals without current raw event context cannot be safely
  reconstructed and return `insufficient_data`.
- Correlated or collinear lower metrics may share or redistribute
  random-forest importance, so individual importance should not be interpreted
  in isolation.
- The current implementation evaluates contemporaneous bounded alignment; it
  does not estimate lagged causal effects.
