# Inference Performance Degradation Perception

This experimental module detects sustained degradation in inference-performance
time series such as `timing_s/step`. It is deliberately isolated under
`experiment/degradation_perception/`: it does not modify the existing
RL-Insight monitor, Recipe, Prometheus, or Grafana pipelines.

The implementation follows the approved statistical workflow:

```text
historical standard data
  -> full-history adaptive KDE
  -> density peaks and unfiltered density valleys
  -> valley-bounded peak influence regions
  -> time-contiguous candidate segments
  -> strict three-part stability voting
  -> one KDE threshold model per stable mode
  -> inference point classification
  -> candidate abnormal intervals
  -> four-condition interval validation
  -> history confirmation
  -> JSON Response
```

It is not a neural-network training component and is not yet connected to the
production Prometheus query path. A strict adapter accepts already-fetched
single-series `query_range` responses, but this module does not issue PromQL or
HTTP requests itself.

## Install

From the repository root, install the project, degradation dependencies, and
test dependencies:

```powershell
python -m pip install -e ".[degradation,test]"
```

The local algorithm needs NumPy and SciPy. Paramiko is only used by the remote
adapter.

## One-command example

The repository includes deterministic demonstration data. Run:

```powershell
python -m experiment.degradation_perception.main --path experiment/degradation_perception/sample_data.json --metrics timing_s/step
```

The sample contains a stable historical mode near `1.0`, followed by six
continuous inference values near `1.5`. With the bundled `UP` configuration,
the result contains a formal abnormal interval.

To restrict only the inference window:

```powershell
python -m experiment.degradation_perception.main --path experiment/degradation_perception/sample_data.json --start-time 100 --end-time 109 --metrics timing_s/step --task-id demo
```

Multiple metrics are ordinary sequential arguments:

```powershell
python -m experiment.degradation_perception.main --path M:\data\input.json --metrics timing_s/step rollout_latency --config-dir M:\data\metric-configs
```

The CLI writes exactly one compact JSON object to stdout. Runtime errors also
produce a JSON error object and a non-zero exit status. The serializer rejects
non-standard `NaN` and `Infinity` output.

## Canonical local input contract

`--path` must name one existing UTF-8 `.json` file. Directories, globs, CSV,
plain training logs, and implicit file discovery are not accepted. The root
object must contain exactly `standard` and `inference`; each section is keyed
by the original metric name:

```json
{
  "standard": {
    "timing_s/step": {
      "timestamps": [1, 2, 3],
      "values": [1.00, 1.01, 0.99]
    }
  },
  "inference": {
    "timing_s/step": {
      "timestamps": [10, 11, 12],
      "values": [1.00, 1.30, 1.40]
    }
  }
}
```

Important rules:

- `standard` and `inference` must both be present, even if one is empty.
- A metric entry contains either exactly `timestamps` and `values` arrays or
  one complete Prometheus `query_range` response.
- Missing metrics are not a file-format error; they become state `1` or `2`
  according to the per-metric state machine.
- Timestamp/value lengths are checked before paired iteration. The loader never
  uses `zip()` to silently truncate mismatched arrays.
- `--start-time` and `--end-time` form an inclusive window and apply only to
  inference data. Standard data is never windowed.
- Metric names remain JSON object keys. A slash in `timing_s/step` is data, not
  a path separator.

## Prometheus range-query input

For already-fetched Prometheus data, wrap the standard-window and
inference-window responses separately:

```json
{
  "standard": {
    "rl_insight_monitor_timing_s_step": {
      "status": "success",
      "data": {
        "resultType": "matrix",
        "result": [{
          "metric": {"worker": "trainer_0"},
          "values": [[1710000000, "1.00"], [1710000015, "1.01"]]
        }]
      }
    }
  },
  "inference": {
    "rl_insight_monitor_timing_s_step": {
      "status": "success",
      "data": {
        "resultType": "matrix",
        "result": [{
          "metric": {"worker": "trainer_0"},
          "values": [[1710001000, "1.50"], [1710001015, "1.51"]]
        }]
      }
    }
  }
}
```

Run it with the exact metric key used in the wrapper:

```powershell
python -m experiment.degradation_perception.main --path prometheus.json --metrics rl_insight_monitor_timing_s_step --source-type prometheus
```

The adapter accepts an empty result or exactly one scalar label series. It
rejects multiple series rather than silently overwriting equal timestamps; use
PromQL aggregation or a complete label selector first. Native histograms must
be reduced to a scalar series (for example with `histogram_quantile`). Counter
metrics should normally be queried through `rate`, `irate`, or `increase`
because a persistent counter trend is intentionally rejected by the stable
segment detector. Prometheus string values are converted during preprocessing,
and `NaN`/`Inf` pairs are dropped.

The metric key is not automatically renamed. In particular, the monitor's
Prometheus exposition may normalize `timing_s/step` to an underscore-based,
namespace-prefixed name. Use the `__name__` returned by the actual query.

This explicit two-phase JSON schema is an independent, deterministic adapter
chosen because neither the approved behavior specification nor the existing
repository defines a legacy degradation-log schema. It must not be described
as the unavailable legacy format.

## Remote JSON Lines contract

The remote adapter can feed incremental JSON Lines into
`parse_dataset_text(..., suffix=".jsonl")`. Every non-empty line must explicitly
contain `phase`, `timestamp`, and a `metrics` object:

```json
{"phase":"standard","timestamp":1,"metrics":{"timing_s/step":1.0}}
{"phase":"inference","timestamp":10,"metrics":{"timing_s/step":1.5}}
```

`phase` must be exactly `standard` or `inference`. It is never inferred from a
timestamp, filename, missing bound, or current wall-clock time. Inference rows
are filtered by the optional inclusive time window; standard rows are retained.

## Preprocessing semantics

Each metric is processed independently in this fixed order:

1. Check for `None`, empty arrays, and unequal lengths.
2. Convert timestamps and values to numeric types.
3. Drop an invalid or non-finite timestamp/value pair together.
4. Stable-sort aligned pairs by timestamp.
5. Deduplicate timestamps, retaining the last valid value from original input
   order for equal timestamps.
6. Check the configured minimum valid sample count.

Numeric timestamps retain their supplied unit. Aware datetimes are converted to
Unix seconds; naive datetimes are deterministically interpreted as UTC. Boolean
values are rejected rather than treated as `0` or `1`. Identity normalization
is used because no approved replacement formula is available; this choice is an
extension point and is not claimed to be equivalent to unavailable legacy code.

## Stable segments and thresholds

`StableSegmentDetector` performs the required full-history KDE, calls
`scipy.signal.find_peaks` on both density and negative density, filters only
positive density peaks, and retains all negative-density peaks as valleys.
Valleys split the numeric influence regions. Values in the same region are
still split when an intervening sample leaves that region or when the time gap
exceeds the configured continuity limit.

Candidate values are divided exactly as follows:

```python
n = len(values)
part_size = n // 3
part_1 = values[0:part_size]
part_2 = values[part_size:2 * part_size]
part_3 = values[2 * part_size:n]
```

All remainder points enter the third part. Each part uses its own population
standard deviation (`ddof=0`). Fewer than three points cannot pass. The current
independent stability-vote design performs all six directed comparisons between
the three subsegment means: `1 in 2`, `2 in 1`, `2 in 3`, `3 in 2`, `1 in 3`,
and `3 in 1`. Each comparison uses the reference subsegment's own `ddof=0`
standard deviation with a deterministic floor of 2% of the absolute reference
mean (plus machine precision). At least four flags must pass. This flag set and
the zero-deviation floor are independently selected engineering details because
the approved material fixes the call chain and vote count, but not every flag.

`is_within_std` keeps the approved `1.05` coefficient in its actual boundary
calculation and uses strict boundaries. The following values are behavioral
contracts for mean `1.0` and standard deviation `0.1`:

```text
1.19 -> false
1.18 -> true
0.81 -> false
0.82 -> true
```

Each stable segment produces its own KDE model. The upper base threshold is the
KDE-CDF `1 - alpha` quantile; the lower base threshold is the `alpha` quantile.
The configured upper and lower ratios (both at least `1`) expand each KDE
interval outward. Expansion is sign-aware, so positive lower bounds move down
and negative upper bounds move up. Multiple mode thresholds are never averaged.
An inference point is normal if it is compatible with any applicable normal
mode, and abnormal only if it is incompatible with every mode.

## Formal abnormal intervals

Nearby abnormal points may form a candidate interval while retaining a bounded
number of intervening normal points. A formal interval must satisfy all four
conditions, not a vote:

```text
duration > 0.5
abnormal point count >= 5
abnormal rate > 0.60
all adjacent timestamps satisfy the configured continuity gap
```

Diagnostics retain the required names: `condition_1` is duration,
`condition_2` is abnormal count, `condition_3` is abnormal rate, and
`condition_4` is timestamp continuity.

The fourth condition is the independently selected continuity rule required by
the approved specification. By default, the allowable gap is three times the
median positive sample gap; an explicit maximum can override it. Boundary
comparisons are intentionally strict where shown above.

Detection-window boundaries are adjusted only after interval validation. The
start is clamped to the requested/observed start and reduced by one current time
unit; the end uses the exclusive stop index, is clamped, and is increased by one
current time unit. The value `1` therefore means one step or one raw timestamp
unit, not unconditionally one second.

## Time modes

Time rules are centralized in `time_utils.py`:

```text
training log: value < 10000  -> step
training log: value >= 10000 -> millisecond
remote monitor: value > 10000 -> value / 10000 / 60 for display
Prometheus: preserve Unix seconds
```

The remote boundary is strictly greater than `10000`; it intentionally differs
from the training-log boundary.

## State and Response

Every metric owns an independent state:

```text
0: enough data; detection completed
1: standard data insufficient
2: inference data insufficient
```

Standard insufficiency is checked first. State `1` or `2` never fabricates
thresholds or degradation intervals and is never counted as an abnormal
history result. A recoverable failure in one metric does not stop other metrics.
An omitted task ID becomes `"default"` before it is used as a key or serialized.

The minimum Response shape is:

```json
{
  "taskId": "default",
  "states": {"timing_s/step": 0},
  "results": {
    "timing_s/step": {
      "state": 0,
      "message": "",
      "thresholds": [],
      "abnormalTimeRange": []
    }
  },
  "abnormalTimeRange": {"timing_s/step": []}
}
```

History is keyed independently by `(task_id, metric)`. Only valid state-0
detections enter its bounded queue. The first version uses an in-memory store;
therefore history survives repeated calls on the same detector/store but does
not persist across separate one-shot CLI processes. Remote monitoring should
reuse a detector or injected history store. Disk persistence is deliberately
not implied.

The same detector instance also caches each successfully validated standard
threshold model. This allows later remote polls containing only inference rows
to reuse the established baseline. The cache is rejected if the metric
configuration changes, and it is process-local: after a monitor restart, a
source batch must provide standard rows again before inference-only batches can
be evaluated.

## Configuration and paths

The bundled defaults include:

```yaml
abnormal_type: UP
alpha: 0.01
upper_ratio: 1.15
lower_ratio: 1.15
```

Per-metric configuration is created only by an explicit loading operation.
Importing the package does not create files. Existing user configuration is not
overwritten. Metric names are converted to deterministic safe filenames, for
example:

```text
timing_s/step -> timing_s__step.yaml
```

The original metric name remains configuration data. Path containment checks
prevent metric names from escaping the chosen configuration directory.

`common_config.yaml` controls history confirmation with `n_keep_result` and
`n_keep_abnormal`. The bundled `1/1` values make the one-shot sample immediately
observable while retaining the history abstraction.

## Remote Monitor safety

`remote_monitor.py` calls `DegradationPerception` directly and never invokes
the CLI. SSH, SFTP, remote-file, and Docker stream resources are closed in
`finally` paths. Host keys are verified, tests use mocks, offsets advance only
after successful parsing/detection, and a remote read failure produces no
degradation interval. Do not commit a real `monitor_config.yaml`; use the
placeholder-only `monitor_config.example.yaml`.

## Files

```text
main.py                    CLI boundary
algorithm.py               per-metric orchestration and history
stable_segment_detector.py KDE modes and three-part voting
kde_utils.py               KDE, peaks, valleys, CDF quantiles
interval_utils.py          candidate and formal intervals
preprocessing.py           strict input, Prometheus matrix adapter, preprocessing
normalization.py           identity-normalization extension point
time_utils.py              time modes and boundary adjustment
serialization.py           recursive JSON-native conversion
config_loader.py           safe YAML loading/copying
perception_config.py       shared types and constants
remote_monitor.py          mocked/testable SSH and Docker adapter
sample_data.json           one-command local demonstration
tests/                     unit and integration tests
```

## Verification

Run the module checks from the repository root:

```powershell
python -m compileall experiment/degradation_perception
python -m pytest -q experiment/degradation_perception/tests
python -m pytest -q tests/monitor/ut tests/recipe/data
git diff --check
```

The tests lock down the approved peak/valley chain, deterministic constant-data
jitter, `1 - alpha` KDE quantile, three-part remainder placement, golden
`is_within_std` boundaries, strict interval thresholds, timestamp boundaries,
state isolation, path safety, JSON serialization, CLI output, and mocked remote
resource cleanup.
