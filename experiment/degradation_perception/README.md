# Inference Performance Degradation Perception

This experimental module detects sustained degradation in a single
inference-performance metric, such as `timing_s/step`. It builds one or more
normal modes from historical data, classifies inference points, and returns
validated abnormal time ranges.

The module is isolated under `experiment/degradation_perception/` and does not
change the existing RL-Insight monitor, Recipe, Prometheus, or Grafana
pipelines.

## Current Status and Limitations

Supported:

- local UTF-8 JSON input with explicit `standard` and `inference` sections;
- one or more metrics, processed independently;
- `UP`, `DOWN`, and `BOTH` degradation directions;
- already-fetched, single-series Prometheus `query_range` matrix responses;
- incremental remote JSON Lines from an SSH file or Docker logs.

Not supported:

- direct PromQL or HTTP requests;
- raw text training logs, CSV, directories, globs, or implicit file discovery;
- automatic aggregation of multiple Prometheus label series;
- persistent history or threshold caches across processes.

`--source-type training_log` selects timestamp interpretation only. It does not
mean that the CLI can parse a raw training-log text file.

## Quick Start

Run all commands from the repository root.

### Install

For local detection:

```powershell
python -m pip install -e ".[degradation]"
```

To run the module tests as well:

```powershell
python -m pip install -e ".[degradation,test]"
```

### Run

```powershell
$result = python -m experiment.degradation_perception.main `
  --path experiment/degradation_perception/sample_data.json `
  --metrics timing_s/step | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) { throw "Detection command failed" }
$result.states.'timing_s/step'
$result.abnormalTimeRange.'timing_s/step' | ConvertTo-Json -Depth 6
```

The bundled sample produces state `0` and one formal abnormal interval. The
relevant fields from the actual response are:

```json
{
  "taskId": "default",
  "states": {
    "timing_s/step": 0
  },
  "abnormalTimeRange": {
    "timing_s/step": [
      {
        "startTime": 103.0,
        "endTime": 110.0,
        "duration": 5.0,
        "abnormalPointCount": 6,
        "totalPointCount": 6,
        "abnormalRate": 1.0,
        "abnormalType": "UP",
        "validationDetail": {
          "condition_1": true,
          "condition_2": true,
          "condition_3": true,
          "condition_4": true
        }
      }
    ]
  }
}
```

### How to Read the Result

- `state = 0` means that enough data was available and detection completed. It
  does **not** mean that no degradation was found.
- Check `abnormalTimeRange.<metric>` for confirmed degradation. An empty array
  means that no formal interval was confirmed.
- State `1` means insufficient standard data.
- State `2` means insufficient inference data.

For a strict smoke-test closure, assert the expected state and interval:

```powershell
if ($result.states.'timing_s/step' -ne 0) {
  throw "Expected completed detection"
}
if (@($result.abnormalTimeRange.'timing_s/step').Count -ne 1) {
  throw "Expected exactly one abnormal interval"
}
"Smoke test passed"
```

## Input Format

`--path` must point to one existing UTF-8 `.json` file. The root object must
contain `standard` and `inference`, and each section is keyed by the original
metric name:

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

The timestamp and value arrays must have equal lengths. Time filtering is
inclusive and applies only to `inference`; `standard` is never windowed.
Metric names remain JSON keys, so the slash in `timing_s/step` is data rather
than a path separator.

See [Input and Output](docs/INPUT_OUTPUT.md) for the complete local,
Prometheus, JSON Lines, time, and response contracts.

## CLI Arguments

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--path` | Yes | None | Input UTF-8 JSON file. |
| `--metrics` | No | `timing_s/step` | One or more metric keys separated by spaces. |
| `--start-time` | No | None | Inclusive lower bound for inference data only. |
| `--end-time` | No | None | Inclusive upper bound for inference data only. |
| `--task-id` | No | `default` | Isolation key for in-process history confirmation. |
| `--source-type` | No | `training_log` | One of `training_log`, `prometheus`, or `remote_monitor`. |
| `--config-dir` | No | Module `config/` | Directory for per-metric YAML files. |

The CLI writes exactly one compact JSON object to stdout. A runtime failure
writes a JSON error object and exits with a non-zero status.

## Output Fields

| Field | Meaning |
| --- | --- |
| `taskId` | Normalized task identifier. |
| `states.<metric>` | Per-metric state: `0`, `1`, or `2`. |
| `results.<metric>` | Thresholds, diagnostics, current intervals, and history result. |
| `abnormalTimeRange.<metric>` | Confirmed formal abnormal intervals. |
| `startTime` / `endTime` | Display boundaries after one raw time-unit of boundary padding. |
| `duration` | Raw candidate span before boundary padding: last raw timestamp minus first raw timestamp. |
| `abnormalPointCount` | Number of abnormal points in the interval. |
| `totalPointCount` | Total abnormal and retained normal points in the interval. |
| `abnormalRate` | `abnormalPointCount / totalPointCount`. |
| `validationDetail` | Results of the four mandatory interval checks. |

### Why `duration` May Differ from `endTime - startTime`

`duration` is calculated from the original candidate interval before output
boundaries are padded. It is not recalculated after padding.

For the bundled sample:

```text
raw abnormal interval: [104, 109]
duration:              109 - 104 = 5

display boundary padding:
startTime:             104 - 1 = 103
endTime:               109 + 1 = 110
```

Therefore, `duration = 5` while `endTime - startTime = 7`. This is expected
field semantics, not a failed calculation. `duration` uses the raw input time
unit; it is not the number of points.

## Configuration

On first explicit detection, the module creates a per-metric YAML file from
`default_config.yaml` if it does not already exist. Existing files are not
overwritten. For example:

```text
timing_s/step -> timing_s__step.yaml
```

The main controls are:

```yaml
abnormal_type: UP
alpha: 0.01
upper_ratio: 1.15
lower_ratio: 1.15
```

`abnormal_type` accepts `UP`, `DOWN`, or `BOTH`. Changes are loaded on the next
detection call. `common_config.yaml` controls in-process history confirmation
through `n_keep_result` and `n_keep_abnormal`.

The complete field list and validation rules are documented in
[`default_config.yaml`](default_config.yaml) and
[Algorithm](docs/ALGORITHM.md).

## Remote Monitor

Copy `monitor_config.example.yaml` to an untracked local file, fill in the SSH
source, and call `run_remote_monitor()` programmatically. The remote input must
be UTF-8 JSON Lines with explicit `phase`, `timestamp`, and `metrics` fields.

Do not commit credentials or a real monitor configuration. See
[Remote Monitor](docs/REMOTE_MONITOR.md) for configuration, execution, offset,
cache, and failure-recovery behavior.

## Verification

Use the following sequence to form a reproducible test closure.

1. Run the smoke test in [Quick Start](#quick-start) and confirm:
   command exit code `0`, metric state `0`, and exactly one abnormal interval.
2. Run all module tests:

   ```powershell
   python -m pytest -q experiment/degradation_perception/tests
   ```

3. Confirm that the module still aligns with the related RL-Insight paths.
   These tests require the main project Recipe dependencies:

   ```powershell
   python -m pip install -e ".[recipe,degradation,test]"
   python -m pytest -q tests/monitor/ut tests/recipe/data
   ```

4. Check compilation and the final diff:

   ```powershell
   python -m compileall experiment/degradation_perception
   git diff --check
   git diff -- experiment/degradation_perception
   ```

The closure is successful only when the sample assertions and the selected
test suites all pass. `.[degradation,test]` is sufficient for the module tests;
it does not install the Recipe stack needed by the cross-project checks.

## Common Errors

| Symptom | Check |
| --- | --- |
| State `1` | Standard metric is missing or has too few valid points. |
| State `2` | Inference metric is missing, filtered out, or has too few valid points. |
| JSON error with non-zero exit | Verify the file path, UTF-8 JSON shape, metric key, and equal array lengths. |
| Prometheus input rejected | Require `status=success`, `resultType=matrix`, and at most one scalar series. |
| Raw training log rejected | Convert it to the explicit two-phase JSON contract first. |
| Cross-project tests miss `pandas`, `loguru`, or `torch` | Install `.[recipe,degradation,test]`. |

## Project Structure

```text
main.py                    CLI boundary
algorithm.py               per-metric orchestration and history
stable_segment_detector.py stable modes and three-part voting
kde_utils.py               KDE, peaks, valleys, and CDF quantiles
interval_utils.py          candidate and formal intervals
preprocessing.py           strict input and Prometheus adapter
time_utils.py              time modes and boundary adjustment
remote_monitor.py          SSH file and Docker-log adapter
sample_data.json           deterministic smoke-test data
tests/                     unit and integration tests
docs/                      detailed technical documentation
```

## Detailed Documentation

- [Algorithm](docs/ALGORITHM.md): detection flow, KDE modes, thresholds, and
  interval validation.
- [Input and Output](docs/INPUT_OUTPUT.md): complete schemas, states, time
  semantics, and `duration`.
- [Design Notes](docs/DESIGN_NOTES.md): deterministic engineering choices and
  compatibility boundaries.
- [Remote Monitor](docs/REMOTE_MONITOR.md): configuration, execution, offsets,
  caches, and safety behavior.
