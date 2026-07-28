# Input and Output

This document defines the accepted datasets, time semantics, states, and
response fields. For the shortest runnable example, see the
[module README](../README.md).

## Local JSON Contract

The CLI accepts one existing UTF-8 `.json` file. Its root must contain exactly
the `standard` and `inference` sections:

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

Rules:

- both sections must exist, even when one is empty;
- a canonical metric entry contains exactly `timestamps` and `values`;
- the arrays must have equal lengths;
- missing requested metrics become state `1` or `2`, rather than a file-format
  error;
- `--start-time` and `--end-time` are inclusive and filter inference only;
- metric keys are preserved exactly.

Directories, globs, CSV, and raw text training logs are not accepted.

## Prometheus `query_range` Contract

Already-fetched standard and inference responses can be placed under their
metric key:

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

Run with the exact wrapper key:

```powershell
python -m experiment.degradation_perception.main `
  --path prometheus.json `
  --metrics rl_insight_monitor_timing_s_step `
  --source-type prometheus
```

The adapter requires `status=success`, `resultType=matrix`, and either an empty
result or exactly one scalar label series. Aggregate or select labels in PromQL
before exporting the response. Reduce native histograms to a scalar series.
Query counters through `rate`, `irate`, or `increase` when a rate is the
intended performance signal.

Prometheus string values are converted to numbers. Non-finite pairs are
dropped. The module does not issue PromQL or HTTP requests and does not rename
metric keys.

## Remote JSON Lines Contract

Every non-empty UTF-8 line must contain an explicit phase, timestamp, and
metrics object:

```json
{"phase":"standard","timestamp":1,"metrics":{"timing_s/step":1.0}}
{"phase":"inference","timestamp":10,"metrics":{"timing_s/step":1.5}}
```

`phase` must be `standard` or `inference`; it is never inferred from time,
filename, or missing fields. Inference rows use the optional inclusive time
window, while standard rows are retained.

## Time Semantics

Numeric timestamps retain their supplied unit. Aware datetimes are converted
to Unix seconds, and naive datetimes are interpreted as UTC.

Source display rules are:

```text
training_log:  raw numeric timestamps are preserved in output
prometheus:    Unix seconds are preserved
remote_monitor with value > 10000:
               displayed as value / 10000 / 60
```

`--source-type training_log` does not parse raw log text. It only selects the
module's training-time convention.

## Metric States

Each metric has an independent state:

```text
0: enough data; detection completed
1: standard data insufficient
2: inference data insufficient
```

State `0` does not mean “no anomaly.” Read
`abnormalTimeRange.<metric>` to determine whether degradation was confirmed.
States `1` and `2` do not fabricate thresholds or intervals.

## Response Shape

The stable top-level shape is:

```json
{
  "taskId": "default",
  "states": {
    "timing_s/step": 0
  },
  "results": {
    "timing_s/step": {
      "state": 0,
      "message": "",
      "thresholds": [],
      "abnormalTimeRange": []
    }
  },
  "abnormalTimeRange": {
    "timing_s/step": []
  }
}
```

`results.<metric>` also contains detailed point diagnostics, current intervals,
and history confirmation fields when detection completes.

The CLI prints exactly one standards-compliant compact JSON object. It rejects
`NaN` and `Infinity`. Runtime errors use this shape and a non-zero exit code:

```json
{
  "ok": false,
  "error": {
    "type": "ValueError",
    "message": "..."
  }
}
```

## Interval Fields

| Field | Definition |
| --- | --- |
| `startTime` | Padded and source-formatted display start. |
| `endTime` | Padded and source-formatted display end. |
| `duration` | Last raw candidate timestamp minus first raw candidate timestamp, calculated before padding. |
| `abnormalPointCount` | Abnormal points inside the candidate. |
| `totalPointCount` | All retained points inside the candidate. |
| `abnormalRate` | Abnormal points divided by total points. |
| `abnormalType` | `UP` or `DOWN` for this interval. |
| `maximumAllowedGap` | Continuity limit used by validation. |
| `validationDetail` | Boolean results for the four mandatory checks. |

## Duration and Display Boundaries

`duration` and `startTime`/`endTime` intentionally describe different stages:

1. the candidate interval is identified;
2. `duration` is calculated from its first and last raw timestamps;
3. the candidate passes interval validation;
4. output boundaries are clamped and padded by one raw time unit;
5. display conversion is applied for the selected source type.

Example:

```text
raw abnormal timestamps: 104, 105, 106, 107, 108, 109
raw interval:            [104, 109]
duration:                109 - 104 = 5

padded startTime:        104 - 1 = 103
padded endTime:          109 + 1 = 110
```

Consequently:

```text
duration != endTime - startTime
5        != 110 - 103
```

`duration` is a time span in the raw input unit. It is neither the number of
points nor the padded display span. For six consecutive integer timestamps,
the raw span is five units.
