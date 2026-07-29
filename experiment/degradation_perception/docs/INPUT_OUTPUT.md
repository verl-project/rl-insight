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

The canonical loader used by `main.py` requires `status=success`,
`resultType=matrix`, and either an empty result or exactly one scalar label
series. Aggregate or select labels in PromQL before exporting that direct
input.

The separate offline simulation adapter accepts the same complete response and
can apply an explicit `select_by_labels` policy. The filtered result must still
be unique; it never selects `result[0]` or aggregates workers implicitly. See
[Prometheus Matrix Simulation](PROMETHEUS_SIMULATION.md) for the outer package,
diagnostics, structured errors, and runnable example.

Prometheus string values are converted to numbers. Non-finite pairs are
dropped. The direct `main.py --path` loader does not issue PromQL or HTTP
requests and does not rename metric keys. The separate
`prometheus_workflow.py` entry point does issue `query_range` requests from its
own YAML, then sends each full response through the same strict converter.
Each metric declares separate `standard_query` and `inference_query` values.
Both queries must isolate the intended run with labels that actually exist in
that deployment; an unlabelled global aggregation across tasks is unsafe.

Metric-type responsibilities remain at the real query layer:

- Gauge values may be used directly; the simulator generates only Gauge-style
  scalar values.
- Counter values should use `rate()` or `increase()` in real PromQL when that
  derived signal is intended.
- Histogram data should use a quantile, mean, or another scalar PromQL
  expression before it reaches this module.

The adapter does not see a `_total` suffix and guess or calculate a rate. It
does not implement PromQL semantics. A native histogram-only response is
reported as `unsupported_native_histogram`, not silently ignored.

The offline simulation validates Prometheus format compatibility and
end-to-end algorithm behavior. It does not validate real production
Prometheus data.

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
remote_monitor series with every value <= 10000:
               raw values are preserved
remote_monitor series containing a value > 10000:
               every display boundary uses value / 10000 / 60
```

The remote-monitor mode is resolved once from the complete inference series,
not independently for each interval endpoint. A series crossing `10000`
therefore remains monotonic. Explicit `training_log` and `prometheus` source
types take priority over numeric magnitude.

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

Configuration, malformed metric input, and unexpected per-metric detection
failures are not business states. The failed metric is omitted from `states`
and appears under the optional `metricErrors.<metric>` object with a stable
`code`, exception `type`, and redacted `message`. Other metrics continue.

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

When one metric fails independently, the response additionally contains:

```json
{
  "metricErrors": {
    "broken/metric": {
      "code": "metric_input_error",
      "type": "DataValidationError",
      "message": "metric input could not be validated"
    }
  }
}
```

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
