# Prometheus Matrix Simulation

This workflow provides deterministic offline data for validating Prometheus
`query_range` compatibility and the complete degradation-perception pipeline.
It does not issue HTTP requests or PromQL queries.

> This test validates Prometheus format compatibility and end-to-end algorithm
> behavior. It is not validation of real production Prometheus data.

## Three Different Formats

The workflow deliberately keeps three layers separate:

| Layer | Purpose | Root shape |
| --- | --- | --- |
| Prometheus matrix response | One real `query_range` response for one query | `status` plus `data.resultType/result` |
| Offline simulation package | Project-owned container pairing logical metrics with queries and full responses | `formatVersion`, `source`, `queryStepSeconds`, `standard`, `inference` |
| Algorithm input | Existing KDE and association contract after conversion | exactly `standard` and `inference`, each containing `timestamps` and `values` |

The outer simulation package is **not** a native Prometheus response. Every
entry's nested `response` is one complete native-style matrix response.
Adapter diagnostics are saved separately and never added to the algorithm
input root.

## Prometheus Matrix Response

Each response follows the scalar `query_range` matrix shape:

```json
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {
          "__name__": "rl_insight_monitor_timing_s_step",
          "project": "verl",
          "experiment_name": "mock_inference",
          "job": "trainer_metrics",
          "instance": "127.0.0.1:9092",
          "worker": "trainer_0"
        },
        "values": [
          [1785301200.0, "1.002"],
          [1785301210.0, "0.998"]
        ]
      }
    ]
  }
}
```

Timestamps are numeric Unix seconds. Scalar values remain strings in the raw
response, matching Prometheus JSON. The response may contain zero, one, or
multiple label series; selection is always explicit.

Native histogram-only series are rejected with
`unsupported_native_histogram`. When both scalar `values` and `histograms`
exist, scalar values are used and diagnostics record that histogram samples
were present.

## Offline Package

The package maps each logical algorithm metric to its PromQL text and response:

```json
{
  "formatVersion": 1,
  "source": "simulated_prometheus_query_range",
  "queryStepSeconds": 10,
  "standard": {
    "timing_s/step": {
      "query": "rl_insight_monitor_timing_s_step{experiment_name=\"mock_standard\"}",
      "seriesPolicy": "exactly_one",
      "response": {}
    }
  },
  "inference": {
    "timing_s/step": {
      "query": "rl_insight_monitor_timing_s_step{experiment_name=\"mock_inference\"}",
      "seriesPolicy": "exactly_one",
      "response": {}
    }
  }
}
```

Logical names and Prometheus names are independently declared in the data.
The adapter does not rename metrics or infer a replacement from characters
such as `/` or `_`.

The default `seriesPolicy` is `exactly_one`:

- zero series returns `no_series`;
- one series is converted;
- multiple series return `multiple_series` with every series' labels.

Explicit selection uses exact label equality:

```json
{
  "seriesPolicy": "select_by_labels",
  "selectLabels": {
    "instance": "127.0.0.1:9092",
    "worker": "trainer_0"
  }
}
```

The filtered result must still contain exactly one series. The first version
does not aggregate workers or replicas. A selector with no match returns
`no_matching_series`; one that still matches multiple series returns
`multiple_matching_series`. Error details include returned and matching label
sets. Real-workflow diagnostics also include the phase-specific query window.

## Cleaning Rules

For the selected scalar series, the adapter:

1. requires each sample to be `[timestamp, value]`;
2. requires a numeric Unix-seconds timestamp and a string value;
3. converts both to finite floats while filtering value `NaN` and infinities;
4. sorts by timestamp;
5. keeps the last valid value for a duplicate timestamp;
6. rejects an empty result after cleaning;
7. preserves Unix seconds without dividing by `1000`, `10000`, or `60`;
8. verifies strict JSON serialization with `allow_nan=False`.

The output remains the existing algorithm contract:

```json
{
  "standard": {
    "timing_s/step": {"timestamps": [], "values": []}
  },
  "inference": {
    "timing_s/step": {"timestamps": [], "values": []}
  }
}
```

## Generated Scenario

The fixed-seed default package contains:

| Metric | Standard behavior | Inference behavior | Expected association |
| --- | --- | --- | --- |
| `timing_s/step` | stable near `1.0` | normal 0-69, high 70-119, recovered 120-179 | explicit target |
| `kv_cache_usage_perc` | stable near `46` | high for 100% of the target event | rank 1 |
| `response_length_mean` | stable near `820` | high for about 98% of the event | rank 2 |
| `num_requests_swapped` | low integer counts | high integer counts for about 90% | rank 3 |
| `e2e_request_latency` | stable near `1.45` | high for about 78% of the event | rank 4 |
| `global_seqlen_minimax_diff` | stable near `65` | noisy high values for about 64% | rank 5 |
| `unrelated_metric` | stable normal noise | same normal behavior | excluded as not abnormal |
| `constant_metric` | fixed at `5` | fixed at `5` | excluded as a constant series |
| `sparse_metric` | sparse stable observations | sparse observations including target-window anomalies | excluded for coverage or aligned-point count |

Dense standard series contain 120 points, dense inference series contain 180
points, and the nominal query step is 10 seconds. `DEFAULT_SEED` makes package
generation and random-forest configuration repeatable.

## Metric Types and PromQL Boundary

The simulator generates scalar samples representing Gauge or already-derived
scalar PromQL results:

- **Gauge:** may be queried directly as algorithm input.
- **Counter:** real PromQL should use `rate()` or `increase()` according to the
  intended signal before exporting a matrix response.
- **Histogram:** real PromQL should first calculate a quantile, mean, or other
  scalar expression.

The adapter never sees `_total` and guesses a rate, and it never implements
PromQL semantics. Counter and histogram conversion belongs to the real query
layer.

## Run End to End

Run from the repository root after installing the experiment requirements.

Windows PowerShell:

```powershell
python -m experiment.degradation_perception.simulated_prometheus `
  --output-dir .\.run-output\prometheus-simulation `
  --run-analysis
```

Linux Bash:

```bash
python -m experiment.degradation_perception.simulated_prometheus \
  --output-dir ./.run-output/prometheus-simulation \
  --run-analysis
```

Without `--config-dir`, the simulator safely creates its runtime configs under
the output directory. If the option is supplied, it must name a dedicated
test-only directory because the simulator updates its per-metric YAML files.

## Outputs

An analysis run writes:

```text
simulated_prometheus_matrix.json  raw offline package
converted_algorithm_input.json   strict standard/inference input
adapter_diagnostics.json          selection and cleaning diagnostics
analysis_result.json              unchanged public KDE/association result
top5_result.json                  compact tester-facing Top5 result
validation_summary.json           machine-readable acceptance checks
```

Acceptance succeeds only when the process exits with code `0`, stdout contains
`"ok": true`, and every `validation_summary.json.checks` value is `true`.

`analysis_result.json` continues to expose `states`, `results`,
`abnormalTimeRange`, and `associationAnalysis`. Each event's
`topAssociations` contains rank, metric, contribution, Pearson/Spearman,
selected signed correlation, RF importance, coverage, and aligned sample
count.

Contribution percentages are normalized across **all** valid candidates and
then truncated to Top 5. The displayed Top 5 is not renormalized, so it may sum
to less than 100% when more than five candidates are valid. Contribution is a
ranking score, not a causal probability.

## Limits

- No real Prometheus server is contacted.
- Real metric names and labels still require deployment-environment review.
- Real Counter and Histogram signals must be converted in PromQL.
- Multiple workers or replicas are not implicitly aggregated.
- The deterministic scenario demonstrates expected algorithm behavior but is
  not evidence about production distributions.
- Association contribution is not a root-cause or causal probability.

For real HTTP acquisition, use
[`prometheus_workflow.example.yaml`](../prometheus_workflow.example.yaml) and
the tester workflow in the main [README](../README.md).
