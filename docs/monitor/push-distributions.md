# Distributions in push sinks

A `histogram` rollout rule retains the scrape-interval increments of classic
Prometheus histograms as `<name>.bucket` counters (with original `le` bounds),
`<name>.sum` and `<name>.count`. It also derives `<name>.mean` and
`<name>.quantile` gauges, with a `quantile` tag (default 0.5, 0.9, 0.95, 0.99).
These rules support nonnegative observations such as latencies and lengths.

```yaml
push:
  rollout:
    metrics:
      - source: vllm:time_to_first_token_seconds
        type: histogram
        name: ttft
        scale: 1000000
        presence: source_available
      - type: counter_ratio
        numerator: vllm:prefix_cache_hits_total
        denominator: vllm:prefix_cache_queries_total
        name: cache_hit_rate
```

`scale` affects mean/quantile gauges only; raw counters retain source units.
Quantiles use linear interpolation of bucket increments, matching the classic
histogram estimator. The first scrape, resets, changed bucket layouts and
inconsistent histograms establish a baseline without emitting fake quantiles.
Idle windows retain zero counter increments but have no latency observations.

Derived quantiles describe **one source label set in one scrape interval**.
Do not average them across replicas or time and label that a request percentile.
Use the preserved bucket counters in a histogram-capable backend for aggregated
quantiles. A dashboard that downsamples derived gauges must identify that
operation (for example, max of per-scrape P99).

`counter_ratio` joins numerator/denominator deltas by identical source labels.
Zero denominator increments, missing series and resets do not emit a ratio.
Optional `scale` and `offset` apply to the resulting ratio; e.g. acceptance length
can use accepted tokens / draft steps + 1 for the bonus token.

`ratio` divides two gauge sources and joins them by identical source labels as
well, so a scrape that exposes one series per `model_name` or `replica` produces
one ratio per series, carrying that series' labels. A denominator with a single
label-free series is used as the shared divisor for every numerator series.
Series without a matching denominator, or whose own denominator is zero, are
skipped individually instead of suppressing the whole rule.

Any rule may set `presence: <metric-name>` to emit a source-presence gauge with
`metric=<rule-name>`. Missing sources produce presence=0, not fake zero latency.
A `counter_delta` rule may also select an exact histogram `_sum`/`_count` sample,
for example when measuring prefill token throughput from request_prompt_tokens_sum.
