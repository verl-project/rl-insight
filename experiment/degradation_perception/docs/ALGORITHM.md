# Algorithm

This document describes the detailed detection flow. For installation and the
test closure, start with the [module README](../README.md).

## Detection Flow

For each metric independently, the module:

1. preprocesses and aligns historical and inference points;
2. finds one or more stable historical performance modes;
3. builds a KDE threshold model for every stable mode;
4. marks an inference point abnormal only when it is incompatible with every
   applicable normal mode;
5. merges nearby abnormal points into candidate intervals;
6. applies four mandatory interval checks;
7. applies in-process history confirmation and returns formal intervals.

## Preprocessing

The processing order is fixed:

1. reject missing, empty, or unequal timestamp/value arrays;
2. convert timestamps and values to numeric types;
3. drop an invalid or non-finite timestamp/value pair together;
4. stable-sort aligned pairs by timestamp;
5. deduplicate timestamps, retaining the last valid value in original input
   order;
6. check the configured minimum valid sample count.

Boolean values are rejected instead of being treated as `0` or `1`.
Normalization is currently identity normalization.

## Stable Modes

`StableSegmentDetector` runs KDE over the full historical value range. It calls
`scipy.signal.find_peaks` on both density and negative density, filters density
peaks, and retains all detected density valleys. The valleys divide the value
range into peak influence regions.

Points in the same value region are still split if an intervening point leaves
the region or if the timestamp gap exceeds the continuity limit.

## Three-Part Stability Vote

A candidate segment is split as follows:

```python
n = len(values)
part_size = n // 3
part_1 = values[0:part_size]
part_2 = values[part_size:2 * part_size]
part_3 = values[2 * part_size:n]
```

Remainder points enter the third part. Each part uses population standard
deviation (`ddof=0`), and fewer than three points cannot pass.

The detector evaluates all six directed mean comparisons:

```text
1 in 2, 2 in 1, 2 in 3, 3 in 2, 1 in 3, 3 in 1
```

Each comparison uses the reference part's standard deviation with a
deterministic floor of 2% of the absolute reference mean plus machine
precision. At least four flags must pass by default.

`is_within_std` uses a strict boundary and the configured coefficient, `1.05`
by default. For mean `1.0` and standard deviation `0.1`:

```text
1.19 -> false
1.18 -> true
0.81 -> false
0.82 -> true
```

## KDE Thresholds

Every stable mode keeps its own KDE model:

- upper base threshold: KDE-CDF quantile `1 - alpha`;
- lower base threshold: KDE-CDF quantile `alpha`;
- `upper_ratio` and `lower_ratio`, both at least `1`, expand the normal
  interval outward;
- expansion is sign-aware, including negative-valued metrics.

Thresholds from multiple modes are never averaged. An inference point is
normal if it matches any applicable normal mode and abnormal only if it matches
none.

`abnormal_type` controls which direction is evaluated:

- `UP`: values above normal modes;
- `DOWN`: values below normal modes;
- `BOTH`: either direction.

## Formal Abnormal Intervals

Nearby abnormal points may form a candidate interval while retaining the
configured number of intervening normal points. A formal interval must satisfy
all four checks:

```text
condition_1: duration > minimum_duration
condition_2: abnormal point count >= minimum_abnormal_points
condition_3: abnormal rate > minimum_abnormal_rate
condition_4: every adjacent timestamp gap is within the continuity limit
```

The bundled defaults are:

```text
minimum_duration:        0.5
minimum_abnormal_points: 5
minimum_abnormal_rate:   0.60
```

The default continuity limit is three times the median positive sample gap.
`maximum_time_gap` can override it.

Output boundaries are padded only after these checks. See
[Duration and Display Boundaries](INPUT_OUTPUT.md#duration-and-display-boundaries)
for the exact field semantics.

## History and Threshold Cache

History is keyed by `(task_id, metric)`. Only successful state-`0` detections
enter the bounded history queue. Threshold models are cached after a valid
standard window so later remote inference-only batches can reuse them.

Both stores are process-local. A one-shot CLI invocation starts a new process,
and a remote monitor restart requires standard data again.

## Configuration Reference

The complete template is
[`default_config.yaml`](../default_config.yaml). It contains:

- metric direction, KDE tail probability, and threshold expansion ratios;
- minimum standard and inference point counts;
- normalization and KDE settings;
- stable-segment voting and continuity settings;
- abnormal-interval validation and continuity settings.

Per-metric settings are loaded on each detection call. History confirmation is
configured separately in [`common_config.yaml`](../common_config.yaml).
