# Design Notes

This document records deterministic engineering choices and compatibility
boundaries. User-facing setup and verification remain in the
[module README](../README.md).

## Scope

The module implements the specified statistical workflow, but no complete
legacy degradation implementation or legacy degradation-log schema is present
in this repository. Where behavior was not fully specified, the module uses
deterministic, test-covered rules.

These rules should not be presented as byte-for-byte compatibility with an
unavailable implementation or input format.

## Deterministic Choices

The current implementation uses:

- identity normalization as an explicit extension point;
- full-history KDE before peak and valley analysis;
- all detected density valleys as value-region separators;
- six directed three-part stability comparisons;
- a 2% reference-mean floor for zero or near-zero standard deviation;
- at least four passing stability flags by default;
- one KDE threshold model per stable mode, without averaging modes;
- sign-aware ratio expansion for positive and negative thresholds;
- a continuity-based fourth formal-interval condition;
- three times the median positive sample gap as the default continuity limit;
- explicit phase labels for remote JSON Lines;
- rejection, rather than implicit merging, of multiple Prometheus series;
- optional exact label selection in the offline simulation adapter, with the
  filtered result still required to be unique.

The rules are centralized in the implementation and locked down by tests so
that future changes are reviewable.

## Configuration Safety

Per-metric configuration is created only during explicit loading. Importing the
package does not create files, and existing user configuration is not
overwritten.

The package-owned `default_config.yaml` is a read-only template. Without an
explicit `--config-dir`, generated per-metric files use
`%APPDATA%/rl-insight/degradation-perception` on Windows and
`${XDG_CONFIG_HOME:-~/.config}/rl-insight/degradation-perception` on POSIX.
An explicit directory always takes precedence.

Metric names are converted to deterministic safe filenames, while the original
metric name remains inside the configuration:

```text
timing_s/step -> timing_s__step.yaml
```

Path-containment checks prevent metric names from escaping the selected
configuration directory. Filename collisions are rejected when the stored raw
metric does not match the requested metric.

Configuration uses strict known-key validation. `upper_ratio` and
`lower_ratio` must be at least `1`, and `abnormal_type` must be `UP`, `DOWN`, or
`BOTH`.

## Input Safety

The local loader requires one explicit UTF-8 JSON file. It validates
timestamp/value lengths before paired processing, so mismatched arrays cannot
be silently truncated.

The direct canonical Prometheus adapter accepts at most one scalar series. The
offline simulation adapter may select one series by explicit labels, but it
does not merge equal timestamps across different label identities.

The real workflow requires separate standard and inference PromQL for every
logical metric. Queries must isolate the intended task before any aggregation;
the workflow records the actual phase, query, query window, returned count, and
candidate labels in diagnostics.

The CLI serializer produces standard JSON and rejects `NaN` and `Infinity`.
A recoverable failure in one metric does not stop unrelated metrics. Such an
error is reported in `metricErrors`, not reclassified as business state `1`.

## Process-Local State

History and standard-threshold caches are in memory:

- repeated calls on the same detector or injected history store can share
  history;
- separate one-shot CLI processes cannot;
- a long-lived remote monitor can reuse a valid standard model for later
  inference-only batches;
- a process restart requires standard data to establish the model again.

Disk persistence is not implied.

## Verification Boundaries

The module tests cover KDE and peak/valley behavior, deterministic
constant-data handling, three-part splitting, threshold direction, interval
conditions, time boundaries, state isolation, path safety, JSON serialization,
CLI output, Prometheus input, and mocked remote cleanup.

The related monitor and Recipe tests verify repository alignment, but they
require the Recipe dependency set. See
[Test and Verify](../README.md#test-and-verify) for the exact commands and success
criteria.
