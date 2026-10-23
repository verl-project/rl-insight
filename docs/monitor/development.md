# Monitor Development

This page describes the Ray monitoring implementation and compatibility
boundaries for contributors. For installation and usage, see
[Server Installation](./server_installation.md) and [Quick Start](./quick_start.md).

## Module responsibilities

| Existing module | Responsibility |
| --- | --- |
| `client/ray_monitor_client.py` | Resolve the Ray-job-scoped Hub and retain the creator's Run-close token. Worker clients drain their submitted events without ending the Run. |
| `collector/ray_monitor_hub.py` | Create the Run, collect SDK events, expose metrics on the bound port, export traces, and perform owner-controlled shutdown. |
| `server/registry.py` | Persist Run identity and lifecycle metadata; track target ownership and expiring registrations in memory. |
| `server/http_api.py` | Accept Run and target requests and periodically project active registrations into service discovery. |
| `utils/prometheus_utils.py` | Publish and renew registrations; maintain the existing Prometheus target file. |
| `server/runtime.py` | Launch services, prepare scrape configuration, and provision the existing dashboards with fixed query data sources. |

Paths above are relative to `rl_insight/`. Framework integrations continue using
the existing SDK and target registration calls; concurrency adds no
framework-specific adapter or exporter wrapper. The push backend is unchanged.

## Collection and registration

Each independent Ray job has a named, non-detached Hub and its own RunContext.
The SDK creates the timestamp-prefixed Run ID before registration and retries
with that identity. Workers initialized before the configured training client
buffer events and attach to its Run; they do not create an unnamed execution.
RunContext is defined once in `utils/protocol.py` and returned directly through
Ray, including connection information and live event counts.
The default `prometheus.metrics_report_port` is `0`; the Hub records the actual
port after binding. An explicit port retains fixed-port behavior.

Metrics use the existing direct collection path:

```text
SDK / native exporter -> target registration -> file_sd
Prometheus -> actual Hub / native exporter metrics endpoint
```

The server accepts registrations at `POST /api/v1/prometheus/targets`; it does
not fetch or aggregate metrics. Trace events continue through OTLP to Tempo.
Prometheus must reach every registered exporter address and port.

With RunContext, the SDK includes a process-level `publisher_id`. One background
thread per SDK process renews each target independently at TTL/3. The default
TTL is 30 seconds; `server.endpoint_ttl` accepts values from 1 to 3600 seconds.
Transient network failures and address conflicts are retried. Ended or unknown
Runs and other terminal request errors stop renewal.

The registry prevents an address from belonging to different active Runs or
publishers. Discovery metadata retains ownership even when exporter samples
already carry `run_id`. Internal `__rl_insight_run_id` and
`__rl_insight_publisher` labels are not stored as metric labels. Existing metric
names and scrape profiles remain in use.

The server projects registrations every five seconds and after target updates.
Prometheus then observes its configured file discovery refresh. The target file
is `data/targets/prometheus-targets.yml`; updates use a cross-process file lock
and atomic replacement. Existing registration paths also retain the Prometheus
reload request for compatibility.

## Run lifecycle

The Hub creator holds the owner token; the registry persists its hash. Creator
`finish()` performs these operations:

1. Mark the Run `closing` and persist its admitted `(job, address, publisher)`
   tuples. Existing registrations may renew; new targets are rejected.
2. Flush OTLP and wait for Prometheus to observe the SDK flush marker.
3. Mark the Run `ended`, remove only its targets, stop local renewal, and close
   the Hub's metrics HTTP endpoint.

Worker `finish()` drains its prior event submissions with an ordered Hub Context
read. It does not wait for a Prometheus scrape or OTLP export. Multiple workers
can therefore finish without serial backend waits in the Hub. Their events stay
in the shared collectors until the creator flushes once. Creator shutdown must
follow worker completion; the Hub rejects events once closing starts.
`final_collection_confirmed` records
whether the SDK flush succeeded. It does not confirm the last snapshot of
independently exiting native exporters, or assert training success.

`end_run()` releases local resources even when the close request fails. There is
currently no persistent retry of that request, so the registry may retain a
`registered` or `closing` Run after a shutdown-time server outage. Target expiry
does not automatically mark a Run failed or ended.

## Persistence and recovery

Run identity, allowlisted configuration and lifecycle metadata live only in
`data/runs.sqlite`. There are no per-Run JSON snapshots or copied telemetry
files. Prometheus and Tempo use shared storage under `data/prometheus` and
`data/tempo`.

Renewable target leases are intentionally not persisted. At server startup,
stale renewable discovery entries are removed; surviving SDK processes
re-register the same Run. The persisted closing admission list permits original
targets to recover without admitting new ones. Recovery requires the original
registry and a reachable server at the original address; a collection gap is
possible.

If a publisher process exits, its targets expire after their last successful
renewal plus the TTL and discovery refresh delays. If only an exporter exits
inside a surviving publisher process, renewal continues and Prometheus reports
`up=0`. No server-side endpoint probing infers process health.

The Hub has no explicit automatic restart or event replay. SDK metrics and
queued trace spans are held in memory before collection/export; abnormal exits
can lose them. Registration recovery is not a lossless telemetry guarantee.

The foreground runtime stops the stack when a managed service exits. Automatic
process restart is the deployment supervisor's responsibility. Use one server
process per stack; the registry does not provide distributed coordination or HA.

## Dashboard provisioning

Bundled JSON templates are copied from
`config/services/grafana/dashboards/` into `~/.rl-insight/runtime/dashboards`.
Grafana folders follow the directory structure. External dashboard directories
use the same destination; user instructions are in
[Custom Grafana Dashboards](./custom_dashboards.md).

Run-aware views reuse the latest bundled panels and layout, including training
step plots backed by `metric_log` point records in Tempo. The **Experiment**
variable supports multiple executions and All. Its display label contains the
experiment name, UTC creation time and short ID; its value selects Run IDs.
Choices depend on the project and observed metrics, independently of chart time.

Grafana uses fixed Prometheus and Tempo query data sources. The server validates
the selected Run IDs and constrains metric and trace expressions to that set.
No per-Run data sources or query-time Grafana mutations are needed. Timeline
identity metadata is hidden from plotted states; multi-Run lanes remain distinct.

## Compatibility and limits

- The default Hub port changes from `9092` to OS-assigned. Explicitly configure
  a port where static discovery or firewall policies require one.
- Creator `finish()` now ends the Run. It is not a pause/resume operation.
- Legacy clients without a publisher ID, and Run-less/manual registrations,
  retain persistent registration without automatic expiry.
- A new SDK can register directly with a server predating renewable targets,
  but cannot obtain renewal and owner-close behavior from it. Upgrade the
  server before SDK clients to use concurrent Run lifecycle management.
- The Run database migration adds lifecycle columns. The PR201 registry uses
  positional inserts and cannot create new Runs against the
  migrated database. Direct downgrade is unsupported; retain a pre-upgrade
  backup if rollback is needed.
- Direct scraping does not verify response ownership on each scrape. TTL limits
  stale registration lifetime but cannot eliminate address reuse within that
  window. Run query isolation is not tenant authentication.
- Prometheus controls scrape concurrency and budgets. Dual-job acceptance is
  not a high-concurrency or multi-node capacity benchmark. Platform proxy mode
  and external forwarding are outside this implementation.

## Validation coverage

`tests/monitor/e2e/test_run_isolation.py` covers direct discovery, address
ownership, independent Run closure, SDK re-registration after server restart,
lease expiry, address reuse, closing admission recovery, and independent Ray
jobs. These tests exercise application restart and simulated registration loss;
they do not establish recovery from every process-kill scenario.
