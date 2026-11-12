# Data Inspection

`rl-insight data inspect` reads a persisted RL-Insight data directory and lists the runs it contains. It reports each run's first and last observed timestamps across Prometheus and Tempo. The command does not require the server stack to be running.

## Inspect the default data directory

```bash
rl-insight data inspect
```

## Inspect a custom data directory

```bash
rl-insight data inspect --log-dir /path/to/rl-insight-data
```

`--log-dir` points to the same data directory used by `rl-insight server start --log-dir`.

| Option | Default | Description |
|---|---|---|
| `--log-dir` | `~/.rl-insight/data` | Data directory to inspect. |
| `--promtool-bin` | Auto-detected | Optional path to `promtool`; standard installs do not need it. |

`promtool` is found automatically through `PATH`, then under
`~/.rl-insight/services/prometheus`. Only set `--promtool-bin` when:

- Prometheus is installed in a custom location that is not on `PATH`.
- You inspect a copied data directory in a container or minimal environment without the normal RL-Insight services.
- Multiple Prometheus versions are installed and you need to force a specific `promtool`.

## Output

The output lists each project, run ID, and merged time range. Runs with the same original experiment name appear separately because each execution receives a unique run ID.

```text
+---------------------------+------------------------------------------------+-------------------------+
| Project                   | Run ID                                         | Time Range              |
+---------------------------+------------------------------------------------+-------------------------+
| deepeyes_rlinsight        | deepeyes_step50_val3_20260827_1655             | 2026-08-27 09:47～09:48 |
|                           +------------------------------------------------+-------------------------+
|                           | no_npu_dashboard_smoke_20260827_083955_2301620 | 2026-08-27 08:40～09:24 |
+---------------------------+------------------------------------------------+-------------------------+
```

The time range combines the earliest start and latest end found for a run across Prometheus and Tempo. If a range crosses days, the end is rendered on a second line. If no run data is found, the command prints `Data not found.`

## Notes

- Prometheus samples are read through `promtool`.
- Tempo traces are read from local Parquet files with PyArrow.
- The default `promtool` search order is PATH, then `~/.rl-insight/services/prometheus`.
