# Server Data Status

`rl-insight server status` reads a persisted RL-Insight data directory and lists the experiments it contains. It reports each experiment's first and last observed timestamps across Prometheus and Tempo. The command does not require the server stack to be running.

## Inspect the default data directory

```bash
rl-insight server status
```

## Inspect a custom data directory

```bash
rl-insight server status --log-dir /path/to/rl-insight-data
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

The output lists each project, experiment, and merged UTC time range. Experiments with the same original name appear separately because each execution receives a unique `experiment_name`.

```text
+---------------------------+------------------------------------------------+-------------------------+
| Project                   | Experiment                                     | Time Range (UTC)        |
+---------------------------+------------------------------------------------+-------------------------+
| deepeyes_rlinsight        | deepeyes_step50_val3_20260827_1655             | 2026-08-27 09:47～09:48 |
|                           +------------------------------------------------+-------------------------+
|                           | no_npu_dashboard_smoke_20260827_083955_2301620 | 2026-08-27 08:40～09:24 |
+---------------------------+------------------------------------------------+-------------------------+
```

The time range combines the earliest start and latest end found for an experiment across Prometheus and Tempo. Timestamps are displayed in UTC. If a range crosses days, the end is rendered on a second line. If no experiment data is found, the command prints `Data not found.`

## Notes

- Prometheus samples are read through `promtool`, which must be version 2.54.0 or newer to support `tsdb dump-openmetrics`.
- Tempo traces are read from local Parquet files with PyArrow.
- The default `promtool` search order is PATH, then `~/.rl-insight/services/prometheus`; Windows `promtool.exe` is recognized.
