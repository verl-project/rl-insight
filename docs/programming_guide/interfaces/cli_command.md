# CLI Command

This page is the complete command reference for the `rl-insight` executable.

## Global options

| Command | Option | Default | Description |
|---|---|---|---|
| `rl-insight` | `--version` | - | Print the RL-Insight version and exit. |
| `rl-insight` | `--log-level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Set Python logging level. |

## Command summary

| Command | Purpose |
|---|---|
| `rl-insight server install` | Install and manage Prometheus, Tempo, and Grafana binaries. |
| `rl-insight server start` | Start the RL-Insight service stack. |
| `rl-insight server stop` | Stop the RL-Insight service stack. |
| `rl-insight server status` | List persisted projects, experiments, and time ranges without starting services. |
| `rl-insight server targets add <target_file>` | Add Prometheus scrape targets from a YAML file. |

## `rl-insight server install`

| Option | Default | Description |
|---|---|---|
| `--config <path>` | Bundled `config.yaml` | Server YAML used for service versions and settings. |
| `--install-dir <path>` | `~/.rl-insight/services` | Managed installation directory. |
| `--force` | Off | Reinstall services even when compatible binaries exist. |
| `--local-archive <dir>` | Disabled | Use local release archives instead of downloading. |

```bash
rl-insight server install
```

- For platform support, direct installation, custom installation directories, and offline installation, see [Server Installation](../../tutorial/server_installation.md).
- For server YAML keys and configuration precedence, see [Configuration](configuration.md).

## `rl-insight server start`

| Option | Default | Description |
|---|---|---|
| `--config <path>` | Bundled `config.yaml` | Server YAML used for services and ports. |
| `--log-dir <path>` | `~/.rl-insight/data` | Directory for Prometheus, Tempo, Grafana, and target state. |
| `--auto-port` | Off | Keep available configured ports and replace occupied ports automatically. |
| `--extra-dashboard-dir <path>` | Disabled | Load extra Grafana dashboard files from this directory. |
| `--detach` | Foreground | Start in background and return immediately. |
| `--attach-logs` | Foreground | Run in foreground and stream service logs. |

`--detach` and `--attach-logs` are mutually exclusive.

```bash
rl-insight server start --detach
```

- `--log-dir` is the persisted data directory. For custom data directories, backup, and restore, see [Data Directory Migration](../../advanced_features/data_directory_migration.md).
- `--config` controls server services and ports. For YAML settings and precedence, see [Configuration](configuration.md).
- `--extra-dashboard-dir` loads integration-owned dashboards without modifying the installed package. For file layout and provisioning details, see [Custom Grafana Dashboards](../../advanced_features/custom_grafana_dashboards.md).
- For service startup, URLs, and troubleshooting, see [Server Installation](../../tutorial/server_installation.md).

## `rl-insight server stop`

| Option | Default | Description |
|---|---|---|
| `--config <path>` | Bundled `config.yaml` | Server YAML used to locate the managed stack. |

```bash
rl-insight server stop
```

Stopping the stack does not delete collected data. For backup and restore workflows, see [Data Directory Migration](../../advanced_features/data_directory_migration.md).

## `rl-insight server status`

| Option | Default | Description |
|---|---|---|
| `--log-dir <path>` | `~/.rl-insight/data` | Persisted data directory to inspect. |
| `--promtool-bin <path>` | Auto-detected | Path to the `promtool` binary used to read Prometheus data. |

```bash
rl-insight server status --log-dir /path/to/rl-insight-data
```

For output format, data sources, and detailed behavior, see [Local Persisted Data Status](../../advanced_features/local_persisted_data_status.md).

## `rl-insight server targets add`

| Argument / Option | Required | Description |
|---|---|---|
| `<target_file>` | Yes | YAML file containing Prometheus jobs and targets. |
| `--config <path>` | No | Server YAML; defaults to the bundled `config.yaml`. |

```bash
rl-insight server targets add targets.yaml
```

This command only registers targets and reloads Prometheus; it does not manage exporter lifecycle. For exporter-specific target files, see [Hardware Monitoring](../../advanced_features/hardware/index.md).
