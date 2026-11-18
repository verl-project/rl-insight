# Data Directory Migration

RL-Insight stores Prometheus metrics, Tempo traces, Grafana state, and scrape-target state under a single data directory. For whole-instance backup or migration, copy that directory and start the server against the copy with `--log-dir`.

This is simpler and more reliable than exporting and importing individual records: the on-disk formats remain unchanged, timestamps are preserved, and the complete stack state moves together.

## Default data directory

By default, RL-Insight uses:

```text
~/.rl-insight/data
```

The directory contains:

```text
data/
├── prometheus/
├── tempo/
├── grafana/
└── targets/
```

## Start with a custom data directory

Use `server start --log-dir` to point RL-Insight at a different data directory:

```bash
rl-insight server start --log-dir /path/to/rl-insight-data
```

`--log-dir` is intentionally aligned with TensorBoard's `--logdir`: it is the directory that contains the persisted run data.

| Option | Default | Description |
|---|---|---|
| `--log-dir` | `~/.rl-insight/data` | Data directory used by Prometheus, Tempo, Grafana, and target state. |

The directory is created automatically if it does not exist.

## Migrate a complete instance

1. Stop the source server:

   ```bash
   rl-insight server stop
   ```

2. Archive the entire data directory:

   ```bash
   tar -czf rl-insight-data.tar.gz -C ~/.rl-insight data
   ```

3. Copy the archive to the target machine:

   ```bash
   scp rl-insight-data.tar.gz user@target-host:/tmp/
   ```

4. Extract it to the target location:

   ```bash
   mkdir -p /path/to/rl-insight
   tar -xzf /tmp/rl-insight-data.tar.gz -C /path/to/rl-insight
   ```

5. Start the target server against that directory:

   ```bash
   rl-insight server start --log-dir /path/to/rl-insight/data
   ```

## Resume an archived instance on the same machine

```bash
rl-insight server stop
tar -xzf rl-insight-data.tar.gz -C /path/to/restore
rl-insight server start --log-dir /path/to/restore/data
```

## Recommendations

- Stop the server before copying the data directory.
- Keep the same Prometheus and Tempo versions on the target machine when possible.
- Preserve file ownership and permissions if the target user differs.
- Use a fresh, empty target directory when you want a clean restore.
- If you need to run source and target instances simultaneously, also change the service ports in the server config.

## Troubleshooting

### Data is not visible after migration

- Confirm that the target directory contains `prometheus/`, `tempo/`, `grafana/`, and `targets/`.
- Confirm that the server was started with the correct `--log-dir`.
- Check the service logs under `~/.rl-insight/services/logs/`.

### The target server will not start

- Verify that the target directory is writable.
- Verify that the target machine has compatible Prometheus and Tempo versions.
- Verify that the configured ports are not already in use.
