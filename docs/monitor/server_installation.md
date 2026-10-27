# Server Installation

Install and start one RL-Insight server stack, then connect your training jobs to
it. For a runnable training example, see [Quick Start](./quick_start.md).

## Requirements

Install the Python package from the repository root:

```bash
pip install -e .
```

The automatic service installer supports Linux x64/arm64 and Windows x64.
When upgrading an existing deployment, update the server before the training
environments.

| Service | Required version | Installed version |
| --- | --- | --- |
| Prometheus | `>= 2.30.0` | `2.54.1` |
| Tempo | `>= 2.0.0` | `2.6.1` |
| Grafana | `>= 13.0.0` | `13.0.0` |

## Install services

Choose one installation method.

### Online installation

Use this if the server machine can reach GitHub release downloads and
`dl.grafana.com`:

```bash
rl-insight server install
```

Services are installed under `~/.rl-insight/services`. Use `--force` to reinstall,
or `--install-dir /opt/rl-insight/services` to choose another location.

### Offline installation

1. Run `rl-insight server install` on the target machine to see the required
   archive filenames and download URLs.
2. Download all three archives on a connected machine and copy them into one
   directory on the target machine. Keep the filenames unchanged.
3. Install from that directory:

```bash
rl-insight server install --local-archive /path/to/archives
```

Include all three archives; missing files trigger an online download attempt.

### Use existing service binaries

If compatible binaries are already on `PATH`, skip installation and start the
server. Otherwise, set their paths in a YAML file. Keep the full Grafana
installation directory alongside its executable.

```yaml
prometheus:
  binary_path: /opt/prometheus/prometheus
tempo:
  binary_path: /opt/tempo/tempo
grafana:
  binary_path: /opt/grafana/bin/grafana
```

Pass this file to the start command with `--config server.yaml`.

## Start and stop

```bash
rl-insight server start
```

Use the server and Grafana URLs printed at startup. Press `Ctrl+C` to stop the
foreground stack.

| Option | Use |
| --- | --- |
| `--detach` | Run in the background. |
| `--attach-logs` | Run in the foreground and stream service logs. |
| `--auto-port` | Choose available ports when configured ports are occupied. Use the newly printed URLs after each start. |
| `--log-dir /path/to/data` | Choose where monitoring data is saved. |
| `--config server.yaml` | Load a custom server configuration. |

To stop a background stack:

```bash
rl-insight server stop
```

If you started with `--config`, pass the same file when stopping. Check that the
server is reachable before launching training:

```bash
curl -fsS http://<server-ip>:18080/healthz
```

The expected response is `{"status":"ok"}`. Replace the address and port with the
printed server URL.

## Connect training jobs

Set this in the environment used to launch each training job:

```bash
export RL_INSIGHT_SERVER_URL=http://<server-ip>:18080
```

Use an address reachable from every training node, and make the same setting
available to all workers. For multiple jobs, use the same server URL and leave
the training metrics port unset so each job receives an available port. See
[Monitor concurrent training jobs](./quick_start.md#monitor-concurrent-training-jobs).

For training on other machines, allow these connections:

| From | To | Default port |
| --- | --- | --- |
| Training processes | RL-Insight server | `18080` |
| Training processes | Trace ingestion | `4318` |
| Prometheus | Training metrics endpoints | Dynamically assigned, or explicitly configured per job |
| Your browser | Grafana | `3000` |

If firewall rules require fixed training metrics ports, set
`prometheus.metrics_report_port` in the training monitoring configuration. Jobs
on the same machine must use different ports. With `--auto-port`, use the service
ports printed at startup instead of the defaults above.

## Run with Docker

Build an image from the repository root; it includes the server services:

```bash
docker build -f docker/Dockerfile -t rl-insight:latest .
```

On Linux, use host networking so training processes can reach the service ports:

```bash
docker run -d --name rl-insight --network host --stop-timeout 45 \
    --shm-size=2g \
    -v rl-insight-data:/root/.rl-insight/data \
    rl-insight:latest
```

With bridge networking, publish the ports explicitly:

```bash
docker run -d --name rl-insight --stop-timeout 45 --shm-size=2g \
    -p 18080:18080 -p 9090:9090 -p 3200:3200 -p 4318:4318 -p 13000:3000 \
    -v rl-insight-data:/root/.rl-insight/data \
    rl-insight:latest
```

In this example, open Grafana at `http://<server-ip>:13000`. The server container
must also be able to reach the training metrics endpoints.

Check logs and stop the container with:

```bash
docker logs -f rl-insight
docker stop rl-insight
```

For offline distribution, use `docker save` on the build machine and `docker load`
on the target machine. Use an image built for the target CPU architecture.

## Keep monitoring data

The default data directory is `~/.rl-insight/data`. Use the same directory when
restarting the server to retain experiment history. Stopping the server does not
delete collected data. The Docker examples persist it in the `rl-insight-data`
volume.

Back up the data directory before upgrading. To roll back a server version,
restore its matching pre-upgrade backup.

Prometheus and Tempo retain data for `30d` by default. To change retention, add
these settings to your server YAML and start with `--config`:

```yaml
prometheus:
  retention_time: 30d
tempo:
  retention_time: 30d
```

For backup and restore, see [Data Directory Migration](./data_migration.md).
