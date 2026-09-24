# Server Installation

RL-Insight Monitor needs three services before online monitoring can run:

| Service    | Role                                | Required version | Installer version |
| ---------- | ----------------------------------- | ---------------- | ----------------- |
| Prometheus | stores and queries training metrics | `>= 2.30.0`    | `2.54.1`        |
| Tempo      | stores and queries RL state traces  | `>= 2.0.0`     | `2.6.1`         |
| Grafana    | shows dashboards and trace views    | `>= 13.0.0`    | `13.0.0`        |

Choose one of the three approaches below depending on your network environment. To run the same stack in a container instead of installing services on the host, see the **Run with Docker** section below.

## Supported Platforms

Automatic installation supports Linux and Windows x64.

| OS family                    | CPU architectures                               |
| ---------------------------- | ----------------------------------------------- |
| Ubuntu / Debian              | `amd64` / `x86_64`, `arm64` / `aarch64` |
| CentOS / RHEL / Rocky / Alma | `amd64` / `x86_64`, `arm64` / `aarch64` |
| Windows | `amd64` / `x86_64` |

---

## Approach 1: Direct Installation (Official Source)

Suitable for users whose nodes can reach GitHub release assets and `dl.grafana.com` directly. The installer downloads and manages everything.

```bash
rl-insight server install
```

The command downloads Prometheus, Tempo, and Grafana into `~/.rl-insight/services`, then prints a dependency summary.

Then start the stack:

```bash
rl-insight server start
```

Use `rl-insight server start --auto-port` (optionally with `--detach`) to retain available configured ports and replace occupied ports with OS-assigned ports. This includes Tempo's internal gRPC and memberlist ports. Automatic mode waits for readiness and makes up to three startup attempts if a port is taken meanwhile. Actual ports are saved in the runtime config for service discovery, Grafana, and `server targets add`; original YAML is unchanged. Use the printed server and Grafana URLs, which may change between starts. Fixed-port mode is unchanged.

To use a custom data directory, pass `--log-dir`:

```bash
rl-insight server start --log-dir /path/to/rl-insight-data
```

The directory is created automatically and is used by Prometheus, Tempo, Grafana, and target state. See [Data Directory Migration](./data_migration.md) for backup and restore instructions.

Useful variants:

```bash
# Reinstall even if binaries already exist.
rl-insight server install --force

# Use a different managed install directory.
rl-insight server install --install-dir /opt/rl-insight/services

# Start in background and stop later.
rl-insight server start --detach
rl-insight server stop
```

---

## Approach 2: Offline Installation With Pre-downloaded Archives

Suitable for nodes that cannot access official sources or have no internet connection at all. Pre-download the archives on another machine and provide them locally.

### 1. Get the download list

Run `rl-insight server install` on the target node. Even if it cannot download, it prints the planned URLs before attempting:

```text
Planned downloads:
  prometheus   2.54.1    https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz
  tempo        2.6.1     https://github.com/grafana/tempo/releases/download/v2.6.1/tempo_2.6.1_linux_amd64.tar.gz
  grafana      13.0.0    https://dl.grafana.com/oss/release/grafana-13.0.0.linux-amd64.tar.gz
```

### 2. Download the archives

Use a machine with network access to download the files listed above. The filenames must match exactly.

### 3. Install from the local directory

Place the three release archives (`.tar.gz` or `.zip`, depending on platform) in a single directory and run:

```bash
rl-insight server install --local-archive /path/to/archives
```

RL-Insight checks the local directory for each archive by exact filename. Archives that match are copied and used directly; any missing archive falls back to the configured download URL. The version is verified implicitly — the archive filename must include the version configured in `install_version`.

---

## Approach 3: Manual Installation (No Installer)

Suitable for air-gapped environments, centralized operations with unified deployment tooling, or users who need full control over binary placement and custom deployment requirements.

RL-Insight searches for binaries in this order: manifest.json → `~/.rl-insight/services` → system PATH → system fixed paths. If your binaries are already on `PATH` or in a standard location, simply run `rl-insight server start`. To point at an arbitrary path, use `binary_path` in config:

```yaml
prometheus:
  binary_path: /opt/custom/prometheus
tempo:
  binary_path: /opt/custom/tempo
grafana:
  binary_path: /opt/custom/grafana-server
```

### 1. Get the archives

Follow Approach 2 to identify and download the three release archives (`.tar.gz` or `.zip`, depending on platform).

### 2. Extract and place

On Windows, extract the archives and configure `binary_path` for each service; keep the full Grafana directory. The commands below are for Linux.

```bash
PROMETHEUS_ARCHIVE=prometheus-2.54.1.linux-arm64.tar.gz
TEMPO_ARCHIVE=tempo_2.6.1_linux_arm64.tar.gz
GRAFANA_ARCHIVE=grafana-13.0.0.linux-arm64.tar.gz

# Prometheus — extracted directory name matches the archive (minus .tar.gz)
PROMETHEUS_DIR="${PROMETHEUS_ARCHIVE%.tar.gz}"
TMP_DIR="$(mktemp -d)"
tar -xzf "$PROMETHEUS_ARCHIVE" -C "$TMP_DIR"
mkdir -p ~/.rl-insight/services/prometheus
cp "$TMP_DIR/$PROMETHEUS_DIR/prometheus" ~/.rl-insight/services/prometheus/prometheus
chmod +x ~/.rl-insight/services/prometheus/prometheus

# Tempo — extracts a flat binary, no directory
TMP_DIR="$(mktemp -d)"
tar -xzf "$TEMPO_ARCHIVE" -C "$TMP_DIR"
mkdir -p ~/.rl-insight/services/tempo
cp "$TMP_DIR/tempo" ~/.rl-insight/services/tempo/tempo
chmod +x ~/.rl-insight/services/tempo/tempo

# Grafana — directory name drops the arch suffix; read it from the tarball
GRAFANA_DIR=$(tar -tzf "$GRAFANA_ARCHIVE" | head -1 | cut -d/ -f1)
TMP_DIR="$(mktemp -d)"
tar -xzf "$GRAFANA_ARCHIVE" -C "$TMP_DIR"
mkdir -p ~/.rl-insight/services/grafana
cp -a "$TMP_DIR/$GRAFANA_DIR" ~/.rl-insight/services/grafana/$GRAFANA_DIR
```

### 3. Verify and start

```bash
~/.rl-insight/services/prometheus/prometheus --version
~/.rl-insight/services/tempo/tempo --version
~/.rl-insight/services/grafana/$GRAFANA_DIR/bin/grafana --version

rl-insight server start
```

---

## Run with Docker

The project also provides an all-in-one image that bundles the CLI together with Prometheus, Tempo, and Grafana. With this approach nothing is installed on the host — no Node.js, no Grafana Labs repository, and no service binaries. Docker is the only requirement.

Build the image from the repository root:

```bash
docker build -f docker/Dockerfile -t rl-insight:latest .
```

To select a different Python base image or pip index, pass `PYTHON_VERSION` / `PIP_INDEX_URL` as `--build-arg` values.

Multi-architecture (amd64/arm64) images are also published to the GitHub Container Registry on every `v*` tag:

```bash
docker pull ghcr.io/verl-project/rl-insight:latest
```

A rolling `nightly` image is additionally rebuilt from `main` (daily schedule and pushes), so you can try the latest unreleased changes without waiting for a version tag:

```bash
docker pull ghcr.io/verl-project/rl-insight:nightly
```

Every nightly build also pushes an immutable `nightly-<date>-<sha>` tag (e.g. `nightly-20260925-3c92f22`), and the same name exists as a git tag in the repository, so a given build can be pinned and reproduced exactly. `latest` always tracks the most recent stable release and is never updated by nightly builds.

Start the full stack on Linux:

```bash
docker run -d --name rl-insight --network host --stop-timeout 45 \
    --shm-size=2g \
    -v rl-insight-data:/home/rl-insight/.rl-insight/data \
    rl-insight:latest
```

| Option                     | Meaning                                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `--network host`         | Services bind`0.0.0.0`, so ports 18080 / 9090 / 3200 / 3000 are published on the host directly (Linux only). |
| `--stop-timeout 45`      | Allow time for graceful shutdown (Prometheus data flush) before SIGKILL.                                       |
| `--shm-size=2g`          | Shared memory headroom; the Docker default of 64MB is too small.                                               |
| `-v rl-insight-data:...` | Persist server data in a named volume (see**Data Persistence** below).                                   |

The container runs as the non-root user `rl-insight` (uid 10001), matching the non-root deployment recommended above. No custom data path is configured: `~` simply resolves to `/home/rl-insight` for that user, so data still lives in the default `~/.rl-insight` layout described in **Data Persistence** below. The named volume mounts exactly that directory, which the image also declares as a `VOLUME`.

### Port Mapping

With `--network host`, service ports land on the host unchanged and cannot be remapped. On macOS/Windows, or when a host port is already taken by another process, use bridge networking and publish ports explicitly. The container port after `:` is fixed by each service; the host port before it is freely configurable:

```bash
docker run -d --name rl-insight --stop-timeout 45 --shm-size=2g \
    -p 18080:18080 -p 9090:9090 -p 3200:3200 -p 13000:3000 \
    -v rl-insight-data:/home/rl-insight/.rl-insight/data \
    rl-insight:latest
```

The example above maps Grafana to host port 13000 to avoid a conflict with an existing Grafana instance; browse `http://<server-ip>:13000`.

| Container port | Service           |
| -------------- | ----------------- |
| 18080          | rl-insight web UI |
| 9090           | Prometheus        |
| 3200           | Tempo             |
| 3000           | Grafana           |

### Health Check and Logs

The image defines a `HEALTHCHECK` that polls all four service endpoints every 30s:

```bash
docker inspect --format '{{json .State.Health}}' rl-insight
docker logs -f rl-insight
```

The repository also provides `tests/monitor/special_e2e/docker_smoke_test.sh`, the same four-stage container test used by CI (healthy start, graceful stop, restart, core-service failure):

```bash
tests/monitor/special_e2e/docker_smoke_test.sh rl-insight:latest
```

### Offline Image Distribution

Where a registry is unavailable, distribute the image as a file:

```bash
docker save rl-insight:latest | gzip > rl-insight-latest-linux-arm64.tar.gz
```

Name the file with the architecture it was built for. On the target host, `docker load -i rl-insight-latest-linux-arm64.tar.gz` restores the original name:tag.

---

## Data Persistence

RL-Insight keeps server data on disk. By default, data is stored under `~/.rl-insight/data`:

| Service    | Persistent data                                                          |
| ---------- | ------------------------------------------------------------------------ |
| Prometheus | `~/.rl-insight/data/prometheus` TSDB blocks                            |
| Tempo      | `~/.rl-insight/data/tempo/traces` and `~/.rl-insight/data/tempo/wal` |
| Grafana    | `~/.rl-insight/data/grafana` data, logs, and plugins                   |

`Ctrl+C` and `rl-insight server stop` stop processes only. They do not delete collected metrics, traces, or dashboard state.

Prometheus and Tempo retain data for `30d` by default:

```yaml
prometheus:
  retention_time: 30d
tempo:
  retention_time: 30d
```

Set a different persistent directory in server YAML:

```yaml
server:
  data_dir: /path/to/rl-insight/data
```

## Next Step

Continue with [Quick Start](./quick_start.md) to start the stack, instrument training code, and open Grafana.
