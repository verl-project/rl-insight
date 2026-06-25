# Server Installation

RL-Insight Monitor needs three Linux services before online monitoring can run:

| Service | Role | Required version | Installer version |
|---|---|---|---|
| Prometheus | stores and queries training metrics | `>= 2.30.0` | `2.54.1` |
| Tempo | stores and queries RL state traces | `>= 2.0.0` | `2.6.1` |
| Grafana | shows dashboards and trace views | `>= 13.0.0` | `13.0.0` |

Choose one of the three approaches below depending on your network environment.

## Supported Linux Platforms

Automatic and manual installation are Linux-only.

| OS family | CPU architectures |
|---|---|
| Ubuntu / Debian | `amd64` / `x86_64`, `arm64` / `aarch64` |
| CentOS / RHEL / Rocky / Alma | `amd64` / `x86_64`, `arm64` / `aarch64` |

Windows and macOS can run the training-side Python APIs, but RL-Insight does not manage local Prometheus, Tempo, or Grafana services there yet.

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

Place the three `.tar.gz` files in a single directory and run:

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

Follow Approach 2 to identify and download the three `.tar.gz` files.

### 2. Extract and place

The default managed directory is `~/.rl-insight/services`. All three services extract the same way: `tar -xzf` into a temp directory, then copy the output into place.

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

## Data Persistence

RL-Insight keeps server data on disk. By default, data is stored under `~/.rl-insight/data`:

| Service | Persistent data |
|---|---|
| Prometheus | `~/.rl-insight/data/prometheus` TSDB blocks |
| Tempo | `~/.rl-insight/data/tempo/traces` and `~/.rl-insight/data/tempo/wal` |
| Grafana | `~/.rl-insight/data/grafana` data, logs, and plugins |

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
