# Server Installation

RL-Insight Monitor needs three Linux services before online monitoring can run:

| Service | Role | Required version | Installer version |
|---|---|---:|---:|
| Prometheus | stores and queries training metrics | `>= 2.30.0` | `2.54.1` |
| Tempo | stores and queries RL state traces | `>= 2.0.0` | `2.6.1` |
| Grafana | shows dashboards and trace views | `>= 13.0.0` | `13.0.0` |

The recommended path is `rl-insight server install`. If the installer cannot download release assets because of network restrictions, use the manual commands below and then run `rl-insight server start`.

## Supported Linux Platforms

Automatic and manual installation are Linux-only.

| OS family | CPU architectures |
|---|---|
| Ubuntu / Debian | `amd64` / `x86_64`, `arm64` / `aarch64` |
| CentOS / RHEL / Rocky / Alma | `amd64` / `x86_64`, `arm64` / `aarch64` |

Windows and macOS can run the training-side Python APIs, but RL-Insight does not manage local Prometheus, Tempo, or Grafana services there yet.

## Option 1: RL-Insight Installer

From the Python environment where `rl-insight` is installed:

```bash
rl-insight server install
```

The command downloads Prometheus, Tempo, and Grafana into:

```text
~/.rl-insight/services
```

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

## Option 2: Manual Managed Install

Use this path when `rl-insight server install` fails because the node cannot reach GitHub release assets or `dl.grafana.com`.

RL-Insight automatically searches the default managed directory, so installing the binaries under `~/.rl-insight/services` is enough for `rl-insight server start`.

### 1. Prepare Paths And Architecture

```bash
export RL_INSIGHT_SERVICES="$HOME/.rl-insight/services"
mkdir -p "$RL_INSIGHT_SERVICES"

case "$(uname -m)" in
  x86_64|amd64) ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac
```

### 2. Install Prometheus

```bash
PROMETHEUS_VERSION=2.54.1
TMP_DIR="$(mktemp -d)"

curl -fL \
  -o "$TMP_DIR/prometheus.tar.gz" \
  "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz"

tar -xzf "$TMP_DIR/prometheus.tar.gz" -C "$TMP_DIR"
mkdir -p "$RL_INSIGHT_SERVICES/prometheus"
cp "$TMP_DIR/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/prometheus" \
  "$RL_INSIGHT_SERVICES/prometheus/prometheus"
chmod +x "$RL_INSIGHT_SERVICES/prometheus/prometheus"

"$RL_INSIGHT_SERVICES/prometheus/prometheus" --version
```

### 3. Install Tempo

```bash
TEMPO_VERSION=2.6.1
TMP_DIR="$(mktemp -d)"

curl -fL \
  -o "$TMP_DIR/tempo.tar.gz" \
  "https://github.com/grafana/tempo/releases/download/v${TEMPO_VERSION}/tempo_${TEMPO_VERSION}_linux_${ARCH}.tar.gz"

tar -xzf "$TMP_DIR/tempo.tar.gz" -C "$TMP_DIR"
mkdir -p "$RL_INSIGHT_SERVICES/tempo"
cp "$TMP_DIR/tempo" "$RL_INSIGHT_SERVICES/tempo/tempo"
chmod +x "$RL_INSIGHT_SERVICES/tempo/tempo"

"$RL_INSIGHT_SERVICES/tempo/tempo" --version
```

### 4. Install Grafana

```bash
GRAFANA_VERSION=13.0.0
TMP_DIR="$(mktemp -d)"

curl -fL \
  -o "$TMP_DIR/grafana.tar.gz" \
  "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz"

tar -xzf "$TMP_DIR/grafana.tar.gz" -C "$TMP_DIR"
mkdir -p "$RL_INSIGHT_SERVICES/grafana"
cp -a "$TMP_DIR/grafana-v${GRAFANA_VERSION}" \
  "$RL_INSIGHT_SERVICES/grafana/grafana-v${GRAFANA_VERSION}"

"$RL_INSIGHT_SERVICES/grafana/grafana-v${GRAFANA_VERSION}/bin/grafana-server" --version
```

### 5. Verify RL-Insight Can Find The Services

```bash
rl-insight server start
```

If the services start successfully, Grafana, Prometheus, Tempo, and the OTLP trace endpoint are printed in the terminal.

## Installing From An Internal Mirror Or Local Files

If compute nodes cannot access the public URLs, download the three archives from another machine or mirror them internally:

```text
prometheus-2.54.1.linux-amd64.tar.gz
prometheus-2.54.1.linux-arm64.tar.gz
tempo_2.6.1_linux_amd64.tar.gz
tempo_2.6.1_linux_arm64.tar.gz
grafana-13.0.0.linux-amd64.tar.gz
grafana-13.0.0.linux-arm64.tar.gz
```

Then replace the `curl` lines above with local archive paths. For example:

```bash
PROMETHEUS_ARCHIVE=/path/to/prometheus-2.54.1.linux-${ARCH}.tar.gz
TMP_DIR="$(mktemp -d)"
tar -xzf "$PROMETHEUS_ARCHIVE" -C "$TMP_DIR"
mkdir -p "$RL_INSIGHT_SERVICES/prometheus"
cp "$TMP_DIR/prometheus-2.54.1.linux-${ARCH}/prometheus" \
  "$RL_INSIGHT_SERVICES/prometheus/prometheus"
chmod +x "$RL_INSIGHT_SERVICES/prometheus/prometheus"
```

Use the same pattern for Tempo and Grafana.

## System Package Install

Company-managed environments may prefer system packages. This is supported when the versions meet the table above and the binaries are available on `PATH` or in common Linux locations such as `/usr/bin`, `/usr/local/bin`, `/usr/sbin`, or `/usr/share/grafana/bin`.

Check versions after installation:

```bash
prometheus --version
tempo --version
grafana-server --version
```

Ubuntu / Debian example:

```bash
sudo apt-get update
sudo apt-get install -y prometheus
```

CentOS / RHEL family example:

```bash
sudo yum install -y prometheus
# or, on newer distributions:
sudo dnf install -y prometheus
```

Grafana is usually installed from Grafana's official APT/RPM repository. Tempo is commonly installed from a Grafana package repository or release archive.

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

