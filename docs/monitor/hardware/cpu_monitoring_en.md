# CPU Hardware Monitoring

Prometheus `node_exporter` provides CPU, memory, and network metrics. Run one node_exporter on every monitored node.

Run Sections 1-3 on each monitored node. Run Section 4 on the host where RL-Insight Server is running.

## 1. Check the existing CPU service

Check first so an existing exporter can be reused, avoiding a duplicate installation and port conflict:

```bash
command -v node_exporter || command -v prometheus-node-exporter
pgrep -af 'node_exporter|prometheus-node-exporter'
curl --noproxy '*' -fsS http://<NODE_IP>:9100/metrics | head
```

Continue according to the result:

- `/metrics` returns successfully: skip installation and startup, then go directly to Section 4, "Register CPU monitoring endpoints on the RL-Insight Server host."
- A binary exists but `/metrics` is unreachable: skip installation and go to Section 3, "Start and verify node_exporter." Check that another process is not using the selected port.
- No binary exists: continue with Section 2.

The default port is `9100`. An existing service may use another port; use its actual `/metrics` address.

## 2. Install node_exporter

The following script installs the official Prometheus `node_exporter 1.12.0`. Download and extraction run as the current user; only installation into `/usr/local/bin` uses `sudo`:

```bash
(
set -euo pipefail

VERSION=1.12.0
case "$(uname -m)" in
  aarch64|arm64) ARCH=arm64 ;;
  x86_64|amd64) ARCH=amd64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

ARCHIVE="node_exporter-${VERSION}.linux-${ARCH}.tar.gz"
BASE_URL="https://github.com/prometheus/node_exporter/releases/download/v${VERSION}"

curl -fLO "${BASE_URL}/${ARCHIVE}"
tar -xzf "${ARCHIVE}"
sudo install -m 0755 \
  "node_exporter-${VERSION}.linux-${ARCH}/node_exporter" \
  /usr/local/bin/node_exporter
)
```

Running the binary directly does not require a `node_exporter` user or a systemd service file.

## 3. Start and verify node_exporter

Start the binary directly:

```bash
nohup /usr/local/bin/node_exporter \
  --web.listen-address=:9100 &
```

`:9100` listens on port `9100` on every local interface. `nohup` runs the process in the background; collection stops if the process exits. Use Docker, Supervisor, or another process manager for production deployments.

`<NODE_IP>` is the local IP of the monitored node. Verify on that node:

```bash
curl --noproxy '*' -fsS http://<NODE_IP>:9100/metrics | head
```

Restrict access to the RL-Insight Server with a firewall. To use another port, change `--web.listen-address=:<PORT>` and register that same port.

## 4. Register CPU monitoring endpoints on the RL-Insight Server host

Run all commands in this section on the host where RL-Insight Server is running. Create `cpu_targets.yaml`:

```yaml
jobs:
  - job_name: node-exporter
    targets:
      - target: "<NODE_01_IP>:9100"
        labels:
          node: node-01
      - target: "<NODE_02_IP>:9100"
        labels:
          node: node-02
```

Confirm that RL-Insight Server is running on the current host. Skip the first command if it is already running:

```bash
rl-insight server start --detach
rl-insight server targets add cpu_targets.yaml
```

`targets add` only registers targets and reloads Prometheus; it does not manage node_exporter. After registration, view CPU, actual memory usage, and network throughput in the RL-Insight Grafana dashboards.

![RL-Insight CPU hardware metrics](https://github.com/mengchengTang/verl-data/blob/master/cpu%E6%8C%87%E6%A0%87.png?raw=1)
