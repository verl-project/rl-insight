# Ascend NPU Hardware Monitoring

MindCluster NPU Exporter provides Ascend NPU metrics. Run one exporter on every monitored NPU node.

Run Sections 1-3 on each monitored NPU node. Run Section 4 on the host where RL-Insight Server is running.

## 1. Check the existing NPU service

Check first so an existing exporter can be reused, avoiding a duplicate installation and port conflict:

```bash
npu-smi info
command -v npu-exporter
pgrep -af npu-exporter
curl --noproxy '*' -fsS http://<NODE_IP>:8082/metrics | head
```

`npu-smi info` must display the NPU correctly before the exporter can collect through the driver. Continue according to the result:

- `/metrics` returns successfully: skip installation and startup, then go directly to Section 4, "Register NPU monitoring endpoints on the RL-Insight Server host."
- A binary exists but `/metrics` is unreachable: skip installation and go to Section 3, "Start and verify NPU Exporter." Check that another process is not using the selected port.
- No binary exists: continue with Section 2.

## 2. Install NPU Exporter

The following script installs the MindCluster `26.0.0` binary layout. Download and extraction run as the current user; commands that write system directories use `sudo`:

```bash
(
set -euo pipefail

VERSION=26.0.0
ARCH="$(uname -m)"
case "${ARCH}" in
  aarch64|x86_64) ;;
  *) echo "Unsupported architecture: ${ARCH}"; exit 1 ;;
esac

PACKAGE="Ascend-mindxdl-npu-exporter_${VERSION}_linux-${ARCH}.zip"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

curl -fL \
  "https://gitcode.com/Ascend/mind-cluster/releases/download/v${VERSION}/${PACKAGE}" \
  -o "${WORK_DIR}/${PACKAGE}"
unzip -q "${WORK_DIR}/${PACKAGE}" -d "${WORK_DIR}/package"

sudo install -m 0500 "${WORK_DIR}/package/npu-exporter" /usr/local/bin/npu-exporter
sudo install -m 0400 "${WORK_DIR}/package/metricConfiguration.json" /usr/local/metricConfiguration.json
sudo install -m 0400 "${WORK_DIR}/package/pluginConfiguration.json" /usr/local/pluginConfiguration.json
sudo install -d -m 0750 /var/log/mindx-dl/npu-exporter
)
```

Running the binary directly does not require a systemd service file. See the [official Ascend installation guide](https://www.hiascend.com/document/detail/zh/mindcluster/2600/clustersched/dlug/docs/zh/scheduling/installation_guide/03_installation/manual_installation/03_npu_exporter.md) for non-root installation and version-specific details.

## 3. Start and verify NPU Exporter

`<NODE_IP>` is the local IP of the machine where NPU Exporter is being installed. Replace it, then start the binary directly:

> **Note:** The following command requires the current user to have valid `sudo` credentials or to run it from a root shell. After `nohup` starts, it cannot accept a `sudo` password interactively.

```bash
nohup sudo env \
  LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common \
  GOGC=50 GOMAXPROCS=2 GODEBUG=madvdontneed=1 \
  /usr/local/bin/npu-exporter \
  -ip=<NODE_IP> \
  -port=8082 \
  -updateTime=5 \
  -logFile=/var/log/mindx-dl/npu-exporter/npu-exporter.log &
```

`/usr/local/Ascend/driver` is the Ascend driver installation directory. NPU Exporter uses `LD_LIBRARY_PATH` to load driver libraries such as `libdcmi.so` from this directory. If the driver is installed elsewhere, replace these paths with the actual driver path.

`nohup` runs the process in the background, and Exporter logs go to the file specified by `-logFile`. Collection stops if the process exits; use Docker, Supervisor, or another process manager for production deployments.

Verify on the current NPU node:

```bash
curl --noproxy '*' -fsS http://<NODE_IP>:8082/metrics | head
```

## 4. Register NPU monitoring endpoints on the RL-Insight Server host

Run all commands in this section on the host where RL-Insight Server is running. Create `npu_targets.yaml`:

```yaml
jobs:
  - job_name: npu-exporter
    targets:
      - target: "<NODE_01_IP>:8082"
        labels:
          node: node-01
      - target: "<NODE_02_IP>:8082"
        labels:
          node: node-02
```

Confirm that RL-Insight Server is running on the current host. Skip the first command if it is already running:

```bash
rl-insight server start --detach
rl-insight server targets add npu_targets.yaml
```

`targets add` only registers targets and reloads Prometheus; it does not manage NPU Exporter. After registration, view NPU utilization, memory, power, temperature, and network throughput in the RL-Insight Grafana dashboards.

![RL-Insight NPU hardware metrics](https://github.com/mengchengTang/verl-data/blob/master/npu%E6%8C%87%E6%A0%87.png?raw=1)
