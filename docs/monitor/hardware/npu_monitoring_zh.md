# 昇腾 NPU 硬件监控

昇腾 NPU 指标由 MindCluster NPU Exporter 提供。每个需要监控的 NPU 节点都要运行一个 NPU Exporter。

第 1～3 节在需要监控的 NPU 节点上执行；第 4 节在运行 RL-Insight Server 的机器上执行。

## 1. 检查现有 NPU 服务

先检查是为了复用机器上已有的 NPU Exporter，避免重复安装和端口冲突：

```bash
npu-smi info
command -v npu-exporter
pgrep -af npu-exporter
curl --noproxy '*' -fsS http://<NODE_IP>:8082/metrics | head
```

`npu-smi info` 必须能正常显示 NPU，Exporter 才能通过驱动采集指标。按检查结果继续：

- `/metrics` 能正常返回：Exporter 已可用，跳过安装和启动，直接到第 4 节“在 RL-Insight Server 机器注册 NPU 监控端口”。
- 找到二进制，但 `/metrics` 无法访问：跳过安装，到第 3 节“启动并验证 NPU Exporter”。先确认实际端口没有被其他进程占用。
- 没有找到二进制：继续执行第 2 节。

## 2. 安装 NPU Exporter

以下脚本按昇腾 MindCluster `26.0.0` 二进制布局安装。下载和解压使用当前用户，需要写系统目录的命令使用 `sudo`：

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

直接运行二进制不需要 systemd service 文件。非 root 安装和版本差异请参考[昇腾官方安装文档](https://www.hiascend.com/document/detail/zh/mindcluster/2600/clustersched/dlug/docs/zh/scheduling/installation_guide/03_installation/manual_installation/03_npu_exporter.md)。

## 3. 启动并验证 NPU Exporter

`<NODE_IP>` 就是当前安装 NPU Exporter 这台机器的本机 IP。替换后直接启动二进制：

> **注意：** 以下命令要求当前用户已经具有有效的 `sudo` 凭据，或者在 root shell 中执行。`nohup` 启动后无法交互式输入 `sudo` 密码。

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

`/usr/local/Ascend/driver` 是昇腾驱动的安装目录。NPU Exporter 通过 `LD_LIBRARY_PATH` 从该目录加载 `libdcmi.so` 等驱动库；如果驱动安装在其他目录，请将命令中的路径替换为实际驱动路径。

`nohup` 让进程在后台运行，Exporter 日志写入 `-logFile` 指定的文件。进程退出后采集会停止，生产环境建议交给 Docker、Supervisor 或其他进程管理器。

在当前 NPU 节点验证：

```bash
curl --noproxy '*' -fsS http://<NODE_IP>:8082/metrics | head
```

## 4. 在 RL-Insight Server 机器注册 NPU 监控端口

以下操作都在运行 RL-Insight Server 的机器上执行。创建 `npu_targets.yaml`：

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

确认当前机器的 RL-Insight Server 已启动；服务已经运行时无需重复启动：

```bash
rl-insight server start --detach
rl-insight server targets add npu_targets.yaml
```

`targets add` 只注册 target 并 reload Prometheus，不管理 NPU Exporter 的生命周期。注册成功后，在 RL-Insight Grafana 看板中查看 NPU 利用率、显存、功耗、温度和网络吞吐等指标。

![RL-Insight NPU 硬件指标](https://github.com/mengchengTang/verl-data/blob/master/npu%E6%8C%87%E6%A0%87.png?raw=1)
