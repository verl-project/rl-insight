# CPU 硬件监控

CPU、内存和网络指标由 Prometheus `node_exporter` 提供。每个需要监控的节点都要运行一个 node_exporter。

第 1～3 节在需要监控的节点上执行；第 4 节在运行 RL-Insight Server 的机器上执行。

## 1. 检查现有 CPU 服务

先检查是为了复用机器上已有的 node_exporter，避免重复安装和端口冲突：

```bash
command -v node_exporter || command -v prometheus-node-exporter
pgrep -af 'node_exporter|prometheus-node-exporter'
curl --noproxy '*' -fsS http://<NODE_IP>:9100/metrics | head
```

按检查结果继续：

- `/metrics` 能正常返回：Exporter 已可用，跳过安装和启动，直接到第 4 节“在 RL-Insight Server 机器注册 CPU 监控端口”。
- 找到二进制，但 `/metrics` 无法访问：跳过安装，到第 3 节“启动并验证 node_exporter”。先确认实际端口没有被其他进程占用。
- 没有找到二进制：继续执行第 2 节。

默认端口是 `9100`，已有服务也可能使用其他端口，以实际 `/metrics` 地址为准。

## 2. 安装 node_exporter

以下脚本安装 Prometheus 官方 `node_exporter 1.12.0`。下载和解压使用当前用户，只有写入 `/usr/local/bin` 时使用 `sudo`：

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

直接运行二进制不需要创建 `node_exporter` 用户，也不需要 systemd service 文件。

## 3. 启动并验证 node_exporter

直接启动二进制：

```bash
nohup /usr/local/bin/node_exporter \
  --web.listen-address=:9100 &
```

`:9100` 表示监听所有本地网络接口的 `9100` 端口。`nohup` 让进程在后台运行；进程退出后采集会停止。生产环境建议交给 Docker、Supervisor 或其他进程管理器。

`<NODE_IP>` 是当前被监控节点的本机 IP。在该节点验证：

```bash
curl --noproxy '*' -fsS http://<NODE_IP>:9100/metrics | head
```

请通过防火墙限制为仅允许 RL-Insight Server 访问。需要更换端口时，修改 `--web.listen-address=:<PORT>`，注册时填写相同端口。

## 4. 在 RL-Insight Server 机器注册 CPU 监控端口

以下操作都在运行 RL-Insight Server 的机器上执行。创建 `cpu_targets.yaml`：

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

确认当前机器的 RL-Insight Server 已启动；服务已经运行时无需重复启动：

```bash
rl-insight server start --detach
rl-insight server targets add cpu_targets.yaml
```

`targets add` 只注册 target 并 reload Prometheus，不管理 node_exporter 的生命周期。注册成功后，在 RL-Insight Grafana 看板中查看 CPU、实际内存用量和网络收发带宽。

![RL-Insight CPU 硬件指标](https://github.com/mengchengTang/verl-data/blob/master/cpu%E6%8C%87%E6%A0%87.png?raw=1)
