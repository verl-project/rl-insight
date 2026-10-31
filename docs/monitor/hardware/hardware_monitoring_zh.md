# 硬件监控

开启硬件监控后即可采集节点和设备指标，无需额外部署 exporter 或开放端口。

## 1. 前置条件

系统采集器（CPU、内存、网络、磁盘）依赖 psutil，该库是 RL-Insight 的内置依赖。厂商加速器为可选项：

| 加速器     | 要求                               | 说明                                  |
| ---------- | ---------------------------------- | ------------------------------------- |
| NVIDIA GPU | `pip install rl-insight[nvidia]` | 可选依赖组，安装 `nvidia-ml-py`     |
| 昇腾 NPU   | `npu-smi` 位于 `PATH`          | 随昇腾驱动提供，支持 Atlas A2、A3、A5 |

缺少厂商工具只影响对应设备的指标，训练和主机监控可以继续。

## 2. 开启硬件监控

硬件监控默认关闭，需通过 `rl_insight.init()` 的 `hardware` 配置段显式开启：

```python
import rl_insight

rl_insight.init(
    project="verl",
    experiment_name="ppo-run",
    config={
        "server": {"url": "http://<SERVER_IP>:18080"},
        "hardware": {
            "enabled": True,                # 默认值 False；设为 True 开启
            "interval_seconds": 10,         # 采样间隔，最小 5 秒
            "collectors": [                 # 默认 ["host", "device"]
                "host", "device",           # 按需加入 "ray_actor_process"、"self_process"
            ],
        },
    },
)
```

- 默认采集 `host`（CPU、内存、网络、磁盘）和 `device`（NVIDIA GPU、昇腾 NPU）。要查看当前 Ray 作业各 worker 的 CPU 和内存，加入 `ray_actor_process`；要查看监控进程自身的用量，加入 `self_process`。
- 默认采样间隔为 10 秒，低于 5 秒的配置按 5 秒处理。
- Ray 监控覆盖初始化时的本地节点，以及当时运行本作业活跃 actor 的节点。后续加入的节点不在此次采集范围内。`finish()` 的作用见 [Python API](../api.md)。

## 3. 指标

所有指标均为 Prometheus gauge，使用 `rlinsight_` 前缀命名，并以 RL-Insight 指标命名空间暴露，例如 `rl_insight_monitor_rlinsight_cpu_utilization_percent`。速率类指标由单调计数器差分计算，按每秒上报。NPU 指标名遵循 NPU Exporter 命名规范；GPU 与主机指标遵循对应 exporter 的语义。

| 分类           | 指标                                                                                                                                                                                                                                                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU            | `rlinsight_cpu_utilization_percent`                                                                                                                                                                                                                                                                                                             |
| 内存           | `rlinsight_memory_used_bytes`、`rlinsight_memory_utilization_percent`                                                                                                                                                                                                                                                                         |
| 网络           | `rlinsight_network_receive_bytes_per_second`、`rlinsight_network_transmit_bytes_per_second`                                                                                                                                                                                                                                                   |
| 磁盘           | `rlinsight_disk_read_bytes_per_second`、`rlinsight_disk_write_bytes_per_second`、`rlinsight_disk_utilization_percent`                                                                                                                                                                                                                       |
| Ray actor 进程 | `rlinsight_ray_process_cpu_percent`、`rlinsight_ray_process_rss_bytes`                                                                                                                                                                                                                                                                        |
| 自监控进程     | `rlinsight_process_cpu_percent`、`rlinsight_process_rss_bytes`                                                                                                                                                                                                                                                                                |
| NVIDIA GPU     | `rlinsight_gpu_utilization_percent`、`rlinsight_gpu_memory_used_bytes`、`rlinsight_gpu_memory_utilization_percent`、`rlinsight_gpu_temperature_celsius`、`rlinsight_gpu_power_watts`、`rlinsight_gpu_info`                                                                                                                            |
| 昇腾 NPU       | `rlinsight_npu_chip_info_utilization`、`rlinsight_npu_chip_info_hbm_used_memory`、`rlinsight_npu_chip_info_hbm_utilization`、`rlinsight_npu_chip_info_hbm_bandwidth_utilization`、`rlinsight_npu_chip_info_temperature`、`rlinsight_npu_chip_info_power`、`rlinsight_npu_chip_info_health_status`、`rlinsight_npu_chip_info_name` |

NPU 内存用量以 MB 为单位。磁盘利用率测量根文件系统（`path="/"`）。用 `node` 区分主机，用 `device_index` 区分同一主机上的物理设备。GPU 可见性筛选要求 `CUDA_VISIBLE_DEVICES` 使用整数编号。使用 UUID 或 MIG 标识时，指标可能涵盖 NVML 可见的所有 GPU。

Ray worker 指标带有 `pid`、`worker_role`、`local_rank` 和 `device_index`。监控进程自身的指标用 `pid` 区分，没有设备编号或本地 rank。

## 4. 在 Grafana 中查看

打开 trainer dashboard 的硬件区。面板使用内置采集指标。用 `hardware: Node` 选择主机，用 `hardware: Device` 筛选设备及关联的 worker。共享节点的主机指标包括该节点上的所有工作负载。进程面板需要开启相应的可选采集器才有数据。所选时段内有对应厂商的数据时，才会显示该厂商的设备行。

## 5. 故障排查

加速器面板为空时，在训练环境中检查 `npu-smi info` 能否列出设备，或确认已安装 NVIDIA 驱动和 `rl-insight[nvidia]`。Ray 节点缺失时，检查节点到服务端的连接和 Ray State API。多节点发现需要安装 `ray[default]` 并启用 Ray dashboard。

## 旧版 exporter 监控

基于 exporter 的硬件监控（node_exporter、NPU Exporter、DCGM）将在 0.5.0 版本移除。上述内置监控在无需部署独立 exporter 的情况下提供相同的硬件面板。旧版指南保留用于已有的 Prometheus 部署：

- [CPU、内存与网络 exporter（将在 0.5.0 版本移除）](cpu_monitoring_zh.md)
- [NPU Exporter（将在 0.5.0 版本移除）](npu_monitoring_zh.md)
