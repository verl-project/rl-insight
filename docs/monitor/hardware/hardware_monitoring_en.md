# Hardware Monitoring

Enable hardware monitoring to collect node and device metrics without deploying an exporter or opening another port.

## 1. Prerequisites

The system collectors (CPU, memory, network and disk) use psutil, which is a built-in dependency of RL-Insight. Vendor accelerators are optional:

| Accelerator | Requirement                        | Note                                                               |
| ----------- | ---------------------------------- | ------------------------------------------------------------------ |
| NVIDIA GPU  | `pip install rl-insight[nvidia]` | Optional extra; installs `nvidia-ml-py`                          |
| Ascend NPU  | `npu-smi` on `PATH`            | Included with the Ascend driver; Atlas A2, A3 and A5 are supported |

Missing vendor tools affect only the corresponding device metrics. Training and host monitoring continue.

## 2. Enable hardware monitoring

Hardware monitoring is disabled by default. It is enabled and configured through the `hardware` section of `rl_insight.init()`:

```python
import rl_insight

rl_insight.init(
    project="verl",
    experiment_name="ppo-run",
    config={
        "server": {"url": "http://<SERVER_IP>:18080"},
        "hardware": {
            "enabled": True,                # default is False; set True to enable
            "interval_seconds": 10,         # sampling interval, minimum 5
            "collectors": [                 # defaults to ["host", "device"]
                "host", "device",           # add "ray_actor_process", "self_process" as needed
            ],
        },
    },
)
```

- The default collectors are `host` (CPU, memory, network and disk) and `device` (NVIDIA GPU and Ascend NPU). Add `ray_actor_process` for CPU and memory by worker in the current Ray job, or `self_process` for the process hosting the monitor.
- Sampling defaults to 10 seconds. Intervals below 5 seconds are raised to 5.
- Ray monitoring covers the local node and nodes hosting active actors of the job at initialization. Nodes added later are outside that snapshot. See [Python API](../api.md) for `finish()` behavior.

## 3. Metrics

All metrics are Prometheus gauges named with the `rlinsight_` prefix and exposed with the RL-Insight metric namespace, for example `rl_insight_monitor_rlinsight_cpu_utilization_percent`. Rates are computed from monotonic counters and reported per second. NPU metric names follow the NPU Exporter naming convention; GPU and host metrics follow the corresponding exporter semantics.

| Family            | Metrics                                                                                                                                                                                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU               | `rlinsight_cpu_utilization_percent`                                                                                                                                                                                                                           |
| Memory            | `rlinsight_memory_used_bytes`, `rlinsight_memory_utilization_percent`                                                                                                                                                                                     |
| Network           | `rlinsight_network_receive_bytes_per_second`, `rlinsight_network_transmit_bytes_per_second`                                                                                                                                                                 |
| Disk              | `rlinsight_disk_read_bytes_per_second`, `rlinsight_disk_write_bytes_per_second`, `rlinsight_disk_utilization_percent`                                                                                                                                      |
| Ray actor process | `rlinsight_ray_process_cpu_percent`, `rlinsight_ray_process_rss_bytes`                                                                                                                                                                                     |
| Self process      | `rlinsight_process_cpu_percent`, `rlinsight_process_rss_bytes`                                                                                                                                                                                             |
| NVIDIA GPU        | `rlinsight_gpu_utilization_percent`, `rlinsight_gpu_memory_used_bytes`, `rlinsight_gpu_memory_utilization_percent`, `rlinsight_gpu_temperature_celsius`, `rlinsight_gpu_power_watts`, `rlinsight_gpu_info`                                                |
| Ascend NPU        | `rlinsight_npu_chip_info_utilization`, `rlinsight_npu_chip_info_hbm_used_memory`, `rlinsight_npu_chip_info_hbm_utilization`, `rlinsight_npu_chip_info_hbm_bandwidth_utilization`, `rlinsight_npu_chip_info_temperature`, `rlinsight_npu_chip_info_power`, `rlinsight_npu_chip_info_health_status`, `rlinsight_npu_chip_info_name` |

NPU memory usage is reported in MB. Disk utilization measures the root filesystem (`path="/"`). Use `node` to distinguish hosts and `device_index` to distinguish physical devices on a host. GPU visibility filtering requires integer `CUDA_VISIBLE_DEVICES` indexes. UUID and MIG selectors can result in metrics for all GPUs visible to NVML.

Ray worker metrics include `pid`, `worker_role`, `local_rank` and `device_index`. Self-process metrics identify the monitor process by `pid` and have no device or local rank.

## 4. View in Grafana

Open the hardware section of the trainer dashboard. The panels use built-in metrics. `hardware: Node` selects hosts, and `hardware: Device` filters devices and their workers. Host panels include all workloads on a shared node. Process panels need their corresponding optional collector. Vendor rows appear when the selected time range contains data for that vendor.

## 5. Troubleshooting

For an empty accelerator panel, check that `npu-smi info` lists the devices inside the training environment or that the NVIDIA driver and `rl-insight[nvidia]` are installed. For missing Ray nodes, check server connectivity and Ray State API availability. Multi-node discovery requires `ray[default]` with the Ray dashboard enabled.

## Legacy exporter-based monitoring

Exporter-based hardware monitoring (node_exporter, NPU Exporter and DCGM) will be removed in version 0.5.0. The built-in monitor described above provides the same hardware panels without a separate exporter deployment. The legacy guides remain available for existing Prometheus deployments:

- [CPU, memory and network exporters (removal in 0.5.0)](cpu_monitoring_en.md)
- [NPU Exporter (removal in 0.5.0)](npu_monitoring_en.md)
