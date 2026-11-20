# Hardware Monitoring

RL-Insight collects hardware metrics through a background monitor started by `rl_insight.init()`. The monitor samples CPU, memory, network, disk, NVIDIA GPU and Ascend NPU metrics and reports them through the regular RL-Insight metric pipeline. No exporter installation, port exposure or target registration is required.

## Built-in hardware monitoring

Hardware monitoring is disabled by default and must be enabled explicitly with `config={"hardware": {"enabled": True}}` before it runs alongside the training process. No separate dashboard or exporter deployment is needed.

- Collectors: a `host` collector for CPU, memory, network and disk (psutil), plus an auto-detected `device` collector for NVIDIA GPU (NVML) or Ascend NPU (`npu-smi`)
- Capability detection is automatic: missing vendor tooling disables only the affected backend and is visible in diagnostics; it never fails training
- In a Ray job, one monitor is created per node, so host samples are not duplicated by multiple workers
- Guide: [中文](hardware_monitoring_zh.md) / [English](hardware_monitoring_en.md)

The built-in monitor is the recommended source for the hardware panels in the trainer dashboards.

## Legacy exporter-based monitoring (removal planned in 0.5.0)

Exporter-based monitoring will be removed in version 0.5.0. The existing guides remain available for deployments that already operate Prometheus:

| Metrics | Exporter | Default port | Guide |
|---|---|---:|---|
| CPU, memory, network | node_exporter | `9100` | [中文](cpu_monitoring_zh.md) / [English](cpu_monitoring_en.md) |
| Ascend NPU | NPU Exporter | `8082` | [中文](npu_monitoring_zh.md) / [English](npu_monitoring_en.md) |

RL-Insight only registers exporter endpoints with Prometheus; installation, startup, shutdown and upgrades remain under user control. Exporter targets are not read by the hardware panels: the trainer dashboards, such as `verl_trainer_v1_with_vllm_engine`, query only the built-in RL-Insight series.

To stop running the built-in collector (for example, because Prometheus already scrapes the exporters), disable it:

```python
rl_insight.init(config={"hardware": {"enabled": False}})
```

```{toctree}
:maxdepth: 1
:hidden:

Built-in Hardware Monitoring <hardware_monitoring_en>
CPU Hardware Monitoring <cpu_monitoring_en>
Ascend NPU Hardware Monitoring <npu_monitoring_en>
```
