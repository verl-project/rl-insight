# Hardware Monitoring

RL-Insight collects hardware metrics through a background monitor started by `rl_insight.init()`. The monitor samples CPU, memory, network, disk, NVIDIA GPU and Ascend NPU metrics and reports them through the regular RL-Insight metric pipeline. No exporter installation, port exposure or target registration is required.

## Built-in hardware monitoring

Hardware monitoring is disabled by default and must be enabled explicitly with `config={"hardware": {"enabled": True}}` before it runs alongside the training process. No separate dashboard or exporter deployment is needed.

- Collectors: CPU, memory, network and disk (psutil), NVIDIA GPU (NVML) and Ascend NPU (`npu-smi`)
- Capability detection is automatic: missing vendor tooling disables only the affected collector and is visible in diagnostics; it never fails training
- In a Ray job, one monitor is created per node, so host samples are not duplicated by multiple workers
- Guide: [中文](hardware_monitoring_zh.md) / [English](hardware_monitoring_en.md)

The built-in monitor is the recommended source for the hardware panels in the trainer dashboards.

## Legacy exporter-based monitoring (deprecated)

Exporter-based monitoring is deprecated. The existing guides remain available for deployments that already operate Prometheus:

| Metrics | Exporter | Default port | Guide |
|---|---|---:|---|
| CPU, memory, network | node_exporter | `9100` | [中文](cpu_monitoring_zh.md) / [English](cpu_monitoring_en.md) |
| Ascend NPU | NPU Exporter | `8082` | [中文](npu_monitoring_zh.md) / [English](npu_monitoring_en.md) |

RL-Insight only registers exporter endpoints with Prometheus; installation, startup, shutdown and upgrades remain under user control. To avoid duplicate panels, use one source per metric family: keep built-in collection enabled, or disable it when the exporters remain the only source:

```python
rl_insight.init(config={"hardware": {"enabled": False}})
```

Set `RL_INSIGHT_HARDWARE_ENABLED=0` instead when a launcher should toggle collection without changing the training configuration. The environment variable takes precedence over `hardware.enabled`.

Both sources feed the hardware metric row of the trainer dashboards, such as `verl_trainer_v1_with_vllm_engine`. Each hardware panel carries two queries per series: the built-in query and the exporter query. The server shows exactly one of them when it stages the dashboards. Each metric family switches to its exporter query only when the matching scrape target was registered through `rl-insight add-targets` under a recognised job name (`node-exporter` for CPU, memory, network and disk; `npu-exporter` for NPU; `dcgm` for GPU; the match is a case-insensitive prefix). Families whose exporter is absent keep the built-in query, so registering only some exporters does not blank the other panels. The panels never mix sources with PromQL `or`, so either mode lights up the same panels without duplicate lines.

```{toctree}
:maxdepth: 1
:hidden:

CPU Hardware Monitoring <cpu_monitoring_en>
Ascend NPU Hardware Monitoring <npu_monitoring_en>
```
