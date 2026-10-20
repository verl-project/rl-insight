# Hardware Monitoring

RL-Insight offers two hardware monitoring modes:

| Mode | Setup | When to use |
|---|---|---|
| In-process monitoring (default) | None; call `rl_insight.init()` | Recommended for RL training jobs |
| External exporters | Install and register an exporter on every node | Existing Prometheus setups, or machines without the training process |

## In-process monitoring (default)

Hardware metrics are collected by a background monitor started from the training process itself. No exporter installation, port exposure, or target registration is required.

- Collectors: CPU, memory, network, disk (psutil), NVIDIA GPU (NVML), Ascend NPU (`npu-smi`)
- Capability detection is automatic: missing vendor tooling disables only that collector and is visible in diagnostics; it never fails training
- In a Ray job, one collector per node is elected automatically, so host samples are not duplicated by multiple workers
- Guide: [中文](in_process_hardware_monitoring_zh.md) / [English](in_process_hardware_monitoring_en.md)

## External exporters (optional)

Exporters keep working for users who already operate Prometheus. RL-Insight only registers their endpoints; installation, startup, shutdown, and upgrades remain under user control.

| Metrics | Exporter | Default port | Guide |
|---|---|---:|---|
| CPU, memory, network | node_exporter | `9100` | [中文](cpu_monitoring_zh.md) / [English](cpu_monitoring_en.md) |
| Ascend NPU | NPU Exporter | `8082` | [中文](npu_monitoring_zh.md) / [English](npu_monitoring_en.md) |

To avoid duplicate panels, prefer one source per metric family: keep in-process collection enabled, or disable it and use the exporters:

```python
rl_insight.init(config={"hardware": {"enabled": False}})
```

Set `RL_INSIGHT_HARDWARE_ENABLED=0` instead when a launcher should toggle collection without changing the training config (the environment variable takes precedence over `hardware.enabled`).

Both sources feed the hardware metric row of the trainer dashboards (e.g. `verl_trainer_v1_with_vllm_engine`). Each hardware panel carries two queries per series (an in-process query and an exporter query), and the server shows exactly one of them when it stages the dashboards — each metric family switches to its exporter query only when the matching scrape target was registered through `rl-insight add-targets` under a recognised job name (`node-exporter` for CPU/memory/network/disk, `npu-exporter` for NPU, `dcgm` for GPU; the match is a case-insensitive prefix). Families whose exporter is absent keep the in-process query, so registering only some exporters does not blank the other panels. The panels never mix sources with PromQL `or`, so either mode lights up the same panels without duplicate lines.
