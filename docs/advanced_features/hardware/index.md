# Hardware Monitoring

RL-Insight collects hardware metrics from exporters through Prometheus. Install and run the exporter on every node you want to monitor:

| Metrics | Exporter | Default port | Guide |
|---|---|---:|---|
| CPU, memory, network | node_exporter | `9100` | [CPU Hardware Monitoring](cpu_monitoring_en.md) |
| Ascend NPU | NPU Exporter | `8082` | [Ascend NPU Hardware Monitoring](npu_monitoring_en.md) |


```{toctree}
:maxdepth: 1

CPU Hardware Monitoring <cpu_monitoring_en>
Ascend NPU Hardware Monitoring <npu_monitoring_en>
CPU 硬件监控 <cpu_monitoring_zh>
昇腾 NPU 硬件监控 <npu_monitoring_zh>
```

Check whether the exporter already exists before installing it. RL-Insight only registers its endpoint with Prometheus; exporter installation, startup, shutdown, and upgrades remain under user control.

After registration, view the metrics in the RL-Insight Grafana dashboards.

```{toctree}
:maxdepth: 1
:hidden:

CPU Hardware Monitoring <cpu_monitoring_en>
Ascend NPU Hardware Monitoring <npu_monitoring_en>
```
