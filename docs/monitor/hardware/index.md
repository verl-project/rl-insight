# Hardware Monitoring

RL-Insight collects hardware metrics from exporters through Prometheus. Install and run the exporter on every node you want to monitor:

| Metrics | Exporter | Default port | Guide |
|---|---|---:|---|
| CPU, memory, network | node_exporter | `9100` | [中文](cpu_monitoring_zh.md) / [English](cpu_monitoring_en.md) |
| Ascend NPU | NPU Exporter | `8082` | [中文](npu_monitoring_zh.md) / [English](npu_monitoring_en.md) |

Check whether the exporter already exists before installing it. RL-Insight only registers its endpoint with Prometheus; exporter installation, startup, shutdown, and upgrades remain under user control.

After registration, view the metrics in the RL-Insight Grafana dashboards.
