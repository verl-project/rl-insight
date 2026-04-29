# Insight 框架概览

## 1. 框架概览
<div align="center">
 <img src="https://raw.githubusercontent.com/verl-project/rl-insight/main/assets/rl_insight_framework.svg" width="600" alt="rl-insight-arch.png">
</div>

图中绿色为数据侧模块，蓝色为功能侧模块；上述部分规划在 RL-Insight 内落地。红色模块 **CollectController** / **Collector** 已由 [verl.DistProfiler](https://verl.readthedocs.io/en/latest/perf/verl_profiler_system.html) 提供，短期内本仓库不会关注。

- **InputData / OutputData**：描述流水线两端的数据形态与约束。输入侧不限定单一 RL 框架，也可来自其它框架或离线整理产物；输出侧为各分析能力的结构化结果，供后续 **Visualizer** 等步骤消费。在实现上与 **DataRule** 对齐：`DataEnum`、`ValidationRule`（`rules.py`）、`DataChecker`，以及 **Parser** / **Visualizer** 的 `input_type`。
- **Offline/Online Parser**：负责将特定数据进行进一步加工解析的过程。
- **Plugin**：在线监控场景下，便于在第三方监控栈上做二次开发（接入、展示、导出等）。
- **Metric**：提供各种指标计算等进阶分析能力，这些指标通常是在 RL 精度与性能调试过程中非常有用的关键特征。
- **Collector**：跨平台 / 跨工具的数据采集、上报能力。
- **CollectController**：决定 **Collector** 采集时机与采集内容，通常会和特定的强化学习流程高度耦合。

---

## 2. 模块简介

| Concept | Location | Role |
|---------|----------|------|
| Entry | `rl_insight/main.py`, `rl_insight/pipeline/` | `main` 对接 CLI；`pipeline` 定义业务流程并选择 **Parser** / **Visualizer**。 |
| DataRule | `rl_insight/data/data_checker.py`, `rl_insight/data/rules.py` | `DataEnum` 区分数据阶段；`DataChecker` 按类型执行对应的 `ValidationRule`。 |
| Parser | `rl_insight/parser/parser.py`, `rl_insight/parser/*_parser.py` | 基于约定的 `input_type` 做解析；字段约定见 `rl_insight/utils/schema.py`（`DataMap`、`EventRow`、`Constant` 等）。 |
| Visualizer | `rl_insight/visualizer/visualizer.py`, `rl_insight/visualizer/timeline_visualizer.py`, … | 消费 **Parser** 输出，基于约定的 `input_type` 做可视化。 |

---