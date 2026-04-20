# RL-Insight - GMM 专家负载热力图可视化工具

## 一、简介

RL-Insight 是一个强化学习性能数据快速分析的可视化工具，基于 VeRL 框架采集的 profiling 数据、GMM 专家负载数据进行解析。其中 GMM 专家负载热力图功能用于可视化 MoE（Mixture of Experts）模型中专家的负载分布情况。

模块划分、流水线与扩展步骤见 [架构与开发指导](./architecture_and_guideline.md)。更完整的数据目录与 JSON 字段约定见 [数据规格与格式说明](./data/data_specification.md)。

### 主要功能

- **GMM 数据解析**：支持解析 VeRL 框架采集的 GMM 专家负载数据
- **专家负载热力图**：生成热力图展示不同专家在不同阶段的负载情况
- **灵活的过滤功能**：支持按 step、role、rank 进行数据过滤，自定义可视化范围

### 软件依赖

依赖版本以仓库根目录 [`requirements.txt`](../requirements.txt) 为准（含 **pandas、plotly、numpy、loguru** 等）。开发/运行前请安装：

```bash
pip install -r requirements.txt
```

若需从本地源码直接运行 `python -m rl_insight.main`，建议再执行：

```bash
pip install -e .
```

## 二、快速使用

### 2.1 采集 GMM 数据

使用 VeRL 框架，基于 msprobe 采集 GMM 专家负载数据，详细参考：

[VeRL 集成 msprobe 数据采集](https://github.com/verl-project/verl/pull/5186)
[VeRL 采集专家负载数据实践](https://github.com/verl-project/verl/issues/5985)

### 2.2 执行分析脚本

#### GMM 热力图使用示例

```bash
python -m rl_insight.main
    --input-path <gmm_data_path>
    --input-type gmm_data
    --profiler-type gmm
    --vis-type gmm_heatmap
    --rank-list all
    --step 1
    --role actor_compute_log_prob
    --output-path <output_path>
```

或修改并直接使用 `examples/gmm_exec.sh` 脚本:

```bash
bash examples/gmm_exec.sh
```

## 三、命令行参数

以下说明与 `python -m rl_insight.main --help` 保持一致；若有出入以命令行帮助为准。

| 参数 | 默认值 | 说明 |
|------|--------|----|
| `--input-path` | （必填，无默认值） | GMM 数据的根目录路径 |
| `--input-type` | `multi_json_mstx` | 输入数据类型，GMM 功能需设置为 `gmm_data` |
| `--profiler-type` | `mstx` | 性能数据种类，GMM 功能需设置为 `gmm` |
| `--output-path` | `output` | 输出路径，若为文件夹则在其中生成 `gmm_heatmap.png` |
| `--vis-type` | `html` | 可视化类型，GMM 功能需设置为 `gmm_heatmap` |
| `--rank-list` | `all` | Rank ID 列表，默认 `all` 表示所有 rank,可指定多个 rank 用逗号分隔 |
| `--pipeline-type` | `OfflineInsightPipeline` | 流水线实现类型 |
| `--step` | 无默认值 | 特定的 step 进行可视化（可选） |
| `--role` | 无默认值 | 特定的 role 进行可视化（可选） |
| `--rank` | 无默认值 | 特定的 rank 进行可视化（可选） |
| `--dpi` | 150 | 热力图输出的 DPI（默认 150） |
| `--cmap` | viridis | 热力图的颜色映射（默认 viridis） |

## 四、输出说明

工具会在指定的输出路径下生成 PNG 文件（文件名默认为 `gmm_heatmap.png`），包含：

- **专家负载热力图**：展示不同专家在不同时间点的负载情况
- **Segments 图**：顶部显示按 step、role、rank 分组的时间段
- **颜色编码**：使用 viridis 颜色映射，负载越高颜色越深
- **分隔线**：清晰分隔不同的 step、role、rank 组合
- **图例**：显示各个 segment 对应的 step、role、rank 信息

### 图表解读

1. **热力图区域**：
   - Y 轴：专家索引
   - X 轴：时间点（按 op 索引）
   - 颜色深度：表示专家负载程度（tokens per expert）

2. **Segments 图**：
   - 显示按 step、role、rank 分组的时间段
   - 每个 segment 有独特的颜色
   - 分隔线清晰标识不同分组的边界

3. **X 轴标签**：
   - 显示 op 索引（MoE layer / fused matmul id）
   - 当时间点较多时会自动调整显示密度

## 五、注意事项

1. GMM 热力图功能需要使用 `--input-type gmm_data` 和 `--profiler-type gmm` 参数
2. 当 `--output-path` 只指定文件夹路径时，会在该文件夹中生成 `gmm_heatmap.png` 文件
3. 当不指定 `--step`、`--role` 或 `--rank` 参数时，默认显示所有数据
4. 对于大量数据，工具会自动调整图表大小和标签显示密度，确保可读性
5. 数据文件需包含有效的专家负载数据，包括 step、role、rank_id、stage、expert_index 和 load 等字段

目录与 JSON 字段的集中说明另见 [数据规格与格式说明](./data/data_specification.md)。运行时校验逻辑以 `rl_insight.data.DataChecker` 及 [`rl_insight/data/rules.py`](../rl_insight/data/rules.py) 中的规则定义为准。