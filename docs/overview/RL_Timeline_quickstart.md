# RL Timeline 可视化工具

## 1. 简介

RL-Insight 是一个强化学习性能数据快速分析的可视化工具，基于 VeRL 框架采集的 profiling 数据进行解析，生成强化学习各阶段的 Timeline 图表。

模块划分、流水线与扩展步骤见 [架构与开发指导](./architecture_and_guideline.md)。更完整的数据目录与 JSON 字段约定见 [数据规格与格式说明](../data/data_specification.md)。

### 1.1 主要功能

- **数据解析**：支持解析 VeRL 框架采集的多格式 profiling 数据
- **并行处理**：利用多进程并行解析多个 Rank 的性能数据，提升处理效率
- **Timeline 可视化**：生成交互式 Timeline 甘特图，直观展示各 Rank 的事件分布
- **性能分析**：通过 Timeline 图表观察卡间负载不均衡、推理长尾等问题，帮助性能调优



## 2. 安装

需要 **Python >= 3.10**。

```bash
pip install rl-insight
```

若需使用最新未发布功能，可从源码安装：

```bash
git clone https://github.com/verl-project/rl-insight.git
cd rl-insight
pip install -r requirements.txt
pip install -e .
```


## 3. 快速使用

### 3.1 采集 Profiling 数据

使用 VeRL 框架采集性能数据，详细参考：

[VeRL NPU Profiling 教程](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/performance/ascend_profiling_zh.rst)

[VeRL GPU Profiling 教程](https://github.com/verl-project/verl/blob/main/docs/perf/nsight_profiling.md)

### 3.2 执行分析脚本

#### 3.2.1 MSTX 使用示例

```bash
python -m rl_insight.main \
   input.path=<profiling_data_path> \
   timeline.parser.type=mstx \
   output.path=<output_path>
```

或修改并直接使用 `examples/mstx_exec.sh` 脚本:

```bash
bash examples/mstx_exec.sh
```

#### 3.2.2 Torch Profiler 解析示例

工具支持解析 PyTorch Profiler 采集的性能数据（`torch` 类型）。

```bash
python -m rl_insight.main \
    input.path=<torch_profiling_data_path> \
    timeline.parser.type=torch \
    output.path=<output_path>
```

或修改并直接使用 `examples/torch_profiler_exec.sh` 脚本:

```bash
bash examples/torch_profiler_exec.sh
```

#### Nvtx Profiler 解析示例

工具支持解析 Nvtx Profiler 采集的性能数据（'nvtx' 类型）。

```bash
python -m rl_insight.main \
    input.path=<nvtx_profiling_data_path> \
    timeline.parser.type=nvtx \
    output.path=<output_path>
```

或修改并直接使用 `examples/nvtx_exec.sh` 脚本:

```bash
bash examples/nvtx_exec.sh
```

## 4. 命令行参数

以下说明与 `python -m rl_insight.main -h` 保持一致；若有出入以命令行帮助为准。

### 4.1 公共参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input.path` | （必填） | Profiling 数据的根目录路径 |
| `input.rank_list` | `all` | Rank ID 列表，如 `0,1,2` 或 `all` |
| `output.path` | `output` | 输出目录 |
| `preset` | 自动推断 | 预设名称：`timeline`、`heatmap`（根据 CLI 参数自动推断） |
| `config_path` | 无 | YAML 配置文件路径 |

### 4.2 Timeline 专属参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timeline.parser.type` | `mstx` | 数据源类型：`mstx`、`torch`、`nvtx` |
| `timeline.visualizer.type` | `html` | 可视化类型：`html`、`png` |
| `timeline.visualizer.width` | `2000` | 图片宽度（仅 png） |
| `timeline.visualizer.scale` | `2` | 图片缩放因子（仅 png） |

## 5. 输出说明

工具会在指定的输出路径下生成 HTML 文件（文件名默认为 `rl_timeline.html`），包含：

- **交互式 Timeline 甘特图**：展示各 Rank 在不同时间段的事件分布
- **悬停信息**：鼠标悬停显示事件详细信息（名称、开始/结束时间、持续时间等）
- **排序功能**：支持按默认排序或按 Rank ID 排序
- **缩放与导航**：支持图表缩放和时间轴导航

当前支持指定生成 PNG 图片格式（文件名默认为 `rl_timeline.png`），包含：

- **静态 Timeline 甘特图**：展示各 Rank 在不同时间段的事件分布
- **样式规范**：白底高清、白色边框分隔任务条、网格线辅助对齐
- **排序功能**：当前仅支持按 Rank ID 排序
- **自适应高度**：根据节点数量自动调整图片高度，保证所有分层完整展示

### 5.1 图表交互功能

1. **Hover 模式切换**：
   - "Hover: Current Only" - 仅显示当前悬停的事件信息
   - "Hover: All Ranks" - 显示所有 Rank 在同一时间点的信息

2. **Y 轴排序切换**：
   - "Sort: Default" - 默认排序
   - "Sort: By Rank ID" - 按 Rank ID 排序

3. **导出图片**：点击右上角相机图标可导出 PNG 图片

## 6. 注意事项

1. RL 分析功能当前仅支持处理所有 Rank（`input.rank_list` 参数暂不支持过滤功能）
2. 至少采集 level0 及以上数据（不支持 level_none 级数据）
3. 采用离散模式采集 `discrete=True`
4. MSTX 数据满足以下要求：
   - 采集数据需经过解析，仅支持使用离线解析方式（analyse=False）
   - 离线解析参考 [MSTX profiling 离线解析](../utils/mstx_preprocessing.md)
   - 输入路径下需包含 `*_ascend_pt` 目录
   - 每个 ascend_pt 目录下需包含 `profiler_info_*.json` 文件
   - trace_view.json 文件位于 `ASCEND_PROFILER_OUTPUT` 子目录中
5. torch 数据满足以下要求：
   - 输入路径下需包含以 `.json.gz` 结尾的 PyTorch Profiler 数据文件，即 verl 仓目前默认 torch_profile 采集数据保存格式
   - 系统会自动过滤包含 `async_llm` 关键字的文件
   - 每个数据文件需包含有效的 `traceEvents` 和 `distributedInfo` 字段
   - Rank ID 将从 `distributedInfo.rank` 字段自动提取
6. NVTX 数据满足以下要求：
   - 需要设置discrete=True的方式进行profile数据采集，采集数据文件名后缀为nsys-rep
   - 采集数据需要经过nsys解析为jsonl文件，文件格式为`worker_process_<PID>.<RID>.nsys-rep`都需要解析，解析命令为`nsys export --type jsonlines --force-overwrite=true -o "输出目录" "要转换的nsys-rep文件"`，更详细的解析命令请参考[Nvidia官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-export-command-switch-options)
   - 输入路径下需包含文件名格式为 `worker_process_<PID>.<RID>.jsonl`的数据文件，即 上一步解析后的文件，且所有要解析的数据文件要在同一个目录下。
   - 系统会自动筛选出满足`worker_process_<PID>.<RID>.jsonl`格式的文件

目录与 JSON 字段的集中说明另见 [数据规格与格式说明](../data/data_specification.md)。运行时校验逻辑以 `rl_insight.data.DataChecker` 及 [`rl_insight/data/rules.py`](../../rl_insight/data/rules.py) 中的规则定义为准。
