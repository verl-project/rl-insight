# Memory 分析模块

## 1. 简介

RL-Insight 的 Memory 分析模块包含 Memory Parser（解析）和 Memory Visualizer（可视化）两部分，完整的 Pipeline 如下：

```
profiling 数据 → MemoryParser → DataChecker → MemoryVisualizer → HTML 交互图表
```

模块划分、流水线与扩展步骤见 [架构说明](./architecture.md)。更完整的数据目录与 JSON 字段约定见 [数据规格与格式说明](../data/data_specification.md)。

### 1.1 主要功能

**Memory Parser**（`memory.parser.type=memory`）：

- **内存分配解析**：解析 Ascend Profiler 输出的 `operator_memory.csv`，提取算子级内存分配/释放记录
- **调用栈关联**：通过 `trace_view.json` 中的 `cpu_op` 事件，为每条内存记录匹配 Python 调用栈，便于定位内存申请源头
- **并行处理**：利用多进程并行解析多个 Rank 的内存数据，提升处理效率
- **结构化输出**：输出标准化的 DataFrame，包含 `name`、`size_kb`、`start_time_ms`、`duration_ms`、`total_allocated_mb`、`call_stack` 等字段

**Memory Visualizer**（`memory.visualizer.type=memory_html`）：

- **双图表交互展示**：Chart1 累计内存趋势折线图 + Chart2 算子甘特图，x 轴实时联动缩放
- **调用栈回溯**：点击任意 bar 查看该内存事件的完整调用栈
- **重叠事件检测**：同算子同时刻的多个内存分配自动标注
- **大数据分段**：百万级事件自动按时间窗口分片（最多 20 个 HTML 文件），段间智能导航
- **自包含输出**：生成独立 HTML + JS 文件对，浏览器直接打开无需服务器

### 1.2 软件依赖

除 RL-Insight 公共依赖外，Memory Parser 额外依赖：

| 库 | 用途 | 安装 |
|----|------|------|
| `ijson` | 流式解析大 JSON（`trace_view.json` 可达数百 MB） | `pip install ijson` |

## 2. 输入数据

### 2.1 目录结构

```text
<input-path>/
└── <role>/
    └── <date>_<time>_ascend_pt/
        ├── profiler_info_<rank_id>.json
        ├── profiler_metadata.json
        └── ASCEND_PROFILER_OUTPUT/
            ├── operator_memory.csv
            └── trace_view.json
```

### 2.2 数据要求

1. **采集方式**：使用 Ascend Profiler 采集，至少采集 level0 及以上数据，采用离散模式采集（`discrete=True`）
2. **离线解析**：采集数据需经过离线解析（`analyse=False`），离线解析参考 [MSTX 预处理](../utils/mstx_preprocessing.md)
3. **`operator_memory.csv`**：Ascend Profiler 输出的算子级内存分配记录，包含 `Name`、`Size(KB)`、`Allocation Time(us)`、`Duration(us)`、`Allocation Total Allocated/Reserved/Active(MB)`、`Device Type` 等字段
4. **`trace_view.json`**：完整时间线事件，用于调用栈关联。需包含 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件；文件可能较大，Parser 内部使用 `ijson` 流式解析
5. **`profiler_info_*.json`**：用于提取 `rank_id`
6. **`profiler_metadata.json`**：用于提取 `role`

### 2.3 `operator_memory.csv` 关键字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `Name` | str | 算子名称 |
| `Size(KB)` | float | 正数=申请，负数=释放 |
| `Allocation Time(us)` | float | 申请/释放时间戳（微秒） |
| `Duration(us)` | float | 占用时长（可能为空，表示未释放） |
| `Allocation Total Allocated(MB)` | float | 申请时刻累计已分配 |
| `Allocation Total Reserved(MB)` | float | 申请时刻累计预留 |
| `Allocation Total Active(MB)` | float | 申请时刻累计活跃 |
| `Device Type` | str | 设备类型（如 `NPU:0`） |

## 3. 快速使用

### 3.1 采集 Profiling 数据

使用 VeRL 框架 + Ascend Profiler 采集内存数据，详细参考：

[VeRL NPU Profiling 教程](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/performance/ascend_profiling_zh.rst)

### 3.2 离线解析

若 `ASCEND_PROFILER_OUTPUT` 目录尚未生成，需先执行离线解析：

```bash
python -m recipe.utils.mstx_preprocessing <profiling_data_path>
```

详见 [MSTX 预处理](../utils/mstx_preprocessing.md)。

### 3.3 执行 Memory Pipeline（Parser + Visualizer）

#### CLI 方式

```bash
# 完整 Pipeline：Parser → DataChecker → Visualizer
python -m recipe.main \
    input.path=<profiling_data_path> \
    memory.parser.type=memory \
    memory.visualizer.type=memory_html \
    output.path=<output_path>
```

执行后在 `output.path` 目录下生成：
- `memory_timeline_NN.html` — 交互式可视化页面（浏览器打开）
- `detail_data_NN.js` — 渲染数据文件

#### Python API 方式

```python
from recipe.parser import MemoryClusterParser
from recipe.visualizer import MemoryVisualizer
from recipe.data import DataChecker, DataEnum

# 1. 解析
parser = MemoryClusterParser({"input": {"rank_list": "all"}})
df = parser.run("<profiling_data_path>")

# 2. 校验（包含列结构和内容合法性）
DataChecker(DataEnum.MEMORY_SUMMARY, df).run()

# 3. 可视化
visualizer = MemoryVisualizer({"output": {"path": "<output_path>"}})
html_path = visualizer.run(df)
print(f"可视化输出: {html_path}")
```

### 3.4 Memory Visualizer 交互说明

在浏览器中打开 `memory_timeline_00.html`：

| 交互 | 操作 | 效果 |
|------|------|------|
| 缩放时间范围 | 拖拽 Chart1 数据区域或底部滑块 | Chart1 / Chart2 联动缩放 |
| 查看具体事件 | 悬停 Chart2 的 bar | 显示算子名、大小、起止时间、重叠事件 |
| 查看调用栈 | 点击 Chart2 的 bar | 详情面板展示完整 call_stack |
| 导航重叠事件 | 详情面板中点击重叠事件链接 | 跳转到对应 bar 的详情 |
| 精确时间输入 | 修改 Left / Right 输入框后回车 | 精确设定可视范围 |
| 跨段跳转 | 拖动到段边界外 | 顶部出现 Seg N → 跳转链接 |
| 拖拽平移 | 在图表区域按住拖动 | 平移时间范围 |

### 3.5 命令行参数

Memory 模块相关参数：

| 参数 | 必填 | 说明 |
|------|------|------|
| `input.path` | ✅ | Profiling 数据根目录路径 |
| `memory.parser.type` | ✅ | 指定 `memory` |
| `memory.visualizer.type` | ✅ | 指定 `memory_html` |
| `output.path` | 否 | 输出目录路径（默认 `output`） |
| `input.rank_list` | 否 | Rank 过滤，默认 `all` |

## 4. 输出说明

### 4.1 Visualizer 最终输出

Pipeline 执行后在 `output.path` 目录下生成：

| 文件 | 说明 |
|------|------|
| `memory_timeline_NN.html` | 自包含交互式可视化页面，浏览器直接打开（~18 KB） |
| `detail_data_NN.js` | 渲染数据文件，由 HTML 自动加载 |

`NN` 为两位序号（`00` ~ `19`）。若数据量超过 5000 条 bar 会自动分段，每个时间段生成一对文件。各段左上角提供 Prev/Next 导航链接，跨段拖动时自动提示跳转。

### 4.2 Parser 中间输出（DataFrame）

以下字段表供 **Python API 用户** 使用 `parser.run()` 的返回结果时参考。CLI 方式无需关注。

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 算子名称（如 `aten::empty`、`aten::matmul`） |
| `role` | str | RL 角色名称（如 `actor_update`、`actor_compute_log_prob`） |
| `rank_id` | int | Rank 标识 |
| `call_stack` | str | 完整 Python 调用栈（以 `;\r\n` 分隔）；未匹配到时为空字符串 |
| `call_stack_top` | str | 调用栈顶层入口（用户代码入口）；未匹配到时为空字符串 |
| `size_kb` | float | 内存大小（KB），正数=申请，负数=释放 |
| `start_time_ms` | float | 内存申请/释放时间（ms） |
| `duration_ms` | float | 内存占用时长（ms）；未释放时为 `0.0` |
| `total_allocated_mb` | float | 申请时刻累计已分配内存（MB） |
| `total_reserved_mb` | float | 申请时刻累计预留内存（MB） |
| `total_active_mb` | float | 申请时刻累计活跃内存（MB） |
| `device_type` | str | 设备类型（如 `NPU:0`） |

### 4.3 Parser 输出示例

```
              name               role  rank_id  size_kb  start_time_ms  duration_ms  ... call_stack_top
0    aten::empty      actor_update        0   1024.0          1000.100000      0.00000  ...  fsdp2.py(112): train_batch
1    aten::empty      actor_update        0   2048.0          3000.050000      0.00000  ...  fsdp2.py(120): train_batch
2  aten::matmul      actor_update        0   4096.0          2000.500000      0.00000  ...  model.py(60): forward
3  aten::unknown      actor_update        0    512.0          6000.000000      0.00000  ...  (empty)
4    aten::empty      actor_update        0  -1024.0          7000.000000      0.00000  ...  (empty)
```

### 4.4 调用栈匹配说明

Memory Parser 通过以下策略将 `operator_memory.csv` 中的内存记录与 `trace_view.json` 中的调用栈关联：

1. 在 `trace_view.json` 中筛选 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件
2. 按 `name` 分组，组内按 `ts`（算子开始时间）升序排序
3. 对每条内存记录，在同名算子组中查找 `ts ≤ Allocation Time` 的最近一条记录
4. 匹配语义：`ts` 是算子开始执行时间，`Allocation Time` 是算子内触发内存分配的时间，因此 `Allocation Time ≥ ts`
5. 未匹配到的记录，`call_stack` 和 `call_stack_top` 字段为空字符串

### 4.5 DataChecker 校验

Pipeline 在 Parser 输出和 Visualizer 输入之间自动执行 `DataEnum.MEMORY_SUMMARY` 校验：

| 规则 | 校验内容 |
|------|----------|
| `ParserOutputValidatorRule` | DataFrame 必须包含 `name`、`size_kb`、`start_time_ms`、`duration_ms`、`total_allocated_mb` |
| `MemoryContentRule` | 数值列可转 float；`name` 不含空值或 NaN；`size_kb` 至少存在一个正值 |

校验失败会抛出 `DataValidationError` 并列出具体原因。

## 5. 局限性

1. **Rank 过滤**：当前 `input.rank_list` 仅支持 `all`，暂不支持过滤指定 Rank
2. **调用栈匹配精度**：调用栈匹配基于算子名 + 时间戳二分查找，若同一算子在极短时间内多次调用且分配内存，可能匹配到非精确的调用栈
3. **大文件性能**：`trace_view.json` 可达数百 MB 甚至数 GB，使用 `ijson` 流式解析可避免内存溢出，但解析速度仍受文件大小影响
4. **仅支持 Ascend NPU**：当前仅支持 Ascend Profiler 输出格式，暂不支持 GPU（CUDA）内存分析
5. **Duration 为空**：若内存尚未释放，`Duration(us)` 为空，Parser 将 `duration_ms` 设为 `0.0`
6. **采集级别**：至少需要 level0 及以上数据，不支持 `level_none` 级数据
7. **离散模式**：需采用离散模式采集（`discrete=True`）
