# Memory Parser - Ascend NPU 内存分析

## 1. 简介

Memory Parser 是 RL-Insight 的内存分析模块，基于 Ascend Profiler 采集的内存 Profiling 数据进行解析，为 RL 训练的内存瓶颈分析提供数据支撑。

模块划分、流水线与扩展步骤见 [架构说明](./architecture.md)。更完整的数据目录与 JSON 字段约定见 [数据规格与格式说明](../data/data_specification.md)。

### 1.1 主要功能

- **内存分配解析**：解析 Ascend Profiler 输出的 `operator_memory.csv`，提取算子级内存分配/释放记录
- **调用栈关联**：通过 `trace_view.json` 中的 `cpu_op` 事件，为每条内存记录匹配 Python 调用栈，便于定位内存申请源头
- **并行处理**：利用多进程并行解析多个 Rank 的内存数据，提升处理效率
- **结构化输出**：输出标准化的 `MemoryEventRow` DataFrame，供下游 Visualizer 或自定义分析脚本消费

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

### 3.3 执行 Memory Parser

#### CLI 方式

```bash
python -m recipe.main \
    --input-path <profiling_data_path> \
    --input-type ascend_memory \
    --profiler-type memory \
    --output-path <output_path>
```

#### Python API 方式

```python
from recipe.parser import MemoryClusterParser
from recipe.data import DataChecker, DataEnum

parser = MemoryClusterParser({"rank_list": "all"})

# 校验输入数据
DataChecker(DataEnum.ASCEND_MEMORY, "<profiling_data_path>").run()

# 解析数据
df = parser.run("<profiling_data_path>")

# 校验输出数据
DataChecker(DataEnum.SUMMARY_MEMORY_EVENT, df).run()

# df 为 pd.DataFrame，可直接用于后续分析
print(df.head())
print(df[["name", "size_kb", "call_stack_top", "start_time_ms"]])
```

### 3.4 命令行参数

以下仅列出 Memory Parser 相关参数，完整参数表见 [RL Timeline quickstart](./RL_Timeline_quickstart.md)。

| 参数 | Memory Parser 所需值 | 说明 |
|------|----------------------|------|
| `--input-type` | `ascend_memory` | 输入数据类型 |
| `--profiler-type` | `memory` | 指定使用 Memory Parser |

## 4. 输出说明

### 4.1 输出格式

Parser 的 `run()` 方法返回 `pd.DataFrame`，每行对应一条内存分配/释放记录，包含以下字段：

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

### 4.2 输出示例

```
              name               role  rank_id  size_kb  start_time_ms  duration_ms  ... call_stack_top
0    aten::empty      actor_update        0   1024.0          1000.100000      0.00000  ...  fsdp2.py(112): train_batch
1    aten::empty      actor_update        0   2048.0          3000.050000      0.00000  ...  fsdp2.py(120): train_batch
2  aten::matmul      actor_update        0   4096.0          2000.500000      0.00000  ...  model.py(60): forward
3  aten::unknown      actor_update        0    512.0          6000.000000      0.00000  ...  (empty)
4    aten::empty      actor_update        0  -1024.0          7000.000000      0.00000  ...  (empty)
```

### 4.3 调用栈匹配说明

Memory Parser 通过以下策略将 `operator_memory.csv` 中的内存记录与 `trace_view.json` 中的调用栈关联：

1. 在 `trace_view.json` 中筛选 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件
2. 按 `name` 分组，组内按 `ts`（算子开始时间）升序排序
3. 对每条内存记录，在同名算子组中查找 `ts ≤ Allocation Time` 的最近一条记录
4. 匹配语义：`ts` 是算子开始执行时间，`Allocation Time` 是算子内触发内存分配的时间，因此 `Allocation Time ≥ ts`
5. 未匹配到的记录，`call_stack` 和 `call_stack_top` 字段为空字符串

## 5. 局限性

1. **Rank 过滤**：当前 `--rank-list` 参数仅支持 `all`，暂不支持过滤指定 Rank
2. **可视化**：Memory Parser 当前输出 `MemoryEventRow` DataFrame，尚无对应的内置 Visualizer，需通过 Python API 自行分析或导出
3. **调用栈匹配精度**：调用栈匹配基于算子名 + 时间戳二分查找，若同一算子在极短时间内多次调用且分配内存，可能匹配到非精确的调用栈
4. **大文件性能**：`trace_view.json` 可达数百 MB，使用 `ijson` 流式解析可避免内存溢出，但解析速度仍受文件大小影响
5. **仅支持 Ascend NPU**：当前仅支持 Ascend Profiler 输出格式，暂不支持 GPU（CUDA）内存分析
6. **Duration 为空**：若内存尚未释放，`Duration(us)` 为空，Parser 将 `duration_ms` 设为 `0.0`
7. **采集级别**：至少需要 level0 及以上数据，不支持 `level_none` 级数据
8. **离散模式**：需采用离散模式采集（`discrete=True`）
