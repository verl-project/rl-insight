# Memory 模块开发者指南

本文面向 Memory 模块（Parser + Visualizer）的开发者，介绍设计思路、内部接口约定以及扩展开发指南。用户侧快速入门见 [Memory Quickstart](../overview/memory_quickstart.md)。

## 1. 设计概述

### 1.1 背景与目标

RL-Insight 已有时序分析 Parser（`mstx` / `torch`），基于 `EventRow` 数据模型输出算子级时间线事件。Memory Parser 新增**内存分析**能力，从 Ascend Profiler 输出的内存相关文件中提取内存分配信息，为 RL 训练的内存瓶颈分析提供数据支撑。

**核心目标**：在每条内存申请记录中补全以下信息：

- 调用栈（Call Stack）—— 定位内存申请源头
- 内存申请大小（Size）—— 量化内存消耗
- 内存申请时间（Allocation Time）—— 时序关联
- 内存占用时长（Duration）—— 生命周期分析

### 1.2 数据流

```
InputData (ASCEND_MEMORY)
    │
    ├── DataChecker 校验
    │
    ▼
MemoryClusterParser
    │
    ├── allocate_prof_data()     ← 扫描目录，构建 DataMap
    │
    ├── mapper_func()            ← 多进程并行调度
    │   └── parse_analysis_data()
    │       ├── _build_call_stack_index()  ← 流式解析 trace_view.json
    │       └── _parse_operator_memory()   ← 解析 operator_memory.csv
    │           └── _match_call_stack()    ← 二分查找调用栈
    │
    ├── reducer_func()           ← 汇总排序
    │
    ▼
┌───────────────────────────────┐
│  OutputData (MEMORY_SUMMARY)  │ → pd.DataFrame
│  DataChecker 校验             │
│    ├── ParserOutputValidator  │    列结构：5 个必需列
│    └── MemoryContentRule      │    内容：数值/非空/正值
└───────────────────────────────┘
    │
    ▼
MemoryVisualizer
    │
    └── generate_memory_timeline()  → HTML + detail_data.js
```

### 1.3 类结构

```
BaseClusterParser (recipe/parser/parser.py)
  └── MemoryClusterParser (recipe/parser/memory_parser.py)
        ├── input_type = DataEnum.ASCEND_MEMORY
        ├── allocate_prof_data()          → 扫描目录，构建 DataMap
        ├── parse_analysis_data()         → 主解析流程
        ├── _build_call_stack_index()     → 流式解析 trace_view.json，构建调用栈索引
        ├── _parse_operator_memory()      → 解析 operator_memory.csv，输出 dict[str, Any]
        ├── _match_call_stack()           → name + ts 匹配调用栈
        ├── _get_data_map()               → 构建 (role, rank_id) → [path] 映射
        ├── _get_rank_path_with_role()    → 生成 DataMap 列表
        ├── _get_profiler_data_path()     → 拼接 ASCEND_PROFILER_OUTPUT 路径
        ├── _get_rank_id()                → 从 profiler_info_*.json 提取 rank_id
        ├── _get_task_role()              → 从 profiler_metadata.json 提取 role
        └── _extract_timestamp_key()      → 提取目录名中的时间戳排序键

BaseVisualizer (recipe/visualizer/visualizer.py)
  └── MemoryVisualizer (recipe/visualizer/memory_visualizer.py)
        ├── input_type = DataEnum.MEMORY_SUMMARY
        ├── run(data)                     → 委派 generate_memory_timeline
        ├── generate_memory_timeline()    → 主流程：过滤→向量化→输出
        ├── _build_chart1_data()          → 双指针滑动窗口，Chart1 hover 数据
        └── _build_memory_html()          → JSON 序列化 + 模板占位符替换
```

---

## 2. 内部接口

### 2.1 注册与数据类型

```python
@register_cluster_parser("memory")
class MemoryClusterParser(BaseClusterParser):
    input_type: DataEnum = DataEnum.ASCEND_MEMORY
```

| 属性 | 值 | 说明 |
|------|-----|------|
| 注册名 | `"memory"` | CLI `timeline.parser.type=memory` |
| `input_type` | `DataEnum.ASCEND_MEMORY` | 输入数据类型 |
| 输出类型 | `DataEnum.MEMORY_SUMMARY` | Parser 输出 / Visualizer 输入 |

`DataEnum` 新增值（定义在 `recipe/data/data_checker.py`）：

```python
class DataEnum(Enum):
    ASCEND_MEMORY = "ascend_memory"    # Memory Parser 输入
    MEMORY_SUMMARY = "memory_summary"  # Memory Parser 输出 / Visualizer 输入
```

### 2.2 MemoryEventRow

定义在 `recipe/utils/schema.py`：

```python
class MemoryEventRow(TypedDict):
    name: str                    # 算子名称
    role: str                    # RL 角色
    rank_id: int                 # 进程 rank
    call_stack: str              # 完整调用栈（";\r\n" 分隔）
    call_stack_top: str          # 调用栈顶层入口
    size_kb: float               # 内存大小（KB），正数=申请，负数=释放
    start_time_ms: float       # 内存申请时间（ms），与 BaseClusterParser.reducer_func 的排序键对齐
    duration_ms: float           # 内存占用时长（ms），0 表示未释放
    total_allocated_mb: float    # 申请时刻累计已分配内存（MB）
    total_reserved_mb: float     # 申请时刻累计预留内存（MB）
    total_active_mb: float       # 申请时刻累计活跃内存（MB）
    device_type: str             # 设备类型
```

**设计决策**：

- `size_kb`：正数表示内存申请，负数表示内存释放，与 `operator_memory.csv` 中的 `Size(KB)` 语义一致
- `duration_ms`：若 `Duration(us)` 有值则转换，无值则为 `0.0`（表示内存尚未释放）
- `call_stack_top`：取调用栈第一行（用户代码入口），便于快速定位，无需解析完整调用栈
- 时间单位统一为毫秒（ms），与现有 `EventRow` 保持一致

### 2.3 allocate_prof_data()

复用 MstxClusterParser 的目录扫描逻辑：

1. 遍历 `input_path`，找到所有 `<date>_<time>_ascend_pt` 目录
2. 从 `profiler_metadata.json` 提取 `role`，从 `profiler_info_*.json` 提取 `rank_id`
3. 按 `_extract_timestamp_key` 提取的时间戳对同 (role, rank_id) 下的目录排序
4. `profiler_data_path` 指向 `ASCEND_PROFILER_OUTPUT` 目录（注意：与 MstxClusterParser 指向 `trace_view.json` 文件不同，Memory Parser 指向目录，因为需要同时访问 `trace_view.json` 和 `operator_memory.csv`）

**关键差异**：

```python
# MstxClusterParser
def _get_profiler_data_path(self, rank_id, data_path):
    return os.path.join(data_path, Constant.ASCEND_PROFILER_OUTPUT, "trace_view.json")

# MemoryClusterParser
def _get_profiler_data_path(self, rank_id, data_path):
    return os.path.join(data_path, Constant.ASCEND_PROFILER_OUTPUT)
```

### 2.4 parse_analysis_data()

主解析流程，接收单个 Rank 的数据路径，返回 `list[dict[str, Any]]`：

```
输入: profiler_data_path (ASCEND_PROFILER_OUTPUT 目录), rank_id, role

步骤1: 构建调用栈索引
       _build_call_stack_index(trace_view_path)
       → 流式解析 trace_view.json (ijson)
       → 筛选 cat=="cpu_op" 且含 "Call stack" 的事件
       → 按 name 分组，组内按 ts 排序
       → 返回 dict[str, list[{ts, dur, call_stack}]]

步骤2: 解析 operator_memory.csv
       _parse_operator_memory(csv_path, call_stack_index, rank_id, role)
       → 逐行构建 MemoryEventRow:
          a. 时间转换: us → ms (÷ 1000)
          b. 调用栈匹配: _match_call_stack(name, allocation_time, index)
             未命中 → call_stack = "", call_stack_top = ""
          c. 提取 call_stack_top: 取调用栈第一行
          d. duration_ms: Duration(us) 有值则转换，无值则为 0

步骤3: 返回 list[dict[str, Any]]
```

### 2.5 _build_call_stack_index()

流式解析 `trace_view.json`，构建调用栈索引：

- **输入**：`trace_view.json` 文件路径
- **输出**：`dict[str, dict]`，每个值包含 `"entries"`（`list[{ts, dur, call_stack}]`，按 `ts` 升序排序）和 `"ts_list"`（预提取的 `ts` 列表，供二分查找直接使用，避免每次调用重建列表）
- **过滤条件**：仅保留 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件
- **流式解析**：使用 `ijson.items(f, "item")` 逐条读取，避免将整个 JSON 加载到内存
- **排序**：组内按 `ts` 升序排序，确保后续二分查找的正确性

### 2.6 _match_call_stack()

调用栈匹配算法，基于二分查找：

- **输入**：算子名 `name`、分配时间 `allocation_time_us`、调用栈索引
- **输出**：`(call_stack, call_stack_top)`，未命中返回 `("", "")`
- **匹配策略**：在同名算子组中，查找 `ts ≤ allocation_time` 的最近一条记录
- **匹配语义**：`ts` 是算子开始执行时间，`Allocation Time` 是算子内触发内存分配的时间，因此 `Allocation Time ≥ ts`；取 `ts` 最接近 `Allocation Time` 的一条即为触发该次内存分配的算子调用
- **允许多对一**：一个算子可能分配多次内存，因此多条 `operator_memory` 记录可以匹配到同一条 `trace_view` 记录

```python
import bisect

idx = bisect.bisect_right(ts_list, allocation_time_us) - 1
if idx < 0:
    return "", ""
```

### 2.7 DataChecker 校验

**输入侧**：`DataEnum.ASCEND_MEMORY` 当前未注册校验规则（`DataChecker.rules` 中为空列表），输入校验由 Parser 内部的文件存在性检查承担。

**输出侧**：`DataEnum.MEMORY_SUMMARY` 已注册两条校验规则：

| 规则 | 类 | 校验内容 |
|------|-----|----------|
| 列结构 | `ParserOutputValidatorRule` | DataFrame 需包含 `name`、`size_kb`、`start_time_ms`、`duration_ms`、`total_allocated_mb` |
| 内容合法性 | `MemoryContentRule` | 数值列可转 float；`name` 不含空值；`size_kb` 至少一个正值 |

常量定义在 `recipe/utils/schema.py`（`MEMORYKEYS`），规则实现在 `recipe/data/rules.py`（`MemoryContentRule`）。

如需增加校验，参考 [DataRule 扩展说明](./rule_extending_guide.md)。

### 2.8 Visualizer 注册与数据类型

```python
@register_cluster_visualizer("memory_html")
class MemoryVisualizer(BaseVisualizer):
    input_type: DataEnum = DataEnum.MEMORY_SUMMARY
```

| 属性 | 值 | 说明 |
|------|-----|------|
| 注册名 | `"memory_html"` | CLI `timeline.visualizer.type=memory_html` |
| `input_type` | `DataEnum.MEMORY_SUMMARY` | 接收 Parser 输出的 DataFrame |

渲染常量：

| 常量 | 值 | 说明 |
|------|-----|------|
| `_MAX_TIMELINE_POINTS` | 2000 | Chart1 折线最大点数，超出均匀下采样 |
| `_HOVER_TOP_N` | 10 | Chart1 hover 显示前 N 大活跃算子 |

### 2.9 generate_memory_timeline()

主流程，串联 5 个处理阶段：

```
输入: pd.DataFrame（MEMORYKEYS 5 列）
  │
  ├── 阶段1: 过滤 size_kb > 0，计算 end_time_ms、size_mb
  ├── 阶段2: 向量化构建 Chart1 事件序列
  │     └── np.concatenate([starts, ends]) → groupby("time") → cumsum()
  ├── 阶段3: Chart2 并行数组构建
  │     └── 7 列 .tolist() 预提取，CS_POOL + CS_IDX 池化去重
  ├── 阶段4: _build_chart1_data() → tl_xy, tl_active
  └── 阶段5: _build_memory_html() → 写入单文件
  │
  ▼
输出: memory_timeline_{role}_rank{id}.html 路径
```

**输入校验**（直接返回 None）：
- `data is None` 或 `data.empty`
- 过滤后全部 `size_kb <= 0` 导致 `data.empty`

### 2.10 _build_chart1_data()

静态方法，双指针滑动窗口算法，O(N+M)：

```python
active_intervals = []
interval_idx = 0
for point in memory_timeline:
    # 入队：start <= t 的 interval
    while interval_idx < n and intervals[interval_idx][0] <= t:
        active_intervals.append(...)
    # 惰性删除：end <= t 已结束
    active_intervals = [x for x in active_intervals if x[1] > t]
    # 按 size_kb 降序取 top-10 存入 tl_active
```

返回 `(tl_xy, tl_active)`，`tl_active` 每点存 `[活跃总数, [[算子名, size], ...]]`，供 Chart1 hover 使用。

### 2.11 _build_memory_html()

生成一对文件字符串：

- **detail_data.js**：13 个 JS 变量声明，使用紧凑 JSON（`separators=(",", ":")`）
- **HTML**：读取 `memory_template.html`，替换占位符：
  - `__DATA_FILE__` → `detail_data_{role}_rank{id}.js` 文件名

---

## 3. 扩展指南

### 3.1 新增内存数据校验规则

适用于：为 `ASCEND_MEMORY` 增加输入校验。

1. 在 `recipe/data/rules.py` 中继承 `ValidationRule`，实现 `check()` 和 `error_message`
2. 在 `DataChecker.rules` 中为新类型挂载规则

示例——为 `ASCEND_MEMORY` 增加 `operator_memory.csv` 存在性校验：

```python
# recipe/data/rules.py
class AscendMemoryFileExistsRule(ValidationRule):
    """校验 ASCEND_PROFILER_OUTPUT 下存在 operator_memory.csv"""

    def check(self, data) -> bool:
        if not isinstance(data, str):
            self._error_message = "Data object is not a path"
            return False
        root_path = Path(data)
        if not root_path.exists():
            self._error_message = f"Source path does not exist: {data}"
            return False
        ascend_pt_pattern = str(root_path / "*" / "*_ascend_pt")
        ascend_pt_folders = glob.glob(ascend_pt_pattern)
        for folder in ascend_pt_folders:
            csv_path = Path(folder) / "ASCEND_PROFILER_OUTPUT" / "operator_memory.csv"
            if not csv_path.exists():
                self._error_message = f"operator_memory.csv not found in {folder}"
                return False
        return True

    @property
    def error_message(self) -> str:
        return self._error_message
```

```python
# recipe/data/data_checker.py
class DataChecker:
    rules: dict[DataEnum, List[ValidationRule]] = {
        DataEnum.ASCEND_MEMORY: [PathExistsRule(), AscendMemoryFileExistsRule()],
        ...
    }
```

### 3.2 已有 Memory Visualizer

Memory Visualizer 已完成实现，代码位于 `recipe/visualizer/memory_visualizer.py`。内部接口详见 [2.8 - 2.11](#28-visualizer-注册与数据类型)。

如需新增其他形式的 Memory Visualizer（如热力图），参考 [Visualizer 扩展指南](./extending_guide.md)。

### 3.3 支持 GPU（CUDA）内存分析

适用于：将 Memory Parser 扩展至支持 CUDA 内存 Profiling 数据。

1. 在 `DataEnum` 中新增 `CUDA_MEMORY = "cuda_memory"`
2. 新增 `CudaMemoryParser`，继承 `BaseClusterParser`，实现 `allocate_prof_data()` 和 `parse_analysis_data()`
3. 使用 `@register_cluster_parser("cuda_memory")` 注册
4. 定义 `CudaMemoryEventRow`（若字段与 `MemoryEventRow` 不同）或复用 `MemoryEventRow`
5. 在 `DataChecker.rules` 中为新类型挂载校验规则
6. 更新 `main.py` 中对应 parser type 的注册名
7. 在 `docs/recipe/data/data_specification.md` 中补充数据形态说明

### 3.4 改进调用栈匹配精度

当前匹配策略基于算子名 + 时间戳二分查找，在极短时间内同一算子多次调用时可能匹配不精确。可能的改进方向：

1. **增加 `dur` 约束**：匹配时额外要求 `allocation_time ≤ ts + dur`，即分配时间必须在算子执行时间范围内
2. **增加 `tid` 约束**：若 `operator_memory.csv` 可提供线程信息，可按 `name + tid` 分组匹配
3. **最近邻匹配**：在 `ts ≤ allocation_time` 的候选中，选择 `allocation_time - ts` 最小的一条

---

## 4. 依赖

**Parser**：

| 库 | 用途 | 说明 |
|----|------|------|
| `ijson` | 流式解析大 JSON | 需安装: `pip install ijson` |
| `csv` | 解析 CSV | 标准库 |
| `bisect` | 二分查找调用栈 | 标准库 |

**Visualizer**：无额外依赖（`numpy`、`pandas`、`loguru`、`omegaconf` 均为项目公共依赖；Plotly.js 通过 HTML 模板 CDN 加载）。

## 5. 测试

测试文件：

| 测试文件 | 覆盖范围 |
|----------|----------|
| `tests/recipe/parser/test_memory_parser.py` | Parser 单元测试：注册、`_build_call_stack_index`、`_match_call_stack`、`_parse_operator_memory`、`allocate_prof_data` |
| `tests/recipe/special_e2e/test_memory_e2e.py` | E2E：完整 Pipeline（Parser + DataChecker + Visualizer），使用 `data/recipe/memory_data/` 样本数据 |