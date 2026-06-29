# 数据规格与格式说明

本文说明 RL-Insight 当前支持各种数据规格的目录布局和数据要求，便于采集与对接。

流水线在校验阶段会使用 `recipe.data.DataChecker` 注册的规则；通用规则见 [`recipe/data/rules.py`](../../../recipe/data/rules.py)，VeRL 日志规则见 [`recipe/data/verl_log_rules.py`](../../../recipe/data/verl_log_rules.py)。**具体校验项以代码为准**，部分规则可能尚未接入 `DataChecker.rules`，文档仅描述数据侧约定。

## 1. Torch Profiler 数据
### 1.1 目录结构

```text
<profile-data-path>/
└── <role>/
    └── prof_*.json.gz
```
参考：[`./data/recipe/torch_data`](../../../data/recipe/torch_data)

### 1.2 文件内容要点

- 解压/解析后的 JSON 需包含 **`distributedInfo`**（如 `rank`）与 **`traceEvents`**（Chrome Trace 风格事件列表）。
- 事件中用于绘制的区间一般为 `ph: "X"`，并带有 `ts`、`dur` 等字段（时间单位以文件内约定为准）。

### 1.3 内容示例（节选）

完整文件体积较大，此处仅保留与解析相关的关键字段示意：

```json
{
  "schemaVersion": 1,
  "distributedInfo": {
    "backend": "cpu:gloo,cuda:nccl",
    "rank": 0,
    "world_size": 2
  },
  "traceEvents": [
    {
      "ph": "X",
      "name": "cudaMemGetInfo",
      "pid": 369418,
      "tid": 1722878400,
      "ts": 4541015316353.111,
      "dur": 10083720.552,
      "args": {}
    },
    {
      "name": "process_name",
      "ph": "M",
      "pid": 369418,
      "tid": 0,
      "args": { "name": "ray::WorkerDict.actor_rollout_compute_log_prob" }
    }
  ]
}
```

## 2. MSTX（Ascend）Profiling 数据

### 2.1 目录结构

```text
<profile-data-path>/
└── <role>/
    └── *_ascend_pt/
        ├── profiler_info_*.json
        └── ASCEND_PROFILER_OUTPUT/
            └── trace_view.json
```
参考：[`./data/recipe/mstx_data`](../../../data/recipe/mstx_data)

### 2.2 trace_view.json 要点

- 为事件数组；解析侧会识别元数据事件（如 `ph: "M"`）以及 **`name` 为 `Overlap Analysis`** 的进程上下文，并消费其中的 **`ph: "X"`** 等区间事件。
- 区间事件通常带有 `ts`、`dur`（具体类型以采集导出为准）。

### 2.3 内容示例（节选）

```json
[
  {
    "name": "process_name",
    "pid": 3550586784,
    "tid": 0,
    "ph": "M",
    "args": { "name": "Overlap Analysis" }
  },
  {
    "name": "Computing",
    "pid": 3550586784,
    "tid": 2,
    "ts": "1773285899055563.748",
    "dur": 53.301,
    "ph": "X",
    "args": {}
  },
]
```

### 2.4 输入数据要求
MSTX 输入当前包含三类检查：

- `PathExistsRule`  
  检查输入对象是否为目录路径，且目录存在

- `MstxJsonFileExistsRule`  
  检查 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` 是否存在，并检查 `profiler_info_*.json` 是否存在

- `MstxJsonFieldValidRule`  
  检查相关 JSON 文件是否非空，并验证关键字段是否齐全

其中：
- `trace_view.json` 要求事件包含 `ph`、`name`、`pid`、`tid`
- `profiler_info_*.json` 要求包含 `config`、`start_info`、`end_info`、`torch_npu_version`、`cann_version`、`rank_id`

## 3. NVTX Profiling 数据
### 3.1 目录结构

```text
<profile-data-path>/
  ├── worker_process_*.*.jsonl
  └── worker_process_*.*.jsonl
```

### 3.2 worker_process_*.*.jsonl 要点

- jsonl文件包括以`color`开头的条目，条目中的`eventType`等于60，且包含`start`、`end`、`textId`字段
- jsonl文件需有包括全局时间信息的条目，该条目中有`startTime`信息
- jsonl文件需有包含RANK信息的条目，对应的，该条目的`value`中有`RANK`信息

### 3.3 内容示例（节选）

```jsonl
{"id":38,"table":"StringIds","value":"compute_log_prob"}
{"duration":21068815496,"globalVid":281474976710656,"startTime":6589107243593703,"stopTime":6589128312409199,"table":"ANALYSIS_DETAILS"}
{"color":255,"domainId":0,"end":21019655556,"eventType":60,"globalTid":282747880941798,"rangeId":1,"start":20979323,"table":"NVTX_EVENTS","textId":38}
{"name":"PROCESS_0:ENVIRONMENT_VARIABLE","table":"META_DATA_CAPTURE","value":"RANK=\"0\""}
```

## 4. 输出生成summary_event数据

### 4.1 格式示例

```
<summary-event-data-path>/
└── summary_event_dataframe_sample.json
```
参考：[`./data/recipe/summary_event_data`](../../../data/recipe/summary_event_data)

解析后汇总生成的数据文件 summary_event_dataframe_sample.json，内容必须包含"role", "name", "rank_id", "start_time_ms", "end_time_ms"字段，文件内容示例：

```
[
  {
    "name":"agent_loop_rollout_replica_0",
    "role":"agent_loop_rollout_replica_0",
    "domain":"default",
    "start_time_ms":1773285888698.7263183594,
    "end_time_ms":1773285890928.7919921875,
    "duration_ms":2230.06575,
    "rank_id":1,
    "tid":3555733409
  },
  {
    "name":"agent_loop_rollout_replica_0",
    "role":"agent_loop_rollout_replica_0",
    "domain":"default",
    "start_time_ms":1773285888698.7546386719,
    "end_time_ms":1773285890928.1730957031,
    "duration_ms":2229.4185,
    "rank_id":0,
    "tid":3555714976
  },
]
```

### 4.2 输出数据校验
输出侧校验的目标，是保证 parser 的产出能够被 visualizer 正常消费。

当前 `SUMMARY_EVENT` 类型使用 `ParserOutputValidatorRule` 进行检查，重点包括：

- 输出必须是 `pandas.DataFrame`
- DataFrame 不能为空
- 必须包含关键字段列：
  - `role`
  - `name`
  - `rank_id`
  - `start_time_ms`
  - `end_time_ms`



## 5. VeRL 训练日志（可选校验）

`DataEnum.VERL_LOG` 对 **单个** VeRL 训练 `.log` 文件做存在性与关键指标子串校验（例如 `DataChecker` 或 [`tests/recipe/data/check_verl_log.py`](../../../tests/recipe/data/check_verl_log.py)）。路径必须是文件，不能是目录。

### 5.1 校验规则（以代码为准）

1. **存在与路径**（`VerlLogExistRule`）：扩展名为 `.log`，文件非空，且能被识别为 VeRL 日志：文件名中含 `verl`（不区分大小写），或文件开头约 64KiB 内容中含 `verl`。
2. **关键子串**（`VerlLogKeyParamsRule`）：日志正文（读取至多约 2MiB，**不区分大小写**）须**同时包含**以下子串，定义见 [`recipe/data/verl_log_rules.py`](../../../recipe/data/verl_log_rules.py) 中 `DEFAULT_REQUIRED_KEYWORDS`：

   - `verl`
   - `actor/loss`
   - `critic/score/mean`
   - `critic/rewards/mean`
   - `response_length/mean`
   - `actor/grad_norm`
   - `training/global_step`
   - `training/epoch`
   - `actor/lr`
   - `actor/entropy`
   - `Training Progress:`（tqdm 类进度条前缀，完整日志中常见）

   若仅存在 `step:` 而日志未打印 `training/global_step` / `training/epoch` 字面量，将不通过。可按业务在代码中传入自定义 `required_keywords` 放宽或收紧。

### 5.2 `data/recipe/verl_data/` 示例数据

仓库 [`data/recipe/verl_data/`](../../../data/recipe/verl_data/) 下提供：

- **`good_minimal_verl.log`**：体量很小的合成日志，覆盖当前必填子串，**推荐**用于脚本/文档中的快速校验示例。
- **负面样例**（用于手工跑 `check_verl_log.py` 或自测规则；说明文字已避免误包含上述关键字）：

| 文件 | 典型失败原因 |
| --- | --- |
| `bad_exist_empty_verl.log` | 空文件 |
| `bad_exist_unbranded.log` | 无 VeRL 标识（文件名与正文均不含 `verl`） |
| `bad_keys_startup_only_verl.log` | 仅启动信息，缺指标类关键字 |
| `bad_keys_five_legacy_metrics_verl.log` | 仅有部分指标，缺全局步进/epoch/lr/entropy 等 |
| `bad_keys_no_training_step_tokens_verl.log` | 有 `step=` 但未出现 `training/global_step`、`training/epoch` 子串 |
| `bad_keys_no_entropy_verl.log` | 缺 `actor/entropy` |

`*.log` 若被根目录 `.gitignore` 忽略，需本地自备或使用 `git add -f` 将约定路径纳入版本库。

### 5.3 命令示例

```bash
python tests/recipe/data/check_verl_log.py data/recipe/verl_data/good_minimal_verl.log
```

## 5. GMM 专家负载dump数据

GMM 热力图输入类型为 `DataEnum.GMM_DATA`。**路径约定、参数与示意图**见 [`docs/recipe/overview/gmm_heatmap_quickstart.md`](../overview/gmm_heatmap_quickstart.md)。本节补充数据侧目录与文件格式说明。

### 5.1 目录结构

解析器会递归查找文件名后缀为 `group_list.pt` 的文件，且**必须**位于名为 `dump_tensor_data` 的目录下；路径中需能匹配 `step_{整数}` 与 `rank{整数}`，并与正式采集层级一致（与常见训练 dump 目录相同，例如本地完整数据可放在 `gmm_dump/` 等任意根目录名之下，只要子目录层级符合下述约定即可。

```text
<gmm-root>/
├── step_1/                                # 训练步骤
│   ├── actor_compute_log_prob/            # 角色/阶段（文件夹名即 role）
│   │   └── rank0/                         # Rank ID
│   │       └── dump_tensor_data/          # 张量 dump 目录（必填层级名）
│   │           ├── NPU.npu_grouped_matmul.0.forward.kwargs.group_list.pt
│   │           ├── NPU.npu_grouped_matmul.1.forward.kwargs.group_list.pt
│   │           └── ...
│   └── actor_update/
│       └── rank0/
│           └── dump_tensor_data/
│               ├── NPU.npu_grouped_matmul.0.forward.kwargs.group_list.pt
│               └── ...
├── step_2/
└── ...
```

参考（仓库内**最小可解析**示例，体量极小，便于测试与文档对照）：[`../../../data/recipe/gmm_data`](../../../data/recipe/gmm_data)

### 5.2 `group_list.pt` 文件内容

- 文件为 PyTorch `torch.save` 序列化对象；解析侧优先以 `weights_only=True` 加载，需为 **`torch.Tensor` 或 `numpy.ndarray`**，语义为一维 **expert 负载**：`reshape(-1)` 后第 `i` 个元素对应 `expert_index == i` 分到该 expert 的 **token 数（非负整数）**。
- 文件名需能被解析器识别为 GMM 算子序号，典型 pattern 为  
  `NPU.npu_grouped_matmul.<op_index>.forward.kwargs.group_list.pt`  
  其中 `<op_index>` 会映射为汇总表中的 `stage`（层/算子序号）。

### 5.3 输入数据校验

`GmmDataRule`（见 [`recipe/data/rules.py`](../../../recipe/data/rules.py)）要求：

- 输入为已存在的目录路径；
- 其下至少存在一个位于 `dump_tensor_data` 路径段中的 `*group_list.pt` 文件。

## 6. Ascend Memory Profiling 数据

Memory 模块输入类型为 `DataEnum.ASCEND_MEMORY`（CLI：`timeline.parser.type=memory`）。用户侧快速入门见 [Memory Quickstart](../overview/memory_quickstart.md)，开发者指南见 [Memory Guide](../developer_guides/memory_guide.md)。

### 6.1 目录结构

```text
<profile-data-path>/
└── <role>/
    └── <date>_<time>_ascend_pt/
        ├── profiler_info_<rank_id>.json
        ├── profiler_metadata.json
        └── ASCEND_PROFILER_OUTPUT/
            ├── operator_memory.csv
            └── trace_view.json
```

参考：[`./data/recipe/memory_data`](../../../data/recipe/memory_data)

### 6.2 operator_memory.csv 要点

| 字段 | 类型 | 说明 |
|------|------|------|
| `Name` | str | 算子名称 |
| `Size(KB)` | float | 正数=申请，负数=释放 |
| `Allocation Time(us)` | float | 申请/释放时间戳（微秒），末尾可能含制表符 |
| `Duration(us)` | float | 占用时长（可能为空，表示未释放） |
| `Allocation Total Allocated(MB)` | float | 申请时刻累计已分配 |
| `Allocation Total Reserved(MB)` | float | 申请时刻累计预留 |
| `Allocation Total Active(MB)` | float | 申请时刻累计活跃 |
| `Device Type` | str | 设备类型（如 `NPU:0`） |

### 6.3 trace_view.json 要点（Memory Parser 视角）

Memory Parser 仅消费 `trace_view.json` 中 `cat=="cpu_op"` 且 `args` 中含 `"Call stack"` 的事件，用于调用栈关联。文件可能较大（数百 MB），Parser 使用 `ijson` 流式解析。

关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cat` | str | 事件类别，Memory Parser 仅消费 `"cpu_op"` |
| `name` | str | 算子名称，用于与 `operator_memory.csv` 的 `Name` 匹配 |
| `ts` | str | 开始时间戳（微秒），JSON 中为字符串类型 |
| `dur` | float | 持续时间（微秒） |
| `args.Call stack` | str | Python 调用栈，以 `";\r\n"` 分隔 |

### 6.4 输出 memory_summary 数据

Memory 模块输出类型为 `DataEnum.MEMORY_SUMMARY`，为 `pd.DataFrame`，每行对应一条内存分配/释放记录。完整字段说明见 [Memory Guide](../developer_guides/memory_guide.md) 中 `MemoryEventRow` 定义。

**DataChecker 校验要求**（`MEMORY_SUMMARY` 已注册规则）：

| 规则 | 校验内容 |
|------|----------|
| `ParserOutputValidatorRule` | DataFrame 必须包含 `name`、`size_kb`、`start_time_ms`、`duration_ms`、`total_allocated_mb` |
| `MemoryContentRule` | 数值列可转 float；`name` 不含空值；`size_kb` 至少一个正值 |

**DataFrame 完整字段**：

- `name`、`role`、`rank_id`
- `call_stack`、`call_stack_top`
- `size_kb`、`start_time_ms`、`duration_ms`
- `total_allocated_mb`、`total_reserved_mb`、`total_active_mb`
- `device_type`

### 6.5 输入数据校验

`DataEnum.ASCEND_MEMORY` 当前未注册校验规则（`DataChecker.rules` 中为空列表），输入校验由 `MemoryClusterParser.parse_analysis_data()` 内部的文件存在性检查承担。如需增加校验规则，参考 [DataRule 扩展说明](../developer_guides/rule_extending_guide.md)。
