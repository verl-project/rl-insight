# 数据规格与格式说明

本文说明 RL-Insight 当前支持各种数据规格的目录布局和数据要求，便于采集与对接。

流水线在校验阶段会使用 `rl_insight.data.DataChecker` 注册的规则；通用规则见 [`rl_insight/data/rules.py`](../../rl_insight/data/rules.py)，VeRL 日志规则见 [`rl_insight/data/verl_log_rules.py`](../../rl_insight/data/verl_log_rules.py)。**具体校验项以代码为准**，部分规则可能尚未接入 `DataChecker.rules`，文档仅描述数据侧约定。

## 一、Torch Profiler 数据

### 目录结构

```text
<profile-data-path>/
└── <role>/
    └── prof_*.json.gz
```

### 文件内容要点

- 解压/解析后的 JSON 需包含 **`distributedInfo`**（如 `rank`）与 **`traceEvents`**（Chrome Trace 风格事件列表）。
- 事件中用于绘制的区间一般为 `ph: "X"`，并带有 `ts`、`dur` 等字段（时间单位以文件内约定为准）。

### 内容示例（节选）

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

## 二、MSTX（Ascend）Profiling 数据

### 目录结构

```text
<profile-data-path>/
└── <role>/
    └── *_ascend_pt/
        ├── profiler_info_*.json
        └── ASCEND_PROFILER_OUTPUT/
            └── trace_view.json
```

### trace_view.json 要点

- 为事件数组；解析侧会识别元数据事件（如 `ph: "M"`）以及 **`name` 为 `Overlap Analysis`** 的进程上下文，并消费其中的 **`ph: "X"`** 等区间事件。
- 区间事件通常带有 `ts`、`dur`（具体类型以采集导出为准）。

### 内容示例（节选）

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

## 三、生成summary_event数据 格式示例

```
<summary-event-data-path>/
└── summary_event_dataframe_sample.json
```

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

## 四、VeRL 训练日志（可选校验）

`DataEnum.VERL_LOG` 对 **单个** VeRL 训练 `.log` 文件做存在性与关键指标子串校验（例如 `DataChecker` 或 [`tests/data/check_verl_log.py`](../../tests/data/check_verl_log.py)）。路径必须是文件，不能是目录。

### 校验规则（以代码为准）

1. **存在与路径**（`VerlLogExistRule`）：扩展名为 `.log`，文件非空，且能被识别为 VeRL 日志：文件名中含 `verl`（不区分大小写），或文件开头约 64KiB 内容中含 `verl`。
2. **关键子串**（`VerlLogKeyParamsRule`）：日志正文（读取至多约 2MiB，**不区分大小写**）须**同时包含**以下子串，定义见 [`rl_insight/data/verl_log_rules.py`](../../rl_insight/data/verl_log_rules.py) 中 `DEFAULT_REQUIRED_KEYWORDS`：

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

### `data/verl_data/` 示例数据

仓库 [`data/verl_data/`](../../data/verl_data/) 下提供：

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

### 命令示例

```bash
python tests/data/check_verl_log.py data/verl_data/good_minimal_verl.log
```
