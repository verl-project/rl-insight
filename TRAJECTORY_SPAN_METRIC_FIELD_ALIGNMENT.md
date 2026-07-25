# Trajectory Span 与 Metric 字段对齐方案

## 1. 文档状态

- 状态：提议稿
- 适用范围：RL rollout trajectory 在线观测链路
- 生产端：Agent Loop `trace_op` decorator 或 `trace_span` direct reporting、trajectory metric exporter
- 适配端：Producer 自己的无状态 trajectory integration adapter
- 转发端：Monitor Client → Monitor Hub → `OpenTelemetryTraceCollector` → Tempo（透明转发）
- 消费端：Grafana Tempo State Timeline、Prometheus dashboard
- 协议版本：v1

本文在 trajectory monitoring RFC 和 Grafana 前端实验 producer 的基础上，确定 Tempo span 与 Prometheus metric 的字段职责、字段集合、命名、基数约束和兼容策略。

## 2. 核心结论

Trajectory 数据按用途分为三类：

| 数据类别 | 主要用途 | 存储位置 |
|---|---|---|
| Step 明细、展示摘要、trajectory 分组身份 | Timeline 展示、单条轨迹排障 | Tempo span |
| Reward、耗时、token 数、成功率等数值摘要 | 聚合、趋势、分位数、告警 | Prometheus metric |
| 完整消息、完整 thought/response、tool arguments/observation、原始 token/logprob/mask | 精确回放、离线分析 | Trajectory/File Store |

具体决策如下：

1. 一个 trajectory step 对应一个独立 root span，不构造 parent/child waterfall。
2. `state_lane_id` 是前端重建 trajectory lane 的唯一分组键。
3. span 显式发送每条 trajectory 的 `run_id`；`project` 和 `experiment_name` 表示包含多条 trajectory 的训练级上下文，不能替代它。
4. `finish_reason` 是 step 结束原因，`state_name` 是 Timeline 展示状态；v1 中二者使用同一个标准化值。
5. `reward` 是 trajectory 级结果，在 span 中可选，在 metric 中以数值形式统计。
6. `sample_success_count`、`sample_total_trajs`、token 长度、耗时等聚合值不复制到每个 step span。
7. `run_id`、`uid`、sample/session/traj/turn 等高基数身份不得成为 Prometheus label。

## 3. 目标与非目标

### 3.1 目标

- 为 producer integration adapter 输出和 Grafana 定义单一、稳定的 Tempo span attribute contract。
- 明确哪些数据属于 span，哪些属于 metric，避免重复存储和含义混杂。
- 支持 Grafana 按 Run → Sample → Session → Trajectory → Step 重建 Timeline。
- 控制 Tempo 数据量、Prometheus label 基数和敏感内容暴露。
- 允许后续从 receive-time 平滑迁移到 execution-time。

### 3.2 非目标

- 不在 Tempo 中保存完整 trajectory artifact。
- 不在 Prometheus 中支持按单个 sample 或 trajectory 精确查询。
- 不在 v1 引入嵌套 trace waterfall。
- 不在 v1 定义任意 `reward_extra_info` 到 metric 的动态映射。
- 不使用 span 的 receive-time duration 代替模型真实执行耗时。

## 4. 领域术语

### 4.1 `finish_reason`

`finish_reason` 表示一个 step 为什么结束，是 trajectory 的业务事实字段。

v1 使用稳定的有界枚举，例如：

```text
tool_calls
stop
length
max_step_limit
format_error
error
unknown
```

上游别名和空值由 producer integration adapter 在上报前统一归一化，例如：

```text
tool_call  -> tool_calls
max_tokens -> length
"" / None -> unknown
```

不将任意异常文本直接作为 `finish_reason`。详细异常内容应进入日志或 trajectory store。

### 4.2 `state_name`

`state_name` 是 Grafana State Timeline 的展示字段。

v1 不引入额外展示映射：

```text
span.name = state_name = canonical_finish_reason
```

因此，v1 中 `state_name` 与 `finish_reason` 值相同，但语义不同：

- `finish_reason`：精确表达 step 结束原因；
- `state_name`：提供给 Timeline 的状态名称。

未来如果 Timeline 需要更粗粒度的颜色或状态分组，可以在新协议版本中映射：

| `finish_reason` | 可能的 `state_name` |
|---|---|
| `format_error` | `error` |
| `tool_timeout` | `error` |
| `max_step_limit` | `terminated` |
| `tool_calls` | `tool` |
| `stop` | `completed` |

该映射只能由 producer integration adapter 统一实现，RL-Insight 后端和前端不得另行维护第二套业务映射。

### 4.3 `run_id`、`uid` 与索引

- `run_id`：唯一标识一条 trajectory 的随机 UUID，该 trajectory 的所有 step 共享此 ID。当前 Uni-Agent 在 Agent Loop invocation 入口创建，是因为当前一次 invocation 对应一条 trajectory。
- `uid`：数据集或业务 sample 的稳定 UID，用于展示和查询。
- `sample`：trajectory 所属的 sample index。
- `session`：sample 内的 session index。
- `traj`：session 内的 trajectory index。
- `turn`：trajectory 内的 step index。

索引字段统一序列化为字符串，以保持 OTEL/TraceQL 查询行为稳定。

## 5. Tempo Span 协议

### 5.1 Span 映射

```text
一个 Step -> 一个 OTLP root span
```

每个 span 的原生字段为：

| 字段 | 来源 | 说明 |
|---|---|---|
| `name` | 标准化后的 `finish_reason` | span 名称 |
| `start_time_ns` | 上一个时间边界或真实 step 开始时间 | 不是 attribute |
| `end_time_ns` | 当前时间边界或真实 step 结束时间 | 不是 attribute |
| `attributes` | 本节定义的字段 | 前端共享协议 |

当前事件没有真实 step 起止时间时，使用 receive-time 区间，并通过 `trajectory.timing_source` 明确标注。

### 5.2 Resource 与 Monitor Init 属性

`service.name` 放在 OTEL Resource。`project` 和 `experiment_name` 由 monitor init-level labels 统一注入每个 span attribute，不由 trajectory producer 重复构造：

| 属性 | 位置 | 建议值/来源 |
|---|---|---|
| `service.name` | OTEL Resource | 稳定服务名，例如 `rl-insight-trajectory` |
| `project` | Span attribute | monitor `init()` label |
| `experiment_name` | Span attribute | monitor `init()` label |

实验 producer 使用的 `agent-loop-poc` 不作为生产默认 `service.name`。

### 5.3 必选 Span Attributes

| 属性 | 类型 | 来源/生成方式 | 作用 |
|---|---|---|---|
| `run_id` | string | trajectory 随机 UUID | 唯一定位一条 trajectory |
| `state_lane_id` | string | 固定等于 `run_id` | Timeline lane 分组键 |
| `sample` | string | `sample_index` | sample 层级 |
| `session` | string | `session_index` | session 层级 |
| `traj` | string | `trajectory_index` | trajectory 层级 |
| `turn` | string | `step.step_idx` | step 顺序 |
| `uid` | string | sample UID | 业务身份和排障 |
| `monitor.trace_source` | string | 固定 `trajectory` | 与训练状态 span 区分 |
| `state_name` | string | canonical finish reason | Timeline 展示 |
| `finish_reason` | string | canonical finish reason | step 结束原因 |
| `type` | string | 从 step 派生 | `tool` 或 `llm` |
| `tools` | string | 工具名 JSON 数组 | 工具展示和查询 |
| `content` | string | thought/response 摘要 | Timeline 详情摘要 |
| `trajectory.timing_source` | string | 时间来源 | `receive_time` 或 `execution_time` |

`monitor.trace_segment` 不是 v1 trajectory contract 字段。现有 `trace_state` / `trace_op`
仍会分别发送 `state_interval` / `duration` 作为兼容性 metadata，但 Hub、Tempo 和当前
Grafana 均不依赖它，新 trajectory producer 不需要主动构造。

### 5.4 Lane 格式

`state_lane_id` 直接等于 `run_id`：

```text
state_lane_id = run_id
```

示例：

```text
550e8400-e29b-41d4-a716-446655440000
```

规则：

- lane 中不再包含 `uid`；`uid` 作为独立 attribute 保留。
- 不增加另一个含义相同的 `lane` 字段。
- `uid` 和 sample/session/traj 作为层级展示与查询字段，不参与 lane identity 生成。
- 即使两条 trajectory 的 `uid` 和 sample/session/traj 完全相同，也必须使用不同 `run_id`。

### 5.5 `type` 与 `tools`

v1 由 producer integration adapter 根据 step 数据派生并编码：

```python
tool_names = [str(name) for name in data.tool_names]
step_type = "tool" if tool_names else "llm"
tools = json.dumps(tool_names, ensure_ascii=False)
```

规则：

- 无工具时 `tools` 固定为 `[]`，不使用空字符串。
- 保留工具出现顺序。
- 使用 `ensure_ascii=False`，避免中文工具名被转义。
- `tools` 只保存名称，不保存 arguments 和 observation。

### 5.6 `content`

```python
content = (step.thought or step.response or "")[:500]
```

规则：

- 优先 thought，为空时使用 response。
- 最多保留 500 个字符。
- producer integration adapter 在调用 `trace_op` / `trace_span` 前应用统一的
  secret/credential redaction、字段选择和截断。
- 不保存完整消息、完整 response 或 tool observation。
- metadata-only 隐私模式可以将 `content` 置为空字符串，但字段仍保留以稳定前端 contract。

### 5.7 可选 Span Attributes

| 属性 | 发送条件 | 说明 |
|---|---|---|
| `reward` | trajectory reward 已知 | trajectory 级最终 reward，v1 按前端兼容要求编码为 string |
| `session_id` | 存在跨进程稳定、用户可搜索的业务 session ID | 不等同于 `session` index |

#### Reward 规则

当前在线链路可能在 step span 发出后才得到 reward，已写入 Tempo 的 span 无法回填。因此：

1. reward 未知时省略 `reward`，不发送空字符串。
2. 在线逐 step 上报不承诺每个 span 都有 reward。
3. 如果终止事件已经携带 reward，可在最后一个 step span 上发送。
4. 离线完整 trajectory export 可以在所有 step span 上复制同一个 trajectory reward，以兼容当前前端实验 producer。
5. 前端必须容忍 `reward` 缺失。
6. reward 的数值统计以 Prometheus metric 为准。

### 5.8 不进入 Span 的字段

| 字段 | 原因 | 应进入的位置 |
|---|---|---|
| `sample_success_count` | sample 级聚合，复制到每个 step 会重复 | Metric/Trajectory Store |
| `sample_total_trajs` | sample 级聚合 | Metric/Trajectory Store |
| `prompt_len` / `response_len` / `seq_len` | trajectory 数值摘要 | Metric |
| `num_turns` | trajectory 数值摘要，可从 spans 推导 | Metric |
| `execution_time` | 性能分布 | Metric |
| `reward_extra_info` 整体 dict | schema 和基数不稳定 | Store；allowlist 后转 Metric |
| `prompt_ids` / `response_ids` | 大数组 | Trajectory Store |
| `response_mask` / `response_logprobs` | 大数组 | Trajectory Store |
| `routed_experts` | 大数组 | Blob/Trajectory Store |
| 完整 messages/thought/response | 数据量和隐私风险 | Trajectory Store |
| tool arguments/observation 全文 | 数据量和隐私风险 | Trajectory Store |

### 5.9 Span Contract 示例

```json
{
  "name": "tool_calls",
  "start_time_ns": 1784772000000000000,
  "end_time_ns": 1784772001200000000,
  "attributes": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "state_lane_id": "550e8400-e29b-41d4-a716-446655440000",
    "sample": "3",
    "session": "2",
    "traj": "1",
    "turn": "5",
    "uid": "task-0098",
    "monitor.trace_source": "trajectory",
    "state_name": "tool_calls",
    "finish_reason": "tool_calls",
    "type": "tool",
    "tools": "[\"搜索\", \"calculator\"]",
    "content": "先检索相关资料，再核对计算结果。",
    "trajectory.timing_source": "receive_time"
  }
}
```

### 5.10 代码形式的 Span Contract

以下类型是 producer integration adapter 输出与 dashboard 共享协议的代码形式。
`encode_trajectory_span()` 是无状态的参考编码函数，用于集中展示 finish reason
归一化、lane 生成、字段编码和可选字段处理规则。它不累积 step、
不保存 trajectory 状态，不属于在线 builder 链路。RL-Insight 通用 tracing interface、
Hub 和 OpenTelemetry collector 均不解释 `TrajectorySpan`。

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, TypedDict
from uuid import UUID

try:
    # Python 3.11+
    from typing import NotRequired
except ImportError:
    # Python 3.9/3.10
    from typing_extensions import NotRequired


TRACE_SOURCE = "trajectory"
CONTENT_MAX_CHARS = 500

StepType = Literal["tool", "llm"]
TimingSource = Literal["receive_time", "execution_time"]
CanonicalFinishReason = Literal[
    "tool_calls",
    "stop",
    "length",
    "max_step_limit",
    "format_error",
    "error",
    "unknown",
]


TrajectorySpanAttributes = TypedDict(
    "TrajectorySpanAttributes",
    {
        # Identity and lane reconstruction
        "run_id": str,
        "state_lane_id": str,
        "sample": str,
        "session": str,
        "traj": str,
        "turn": str,
        "uid": str,

        # Trace protocol
        "monitor.trace_source": str,
        "state_name": str,
        "finish_reason": str,
        "type": StepType,
        "tools": str,
        "content": str,
        "trajectory.timing_source": TimingSource,

        # Optional: omit when unavailable; never encode missing reward as 0 or "".
        "reward": NotRequired[str],
        "session_id": NotRequired[str],
    },
)


class TrajectorySpan(TypedDict):
    """One trajectory step mapped to one OTLP root span."""

    name: str
    start_time_ns: int
    end_time_ns: int
    attributes: TrajectorySpanAttributes


@dataclass(frozen=True)
class TrajectoryStepSpanInput:
    """Producer adapter input for one completed trajectory step.

    `thought` and `response` must have passed the configured redactor before
    entering this interface.
    """

    run_id: str
    uid: str
    sample_index: int
    session_index: int
    trajectory_index: int
    step_index: int
    finish_reason: Optional[str]
    start_time_ns: int
    end_time_ns: int

    tool_names: Sequence[str] = ()
    thought: str = ""
    response: str = ""
    timing_source: TimingSource = "receive_time"
    reward: Optional[float] = None
    session_id: Optional[str] = None


_FINISH_REASON_ALIASES: dict[str, CanonicalFinishReason] = {
    "tool_call": "tool_calls",
    "tool_calls": "tool_calls",
    "completed": "tool_calls",
    "completed_with_tool_errors": "error",
    "finished": "stop",
    "turn_done": "stop",
    "stop": "stop",
    "max_tokens": "length",
    "token_limit": "length",
    "length": "length",
    "max_step_limit": "max_step_limit",
    "format_error": "format_error",
    "terminal_dead": "error",
    "timeout_budget_exhausted": "error",
    "unknown_error": "error",
    "error": "error",
    "unknown": "unknown",
}


def canonicalize_finish_reason(value: Optional[str]) -> CanonicalFinishReason:
    """Map upstream aliases and arbitrary/empty values to the v1 enum."""
    normalized = (value or "").strip().lower()
    return _FINISH_REASON_ALIASES.get(normalized, "unknown")


def encode_trajectory_span(data: TrajectoryStepSpanInput) -> TrajectorySpan:
    """Encode one trajectory step into a v1 Tempo span.

    Invariants:
    - `run_id` is a valid UUID string identifying exactly one trajectory.
    - all hierarchy indices are non-negative;
    - `end_time_ns` is strictly greater than `start_time_ns`;
    - reward, when present, is finite;
    - every OTLP attribute value emitted by this contract is a scalar string.
    """
    try:
        UUID(data.run_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("run_id must be a valid trajectory UUID") from exc

    indices = (
        data.sample_index,
        data.session_index,
        data.trajectory_index,
        data.step_index,
    )
    if any(index < 0 for index in indices):
        raise ValueError("trajectory hierarchy indices must be non-negative")

    if data.end_time_ns <= data.start_time_ns:
        raise ValueError("end_time_ns must be greater than start_time_ns")

    if data.reward is not None and not math.isfinite(data.reward):
        raise ValueError("reward must be finite when present")

    finish_reason = canonicalize_finish_reason(data.finish_reason)
    tool_names = [str(name) for name in data.tool_names]
    step_type: StepType = "tool" if tool_names else "llm"
    content = (data.thought or data.response or "")[:CONTENT_MAX_CHARS]
    lane = data.run_id

    attributes: TrajectorySpanAttributes = {
        "run_id": data.run_id,
        "state_lane_id": lane,
        "sample": str(data.sample_index),
        "session": str(data.session_index),
        "traj": str(data.trajectory_index),
        "turn": str(data.step_index),
        "uid": data.uid,
        "monitor.trace_source": TRACE_SOURCE,
        "state_name": finish_reason,
        "finish_reason": finish_reason,
        "type": step_type,
        "tools": json.dumps(tool_names, ensure_ascii=False),
        "content": content,
        "trajectory.timing_source": data.timing_source,
    }

    if data.reward is not None:
        attributes["reward"] = str(data.reward)
    if data.session_id:
        attributes["session_id"] = data.session_id

    return {
        "name": finish_reason,
        "start_time_ns": data.start_time_ns,
        "end_time_ns": data.end_time_ns,
        "attributes": attributes,
    }
```

调用示例：

```python
span = encode_trajectory_span(
    TrajectoryStepSpanInput(
        run_id="550e8400-e29b-41d4-a716-446655440000",
        uid="task-0098",
        sample_index=3,
        session_index=2,
        trajectory_index=1,
        step_index=5,
        finish_reason="tool_call",  # producer adapter 归一化为 tool_calls
        start_time_ns=1784772000000000000,
        end_time_ns=1784772001200000000,
        tool_names=("搜索", "calculator"),
        thought="先检索相关资料，再核对计算结果。",
    )
)

trace_span(**span)
```

`trace_span()` 是 RL-Insight 通用直接上报 interface；`trace_op()` 同时支持同步与异步
decorator，自动计时后也必须调用同一个 `trace_span()` 实现。两种调用形式都产生通用
TRACE event；MonitorHub 和 OpenTelemetry collector 原样转发，均不解释本协议。

该接口的输出必须满足：

```text
span["name"]
    == span["attributes"]["state_name"]
    == span["attributes"]["finish_reason"]
```

## 6. Prometheus Metric 协议

### 6.1 Metric 的职责

Metric 表达 trajectory、sample 和 tool 的数值统计，用于：

- count/rate；
- reward 和长度分布；
- duration 分布和分位数；
- success/error/truncation 趋势；
- 工具调用与耗时统计；
- 告警。

Metric 不承担单条 trajectory 的身份重建和文本详情展示。

### 6.2 允许的低基数 Labels

按 metric 的用途选择必要 label，不要求每个 metric 机械携带全部 label。

| Label | 约束 | 适用场景 |
|---|---|---|
| `project` | 配置值，有界 | 通用 |
| `experiment_name` | 配置值，数量受控 | 通用 |
| `framework` | 枚举，如 `verl` | 通用 |
| `task_domain` 或 `data_source` | 必须有界 | reward/成功率分析 |
| `reward_source` | 有界配置名 | reward metric |
| `final_exit_reason` | 标准化枚举 | trajectory count/duration |
| `success` | `true` / `false` / `unknown` | trajectory count |
| `tool_name` | 仅允许配置中声明的工具名 | tool metric |
| `tool_status` | `ok` / `timeout` / `syntax_error` / `skipped` / `error` | tool metric |
| `timing_source` | `execution_time` / `receive_time` | 仅在确有混合时间源时使用 |

如果某个固定 label 的值不可用，使用有界值 `unknown`，不使用异常文本或任意 metadata。

### 6.3 禁止作为 Prometheus Label 的字段

```text
run_id
trajectory_id
uid
sample
session
session_id
traj
turn
state_lane_id
tool_call_id
content
tools JSON
prompt / response / thought / action / observation
任意文件路径
任意动态 metadata key/value
```

这些字段会造成高基数或敏感数据泄露。

`run_id` 必须保留在 Tempo span，但默认不得成为 Prometheus label。如果需要 Metric → Trace 跳转，优先使用 exemplar，而不是增加 `run_id` 或 `trajectory_id` label。

### 6.4 Trajectory Metrics

下表使用逻辑 metric 名称；实际 exporter 可以统一增加 `rl_insight_` namespace。

| Metric | 类型 | 数值来源 | Labels | 缺失处理 |
|---|---|---|---|---|
| `trajectory_total` | Counter | trajectory 完成事件，每条 +1 | `project`, `experiment_name`, `framework`, `final_exit_reason`, `success` | 必须产生 |
| `trajectory_reward` | Histogram | `reward_score` | `project`, `experiment_name`, `task_domain`, `reward_source` | reward 未知时不 observe |
| `trajectory_turns` | Histogram | `num_turns` | `project`, `experiment_name`, `final_exit_reason` | 无 step 时 observe 0 |
| `trajectory_tool_calls` | Histogram | `total_tool_calls` | `project`, `experiment_name` | 无工具时 observe 0 |
| `trajectory_format_errors` | Histogram | trajectory 内 format-error 数 | `project`, `experiment_name` | 无时 observe 0 |
| `trajectory_truncated_total` | Counter | truncation 事件 | `project`, `experiment_name`, `final_exit_reason` | 未知时不增加 |
| `trajectory_preemptions` | Histogram | `num_preempted` | `project`, `experiment_name` | `-1`/未知时不 observe |

Reward 使用 Histogram，而不是无 trajectory identity 的 Gauge。Histogram 可以直接提供 count、sum、均值和分位数，且不会因新 trajectory 到达而覆盖上一条结果。

### 6.5 Token Metrics

| Metric | 类型 | 定义 |
|---|---|---|
| `trajectory_prompt_tokens` | Histogram | `prompt_len` |
| `trajectory_response_tokens` | Histogram | `response_len` |
| `trajectory_sequence_tokens` | Histogram | `seq_len` |
| `trajectory_action_tokens` | Histogram | `response_mask == 1` 的 token 数 |
| `trajectory_environment_tokens` | Histogram | response 中 `response_mask == 0` 的 token 数 |

Token metric 只发送数值计数，不发送 token ID、mask 或 logprob 数组。

建议 labels 仅使用：

```text
project, experiment_name, framework
```

如确有按 task domain 对比长度分布的需求，可以增加有界 `task_domain`，但不使用 `uid` 或 sample ID。

### 6.6 耗时 Metrics

| Metric | 类型 | 数据来源 |
|---|---|---|
| `trajectory_execution_seconds` | Histogram | trajectory 真实执行时间 |
| `trajectory_generation_seconds` | Histogram | `metrics.generate_sequences` |
| `trajectory_tool_seconds` | Histogram | `metrics.tool_calls` |
| `trajectory_reward_compute_seconds` | Histogram | `metrics.compute_score` |

Tempo span 当前的 receive-time interval 只用于 Timeline，不得作为模型生成或工具执行性能指标。

如果未来 step 携带真实 `started_at_ns` / `ended_at_ns`，可将 `trajectory.timing_source` 切换为 `execution_time`，但 metric 仍优先使用上游明确提供的执行耗时。

### 6.7 Tool Metrics

| Metric | 类型 | Labels | 说明 |
|---|---|---|---|
| `trajectory_tool_call_total` | Counter | `project`, `experiment_name`, `tool_name`, `tool_status` | 每次工具调用 +1 |
| `trajectory_tool_duration_seconds` | Histogram | `project`, `experiment_name`, `tool_name`, `tool_status` | 工具真实耗时 |
| `trajectory_tool_reward` | Histogram，后续可选 | `project`, `experiment_name`, `tool_name` | 仅在 tool reward 语义稳定后启用 |

v1 不默认启用 `trajectory_tool_reward`，原因是当前部分工具实现默认返回 `0.0`，且 tool reward 与最终 trajectory reward 的组合规则尚未统一。

### 6.8 Sample Metrics

原前端实验 producer 的：

```text
sample_success_count
sample_total_trajs
```

不再复制到每个 step span。若需要聚合 sample 行为，在 sample 完成时发送：

| Metric | 类型 | 说明 |
|---|---|---|
| `sample_total` | Counter | 每个完成的 sample +1 |
| `sample_trajectory_count` | Histogram | 每个 sample 的 trajectory 总数 |
| `sample_successful_trajectory_count` | Histogram | 每个 sample 的成功 trajectory 数 |
| `sample_success_ratio` | 查询派生 | `successful / total`，不要求单独存储 |

如果 UI 需要查看某个具体 sample 的精确 `success_count` 和 `total_trajs`，应查询 trajectory store。Prometheus 不应通过增加 sample ID label 来支持该查询。

### 6.9 `reward_extra_info`

禁止把完整字典序列化为 label 或单个 JSON metric 字段。

只允许将经过 allowlist 的稳定数值键转换为独立 metric，例如：

```text
acc
format_score
overlong_reward
```

要求：

1. 每个键预先声明名称、类型、单位和 histogram buckets。
2. 动态键只保留在 trajectory store。
3. 字符串 explanation、error message 不进入 label。
4. `reward_score` 与中间 tool/turn reward 必须使用不同 metric 名称。

## 7. 字段来源映射

| 观测字段 | 推荐来源 | 目标 |
|---|---|---|
| `run_id` | trajectory identity | Span |
| `uid` | sample identity | Span |
| sample/session/traj/turn | immutable trace context / `step_idx` | Span |
| `finish_reason` | `step.exit_reason` 标准化 | Span + 部分 Metric label |
| `type` | 是否存在 `tool_results` | Span |
| `tools` | `tool_results[].name` | Span |
| `content` | `step.thought` 或 `step.response` 摘要 | Span |
| `reward` | trajectory `reward_score` | 可选 Span + Histogram |
| `prompt_len` | token data 长度 | Histogram |
| `response_len` | response/mask 长度 | Histogram |
| `num_turns` | trajectory step 数 | Histogram |
| tool count/status/duration | `tool_results` | Counter/Histogram |
| generation/tool/reward latency | Agent Loop metrics | Histogram |
| `num_preempted` | Agent Loop metrics | Histogram，未知不发送 |

## 8. 前端消费约定

Grafana 前端可以依赖以下条件：

1. `run_id` 存在，同一 trajectory 的所有 step 相同，不同 trajectory 的值不同。
2. `state_lane_id == run_id`。
3. `sample`、`session`、`traj`、`turn` 是字符串。
4. `state_name` 与 `finish_reason` 在 v1 中相等。
5. `type` 只有 `tool` 或 `llm`。
6. `tools` 始终是合法 JSON 数组字符串。
7. `content` 字段存在，但可能为空，长度不超过 500 字符。
8. `reward` 可能缺失，前端不得将缺失等同于 `0`。
9. `trajectory.timing_source=receive_time` 时，Timeline duration 不能解释为真实模型执行耗时。
10. sample 成功数、trajectory 总数、token/reward 聚合从 metric 或 trajectory store 获取，而不是从每个 step span 读取。

## 9. 兼容与迁移

### 9.1 当前差异

| 项目 | 当前 backend 草案 | 前端实验 producer | v1 结论 |
|---|---|---|---|
| lane | `uid=.../sample=.../session=.../traj=...` | `run=.../sample=.../session=.../traj=...` | `state_lane_id = run_id` |
| `run_id` | 未显式发送 | 必选 | 必选，唯一标识 trajectory |
| `monitor.trace_source` | 有 | 未列为主要字段 | 保留 |
| `reward` | 当前在线路径没有 | 字符串，可能为空 | 可选；未知时省略 |
| `session_id` | 开放项 | 最终回复未要求 | 可选，默认不发送 |
| sample counts | 开放项 | 最终回复未要求 | 移至 Metric/Store |
| tools 编码 | 默认 JSON | `ensure_ascii=False` | 明确使用 `ensure_ascii=False` |
| `service.name` | 继承现有 exporter | `agent-loop-poc` | 使用生产稳定名称 |

### 9.2 上线顺序

1. producer、RL-Insight 和前端共同确认透明转发及本字段表。
2. 前端先兼容 `reward` 缺失和新 lane 格式。
3. producer integration adapter 为每条 trajectory 分配随机 `run_id`，设置 `state_lane_id = run_id`，统一 finish reason 和 tools 编码，并完成脱敏。
4. 增加 producer adapter contract 单元测试和端到端 Tempo smoke test。
5. trajectory 完成事件具备 reward/token/耗时后，再接入 metrics。
6. 新 metric 先在测试 namespace 验证 label cardinality，再进入正式 dashboard。

## 10. 基数、数据量与隐私约束

### 10.1 Tempo

- 身份字段允许高基数，因为用于单条 trace 查询和 lane 重建。
- `content` 最多 500 字符，并经过 redaction。
- 不保存原始 token 数组和完整 observation。
- `finish_reason`、`type` 必须是有界值。
- retention 必须由部署配置明确控制。

### 10.2 Prometheus

- 不允许 run/sample/trajectory/turn/tool-call 级身份 label。
- 动态 tool name 必须归一化到配置声明集合，未知工具使用 `unknown`。
- error message、任意 metadata 和路径不能成为 label value。
- histogram buckets 必须按真实数据范围配置，避免为每个实验动态生成 bucket。

### 10.3 Reward 缺失语义

以下三种状态必须区分：

```text
reward 缺失  -> 尚未计算或不可用
reward = 0   -> 已计算，结果为零
reward < 0   -> 已计算，结果为负值
```

因此，缺失 reward 不得编码成 `""`、`0` 或 `NaN`。

## 11. 验收标准

### 11.1 Span Contract 测试

- 每个 step 精确产生一个 root span。
- `span.name == state_name == finish_reason`。
- lane 格式与前端完全一致。
- 同一 trajectory 的所有 step 使用同一个 lane。
- 不同 run/sample/session/traj 不产生 lane 冲突。
- identity 字段均为字符串。
- 无工具时 `type=llm` 且 `tools=[]`。
- 有工具时 `type=tool`，工具名顺序稳定，中文不转义。
- `content` 优先 thought，回退 response，最大 500 字符。
- receive-time span 明确携带 `trajectory.timing_source=receive_time`。
- reward 未知时属性不存在；reward 为 `0` 时正确发送 `"0"` 或等价稳定字符串。
- span 中不存在 token arrays、完整 observation 和 sample aggregate。

### 11.2 Metric 测试

- 每条已完成 trajectory 对 `trajectory_total` 恰好增加一次。
- reward 为 `0` 时 histogram 正常 observe；reward 缺失时不 observe。
- `num_preempted=-1` 不作为真实数值发送。
- tool count/status 与 trajectory 数据一致。
- token count 与 mask 定义一致。
- 所有 label key/value 都在 allowlist 内。
- metric 中不存在 `run_id`、`uid`、sample/session/traj/turn 等高基数 label。
- receive-time interval 不被写入真实执行耗时 metric。

### 11.3 端到端验证

- Tempo span 数量等于输入 step 数量。
- Grafana 可按 project/experiment 查看多条 trajectory，并按 `run_id` 重建单条 trajectory 的完整 lane。
- 前端能展示非 ASCII 工具名和 content 摘要。
- reward 缺失时页面正常展示，不误显示为 0。
- Prometheus 可以展示 trajectory count、reward、token 和 duration 分布。
- 时间范围内 series 数量符合预设 cardinality budget。

## 12. 最终字段清单

### 12.1 Span v1

必选：

```text
run_id
state_lane_id
sample
session
traj
turn
uid
monitor.trace_source
state_name
finish_reason
type
tools
content
trajectory.timing_source
```

可选：

```text
reward
session_id
```

Resource：

```text
service.name
project
experiment_name
```

### 12.2 Metric v1

核心：

```text
trajectory_total
trajectory_reward
trajectory_execution_seconds
trajectory_generation_seconds
trajectory_tool_seconds
trajectory_reward_compute_seconds
trajectory_turns
trajectory_tool_calls
trajectory_prompt_tokens
trajectory_response_tokens
trajectory_sequence_tokens
trajectory_action_tokens
trajectory_environment_tokens
trajectory_tool_call_total
trajectory_tool_duration_seconds
```

可选扩展：

```text
trajectory_format_errors
trajectory_truncated_total
trajectory_preemptions
sample_total
sample_trajectory_count
sample_successful_trajectory_count
trajectory_tool_reward
```

该清单是 v1 producer 与 Grafana 的共享接口。新增、删除或改变字段语义时，需要 producer 与 dashboard 同步评审；不得只在一端静默改变。
