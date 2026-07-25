# RL-Insight `dev`：Trace Interface → Tempo 最终实施计划

## 1. 状态与范围

- 状态：方案已整理，待评审实施
- 代码基线：`rl-insight/dev@9e77cd3672535f35e2071fb95d4de51bb74d7966`
- 实施分支：`feature/trajectory-span-tempo`
- 上游协议文档：`TRAJECTORY_SPAN_METRIC_FIELD_ALIGNMENT.md`
- 目标：提供通用同步/异步 decorator 和直接 span 上报 interface，并复用现有 Tempo 链路
- 不包含：Uni-Agent 接入、Grafana dashboard、Prometheus trajectory metric

如果实施时 `dev` 已不再指向上述提交，必须先重新核对 `api.py`、Hub、OpenTelemetry
collector 和测试差异并更新本计划，不能直接套用旧假设。

## 2. 已确认决策

1. 保留已有公开 `trace_op()` 和 `trace_state()`。
2. 新增公开 `trace_span()`，接收完整 name、起止时间和 attributes。
3. `trace_op()` 同时支持 `def` 与 `async def`。
4. 同步/异步 wrapper 共用同一套内部 invocation implementation。
5. `trace_op()` 最终调用 `trace_span()`，不维护第二套 TRACE event 发送逻辑。
6. 不新增 `trace_trajectory()`、`TempoSampleRecord`、builder 或 converter。
7. Hub、OpenTelemetry collector、RL-Insight Server 和 Tempo 配置保持不变。
8. RL-Insight 不校验、不补齐、不解释 trajectory 数据协议字段。
9. Producer integration adapter 准备最终 name/time/attributes；后端透明转发。
10. 不新增 `segment` 参数。
11. 现有 `trace_op` 继续写入 `monitor.trace_segment=duration`；现有 `trace_state`
    继续写入 `monitor.trace_segment=state_interval`，仅为兼容，不纳入新 trajectory contract。
12. 旧 `rl_insight/experimental/` trajectory builder/file/HTML 原型全部删除。

## 3. 当前 `dev` 现状

### 3.1 已有链路

```text
trace_state / trace_op
  -> rl_insight.api._emit_trace_span
  -> MonitorRayClient.apply_event
  -> MonitorHubActor.apply_event
  -> MonitorHubActor._handle_trace
  -> OpenTelemetryTraceCollector.record_span
  -> OTLP/HTTP /v1/traces
  -> Tempo
```

链路已经支持：

- Ray client fire-and-forget；
- TRACE event 原样发送；
- 显式 `start_time_ns` / `end_time_ns`；
- root span 和 arbitrary attributes；
- Tempo OTLP/HTTP receiver；
- Tempo 查询 smoke test。

当前缺口：

- `_emit_trace_span()` 是私有 implementation，没有公开直接上报 interface；
- `trace_op()` 只支持同步函数；
- async function 会告警并原样返回，不产生 span；
- `trace_op()` 不能从成功返回值生成最终 attributes 或 span name；
- E2E 只覆盖 `trace_state()`，没有覆盖 direct interface。

### 3.2 既有 trace interface 的数据处理

| Interface | TRACE event 生成前 | Event 发送后 |
|---|---|---|
| `trace_state` | lane 默认值、保留字段、时间、overlap 合并/丢弃 | Hub→Tempo 透明转发 |
| `trace_op` | name、static/extra labels、duration、时间 | Hub→Tempo 透明转发 |
| `trace_span`（目标） | 复制 name/time/attributes，合入 init attributes | Hub→Tempo 透明转发 |

`trace_state()` 现有语义：

- `state_lane_id` 缺失时使用当前进程 ID；
- 强制写入 `state_name`、`state_lane_id`、`monitor.trace_segment=state_interval`；
- 同 lane、同 state 的重叠调用合并为一个 span；
- 同 lane 被其他 state 占用时，新 state 被 shadow，不产生 span；
- 一个 context 调用不保证对应一个 TRACE event。

`trace_op()` 现有语义：

- name 默认 `func.__qualname__`；
- `extra_labels(first_arg)` 覆盖同名 static label；
- 强制写入 `monitor.trace_segment=duration`；
- 正常返回或抛异常都会尝试发送一个 span；
- 不做 overlap 合并；
- async function 当前不包装。

`_emit_trace_span()` 现有语义：

- 合入 `process_id`、`project`、`experiment_name`；
- init attributes 在前，调用方 attributes 在后，同名调用方值会覆盖 init 值；
- 将 start/end 转成整数；
- 构造 `kind=TRACE` event 并调用 client。

Hub 只执行 `dict(attributes)` 和 `int(start/end)`；OpenTelemetry collector 只按显式时间
创建和结束 root span。两者都没有协议转换、字段派生或合并。

透明转发不变量从 `_emit_trace_span()` 生成 TRACE event 后开始：

```text
TRACE event 中的 name/time/attributes == Tempo 中的 name/time/attributes
```

Trajectory 要求“一 step 一 span”，因此不能使用带 lane 合并语义的 `trace_state()`；
应使用新增的 `trace_span()` 或增强后的 `trace_op()`。

## 4. 目标公开 interface

### 4.1 `trace_span`：直接上报

```python
def trace_span(
    *,
    name: str,
    start_time_ns: int,
    end_time_ns: int,
    attributes: Mapping[str, Any],
) -> None:
    """Report one completed span through the existing TRACE event path."""
```

Interface 语义：

- `name` 是最终 span name，后端不覆盖；
- 时间是 Unix epoch nanoseconds，调用方负责真实性；
- 不重新计时；
- 不验证 `end_time_ns > start_time_ns` 等业务规则；
- 不解析 `StepOutput`；
- 不归一化 trajectory 字段；
- 复制 attributes，调用方后续修改原 mapping 不影响已提交 event；
- monitoring disabled 时立即返回；
- 同步 fire-and-forget，不返回 Ray object ref，不需要 `await`；
- client submission 保持现有 `_emit_trace_span()` 失败语义，本 PR 不新增重试或吞错。

实现收敛到现有 private implementation：

```python
def trace_span(...):
    _emit_trace_span(
        name=name,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        attributes=dict(attributes),
    )
```

`trace_span()` 不自动添加 `monitor.trace_segment`。

### 4.2 `trace_op`：同步/异步 decorator

```python
def trace_op(
    name: str | None = None,
    *,
    extra_labels: Callable[[Any], Mapping[str, Any]] | None = None,
    **static_labels: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...
```

| 参数 | 数据来源 | 时机 |
|---|---|---|
| `name` | decorator 配置 | 默认 `func.__qualname__` |
| `static_labels` | decorator 配置 | 每次调用固定 attributes |
| `extra_labels` | 第一个位置参数 | 函数执行前 |

兼容规则：

- 保留 `name`、`extra_labels`、`**static_labels` 现有用法；
- 不新增 `segment` 参数；
- 继续固定写入 `monitor.trace_segment=duration`；
- callback 异常属于观测异常，不能替换业务返回值、业务异常或 async cancellation。
- 不新增基于返回值派生 attributes/name 的 callback；输出派生字段由调用方自行计算后走
  `trace_span()` 上报，避免在 decorator 上叠加尚无消费者的接口面。

字段覆盖顺序：

```text
static_labels
  -> extra_labels
  -> monitor.trace_segment=duration
```

name 覆盖顺序：

```text
explicit name / func.__qualname__
```

Callback fallback：

- `extra_labels` 失败：warning，继续执行业务函数，保留 static labels；
- 不捕获业务函数的 `BaseException`，只通过 `finally` 尝试关闭 span。

### 4.3 共享同步/异步 implementation

公开 interface 只有一个 `trace_op`。根据 `inspect.iscoroutinefunction(func)` 选择 wrapper：

```python
if inspect.iscoroutinefunction(func):

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        with _trace_op_invocation(...):
            return await func(*args, **kwargs)

    return async_wrapper

@functools.wraps(func)
def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
    with _trace_op_invocation(...):
        return func(*args, **kwargs)

return sync_wrapper
```

`_trace_op_invocation` 是 private implementation，统一负责：

- disabled 快速路径；
- pre-call attributes；
- start time；
- end time；
- 调用 `trace_span()`；
- extra_labels warning/fallback。

同步与异步 wrapper 只保留 `func(...)` / `await func(...)` 的必要差异，不复制 event
构造和发送逻辑。不新增公开 `async_trace_op`。

### 4.4 `trace_state`

- 保持公开导出；
- 保持 lane、overlap、shadow 和时间语义；
- 继续直接调用 `_emit_trace_span()`；
- 不强制迁移到 `trace_span()`，避免扩大行为变化。

## 5. Attribute 类型与数据协议

### 5.1 AttributeValue

公开注解暂时保持 `Mapping[str, Any]`，兼容已有 labels；实际值必须符合 OpenTelemetry：

- `str`、`bool`、`int`、`float`；
- 上述同类标量组成的 sequence。

不支持：

- nested mapping；
- Pydantic model；
- 任意业务对象；
- 混合类型 sequence。

RL-Insight 不自动执行 `str(value)`、JSON 编码或类型修复。复杂字段由调用方上报前编码，
例如 tools 使用 JSON string。

### 5.2 Trajectory Span contract

Trajectory Span contract 是 producer 与 dashboard 的独立协议，不是 RL-Insight tracing
interface 的实现依赖。

Producer integration adapter 负责：

- 每条 trajectory 创建随机 UUID `run_id`；
- 同一 trajectory 内复用 `run_id`；
- 记录真实 step start/end；
- 标准化 `finish_reason`；
- 构造 `state_name`、`state_lane_id`、tools、type 和其他 attributes；
- 在上报前完成敏感信息脱敏和 content 截断；
- 输出最终 contract-ready name/time/attributes。

RL-Insight 不负责：

- 生成或验证 `run_id`；
- 识别普通 span 与 trajectory span；
- 标准化 finish reason；
- 生成 lane；
- 转换 tools/content/reward；
- 校验完整 contract。

`monitor.trace_segment` 不属于新 trajectory contract。现有 `trace_state` / `trace_op`
仍发送该兼容性 metadata，但新 producer 无需主动构造。

Grafana 只处理颜色、列名、排序、单位和缺失值等展示逻辑，不重新定义 finish reason、
lane identity 或字段类型。

## 6. 文件修改清单

### 6.1 修改（M）

#### `rl_insight/api.py`

- `__all__` 增加 `trace_span`；
- 新增公开 `trace_span()`；
- 支持 sync/async wrapper；
- 删除 async no-op warning 分支；
- 新增 private 共享 invocation implementation；
- `trace_op()` 最终调用 `trace_span()`；
- `trace_state()` 行为保持不变；
- `_emit_trace_span()` event shape 和 merge 顺序保持不变。

#### `rl_insight/__init__.py`

只新增 lazy export：

```diff
 "trace_op": ".api",
+"trace_span": ".api",
 "trace_state": ".api",
```

`trace_state` 是既有内容，不是本次新增。

#### `tests/monitor/ut/test_api.py`

保留既有测试并增加第 7 节测试矩阵。

#### `tests/monitor/special_e2e/test_monitor_smoke.py`

- 增加 direct interface Tempo smoke；
- 增加 contract-ready trajectory span 透明转发验证；
- 增加两个同名 span 不合并验证；
- 保留既有 metric 和 `trace_state` smoke。

#### `README.md`

- Training API 表增加 `trace_span`；
- `trace_op` 描述改为支持 sync/async；
- 增加 direct reporting 示例；
- 说明 epoch nanoseconds、AttributeValue 和透明转发；
- 不加入 Uni-Agent、Grafana 或 trajectory 字段转换实现。

### 6.2 新增（A）

```text
A  TRAJECTORY_SPAN_METRIC_FIELD_ALIGNMENT.md
A  TRAJECTORY_SPAN_TO_TEMPO_DEV_PLAN.md
```

### 6.3 删除（D）

旧 experimental prototype 无仓库内外部引用，删除整个原型：

```text
D  rl_insight/experimental/README.md
D  rl_insight/experimental/__init__.py
D  rl_insight/experimental/builder.py
D  rl_insight/experimental/generate_data.py
D  rl_insight/experimental/server.py
D  rl_insight/experimental/samples/__init__.py
D  rl_insight/experimental/samples/base.py
D  rl_insight/experimental/samples/file_sample.py
D  rl_insight/experimental/samples/sample.py
```

不保留空 builder interface、`BaseSample`、HTML server、converter 或空 package。

### 6.4 明确不修改（UNCHANGED）

```text
rl_insight/client/ray_monitor_client.py
rl_insight/collector/ray_monitor_hub.py
rl_insight/utils/opentelemetry_utils.py
rl_insight/config/services/tempo/tempo.yaml
rl_insight/server/**
rl_insight/config/services/grafana/**
tests/monitor/ut/client/test_ray_monitor_client.py
tests/monitor/ut/collector/test_ray_monitor_hub.py
tests/monitor/ut/utils/test_opentelemetry_utils.py
```

如果实施 diff 超出以上范围，必须先说明原因并更新本计划。

## 7. 单元测试矩阵

`tests/monitor/ut/test_api.py` 增加：

| ID | 场景 | 预期 |
|---|---|---|
| U1 | direct report | 生成完整 TRACE event |
| U2 | explicit time | start/end 原样保留 |
| U3 | init attributes | 合入 process/project/experiment |
| U4 | attribute copy | 原 mapping 后续修改不影响 event |
| U5 | sync decorator | 返回值与既有行为不变 |
| U6 | async decorator | await 前开始、完成后结束 |
| U7 | coroutine identity | 包装后仍是 coroutine function |
| U8 | sync exception | 原异常传播并尝试结束 span |
| U9 | async exception/cancel | 原异常或 cancellation 传播 |
| U10 | extra_labels failure | warning + fallback，保留 static labels，不替换业务结果 |
| U11 | disabled | 不计时、不调用 extra_labels、不发送 event |
| U12 | direct/decorator parity | 最终 TRACE event schema 相同 |
| U13 | trace_state regression | lane merge/shadow 行为不变 |

不在本文件校验完整 Trajectory Span contract；该测试属于后续 producer integration adapter。

## 8. Step-by-step 执行与验证

### Step 0：准备分支

```bash
git switch dev
git pull --ff-only
git switch -c feature/trajectory-span-tempo
```

验证：

```bash
git merge-base dev feature/trajectory-span-tempo
git log --oneline dev..feature/trajectory-span-tempo
```

预期：merge base 是已确认的最新 `dev`，新分支没有功能提交。

旧 `feature/trace-trajectory-tempo` PR 标记 superseded；新 PR 后续引用旧 PR，不整体
cherry-pick `b27e97f`，不 force-push 重写旧 PR。

### Step 1：添加 API 单元测试（Red）

修改 `tests/monitor/ut/test_api.py`，加入 U1-U13。

```bash
pytest tests/monitor/ut/test_api.py
```

预期：新增 direct/async 用例失败；既有测试通过。

### Step 2：实现 tracing interface（Green）

修改：

```text
rl_insight/api.py
rl_insight/__init__.py
```

```bash
pytest tests/monitor/ut/test_api.py
```

预期：U1-U13 与既有 API 测试全部通过。

### Step 3：更新文档

更新 README 和两份设计文档，明确 interface、AttributeValue、producer contract 和透明转发。

```bash
pytest tests/doc/test_docs_urls.py
```

### Step 4：删除旧 experimental 原型

删除第 6.3 节文件。

```bash
rg "rl_insight\.experimental|TrajectoryBuilder|FileSampleRecord" rl_insight tests README.md docs
```

预期：退出码为 1；代码、测试和公开用户文档中无旧原型引用。

### Step 5：运行 monitor 单元测试回归

```bash
pytest tests/monitor/ut
```

验证 Ray client、Hub、OpenTelemetry、metric 和既有 trace 行为无回归。

### Step 6：扩展并运行 Tempo E2E

修改 `tests/monitor/special_e2e/test_monitor_smoke.py`：

- 每次生成唯一 `test_run`；
- 上报一个普通 direct span；
- 上报 contract-ready trajectory span；
- 连续上报两个同名 span；
- 轮询 Tempo 直到至少两个目标 span或超时；
- 不依赖结果顺序；
- 超时输出最后一次响应。

远程 Linux 管理栈运行：

```bash
pytest tests/monitor/special_e2e/test_monitor_smoke.py
```

验证：

- direct span 可查询；
- contract-ready name/time/attributes 原样保存；
- 两个同名 event 保存为两个独立 span；
- 既有 `trace_state` 和 metric smoke 继续通过。

### Step 7：核对 diff allowlist

```bash
git diff --name-status dev...HEAD
```

预期只出现第 6 节列出的 M/A/D 文件。若超出，先更新计划并说明原因。

### Step 8：提交前检查

在项目规定的远程开发环境执行：

```bash
pre-commit run --all-files
pytest
```

不在本地启动 NPU/完整部署环境；未经指令不 commit/push。

## 9. PR 组织建议

保持一个 PR，按以下提交顺序方便 review：

1. API unit tests；
2. `trace_span` + sync/async `trace_op` implementation；
3. Tempo smoke test；
4. 删除 experimental prototype；
5. README 和设计文档。

建议 PR 标题：

```text
[monitor-api] feat: add direct and async trace span reporting
```

## 10. 风险与控制

| 风险 | 控制 |
|---|---|
| async wrapper 在 coroutine 创建时结束 | 时间记录放在 async wrapper 的 await 生命周期内 |
| callback 异常影响业务 | callback warning/fallback，业务结果和异常优先 |
| submission 失败语义不一致 | 本 PR 保持现状；统一 best-effort 另立改造 |
| 任意对象进入 OTEL attributes | README 限定 AttributeValue，不做隐式转换 |
| trajectory 规则渗入 RL-Insight | interface 测试只验证通用 name/time/attributes |
| Hub 演变成协议处理中心 | 禁止按 trace source 分支或改写字段 |
| producer 字段不一致 | producer adapter contract tests 上报前发现 |
| Tempo E2E 最终一致性 | 唯一 test_run、轮询、无序断言、保留最后响应 |
| 同名 span 被错误合并 | E2E 连续发送同名 span并验证数量 |
| 删除 experimental 影响外部试用者 | PR 明确 breaking cleanup；该 package 仍为 experimental |

## 11. Definition of Done

- 新分支基于已确认的 `dev`。
- `trace_span` 可公开导入并直接上报。
- `trace_op` 支持 sync/async，包装后 coroutine identity 保持。
- sync/async wrapper 共用 invocation implementation。
- extra_labels callback 失败 warning/fallback 有测试。
- disabled、异常、取消、attribute copy 有测试。
- `trace_state` merge/shadow 行为无回归。
- Hub、OpenTelemetry、Server、Tempo 配置没有修改。
- 没有 converter、builder 或第二套发送链路。
- `monitor.trace_segment` 未新增参数，也不属于新 trajectory contract。
- 旧 experimental prototype 已删除且无孤立引用。
- monitor 单元测试、文档测试、Tempo E2E、pre-commit 和全量 pytest 通过。
- 最终 diff 与第 6 节 allowlist 一致。
- 未经用户指令没有 commit 或 push。
