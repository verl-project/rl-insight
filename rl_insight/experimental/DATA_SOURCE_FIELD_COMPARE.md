# 本分支相对上游合同多出的数据（Repeat 专用）

> 分支：`youjinlin/agent-loop-repeat-poc`  
> 范围：**只列上游三类存储合同里没有约定、但 Repeat 看板依赖的数据**  
> 原料：PR#120 `SampleRecord` 树（`uid` / `sample_index` / `session_index` / `trajectory_index` / `reward_score` / `num_turns` / finish 是否 `stop`）  
> 旁路：`run_id` 不在 PR#120 树内，由 export / 训练侧 run 维度提供  
> 布局：完整嵌套标题来自 demo Prom `*_info`（`${var:text}`）；`query_result` + 顺序安全的 value/text regex；四层嵌套 Rows Repeat；**无** `allValue`。  
> 注意：不要用「`$sample` id + Stat」那一版刺探布局——那不是早期验收的标准标题形态。

上游已有「数值摘要 → Prometheus」的意图；本分支额外要的是下面两块。**全部落在 Prometheus gauge**（Tempo / File Store 不写这些）。

---

## Gauge 怎么设（读数前先看）

Prometheus 没有「只存一组标签」的类型；本分支用 **gauge** 两种用法：

| 用法 | 值怎么设 | 信息在哪 | 本分支例子 |
|------|----------|----------|------------|
| 存在性 / 行信息 | 固定 `.set(1)` | **labels**（id、`title`） | `agent_loop_*_info` |
| 业务数值 | `.set(真实数字)` | **值** + labels | `agent_loop_traj_reward` 等 |

- `*_info` 的 `1` ≠ 业务得分，只表示「这条 series / 这一行现在存在」；节点没了就删掉这条 series，而不是写成 `0`。
- Grafana 行标题读的是 label `title`，不是这个 `1`。
- `turns` / `reward` / `success` 的值会随轨迹变化（success 用 `0.0` / `1.0`）。

---

## 数据类型：上游未定，本分支已定

上游合同只说 reward / 成功率等「数值摘要进 Prometheus」，**没有规定**用 gauge 还是 counter、值是 float 还是 int、success 怎么编码、labels 用 string 还是别的。

本分支 POC **已选定**（上游可对齐或改名，但对接本看板需兼容）：

| 项 | 本分支确定 |
|----|------------|
| Prom 类型 | 一律 **gauge** |
| `*_info` 的值 | **float，固定 `1`** |
| `title`、层级 id labels（`run_id`/`sample`/`session`/`traj`） | **string**（`sample` 等由 PR#120 的 int index `str(...)` 而来） |
| `agent_loop_traj_turns` / `reward` | **float**（随轨迹变） |
| `agent_loop_traj_success` | **float，`0.0` 或 `1.0`**（不成功 / 成功） |

---

## 字段来源与派生边界

### Tempo turn span

| 字段 | 来源 |
|------|------|
| `sample` / `session` / `traj` / `turn` | 上游 SampleRecord 索引 / `Step.step_idx`，仅转为 string |
| `finish_reason` | 上游每个 `Step.exit_reason`；末步缺失时回退到 trajectory tag；仍缺失则明确写 `unknown` |
| `tools` | 上游 `Step.tool_results[].name` 的 JSON 编码 |
| `content` | 上游 `Step.thought or Step.response`，截断到 500 字符（demo 有 `thought`） |
| `type` | 根据 `tool_results` 是否为空派生为 `tool` / `llm`，上游无同名原始字段 |
| `run_id` | export 侧生成/传入，上游 SampleRecord 没有 run 维度 |
| `state_lane_id` | export 侧由 run/sample/session/traj 拼接的可视化标识 |
| `service.name` / `monitor.trace_segment` | export 侧固定的传输/查询标识 |
| span/trace id | OpenTelemetry 自动生成 |
| span start/end/duration | Demo export 的合成时间（每 turn 默认 1 秒并锚定当前时间）；上游当前无 turn 时间戳，不能解释为真实耗时 |

`generate_data` 原本在事件里生成每 turn 的 `finish_reason`，但
`TrajectoryBuilder` 曾只保留末步原因。当前 Builder 已恢复
`finish_reason → Step.exit_reason` 的逐 turn 持久化。Timeline 直接使用
`span.finish_reason` 并仅在显示层重命名为 `State`，不再写重复的
`span.state_name`。

### 待上游反馈（可视化尚未完成的「展示数据」）

以下字段 **demo fixture 从未产出，本分支也不编造**；需要上游确认真实事件流能否提供。在确认之前，看板不展示对应列/面板：

| 缺口 | 需要上游事件带什么 | 影响的展示意图 |
|------|---------------------|----------------|
| 原始模型输出（解析/替换命令之前） | `step.assistant_msg` | Turn details 的 Response / 原始输出 |
| 完整对话历史 / 模型输入 prompt | `trajectory_begin.messages` + 每步 `assistant_msg` | Request / prompt |
| 真实 token id 序列 | `prompt_ids` / `response_ids` / `response_mask` | Token summary |

### Prometheus

| 字段/指标 | 来源 |
|-----------|------|
| 层级 id、`uid` | 上游 SampleRecord |
| `agent_loop_traj_turns` | 上游 `num_turns`；缺失时按 `steps` 数确定性计算 |
| `agent_loop_traj_reward` | 仅使用上游 `reward_score`；缺失时不发布，不再由 success 伪造 |
| `agent_loop_traj_success` | 与上游 Demo viewer 一致：`reward_score > 0` 时为 1 |
| `*_info.title` | 本分支对真实层级、turn/reward/success 的聚合展示字符串 |
| `sess_key` / `leaf` / `agent_loop_unit` | Repeat 枚举用的派生标识/存在性指标 |


---

## 可视化上各管什么

看板目标形态（标题 = `${var:text}`；全部层级均为 Rows）：

```text
Run · {run_id} · samples N · success a/b          ← ${run_id:text}  Rows Repeat
  Sample 0 · …                                     ← ${sample:text}  Rows Repeat
    Session 0 · …                                  ← ${session:text} Rows Repeat
      Trajectory Overview
      Trajectory #t · reward R · X turns           ← ${traj:text}    Rows Repeat
        sequence + details
```

| # | 数据 | 可视化层面干什么 |
|---|------|------------------|
| 1 | `agent_loop_run_info` + `title` | Run 行；Rows Repeat `$run_id` |
| 2 | `agent_loop_sample_info` + `title` | Sample 行；Rows Repeat `$sample` |
| 3 | `agent_loop_session_info` + `title` | Session 行；Rows Repeat `$session`；变量由当前 Sample section 限定 |
| 4 | `agent_loop_traj_info` + `title` | Traj 行；Rows Repeat `$traj`；变量由当前 Session section 限定 |
| 5 | `agent_loop_traj_{turns,reward,success}` | traj 级 float；亦写入 traj `title` |
| 6 | Tempo `span.content`（以及 turn/type/tools/finish_reason） | Turn details 表格；Content 可 `inspect` 展开 |

---

## 总览

| # | 本分支数据 | 存放 | 值类型 |
|---|------------|------|--------|
| 1 | `agent_loop_*_info` + label `title` | Prometheus | gauge，值固定 `1`；`title` 为 string |
| 2 | `agent_loop_traj_{turns,reward,success}` | Prometheus | gauge，值为 float |

---

## 1. `agent_loop_*_info` + `title` — 行标题与变量

**组成：** 每层（run / sample / session / traj）一条 series；值仍是 `1`，展示文案在 label **`title`**。有几条 series ≈ 这一层 Repeat 几行。

| metric | id labels | `title` ← PR#120 怎么拼 |
|--------|-----------|-------------------------|
| `agent_loop_run_info` | `run_id` | sample 数；其下 traj success 数 / 总数 |
| `agent_loop_sample_info` | `run_id`,`sample` | success / 总 turns / session 数 |
| `agent_loop_session_info` | `run_id`,`sample`,`session`,`sess_key` | `sess_key=sample=S/session=C`（跨 sample 唯一）；title=Session 聚合 |
| `agent_loop_traj_info` | `run_id`,`sample`,`session`,`traj`,`leaf`,`title` | `leaf=sample=S/session=C/traj=T`；`title`=`Trajectory #T · reward · turns` |


**success：** `reward_score > 0`，与上游 Demo viewer 保持一致。

**本分支定义**（`prom_export.py`）：

```python
g_run_info = Gauge(
    "agent_loop_run_info",
    "Agent Loop run row title",
    ("run_id", "title"),
)
g_sample_info = Gauge(
    "agent_loop_sample_info",
    "Agent Loop sample row title",
    ("run_id", "sample", "title"),
)
g_session_info = Gauge(
    "agent_loop_session_info",
    "Agent Loop session row title",
    ("run_id", "sample", "session", "sess_key", "title"),
)
g_traj_info = Gauge(
    "agent_loop_traj_info",
    "Agent Loop traj row title",
    ("run_id", "sample", "session", "traj", "leaf", "title"),
)
```

**本分支写入（拼 title + sess_key/leaf + set）：**

```python
sess_key = f"sample={sample}/session={session}"
leaf = f"sample={sample}/session={session}/traj={traj}"
traj_title = f"Trajectory #{traj} · reward {reward} · {turns} turns"
g_session_info.labels(..., sess_key=sess_key, title=session_title).set(1)
g_traj_info.labels(**labels, leaf=leaf, title=traj_title).set(1)
```

**看板怎么用**（`build_repeat_dashboard.py`）：变量使用 `includeAll`，但不设置
`allValue`。Grafana 13.1 的 section-level variables 将 Sample、Session 和
Trajectory 查询限定在各自父级 Repeat 实例内，因此变量 value 使用当前层
`sample` / `session` / `traj` 即可。`sess_key` / `leaf` 保留在 Prom 合同中，
但当前 Rows 看板不依赖它们。

---

## 2. POC 指标名：`agent_loop_traj_{turns,reward,success}`

上游有「进 Prom」的意图，**未定正式名**。本分支 POC——对应 PR#120 **traj 上已有的一级字段**，不是 run/session 聚合：

| metric | 值类型 | ← PR#120 |
|--------|--------|----------|
| `agent_loop_traj_turns` | float | `num_turns`（或 `len(steps)`） |
| `agent_loop_traj_reward` | float | `reward_score` |
| `agent_loop_traj_success` | float `0\|1` | `reward_score > 0` → `1.0` |

labels：`run_id,sample,session,traj`（均为 string）。  
`sample` ← `str(sample_index)`，`session` ← `str(session_index)`，`traj` ← `str(trajectory_index)`；`run_id` 为旁路。

**本分支定义 + 写入：**

```python
TRAJ_LABELS = ("run_id", "sample", "session", "traj")

g_turns = Gauge(
    "agent_loop_traj_turns",
    "Agent Loop traj turn count",
    TRAJ_LABELS,
)
g_reward = Gauge(
    "agent_loop_traj_reward",
    "Agent Loop traj reward",
    TRAJ_LABELS,
)
g_success = Gauge(
    "agent_loop_traj_success",
    "Agent Loop traj success (1=reward_score > 0, 0=otherwise)",
    TRAJ_LABELS,
)

g_turns.labels(**labels).set(float(turns))
g_reward.labels(**labels).set(float(reward))
g_success.labels(**labels).set(1.0 if ok else 0.0)
```

（同一段循环里还会写 `g_traj_info`，见 `prom_export.publish_sample_runs`。）

---

## 接线注意

1. 树必须是真实子树（非笛卡尔积）；节点消失时对应 label 组合应从 `/metrics` 去掉。  
2. `sample` / `session` / `traj` 在 Prom 里是 **string**，与 Tempo span attr、Grafana 变量一致。  
3. 完整案例：`rl_insight/experimental/prom_export.py`（`publish_sample_runs`）、`export_to_tempo.py`（PR#120→Prom+Tempo）、`build_repeat_dashboard.py`（variables）。
