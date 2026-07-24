# experimental — 实验轨迹数据流水线（Demo）

> **当前阶段：原型演示。** 完整链路是「造数 → Builder → SampleRecord → TempoSpanMapper → Tempo → Rebuild → Grafana」。中间的事件协议与 Builder 层保持稳定；展示端是 Grafana Agent Loop（无 HTML Timeline）。

---

## 模块化设计

按流水线拆成两块，职责分开、依赖单向：

| 块 | 目录 / 文件 | 管什么 |
|----|-------------|--------|
| **源数据（#120）** | `samples/` + `builder.py` + `generate_data.py` | 事件 → `SampleRecord` 树；完整字段在这里 |
| **可视化（本 PR）** | `agent_loop/` + `export_to_tempo.py` | 按协议子集进 Tempo，再 Rebuild 成 Grafana 嵌套面板 |

**设计要点：**

1. **模块化**：写 Tempo / 读 Tempo / 建树 / 写 Grafana / 编排各占一个模块，改一层不动另一层。
2. **面向对象**：每阶段一个类（对齐 `TrajectoryBuilder`）——构造注入、公开方法少。
3. **两套模型**：写路径用 `SampleRecord`；读路径只用 `TempoSpan`（Tempo 里的 attributes）。viz **不** import SampleRecord。
4. **薄门面**：CLI / HTTP 调 `export_samples_to_tempo`、`rebuild_from_tempo` 等函数即可，不必绑死类名。

### 模块怎么读

按子包拆开：

| 图上区域 | 对应目录 | 在干什么 |
|----------|----------|----------|
| **Source (#120)** | `samples/` + `builder` / `generate_data` | 造完整 `SampleRecord`；全文/token 等留在这里 |
| **write/** | `agent_loop/write/` | 写路径：`TempoSpanMapper` 按显式 attribute 子集打 OTLP，带 `run_id` 进 Tempo |
| **Tempo** | （外部） | span 存储 / 检索 |
| **read/** | `agent_loop/read/` | 读路径零件：`TempoClient` 拉 span，`RunHierarchyBuilder` 建成 Run 树 |
| **rebuild/** | `agent_loop/rebuild/` | 编排：Client → Hierarchy（含时间窗过滤）→ DashboardWriter |
| **dashboard/** | `agent_loop/dashboard/` | 把树写成 Grafana runtime JSON（Turn details **无** Reward） |

入口也画在边上：`export_to_tempo.py` → write；`http_api` Rebuild → rebuild。  
底注：**只有 write/mapper 碰 SampleRecord**；可视化链路只用 `TempoSpan`。

```
数据生成                 数据处理                      可视化
────────────────────────────────────────────────────────────────
模拟脚本 ──┐             ┌─ SampleRecord (内存) ─┐
           ├── Builder ──┤                       ├── write/TempoSpanMapper
uni-agent ─┘     ▲       └─ FileSampleRecord ───┘         │
                 │                                        ▼
           两种事件格式（稳定）                          Tempo
                                                          │
                                                          ▼
                         rebuild/（complete → filter_to_window → dashboard）
                                                          │
                                                          ▼
                                      Grafana agent_loop_trajectory
                                      （点 Rebuild；Turn details 无 Reward）
```

---

## experimental 目录怎么分

```
experimental/
  ├── README.md                 # 本文档
  ├── __init__.py               # 包入口（懒导出 Builder / Sample 类型）
  ├── builder.py                # TrajectoryBuilder：事件 → BaseSample
  ├── generate_data.py          # 模拟造数 CLI / 库函数 generate()
  ├── export_to_tempo.py        # CLI：造数 + Mapper → Tempo（不 Rebuild）
  ├── samples/                  # 源数据层
  └── agent_loop/               # 可视化 OO 包
```

### 根目录文件

| 文件 | 功能 |
|------|------|
| `__init__.py` | 对外懒导出 `TrajectoryBuilder`、`SampleRecord` 等，避免 server 侧误拉重依赖 |
| `builder.py` | **数据处理核心**：吃 `trajectory_begin` / `step` 两种事件，经 `BaseSample` 建树 |
| `generate_data.py` | **造数**：模拟 agent 推理事件；可落盘或内存；也可被 `export_to_tempo` 调用 |
| `export_to_tempo.py` | **导出入口**：`generate` → `TempoSpanMapper.export`；看盘需再点 Grafana Rebuild |
| `README.md` | 架构、模块分配、用法（本文档） |

### `samples/` — 源数据

| 文件 | 功能 |
|------|------|
| `base.py` | `BaseSample` Protocol（六个 CRUD 方法） |
| `sample.py` | `SampleRecord` 内存实现（Pydantic：Sample→Session→Traj→Step） |
| `file_sample.py` | `FileSampleRecord` 文件实现（每 traj 一个 JSON） |
| `__init__.py` | 导出上述类型 |

完整字段（含 token、tool action/observation 等）留在这里；**不**全部进 Tempo。

### `agent_loop/` — 可视化

| 路径 | 类 / 角色 | 功能 |
|------|-----------|------|
| `constants.py` | — | `service.name`、Grafana uid/slug、默认 URL（包根共享） |
| `__init__.py` | — | 包对外 API（`export_samples_to_tempo` / `rebuild_from_tempo` 等） |
| `write/` | WRITE | SampleRecord → Tempo |
| `write/mapper.py` | `TempoSpanMapper` | SampleRecord → OTLP span（写 Tempo；显式 attribute 子集） |
| `read/` | READ | Tempo → hierarchy |
| `read/client.py` | `TempoClient` / `TempoSpan` | HTTP 读 Tempo → span 列表 |
| `read/hierarchy.py` | `RunHierarchyBuilder` | 分组 / 补全 / **按时间窗过滤** / 建 Run 树 |
| `dashboard/` | Grafana | 树 → dashboard JSON |
| `dashboard/writer.py` | `AgentLoopDashboardWriter` | 树 → Grafana 嵌套 JSON（读同目录 `panel_templates.json`） |
| `dashboard/panel_templates.json` | — | overview / sequence / details 面板模板（details **无** Reward 列） |
| `rebuild/` | 编排 | Client + Hierarchy + Writer |
| `rebuild/service.py` | `AgentLoopRebuild` | fetch → complete → **filter_to_window** → 写盘（新 run 在前） |

**依赖边界：** 只有 `write.mapper` 可碰 `samples`；`read` / `dashboard` / `rebuild` 不 import SampleRecord。

---

## Builder：不变的中间层

Builder 接收两种事件，通过 `BaseSample` 驱动下游存储。**链路中唯一设计为不变的是这两种事件格式。**

### 两种事件类型

**`trajectory_begin`** — 新轨迹开始

- `reason: "initial"` — session 刚创建
- `reason: "split"` — 消息前缀不匹配，开新 chain
- `reason: "budget"` — 长度预算耗尽后继续

```json
{
    "event": "trajectory_begin",
    "uid": "task-0001",
    "sample_index": 0,
    "session_index": 0,
    "trajectory_index": 0,
    "reason": "initial",
    "prompt_len": 18295
}
```

必填只有 `uid`；`trajectory_index` 缺省时自动递增。

**`step`** — 一步推理（模型思考 + 工具调用）

```json
{
    "event": "step",
    "uid": "task-0001",
    "step_index": 1,
    "finish_reason": "tool_calls",
    "completion_tokens": 200,
    "thought": "Let me explore the codebase first...",
    "tool_results": [
        {"name": "Bash", "action": "ls -la", "observation": "main.py", "status": "ok"}
    ]
}
```

`finish_reason`：`tool_calls` 继续；`stop` / `length` 结束并前进 cursor。

### 使用方式

```python
from rl_insight.experimental import TrajectoryBuilder
from rl_insight.experimental.samples import FileSampleRecord

builder = TrajectoryBuilder()  # 内存 SampleRecord
# 或：
builder = TrajectoryBuilder(
    lambda uid, si: FileSampleRecord.create("/data", uid=uid, sample_index=si)
)

builder.feed({"event": "trajectory_begin", "uid": "task-0001"})
builder.feed({"event": "step", "uid": "task-0001", ...})
samples = builder.samples
```

可视化：对已有树用 Mapper / `export_to_tempo.py` 写 Tempo，不是再实现一个 `BaseSample`。

| 场景 | 推荐 |
|------|------|
| 实验脚本、分析 | `SampleRecord` |
| 分布式并发写 | `FileSampleRecord` |
| 进 Grafana | `agent_loop.write.mapper` → Tempo |

---

## 怎么跑

**只造数（可落盘）：**

```bash
python rl_insight/experimental/generate_data.py /tmp/my-trajs --stream
python rl_insight/experimental/generate_data.py --memory --samples 4
```

**造数并导出 Tempo：**

```bash
python rl_insight/experimental/export_to_tempo.py --samples 2 --seed 42
```

**看盘：**

1. 启动 rl-insight server（Tempo + Grafana）
2. 跑 `export_to_tempo.py`
3. 打开 Grafana `agent_loop_trajectory`，点 **Rebuild Agent Loop**

Rebuild 调 `rebuild_from_tempo`：拉 Tempo → 分组/补全 → **只保留与 Grafana `from`/`to` 时间重叠的 span/run**（窗外 run 不建树）→ 新 run 在前 → 写 runtime dashboard JSON。

这样树上出现的 Run 都能在当前时间窗里查到时序数据，避免「有树壳、面板报 Data does not have a time field」。
