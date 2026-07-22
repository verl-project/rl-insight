# experimental — 实验轨迹数据流水线（Demo）

> **当前阶段：原型演示。** 这里展示的是一条完整的数据链路——从生成到处理再到展示。链路中的每个组件在后续都可以独立替换，但位于链路中间的事件协议和 Builder 层保持不变。

## 整体架构

```
数据生成                      数据处理                        数据展示
─────────────────────────────────────────────────────────────────────
                                                           ┌─ HTML Timeline (当前)
模拟脚本 ──┐                  ┌─ SampleRecord (内存)       │
           ├── Builder ── BaseSample ─┤                     ├─ Grafana (后续)
uni-agent ─┘     ▲           └─ FileSampleRecord (文件) ──┤
                 │                  └─ GrafanaRecord (后续) ┘
                 │
           两种事件格式（稳定不变）
```

整条链路分为三段，每段都可以独立演进：

| 阶段 | 当前 | 后续替换 |
|------|------|----------|
| **数据生成** | Python 脚本模拟 agent 推理过程 | 接入 uni-agent Gateway，在实际 rollout 时发出相同的事件 |
| **数据处理** | Builder 接收事件，驱动 `SampleRecord`（内存）或 `FileSampleRecord`（文件）落盘 | 新增面向 rl-insight + Grafana 的 `BaseSample` 实现，将数据写入 Prometheus 或 JSON API |
| **数据展示** | Timeline HTML 页面，轮询文件系统渲染时序图 | 切换到 Grafana 面板，读取新 `BaseSample` 实现暴露的指标 |

**链路中唯一设计为不变的，是 Builder 的事件协议。** 这个协议足够薄——只有两种 JSON 事件——但它足以表达 agent 推理的完整生命周期。数据生成端无论怎么换（模拟脚本、uni-agent、其他框架），数据处理端无论用什么存储（内存、文件、时序数据库），只要两边通过 Builder 对接，就不需要互相感知对方的存在。

## Builder：不变的中间层

Builder 是整个流水线的核心枢纽。它接收两种事件，通过 `BaseSample` 接口驱动下游存储，对上游生成端和下游存储端都保持透明。

### 两种事件类型

**`trajectory_begin`** — 新轨迹开始

新轨迹的创建时机有三种，对应 Gateway 层的实际行为：

- `reason: "initial"` — session 刚创建，没有任何历史上下文
- `reason: "split"` — 新请求的消息前缀与当前 chain 不匹配，旧 chain 固化，开新 chain
- `reason: "budget"` — response 长度预算耗尽后被截断，在已有历史基础上继续

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

必填字段只有 `uid`，其余都可以缺省。`trajectory_index` 不传时自动递增。

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

`finish_reason` 字段控制 Builder 的状态流转：

- `tool_calls` → 轨迹继续，cursor 停留在当前 trajectory
- `stop` → 轨迹正常结束，cursor 前进到下一个 trajectory_index
- `length` → 轨迹被截断结束，标记为 truncated，cursor 前进

**典型时序**——一个 sample 的某次 session 的事件流：

```
trajectory_begin  (reason=initial)     ← 首次开始
step  (finish_reason=tool_calls)       ← 工具调用，继续
step  (finish_reason=tool_calls)       ← 继续
step  (finish_reason=stop)             ← 正常结束

trajectory_begin  (reason=split)       ← 不匹配，开新 chain
step  (finish_reason=tool_calls)
step  (finish_reason=length)           ← 被截断

trajectory_begin  (reason=budget)      ← 截断后继续
step  (finish_reason=stop)
```

### Builder 使用方式

Builder 通过工厂函数解耦存储端，换存储只需换工厂：

```python
from rl_insight.experimental import TrajectoryBuilder

# 内存版
builder = TrajectoryBuilder()

# 文件版——只换这一行
builder = TrajectoryBuilder(
    lambda uid, si: FileSampleRecord.create("/data", uid=uid, sample_index=si)
)

# 后续 Grafana 版——也是换这一行
builder = TrajectoryBuilder(
    lambda uid, si: GrafanaRecord.create("http://prom:9090", uid=uid, sample_index=si)
)

# 接收事件
builder.feed({"event": "trajectory_begin", "uid": "task-0001"})
builder.feed({"event": "step", "uid": "task-0001", ...})

# 批量加载
builder.feed_jsonl("events.jsonl")
samples = builder.samples
```

## BaseSample 接口

`BaseSample` 是一个 Python Protocol，定义了六个方法。任何对象只要实现了这六个方法，就能作为 Builder 的下游目标——不需要继承任何基类，不依赖任何框架。

| 方法 | 用途 |
|------|------|
| `new_trajectory(session_index)` | 在指定 session 下创建一条新轨迹，返回轨迹对象 |
| `get_trajectory(session_index, traj_idx)` | 读取一条轨迹，不存在时返回 None |
| `add_step(session_index, traj_idx, step)` | 向轨迹追加一个推理 step |
| `finish_trajectory(session_index, traj_idx, reason, status)` | 标记轨迹结束，记录退出原因和状态 |
| `set_trajectory_reward(session_index, traj_idx, score)` | 设置轨迹的 reward 分数 |
| `set_trajectory_token_data(session_index, traj_idx, ...)` | 设置 token 级别的数据（prompt_ids、response_mask 等） |

这是一个极简的 CRUD 接口。它不预设存储方式，不关心并发模型，也不包含任何业务逻辑。所有业务语义（何时创建轨迹、何时结束、如何从 step 推断 finish_reason）都在 Builder 层完成。

## 两种 BaseSample 实现

### SampleRecord — 内存版

数据完全驻留在 Pydantic 模型字段中。适合单进程分析、实验 JSONL 加载、以及任何需要快速随机访问的场景。

内部层级结构与 uni-agent 保持一致：

```
SampleRecord          ← 一个 RL 训练样本
  └── SessionRecord   ← 对应一次 GatewaySession（一次 rollout 尝试）
        └── TrajectoryRecord  ← 对应一条 chain / trajectory
              └── Step        ← 一次模型调用 + 工具执行
                    └── ToolResult  ← 一次工具调用结果
```

- **Step** 对应 uni-agent 的 `StepOutput`：包含 `step_idx`、`thought`、`response`、`tool_results`、`done`、`exit_reason`
- **ToolResult** 对应 uni-agent 的 `ToolResult`：包含 `tool_call_id`、`name`、`action`、`observation`、`status`、`execution_time`
- **TrajectoryRecord** 同时承载 token 级别数据（对应 `gateway.types.Trajectory`）和 step 列表

序列化方式为 Pydantic 的 `model_dump()` / `model_validate()`，仅支持 JSON 格式。

对外入口是 `SampleRecord.create(uid=..., sample_index=...)`，所有轨迹操作通过 `SampleRecord` 的方法完成，外部不直接操作 `SessionRecord` 或 `TrajectoryRecord`。

### FileSampleRecord — 文件版

每条轨迹存为一个独立的 JSON 文件，无内存常驻状态。适合分布式场景：多个进程写同一个 sample 的不同轨迹不会冲突。

目录组织：

```
{root_dir}/{uid}/
  ├── _index.json       # 索引：记录 sample_index 和已有的 (session, trajectory) 列表
  ├── t_0_0.json        # session 0, trajectory 0（完整的 TrajectoryRecord JSON）
  ├── t_0_1.json        # session 0, trajectory 1
  └── t_1_0.json        # session 1, trajectory 0
```

关键设计：

- **一个轨迹一个文件**：不同 `(session_index, trajectory_index)` 的组合对应不同文件，写操作天然隔离
- **原子写入**：先写临时文件 → fsync → rename，不会读到半写状态
- **轻量索引**：`_index.json` 记录轨迹列表，读操作不需要扫描目录
- **按需加载**：调用 `load()` 方法可以把磁盘上所有轨迹批量读入 `SampleRecord`，用于聚合分析

由于文件系统需要定位到具体目录，创建时必须提供 `uid` 参数（这是 `FileSampleRecord` 与 `SampleRecord` 在接口上的唯一差异：`SampleRecord` 的 `uid` 是字段，`FileSampleRecord` 的 `uid` 同时是路径组成部分）。

### 选型指南

| 场景 | 推荐实现 |
|------|----------|
| 实验脚本、notebook 分析、JSONL 加载 | `SampleRecord` |
| 分布式 rollout、多进程并发写入 | `FileSampleRecord` |
| 后续 rl-insight 指标采集 | 新的 `BaseSample` 实现（如 GrafanaRecord） |

## 数据生成（当前：模拟脚本）

```bash
python rl_insight/experimental/generate_data.py /tmp/my-trajs --stream
```

模拟 12 个编程任务（如修复 bug、优化查询、重构中间件）的 agent 推理过程。每条轨迹包含 2-7 个 step，使用 Bash / Read / Edit / finish 等工具，约 35% 的样本最终 reward=1。

`--stream` 模式下每秒生成一条轨迹，约一分钟完成全部 60 条左右的轨迹。默认每次启动前清空目标目录（`--no-clean` 可跳过）。

**后续替换方向**：在 uni-agent 的 Gateway 层，每次 `trajectory_begin` 和 `step` 发生时发出相同格式的 JSON 事件，喂给 Builder 即可。生成端不需要感知下游是内存、文件还是 Grafana。

## 数据展示（当前：Timeline HTML）

```bash
python rl_insight/experimental/server.py /tmp/my-trajs --port 8080
```

浏览器打开 `http://localhost:8080`，提供：

- 样本级总览：session 数、总 turn 数、总轨迹数、成功数
- 时序色块图：每个 step 用颜色编码工具类型（Bash 绿 / Read 青 / Edit 黄 / LLM 蓝），被截断的轨迹带红色边框
- Session 切换：每个 sample 的多轮 rollout 可以分 tab 查看
- 交互细节：Hover 显示 step 详情，点击展开完整思考内容
- 自动刷新：每 5 秒轮询文件系统，适合配合 `--stream` 实时观察生成进度

**后续替换方向**：Grafana 面板读取新 `BaseSample` 实现暴露的指标，不再轮询文件系统。

## 目录结构

```
experimental/
  ├── README.md           # 本文档
  ├── __init__.py         # 公开导出
  ├── base.py             # BaseSample Protocol 定义（六个方法）
  ├── sample.py           # SampleRecord 内存版（Pydantic 实现）
  ├── file_sample.py      # FileSampleRecord 文件版
  ├── builder.py          # TrajectoryBuilder 事件驱动适配层
  ├── generate_data.py    # 模拟数据生成脚本
  └── server.py           # Timeline HTML 可视化服务
```
