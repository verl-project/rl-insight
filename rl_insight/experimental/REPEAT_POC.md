# Agent Loop：Grafana Repeat 方案说明

> 分支：`youjinlin/agent-loop-repeat-poc`  
> **目标**：嵌套布局与 Rebuild/仿真模板一致，且 **行数随 Prom 枚举软刷新**（Grafana Repeat）。

---

## 目标形态

```text
Run · {run_id} · samples N · success a/b          ← ${run_id:text}   repeat=$run_id
  Sample {i} · success a/b · X turns · Y sessions ← ${sample:text}   repeat=$sample
    Session {i} · success a/b · X turns · Y trajs ← ${session:text}  repeat=$session
      [hideHeader] overview (Tempo)
      Trajectory #{i} · reward R · X turns        ← ${traj:text}     repeat=$traj
        sequence + details (Tempo)
```

- 变量：`includeAll` + `current=All`（展开每一项）；**不要**设 `allValue: .*`（会串 sample 的 session）。
- `refresh: onTimeRangeChanged` + 看板 `autoRefresh`：Prom 里节点增减后，Refresh / 自动刷新会重拉变量并重画行数。
- 轨迹正文仍查 Tempo（具体 `run/sample/session/traj`）。

---

## 产物

| 文件 | 作用 |
|------|------|
| `build_repeat_dashboard.py` | 生成带嵌套 Repeat 的 `agent_loop_trajectory.json` |
| `agent_loop_panel_templates.json` | overview / sequence / details |
| `prom_export.py` | SampleRecord → `*_info` + traj gauges |
| `export_to_tempo.py` | generate → Prom + Tempo（**不**改看板 JSON） |
| `tempo_export.py` | SampleRecord → Tempo spans |

```bash
# 1) 生成/更新 Repeat 看板（改布局时跑）
python -m rl_insight.experimental.build_repeat_dashboard
cp rl_insight/config/services/grafana/dashboards/agent_loop_trajectory.json \
  ~/.rl-insight/runtime/dashboards/

# 2) 喂数据（数量变化靠 Prom；看板靠 Refresh 软刷新行数）
python rl_insight/experimental/export_to_tempo.py --samples 2 --seed 42
```

`export_to_tempo.py` 默认把 `127.0.0.1:9108` 注册到当前 runtime
Prometheus 并触发 reload；只有抓取目标已由外部配置时才使用
`--no-register-scrape`。

编辑行时 Repeat by variable 应为 `run_id` / `sample` / `session` / `traj`（不是 Disable repeating）。

当 `agent_loop_run_info` 为空时，隐藏 Run 及其全部子级，避免 Grafana
为 `includeAll` 的虚拟选项渲染四层 `All`。

---

## Prometheus 合同

| 指标 | labels | 值 |
|------|--------|-----|
| `agent_loop_traj_turns` / `reward` / `success` | `run_id,sample,session,traj` | float / 0\|1 |
| `agent_loop_run_info` | `run_id,title` | `1` |
| `agent_loop_sample_info` | `run_id,sample,title` | `1` |
| `agent_loop_session_info` | `run_id,sample,session,title` | `1` |
| `agent_loop_traj_info` | `run_id,sample,session,traj,leaf,title` | `1`；title=行标题 |

变量：`query_result(*_info)` + `value`/`text` 命名捕获组。

---

## Tempo

- `service.name` = `agent-loop-poc`
- `state_lane_id` = `run={run_id}/sample={sample}/session={session}/traj={traj}`
- reward 只在 Prom / 行标题，不写 Tempo turn span
