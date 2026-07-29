# Inference Performance Degradation Perception
这个实验模块会完成整条链路：

```text
读取 standard 健康窗口和 inference 待检测窗口
→ 每个指标分别做 KDE 异常检测
→ 找到 timing_s/step 的确认异常区间
→ 从同一异常窗口筛选下游异常指标
→ 计算 Pearson/Spearman 相关系数和随机森林重要性
→ 返回异常贡献度最高的 5 个指标及百分比
```

测试员不需要修改 Python 代码。所有命令都从仓库根目录运行。第一次使用
时，如果项目还没有准备好 `.venv`，在 PowerShell 依次执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[degradation,test]"
.\.venv\Scripts\python.exe -m pip install -r experiment/degradation_perception/requirements.txt
```

`requirements.txt` 中包含随机森林所需的 `scikit-learn`。两条安装命令都
需要执行：根 `degradation` extra 提供 NumPy、SciPy 和 Paramiko，实验目录
的 requirements 补充 scikit-learn。

### 当前集成边界（仅记录）

- `OUT_OF_SCOPE-1`：`rl_insight.init(config={"server": {"url": ...}})` 与
  `RL_INSIGHT_SERVER_URL` 尚未完全贯通。提交训练任务前必须显式设置该环境
  变量；多节点 Ray 还必须把它加入 `runtime_env.env_vars`。例如 Bash 使用
  `export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"`，PowerShell 使用
  `$env:RL_INSIGHT_SERVER_URL = "http://<server-ip>:18080"`。
- `OUT_OF_SCOPE-2`：Prometheus workflow 尚未注册为根 console script，当前
  入口是 `python -m experiment.degradation_perception.prometheus_workflow`。
- `OUT_OF_SCOPE-3`：SciPy/scikit-learn 尚未同时进入正确的根安装 extra，所
  以必须按上文同时安装 `.[degradation,test]` 和实验 requirements。

这些边界需要修改 `rl_insight/**` 或根 `pyproject.toml`，本实验修复不改动
那些文件。

### 路径 A：没有 Prometheus，先验收完整算法

复制并执行这一行：

```powershell
.\.venv\Scripts\python.exe -m experiment.degradation_perception.simulated_prometheus --output-dir .\.run-output\degradation-acceptance --run-analysis
```

满足以下三项即通过：

1. 命令退出码为 `0`，屏幕最后的 JSON 中 `"ok": true`；
2. `.run-output\degradation-acceptance\validation_summary.json` 的
   `checks` 每一项都是 `true`；
3. 打开 `.run-output\degradation-acceptance\top5_result.json`，能看到
   `timing_s/step` 的一个异常事件和 5 个按贡献度排序的下游指标。

直接查看测试员结果：

```powershell
Get-Content .\.run-output\degradation-acceptance\top5_result.json
```

默认固定种子场景会按以下顺序返回：

```text
1. kv_cache_usage_perc
2. response_length_mean
3. num_requests_swapped
4. e2e_request_latency
5. global_seqlen_minimax_diff
```

百分比由实际相关系数和随机森林结果计算，不是写死的。五个有效候选在此
场景中合计为 `100%`。

### 路径 B：连接真实 Prometheus

1. 复制模板，不要直接改模板：

```powershell
Copy-Item experiment\degradation_perception\prometheus_workflow.example.yaml .\prometheus_workflow.local.yaml
notepad .\prometheus_workflow.local.yaml
```

2. 替换模板中的 URL、时间窗和查询占位值：

   - `prometheus.base_url`：Prometheus 根地址，不要追加
     `/api/v1/query_range`；
   - `windows.standard`：一段已知健康、稳定的历史时间；
   - `windows.inference`：怀疑发生退化、需要检测的时间；
   - 每个指标的 `standard_query` 和 `inference_query`：分别只选择对应健康
     任务和待检测任务的 PromQL。

3. 执行一键流程：

```powershell
.\.venv\Scripts\python.exe -m experiment.degradation_perception.prometheus_workflow --config .\prometheus_workflow.local.yaml --output-dir .\.run-output\degradation-real
```

4. 查看最终结果：

```powershell
Get-Content .\.run-output\degradation-real\top5_result.json
```

工作流会自动对每个指标、每个时间窗调用
`/api/v1/query_range`，保存原始响应，转换成算法输入，对所有指标执行
KDE，再运行相关系数和随机森林，最后写出：

```text
prometheus_query_responses.json  两个窗口的完整原始响应
converted_algorithm_input.json   清洗后的 standard/inference 数据
adapter_diagnostics.json         序列选择与样本清洗诊断
analysis_result.json             完整 KDE 和关联分析结果
top5_result.json                 测试员只需查看的简化结果
runtime_config/                  本次运行自动生成的指标配置
```

若显式传入工作流 `--config-dir`，请使用专用目录。语义一致的已有指标配置会
原样复用（包括注释和格式）；`abnormal_type` 或 `association` 与工作流配置
冲突时会返回 `existing_metric_config_conflict`，不会覆盖原文件。

`main --path` 和 `simulated_prometheus` 都是离线 JSON 路径，不会发起 HTTP；
只有上面的 `prometheus_workflow` 命令会连接真实 Prometheus。

真实 Prometheus 验收时请检查：

- 命令退出码为 `0` 且 stdout 中 `"ok": true`；
- `analysis_result.json` 中每个 `states.<指标>` 都是 `0`；
- 对刻意选择的异常窗口，`top5_result.json` 中
  `anomalyDetected` 为 `true`、`eventCount` 大于 `0`，并有预期的
  `top5`。

`state = 0` 只表示 KDE 已完成，不表示指标正常。真实窗口本来没有异常时，
`target_not_abnormal` 和空 Top5 是合理结果，不属于 Prometheus 接入失败。

### PromQL 必须满足的条件

- 每条查询最终必须返回一个标量时间序列。
- `standard_query` 与 `inference_query` 必须分别隔离目标任务。先查看现场
  实际 labels，再从 `project`、`experiment_name`、`instance`、`worker`、
  `replica`、`run_id` 等实际存在的字段中选择足以唯一定位任务的组合；不要
  假设所有环境都有全部 labels。
- 禁止用不带任务 selector 的全局 `avg(...)` 或 `sum(...)` 作为默认查询。
- Gauge 可以直接查询。
- Counter 必须先使用 `rate()` 或 `increase()`。
- Histogram 必须先用 `histogram_quantile()`，或用
  `rate(sum) / rate(count)` 转成标量。
- 同一任务确有多个 worker、engine 或 replica 时，先用任务 labels 隔离，
  再在该任务范围内聚合，或用下述精确 labels 选一条。算法绝不会悄悄取
  `result[0]`。

如果查询必须保留多条 label series，可在对应指标下精确选择一条：

```yaml
series_policy: select_by_labels
select_labels:
  instance: "10.0.0.8:9092"
  worker: "trainer_0"
```

筛选后仍必须恰好剩一条。Bearer Token 只允许从环境变量读取，例如先执行
`$env:PROM_TOKEN = "..."`，再把 YAML 中 `bearer_token_env` 写成
`PROM_TOKEN`；不要把 Token 直接写进 YAML。

### 常见失败怎么处理

| 屏幕中的错误或状态 | 含义与处理 |
| --- | --- |
| `prometheus_connection_error` | 地址、网络、TLS 或端口不通；默认会直连内部 Prometheus，只有必须经过系统代理时才把 `use_environment_proxy` 改为 `true`。 |
| HTTP `401` / `403` | 鉴权失败；检查 Token 环境变量和服务权限。 |
| `no_series` | 该 PromQL 在所选时间窗没有数据；检查时间、label 和指标名。 |
| `multiple_series` | `exactly_one` 查询返回多条 series；检查任务 labels，在已隔离任务内聚合，或配置 `select_by_labels`。 |
| `no_matching_series` | `select_by_labels` 没有匹配项；错误详情包含实际返回的候选 labels。 |
| `multiple_matching_series` | `select_by_labels` 仍匹配多条；增加实际存在且能区分候选的 labels。 |
| `resultType must be matrix` | 返回的不是 range matrix；真实工作流会调用 `query_range`，不要把 instant query 响应手工替换进去。 |
| `unsupported_native_histogram` | 原始 native histogram 不能直接计算；先在 PromQL 中转为标量。 |
| `state = 1` / `state = 2` | standard / inference 有效点数不足；扩大窗口或缩短 step。 |
| `metricErrors.<指标>` | 配置、输入或内部检测错误；该指标不会伪装成 `state = 1`，其他指标仍继续。 |
| `target_not_abnormal` | 主指标没有形成确认异常区间；先确认 inference 窗口确实包含退化。 |
| `partial_success` / `insufficient_data` | 对齐覆盖、公共样本或随机森林样本不足；尽量让所有指标使用相同步长和时间窗。 |

建议准备至少 `120` 个 standard 点和 `180` 个 inference 点，步长可先用
`10` 秒。代码硬下限更低，但过短窗口很难稳定训练随机森林。

> `abnormalContribution` 是异常窗口内的关联排名分数，不是因果概率，也不
> 能单独证明根因。若有效候选超过 5 个，百分比先在全部有效候选之间归一化
> 再截取 Top5，因此显示的五项之和可能小于 100%。

下面内容是开发者参考，普通测试员无需继续阅读。

This experimental module has two separate responsibilities:

- the existing KDE detector decides whether each metric is abnormal;
- optional association analysis ranks abnormal lower metrics after an explicitly
  configured Top metric has a confirmed anomaly.

Association contribution is a candidate-ranking score. It is not a causal
probability and does not establish a root cause.

## Data Flow

```text
standard/inference multi-metric data
→ existing KDE detection for every selected metric
→ confirmed Top-metric anomaly
→ abnormal lower-metric filtering
→ bounded time alignment
→ Pearson/Spearman correlation
→ random-forest importance from point anomaly labels
→ independently normalized evidence with configured weights
→ Top 5 associations per target event
```

The core association stage stays independent from data acquisition. The
`prometheus_workflow` entry point performs HTTP/PromQL acquisition first, then
passes the converted data through the same KDE and association implementation.

## Supported Inputs

- local UTF-8 JSON with explicit `standard` and `inference` sections;
- one or more metrics selected with space-separated `--metrics` values;
- canonical `timestamps`/`values` series or one already-fetched,
  single-series Prometheus `query_range` matrix;
- real Prometheus `query_range` acquisition through
  `prometheus_workflow.example.yaml`;
- an offline high-fidelity Prometheus package used by the simulator and its
  adapter (the outer package is not direct `main --path` input);
- `UP`, `DOWN`, and `BOTH` KDE directions;
- incremental remote JSON Lines through the Remote Monitor adapter.

The workflow does not implicitly merge multiple Prometheus label series.
Queries must aggregate to one scalar series or use explicit exact-label
selection. The module does not parse raw training logs, CSV files,
directories, or globs.

## Install

Run commands from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[degradation,test]"
python -m pip install -r experiment/degradation_perception/requirements.txt
```

### Linux Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[degradation,test]"
python -m pip install -r experiment/degradation_perception/requirements.txt
```

The local requirements file adds `scikit-learn` for
`RandomForestClassifier` and permutation importance. Existing NumPy and SciPy
dependencies are reused. KDE-only detection remains available when association
analysis is disabled.

## Input Format

The root contract remains exactly `standard` plus `inference`; do not add
association settings to the data JSON.

```json
{
  "standard": {
    "timing_s/step": {
      "timestamps": [1, 2, 3],
      "values": [1.00, 1.01, 0.99]
    },
    "kv_cache_usage_perc": {
      "timestamps": [1, 2, 3],
      "values": [40.0, 41.0, 39.0]
    }
  },
  "inference": {
    "timing_s/step": {
      "timestamps": [10, 11, 12],
      "values": [1.00, 1.30, 1.40]
    },
    "kv_cache_usage_perc": {
      "timestamps": [10, 11, 12],
      "values": [45.0, 70.0, 85.0]
    }
  }
}
```

`standard` is trusted historical-normal data and `inference` is the data to
detect. Timestamps and values must have equal lengths. Every Top and lower
metric must be present in both phases and included in `--metrics`. The short
example shows the shape only; actual detection must satisfy the configured
minimum point counts.

Input preprocessing is shared by KDE and association analysis: invalid pairs
are removed together, timestamps are sorted, and a duplicate timestamp keeps
its last valid value. `--start-time` and `--end-time` are inclusive and apply
only to `inference`.

See [Input and Output](docs/INPUT_OUTPUT.md) for the full canonical,
Prometheus, JSON Lines, and time contracts.

## Prometheus Matrix Simulation

The offline workflow keeps three formats separate:

1. each nested `response` is a complete Prometheus `query_range` matrix;
2. the outer `standard`/`inference` package is a project-owned test container;
3. adapter output is the unchanged algorithm input containing only
   `timestamps` and `values`.

It generates fixed-seed scalar data for the target plus the five
inference-service candidates shown in the tester workflow, together with
unrelated, constant, and sparse exclusion cases. It then applies strict matrix
validation, explicit series selection, finite-value cleaning, Unix-second
preservation, KDE, and association analysis.

Windows PowerShell:

```powershell
python -m experiment.degradation_perception.simulated_prometheus `
  --output-dir .\.run-output\prometheus-simulation `
  --run-analysis
```

Linux Bash:

```bash
python -m experiment.degradation_perception.simulated_prometheus \
  --output-dir ./.run-output/prometheus-simulation \
  --run-analysis
```

The run writes `simulated_prometheus_matrix.json`,
`converted_algorithm_input.json`, `adapter_diagnostics.json`,
`analysis_result.json`, `top5_result.json`, and `validation_summary.json`.
Top 5 is selected only after contribution percentages are normalized across
all valid candidates; the displayed subset is never renormalized.

The simulator does not contact Prometheus. This test validates Prometheus
format compatibility and end-to-end algorithm behavior, not real production
Prometheus data. See [Prometheus Matrix Simulation](docs/PROMETHEUS_SIMULATION.md)
for schemas, selection errors, metric behavior, Gauge/Counter/Histogram rules,
outputs, and limitations.

## Configure a Top Metric

On first detection, a per-metric YAML file is copied from
`default_config.yaml`; existing files are not overwritten:

```text
timing_s/step -> timing_s__step.yaml
```

Without `--config-dir`, generated files go to a user-writable directory:

```text
Windows: %APPDATA%\rl-insight\degradation-perception
POSIX:   ${XDG_CONFIG_HOME:-~/.config}/rl-insight/degradation-perception
```

The package-owned `default_config.yaml` remains a read-only template. An
explicit `--config-dir` takes precedence, and an existing metric file is never
overwritten.

Enable association in the Top metric's file under the selected `--config-dir`:

```yaml
association:
  enabled: true
  target_metrics:
    - "timing_s/step"
  candidate_mode: abnormal_lower_metrics
  weights:
    correlation: 0.5
    random_forest: 0.5
  top_k: 5
  context_ratio: 1.0
  min_aligned_points: 10
  min_rf_samples: 30
  min_coverage_ratio: 0.6
  alignment_tolerance: null
  random_forest:
    n_estimators: 200
    class_weight: balanced
    random_state: 42
    importance_method: permutation
```

User-facing controls:

| Field | Meaning |
| --- | --- |
| `enabled` | Enables post-KDE analysis. Default is `false`. |
| `target_metrics` | Explicit Top metrics; no metric is inferred from values. |
| `weights` | Non-negative correlation/RF weights whose sum must be `1`. |
| `top_k` | Number of ranked associations returned per event. |
| `context_ratio` | Context on each side as a fraction of raw event duration. |
| `min_aligned_points` | Minimum paired points for a candidate. |
| `min_rf_samples` | Minimum common rows before random-forest training. |
| `min_coverage_ratio` | Minimum aligned target-window coverage in `[0, 1]`. |

The first enabled association block among the selected metric configs supplies
the shared settings, so place the Top metric first in `--metrics`. Old YAML
files without `association` are deep-merged with the disabled defaults.

CLI targets override `target_metrics` and enable analysis for that invocation:

```text
--association-target "timing_s/step"
```

When neither YAML nor CLI enables association analysis, the original KDE
response shape is unchanged and has no `associationAnalysis` field.

## Run

The `--metrics` values are separated by spaces, not commas.

### Windows PowerShell

```powershell
$result = python -m experiment.degradation_perception.main `
  --path .\data.json `
  --metrics "timing_s/step" "kv_cache_usage_perc" "response_length_mean" `
  --association-target "timing_s/step" `
  --config-dir .\.run-config | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) { throw "Detection command failed" }
$result.associationAnalysis.targets.'timing_s/step'.events |
  ConvertTo-Json -Depth 12
```

### Linux Bash

```bash
python -m experiment.degradation_perception.main \
  --path ./data.json \
  --metrics "timing_s/step" "kv_cache_usage_perc" "response_length_mean" \
  --association-target "timing_s/step" \
  --config-dir ./.run-config
```

KDE-only smoke test with the bundled data:

```powershell
python -m experiment.degradation_perception.main `
  --path experiment/degradation_perception/sample_data.json `
  --metrics "timing_s/step"
```

The CLI writes exactly one strict JSON object to stdout. Runtime failures are
also JSON and use a non-zero exit code.

`--source-type training_log` preserves raw numeric timestamps, while
`--source-type prometheus` preserves Unix seconds. For `remote_monitor`, the
display mode is selected once for the complete inference series: if any value
is greater than `10000`, every interval boundary uses `value / 10000 / 60`;
endpoints are never converted independently.

## Association Output

Each confirmed target anomaly is analyzed independently:

```json
{
  "associationAnalysis": {
    "enabled": true,
    "status": "success",
    "weights": {
      "correlation": 0.5,
      "randomForest": 0.5
    },
    "targets": {
      "timing_s/step": {
        "status": "success",
        "events": [
          {
            "targetAbnormalRange": {
              "startTime": 103.0,
              "endTime": 110.0
            },
            "analysisWindow": {
              "startTime": 98.0,
              "endTime": 115.0
            },
            "randomForestStatus": "success",
            "topAssociations": [
              {
                "rank": 1,
                "metric": "kv_cache_usage_perc",
                "abnormalContribution": 42.7,
                "pearson": 0.76,
                "spearman": 0.83,
                "selectedCorrelation": 0.83,
                "correlationDirection": "positive",
                "randomForestImportance": 0.42,
                "coverageRatio": 0.95,
                "alignedSampleCount": 40
              }
            ]
          }
        ]
      }
    }
  }
}
```

For each candidate:

```text
corr = max(abs(Pearson), abs(Spearman))

final_score
  = correlation_weight * normalized_corr
  + random_forest_weight * normalized_random_forest_importance
```

With defaults, each normalized evidence source contributes 50%. If one source
is unavailable, the valid source is renormalized to the full contribution and
the event is `partial_success`. `allAssociations` retains the full ranking;
`topAssociations` is truncated to `top_k`, so the displayed subset need not sum
to 100%.

Common statuses:

| Status | Meaning |
| --- | --- |
| `success` | Correlation and random-forest evidence are available. |
| `partial_success` | One source is unavailable or RF used a documented fallback. |
| `target_not_abnormal` | The Top metric has no confirmed formal interval. |
| `no_candidate_metrics` | No lower metric passed event-window filtering. |
| `insufficient_data` | Alignment or both evidence sources are insufficient. |
| `target_metric_missing` | The Top metric was not selected or is absent from inference. |
| `target_detection_failed` | The selected Top metric did not complete KDE detection. |

## Existing KDE Output

The original fields retain their meanings:

| Field | Meaning |
| --- | --- |
| `states.<metric>` | `0` completed, `1` insufficient standard, `2` insufficient inference. |
| `metricErrors.<metric>` | Optional redacted configuration, input, or internal error; failed metrics are absent from `states`. |
| `results.<metric>.pointDiagnostics` | Per-point KDE diagnostics; `abnormal` is the label reused by association analysis. |
| `results.<metric>.currentAbnormalTimeRange` | Formal ranges in the current call before history publication. |
| `abnormalTimeRange.<metric>` | History-confirmed formal ranges. |
| `startTime` / `endTime` | Display boundaries after existing one-unit padding. |
| `duration` | Raw candidate span before boundary padding. |

State `0` means detection completed; it does not mean that no degradation was
found. Read `abnormalTimeRange.<metric>` for confirmed anomalies.

## Test and Verify

### Windows PowerShell

```powershell
python -m pytest -q experiment/degradation_perception/tests
python -m compileall -q experiment/degradation_perception
python -m ruff check experiment/degradation_perception
git diff --check -- experiment
git status --short
git diff --stat -- experiment
git diff -- experiment
```

### Linux Bash

```bash
python -m pytest -q experiment/degradation_perception/tests
python -m compileall -q experiment/degradation_perception
python -m ruff check experiment/degradation_perception
git diff --check -- experiment
git status --short
git diff --stat -- experiment
git diff -- experiment
```

Related cross-project tests require the Recipe dependency group:

```bash
python -m pip install -e ".[recipe,degradation,test]"
python -m pytest -q tests/monitor/ut tests/recipe/data
```

## Remote Monitor

Copy `monitor_config.example.yaml` to an untracked local file, fill in the SSH
source, and call `run_remote_monitor()` programmatically. Do not commit
credentials. See [Remote Monitor](docs/REMOTE_MONITOR.md).

## Project Structure

```text
main.py                    CLI boundary and association target override
algorithm.py               existing per-metric KDE orchestration, then association
association_analysis.py    candidate filtering, evidence, scoring, and ranking
time_alignment.py          event windows and bounded cross-metric alignment
stable_segment_detector.py stable modes and three-part voting
kde_utils.py               KDE, peaks, valleys, and CDF quantiles
interval_utils.py          candidate and formal intervals
preprocessing.py           shared strict input preprocessing
prometheus_matrix_adapter.py explicit matrix validation and series selection
prometheus_workflow.py       real query_range acquisition and one-command analysis
simulated_prometheus.py    fixed-seed package generation and E2E acceptance CLI
result_presentation.py     compact tester-facing Top5 projection
requirements.txt           supplemental scikit-learn dependency
examples/                  deterministic offline Prometheus matrix package
tests/                     unit and end-to-end tests
docs/                      focused technical documentation
```

## Detailed Documentation

- [Association Analysis](docs/ASSOCIATION_ANALYSIS.md): evidence definitions,
  event flow, degradation rules, and limitations.
- [Algorithm](docs/ALGORITHM.md): KDE modes, thresholds, and interval validation.
- [Input and Output](docs/INPUT_OUTPUT.md): complete schemas and time semantics.
- [Prometheus Matrix Simulation](docs/PROMETHEUS_SIMULATION.md): offline package,
  matrix adapter, generated signals, commands, outputs, and production limits.
- [Design Notes](docs/DESIGN_NOTES.md): deterministic engineering decisions.
- [Remote Monitor](docs/REMOTE_MONITOR.md): remote execution and safety.
