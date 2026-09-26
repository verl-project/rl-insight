# Grafana dashboard 开发

> **英文版本：[README.md](README.md)**

RL-Insight 的 Grafana dashboard 以可复用的 Jsonnet 模块维护，并被组合成运行时服务所加载的、已提交的 Grafana JSON 文件。

## 架构

开发期的一半是开发者执行的一条命令（wrapper → 框架核心 → Jsonnet 入口 → 已提交 JSON）；运行时的一半是运行中的服务所做的事（启动 → 暂存 → provisioning → Grafana）。下面每一段箭头都写明下一步实际做了什么。

```text
开发者执行：python tools/grafana/generate_dashboards.py
        │
        │ CALLS（调用）—— 生产薄封装调用框架核心
        │ tools/grafana/framework/generate.py 里的 render()，
        │ 并把 Jsonnet 入口 tools/grafana/dashboards.jsonnet 传给它
        ▼
render(tools/grafana/dashboards.jsonnet)
        │
        │ EVALUATES WITH go-jsonnet（用 go-jsonnet 求值）—— 生成器求值的文件
        │ 就是入口 tools/grafana/dashboards.jsonnet
        ▼
tools/grafana/dashboards.jsonnet  （生成器的 Jsonnet 入口）
    导入 registry：tools/grafana/dashboard_compositions.libsonnet，
    后者再导入可复用模块 tools/grafana/dashboards/*.libsonnet
    导入组合库：tools/grafana/framework/composer.libsonnet，
    它是这个入口调用到的库——不是生成器之前的一个独立顺序节点
        │
        │ COMPOSES（组合）—— 入口对每一个已注册的 composition 调用
        │ composer.compose(modules, dashboard)，把选中的模块合并成
        │ 一个完整的 dashboard 对象
        ▼
{ "<composition-name>": <完整的 dashboard 对象>, ... }
        │
        │ SERIALIZED BY（被谁序列化）—— 薄封装用框架核心的 generated_text()
        │ 把每个对象序列化并写出
        ▼
rl_insight/config/services/grafana/dashboards/verl/<composition-name>.json
    生成好的生产 dashboard，已提交到仓库
        │
        │ COPIED AT STARTUP BY（启动时被谁复制）—— rl_insight/server/runtime.py
        │ 的 _stage_grafana_dashboards() 复制这些已提交的文件
        ▼
<runtime_dir>/dashboards/<composition-name>.json  （运行时暂存副本）
        │
        │ POINTED AT BY（被谁指向）—— _render_grafana_provisioning() 写出
        │ provisioning/dashboards/default.yml，其 file provider 把
        │ options.path 设为 <runtime_dir>/dashboards
        ▼
Grafana
    实际加载并展示 dashboard 的服务
```

同一条链路用文字说明：

1. **调用（Invoke）** —— 开发者执行 `python tools/grafana/generate_dashboards.py`。薄封装调用框架核心 `tools/grafana/framework/generate.py` 里的 `render()`，并把 Jsonnet 入口 `tools/grafana/dashboards.jsonnet` 传给它。
2. **求值（Evaluate）** —— `render()` 用 go-jsonnet 对 `tools/grafana/dashboards.jsonnet` 求值。该入口导入 composition registry `dashboard_compositions.libsonnet`（后者再导入 `dashboards/*.libsonnet` 里的可复用模块）以及组合库 `framework/composer.libsonnet`。composer 是这个入口导入并调用的库——不是生成器在它之前执行的独立步骤。
3. **组合（Compose）** —— 入口对每一个已注册的 composition 调用 `composer.compose(modules, dashboard)`，把选中的模块合并成一个完整的 dashboard 对象。`render()` 返回 `{ "<composition-name>": <dashboard object> }`。
4. **序列化并写入（Serialize and write）** —— 薄封装用框架核心的 `generated_text()` 逐个序列化 dashboard，写入 `rl_insight/config/services/grafana/dashboards/verl/`；这些文件会被提交。
5. **启动时复制（Copy at startup）** —— `rl_insight/server/runtime.py:prepare_files()` 调用 `_stage_grafana_dashboards()`，把已提交的 JSON 复制到 `<runtime_dir>/dashboards`。运行时使用的是这些暂存副本；Grafana 从不直接读取仓库路径。
6. **加载（Load）** —— `_render_grafana_provisioning()` 写出 `provisioning/dashboards/default.yml`，这是一个 file provider，其 `options.path` 指向 `<runtime_dir>/dashboards`。Grafana 启动时读取该 provisioning 文件，扫描它指向的目录，加载其中的暂存 dashboard JSON。

这套结构的目标是：

- 复用通用的 dashboard 内容，而不是复制大型 JSON 文件；
- 让新增和维护 dashboard 变体变得容易；
- 保持现有 RL-Insight / Grafana 运行时行为不变。

## 本文档面向谁？

| 角色                      | 会变化什么？                                      | 我应该做什么？                                                                          |
| ------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| RL-Insight / Grafana 用户 | 运行时没有任何变化                                | 完全像以前一样启动和使用 RL-Insight                                                     |
| Dashboard 开发者          | Dashboard 以模块 + 组合的方式编写                 | 编辑 composition registry，只在需要时添加模块，然后生成并提交 JSON                      |

## 运行时行为

本次重构**变更前后运行时行为完全一致**。运行时只有一条路径，它从不执行 Jsonnet，这里只画一次：

```text
仓库：rl_insight/config/services/grafana/dashboards/verl/*.json
    已提交的 dashboard JSON，在开发期生成（见「架构」）
        │
        │ COPIED AT STARTUP BY（启动时被谁复制）—— RL-Insight 启动时执行
        │ rl_insight/server/runtime.py:prepare_files()，它调用
        │ _stage_grafana_dashboards() 把这些文件复制到运行时目录
        ▼
<runtime_dir>/dashboards/*.json
    暂存副本——运行时读取的是这些副本，从不直接读取仓库路径
        │
        │ POINTED AT BY（被谁指向）—— _render_grafana_provisioning() 写出
        │ provisioning/dashboards/default.yml，其 file provider 把
        │ options.path 设为本目录（指向目录，而不是逐个文件列举）
        ▼
Grafana provisioning
        │
        │ SCANNED BY（被谁扫描）—— Grafana 启动时读取该配置，扫描它指向的
        │ 目录，并加载其中找到的 dashboard 文件
        ▼
Grafana
    实际加载并展示 dashboard 的服务
```

上图中的名词含义：

| 名词                       | 含义                                                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `provisioning`             | Grafana 的“按配置发现 dashboard”机制：Grafana 启动时读取自己的 provisioning 文件，扫描这些文件所指向的目录，并加载其中找到的 dashboard 文件。file provider 指向的是目录，不是逐文件列表。                              |
| `committed dashboard JSON` | 仓库中已生成并提交的最终 Grafana JSON（`rl_insight/config/services/grafana/dashboards/verl/*.json`）。启动时它会被复制到 `<runtime_dir>/dashboards`，Grafana 加载的是这些暂存副本。                                       |
| `Grafana`                  | 实际加载并展示 dashboard 的服务。                                                                                                                                                                                        |
| `gojsonnet`                | 把 Jsonnet 求值为 JSON 的引擎。它只在 dashboard 开发和生成期间运行，绝不在用户运行时执行。                                                                                                                                |

## 生成是自动的吗？

**不是。** 普通的 RL-Insight 用户从不执行生成脚本，启动链路上也没有任何一步会对 Jsonnet 求值。

启动时，`rl_insight/server/runtime.py:prepare_files()` 对 Grafana 做三件事（`runtime.py:112`–`115`）：

1. `_render_grafana_config()` 写出 `grafana.ini`；
2. `_stage_grafana_dashboards()` 把 `grafana.dashboards_dir`（`rl_insight/config/services/grafana/dashboards/`）下**已经提交**的 JSON 复制到运行时目录。它只复制文件——从不执行生成器；
3. `_render_grafana_provisioning()` 写出 `provisioning/dashboards/default.yml`，这是一个 Grafana file provider，其 `options.path` 指向该运行时目录。随后 Grafana 扫描该目录并加载其中的暂存副本——而不是仓库里的文件。

所以生成出来的 JSON 不会在启动时产生；它由开发者生成一次，然后提交。

因此，dashboard 开发者修改 Jsonnet 源文件后必须手动生成——运行 `python tools/grafana/generate_dashboards.py`，再运行 `python tools/grafana/generate_dashboards.py --check`，然后提交重新生成的 JSON。命令以及 `--check` 的比较方式见[生成与校验](#生成与校验)。

## 两个生成层次

生成分成两层。**这两层都不是启动链路中的步骤**，并且仓库当前没有任何 CI 任务会调用它们。

| 层次                   | 文件                                   | 职责                                                                                                                                                                                                                                        |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 生产薄封装（#173）     | `tools/grafana/generate_dashboards.py` | 只提供生产相关的默认值：把框架生成器指向 `tools/grafana/dashboards.jsonnet` 和已提交的输出目录 `rl_insight/config/services/grafana/dashboards/verl/`。它的 `--check` 比较解析后的 JSON 对象，因此仅由序列化导致的 key 顺序差异会被忽略。 |
| 框架核心（通用，#174） | `tools/grafana/framework/generate.py`  | 可复用的渲染/序列化核心：用 go-jsonnet 对 composition 配置求值，以确定性方式序列化（key 排序、固定缩进），并对外提供 `render()` 和 `generated_text()`。                                                                                    |

`generate_dashboards.py` 没有重新实现上述任何逻辑；它从 `tools/grafana/framework/` 导入 `generate`。该 framework 目录属于通用组合改动（#174），本分支没有把它复制进来。

## Dashboard 开发：变更前后

### 变更前

每个 dashboard 都是仓库里一个完整的 Grafana JSON 文件——数千行；新增一个变体的做法是复制这样一个文件再改。

```text
每个 dashboard 一个完整的 Grafana JSON
        │
        │ COPY AND EDIT（复制并修改）—— 每个变体都从现有文件的完整副本开始
        ▼
若干个几乎相同的大型 JSON 文件
        │
        │ REPEAT BY HAND（手工重复）—— 共享的改动必须在每份副本上重做一遍
        ▼
各副本逐渐不一致
```

维护问题在于：一处共享改动（新增一个 panel、重命名一个 variable、修正一个阈值）必须在每份副本里手工重复，没有任何机制让这些副本保持同步，而评审一次改动意味着要读一份数千行的 JSON diff。

### 变更后

dashboard 之间共有的内容被抽成可复用模块。一个模块拥有某个子系统的一份 dashboard 内容——它的 panels、rows 和 variables。composition 指明一个 dashboard 由哪些模块组成，生成器再把它们合并成一个完整的 Grafana JSON 文件。

```text
panels / rows / variables 拆分为可复用模块
        │
        │ SELECT（选择）—— composition 指明组成一个 dashboard 的模块
        ▼
composition registry（dashboard_compositions.libsonnet）
        │
        │ COMPOSE（组合）—— 生成器只合并这些选中的模块
        ▼
每个 dashboard 一个完整的 Grafana JSON 文件
```

这解决了什么：

- 共享改动只需改一次，改在拥有它的模块里，而不是每份副本里；
- 新增一个 dashboard 变体只是加一条注册项，而不是再复制一份大型 JSON；
- dashboard 内容以小型模块的形式被评审，而不是一份巨大的 JSON diff。

## 可复用模块

| 模块         | 职责                                                                             |
| ------------ | -------------------------------------------------------------------------------- |
| `trainer`    | 训练指标：actor、critic、reward、loss、rollout、throughput、timing 等            |
| `controller` | Controller / 编排 / transfer-queue 控制指标                                      |
| `storage`    | 分区、存储和数据传输指标                                                         |
| `trajectory` | Tempo / TraceQL 状态时间线                                                       |
| `vllm`       | vLLM 推理和宿主机侧指标                                                          |
| `sglang`     | SGLang 推理指标                                                                  |
| `npu`        | Ascend NPU 指标                                                                  |

共享的 VERL base 是：

```text
verlBase
= trainer
+ controller
+ storage
+ trajectory
```

当前的生产 composition 是：

```text
vLLM dashboard
= verlBase + vllm + npu

SGLang dashboard
= verlBase + sglang
```

## 新增一个 dashboard

### 仅复用现有模块

如果新的 dashboard 只需要现有内容，则不需要新增模块。

例如：

```text
trainer + trajectory + npu
```

只需在以下文件中新增一个条目：

```text
tools/grafana/dashboard_compositions.libsonnet
```

示例：

```jsonnet
my_verl_dashboard: {
  modules: [
    trainer,
    trajectory,
    npu,
  ],
  dashboard: {
    metadata: {
      name: 'my-verl-dashboard',
      labels: {},
      annotations: {},
    },
    title: 'my_verl_dashboard',
    tags: ['RL-Insight', 'verl'],
    spec: productionSpec,
    variableOrder: [
      'datasource',
      'project',
      'experiment_name',
      'npu_instance',
    ],
    rowOrder: [
      'rl state timeline',
      'training metric',
    ],
  },
},
```

然后生成 JSON：

```bash
python tools/grafana/generate_dashboards.py
```

### 新增一个 engine 或新增内容

如果新的 engine `foo` 有自己的 panel：

1. 新增一个模块：

```text
tools/grafana/dashboards/foo.libsonnet
```

2. 在以下文件中导入它：

```text
tools/grafana/dashboard_compositions.libsonnet
```

3. 新增一个 composition：

```jsonnet
verl_tainer_v1_with_foo_engine: {
  modules: verlBase + [foo],
  dashboard: {
    ...
  },
},
```

4. 生成并校验：

```bash
python tools/grafana/generate_dashboards.py
python tools/grafana/generate_dashboards.py --check
```

新增一个普通的 engine 或 dashboard **不需要**修改 `composer.libsonnet`、`viz.libsonnet`、`framework/generate.py` 或 `dashboards.jsonnet`。

## 扩展现有内容

一个 dashboard 可以复用某个 base 模块并添加额外内容。

例如：

```text
trainer + trainer_extra
```

### 新增一个 row

如果扩展添加的是一个全新的 section，请用 `rows` 定义它。

```jsonnet
{
  panels: [
    ...
  ],
  rows: {
    'custom training metric': {
      ...
    },
  },
}
```

### 向现有 row 添加一个 panel

如果新 panel 应当出现在现有的 `training metric` row 内部，请使用 `rowItems`。

```jsonnet
{
  panels: [{
    key: 'training.custom.foo',
    outputKey: 'panel-custom-foo',
    id: 500,
    title: 'Custom Foo Metric',
    queries: [{
      expr: 'custom_foo_metric',
    }],
  }],

  rowItems: {
    'training metric': [{
      kind: 'GridLayoutItem',
      spec: {
        x: 0,
        y: 100,
        width: 12,
        height: 8,
        element: {
          kind: 'ElementReference',
          name: 'training.custom.foo',
        },
      },
    }],
  },
}
```

然后组合这两个模块：

```jsonnet
modules: [
  trainer,
  trainer_extra,
  controller,
  storage,
  trajectory,
]
```

扩展只做增量添加。它们不会静默覆盖现有的 panel、row 或 variable。

## 我应该修改什么？

| 任务                                 | 通常需要修改的文件                                                |
| ------------------------------------ | ----------------------------------------------------------------- |
| 使用现有模块的新 dashboard           | `dashboard_compositions.libsonnet`                                |
| 新 engine / 新内容                   | 新的 `dashboards/*.libsonnet` + `dashboard_compositions.libsonnet` |
| 扩展现有内容                         | 新的扩展模块 + `dashboard_compositions.libsonnet`                 |
| 新的共享可视化类型                   | framework 变更                                                    |
| 新的 composition 行为                | framework 变更                                                    |

对于普通的 dashboard 新增，不要修改 framework。

## 生成与校验

生成所有已注册的生产 dashboard：

```bash
python tools/grafana/generate_dashboards.py
```

校验已提交的 JSON 与 Jsonnet 源文件一致：

```bash
python tools/grafana/generate_dashboards.py --check
```

生产环境的 `--check` 比较的是解析后的 JSON 对象，因此仅由序列化导致的 key 顺序差异会被忽略。

结构性迁移检查也可在以下文件中使用：

```text
tools/grafana/dashboards/verify_modules.jsonnet
```

## 限制

- composition 只做增量添加；不支持隐式覆盖。
- `rowItems` 只会向现有的、受支持的 `GridLayout` row 追加条目。
- 无法通过 `rowItems` 删除或重排现有的 row 条目。
- 生成的生产 JSON 不应手工维护；应改为更新源模块或 composition 并重新生成。

关于通用的组合规则和 framework 内部实现，参见 [`framework/README.md`](framework/README.md)。
