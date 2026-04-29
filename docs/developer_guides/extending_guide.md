## 扩展指南

### 1. 扩展 **DataRule**

适用于：`InputData` / `OutputData` 的数据类型需要扩展，解析数据语义或字段发生变化，需要新的类型标识与 `ValidationRule`。

1. 在 `DataEnum` 中增加新值（字符串建议与 CLI 一致）。
2. 在 `DataChecker.rules` 中为新 `DataEnum` 挂载 `ValidationRule` 子类（`rules.py`），实现 `check()` / `error_message`。
3. 将能消费该数据的 **Parser** / **Visualizer** 的类属性 `input_type` 设为对应 `DataEnum`。
4. 在 `docs/data/data_specification.md` 中补充数据形态说明。

### 2. 扩展 **Parser** / **Visualizer**

适用于：在仍使用 **OfflineInsightPipeline** 的前提下，新增一种解析后端或一种可视化输出。

**Parser**

1. 新增模块，例如 `rl_insight/parser/my_parser.py`。
2. 继承 `BaseClusterParser`，实现 `run()` 方法。
3. `@register_cluster_parser("<name>")`，保证 `get_cluster_parser_cls("<name>")` 可用。
4. 更新 `main.py` 中 `--profiler-type` 的 help 与相关用户文档。

**Visualizer**

1. 新增模块，例如 `rl_insight/visualizer/my_visualizer.py`。
2. 继承 `BaseVisualizer`，实现 `run()` 方法。
3. `@register_cluster_visualizer("<name>")`，保证 `get_cluster_visualizer_cls("<name>")` 可用。
4. 更新 `main.py` 中 `--vis-type` 的 help 与相关用户文档。

若输入或中间数据形态变化，需同步按上一节扩展 **DataRule**。

### 3. 扩展 **Pipeline**

适用于：全新的处理范式（跳过步骤、插入预处理、多产物、在线多进程流程等）。

1. 在 `rl_insight/pipeline/` 新增类，实现 `__init__(self, config)`、`run(self)`，按需组合 `DataChecker`、`get_cluster_parser_cls`、`get_cluster_visualizer_cls` 等。
2. 在 `main.py` 的 `SUPPORTED_PIPELINE_TYPES` 中注册，例如 `{"MyPipeline": MyPipeline}`。
3. 更新 `--pipeline-type` 的 help，名称与 dict key 一致，并更新文档。
4. 若数据解析或数据类型发生变化，同步扩展 **DataRule** / **Parser** / **Visualizer**。
