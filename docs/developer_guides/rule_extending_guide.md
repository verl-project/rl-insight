# DataRule 扩展说明

DataRule 主要用于定义某一类数据的校验逻辑，开发者如果需要为新的数据类型增加校验逻辑，通常需要完成两步：

1. 继承 `ValidationRule` 并实现自定义 `check()` 函数
2. 在 `DataChecker.rules` 中将数据类型与对应 Rule 进行注册

## 1. 实现自定义 DataRule

### 1.1. `ValidationRule`

自定义数据校验规则需要继承 `ValidationRule`，该基类定义在 `rl_insight/data/rules.py`。

`ValidationRule` 是所有数据校验规则的抽象基类，核心包括三个部分：

- 初始化时定义 `_error_message`，用于保存当前 Rule 的失败原因
- 定义抽象方法 `check(data) -> bool`，要求子类必须实现具体校验逻辑
- 通过 `error_message` 属性向外暴露失败原因，供 `DataChecker` 收集

### 1.2 `check()`
`check(data)` 是每条自定义 DataRule 继承 `ValidationRule`后必须实现的核心方法

- 输入`data`： 是当前待校验的数据对象，由 `DataChecker` 在运行时传入
- 输出：返回 `True` 表示校验通过，返回 `False` 表示校验失败
    - 如果校验失败，应在 Rule 内部设置 `self._error_message`
    - `error_message` 属性用于让 `DataChecker` 收集失败原因

## 2. 注册自定义DataRule

自定义 Rule 实现完成后，需要在 `DataChecker` 中完成注册。注册位置位于 `rl_insight/data/data_checker.py`。

`DataChecker` 中通过 `DataEnum` 表示不同的数据类型，通过 `rules` 字典维护数据类型和 Rule 列表之间的映射关系。

### 2.1. 注册新增数据类型

在 `DataEnum` 中增加新的类型名称，新增 `DataEnum.CUSTOM_DATA`

### 2.2. 注册数据类型和 DataRule 映射关系

新增数据类型后，需要在 `DataChecker.rules` 中将该数据类型和自定义 Rule 绑定起来。

例如，如果新增了一个自定义数据类型 `CUSTOM_DATA`，并实现了一个自定义规则 `CustomDataRule`，则需要在 `rules` 映射表中增加类似关系：

- `DataEnum.CUSTOM_DATA`
  - `CustomDataRule()`

如果一个数据类型需要多个校验步骤，也可以注册多个 Rule。例如：

- `CustomPathExistsRule()`
- `CustomFileExistsRule()`
- `CustomFieldValidRule()`

这样 `DataChecker` 会按照列表顺序依次执行它们。
