# BaseClusterParser 接口说明

`BaseClusterParser` 是所有集群级 Parser 的统一抽象基类，可以基于该基类新增一种解析后端类(参考 `MstxClusterParser`)

## 1. `run()` 执行入口

`BaseClusterParser.run()` 定义了 Parser 的统一执行流程。所有子类只要复用该基类，就会走同一套主流程：

1. `allocate_prof_data(input_data)`：将原始输入路径转换为 `DataMap` 列表。

2. `mapper_func(_data_maps)`：调度 `DataMap` 列表对应的解析任务。

3. `parse_analysis_data(profiler_data_path, rank_id, role)`：由具体 Parser 子类实现，负责单个 rank 对应的 profiler 解析。

4. `reducer_func(mapper_res)`：将所有 rank 返回的事件列表转换为统一的 `DataFrame`。

5. `get_data()`： 聚合多个 rank 结果后保存为 `self.events_summary`，最终的 Parser 输出结果。

## 2. `allocate_prof_data() `

### 2.1. **`allocate_prof_data()`的实现：**
- 接收原始输入路径
- 扫描目录
- 找到每个 rank 对应的数据位置
- 组织成 DataMap 列表

### 2.2. **输入&输出**
- 输入`input_data`：实际为 profile 文件存放的根路径
  - `self.parser.run(self.config.input_path)`
  - 参考:[`Torch Profiler目录结构中的profile-data-path层级`](../data/data_specification.md)
- 输出`DataMap`，需包含以下字段
  - rank_id：当前 profiling 数据对应的 rank 编号
  - role：当前 rank 对应的任务角色，例如 rollout、actor、critic 等
  - profiler_data_path：当前 rank 对应的实际 profiling 文件存放的根路径

## 3. `parse_analysis_data()`

### 3.1. `parse_analysis_data()` 的实现

- 接收单个 rank 对应的 profiler 数据文件路径
- 读取该文件中的 profiling 数据
- 根据当前 Parser 的数据格式，提取需要的事件信息
- 将事件时间统一转换为毫秒
- 组织成 `EventRow` 列表返回

### 3.2. 输入 & 输出

- 输入 `profiler_data_path`：当前 rank 对应的实际 profiler 文件存放的根路径，取自`DataMap`。
  - 参考:[`Torch Profiler目录结构中的profile-data-path层级`](../data/data_specification.md)

- 输入 `rank_id`： 当前 profiling 数据对应的 rank 编号，取自`DataMap`。

- 输入 `role`： 当前 rank 对应的任务角色，用于标识解析出的事件属于哪个 RL 任务阶段，例如生成/计算/更新 Actor 等，取自`DataMap`。

- 输出 `list[EventRow]`：当前 rank 解析得到的事件列表，每个 `EventRow` 至少应包含以下字段：
    - `name`：事件名称
    - `role`：任务角色
    - `domain`：事件类别或分组
    - `start_time_ms`：事件开始时间，单位为毫秒
    - `end_time_ms`：事件结束时间，单位为毫秒
    - `duration_ms`：事件持续时间，单位为毫秒
    - `rank_id`：事件所属 rank
    - `tid`：线程 ID 或进程 ID
