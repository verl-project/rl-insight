# 进程内硬件监控

"进程内"指硬件指标由训练进程自身内部的后台监控采集，不需要独立的 exporter 进程或额外端口：调用 `rl_insight.init()` 后，监控器（Ray 下每个节点一个，非 Ray 下为调用进程内的线程）会周期性采样节点硬件，并通过 RL-Insight 现有的指标链路上报，无需安装任何 exporter。

## 1. 前置条件

系统采集器（CPU、内存、网络、磁盘）依赖 psutil，它是 RL-Insight 的内置依赖。厂商加速器为可选项：

| 加速器     | 要求                               | 说明                             |
| ---------- | ---------------------------------- | -------------------------------- |
| NVIDIA GPU | `pip install rl-insight[nvidia]` | 可选依赖组，安装`nvidia-ml-py` |
| 昇腾 NPU   | `npu-smi` 位于 `PATH`          | 随昇腾驱动提供                   |

缺少厂商工具不会导致训练失败：对应采集器通过状态指标上报为不可用，其余采集器正常工作。

## 2. 开启硬件监控

硬件监控默认开启，通过 `rl_insight.init()` 的 `hardware` 配置段控制：

```python
import rl_insight

rl_insight.init(
    project="verl",
    experiment_name="ppo-run",
    config={
        "server": {"url": "http://<SERVER_IP>:18080"},
        "hardware": {
            "enabled": True,                # 默认值；设为 False 关闭
            "interval_seconds": 10,         # 采样间隔，最小 5 秒
            "collectors": [                 # 启用哪些采集器
                "cpu", "memory", "network", "disk", "gpu", "npu",
                "ray_actor_process",
            ],
            "disk_path": "/",               # 磁盘采集器所测量的文件系统路径
            "worker_role": "trainer",       # 可选；上报进程的角色标签
        },
    },
)
```

说明：

- Ray 作业中，`init()` 会在每个存活节点上创建一个硬件 actor，即使多个 worker 进程各自调用 `init()`，主机级指标每个节点也只采集一份。非 Ray 环境下，监控以线程方式运行在调用进程内。节点 actor 与进程内线程都按 `server.backend` 选择上报通道：`ray` 发往 job 级 Hub，`push` 直接写入配置的 sink。
- `self_process` 采集器已注册但不在默认列表中；当 worker 进程自行调用 `init()` 且需要上报自身 CPU/内存占用时，手动将其加入 `collectors`。它采集的是运行监控的进程本身——Ray 下即节点级硬件 actor，而非某个训练 worker；逐 worker 视图请用 `ray_actor_process`。
- `ray_actor_process` 采集器默认启用：它会逐进程上报本节点上属于当前 Ray job 的每个 Ray actor 的 CPU/线程/内存占用（指标族为 `rlinsight_ray_process_*`），并附带 `pid` 与自动推断的 `worker_role`。Ray State API 的 `list_actors` 是集群级查询，这里按当前 `job_id` 过滤，因此多个作业共享同一节点时不会把其他作业的 worker 计入本作业，也不会污染 `worker_role` / `device_index` 下拉选项。角色来自 actor 的类名：非 colocated 的 `ActorRolloutRefWorker`/`CriticWorker` 与 `FusedWorker_<roles>` 会映射成对应角色；而 verl 默认的 colocated 模式下所有训练 worker 共用一个匿名 `WorkerDict` 类、不含角色信息，会统一上报为 `worker`，此时需借助下述 `local_rank`/`device_index` 区分到具体卡。该采集器依赖 Ray State API，非 Ray 作业下会自动上报为不可用，因此保留在默认列表中不会造成噪声。
- `ray_actor_process` 还会尽量补上两个可与其他面板联动的标签：`local_rank` 取自 actor 名后缀 `<类名>_<pg_idx>:<local_rank>`，即该进程在本节点内的设备序号（verl 用它做 placement group bundle 下标）；`device_index` 从进程环境读取 `ASCEND_RT_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`，与 NPU/GPU 面板使用同一卡号口径，因此可以在同一个下拉里同时筛卡与进程。标签值为多卡列表、取不到环境或解析失败时，该标签上报为空字符串而非省略——同一指标族的标签维度必须固定，否则会让 hub 的 Prometheus 注册表在遇到不同标签集的 worker 时抛 `DuplicateTimeseries` 并丢掉整批样本。rl-insight 自身的节点级 actor（`MonitorHubActor`、`NodeHardwareActor`）会被跳过，避免把监控自身开销当成训练 worker。
- `worker_role` 是**进程级**标签：描述的是上报样本的那个进程，而不是节点。进程级指标族（`rlinsight_ray_process_*` / `rlinsight_process_*`）始终携带该标签键，未配置或推断不出时其值为空字符串；环境变量 `RL_INSIGHT_WORKER_ROLE` 的优先级高于配置项。使用 `ray_actor_process` 时该标签由 Ray actor 类名自动推断，只有 `self_process` 采集器才需要手动设置。
- `interval_seconds` 默认 10 秒，这是一个偏保守的取值，兼顾采样开销与曲线可用性；低于 5 时会触发 `RuntimeWarning` 并被钳制为 5。每一轮都会运行 psutil 采集器，并对每张卡发起一次设备查询；NPU 机器上 `npu-smi` 每轮都会拉起一个子进程（超时 5 秒）——大规模集群可适当调大间隔，仅在需要更细粒度时再调小。
- `disk_path` 指定要上报利用率的文件系统（默认 `/`）；当训练数据或 checkpoint 所在挂载点是压力来源时，指向该路径。
- `RL_INSIGHT_HARDWARE_ENABLED` 可覆盖 `hardware.enabled`（`1/0`、`true/false`、`yes/no`、`on/off`，大小写不敏感），`RL_INSIGHT_WORKER_ROLE` 可覆盖 `hardware.worker_role`。这两个变量只在**实际调用 `init()` 的进程**中读取，解析结果随后作为 node actor 的构造参数下发；node actor 不会再次读取它们。
  - **非 Ray 模式**：每个进程读取自己的环境，因此逐进程开关和角色是有效的。
  - **Ray 模式**：配置是 job 级的。第一个调用 `init()` 并创建 node actor 的进程决定 `enabled`、`interval_seconds`、`collectors` 以及 `self_process` 的 `worker_role`；之后其他 worker 的 `init()` 只会复用已存在的 actor，它们的同名环境变量不会重新配置这些 actor。要关闭整个作业的硬件监控，请在**首个调用 `init()` 的进程**（通常是 driver）或 job 级 `runtime_env.env_vars` 中设置，**不需要在每个训练节点上分别设置**；只在部分训练 worker 上设置 `RL_INSIGHT_HARDWARE_ENABLED=0` 不会停掉已经创建的 node actor。
  - Ray 模式下 `RL_INSIGHT_WORKER_ROLE` 只影响 opt-in 的 `self_process` 采集器（它采集的是 node actor 自身）；逐训练 worker 的角色由 `ray_actor_process` 从 actor 类名自动推断，不需要也不应通过每个 worker 的环境变量设置。
  - 只有 RL-Insight server 地址会通过 hub actor 的 `runtime_env` 自动转发，其余配置通过合并后的 config 和 actor 构造参数传递。

## 3. 指标

所有指标均为 Prometheus gauge，使用 `rlinsight_` 前缀命名（实际暴露时带有 RL-Insight 指标命名空间，如 `rl_insight_monitor_rlinsight_cpu_utilization_percent`）。速率类指标由单调计数器差分计算，按每秒上报。

| 分类       | 指标                                                                                                                                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CPU        | `rlinsight_cpu_utilization_percent`、`rlinsight_cpu_logical_count`、`rlinsight_cpu_physical_count`、`rlinsight_cpu_cgroup_throttled_periods`、`rlinsight_cpu_cgroup_throttled_seconds`                                                             |
| 内存       | `rlinsight_memory_used_bytes`、`rlinsight_memory_available_bytes`、`rlinsight_memory_utilization_percent`、`rlinsight_memory_cgroup_limit_bytes`、`rlinsight_memory_cgroup_usage_bytes`                                                            |
| 网络       | `rlinsight_network_receive_bytes_per_second`、`rlinsight_network_transmit_bytes_per_second`                                                                                                                                                              |
| 磁盘       | `rlinsight_disk_read_bytes_per_second`、`rlinsight_disk_write_bytes_per_second`、`rlinsight_disk_utilization_percent`                                                                                                                                  |
| Ray actor 进程 | `rlinsight_ray_process_cpu_percent`、`rlinsight_ray_process_thread_count`、`rlinsight_ray_process_rss_bytes`                                                                                                                                            |
| 自监控进程 | `rlinsight_process_cpu_percent`、`rlinsight_process_thread_count`、`rlinsight_process_rss_bytes`、`rlinsight_process_rss_percent`                                                                                                                        |
| NVIDIA GPU | `rlinsight_gpu_utilization_percent`、`rlinsight_gpu_memory_used_bytes`、`rlinsight_gpu_memory_total_bytes`、`rlinsight_gpu_memory_utilization_percent`、`rlinsight_gpu_temperature_celsius`、`rlinsight_gpu_power_watts`、`rlinsight_gpu_info` |
| 昇腾 NPU   | `rlinsight_npu_utilization_percent`、`rlinsight_npu_memory_used_bytes`、`rlinsight_npu_memory_total_bytes`、`rlinsight_npu_memory_utilization_percent`、`rlinsight_npu_temperature_celsius`、`rlinsight_npu_power_watts`、`rlinsight_npu_info` |
| 诊断       | `rlinsight_hardware_collector_available`                                                                                                                                                                      |

稳定标签：`project`、`experiment_name`、`run_id`、`node`；设备类指标另有 `device_vendor`、`device_index`。两个进程级指标族的标签契约不同：

- `rlinsight_ray_process_*`（`ray_actor_process`，默认启用）：固定携带 `pid`、`worker_role`、`local_rank`、`device_index`。`worker_role` 由 Ray actor 类名推断，`local_rank` 取自 actor 名的 `<pg_idx>:<local_rank>` 后缀，`device_index` 取自该进程环境中的 `ASCEND_RT_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`；推断不出时为空字符串。
- `rlinsight_process_*`（`self_process`，默认不启用）：固定携带 `pid`、`worker_role`，**没有** `local_rank` / `device_index`。它描述的是运行监控的进程本身——Ray 模式下是 `NodeHardwareActor`，非 Ray 模式下是调用 `init()` 的进程——因此不存在 worker 意义上的 local rank，其环境也不代表训练 worker 的卡绑定；按 `device_index` 或 `local_rank` 过滤这一族不会得到数据。`worker_role` 需要手动配置或通过 `RL_INSIGHT_WORKER_ROLE` 设置。

其余系列均为节点级。`device_index` 在设备类指标上是物理卡号（NPU 为 `npu-smi` 编号，GPU 为 NVML 编号），在 `ray_actor_process` 上是该进程被分配到的同一物理卡号，因此这两类序列可以用同一个 `device_index` 值联合筛选。设备类指标按逻辑设备/芯片逐个上报：NPU 序列会收敛到 `ASCEND_RT_VISIBLE_DEVICES`，GPU 序列在 `CUDA_VISIBLE_DEVICES` 为纯整数索引时收敛到该列表，因此被限定到部分卡的作业只会上报自己使用的卡；非整数形式（PCI 总线号、GPU UUID、MIG id）无法映射到 NVML 索引，此时上报该节点全部 GPU。`*_info` 指标是恒为 1 的 gauge，设备名称与驱动版本通过标签携带。`rlinsight_disk_utilization_percent` 通过 `path` 标签携带所配置的 `disk_path`。cgroup 指标仅在运行时暴露容器限额时出现：`rlinsight_cpu_cgroup_throttled_*` 两个指标自容器启动起累计，如需速率请在 PromQL 中使用 `rate()`。

## 4. 在 Grafana 中查看

硬件指标直接显示在 trainer dashboard（如 `verl_trainer_v1_with_vllm_engine`）的 hardware metric 行，无需单独的 dashboard 或 exporter 部署。每个硬件面板的每条序列内置两条查询：进程内查询（默认可见）与 exporter 查询（默认隐藏），由 rl-insight 服务端在暂存 dashboard 时按指标族分别选择：CPU/内存/网络/磁盘面板只在注册了 `node-exporter` 时切到 exporter 查询，NPU 面板只在注册了 `npu-exporter` 时切换，GPU 面板只在注册了 `dcgm` 时切换（job 名匹配为大小写不敏感前缀，通过 `rl-insight add-targets` 注册）。未注册对应 exporter 的指标族继续显示进程内查询，因此只注册部分 exporter 不会让另一组面板变成 `No data`。面板不会用 PromQL `or` 混合两种来源，也不会在两种来源同时存在时画出重复曲线。设备序列 legend 沿用 exporter 风格（`NPU {{id}}` / `GPU {{id}}`，`device_index` 会被改写到 `id` 标签）。

CPU/内存/网络/磁盘等主机级面板标题统一带 `Node` 前缀，其数值覆盖整机所有进程，而非仅本次训练作业，共享节点上请勿据此归因单作业资源占用。

`Ray Actor Process CPU` / `Ray Actor Process Memory` 面板是进程级面板，legend 为 `{{node}} {{device_index}} {{worker_role}} {{pid}}`，覆盖节点上每个 Ray worker（由默认启用的 `ray_actor_process` 采集器提供）；`Self Process CPU` / `Self Process Memory` 面板则覆盖运行监控的进程本身，legend 为 `{{node}} {{worker_role}} {{pid}}`，仅在启用 `self_process` 后才有数据。顶部 `hardware: Worker Role` 变量按角色筛选，`hardware: Device` 变量按 `device_index` 筛选，并与 NPU/GPU in-process 面板共用同一条件——选中某张卡时，进程面板与设备面板会同步收敛到该卡（此时没有卡标签的 CPU 池类进程会被隐藏，选中 `All` 可恢复全量）。

多机环境下两种模式行为对齐：机器维度统一为 `node` 标签——进程内指标原生携带；exporter/DCGM 序列在查询中由 `instance` 地址合成（IPv6 target 的 `instance` 形如 `[2001:db8::1]:8082`，合成时会去掉方括号，与进程内不带方括号的地址口径一致，面板的 `instance` 过滤同样兼容带方括号的形式）。legend（`{{node}} NPU {{id}}`）在两种模式下渲染一致，多节点序列不会混淆。节点切换由顶部 `hardware: Node` 下拉完成（`node` 标签；当部署缺少进程内身份指标时，回退用 exporter 的 `instance` 主机段合成），选中 `All` 时汇总全部节点，多选则只显示选中的节点。`CPU / Host Metrics` 与 `Ascend NPU Metrics` 两行**不按节点重复生成 tab**：Grafana 的行级重复要求该变量在 dashboard 加载时即可解析出取值，而该变量取自瞬时查询（取数时刻 = 时间范围结束点），在已结束训练的时间窗内会解析为空，整行随之塌缩成一个空 tab、任何面板都不渲染；此外 1024 卡规模下逐节点 tab 也会爆炸。因此两行始终平铺渲染，节点过滤下沉到面板查询（`node=~"$node"`，变量为空时退化为匹配全部）。单机场景该变量只有一个取值，行为与改动前一致。硬件面板中只有 `Ray Actor Process` 两个面板的图例按 `Max` 降序排列，其余面板保持序列自然顺序——NPU/GPU 面板按 rank 排列（`NPU 0`、`NPU 1`…）。

`Ascend NPU Metrics` 与 `NVIDIA GPU Metrics` 两整行**始终渲染**。这两行原本靠隐藏变量探测厂商身份指标来自动显隐，但该探测随数据变旧解析为空（且变量只在 dashboard 加载时解析一次）：训练结束后整行连同其中的面板一起消失，把时间范围移回训练时段也不会恢复。现在某厂商在本节点没有任何数据时由其面板自行显示 `No data`——纯 NPU 机器上 NVIDIA GPU 行是一组空面板，反之亦然。

## 5. 故障排查

某采集器在 `rlinsight_hardware_collector_available` 中为 0：

- `gpu` 为 0：未安装 `nvidia-ml-py`（`pip install rl-insight[nvidia]`），或节点上无 NVIDIA 驱动/设备。
- `npu` 为 0：`npu-smi` 不在 `PATH`。训练容器内通常需要昇腾驱动；宿主机有该工具而容器缺失时，可手动挂载：

```bash
docker run ... -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi ...
```

- 采集器为 1 但对应面板为空：`rlinsight_hardware_collector_available` 反映的是采集器是否成功初始化其工具，而不是节点上是否有设备。`npu-smi` 存在但未报告任何卡（或某一轮查询失败）时，`npu` 仍为 1 而面板无数据；`gpu` 同理。
- 多节点 Ray 集群上主机级序列重复或缺失：检查各节点能否访问 RL-Insight server，以及节点列表是否健康（`ray.nodes()`）。
- 传输失败导致批次被丢弃时，上报侧会打印 `failed to forward ... hardware events; dropping them` 日志；由于承载丢弃计数的传输通道本身就是刚失败的那条，因此不再暴露 dropped 指标。出现该告警时请检查 RL-Insight server 状态。
