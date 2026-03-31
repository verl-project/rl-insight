# RL-Insight - 数据文件目录结构

## 一、采集Torch Profiling 数据目录结构

```
<profile-data-path>/
└── <role>/
    └── prof_*.json.gz
```

数据解析文件 prof_*.json.gz，解析文件内容包含distrubutedInfo、traceEvent等字段，数据内容一般包含ts、dur等字段，解析文件内容示例：

```
{
    "schemaVersion": 1,
    "deviceProperties": [
    {
      "id": 0, "name": "NVIDIA L20", "totalGlobalMem": 47677177856,
      "computeMajor": 8, "computeMinor": 9,
      "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 1536,
      "regsPerBlock": 65536, "warpSize": 32,
      "sharedMemPerBlock": 49152, "numSms": 92
    , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 101376, "sharedMemPerMultiprocessor": 102400
    }
    ],
      "cupti_version": 26,
      "cuda_runtime_version": 12080,
      "cuda_driver_version": 12080,
      "distributedInfo": {"backend": "cpu:gloo,cuda:nccl", "rank": 0, "world_size": 2, "pg_count": 9, "pg_config": [{"pg_name": "0", "pg_desc": "default_pg", "backend_config": "cpu:gloo,cuda:nccl", "pg_size": 4, "ranks": [0, 1, 2, 3]}, {"pg_name": "1", "pg_desc": "mesh_dp", "backend_config": "cpu:gloo,cuda:nccl", "pg_size": 2, "ranks": [0, 2]}, {"pg_name": "3", "pg_desc": "mesh_infer_tp", "backend_config": "cpu:gloo,cuda:nccl", "pg_size": 2, "ranks": [0, 1]}, {"pg_name": "5", "pg_desc": "mesh_infer_pp", "backend_config": "cpu:gloo,cuda:nccl", "pg_size": 1, "ranks": [0]}]},
      "trace_id": "B45DDD976E4D4DDF8E3CFB28A0E2EF25",
  "displayTimeUnit": "ms",
  "baseTimeNanoseconds": 1767189312000000000,
  "traceEvents": [
  {
    "ph": "X", "cat": "cuda_runtime", "name": "cudaMemGetInfo", "pid": 369418, "tid": 1722878400,
    "ts": 4541015316353.111, "dur": 10083720.552,
    "args": {
            "cbid": 30, "correlation": 5
    }
  },
  {
    "name": "process_name", "ph": "M", "ts": 4541015315505.900, "pid": 369418, "tid": 0,
    "args": {
      "name": "ray::WorkerDict.actor_rollout_compute_log_prob"
    }
  },
  {
    "name": "Iteration Start: PyTorch Profiler", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace PyTorch Profiler", "ts": 4541015315462.515
  },
  {
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": 4541019157986.427
  }
  ],
  "traceName": "/tmp/tmpx5qz1t66.json"
}
```

## 二、采集Mstx Profiling 数据目录结构

```
<profile-data-path>/
└── <role>/
    └── *_ascend_pt/
        |── profiler_info_*.json
        └── ASCEND_PROFILER_OUTPUT/
            └── trace_view.json
```

数据解析文件 trace_view.json，解析文件内容必须包含"ph": "M"，且"name": "Overlap Analysis"对应"pid"的数据，该数据一般包含ts、dur等字段，解析文件内容示例：

```
[
  {
    "name": "process_name",
    "pid": 3550586784,
    "tid": 0,
    "ph": "M",
    "args": {
      "name": "Overlap Analysis"
    }
  },
  {
    "name": "Computing",
    "pid": 3550586784,
    "tid": 2,
    "ts": "1773285899055563.748",
    "dur": 53.301,
    "ph": "X",
    "args": {}
  },
]
```

## 三、VeRL 训练日志（可选校验）

`DataEnum.VERL_LOG` 用于对 **单个** VeRL 训练 `.log` 文件做存在性与关键指标子串校验（通过 `DataChecker` 或 `tests/data/check_verl_log.py`）。路径必须是文件，不能是目录。

约定：

- 文件扩展名为 `.log`，且非空。
- 能被识别为 VeRL 日志：文件名中包含 `verl`（不区分大小写），或文件开头约 64KiB 内容中出现 `verl`。
- 日志正文（不区分大小写的子串匹配）需包含 `VerlLogKeyParamsRule.DEFAULT_REQUIRED_KEYWORDS` 中配置的指标名，例如 `critic/score/mean`、`actor/loss` 等；可按项目日志格式在代码中调整该元组。

示例日志放在 **`rl-insight/data/verl_sample/`** 下，例如：

- `data/verl_sample/sample_verl_train.log` — 最小可过校验示例
- `data/verl_sample/good_full_verl.log` — 完整风格日志
- `data/verl_sample/bad_*.log` — 用于手工验证失败场景

在 `rl-insight` 包根目录下手动校验示例：

```bash
python tests/data/check_verl_log.py data/verl_sample/sample_verl_train.log
```
