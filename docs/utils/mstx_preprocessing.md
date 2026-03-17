# mstx_preprocessing 使用文档

## 功能说明

`utils/mstx_preprocessing.py` 是一个 MSTX Profiling 离线解析脚本，用于在调用 main 之前对 NPU profiling 数据进行离线数据解析，从而生成 mstx_parser.py 需处理的目标文件 `trace_view.json`

它的作用是：

- 接收一个 profiling 数据根目录路径
- 对根目录下的所有文件夹调用 `torch_npu.profiler.profiler.analyse`，执行离线解析
- 解析失败时输出错误日志

## 脚本位置

```bash
rl_insight/utils/mstx_preprocessing.py
```

## 使用方法

```bash
python -m rl_insight.utils.mstx_preprocessing <profile-data-path>
```

## 参数说明

脚本接收 1 个位置参数：

| 参数 | 说明 |
|------|------|
| `profile-data-path` | profiling 数据根目录路径 |

## 目录结构示意

`profile-data-path` 下的目录层级如下：

```text
<profile-data-path>/
└── <role>/
    └── *_ascend_pt/
        └── ASCEND_PROFILER_OUTPUT/
```

## 依赖要求

运行脚本前，需要满足以下条件：

- Python 环境可用
- 已安装 `torch_npu`
- 输入路径存在，且目录内容符合 Ascend profiler 离线解析要求

## 注意事项

- `profile-data-path` 目录下需要包含 `<role>/*_ascend_pt/ASCEND_PROFILER_OUTPUT` 这一层级
- 如果解析失败，脚本会打印错误日志
