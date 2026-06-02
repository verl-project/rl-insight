# RL-Insight: Provide performance insight capabilities for RL frameworks.
<div align="center">

[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-blue)](https://deepwiki.com/verl-project/rl-insight)
[![GitHub Repo stars](https://img.shields.io/github/stars/verl-project/rl-insight)](https://github.com/verl-project/rl-insight/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://rl-insight.readthedocs.io/en/latest/)

</div>

RL-Insight provides performance insight capabilities for RL training frameworks. It defines a [general pipeline](https://github.com/verl-project/rl-insight/blob/main/docs/overview/architecture.md) for performance insights. A series of capabilities will be built based on this framework. With a well-defined data protocol, these capabilities can generalize across training frameworks.

<div align="center">
 <img src="https://raw.githubusercontent.com/verl-project/rl-insight/main/assets/rl_insight_framework.svg" width="600" alt="rl-insight-arch">
</div>

## Key Features

**Offline Analysis**
- **Timeline visualization** — interactive HTML Gantt charts for per-rank event timelines across RL training phases, with parallel multi-rank parsing for MSTX, Torch Profiler, and NVTX data sources. PNG export also supported.
- **MoE Expert Load Heatmap** — GMM-clustered heatmaps to visualize expert load distribution in Mixture-of-Experts models, helping identify load imbalance across experts and layers.

**Online Monitoring (Experimental)**
- Real-time observability stack based on **Prometheus + Tempo + Grafana**
- Training-side Python APIs: counter, gauge, histogram metrics plus distributed tracing (`trace_state`, `trace_op`)

## Installation

Python >= 3.10 required.

```bash
pip install rl-insight
```

For the latest unreleased features, install from source:

```bash
git clone https://github.com/verl-project/rl-insight.git
cd rl-insight
pip install -r requirements.txt
pip install -e .
```

## Quickstart

### Timeline Visualization

Parse MSTX, Torch Profiler, or NVTX data and generate an interactive HTML timeline:

```bash
# MSTX
python -m rl_insight.main \
    input.path=<profiling_data_path> \
    timeline.parser.type=mstx \
    output.path=<output_path>

# Torch Profiler
python -m rl_insight.main \
    input.path=<torch_data_path> \
    timeline.parser.type=torch \
    output.path=<output_path>

# NVTX
python -m rl_insight.main \
    input.path=<nvtx_data_path> \
    timeline.parser.type=nvtx \
    output.path=<output_path>
```

Switch visualizer type for PNG output:

```bash
timeline.visualizer.type=html    # interactive timeline (default)
timeline.visualizer.type=png     # static PNG export
```

Convenience scripts are available in `examples/`:

```bash
bash examples/mstx_exec.sh
bash examples/torch_profiler_exec.sh
bash examples/nvtx_exec.sh
```

### MoE Expert Load Heatmap

Visualize expert load distribution in Mixture-of-Experts models:

```bash
bash examples/gmm_exec.sh
```

Or with full CLI control:

```bash
python -m rl_insight.main \
    input.path=<gmm_data_path> \
    output.path=<output_path> \
    heatmap.parser.type=gmm \
    heatmap.visualizer.type=gmm_heatmap \
    heatmap.visualizer.gmm_per_layer=3
```

### Online Monitoring (Experimental)

Start the observability stack and instrument training code:

```bash
rl-insight server start
```

```python
import rl_insight as insight

insight.init()
insight.metric_count("train_step_total", amount=1)
insight.metric_value("reward_mean", value=1.23)

with insight.trace_state("rollout", state_lane_id="trainer_0"):
    run_rollout()
```

See [`experimental/README.md`](experimental/README.md) for full API reference and configuration.

## Roadmap

- Q1 Roadmap https://github.com/verl-project/rl-insight/issues/6
- Q2 Roadmap https://github.com/verl-project/rl-insight/issues/49

## Documentation

- [Architecture & Design](https://github.com/verl-project/rl-insight/blob/main/docs/overview/architecture.md)
- [Offline Timeline Quickstart](https://github.com/verl-project/rl-insight/blob/main/docs/overview/RL_Timeline_quickstart.md)
- [GMM Heatmap Quickstart](https://github.com/verl-project/rl-insight/blob/main/docs/overview/gmm_heatmap_quickstart.md)
- [Memory Parser Guide](https://github.com/verl-project/rl-insight/blob/main/docs/developer_guides/memory_parser_guide.md)
- [Extension Guide](https://github.com/verl-project/rl-insight/blob/main/docs/developer_guides/extending_guide.md)

## Contribution Guide

See [CONTRIBUTING.md](CONTRIBUTING.md).
