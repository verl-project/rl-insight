# Insight Framework

<div align="center">
 <img src="https://github.com/tardis-key/rl-insight/blob/main/assets/rl_insight_framework.svg" width="600" alt="rl-insight-arch.png">
</div>

The green data modules and blue function modules shown in the diagram will be included in RL-Insight. The red modules (CollectController/Collector) has already been implemented in [verl.DistProfiler](https://verl.readthedocs.io/en/latest/perf/verl_profiler_system.html) and will not be integrated into RL-Insight for now.
- InputData: Defines the input structure requirements that the offline data pipeline can process. It does not require mandatory dependencies on specific RL frameworks and can also be obtained through other frameworks or manual processing.
- OutputData: The output results of all analytical capabilities in RL-Insight. They are human-readable (we will do our best) and can be used for further visualization via RL-Insight.
- Offline/Online Parser: Parses raw data in various formats to complete data analysis for corresponding scenarios.
- Plugin: In online monitor, plugins enable secondary development of 3rd-party monitoring tools to implement data reception, visualization, export, and other functions.
- Metric: Provides advanced capabilities such as metric calculation.
- Collector: Provides data collecting capabilities across different platforms and tools.
- CollectController: Controls when and what data the Collector collect at appropriate times.