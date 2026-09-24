# Metric Name Catalog

This review file is generated from `recipe/perf_analysis/degradation/metrics.py`.
It records each exact metric name together with a Chinese description for review.

## 数据类型

- **Gauge**：瞬时量值，每次上报的是当前值（如时延、内存、loss、步编号等）。
- **Counter**：累计值，单调递增，通常需要 `rate()` / `increase()` 转换后才能横向比较。
- **Histogram**：直方图，用 `histogram_quantile()` 求分位。
- **统计值**：由同一批观测聚合出的统计量（mean / max / min / clip_ratio / p50 / p99 等）。
- **派生值**：由其他指标计算得到（如 per-token 时延、吞吐、MFU、相关性 / KL / 散度 / PPL 等）。

## Summary

| Metric role | Count |
|---|---:|
| Global-step ruler | 1 |
| Active scalar targets | 8 |
| Cataloged histogram targets (not modeled directly) | 7 |
| Active candidate names | 102 |
| Cataloged preprocessing candidates (not modeled directly) | 16 |
| Total catalog entries | 134 |

## Global-step ruler

| No. | Exact metric name | 数据类型 | 计算方法 | 中文说明 |
| ---: | --- | --- | --- | --- |
| 1 | rl_insight_monitor_training_global_step | Gauge | 实测 | 当前训练任务的全局训练步编号；用于按 step 对齐其他指标。 |

## Active scalar targets

| No. | Exact metric name | 数据类型 | 计算方法 | 中文说明 |
| ---: | --- | --- | --- | --- |
| 1 | rl_insight_monitor_timing_s_step | Gauge | 实测 | 一个完整训练 step 的总耗时（s）。 |
| 2 | rl_insight_monitor_timing_s_gen | Gauge | 实测 | Rollout 生成阶段的总耗时（s）。 |
| 3 | rl_insight_monitor_timing_s_ref | Gauge | 实测 | 参考模型计算响应 token 对数概率的总耗时（s）。 |
| 4 | rl_insight_monitor_timing_s_adv | Gauge | 实测 | 优势值计算阶段的总耗时（s）。 |
| 5 | rl_insight_monitor_timing_s_old_log_prob | Gauge | 实测 | Actor 旧策略对数概率计算阶段的总耗时（s）。 |
| 6 | rl_insight_monitor_timing_s_update_actor | Gauge | 实测 | Actor 参数更新阶段的总耗时（s）。 |
| 7 | rl_insight_monitor_timing_s_update_weights | Gauge | 实测 | 训练 Actor 向 Rollout 推理引擎同步权重的总耗时（s）。 |
| 8 | rl_insight_monitor_timing_s_testing | Gauge | 实测 | 验证或测试阶段的总耗时（s）。 |

## Cataloged histogram targets

These names are cataloged but require aggregation before direct modeling.

| No. | Exact metric name | 数据类型 | 计算方法 | 中文说明 |
| ---: | --- | --- | --- | --- |
| 1 | vllm:e2e_request_latency_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求从接收到完成的端到端时延直方图（s）。 |
| 2 | vllm:time_to_first_token_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求从接收到产生首个输出 token 的时延直方图（s）。 |
| 3 | vllm:request_time_per_output_token_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求在首个 token 后生成每个输出 token 的平均耗时直方图（s）。 |
| 4 | vllm:request_queue_time_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求在调度队列中的等待时长直方图（s）。 |
| 5 | vllm:request_prefill_time_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求的 Prefill 时长直方图（s）。 |
| 6 | vllm:request_decode_time_seconds | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求的 Decode 时长直方图（s）。 |
| 7 | tq_controller_request_duration_seconds | Histogram | 直方图，histogram_quantile 取分位 | TransferQueue Controller 按操作类型统计的请求处理时延直方图（s）。 |

## Active candidate names

| No. | Category | Exact metric name | 数据类型 | 计算方法 | 中文说明 |
| ---: | --- | --- | --- | --- | --- |
| 1 | latency | rl_insight_monitor_timing_per_token_ms_gen | 派生值 | timing_s_gen × 1000 / 生成 token 数 | Rollout 生成阶段的单 token 平均耗时（ms）。 |
| 2 | latency | rl_insight_monitor_timing_per_token_ms_ref | 派生值 | timing_s_ref × 1000 / 参考 token 数 | 参考模型对数概率计算的单 token 平均耗时（ms）。 |
| 3 | latency | rl_insight_monitor_timing_per_token_ms_adv | 派生值 | timing_s_adv × 1000 / 计算 token 数 | 优势值计算的单 token 平均耗时（ms）。 |
| 4 | latency | rl_insight_monitor_timing_per_token_ms_update_actor | 派生值 | timing_s_update_actor × 1000 / 更新 token 数 | Actor 参数更新的单 token 平均耗时（ms）。 |
| 5 | latency | rl_insight_monitor_perf_time_per_step | Gauge | 实测 | 当前训练 step 耗时（s）。 |
| 6 | latency | rl_insight_monitor_perf_throughput | 派生值 | 总 token / (step 耗时 × 设备数) | 按设备数归一化的训练吞吐量（token/s/device）。 |
| 7 | hardware_resources | rl_insight_monitor_perf_mfu_actor | 派生值 | 实际 FLOPs / 理论峰值 FLOPs | Actor 训练的模型浮点运算利用率，即实际计算吞吐相对理论峰值的比例。 |
| 8 | data_characteristics | rl_insight_monitor_perf_total_num_tokens | Gauge | 本 step 参与计算的 token 总数 | 当前训练 step 中参与计算的 token 总数。 |
| 9 | hardware_resources | rl_insight_monitor_actor_perf_cpu_memory_used_gb | Gauge | 实测（主机 CPU 内存） | Actor 所在主机已使用的 CPU 内存（GB）。 |
| 10 | hardware_resources | rl_insight_monitor_actor_perf_max_memory_allocated_gb | Gauge | PyTorch max_memory_allocated / 1024³ | Actor 运行期间 PyTorch 实际分配的设备内存峰值（GB）。 |
| 11 | hardware_resources | rl_insight_monitor_actor_perf_max_memory_reserved_gb | Gauge | PyTorch max_memory_reserved / 1024³ | Actor 运行期间 PyTorch 内存分配器保留的设备内存峰值（GB）。 |
| 12 | data_characteristics | rl_insight_monitor_prompt_length_mean | 统计值 | mean（批次内统计） | 当前批次 Prompt token 长度的平均值。 |
| 13 | data_characteristics | rl_insight_monitor_prompt_length_max | 统计值 | max（批次内统计） | 当前批次 Prompt token 长度的最大值。 |
| 14 | data_characteristics | rl_insight_monitor_prompt_length_min | 统计值 | min（批次内统计） | 当前批次 Prompt token 长度的最小值。 |
| 15 | data_characteristics | rl_insight_monitor_prompt_length_clip_ratio | 统计值 | 达到上限样本数 / 总样本数 | 当前批次中 Prompt 长度达到配置上限的样本比例。 |
| 16 | data_characteristics | rl_insight_monitor_response_length_mean | 统计值 | mean（批次内统计） | 当前批次生成响应 token 长度的平均值。 |
| 17 | data_characteristics | rl_insight_monitor_response_length_max | 统计值 | max（批次内统计） | 当前批次生成响应 token 长度的最大值。 |
| 18 | data_characteristics | rl_insight_monitor_response_length_min | 统计值 | min（批次内统计） | 当前批次生成响应 token 长度的最小值。 |
| 19 | data_characteristics | rl_insight_monitor_response_length_clip_ratio | 统计值 | 达到上限样本数 / 总样本数 | 当前批次中响应长度达到最大生成长度的样本比例。 |
| 20 | data_characteristics | rl_insight_monitor_response_length_non_aborted_mean | 统计值 | mean（批次内统计） | 排除中止样本后，响应 token 长度的平均值。 |
| 21 | data_characteristics | rl_insight_monitor_response_length_non_aborted_max | 统计值 | max（批次内统计） | 排除中止样本后，响应 token 长度的最大值。 |
| 22 | data_characteristics | rl_insight_monitor_response_length_non_aborted_min | 统计值 | min（批次内统计） | 排除中止样本后，响应 token 长度的最小值。 |
| 23 | data_characteristics | rl_insight_monitor_response_length_non_aborted_clip_ratio | 统计值 | 达到上限样本数 / 总样本数 | 排除中止样本后，响应长度达到最大生成长度的样本比例。 |
| 24 | rollout_quality | rl_insight_monitor_response_aborted_ratio | 统计值 | 比例统计 | 当前批次中 Rollout 响应被中止、响应长度记为零的样本比例。 |
| 25 | data_characteristics | rl_insight_monitor_global_seqlen_mean | 统计值 | mean（批次内统计） | 数据并行负载均衡前，各分区序列总长度的平均值。 |
| 26 | data_characteristics | rl_insight_monitor_global_seqlen_max | 统计值 | max（批次内统计） | 数据并行负载均衡前，各分区序列总长度的最大值。 |
| 27 | data_characteristics | rl_insight_monitor_global_seqlen_min | 统计值 | min（批次内统计） | 数据并行负载均衡前，各分区序列总长度的最小值。 |
| 28 | data_characteristics | rl_insight_monitor_global_seqlen_minmax_diff | 派生值 | max(各分区序列总长度) − min(各分区序列总长度) | 数据并行负载均衡前，各分区序列总长度最大值与最小值之差。 |
| 29 | data_characteristics | rl_insight_monitor_global_seqlen_balanced_max | 统计值 | max（批次内统计） | 数据并行负载均衡后，各分区序列总长度的最大值。 |
| 30 | data_characteristics | rl_insight_monitor_global_seqlen_balanced_min | 统计值 | min（批次内统计） | 数据并行负载均衡后，各分区序列总长度的最小值。 |
| 31 | training_quality | rl_insight_monitor_actor_loss | Gauge | policy loss + KL 项 + 熵项等加权组合 | Actor 优化使用的总损失；由策略损失及配置启用的 KL、熵等项组合而成。 |
| 32 | training_quality | rl_insight_monitor_actor_pg_loss | Gauge | PPO 裁剪后的策略梯度损失 | PPO 裁剪后的策略梯度损失。 |
| 33 | training_quality | rl_insight_monitor_actor_kl_loss | Gauge | KL(π_actor ‖ π_ref) | 当前 Actor 策略与参考策略之间的 KL 损失。 |
| 34 | training_quality | rl_insight_monitor_actor_entropy_loss | Gauge | 熵正则项（含系数与符号） | Actor 目标函数中使用的策略熵聚合项；其符号和权重由熵系数的组合方式决定。 |
| 35 | training_quality | rl_insight_monitor_actor_entropy | Gauge | mean(有效响应 token 上的策略熵) | Actor 策略在有效响应 token 上的平均熵。 |
| 36 | training_quality | rl_insight_monitor_actor_ppo_kl | Gauge | KL(π_current ‖ π_old) 近似 | 当前策略与生成数据时旧策略之间的 PPO KL 近似统计值。 |
| 37 | training_quality | rl_insight_monitor_actor_kl_coef | Gauge | 配置值（KL 系数） | Actor 损失中 KL 正则项当前使用的系数。 |
| 38 | training_quality | rl_insight_monitor_actor_pg_clipfrac | Gauge | 触发裁剪 token / 有效 token | PPO 概率比触发裁剪的有效 token 比例。 |
| 39 | training_quality | rl_insight_monitor_actor_pg_clipfrac_lower | Gauge | 触发下界裁剪 token / 有效 token | PPO 概率比触发下界裁剪的有效 token 比例。 |
| 40 | training_quality | rl_insight_monitor_actor_grad_norm | Gauge | 梯度裁剪后的范数 | 梯度裁剪后记录的 Actor 梯度范数。 |
| 41 | training_quality | rl_insight_monitor_actor_lr | Gauge | 当前优化器学习率 | Actor 优化器当前学习率。 |
| 42 | training_quality | rl_insight_monitor_critic_score_mean | 统计值 | mean（批次内统计） | 排除中止样本后，序列级原始评分的平均值。 |
| 43 | training_quality | rl_insight_monitor_critic_score_max | 统计值 | max（批次内统计） | 排除中止样本后，序列级原始评分的最大值。 |
| 44 | training_quality | rl_insight_monitor_critic_score_min | 统计值 | min（批次内统计） | 排除中止样本后，序列级原始评分的最小值。 |
| 45 | training_quality | rl_insight_monitor_critic_rewards_mean | 统计值 | mean（批次内统计） | 排除中止样本后，序列级最终奖励的平均值。 |
| 46 | training_quality | rl_insight_monitor_critic_rewards_max | 统计值 | max（批次内统计） | 排除中止样本后，序列级最终奖励的最大值。 |
| 47 | training_quality | rl_insight_monitor_critic_rewards_min | 统计值 | min（批次内统计） | 排除中止样本后，序列级最终奖励的最小值。 |
| 48 | training_quality | rl_insight_monitor_critic_returns_mean | 统计值 | mean（批次内统计） | 有效响应 token 上回报值的平均值。 |
| 49 | training_quality | rl_insight_monitor_critic_returns_max | 统计值 | max（批次内统计） | 有效响应 token 上回报值的最大值。 |
| 50 | training_quality | rl_insight_monitor_critic_returns_min | 统计值 | min（批次内统计） | 有效响应 token 上回报值的最小值。 |
| 51 | training_quality | rl_insight_monitor_critic_advantages_mean | 统计值 | mean（批次内统计） | 有效响应 token 上优势值的平均值。 |
| 52 | training_quality | rl_insight_monitor_critic_advantages_max | 统计值 | max（批次内统计） | 有效响应 token 上优势值的最大值。 |
| 53 | training_quality | rl_insight_monitor_critic_advantages_min | 统计值 | min（批次内统计） | 有效响应 token 上优势值的最小值。 |
| 54 | training_quality | rl_insight_monitor_training_epoch | Gauge | 实测 | 当前训练轮次编号。 |
| 55 | data_characteristics | rl_insight_monitor_training_num_turns_mean | 统计值 | mean（批次内统计） | 当前训练批次多轮交互轮数的平均值。 |
| 56 | data_characteristics | rl_insight_monitor_training_num_turns_max | 统计值 | max（批次内统计） | 当前训练批次多轮交互轮数的最大值。 |
| 57 | data_characteristics | rl_insight_monitor_training_num_turns_min | 统计值 | min（批次内统计） | 当前训练批次多轮交互轮数的最小值。 |
| 58 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_staleness_mean | 统计值 | mean（批次内统计） | 异步训练中，轨迹生成所用策略版本相对当前训练策略的陈旧步数平均值。 |
| 59 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_staleness_max | 统计值 | max（批次内统计） | 异步训练中，轨迹生成所用策略版本相对当前训练策略的陈旧步数最大值。 |
| 60 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_staleness_worst_mean | 统计值 | mean（批次内统计） | 批次内按框架 worst 口径统计的轨迹陈旧步数平均值。 |
| 61 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_staleness_worst_max | 统计值 | max（批次内统计） | 批次内按框架 worst 口径统计的轨迹陈旧步数最大值。 |
| 62 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_staleness_worst_min | 统计值 | min（批次内统计） | 批次内按框架 worst 口径统计的轨迹陈旧步数最小值。 |
| 63 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_spans_mean | 统计值 | mean（批次内统计） | 样本内轨迹所跨策略版本范围的平均值。 |
| 64 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_spans_max | 统计值 | max（批次内统计） | 样本内轨迹所跨策略版本范围的最大值。 |
| 65 | rollout_quality | rl_insight_monitor_training_off_policy_trajectory_spans_min | 统计值 | min（批次内统计） | 样本内轨迹所跨策略版本范围的最小值。 |
| 66 | rollout_quality | rl_insight_monitor_training_rollout_probs_diff_mean | 统计值 | mean(|π_rollout − π_actor|) | Rollout 引擎概率与 Actor 重算概率之绝对差的平均值。 |
| 67 | rollout_quality | rl_insight_monitor_training_rollout_probs_diff_max | 统计值 | max(|π_rollout − π_actor|) | Rollout 引擎概率与 Actor 重算概率之绝对差的最大值。 |
| 68 | rollout_quality | rl_insight_monitor_training_rollout_probs_diff_std | 统计值 | std(|π_rollout − π_actor|) | Rollout 引擎概率与 Actor 重算概率之绝对差的标准差。 |
| 69 | rollout_quality | rl_insight_monitor_training_rollout_probs_diff_valid | Gauge | 差值统计是否有效（1/0） | Rollout 与 Actor 概率差统计是否有效的标记；1 为有效，0 为无效。 |
| 70 | rollout_quality | rl_insight_monitor_training_rollout_actor_probs_pearson_corr | 派生值 | Pearson(π_rollout, π_actor 重算) | Rollout 引擎概率与 Actor 重算概率之间的 Pearson 相关系数。 |
| 71 | rollout_quality | rl_insight_monitor_rollout_corr_kl | 派生值 | KL(π_rollout ‖ π_training) 直接估计 | Rollout 策略相对训练策略的直接 KL 估计，即 KL(π_rollout 与 π_training)。 |
| 72 | rollout_quality | rl_insight_monitor_rollout_corr_k3_kl | 派生值 | KL(π_rollout ‖ π_training) K3 估计 | Rollout 策略相对训练策略的 K3 KL 估计；相比直接估计在小差异时更稳定。 |
| 73 | rollout_quality | rl_insight_monitor_rollout_corr_chi2_seq | 派生值 | 序列级 χ²(π_training, π_rollout) | 训练策略与 Rollout 策略之间的序列级卡方散度。 |
| 74 | rollout_quality | rl_insight_monitor_rollout_corr_chi2_token | 派生值 | token 级 χ²(π_training, π_rollout) | 训练策略与 Rollout 策略之间的 token 级卡方散度。 |
| 75 | rollout_quality | rl_insight_monitor_rollout_corr_log_ppl_diff | 派生值 | log PPL_training − log PPL_rollout | 训练策略与 Rollout 策略的 log 困惑度差；定义为 training log PPL 减 rollout log PPL。 |
| 76 | rollout_quality | rl_insight_monitor_rollout_corr_log_ppl_diff_max | 派生值 | max(逐序列 log PPL diff) | 逐序列 log 困惑度差的最大值。 |
| 77 | rollout_quality | rl_insight_monitor_rollout_corr_log_ppl_diff_min | 派生值 | min(逐序列 log PPL diff) | 逐序列 log 困惑度差的最小值。 |
| 78 | rollout_quality | rl_insight_monitor_rollout_corr_log_ppl_abs_diff | 派生值 | mean(|逐序列 log PPL diff|) | 逐序列 log 困惑度差绝对值的平均值。 |
| 79 | rollout_quality | rl_insight_monitor_rollout_corr_ppl_ratio | 派生值 | mean(PPL_training / PPL_rollout) | 逐序列 training PPL 与 rollout PPL 比值的平均值。 |
| 80 | rollout_quality | rl_insight_monitor_rollout_corr_rollout_log_ppl | 派生值 | mean(rollout 逐序列 log PPL) | Rollout 策略逐序列 log 困惑度的批次平均值。 |
| 81 | rollout_quality | rl_insight_monitor_rollout_corr_rollout_ppl | 派生值 | mean(rollout 逐序列 PPL) | Rollout 策略逐序列困惑度的批次平均值。 |
| 82 | rollout_quality | rl_insight_monitor_rollout_corr_training_log_ppl | 派生值 | mean(training 逐序列 log PPL) | 训练策略逐序列 log 困惑度的批次平均值。 |
| 83 | rollout_quality | rl_insight_monitor_rollout_corr_training_ppl | 派生值 | mean(training 逐序列 PPL) | 训练策略逐序列困惑度的批次平均值。 |
| 84 | data_characteristics | rl_insight_monitor_val_aux_num_turns_mean | 统计值 | mean（批次内统计） | 验证批次多轮交互轮数的平均值。 |
| 85 | data_characteristics | rl_insight_monitor_val_aux_num_turns_max | 统计值 | max（批次内统计） | 验证批次多轮交互轮数的最大值。 |
| 86 | data_characteristics | rl_insight_monitor_val_aux_num_turns_min | 统计值 | min（批次内统计） | 验证批次多轮交互轮数的最小值。 |
| 87 | vllm_engine | vllm:request_max_num_generation_tokens | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求所指定最大生成 token 数的分布直方图。 |
| 88 | vllm_engine | vllm:request_prompt_tokens | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求输入或 Prefill token 数的分布直方图。 |
| 89 | vllm_engine | vllm:request_generation_tokens | Histogram | 直方图，histogram_quantile 取分位 | vLLM 请求实际生成 token 数的分布直方图。 |
| 90 | vllm_engine | vllm:kv_cache_usage_perc | Gauge | 已用 KV Cache / 总量（1=100%） | vLLM KV Cache 使用比例；数值 1 表示使用率 100%。 |
| 91 | vllm_engine | vllm:num_requests_running | Gauge | 实测 | vLLM 当前正在模型执行批次中运行的请求数。 |
| 92 | vllm_engine | vllm:num_requests_waiting | Gauge | 实测 | vLLM 当前等待调度的请求数。 |
| 93 | vllm_engine | vllm:num_requests_swapped | Gauge | 实测 | vLLM 旧版调度器中当前处于 swapped 状态的请求数。 |
| 94 | transfer_queue | tq_storage_request_latency_p50 | 统计值 | histogram 50 分位 | TransferQueue 存储请求时延 P50（s）。 |
| 95 | transfer_queue | tq_storage_request_latency_p99 | 统计值 | histogram 99 分位 | TransferQueue 存储请求时延 P99（s）。 |
| 96 | transfer_queue | tq_controller_uptime_seconds | Gauge | 进程启动以来的运行时长 | TransferQueue Controller 运行时长（s）。 |
| 97 | transfer_queue | tq_controller_memory_rss_bytes | Gauge | 进程常驻内存 RSS | TransferQueue Controller 进程常驻内存（B）。 |
| 98 | transfer_queue | tq_partition_production_progress | Gauge | 已生产 / 总量（0–1） | TransferQueue 各分区生产进度（0–1）。 |
| 99 | transfer_queue | tq_partition_consumption_progress | Gauge | 已消费 / 总量（0–1） | TransferQueue 各分区各任务消费进度（0–1）。 |
| 100 | transfer_queue | tq_storage_utilization_ratio | Gauge | 有效键数 / 容量上限 | TransferQueue 存储当前有效键数相对容量上限的比例。 |
| 101 | transfer_queue | tq_storage_memory_rss_bytes | Gauge | 进程常驻内存 RSS | TransferQueue 存储进程常驻内存（B）。 |
| 102 | transfer_queue | tq_storage_request_ops | Counter | 按操作类型累计请求数 | TransferQueue 各存储单元按操作类型统计的已处理请求数。 |

## Cataloged candidates requiring preprocessing

This section follows the current catalog grouping. The vLLM entries are cumulative counters and require rate or increase conversion. Several TransferQueue names ending in `_total` are Gauges in the official implementation; their Chinese descriptions identify that distinction.

| No. | Exact metric name | 数据类型 | 计算方法 | 中文说明 |
| ---: | --- | --- | --- | --- |
| 1 | vllm:generation_tokens_total | Counter | 累计计数，需 rate/increase | vLLM 累计生成的输出 token 数；用于关联前需转换为区间增量或速率。 |
| 2 | vllm:prompt_tokens_total | Counter | 累计计数，需 rate/increase | vLLM 累计处理的 Prompt token 数；用于关联前需转换为区间增量或速率。 |
| 3 | vllm:request_success_total | Counter | 累计计数，需 rate/increase | vLLM 按完成原因统计的累计成功请求数；用于关联前需转换为区间增量或速率。 |
| 4 | vllm:prefix_cache_hits_total | Counter | 累计计数，需 rate/increase | vLLM 前缀缓存累计命中的 token 数；用于关联前需转换为区间增量或速率。 |
| 5 | vllm:prefix_cache_queries_total | Counter | 累计计数，需 rate/increase | vLLM 前缀缓存累计查询的 token 数；用于关联前需转换为区间增量或速率。 |
| 6 | vllm:spec_decode_num_accepted_tokens_total | Counter | 累计计数，需 rate/increase | vLLM 投机解码累计接受的草稿 token 数；用于关联前需转换为区间增量或速率。 |
| 7 | vllm:spec_decode_num_draft_tokens_total | Counter | 累计计数，需 rate/increase | vLLM 投机解码累计生成的草稿 token 数；用于关联前需转换为区间增量或速率。 |
| 8 | vllm:spec_decode_num_drafts_total | Counter | 累计计数，需 rate/increase | vLLM 投机解码累计执行的草稿提议次数；用于关联前需转换为区间增量或速率。 |
| 9 | tq_controller_request_total | Counter | 累计计数，需 rate/increase | TransferQueue Controller 按操作类型累计接收的请求数。 |
| 10 | tq_controller_request_samples_total | Counter | 累计计数，需 rate/increase | TransferQueue Controller 按操作类型累计处理的样本数。 |
| 11 | tq_partitions_total | Gauge | 实测 | TransferQueue 当前活动分区数；官方实现为 Gauge，并非累计 Counter。 |
| 12 | tq_partition_samples_total | Gauge | 实测 | TransferQueue 各分区当前活动样本数；官方实现为 Gauge，并非累计 Counter。 |
| 13 | tq_storage_active_keys_total | Gauge | 实测 | TransferQueue 存储当前有效键数；官方实现为 Gauge，并非累计 Counter。 |
| 14 | tq_storage_capacity_total | Gauge | 实测 | TransferQueue 存储可容纳的最大键数；官方实现为 Gauge，并非累计 Counter。 |
| 15 | tq_global_index_allocated_total | Gauge | 实测 | TransferQueue 当前已分配的全局索引数；官方实现为 Gauge，并非累计 Counter。 |
| 16 | tq_global_index_reusable_total | Gauge | 实测 | TransferQueue 当前可复用的全局索引数；官方实现为 Gauge，并非累计 Counter。 |
