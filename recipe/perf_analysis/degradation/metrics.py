# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric catalog for the degradation experiment."""

from dataclasses import dataclass

GLOBAL_STEP_METRIC = "rl_insight_monitor_training_global_step"

# Fixed scalar latency targets that can trigger degradation events.
TARGET_METRICS = (
    "rl_insight_monitor_timing_s_step",
    "rl_insight_monitor_timing_s_gen",
    "rl_insight_monitor_timing_s_ref",
    "rl_insight_monitor_timing_s_adv",
    "rl_insight_monitor_timing_s_old_log_prob",
    "rl_insight_monitor_timing_s_update_actor",
    "rl_insight_monitor_timing_s_update_weights",
    "rl_insight_monitor_timing_s_testing",
)

# Histogram families are cataloged but require aggregation before modeling.
HISTOGRAM_TARGET_METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "tq_controller_request_duration_seconds",
)


@dataclass(frozen=True)
class MetricSpec:
    """One scalar association candidate and its presentation category."""

    name: str
    category: str


CANDIDATE_CATEGORIES = (
    "latency",
    "training_quality",
    "rollout_quality",
    "data_characteristics",
    "hardware_resources",
    "vllm_engine",
    "transfer_queue",
)

# Keep this in runtime order. Every derived candidate view comes from this one
# catalog, so detection and model-facing presentation cannot drift apart.
CANDIDATE_METRIC_SPECS = (
    MetricSpec("rl_insight_monitor_timing_per_token_ms_gen", "latency"),
    MetricSpec("rl_insight_monitor_timing_per_token_ms_ref", "latency"),
    MetricSpec("rl_insight_monitor_timing_per_token_ms_adv", "latency"),
    MetricSpec("rl_insight_monitor_timing_per_token_ms_update_actor", "latency"),
    MetricSpec("rl_insight_monitor_perf_time_per_step", "latency"),
    MetricSpec("rl_insight_monitor_perf_throughput", "latency"),
    MetricSpec("rl_insight_monitor_perf_mfu_actor", "hardware_resources"),
    MetricSpec("rl_insight_monitor_perf_total_num_tokens", "data_characteristics"),
    MetricSpec(
        "rl_insight_monitor_actor_perf_cpu_memory_used_gb", "hardware_resources"
    ),
    MetricSpec(
        "rl_insight_monitor_actor_perf_max_memory_allocated_gb", "hardware_resources"
    ),
    MetricSpec(
        "rl_insight_monitor_actor_perf_max_memory_reserved_gb", "hardware_resources"
    ),
    MetricSpec("rl_insight_monitor_prompt_length_mean", "data_characteristics"),
    MetricSpec("rl_insight_monitor_prompt_length_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_prompt_length_min", "data_characteristics"),
    MetricSpec("rl_insight_monitor_prompt_length_clip_ratio", "data_characteristics"),
    MetricSpec("rl_insight_monitor_response_length_mean", "data_characteristics"),
    MetricSpec("rl_insight_monitor_response_length_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_response_length_min", "data_characteristics"),
    MetricSpec("rl_insight_monitor_response_length_clip_ratio", "data_characteristics"),
    MetricSpec(
        "rl_insight_monitor_response_length_non_aborted_mean", "data_characteristics"
    ),
    MetricSpec(
        "rl_insight_monitor_response_length_non_aborted_max", "data_characteristics"
    ),
    MetricSpec(
        "rl_insight_monitor_response_length_non_aborted_min", "data_characteristics"
    ),
    MetricSpec(
        "rl_insight_monitor_response_length_non_aborted_clip_ratio",
        "data_characteristics",
    ),
    MetricSpec("rl_insight_monitor_response_aborted_ratio", "rollout_quality"),
    MetricSpec("rl_insight_monitor_global_seqlen_mean", "data_characteristics"),
    MetricSpec("rl_insight_monitor_global_seqlen_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_global_seqlen_min", "data_characteristics"),
    MetricSpec("rl_insight_monitor_global_seqlen_minmax_diff", "data_characteristics"),
    MetricSpec("rl_insight_monitor_global_seqlen_balanced_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_global_seqlen_balanced_min", "data_characteristics"),
    MetricSpec("rl_insight_monitor_actor_loss", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_pg_loss", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_kl_loss", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_entropy_loss", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_entropy", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_ppo_kl", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_kl_coef", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_pg_clipfrac", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_pg_clipfrac_lower", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_grad_norm", "training_quality"),
    MetricSpec("rl_insight_monitor_actor_lr", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_score_mean", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_score_max", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_score_min", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_rewards_mean", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_rewards_max", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_rewards_min", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_returns_mean", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_returns_max", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_returns_min", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_advantages_mean", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_advantages_max", "training_quality"),
    MetricSpec("rl_insight_monitor_critic_advantages_min", "training_quality"),
    MetricSpec("rl_insight_monitor_training_epoch", "training_quality"),
    MetricSpec("rl_insight_monitor_training_num_turns_mean", "data_characteristics"),
    MetricSpec("rl_insight_monitor_training_num_turns_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_training_num_turns_min", "data_characteristics"),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_staleness_mean",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_staleness_max",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_mean",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_max",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_min",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_spans_mean",
        "rollout_quality",
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_spans_max", "rollout_quality"
    ),
    MetricSpec(
        "rl_insight_monitor_training_off_policy_trajectory_spans_min", "rollout_quality"
    ),
    MetricSpec(
        "rl_insight_monitor_training_rollout_probs_diff_mean", "rollout_quality"
    ),
    MetricSpec("rl_insight_monitor_training_rollout_probs_diff_max", "rollout_quality"),
    MetricSpec("rl_insight_monitor_training_rollout_probs_diff_std", "rollout_quality"),
    MetricSpec(
        "rl_insight_monitor_training_rollout_probs_diff_valid", "rollout_quality"
    ),
    MetricSpec(
        "rl_insight_monitor_training_rollout_actor_probs_pearson_corr",
        "rollout_quality",
    ),
    MetricSpec("rl_insight_monitor_rollout_corr_kl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_k3_kl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_chi2_seq", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_chi2_token", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_log_ppl_diff", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_log_ppl_diff_max", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_log_ppl_diff_min", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_log_ppl_abs_diff", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_ppl_ratio", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_rollout_log_ppl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_rollout_ppl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_training_log_ppl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_rollout_corr_training_ppl", "rollout_quality"),
    MetricSpec("rl_insight_monitor_val_aux_num_turns_mean", "data_characteristics"),
    MetricSpec("rl_insight_monitor_val_aux_num_turns_max", "data_characteristics"),
    MetricSpec("rl_insight_monitor_val_aux_num_turns_min", "data_characteristics"),
    MetricSpec("vllm:request_max_num_generation_tokens", "vllm_engine"),
    MetricSpec("vllm:request_prompt_tokens", "vllm_engine"),
    MetricSpec("vllm:request_generation_tokens", "vllm_engine"),
    MetricSpec("vllm:kv_cache_usage_perc", "vllm_engine"),
    MetricSpec("vllm:num_requests_running", "vllm_engine"),
    MetricSpec("vllm:num_requests_waiting", "vllm_engine"),
    MetricSpec("vllm:num_requests_swapped", "vllm_engine"),
    MetricSpec("tq_storage_request_latency_p50", "transfer_queue"),
    MetricSpec("tq_storage_request_latency_p99", "transfer_queue"),
    MetricSpec("tq_controller_uptime_seconds", "transfer_queue"),
    MetricSpec("tq_controller_memory_rss_bytes", "transfer_queue"),
    MetricSpec("tq_partition_production_progress", "transfer_queue"),
    MetricSpec("tq_partition_consumption_progress", "transfer_queue"),
    MetricSpec("tq_storage_utilization_ratio", "transfer_queue"),
    MetricSpec("tq_storage_memory_rss_bytes", "transfer_queue"),
    MetricSpec("tq_storage_request_ops", "transfer_queue"),
)

# Scalar metrics that can be modeled directly with a BOTH policy.
CANDIDATE_METRICS = tuple(spec.name for spec in CANDIDATE_METRIC_SPECS)
CATEGORY_BY_METRIC = {spec.name: spec.category for spec in CANDIDATE_METRIC_SPECS}
CANDIDATE_METRICS_BY_CATEGORY = {
    category: tuple(
        spec.name for spec in CANDIDATE_METRIC_SPECS if spec.category == category
    )
    for category in CANDIDATE_CATEGORIES
}

# Raw counters are cataloged but require rate or increase before modeling.
RATE_REQUIRED_CANDIDATE_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "tq_controller_request_total",
    "tq_controller_request_samples_total",
    "tq_partitions_total",
    "tq_partition_samples_total",
    "tq_storage_active_keys_total",
    "tq_storage_capacity_total",
    "tq_global_index_allocated_total",
    "tq_global_index_reusable_total",
)

CATALOG_TARGET_METRICS = TARGET_METRICS + HISTOGRAM_TARGET_METRICS
CATALOG_CANDIDATE_METRICS = CANDIDATE_METRICS + RATE_REQUIRED_CANDIDATE_METRICS
METRIC_CATALOG = (
    GLOBAL_STEP_METRIC,
    *CATALOG_TARGET_METRICS,
    *CATALOG_CANDIDATE_METRICS,
)

__all__ = [
    "CANDIDATE_CATEGORIES",
    "CANDIDATE_METRICS",
    "CANDIDATE_METRICS_BY_CATEGORY",
    "CANDIDATE_METRIC_SPECS",
    "CATALOG_CANDIDATE_METRICS",
    "CATALOG_TARGET_METRICS",
    "CATEGORY_BY_METRIC",
    "GLOBAL_STEP_METRIC",
    "HISTOGRAM_TARGET_METRICS",
    "METRIC_CATALOG",
    "RATE_REQUIRED_CANDIDATE_METRICS",
    "TARGET_METRICS",
    "MetricSpec",
]
