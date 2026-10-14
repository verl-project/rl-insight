local trainingMetrics = [
  ['training.actor.actor_entropy_loss', 'panel-185', 185, 'actor_entropy_loss'],
  ['training.actor.actor_grad_norm', 'panel-186', 186, 'actor_grad_norm'],
  ['training.actor.actor_kl_coef', 'panel-187', 187, 'actor_kl_coef'],
  ['training.actor.actor_kl_loss', 'panel-188', 188, 'actor_kl_loss'],
  ['training.actor.actor_loss', 'panel-189', 189, 'actor_loss'],
  ['training.actor.actor_lr', 'panel-190', 190, 'actor_lr'],
  ['training.actor.actor_perf_cpu_memory_used_gb', 'panel-191', 191, 'actor_perf_cpu_memory_used_gb'],
  ['training.actor.actor_perf_max_memory_allocated_gb', 'panel-192', 192, 'actor_perf_max_memory_allocated_gb'],
  ['training.actor.actor_perf_max_memory_reserved_gb', 'panel-193', 193, 'actor_perf_max_memory_reserved_gb'],
  ['training.actor.actor_pg_clipfrac', 'panel-194', 194, 'actor_pg_clipfrac'],
  ['training.actor.actor_pg_clipfrac_lower', 'panel-195', 195, 'actor_pg_clipfrac_lower'],
  ['training.actor.actor_pg_loss', 'panel-196', 196, 'actor_pg_loss'],
  ['training.actor.actor_ppo_kl', 'panel-197', 197, 'actor_ppo_kl'],
  ['training.critic.critic_advantages_max', 'panel-198', 198, 'critic_advantages_max'],
  ['training.critic.critic_advantages_mean', 'panel-199', 199, 'critic_advantages_mean'],
  ['training.critic.critic_advantages_min', 'panel-200', 200, 'critic_advantages_min'],
  ['training.critic.critic_returns_max', 'panel-201', 201, 'critic_returns_max'],
  ['training.critic.critic_returns_mean', 'panel-202', 202, 'critic_returns_mean'],
  ['training.critic.critic_returns_min', 'panel-203', 203, 'critic_returns_min'],
  ['training.critic.critic_rewards_max', 'panel-204', 204, 'critic_rewards_max'],
  ['training.critic.critic_rewards_mean', 'panel-205', 205, 'critic_rewards_mean'],
  ['training.critic.critic_rewards_min', 'panel-206', 206, 'critic_rewards_min'],
  ['training.global_seqlen.global_seqlen_balanced_max', 'panel-210', 210, 'global_seqlen_balanced_max'],
  ['training.global_seqlen.global_seqlen_balanced_min', 'panel-211', 211, 'global_seqlen_balanced_min'],
  ['training.global_seqlen.global_seqlen_max', 'panel-212', 212, 'global_seqlen_max'],
  ['training.global_seqlen.global_seqlen_mean', 'panel-213', 213, 'global_seqlen_mean'],
  ['training.global_seqlen.global_seqlen_min', 'panel-214', 214, 'global_seqlen_min'],
  ['training.global_seqlen.global_seqlen_minmax_diff', 'panel-215', 215, 'global_seqlen_minmax_diff'],
  ['training.perf.perf_mfu_actor', 'panel-219', 219, 'perf_mfu_actor'],
  ['training.perf.perf_throughput', 'panel-221', 221, 'perf_throughput'],
  ['training.perf.perf_time_per_step', 'panel-222', 222, 'perf_time_per_step'],
  ['training.perf.perf_total_num_tokens', 'panel-223', 223, 'perf_total_num_tokens'],
  ['training.prompt_length.prompt_length_clip_ratio', 'panel-224', 224, 'prompt_length_clip_ratio'],
  ['training.prompt_length.prompt_length_max', 'panel-225', 225, 'prompt_length_max'],
  ['training.prompt_length.prompt_length_mean', 'panel-226', 226, 'prompt_length_mean'],
  ['training.prompt_length.prompt_length_min', 'panel-227', 227, 'prompt_length_min'],
  ['training.response.response_aborted_ratio', 'panel-228', 228, 'response_aborted_ratio'],
  ['training.response.response_length_clip_ratio', 'panel-229', 229, 'response_length_clip_ratio'],
  ['training.response.response_length_max', 'panel-230', 230, 'response_length_max'],
  ['training.response.response_length_mean', 'panel-231', 231, 'response_length_mean'],
  ['training.response.response_length_min', 'panel-232', 232, 'response_length_min'],
  ['training.response.response_length_non_aborted_clip_ratio', 'panel-233', 233, 'response_length_non_aborted_clip_ratio'],
  ['training.response.response_length_non_aborted_max', 'panel-234', 234, 'response_length_non_aborted_max'],
  ['training.response.response_length_non_aborted_mean', 'panel-235', 235, 'response_length_non_aborted_mean'],
  ['training.response.response_length_non_aborted_min', 'panel-236', 236, 'response_length_non_aborted_min'],
  ['training.rollout.rollout_corr_chi2_seq', 'panel-237', 237, 'rollout_corr_chi2_seq'],
  ['training.rollout.rollout_corr_chi2_token', 'panel-238', 238, 'rollout_corr_chi2_token'],
  ['training.rollout.rollout_corr_k3_kl', 'panel-239', 239, 'rollout_corr_k3_kl'],
  ['training.rollout.rollout_corr_kl', 'panel-240', 240, 'rollout_corr_kl'],
  ['training.rollout.rollout_corr_log_ppl_abs_diff', 'panel-241', 241, '_rollout_corr_log_ppl_abs_diff'],
  ['training.rollout.rollout_corr_log_ppl_diff', 'panel-242', 242, 'rollout_corr_log_ppl_diff'],
  ['training.rollout.rollout_corr_log_ppl_diff_max', 'panel-243', 243, 'rollout_corr_log_ppl_diff_max'],
  ['training.rollout.rollout_corr_log_ppl_diff_min', 'panel-244', 244, 'rollout_corr_log_ppl_diff_min'],
  ['training.rollout.rollout_corr_ppl_ratio', 'panel-245', 245, 'rollout_corr_ppl_ratio'],
  ['training.rollout.rollout_corr_rollout_log_ppl', 'panel-246', 246, 'rollout_corr_rollout_log_ppl'],
  ['training.rollout.rollout_corr_rollout_ppl', 'panel-247', 247, 'rollout_corr_rollout_ppl'],
  ['training.rollout.rollout_corr_training_log_ppl', 'panel-248', 248, 'rollout_corr_training_log_ppl'],
  ['training.rollout.rollout_corr_training_ppl', 'panel-249', 249, 'rollout_corr_training_ppl'],
  ['training.timing.timing_per_token_ms_adv', 'panel-250', 250, 'timing_per_token_ms_adv'],
  ['training.timing.timing_per_token_ms_gen', 'panel-251', 251, 'timing_per_token_ms_gen'],
  ['training.timing.timing_per_token_ms_ref', 'panel-252', 252, 'timing_per_token_ms_ref'],
  ['training.timing.timing_per_token_ms_update_actor', 'panel-253', 253, 'timing_per_token_ms_update_actor'],
  ['training.timing.timing_s_adv', 'panel-254', 254, 'timing_s_adv'],
  ['training.timing.timing_s_gen', 'panel-273', 273, 'timing_s_gen'],
  ['training.timing.timing_s_old_log_prob', 'panel-274', 274, 'timing_s_old_log_prob'],
  ['training.timing.timing_s_ref', 'panel-275', 275, 'timing_s_ref'],
  ['training.timing.timing_s_step', 'panel-278', 278, 'timing_s_step'],
  ['training.timing.timing_s_testing', 'panel-280', 280, 'timing_s_testing'],
  ['training.timing.rl_insight_monitor_timing_s_update_actor', 'panel-281', 281, 'rl_insight_monitor_timing_s_update_actor'],
  ['training.timing.timing_s_update_weights', 'panel-282', 282, 'timing_s_update_weights'],
  ['training.training.training_rollout_actor_probs_pearson_corr', 'panel-285', 285, 'training_rollout_actor_probs_pearson_corr'],
  ['training.training.training_rollout_probs_diff_max', 'panel-286', 286, 'training_rollout_probs_diff_max'],
  ['training.training.training_rollout_probs_diff_mean', 'panel-287', 287, 'training_rollout_probs_diff_mean'],
  ['training.training.training_rollout_probs_diff_std', 'panel-288', 288, 'training_rollout_probs_diff_std'],
  ['training.training.training_rollout_probs_diff_valid', 'panel-289', 289, 'training_rollout_probs_diff_valid'],
  ['training.val.val_aux_num_turns_max', 'panel-290', 290, 'val_aux_num_turns_max'],
  ['training.val.val_aux_num_turns_mean', 'panel-291', 291, 'val_aux_num_turns_mean'],
  ['training.val.val_aux_num_turns_min', 'panel-292', 292, 'val_aux_num_turns_min'],
  ['training.val.val_aux_openai_gsm8k_reward_mean_1', 'panel-293', 293, 'val_aux_openai_gsm8k_reward_mean_1'],
  ['training.val.val_core_openai_gsm8k_acc_mean_1', 'panel-294', 294, 'val_core_openai_gsm8k_acc_mean_1'],
  ['training.training.training_num_turns_max', 'panel-314', 314, 'training_num_turns_max'],
  ['training.training.training_off_policy_trajectory_spans_mean', 'panel-315', 315, 'training_off_policy_trajectory_spans_mean'],
  ['training.training.training_num_turns_mean', 'panel-316', 316, 'training_num_turns_mean'],
  ['training.training.training_off_policy_trajectory_spans_min', 'panel-317', 317, 'training_off_policy_trajectory_spans_min'],
  ['training.training.training_off_policy_trajectory_staleness_max', 'panel-318', 318, 'training_off_policy_trajectory_staleness_max'],
  ['training.training.training_off_policy_trajectory_staleness_worst_max', 'panel-319', 319, 'training_off_policy_trajectory_staleness_worst_max'],
  ['training.training.training_num_turns_min', 'panel-320', 320, 'training_num_turns_min'],
  ['training.training.training_off_policy_trajectory_staleness_mean', 'panel-321', 321, 'training_off_policy_trajectory_staleness_mean'],
  ['training.training.training_off_policy_trajectory_spans_max', 'panel-322', 322, 'training_off_policy_trajectory_spans_max'],
  ['training.training.training_off_policy_trajectory_staleness_worst_min', 'panel-323', 323, 'training_off_policy_trajectory_staleness_worst_min'],
  ['training.training.training_off_policy_trajectory_staleness_worst_mean', 'panel-324', 324, 'training_off_policy_trajectory_staleness_worst_mean'],
];

local trainingPanel(metric) =
  local title = metric[3];
  local metricName =
    if std.startsWith(title, 'rl_insight_monitor_') then title
    else 'rl_insight_monitor_' + (if std.startsWith(title, '_') then title[1:] else title);
  {
    key: metric[0],
    outputKey: metric[1],
    id: metric[2],
    title: title,
    queries: [{
      expr: metricName + '{project=~"$project", experiment_name=~"$experiment_name"}',
    }],
  };

// Reusable RL training metrics and their unchanged production layout.
{
  panels: [
    {
      key: 'training.actor.actor_entropy',
      outputKey: 'panel-184',
      id: 184,
      title: 'actor_entropy',
      queries: [
        {
          expr: 'rl_insight_monitor_actor_entropy{project=~"$project", experiment_name=~"$experiment_name"}',
          extra: {
            instant: false,
          },
        },
      ],
    },
    {
      key: 'training.critic.critic_score_max',
      outputKey: 'panel-207',
      id: 207,
      title: 'critic_score_max',
      queries: [
        {
          expr: 'rl_insight_monitor_critic_score_max{project=~"$project", experiment_name=~"$experiment_name"}',
        },
      ],
      vizPatch: {
        version: '13.0.1',
      },
    },
    {
      key: 'training.critic.critic_score_mean',
      outputKey: 'panel-208',
      id: 208,
      title: 'critic_score_mean',
      queries: [
        {
          expr: 'rl_insight_monitor_critic_score_mean{project=~"$project", experiment_name=~"$experiment_name"}',
        },
      ],
      vizPatch: {
        version: '13.0.1',
      },
    },
    {
      key: 'training.critic.critic_score_min',
      outputKey: 'panel-209',
      id: 209,
      title: 'critic_score_min',
      queries: [
        {
          expr: 'rl_insight_monitor_critic_score_min{project=~"$project", experiment_name=~"$experiment_name"}',
        },
      ],
      vizPatch: {
        version: '13.0.1',
      },
    },
    {
      key: 'training.training.training_epoch',
      outputKey: 'panel-283',
      id: 283,
      title: 'training_epoch',
      queries: [
        {
          expr: 'rl_insight_monitor_training_epoch{project=~"$project", experiment_name=~"$experiment_name"}',
        },
      ],
      vizBase: 'gauge',
    },
    {
      key: 'training.training.training_global_step',
      outputKey: 'panel-284',
      id: 284,
      title: 'training_global_step',
      queries: [
        {
          expr: 'rl_insight_monitor_training_global_step{project=~"$project", experiment_name=~"$experiment_name"}',
        },
      ],
      vizBase: 'gauge',
    },
  ] + [trainingPanel(metric) for metric in trainingMetrics],
  rows: {
    'training metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'training metric',
        collapse: true,
        layout: {
          kind: 'RowsLayout',
          spec: {
            rows: [
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'actor',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_entropy',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_entropy_loss',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_grad_norm',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_kl_coef',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_kl_loss',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_loss',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_lr',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_perf_cpu_memory_used_gb',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_perf_max_memory_allocated_gb',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_perf_max_memory_reserved_gb',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_pg_clipfrac',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_pg_clipfrac_lower',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_pg_loss',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.actor.actor_ppo_kl',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'critic',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_advantages_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_advantages_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_advantages_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_returns_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_returns_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_returns_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_rewards_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_rewards_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_rewards_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_score_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_score_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.critic.critic_score_min',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'global_seqlen',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_balanced_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_balanced_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.global_seqlen.global_seqlen_minmax_diff',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'perf',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.perf.perf_mfu_actor',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.perf.perf_total_num_tokens',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.perf.perf_throughput',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.perf.perf_time_per_step',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'prompt_length',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.prompt_length.prompt_length_clip_ratio',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.prompt_length.prompt_length_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.prompt_length.prompt_length_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.prompt_length.prompt_length_min',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'response',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_aborted_ratio',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_clip_ratio',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_non_aborted_clip_ratio',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_non_aborted_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_non_aborted_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.response.response_length_non_aborted_min',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'rollout',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_chi2_seq',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_chi2_token',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_k3_kl',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_kl',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_log_ppl_abs_diff',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_log_ppl_diff',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_log_ppl_diff_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_log_ppl_diff_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_ppl_ratio',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_rollout_log_ppl',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_rollout_ppl',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_training_log_ppl',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.rollout.rollout_corr_training_ppl',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'timing',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_per_token_ms_adv',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_per_token_ms_gen',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_per_token_ms_ref',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_per_token_ms_update_actor',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_adv',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_gen',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_old_log_prob',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_ref',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.rl_insight_monitor_timing_s_update_actor',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_update_weights',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_step',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.timing.timing_s_testing',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'training',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_epoch',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_global_step',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_num_turns_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_num_turns_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_num_turns_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_rollout_actor_probs_pearson_corr',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_rollout_probs_diff_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_rollout_probs_diff_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 16,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_rollout_probs_diff_std',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_rollout_probs_diff_valid',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_spans_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 24,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_spans_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_spans_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_staleness_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 32,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_staleness_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 40,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_staleness_worst_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 40,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_staleness_worst_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 40,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.training.training_off_policy_trajectory_staleness_worst_min',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'val',
                  collapse: true,
                  layout: {
                    kind: 'GridLayout',
                    spec: {
                      items: [
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.val.val_aux_num_turns_max',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.val.val_aux_num_turns_mean',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.val.val_aux_num_turns_min',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.val.val_aux_openai_gsm8k_reward_mean_1',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 8,
                            width: 8,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'training.val.val_core_openai_gsm8k_acc_mean_1',
                            },
                          },
                        },
                      ],
                    },
                  },
                },
              },
            ],
          },
        },
      },
    },
  },
  variables: {
    datasource: {
      kind: 'DatasourceVariable',
      spec: {
        name: 'datasource',
        pluginId: 'prometheus',
        refresh: 'onDashboardLoad',
        regex: '',
        current: {
          text: '',
          value: '',
        },
        options: [],
        multi: false,
        includeAll: false,
        hide: 'hideVariable',
        skipUrlSync: false,
        description: 'Filter queries of a specific Prometheus type.',
        allowCustomValue: true,
      },
    },
    experiment_name: {
      kind: 'QueryVariable',
      spec: {
        name: 'experiment_name',
        current: {
          text: 'All',
          value: '$__all',
        },
        label: 'training: Experiment Name',
        hide: 'dontHide',
        refresh: 'onDashboardLoad',
        skipUrlSync: false,
        query: {
          kind: 'DataQuery',
          group: 'prometheus',
          version: 'v0',
          datasource: {
            name: '${datasource}',
          },
          spec: {
            query: 'label_values({__name__=~"rl_insight_monitor_.*"}, experiment_name)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'alphabeticalAsc',
        definition: 'label_values({__name__=~"rl_insight_monitor_.*"}, experiment_name)',
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
    project: {
      kind: 'QueryVariable',
      spec: {
        name: 'project',
        current: {
          text: 'All',
          value: '$__all',
        },
        label: 'training: Project',
        hide: 'dontHide',
        refresh: 'onDashboardLoad',
        skipUrlSync: false,
        query: {
          kind: 'DataQuery',
          group: 'prometheus',
          version: 'v0',
          datasource: {
            name: '${datasource}',
          },
          spec: {
            query: 'label_values({__name__=~"rl_insight_monitor_.*"}, project)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'alphabeticalAsc',
        definition: 'label_values({__name__=~"rl_insight_monitor_.*"}, project)',
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
  },
}
