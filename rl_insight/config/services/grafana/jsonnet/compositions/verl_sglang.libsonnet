// VERL trainer dashboard with the SGLang engine content. SGLang never composes
// the NPU module: its own modules already cover the engine metrics.
//
// `_jsonnet` is kept on purpose in the title and name: this dashboard is
// materialized next to the committed static
// `verl_tainer_v1_with_sglang_engine.json`, which keeps its own identity. The
// historical `tainer` spelling is preserved for the same reason.
local trainer = import '../dashboards/trainer.libsonnet';
local controller = import '../dashboards/controller.libsonnet';
local storage = import '../dashboards/storage.libsonnet';
local trajectory = import '../dashboards/trajectory.libsonnet';
local sglang = import '../dashboards/sglang.libsonnet';
local chrome = import 'verl_dashboard.libsonnet';

// Trainer-side content shared by every VERL composition.
local verlBase = [trainer, controller, storage, trajectory];

{
  modules: verlBase + [sglang],
  dashboard: {
    metadata: {
      name: '3891c6d0-6872-4d81-953c-fed38cce5383',
      generation: 1,
      creationTimestamp: '2026-07-08T14:34:31Z',
      labels: {},
      annotations: {},
    },
    title: 'verl_trainer_v1_with_sglang_engine_jsonnet',
    tags: [
      'RL-Insight',
      'verl',
      'sglang',
    ],
    spec: chrome.spec,
    variableOrder: [
      'datasource',
      'project',
      'experiment_name',
      'sglang_model_name',
      'sglang_replica',
      'task_name',
      'op_type',
      'quantile',
    ],
    rowOrder: [
      'rl state timeline',
      'training metric',
      'sglang engine metric',
      'transfer queue metric',
      'device metric',
    ],
  },
}
