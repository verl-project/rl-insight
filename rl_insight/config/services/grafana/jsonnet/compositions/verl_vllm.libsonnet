// VERL trainer dashboard with the vLLM engine and the NPU hardware content.
//
// `_jsonnet` is kept on purpose in the title and name: this dashboard is
// materialized next to the committed static
// `verl_tainer_v1_with_vllm_engine.json`, which keeps its own identity. The
// historical `tainer` spelling is preserved for the same reason.
local trainer = import '../dashboards/trainer.libsonnet';
local controller = import '../dashboards/controller.libsonnet';
local storage = import '../dashboards/storage.libsonnet';
local trajectory = import '../dashboards/trajectory.libsonnet';
local vllm = import '../dashboards/vllm.libsonnet';
local npu = import '../dashboards/npu.libsonnet';
local chrome = import 'verl_dashboard.libsonnet';

// Trainer-side content shared by every VERL composition.
local verlBase = [trainer, controller, storage, trajectory];

{
  modules: verlBase + [vllm, npu],
  dashboard: {
    metadata: {
      name: '4e42cd18-bffc-491e-8ea6-cd49b6bfd74b',
      generation: 45,
      creationTimestamp: '2026-07-08T14:34:31Z',
      labels: {},
      annotations: {},
    },
    title: 'verl_trainer_v1_with_vllm_engine_jsonnet',
    tags: [
      'RL-Insight',
      'verl',
      'vllm',
    ],
    spec: chrome.spec,
    variableOrder: [
      'datasource',
      'project',
      'experiment_name',
      'vllm_model_name',
      'workerid',
      'interval',
      'replica',
      'task_name',
      'op_type',
      'quantile',
      'npu_instance',
    ],
    rowOrder: [
      'rl state timeline',
      'training metric',
      'vllm engine metric',
      'transfer queue metric',
      'hardware metric',
    ],
  },
}
