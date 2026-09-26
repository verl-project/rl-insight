// The single production composition registry.
//
// Developers register a dashboard here: import (or add) the semantic content
// modules, then add one entry with the aggregate `modules` list and the
// dashboard-level `dashboard` config. Nothing else needs to change; the thin
// `dashboards.jsonnet` entrypoint composes every entry in this registry.
local trainer = import 'dashboards/trainer.libsonnet';
local controller = import 'dashboards/controller.libsonnet';
local storage = import 'dashboards/storage.libsonnet';
local trajectory = import 'dashboards/trajectory.libsonnet';
local vllm = import 'dashboards/vllm.libsonnet';
local sglang = import 'dashboards/sglang.libsonnet';
local npu = import 'dashboards/npu.libsonnet';

// Dashboard-level configuration kept outside reusable content modules.
local productionSpec = {
  annotations: [
    {
      kind: 'AnnotationQuery',
      spec: {
        query: {
          kind: 'DataQuery',
          group: 'grafana',
          version: 'v0',
          spec: {},
        },
        enable: true,
        hide: true,
        iconColor: 'rgba(0, 211, 255, 1)',
        name: 'Annotations & Alerts',
        builtIn: true,
      },
    },
  ],
  cursorSync: 'Crosshair',
  editable: true,
  links: [],
  liveNow: false,
  preload: false,
  timeSettings: {
    timezone: 'browser',
    from: 'now-15m',
    to: 'now',
    autoRefresh: '',
    autoRefreshIntervals: [
      '5s',
      '10s',
      '30s',
      '1m',
      '5m',
      '15m',
      '30m',
      '1h',
      '2h',
      '1d',
    ],
    hideTimepicker: false,
    fiscalYearStartMonth: 0,
  },
};

// Reusable trainer-side content shared by every verl trainer composition.
local verlBase = [trainer, controller, storage, trajectory];

{
  compositions: {
    // Historical output filename (`tainer`) is kept on purpose.
    verl_tainer_v1_with_vllm_engine: {
      modules: verlBase + [vllm, npu],
      dashboard: {
        metadata: {
          name: 'a4cabbce-92af-4a84-9399-4deba24ae6d1',
          generation: 45,
          creationTimestamp: '2026-07-08T14:34:31Z',
          labels: {},
          annotations: {},
        },
        title: 'verl_trainer_v1_with_vllm_engine',
        tags: [
          'RL-Insight',
          'verl',
          'vllm',
        ],
        spec: productionSpec,
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
    },
    // Historical output filename (`tainer`) is kept on purpose.
    verl_tainer_v1_with_sglang_engine: {
      modules: verlBase + [sglang],
      dashboard: {
        metadata: {
          name: 'd436bdfd-f96e-4c62-b0b1-8e8315c1757a',
          generation: 1,
          creationTimestamp: '2026-07-08T14:34:31Z',
          labels: {},
          annotations: {},
        },
        title: 'verl_trainer_v1_with_sglang_engine',
        tags: [
          'RL-Insight',
          'verl',
          'sglang',
        ],
        spec: productionSpec,
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
    },
  },
}
