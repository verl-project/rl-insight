// Transfer-queue storage and data-transfer metrics.
{
  panels: [
    {
      key: 'storage.partition_status.samples_per_partition',
      outputKey: 'panel-302',
      id: 302,
      title: 'Samples per Partition',
      queries: [
        {
          expr: 'tq_partition_samples_total',
          editorMode: 'code',
          legend: '{{ partition_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.partition_status.production_progress',
      outputKey: 'panel-303',
      id: 303,
      title: 'Production Progress',
      queries: [
        {
          expr: 'tq_partition_production_progress{task_name=~"$task_name"}',
          editorMode: 'code',
          legend: '{{ partition_id }} / {{ task_name }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'percentunit',
              min: 0,
              max: 1,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              color: {
                mode: 'continuous-GrYlRd',
              },
              custom: {
                fillOpacity: 20,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.partition_status.consumption_progress',
      outputKey: 'panel-304',
      id: 304,
      title: 'Consumption Progress',
      queries: [
        {
          expr: 'tq_partition_consumption_progress{task_name=~"$task_name"}',
          editorMode: 'code',
          legend: '{{ partition_id }} / {{ task_name }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'percentunit',
              min: 0,
              max: 1,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              color: {
                mode: 'continuous-GrYlRd',
              },
              custom: {
                fillOpacity: 20,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.storage_utilization',
      outputKey: 'panel-305',
      id: 305,
      title: 'Storage Utilization',
      queries: [
        {
          expr: 'tq_storage_utilization_ratio',
          editorMode: 'code',
          legend: '{{ storage_unit_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'bargauge',
    },
    {
      key: 'storage.storage_units.active_keys_per_storage_unit',
      outputKey: 'panel-306',
      id: 306,
      title: 'Active Keys per Storage Unit',
      queries: [
        {
          expr: 'tq_storage_active_keys_total',
          editorMode: 'code',
          legend: '{{ storage_unit_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.storage_capacity_vs_active_keys',
      outputKey: 'panel-307',
      id: 307,
      title: 'Storage Capacity vs Active Keys',
      queries: [
        {
          expr: 'tq_storage_capacity_total',
          editorMode: 'code',
          legend: 'capacity {{ storage_unit_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'tq_storage_active_keys_total',
          editorMode: 'code',
          legend: 'active {{ storage_unit_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.storage_process_rss_memory',
      outputKey: 'panel-308',
      id: 308,
      title: 'Storage Process RSS Memory',
      queries: [
        {
          expr: 'tq_storage_memory_rss_bytes',
          editorMode: 'code',
          legend: '{{ storage_unit_id }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'bytes',
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.storage_request_rate_per_second',
      outputKey: 'panel-309',
      id: 309,
      title: 'Storage Request Rate (per second)',
      queries: [
        {
          expr: 'sum by (op_type) (rate(tq_storage_request_ops{op_type=~"$op_type"}[$__rate_interval]))',
          editorMode: 'code',
          legend: '{{ op_type }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'mean',
                'max',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'ops',
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 20,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.storage_request_latency_p50_p99',
      outputKey: 'panel-310',
      id: 310,
      title: 'Storage Request Latency P50 / P99',
      queries: [
        {
          expr: 'tq_storage_request_latency_p50{op_type=~"$op_type"}',
          editorMode: 'code',
          legend: 'p50 {{ op_type }} ({{ storage_unit_id }})',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'tq_storage_request_latency_p99{op_type=~"$op_type"}',
          editorMode: 'code',
          legend: 'p99 {{ op_type }} ({{ storage_unit_id }})',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      transformations: [
        {
          kind: 'Transformation',
          group: 'filterFieldsByName',
          spec: {
            options: {
              include: {
                pattern: 'Time|(${quantile:regex}).*',
              },
            },
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'mean',
                'max',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 's',
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.produced_vs_cleared_samples_per_second',
      outputKey: 'panel-311',
      id: 311,
      title: 'Produced vs Cleared Samples (per second)',
      queries: [
        {
          expr: 'sum(rate(tq_controller_request_samples_total{op_type="NOTIFY_DATA_UPDATE"}[$__rate_interval]))',
          editorMode: 'code',
          legend: 'Produced samples/s',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'sum(rate(tq_controller_request_samples_total{op_type="CLEAR_META"}[$__rate_interval]))',
          editorMode: 'code',
          legend: 'Cleared samples/s',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'mean',
                'max',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'short',
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 15,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'storage.storage_units.active_keys_delta_put_clear_accumulation',
      outputKey: 'panel-312',
      id: 312,
      title: 'Active Keys Delta (PUT - CLEAR accumulation)',
      queries: [
        {
          expr: 'sum(tq_storage_active_keys_total)',
          editorMode: 'code',
          legend: 'Total Active Keys (all storage units)',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            legend: {
              calcs: [
                'lastNotNull',
                'max',
              ],
              displayMode: 'table',
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 'short',
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'green',
                  },
                ],
              },
              color: {
                mode: 'continuous-GrYlRd',
              },
              custom: {
                fillOpacity: 20,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
  ],
  rows: {},
  variables: {
    task_name: {
      kind: 'QueryVariable',
      spec: {
        name: 'task_name',
        current: {
          text: '',
          value: '',
        },
        label: 'transfer queue: Task',
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
            query: 'label_values(tq_partition_consumption_progress, task_name)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values(tq_partition_consumption_progress, task_name)',
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
  },
}
