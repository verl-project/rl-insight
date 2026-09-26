// Transfer-queue controller metrics and the unchanged combined transfer-queue row.
{
  panels: [
    {
      key: 'controller.controller_overview.controller_uptime',
      outputKey: 'panel-295',
      id: 295,
      title: 'Controller Uptime',
      queries: [
        {
          expr: 'tq_controller_uptime_seconds',
          editorMode: 'code',
          legend: '',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'stat',
    },
    {
      key: 'controller.controller_overview.controller_rss_memory',
      outputKey: 'panel-296',
      id: 296,
      title: 'Controller RSS Memory',
      queries: [
        {
          expr: 'tq_controller_memory_rss_bytes',
          editorMode: 'code',
          legend: '',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'stat',
      vizPatch: {
        spec: {
          options: {
            graphMode: 'area',
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
                  {
                    value: 2147483648,
                    color: 'yellow',
                  },
                  {
                    value: 4294967296,
                    color: 'red',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'controller.controller_overview.active_partitions',
      outputKey: 'panel-297',
      id: 297,
      title: 'Active Partitions',
      queries: [
        {
          expr: 'tq_partitions_total',
          editorMode: 'code',
          legend: '',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'stat',
      vizPatch: {
        spec: {
          fieldConfig: {
            defaults: {
              unit: null,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'blue',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'controller.controller_overview.global_indexes_allocated',
      outputKey: 'panel-298',
      id: 298,
      title: 'Global Indexes Allocated',
      queries: [
        {
          expr: 'tq_global_index_allocated_total',
          editorMode: 'code',
          legend: '',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'stat',
      vizPatch: {
        spec: {
          fieldConfig: {
            defaults: {
              unit: null,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'purple',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'controller.controller_overview.reusable_indexes',
      outputKey: 'panel-299',
      id: 299,
      title: 'Reusable Indexes',
      queries: [
        {
          expr: 'tq_global_index_reusable_total',
          editorMode: 'code',
          legend: '',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizBase: 'stat',
      vizPatch: {
        spec: {
          fieldConfig: {
            defaults: {
              unit: null,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'orange',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'controller.request_throughput_latency.controller_request_rate_per_second',
      outputKey: 'panel-300',
      id: 300,
      title: 'Controller Request Rate (per second)',
      queries: [
        {
          expr: 'sum by (op_type) (rate(tq_controller_request_total{op_type=~"$op_type"}[$__rate_interval]))',
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
      key: 'controller.request_throughput_latency.controller_request_latency_p50_p99',
      outputKey: 'panel-301',
      id: 301,
      title: 'Controller Request Latency P50 / P99',
      queries: [
        {
          expr: 'histogram_quantile(0.50, sum by (op_type, le) (rate(tq_controller_request_duration_seconds_bucket{op_type=~"$op_type"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'p50 {{ op_type }}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.99, sum by (op_type, le) (rate(tq_controller_request_duration_seconds_bucket{op_type=~"$op_type"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'p99 {{ op_type }}',
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
  ],
  rows: {
    'transfer queue metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'transfer queue metric',
        collapse: true,
        layout: {
          kind: 'RowsLayout',
          spec: {
            rows: [
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'Controller Overview',
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
                            width: 4,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.controller_overview.controller_uptime',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 4,
                            y: 0,
                            width: 4,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.controller_overview.controller_rss_memory',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 8,
                            y: 0,
                            width: 4,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.controller_overview.active_partitions',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 12,
                            y: 0,
                            width: 4,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.controller_overview.global_indexes_allocated',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 16,
                            y: 0,
                            width: 4,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.controller_overview.reusable_indexes',
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
                  title: 'Request Throughput & Latency',
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
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.request_throughput_latency.controller_request_rate_per_second',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 12,
                            y: 0,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'controller.request_throughput_latency.controller_request_latency_p50_p99',
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
                  title: 'Partition Status',
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
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.partition_status.samples_per_partition',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 12,
                            y: 0,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.partition_status.consumption_progress',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.partition_status.production_progress',
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
                  title: 'Storage Units',
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
                            width: 24,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.storage_utilization',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 8,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.active_keys_per_storage_unit',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 12,
                            y: 8,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.storage_capacity_vs_active_keys',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 16,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.storage_process_rss_memory',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 12,
                            y: 16,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.storage_request_latency_p50_p99',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 24,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.storage_request_rate_per_second',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 32,
                            width: 12,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.produced_vs_cleared_samples_per_second',
                            },
                          },
                        },
                        {
                          kind: 'GridLayoutItem',
                          spec: {
                            x: 0,
                            y: 40,
                            width: 24,
                            height: 8,
                            element: {
                              kind: 'ElementReference',
                              name: 'storage.storage_units.active_keys_delta_put_clear_accumulation',
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
    op_type: {
      kind: 'CustomVariable',
      spec: {
        name: 'op_type',
        query: 'PUT_DATA,GET_DATA,CLEAR_DATA,GET_META,CLEAR_META,NOTIFY_DATA_UPDATE',
        current: {
          text: 'All',
          value: '$__all',
        },
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        label: 'transfer queue: Op Type',
        hide: 'dontHide',
        skipUrlSync: false,
        allowCustomValue: true,
        valuesFormat: 'csv',
      },
    },
    quantile: {
      kind: 'CustomVariable',
      spec: {
        name: 'quantile',
        query: 'p50,p99',
        current: {
          text: 'All',
          value: '$__all',
        },
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        label: 'transfer queue: Quantile',
        hide: 'dontHide',
        skipUrlSync: false,
        allowCustomValue: true,
        valuesFormat: 'csv',
      },
    },
  },
}
