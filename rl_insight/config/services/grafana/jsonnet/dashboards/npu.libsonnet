// Optional Ascend NPU panels and selector used by the vLLM dashboard.
{
  panels: [
    {
      key: 'hardware.ascend_npu_metrics.npu_ai_core_utilization',
      outputKey: 'panel-329',
      id: 329,
      title: 'NPU AI Core Utilization',
      queries: [
        {
          expr: 'npu_chip_info_utilization{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'AI Core utilization by NPU',
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
              unit: 'percent',
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
              min: 0,
              max: 100,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_hbm_utilization',
      outputKey: 'panel-330',
      id: 330,
      title: 'NPU HBM Utilization',
      queries: [
        {
          expr: 'npu_chip_info_hbm_utilization{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'HBM utilization by NPU',
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
              unit: 'percent',
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
              min: 0,
              max: 100,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_hbm_used_memory',
      outputKey: 'panel-331',
      id: 331,
      title: 'NPU HBM Used Memory',
      queries: [
        {
          expr: 'npu_chip_info_hbm_used_memory{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Used HBM memory by NPU',
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
              unit: 'decmbytes',
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
              min: 0,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_power',
      outputKey: 'panel-332',
      id: 332,
      title: 'NPU Power',
      queries: [
        {
          expr: 'npu_chip_info_power{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Power consumption by NPU',
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
              unit: 'watt',
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
              min: 0,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_temperature',
      outputKey: 'panel-333',
      id: 333,
      title: 'NPU Temperature',
      queries: [
        {
          expr: 'npu_chip_info_temperature{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Chip temperature by NPU',
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
              unit: 'celsius',
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
              min: 0,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_health_status',
      outputKey: 'panel-334',
      id: 334,
      title: 'NPU Health Status',
      queries: [
        {
          expr: 'npu_chip_info_health_status{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Health status code reported by NPU Exporter',
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
                fillOpacity: 20,
                showPoints: 'auto',
              },
              min: 0,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_hbm_bandwidth_utilization',
      outputKey: 'panel-335',
      id: 335,
      title: 'NPU HBM Bandwidth Utilization',
      queries: [
        {
          expr: 'npu_chip_info_hbm_bandwidth_utilization{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}}',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'HBM bandwidth utilization by NPU',
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
              unit: 'percent',
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
              min: 0,
              max: 100,
            },
          },
        },
      },
    },
    {
      key: 'hardware.ascend_npu_metrics.npu_network_throughput',
      outputKey: 'panel-336',
      id: 336,
      title: 'NPU Network Throughput',
      queries: [
        {
          expr: 'npu_chip_info_bandwidth_rx{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}} RX',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'npu_chip_info_bandwidth_tx{job="npu-exporter", instance=~"$npu_instance"}',
          editorMode: 'code',
          legend: 'NPU {{id}} TX',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'NPU network receive and transmit throughput',
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
              unit: 'MBs',
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
              min: 0,
            },
          },
        },
      },
    },
  ],
  rows: {},
  variables: {
    npu_instance: {
      kind: 'QueryVariable',
      spec: {
        name: 'npu_instance',
        current: {
          text: 'All',
          value: '$__all',
        },
        label: 'hardware: NPU Node',
        hide: 'hideVariable',
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
            query: 'label_values(npu_chip_info_name{job="npu-exporter"}, instance)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'alphabeticalAsc',
        definition: 'label_values(npu_chip_info_name{job="npu-exporter"}, instance)',
        options: [],
        multi: true,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: false,
      },
    },
  },
}
