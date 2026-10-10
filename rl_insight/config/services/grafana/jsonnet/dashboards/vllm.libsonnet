// vLLM inference content plus the CPU portion of its unchanged hardware row.
{
  panels: [
    {
      key: 'engine.vllm.metric.vllm_token_throughput',
      outputKey: 'panel-1',
      id: 1,
      title: 'vLLM: Token Throughput',
      queries: [
        {
          expr: 'sum by (model_name, engine, replica) (rate(vllm:request_prompt_tokens_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: 'Prompt Tokens/Sec - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'sum by (model_name, engine, replica) (rate(vllm:generation_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: 'Generation Tokens/Sec - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Number of tokens processed per second',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'tokens/s',
              min: 0,
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_time_per_output_token_latency',
      outputKey: 'panel-2',
      id: 2,
      title: 'vLLM: Time Per Output Token Latency',
      queries: [
        {
          expr: 'histogram_quantile(0.99, sum by(le, model_name, engine) (rate(vllm:request_time_per_output_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P99 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.95, sum by(le, model_name, engine) (rate(vllm:request_time_per_output_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P95 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.9, sum by(le, model_name, engine) (rate(vllm:request_time_per_output_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P90 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.5, sum by(le, model_name, engine) (rate(vllm:request_time_per_output_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P50 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: '(sum by(model_name, engine) (rate(vllm:request_time_per_output_token_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))\n/\nsum by(model_name, engine) (rate(vllm:request_time_per_output_token_seconds_count{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'Mean - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Time per output token latency.',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
            legend: {
              calcs: [
                'lastNotNull',
              ],
              displayMode: 'table',
              sortBy: 'Last *',
              sortDesc: false,
            },
            tooltip: {
              mode: 'multi',
            },
          },
          fieldConfig: {
            defaults: {
              unit: 's',
              min: 0,
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_cache_utilization',
      outputKey: 'panel-3',
      id: 3,
      title: 'vLLM: Cache Utilization',
      queries: [
        {
          expr: 'vllm:kv_cache_usage_perc{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}',
          editorMode: 'code',
          legend: 'KV Cache Usage - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Percentage of used cache blocks by vLLM.',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_time_to_first_token_latency',
      outputKey: 'panel-5',
      id: 5,
      title: 'vLLM: Time To First Token Latency',
      queries: [
        {
          expr: '(sum by(model_name, engine) (rate(vllm:time_to_first_token_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))\n/\nsum by(model_name, engine) (rate(vllm:time_to_first_token_seconds_count{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'Average - {{model_name}} - {{engine}} - replica $replica ',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.5, sum by(le, model_name, engine)(rate(vllm:time_to_first_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P50 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.9, sum by(le, model_name, engine)(rate(vllm:time_to_first_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P90 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.95, sum by(le, model_name, engine) (rate(vllm:time_to_first_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P95 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.99, sum by(le, model_name, engine)(rate(vllm:time_to_first_token_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P99 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'P50, P90, P95, and P99 TTFT latency.',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 's',
              min: 0,
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_e2e_request_latency',
      outputKey: 'panel-6',
      id: 6,
      title: 'vLLM: E2E Request Latency',
      queries: [
        {
          expr: 'sum by(model_name, engine) (rate(vllm:e2e_request_latency_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))\n/\nsum by(model_name, engine) (rate(vllm:e2e_request_latency_seconds_count{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: 'Average - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.5, sum by(le, model_name, engine) (rate(vllm:e2e_request_latency_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P50 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.9, sum by(le, model_name, engine) (rate(vllm:e2e_request_latency_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P90 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.95, sum by(le, model_name, engine) (rate(vllm:e2e_request_latency_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P95 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'histogram_quantile(0.99, sum by(le, model_name, engine) (rate(vllm:e2e_request_latency_seconds_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])))',
          editorMode: 'code',
          legend: 'P99 - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Latency from request start to first token returned (in seconds).',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 's',
              min: 0,
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_scheduler_state',
      outputKey: 'panel-7',
      id: 7,
      title: 'vLLM: Scheduler State',
      queries: [
        {
          expr: 'vllm:num_requests_running{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}',
          editorMode: 'code',
          legend: 'Num Running - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'vllm:num_requests_swapped{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}',
          editorMode: 'code',
          legend: 'Num Swapped - {{model_name}} - {{engine}} -  replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'vllm:num_requests_waiting{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}',
          editorMode: 'code',
          legend: 'Num Waiting - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Number of requests in RUNNING, WAITING, and SWAPPED state',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'Requests',
              min: 0,
              custom: {
                fillOpacity: 10,
              },
            },
            overrides: [
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#1F60C4',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byName',
                  options: 'MAX + PENDING',
                },
                properties: [
                  {
                    id: 'color',
                    value: {
                      fixedColor: '#777777',
                      mode: 'fixed',
                    },
                  },
                  {
                    id: 'custom.fillOpacity',
                    value: 0,
                  },
                  {
                    id: 'custom.stacking',
                    value: {
                      group: 'A',
                      mode: 'none',
                    },
                  },
                  {
                    id: 'custom.lineStyle',
                    value: {
                      dash: [
                        10,
                        10,
                      ],
                      fill: 'dash',
                    },
                  },
                ],
              },
              {
                matcher: {
                  id: 'byValue',
                  options: {
                    op: 'gte',
                    reducer: 'allIsZero',
                    value: 0,
                  },
                },
                properties: [
                  {
                    id: 'custom.hideFrom',
                    value: {
                      legend: true,
                      tooltip: true,
                      viz: false,
                    },
                  },
                ],
              },
            ],
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_request_prompt_length',
      outputKey: 'panel-8',
      id: 8,
      title: 'vLLM: Request Prompt Length',
      queries: [
        {
          expr: 'sum by(le, model_name, engine, replica) (increase(vllm:request_prompt_tokens_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{le}} - replica {{replica}}',
          extra: {
            format: 'heatmap',
            fullMetaSearch: false,
            includeNullMetadata: true,
            instant: false,
            useBackend: false,
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Heatmap of request prompt length',
      vizBase: 'heatmap',
    },
    {
      key: 'engine.vllm.metric.vllm_request_generation_length',
      outputKey: 'panel-9',
      id: 9,
      title: 'vLLM: Request Generation Length',
      queries: [
        {
          expr: 'sum by(le, model_name, engine, replica) (increase(vllm:request_generation_tokens_bucket{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{le}} - replica {{replica}}',
          extra: {
            format: 'heatmap',
            fullMetaSearch: false,
            includeNullMetadata: true,
            instant: false,
            useBackend: false,
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Heatmap of request generation length',
      vizBase: 'heatmap',
    },
    {
      key: 'engine.vllm.metric.vllm_finish_reason',
      outputKey: 'panel-10',
      id: 10,
      title: 'vLLM: Finish Reason',
      queries: [
        {
          expr: 'sum by(finished_reason, model_name, engine) (increase(vllm:request_success_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{finished_reason}} - {{model_name}} - {{engine}} - replica $replica',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Number of finished requests by their finish reason: either an EOS token was generated or the max sequence length was reached.',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
          },
          fieldConfig: {
            defaults: {
              custom: {
                fillOpacity: 0,
                lineWidth: 1,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_queue_time',
      outputKey: 'panel-11',
      id: 11,
      title: 'vLLM: Queue Time',
      queries: [
        {
          expr: 'sum by(model_name, engine, replica) (rate(vllm:request_queue_time_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
          },
          fieldConfig: {
            defaults: {
              custom: {
                fillOpacity: 0,
                lineWidth: 1,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_requests_prefill_and_decode_time',
      outputKey: 'panel-12',
      id: 12,
      title: 'vLLM: Requests Prefill and Decode Time',
      queries: [
        {
          expr: 'sum by(model_name, engine, replica) (rate(vllm:request_decode_time_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: 'Decode - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'sum by(model_name, engine, replica) (rate(vllm:request_prefill_time_seconds_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: 'Prefill - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
          },
          fieldConfig: {
            defaults: {
              custom: {
                fillOpacity: 0,
                lineWidth: 1,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_max_generation_token_in_sequence_group',
      outputKey: 'panel-13',
      id: 13,
      title: 'vLLM: Max Generation Token in Sequence Group',
      queries: [
        {
          expr: 'sum by(model_name, engine, replica) (rate(vllm:request_max_num_generation_tokens_sum{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
          },
          fieldConfig: {
            defaults: {
              custom: {
                fillOpacity: 0,
                lineWidth: 1,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.vllm_prefix_cache_hit_rate',
      outputKey: 'panel-28',
      id: 28,
      title: 'vLLM: Prefix Cache Hit Rate',
      queries: [
        {
          expr: 'increase(vllm:prefix_cache_hits_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]) / increase(vllm:prefix_cache_queries_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])',
          editorMode: 'code',
          legend: 'Prefix Cache Hit Rate - {{model_name}} - {{engine}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Percentage of prefix cache queries that resulted in a cache hit (GPU).',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
          },
          fieldConfig: {
            defaults: {
              custom: {
                fillOpacity: 0,
                lineWidth: 1,
                showPoints: 'auto',
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_accepted_vs_drafted_throughput',
      outputKey: 'panel-30',
      id: 30,
      title: 'SpecDecoding: Accepted vs Drafted Throughput',
      queries: [
        {
          expr: 'rate(vllm:spec_decode_num_accepted_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])',
          editorMode: 'code',
          legend: 'Accepted Throughput - {{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'rate(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])',
          editorMode: 'code',
          legend: 'Drafted Throughput - {{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Speculative Decoding metrics showing draft and accepted token rates',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'tokens/s',
              min: 0,
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
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_draft_acceptance_rate.2',
      outputKey: 'panel-31',
      id: 31,
      title: 'SpecDecoding: Draft Acceptance Rate',
      queries: [
        {
          expr: 'sum by (model_name, replica) (rate(vllm:spec_decode_num_accepted_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) / sum by (model_name, replica) (rate(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) * 100',
          editorMode: 'code',
          legend: '{{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Speculative Decoding acceptance rate percentage',
      vizBase: 'stat',
      vizPatch: {
        spec: {
          options: {
            graphMode: 'area',
          },
          fieldConfig: {
            defaults: {
              unit: 'percent',
              min: 0,
              max: 100,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'red',
                  },
                  {
                    value: 80,
                    color: 'yellow',
                  },
                  {
                    value: 90,
                    color: 'green',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_accepted_vs_drafted_tokens',
      outputKey: 'panel-32',
      id: 32,
      title: 'SpecDecoding: Accepted vs Drafted Tokens',
      queries: [
        {
          expr: 'increase(vllm:spec_decode_num_accepted_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])',
          editorMode: 'code',
          legend: 'Accepted - {{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'increase(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])',
          editorMode: 'code',
          legend: 'Drafted - {{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Total accepted and drafted tokens in speculative decoding',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'short',
              min: 0,
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
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_draft_acceptance_rate',
      outputKey: 'panel-33',
      id: 33,
      title: 'SpecDecoding: Draft Acceptance Rate',
      queries: [
        {
          expr: 'sum by (model_name, replica) (rate(vllm:spec_decode_num_accepted_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) / sum by (model_name, replica) (rate(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) * 100',
          editorMode: 'code',
          legend: '{{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Speculative Decoding acceptance rate percentage',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'percent',
              min: 0,
              max: 100,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'red',
                  },
                  {
                    value: 80,
                    color: 'yellow',
                  },
                  {
                    value: 90,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                thresholdsStyle: {
                  mode: 'line',
                },
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_mean_acceptance_length',
      outputKey: 'panel-35',
      id: 35,
      title: 'SpecDecoding: Mean Acceptance Length',
      queries: [
        {
          expr: 'sum by (model_name, replica) (rate(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) / sum by (model_name, replica) (rate(vllm:spec_decode_num_drafts_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Mean number of drafts per speculation',
      vizPatch: {
        spec: {
          options: {
            alertThreshold: true,
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
              unit: 'short',
              min: 0,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'super-light-yellow',
                  },
                  {
                    value: 1,
                    color: 'super-light-green',
                  },
                  {
                    value: 2,
                    color: 'green',
                  },
                ],
              },
              custom: {
                fillOpacity: 10,
                thresholdsStyle: {
                  mode: 'line',
                },
              },
            },
          },
        },
      },
    },
    {
      key: 'engine.vllm.metric.specdecoding_mean_acceptance_length.2',
      outputKey: 'panel-36',
      id: 36,
      title: 'SpecDecoding: Mean Acceptance Length',
      queries: [
        {
          expr: 'sum by (model_name, replica) (rate(vllm:spec_decode_num_draft_tokens_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval])) / sum by (model_name, replica) (rate(vllm:spec_decode_num_drafts_total{model_name=~"$vllm_model_name", engine=~"$workerid", replica=~"$replica"}[$interval]))',
          editorMode: 'code',
          legend: '{{model_name}} - replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Mean number of drafts per speculation',
      vizBase: 'stat',
      vizPatch: {
        spec: {
          options: {
            graphMode: 'area',
          },
          fieldConfig: {
            defaults: {
              unit: 'short',
              min: 0,
              thresholds: {
                steps: [
                  {
                    value: 0,
                    color: 'super-light-yellow',
                  },
                  {
                    value: 1,
                    color: 'super-light-green',
                  },
                  {
                    value: 2,
                    color: 'green',
                  },
                ],
              },
            },
          },
        },
      },
    },
    {
      key: 'hardware.cpu_host_metrics.cpu_utilization',
      outputKey: 'panel-325',
      id: 325,
      title: 'CPU Utilization',
      queries: [
        {
          expr: '100 - avg by (node, instance) (rate(node_cpu_seconds_total{job="node-exporter", mode="idle"}[$__rate_interval])) * 100',
          editorMode: 'code',
          legend: '{{node}} ({{instance}})',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'CPU utilization for all node_exporter targets',
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
      key: 'hardware.cpu_host_metrics.memory_used',
      outputKey: 'panel-326',
      id: 326,
      title: 'Memory Used',
      queries: [
        {
          expr: 'node_memory_MemTotal_bytes{job="node-exporter"} - node_memory_MemAvailable_bytes{job="node-exporter"}',
          editorMode: 'code',
          legend: '{{node}} ({{instance}})',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Used memory for all node_exporter targets',
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
      key: 'hardware.cpu_host_metrics.network_throughput',
      outputKey: 'panel-328',
      id: 328,
      title: 'Network Throughput',
      queries: [
        {
          expr: 'sum by (node, instance) (rate(node_network_receive_bytes_total{job="node-exporter", device!~"lo"}[$__rate_interval]))',
          editorMode: 'code',
          legend: '{{node}} RX',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
        {
          expr: 'sum by (node, instance) (rate(node_network_transmit_bytes_total{job="node-exporter", device!~"lo"}[$__rate_interval]))',
          editorMode: 'code',
          legend: '{{node}} TX',
          labels: {
            'grafana.app/export-label': 'prometheus-1',
          },
        },
      ],
      description: 'Receive and transmit throughput for all node_exporter targets',
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
              unit: 'Bps',
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
  rows: {
    'hardware metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'hardware metric',
        collapse: true,
        layout: {
          kind: 'RowsLayout',
          spec: {
            rows: [
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'CPU / Host Metrics',
                  collapse: false,
                  layout: {
                    kind: 'AutoGridLayout',
                    spec: {
                      maxColumnCount: 2,
                      columnWidthMode: 'standard',
                      rowHeightMode: 'standard',
                      items: [
                        {
                          kind: 'AutoGridLayoutItem',
                          spec: {
                            element: {
                              kind: 'ElementReference',
                              name: 'hardware.cpu_host_metrics.cpu_utilization',
                            },
                            conditionalRendering: {
                              kind: 'ConditionalRenderingGroup',
                              spec: {
                                visibility: 'hide',
                                condition: 'and',
                                items: [
                                  {
                                    kind: 'ConditionalRenderingData',
                                    spec: {
                                      value: false,
                                    },
                                  },
                                ],
                              },
                            },
                          },
                        },
                        {
                          kind: 'AutoGridLayoutItem',
                          spec: {
                            element: {
                              kind: 'ElementReference',
                              name: 'hardware.cpu_host_metrics.memory_used',
                            },
                            conditionalRendering: {
                              kind: 'ConditionalRenderingGroup',
                              spec: {
                                visibility: 'hide',
                                condition: 'and',
                                items: [
                                  {
                                    kind: 'ConditionalRenderingData',
                                    spec: {
                                      value: false,
                                    },
                                  },
                                ],
                              },
                            },
                          },
                        },
                        {
                          kind: 'AutoGridLayoutItem',
                          spec: {
                            element: {
                              kind: 'ElementReference',
                              name: 'hardware.cpu_host_metrics.network_throughput',
                            },
                            conditionalRendering: {
                              kind: 'ConditionalRenderingGroup',
                              spec: {
                                visibility: 'hide',
                                condition: 'and',
                                items: [
                                  {
                                    kind: 'ConditionalRenderingData',
                                    spec: {
                                      value: false,
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
              {
                kind: 'RowsLayoutRow',
                spec: {
                  title: 'Ascend NPU Metrics',
                  collapse: false,
                  layout: {
                    kind: 'TabsLayout',
                    spec: {
                      tabs: [
                        {
                          kind: 'TabsLayoutTab',
                          spec: {
                            title: '$npu_instance',
                            repeat: {
                              mode: 'variable',
                              value: 'npu_instance',
                            },
                            layout: {
                              kind: 'AutoGridLayout',
                              spec: {
                                maxColumnCount: 2,
                                columnWidthMode: 'standard',
                                rowHeightMode: 'standard',
                                items: [
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_ai_core_utilization',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_hbm_utilization',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_hbm_used_memory',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_power',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_temperature',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_health_status',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_hbm_bandwidth_utilization',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
                                              },
                                            },
                                          ],
                                        },
                                      },
                                    },
                                  },
                                  {
                                    kind: 'AutoGridLayoutItem',
                                    spec: {
                                      element: {
                                        kind: 'ElementReference',
                                        name: 'hardware.ascend_npu_metrics.npu_network_throughput',
                                      },
                                      conditionalRendering: {
                                        kind: 'ConditionalRenderingGroup',
                                        spec: {
                                          visibility: 'hide',
                                          condition: 'and',
                                          items: [
                                            {
                                              kind: 'ConditionalRenderingData',
                                              spec: {
                                                value: false,
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
    'vllm engine metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'vllm engine metric',
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
                    name: 'engine.vllm.metric.vllm_token_throughput',
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
                    name: 'engine.vllm.metric.vllm_time_per_output_token_latency',
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
                    name: 'engine.vllm.metric.vllm_cache_utilization',
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
                    name: 'engine.vllm.metric.vllm_time_to_first_token_latency',
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
                    name: 'engine.vllm.metric.vllm_e2e_request_latency',
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
                    name: 'engine.vllm.metric.vllm_scheduler_state',
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
                    name: 'engine.vllm.metric.vllm_request_prompt_length',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 24,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.vllm_request_generation_length',
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
                    name: 'engine.vllm.metric.vllm_finish_reason',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 32,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.vllm_queue_time',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 0,
                  y: 40,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.vllm_requests_prefill_and_decode_time',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 40,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.vllm_max_generation_token_in_sequence_group',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 0,
                  y: 48,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.vllm_prefix_cache_hit_rate',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 48,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_mean_acceptance_length',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 0,
                  y: 56,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_accepted_vs_drafted_throughput',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 56,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_draft_acceptance_rate',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 0,
                  y: 64,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_accepted_vs_drafted_tokens',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 12,
                  y: 64,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_draft_acceptance_rate.2',
                  },
                },
              },
              {
                kind: 'GridLayoutItem',
                spec: {
                  x: 0,
                  y: 72,
                  width: 12,
                  height: 8,
                  element: {
                    kind: 'ElementReference',
                    name: 'engine.vllm.metric.specdecoding_mean_acceptance_length.2',
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
    interval: {
      kind: 'CustomVariable',
      spec: {
        name: 'interval',
        query: '',
        current: {
          text: '5m',
          value: '5m',
        },
        options: [],
        multi: false,
        includeAll: false,
        label: 'vllm: Interval',
        hide: 'dontHide',
        skipUrlSync: false,
        allowCustomValue: true,
        valuesFormat: 'csv',
      },
    },
    replica: {
      kind: 'QueryVariable',
      spec: {
        name: 'replica',
        current: {
          text: '',
          value: '',
        },
        label: 'vllm: Rollout Replica',
        hide: 'dontHide',
        refresh: 'onDashboardLoad',
        skipUrlSync: false,
        description: 'Replica rank of rollout',
        query: {
          kind: 'DataQuery',
          group: 'prometheus',
          version: 'v0',
          spec: {
            qryType: 1,
            query: 'label_values(vllm:request_prompt_tokens_sum,replica)',
            refId: 'PrometheusVariableQueryEditor-VariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values(vllm:request_prompt_tokens_sum,replica)',
        options: [],
        multi: false,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
    vllm_model_name: {
      kind: 'QueryVariable',
      spec: {
        name: 'vllm_model_name',
        current: {
          text: '',
          value: '',
        },
        label: 'vllm: vLLM Model Name',
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
            query: 'label_values(vllm:request_prompt_tokens_sum{}, model_name)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values(vllm:request_prompt_tokens_sum{}, model_name)',
        options: [],
        multi: false,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
    workerid: {
      kind: 'QueryVariable',
      spec: {
        name: 'workerid',
        current: {
          text: '',
          value: '',
        },
        label: 'vllm: vLLM Engine ID',
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
            query: 'label_values(vllm:request_prompt_tokens_sum{}, engine)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values(vllm:request_prompt_tokens_sum{}, engine)',
        options: [],
        multi: false,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
  },
}
