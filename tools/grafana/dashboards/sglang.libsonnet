// SGLang-specific inference content.
{
  panels: [
    {
      key: 'engine.sglang.metric.sglang_end_to_end_request_latency',
      outputKey: 'panel-1',
      id: 1,
      title: 'SGLang: End-to-End Request Latency',
      queries: [
        {
          expr: 'histogram_quantile(0.99, sum by (le, replica) (rate({__name__=~"sglang[:_]e2e_request_latency_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P99',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'histogram_quantile(0.9, sum by (le, replica) (rate({__name__=~"sglang[:_]e2e_request_latency_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P90',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'histogram_quantile(0.5, sum by (le, replica) (rate({__name__=~"sglang[:_]e2e_request_latency_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P50',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'sum by (replica) (rate({__name__=~"sglang[:_]e2e_request_latency_seconds_sum", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])) / sum by (replica) (rate({__name__=~"sglang[:_]e2e_request_latency_seconds_count", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval]))',
          editorMode: 'code',
          legend: 'replica {{replica}} Avg',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'End-to-end request latency from SGLang metrics (official dashboard query; metric prefix sglang:).',
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
      key: 'engine.sglang.metric.sglang_end_to_end_request_latency_heatmap',
      outputKey: 'panel-2',
      id: 2,
      title: 'SGLang: End-to-End Request Latency Heatmap',
      queries: [
        {
          expr: 'sum by (le, replica) (increase({__name__=~"sglang[:_]e2e_request_latency_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval]))',
          legend: 'replica {{replica}} le {{le}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
            format: 'heatmap',
            fullMetaSearch: false,
            includeNullMetadata: true,
            useBackend: false,
          },
        },
      ],
      description: 'Heatmap of end-to-end request latency buckets.',
      vizBase: 'heatmap',
      vizPatch: {
        spec: {
          options: {
            yAxis: {
              unit: 's',
            },
          },
        },
      },
    },
    {
      key: 'engine.sglang.metric.sglang_time_to_first_token_latency',
      outputKey: 'panel-3',
      id: 3,
      title: 'SGLang: Time-To-First-Token Latency',
      queries: [
        {
          expr: 'histogram_quantile(0.99, sum by (le, replica) (rate({__name__=~"sglang[:_]time_to_first_token_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P99',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'histogram_quantile(0.9, sum by (le, replica) (rate({__name__=~"sglang[:_]time_to_first_token_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P90',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'histogram_quantile(0.5, sum by (le, replica) (rate({__name__=~"sglang[:_]time_to_first_token_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])))',
          editorMode: 'code',
          legend: 'replica {{replica}} P50',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
        {
          expr: 'sum by (replica) (rate({__name__=~"sglang[:_]time_to_first_token_seconds_sum", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval])) / sum by (replica) (rate({__name__=~"sglang[:_]time_to_first_token_seconds_count", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval]))',
          editorMode: 'code',
          legend: 'replica {{replica}} Avg',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'Time to first token latency from SGLang metrics.',
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
      key: 'engine.sglang.metric.sglang_time_to_first_token_heatmap',
      outputKey: 'panel-4',
      id: 4,
      title: 'SGLang: Time-To-First-Token Heatmap',
      queries: [
        {
          expr: 'sum by (le, replica) (increase({__name__=~"sglang[:_]time_to_first_token_seconds_bucket", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}[$__rate_interval]))',
          legend: 'replica {{replica}} le {{le}}',
          extra: {
            exemplar: false,
            interval: '',
            queryType: 'randomWalk',
            format: 'heatmap',
            fullMetaSearch: false,
            includeNullMetadata: true,
            useBackend: false,
          },
        },
      ],
      description: 'Heatmap of time-to-first-token latency buckets.',
      vizBase: 'heatmap',
      vizPatch: {
        spec: {
          options: {
            yAxis: {
              unit: 's',
            },
          },
        },
      },
    },
    {
      key: 'engine.sglang.metric.sglang_num_running_requests',
      outputKey: 'panel-5',
      id: 5,
      title: 'SGLang: Num Running Requests',
      queries: [
        {
          expr: '{__name__=~"sglang[:_]num_running_reqs", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}',
          editorMode: 'code',
          legend: 'replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'Number of running requests in SGLang.',
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
              unit: 'none',
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
      key: 'engine.sglang.metric.sglang_token_generation_throughput',
      outputKey: 'panel-6',
      id: 6,
      title: 'SGLang: Token Generation Throughput',
      queries: [
        {
          expr: '{__name__=~"sglang[:_]gen_throughput", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}',
          editorMode: 'code',
          legend: 'replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'Token generation throughput (tokens/s).',
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
      key: 'engine.sglang.metric.sglang_cache_hit_rate',
      outputKey: 'panel-7',
      id: 7,
      title: 'SGLang: Cache Hit Rate',
      queries: [
        {
          expr: '{__name__=~"sglang[:_]cache_hit_rate", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}',
          editorMode: 'code',
          legend: 'replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'SGLang cache hit rate.',
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
      key: 'engine.sglang.metric.sglang_number_queued_requests',
      outputKey: 'panel-8',
      id: 8,
      title: 'SGLang: Number Queued Requests',
      queries: [
        {
          expr: '{__name__=~"sglang[:_]num_queue_reqs", model_name=~"$sglang_model_name", replica=~"$sglang_replica"}',
          editorMode: 'code',
          legend: 'replica {{replica}}',
          extra: {
            exemplar: true,
            interval: '',
            queryType: 'randomWalk',
          },
        },
      ],
      description: 'Number of queued requests in SGLang.',
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
              unit: 'none',
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
  ],
  rows: {
    'sglang engine metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'sglang engine metric',
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
                    name: 'engine.sglang.metric.sglang_end_to_end_request_latency',
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
                    name: 'engine.sglang.metric.sglang_end_to_end_request_latency_heatmap',
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
                    name: 'engine.sglang.metric.sglang_time_to_first_token_latency',
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
                    name: 'engine.sglang.metric.sglang_time_to_first_token_heatmap',
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
                    name: 'engine.sglang.metric.sglang_num_running_requests',
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
                    name: 'engine.sglang.metric.sglang_token_generation_throughput',
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
                    name: 'engine.sglang.metric.sglang_cache_hit_rate',
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
                    name: 'engine.sglang.metric.sglang_number_queued_requests',
                  },
                },
              },
            ],
          },
        },
      },
    },
    'device metric': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'device metric',
        collapse: true,
        conditionalRendering: {
          kind: 'ConditionalRenderingGroup',
          spec: {
            visibility: 'hide',
            condition: 'and',
            items: [
              {
                kind: 'ConditionalRenderingVariable',
                spec: {
                  variable: 'datasource',
                  operator: 'equals',
                  value: '',
                },
              },
            ],
          },
        },
        layout: {
          kind: 'AutoGridLayout',
          spec: {
            maxColumnCount: 3,
            columnWidthMode: 'standard',
            rowHeightMode: 'standard',
            items: [],
          },
        },
      },
    },
  },
  variables: {
    sglang_model_name: {
      kind: 'QueryVariable',
      spec: {
        name: 'sglang_model_name',
        current: {
          text: '',
          value: '',
        },
        label: 'sglang: Model Name',
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
            query: 'label_values({__name__=~"sglang[:_]gen_throughput"}, model_name)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values({__name__=~"sglang[:_]gen_throughput"}, model_name)',
        options: [],
        multi: false,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
    sglang_replica: {
      kind: 'QueryVariable',
      spec: {
        name: 'sglang_replica',
        current: {
          text: '',
          value: '',
        },
        label: 'sglang: Rollout Replica',
        hide: 'dontHide',
        refresh: 'onDashboardLoad',
        skipUrlSync: false,
        description: 'Replica rank of SGLang rollout',
        query: {
          kind: 'DataQuery',
          group: 'prometheus',
          version: 'v0',
          datasource: {
            name: '${datasource}',
          },
          spec: {
            query: 'label_values({__name__=~"sglang[:_]gen_throughput"}, replica)',
            refId: 'StandardVariableQuery',
          },
        },
        regex: '',
        regexApplyTo: 'value',
        sort: 'disabled',
        definition: 'label_values({__name__=~"sglang[:_]gen_throughput"}, replica)',
        options: [],
        multi: false,
        includeAll: true,
        allValue: '.*',
        allowCustomValue: true,
      },
    },
  },
}
