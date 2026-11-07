// Trajectory state-timeline tracing content.
{
  panels: [
    {
      key: 'trajectory.metric.state_timeline',
      outputKey: 'panel-40',
      id: 40,
      title: 'state timeline',
      queries: [
        {
          raw: {
            kind: 'DataQuery',
            group: 'tempo',
            version: 'v0',
            spec: {
              limit: 10000,
              metricsQueryType: 'range',
              query: '{span.state_lane_id!=""}',
              queryType: 'traceql',
              serviceMapUseNativeHistograms: false,
              spss: 100,
              tableType: 'spans',
            },
          },
        },
      ],
      transformations: [
        {
          kind: 'Transformation',
          group: 'calculateField',
          spec: {
            options: {
              binary: {
                left: {
                  matcher: {
                    id: 'byName',
                    options: 'Duration',
                  },
                },
                operator: '/',
                right: {
                  fixed: '1000000',
                },
              },
              mode: 'binary',
              reduce: {
                reducer: 'sum',
              },
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'calculateField',
          spec: {
            options: {
              alias: 'End time',
              mode: 'reduceRow',
              reduce: {
                include: [
                  'Start time',
                  'Duration / 1000000',
                ],
                reducer: 'sum',
              },
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'extractFields',
          spec: {
            options: {
              source: 'state_lane_id',
              format: 'regexp',
              replace: false,
              regExp: '/^(?<state_lane_prefix>.+?)(?:[_-](?<state_lane_num>\\d+))?$/',
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'convertFieldType',
          spec: {
            options: {
              conversions: [
                {
                  destinationType: 'time',
                  targetField: 'Start time',
                },
                {
                  destinationType: 'time',
                  targetField: 'End time',
                },
                {
                  destinationType: 'number',
                  targetField: 'state_lane_num',
                },
              ],
              fields: {},
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'sortBy',
          spec: {
            options: {
              fields: {},
              sort: [
                {
                  field: 'state_lane_num',
                },
              ],
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'sortBy',
          spec: {
            options: {
              fields: {},
              sort: [
                {
                  field: 'state_lane_prefix',
                },
              ],
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'organize',
          spec: {
            options: {
              excludeByName: {
                Duration: true,
                'Duration / 1000000': true,
                Name: true,
                'Span ID': true,
                'Trace Service': true,
                traceIdHidden: true,
                state_lane_prefix: true,
                state_lane_num: true,
              },
              includeByName: {},
              indexByName: {},
              renameByName: {},
            },
          },
        },
        {
          kind: 'Transformation',
          group: 'partitionByValues',
          spec: {
            options: {
              fields: [
                'state_lane_id',
              ],
              keepFields: false,
            },
          },
        },
      ],
      vizBase: 'state-timeline',
    },
  ],
  rows: {
    'rl state timeline': {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'rl state timeline',
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
                  height: 18,
                  element: {
                    kind: 'ElementReference',
                    name: 'trajectory.metric.state_timeline',
                  },
                },
              },
            ],
          },
        },
      },
    },
  },
  variables: {},
}
