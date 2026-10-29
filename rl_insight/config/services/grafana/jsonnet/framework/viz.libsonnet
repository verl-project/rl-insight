// Generated once from the existing dashboards; maintained as Jsonnet source.
{
  heatmap: {
    group: 'heatmap',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          custom: {
            hideFrom: {
              legend: false,
              tooltip: false,
              viz: false,
            },
            scaleDistribution: {
              type: 'linear',
            },
          },
        },
        overrides: [],
      },
      options: {
        annotations: {
          clustering: -1,
          multiLane: false,
        },
        calculate: false,
        cellGap: 1,
        cellValues: {
          unit: 'none',
        },
        color: {
          exponent: 0.5,
          fill: 'dark-orange',
          min: 0,
          mode: 'scheme',
          reverse: false,
          scale: 'exponential',
          scheme: 'Spectral',
          steps: 64,
        },
        exemplars: {
          color: 'rgba(255,0,255,0.7)',
        },
        filterValues: {
          le: 1e-09,
        },
        legend: {
          show: true,
          showLegend: true,
        },
        rowsFrame: {
          layout: 'auto',
          value: 'Request count',
        },
        tooltip: {
          mode: 'single',
          showColorScale: false,
          yHistogram: true,
        },
        yAxis: {
          axisPlacement: 'left',
          reverse: false,
          unit: 'none',
        },
      },
    },
    version: '13.0.2',
  },
  stat: {
    group: 'stat',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          color: {
            mode: 'thresholds',
          },
          thresholds: {
            mode: 'absolute',
            steps: [
              {
                color: 'green',
                value: 0,
              },
            ],
          },
          unit: 's',
        },
        overrides: [],
      },
      options: {
        colorMode: 'value',
        graphMode: 'none',
        justifyMode: 'auto',
        orientation: 'auto',
        percentChangeColorMode: 'standard',
        reduceOptions: {
          calcs: [
            'lastNotNull',
          ],
          fields: '',
          values: false,
        },
        showPercentChange: false,
        text: {},
        textMode: 'auto',
        wideLayout: true,
      },
    },
    version: '13.0.2',
  },
  bargauge: {
    group: 'bargauge',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          color: {
            mode: 'continuous-GrYlRd',
          },
          max: 1,
          min: 0,
          thresholds: {
            mode: 'absolute',
            steps: [
              {
                color: 'green',
                value: 0,
              },
              {
                color: 'yellow',
                value: 0.7,
              },
              {
                color: 'red',
                value: 0.9,
              },
            ],
          },
          unit: 'percentunit',
        },
        overrides: [],
      },
      options: {
        displayMode: 'gradient',
        legend: {
          calcs: [],
          displayMode: 'list',
          placement: 'bottom',
          showLegend: false,
        },
        maxVizHeight: 300,
        minVizHeight: 10,
        minVizWidth: 0,
        namePlacement: 'auto',
        orientation: 'horizontal',
        reduceOptions: {
          calcs: [
            'lastNotNull',
          ],
          fields: '',
          values: false,
        },
        showUnfilled: true,
        sizing: 'auto',
        valueMode: 'color',
      },
    },
    version: '13.0.2',
  },
  'state-timeline': {
    group: 'state-timeline',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          color: {
            mode: 'thresholds',
          },
          custom: {
            axisPlacement: 'auto',
            fillOpacity: 70,
            hideFrom: {
              legend: false,
              tooltip: false,
              viz: false,
            },
            insertNulls: false,
            lineWidth: 0,
            spanNulls: false,
          },
          mappings: [
            {
              options: {
                pattern: '.*compute_log_prob.*',
                result: {
                  color: 'yellow',
                  index: 0,
                },
              },
              type: 'regex',
            },
            {
              options: {
                pattern: '.*generate.*',
                result: {
                  color: 'blue',
                  index: 1,
                },
              },
              type: 'regex',
            },
            {
              options: {
                pattern: '.*compute_ref_log_prob.*',
                result: {
                  color: 'purple',
                  index: 2,
                },
              },
              type: 'regex',
            },
          ],
          thresholds: {
            mode: 'absolute',
            steps: [
              {
                color: 'green',
                value: 0,
              },
            ],
          },
        },
        overrides: [],
      },
      options: {
        alignValue: 'left',
        annotations: {
          clustering: -1,
          multiLane: false,
        },
        legend: {
          displayMode: 'list',
          placement: 'bottom',
          showLegend: true,
        },
        mergeValues: true,
        perPage: 64,
        rowHeight: 0.9,
        showValue: 'auto',
        tooltip: {
          hideZeros: false,
          mode: 'single',
          sort: 'none',
        },
      },
    },
    version: '13.0.2',
  },
  gauge: {
    group: 'gauge',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          color: {
            mode: 'thresholds',
          },
          thresholds: {
            mode: 'absolute',
            steps: [
              {
                color: 'green',
                value: 0,
              },
              {
                color: 'red',
                value: 80,
              },
            ],
          },
        },
        overrides: [],
      },
      options: {
        barShape: 'flat',
        barWidthFactor: 0.5,
        effects: {
          barGlow: false,
          centerGlow: false,
          gradient: true,
        },
        endpointMarker: 'point',
        minVizHeight: 75,
        minVizWidth: 75,
        orientation: 'auto',
        reduceOptions: {
          calcs: [
            'lastNotNull',
          ],
          fields: '',
          values: false,
        },
        segmentCount: 1,
        segmentSpacing: 0.3,
        shape: 'gauge',
        showThresholdLabels: false,
        showThresholdMarkers: true,
        sizing: 'auto',
        sparkline: true,
        textMode: 'auto',
      },
    },
    version: '13.0.2',
  },
  timeseries: {
    group: 'timeseries',
    kind: 'VizConfig',
    spec: {
      fieldConfig: {
        defaults: {
          color: {
            mode: 'palette-classic',
          },
          custom: {
            axisBorderShow: false,
            axisCenteredZero: false,
            axisColorMode: 'text',
            axisLabel: '',
            axisPlacement: 'auto',
            barAlignment: 0,
            barWidthFactor: 0.6,
            drawStyle: 'line',
            fillOpacity: 8,
            gradientMode: 'none',
            hideFrom: {
              legend: false,
              tooltip: false,
              viz: false,
            },
            insertNulls: false,
            lineInterpolation: 'linear',
            lineWidth: 2,
            pointSize: 5,
            scaleDistribution: {
              type: 'linear',
            },
            showPoints: 'never',
            showValues: false,
            spanNulls: false,
            stacking: {
              group: 'A',
              mode: 'none',
            },
            thresholdsStyle: {
              mode: 'off',
            },
          },
          thresholds: {
            mode: 'absolute',
            steps: [
              {
                color: 'green',
                value: 0,
              },
              {
                color: 'red',
                value: 80,
              },
            ],
          },
        },
        overrides: [],
      },
      options: {
        annotations: {
          clustering: -1,
          multiLane: false,
        },
        legend: {
          calcs: [],
          displayMode: 'list',
          placement: 'bottom',
          showLegend: true,
        },
        tooltip: {
          hideZeros: false,
          mode: 'single',
          sort: 'none',
        },
      },
    },
    version: '13.0.2',
  },
}
