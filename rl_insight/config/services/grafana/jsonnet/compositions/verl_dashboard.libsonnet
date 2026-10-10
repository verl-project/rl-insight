// Dashboard-level chrome shared by the VERL Jsonnet dashboards: annotations,
// cursor sync, editability and the default time range. Kept in one place so the
// engine-specific compositions cannot drift apart.
{
  spec: {
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
  },
}
