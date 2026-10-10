# Grafana dashboard framework

RL-Insight builds its Grafana dashboards from **Jsonnet** sources shipped inside
the `rl_insight` package and materializes them when the service starts. Grafana
keeps reading plain JSON, and nobody runs a generator or commits generated
files.

This change is the mechanism layer: the composition registry ships **empty**, so
the runtime stages the bundled static dashboards and registers no Jsonnet one.
The production compositions come with the change that depends on this one.

## How it works

![Jsonnet dashboard file architecture](diagrams/jsonnet-files.svg)
![RL-Insight dashboard startup flow](diagrams/server-start.svg)

`rl-insight server start` rebuilds `<runtime_dir>/dashboards` from scratch: the
bundled static dashboards are staged byte for byte, then
`rl_insight.grafana.renderer` evaluates the entrypoint in process through the
`rjsonnet` binding and writes one `<dashboard-name>.json` per registered
composition. Rendering is additive — an existing file is never overwritten and a
collision fails startup. The runtime never writes to the package sources, the
config or the repository, Grafana provisioning keeps reading plain JSON from the
runtime directory, and no Jsonnet CLI, Go toolchain or compiler is required.

### Source layout

| Path | Purpose |
| --- | --- |
| `jsonnet/dashboards.jsonnet` | Sole entrypoint; composes every registered composition. Normally unchanged. |
| `jsonnet/dashboard_compositions.libsonnet` | Composition registry; empty in this change. |
| `jsonnet/dashboards/*.libsonnet` | Reusable content modules (`panels`, `rows`, `rowItems`, `variables`, `tags`). |
| `jsonnet/framework/composer.libsonnet` | Ordered merge, conflict detection, late layout reference resolution. |
| `jsonnet/framework/viz.libsonnet` | Shared visualization defaults and RFC 7396 viz patches. |
| `rl_insight/grafana/renderer.py` | `render_dashboards`, `materialize_dashboards`, `stale_dashboards`. |
| `tools/grafana/framework/generate.py` | Optional CLI wrapper over the same renderer (debug/CI only). |

### Configuration precedence

| `config.yaml` key | Effect |
| --- | --- |
| `grafana.dashboard_config` | Non-empty: render that Jsonnet config instead of the bundled entrypoint. A missing file or Jsonnet error stops startup before Grafana runs. |
| `grafana.dashboards_dir` | Legacy static directory. When it is not the bundled default it is copied as-is and no Jsonnet runs. |
| `grafana.extra_dashboard_dir` | Merged last, with recursive copy and filename-collision failure. |

## Extending the framework (example only)

```jsonnet
// jsonnet/dashboards/foo.libsonnet: plain data, no behaviour
{
  panels: [{
    key: 'foo.panel', outputKey: 'panel-foo', id: 900, title: 'Foo metric',
    queries: [{ expr: 'foo_metric' }],
  }],
  rows: { 'foo row': { kind: 'RowsLayoutRow', spec: { title: 'foo row',
    collapse: false, layout: { kind: 'GridLayout', spec: { items: [{
      kind: 'GridLayoutItem', spec: { x: 0, y: 0, width: 24, height: 8,
        element: { kind: 'ElementReference', name: 'foo.panel' } } }] } } } } },
  variables: { foo_instance: { kind: 'ConstantVariable', spec: {
    name: 'foo_instance', label: 'Instance', value: 'i0', hide: 'dontHide' } } },
}
```

```jsonnet
// jsonnet/dashboard_compositions.libsonnet: specialize the empty registry
local foo = import 'dashboards/foo.libsonnet';

{ compositions: { foo_dashboard: {
  modules: [foo],
  dashboard: {
    metadata: { name: 'foo-dashboard-id', labels: {}, annotations: {} },
    title: 'foo_dashboard', tags: ['RL-Insight', 'foo'], spec: {},
    variableOrder: ['foo_instance'], rowOrder: ['foo row'],
  },
} } }
```

`rowItems` lets a module append panels to a row another module owns; see
`composer.libsonnet` for the full module schema and composition rules.

## Optional CLI

`tools/grafana/framework/generate.py` renders through the same core for local
debugging and CI. The server never calls it:

```bash
python tools/grafana/framework/generate.py --config <config.jsonnet> --out-dir <dir>
python tools/grafana/framework/generate.py --config <config.jsonnet> --check --expected-dir <dir>
```

Exit codes: `0` success, `1` stale files found by `--check`, `2` rendering
failed.

## Tests

`tests/monitor/ut/` covers the composer, renderer, CLI and startup preparation.
