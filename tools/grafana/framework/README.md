# Grafana dashboard composition framework

A minimal, generic framework for building Grafana dashboards as Jsonnet code:
a **composer** that combines small dashboard modules into one dashboard, and a
**renderer** that evaluates the result into deterministic JSON. The framework
knows nothing about any concrete dashboard (no engine names, no metric names,
no production paths) and ships no fixtures — every behavior below is exercised
by tests that build their inputs in temporary directories.

This document is for **framework contributors**. It covers the module schema,
the composer, `rowItems`, the visualization defaults, the renderer API, the
optional CLI and the tests. Authors who only want to add or extend a dashboard
should read the dashboard-development guide instead.

## Motivation

Dashboards maintained as hand-edited JSON tend to duplicate the same
panel/row/variable definitions across dashboard variants: each variant is a
near-copy of a base dashboard, every tweak has to be repeated by hand in every
copy, variants are hard to extend without forking their rows, and the
committed JSON cannot be checked mechanically. This framework addresses all
three: modules define content once, composition configs decide which modules
make up a dashboard, additive extensions enrich existing rows without forking
them, and `--check` mode fails deterministically when the committed JSON no
longer matches the sources.

## Architecture

The runtime pieces live inside the installed `rl_insight` package, so a wheel
carries everything needed to render — no Jsonnet CLI, no Go toolchain, no
compiler, and no second copy of the framework:

```
rl_insight/grafana/renderer.py                 # runtime core: evaluate + serialize
rl_insight/config/services/grafana/jsonnet/framework/
├── composer.libsonnet   # panel/row/variable machinery + compose()
└── viz.libsonnet        # shared Grafana visualization defaults
tools/grafana/framework/
├── generate.py          # optional thin CLI over the package core
└── README.md            # this file
tests/monitor/ut/test_grafana_framework.py
```

`rl_insight.grafana.renderer` is the single implementation of evaluation and
serialization. `tools/grafana/framework/generate.py` adds argument parsing and
exit codes only — it imports the core and never re-implements the evaluator,
shells out to a Jsonnet CLI, or invokes another Python entry point.

Modules and composition configs are plain data; `composer.compose(modules,
dashboard)` renders one Grafana Dashboard v2 resource. The composer contains
no dashboard-specific knowledge: all content — panels, rows, variables, and
additive row extensions — comes from the modules, and all dashboard-level
decisions come from the composition config.

## Module interface

A module is a plain Jsonnet object — data only, no behavioral code:

| field       | required | content                                                          |
| ----------- | -------- | ---------------------------------------------------------------- |
| `panels`    | yes      | list of panel records: `key`, `outputKey`, `id`, `title`, `queries`, plus optional `description`, `links`, `vizBase`, `vizPatch`, `transformations`, `queryOptions` |
| `rows`      | no       | named `RowsLayoutRow` specs the module **owns** (creates); grid items reference panels by their `key` |
| `rowItems`  | no       | `{ <existing row name>: [GridLayoutItem, ...] }` — items **appended** to a row owned by another (or the same) module; the module does not create the row |
| `variables` | no       | named variable specs                                             |
| `tags`      | no       | tags merged into the dashboard tag list                          |

`vizPatch` is an RFC 7396 merge patch over the shared defaults in
`viz.libsonnet`, so a module only describes what differs from the defaults.

## Composition

1. **Content is concatenated in module order**; nothing is overridden
   implicitly.
2. **Conflicts are errors, not overrides**: duplicate panel `key`,
   `outputKey`, or `id`, duplicate owned row name, or duplicate variable name
   across the composed modules all abort evaluation with a message naming the
   offending entries.
3. **Dashboard-level decisions come from the composition config**: `metadata`,
   `title`, `spec` (chrome such as time range/refresh), `tags`, and the
   `variableOrder` / `rowOrder` that pick and order variables and rows from the
   merged modules.
4. **Ordering references are validated**: an entry in `variableOrder` or
   `rowOrder` that no module provides aborts evaluation.
5. **Tags** are the config's tags followed by module tags in composition
   order, de-duplicated keeping the first occurrence.
6. **References resolve late**: layouts address panels by module-unique `key`;
   the composer rewrites `ElementReference` names to `outputKey`s when
   rendering, so modules never need to know each other's naming.
7. **`rowItems` are append-only extensions** of existing rows: appending runs
   in module composition order (and, per target row, in the order the modules
   are listed); a target row that no module owns fails with
   `unknown row extension target: ...`; a target that is not a
   `RowsLayoutRow` with a `GridLayout` layout carrying `spec.items` fails with
   `unsupported row extension target: ...` — never silently ignored.

## Extending existing modules

A base module owns its rows; extension modules contribute panels into those
rows via `rowItems`. Both files stay generic — this example uses an
`overview` row and an `overview_extra` satellite module:

```jsonnet
// modules/overview.libsonnet (base: owns the row)
{
  panels: [{
    key: 'overview.requests', outputKey: 'overview-requests', id: 1,
    title: 'Request rate',
    queries: [{ expr: 'my_service_requests_total' }],
  }],
  rows: {
    overview: {
      kind: 'RowsLayoutRow',
      spec: {
        title: 'Overview', collapse: false,
        layout: {
          kind: 'GridLayout',
          spec: {
            items: [{
              kind: 'GridLayoutItem',
              spec: {
                x: 0, y: 0, width: 12, height: 8,
                element: { kind: 'ElementReference', name: 'overview.requests' },
              },
            }],
          },
        },
      },
    },
  },
}

// modules/overview_extra.libsonnet (additive extension: owns no rows)
{
  panels: [{
    key: 'overview_extra.saturation', outputKey: 'overview-extra-saturation', id: 2,
    title: 'Saturation',
    queries: [{ expr: 'my_service_saturation_ratio' }],
  }],
  rowItems: {
    overview: [{
      kind: 'GridLayoutItem',
      spec: {
        x: 12, y: 0, width: 12, height: 8,
        element: { kind: 'ElementReference', name: 'overview_extra.saturation' },
      },
    }],
  },
}

// dashboards/my-service.jsonnet
local composer = import 'rl_insight/config/services/grafana/jsonnet/framework/composer.libsonnet';
local overview = import 'modules/overview.libsonnet';
local overview_extra = import 'modules/overview_extra.libsonnet';

{
  'my-service': composer.compose([overview, overview_extra], {
    metadata: { name: 'my-service', uid: 'my-service' },
    title: 'My service',
    tags: ['example'],
    spec: { time: { from: 'now-6h', to: 'now' } },
    variableOrder: ['region'],
    rowOrder: ['overview'],
  }),
}
```

The rendered `Overview` row contains the base item first, then the extension's
item. Several extensions may target the same row — their items append in
composition order — and adding another panel to an existing dashboard is a
new extension module plus one entry in the config's module list, never a
change to the base module.

## Renderer

The core API lives in `rl_insight.grafana.renderer` (also re-exported from
`rl_insight.grafana`):

| function                                       | purpose                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `render_dashboards(config) -> dict`            | evaluate the config and return `{ "<dashboard-name>": <dashboard> }` |
| `generated_text(dashboard) -> str`             | deterministic serialization of one dashboard                       |
| `materialize_dashboards(config, output_dir)`   | render and write one `<dashboard-name>.json` per dashboard         |
| `stale_dashboards(config, expected_dir)`       | the expected files that are missing or differ byte-for-byte        |
| `FRAMEWORK_DIR`                                | package path holding `composer.libsonnet` and `viz.libsonnet`      |

```python
from rl_insight.grafana import renderer

dashboards = renderer.render_dashboards("dashboards/my-service.jsonnet")
renderer.materialize_dashboards("dashboards/my-service.jsonnet", "/tmp/out")
assert renderer.stale_dashboards("dashboards/my-service.jsonnet", "dashboards/expected") == []
```

A composition config evaluates to `{ "<dashboard-name>": <dashboard resource> }`
where each resource is the object returned by `composer.compose(...)`; the
renderer writes one JSON file per name. Output is deterministic: the composer
emits object fields in sorted order and `generated_text` uses fixed
indentation, so re-running over unchanged sources is byte-identical.

Evaluation happens **in-process** through the [`rjsonnet`][rjsonnet] binding, a
base runtime dependency of `rl-insight`. `rjsonnet` publishes CPython ABI3
wheels for Windows (x86/x64), macOS (x86_64/arm64/universal2) and the common
Linux glibc/musl architectures, so installing `rl-insight` is enough on every
platform: no `jsonnet` CLI, no `go install`, and no compiler step. Evaluation
failures raise `renderer.JsonnetRenderError`, whose message names the config
path and keeps the Jsonnet stack trace.

[rjsonnet]: https://pypi.org/project/rjsonnet/

## Optional CLI

`tools/grafana/framework/generate.py` is a convenience wrapper for shell use.
It parses arguments, calls the package core and maps the result to an exit
code; it renders exactly the bytes `render_dashboards` /
`materialize_dashboards` produce.

```bash
# render: writes <dashboard-name>.json into the output directory
python tools/grafana/framework/generate.py \
  --config dashboards/my-service.jsonnet --out-dir /tmp/out

# verify: compare against committed expected files, exit 1 on any mismatch
python tools/grafana/framework/generate.py \
  --config dashboards/my-service.jsonnet \
  --check --expected-dir dashboards/expected
```

Exit codes: `0` success, `1` `--check` found stale files, `2` the config could
not be rendered. The CLI is optional — library callers use the renderer
directly, and nothing in the runtime path depends on it.

## Developer workflow

1. Write (or reuse) modules; decide which module owns each row.
2. List the modules in a composition config and pick `variableOrder` /
   `rowOrder` there.
3. Render through the renderer (or the optional CLI) and commit the JSON next
   to the config.
4. Keep expected files in sync with sources — CI's `--check` run fails on any
   drift, and the tests reject duplicate names, unresolved ordering
   references, bad extension targets, and non-deterministic output.

## Testing

- `pytest tests/monitor/ut/test_grafana_framework.py` — 25 tests. Rendering
  goes through the installed-package core; the optional CLI is covered
  separately, only to prove it is a thin wrapper (same bytes, same exit codes).
  Coverage includes byte determinism, materialization into a fresh output
  directory, check-mode pass/stale-detection, evaluator failures naming the
  config path and the Jsonnet context, a missing config, the absence of any
  subprocess/CLI dependency, the packaged `.libsonnet` assets, every
  composition and conflict rule (duplicate panel `key`/`outputKey`/`id`,
  duplicate owned row/variable names, unresolved `variableOrder`/`rowOrder`
  entries), cross-module reference resolution, composition-order/variable-order
  semantics, and the `rowItems` extension rules (single extension, multiple
  extensions in composition order, unknown target, non-GridLayout target,
  panels + `rowItems` without owned rows). All tests build their Jsonnet inputs
  and expected outputs in `tmp_path`; no fixture JSON is committed.
- `pre-commit run --all-files` covers license headers, formatting, and compile
  checks.

## Limitations

- `rowItems` only appends `GridLayoutItem`s to a `GridLayout` row; it cannot
  remove or reorder existing items, and rows with a non-GridLayout layout
  cannot be extended.
- There is no panel patching, inheritance, or conditional composition by
  design; modules stay plain data and the composer stays small.
- `--check` compares files byte-for-byte, so expected files must be
  regenerated on any source change.

## Going forward

New dashboard variants do not require new framework code. A variant is a
composition config listing the base modules it reuses plus any additive
extension modules that append panels to existing rows via `rowItems`, with
the variable/row order picked in the config. Bringing an existing dashboard
under the framework means re-expressing its sections as modules with the same
schema — a mechanical extraction, no framework changes — after which the
dashboard's JSON is generated and verified by check mode instead of being
hand-edited. Migrating any concrete set of existing dashboards is deliberately
out of scope here and belongs to a separate, dependent change; until such a
migration lands, this framework ships nothing that affects existing
dashboards.
