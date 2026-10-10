// Generic Grafana dashboard composer.
//
// Takes an ordered list of sub-dashboard modules (A, B, C, D, ...) plus a
// dashboard-level composition record, and renders one Grafana Dashboard v2
// resource. The composer knows nothing about any concrete dashboard: modules
// are plain data, and every merge decision follows the explicit rules
// documented in README.md ("Composition rules").
//
// Module schema (all fields plain data, no behavioural code):
//   panels:    [{ key, outputKey, id, title, queries, description?, links?,
//                 vizBase?, vizPatch?, transformations?, queryOptions? }]
//   rows?:     { <name>: <RowsLayoutRow spec> }   (panels referenced by `key`)
//   rowItems?: { <existing row name>: [GridLayoutItem, ...] }
//              additive extension of an existing row, no rows created
//   variables?: { <name>: <variable spec> }
//   tags?:     [tag, ...]
//
// Dashboard schema (owned by the composition config, not by modules):
//   metadata, title, tags, spec (dashboard chrome), variableOrder, rowOrder
local viz = import 'viz.libsonnet';

local has(object, field) = std.objectHas(object, field);

// Guard helper: returns `value` when `condition` holds, otherwise fails
// evaluation with `message`.  Threading the value through the guard keeps the
// check eager despite Jsonnet's lazy evaluation.
local require(condition, message, value) =
  if condition then value else error message;

// Apply an RFC 7396-style merge patch.  Modules only describe the fields that
// differ from the shared visualization defaults in viz.libsonnet.
local mergePatch(base, patch) =
  if std.type(base) != 'object' || std.type(patch) != 'object' then patch
  else
    {
      [field]: base[field]
      for field in std.objectFields(base)
      if !has(patch, field)
    } + {
      [field]:
        if has(base, field) then mergePatch(base[field], patch[field])
        else patch[field]
      for field in std.objectFields(patch)
      if patch[field] != null
    };

local prometheusQuery(query) = {
  kind: 'DataQuery',
  group: 'prometheus',
  version: 'v0',
  spec: {
    editorMode: if has(query, 'editorMode') then query.editorMode else 'builder',
    expr: query.expr,
    legendFormat: if has(query, 'legend') then query.legend else '{{__name__}}',
    range: true,
  } + (if has(query, 'extra') then query.extra else {}),
} + (if has(query, 'labels') then { labels: query.labels } else {});

local panelQuery(query, index) = {
  kind: 'PanelQuery',
  spec: {
    query: if has(query, 'raw') then query.raw else prometheusQuery(query),
    refId: if has(query, 'refId') then query.refId else std.char(std.codepoint('A') + index),
    hidden: if has(query, 'hidden') then query.hidden else false,
  },
};

local panel(record) = {
  kind: 'Panel',
  spec: {
    id: record.id,
    title: record.title,
    description: if has(record, 'description') then record.description else '',
    links: if has(record, 'links') then record.links else [],
    data: {
      kind: 'QueryGroup',
      spec: {
        queries: std.mapWithIndex(function(index, query) panelQuery(query, index), record.queries),
        transformations: if has(record, 'transformations') then record.transformations else [],
        queryOptions: if has(record, 'queryOptions') then record.queryOptions else {},
      },
    },
    vizConfig: mergePatch(
      viz[if has(record, 'vizBase') then record.vizBase else 'timeseries'],
      if has(record, 'vizPatch') then record.vizPatch else {}
    ),
  },
};

// Rewrite `ElementReference` nodes inside layout specs: layouts address panels
// by module-unique `key`, while dashboard elements are keyed by `outputKey`.
// Resolving late keeps module files free of naming collisions.
local replaceReferences(value, outputKeys) =
  if std.type(value) == 'array' then
    [replaceReferences(item, outputKeys) for item in value]
  else if std.type(value) == 'object' then
    if has(value, 'kind') && value.kind == 'ElementReference' then
      value { name: outputKeys[value.name] }
    else
      { [field]: replaceReferences(value[field], outputKeys) for field in std.objectFields(value) }
  else value;

local duplicates(items) =
  local counts = std.foldl(
    function(acc, item) acc { [item + '']: std.get(acc, item + '', 0) + 1 },
    items,
    {}
  );
  [field for field in std.objectFields(counts) if counts[field] > 1];

// Returns `value` unless `items` contains duplicates, in which case evaluation
// fails and names the offending entries.
local requireUnique(items, what, value) =
  local dups = duplicates(items);
  require(
    std.length(dups) == 0,
    'duplicate ' + what + '(s): ' + std.join(', ', dups),
    value
  );

local mergedField(modules, field) =
  std.foldl(
    function(acc, module)
      acc + (if has(module, field) then module[field] else {}),
    modules,
    {}
  );

// Compose `modules` into one dashboard following the rules in README.md.
// `dashboard` carries the dashboard-level decisions: metadata, title, tags,
// chrome `spec`, and the variable/row ordering.
local compose(modules, dashboard) =
  local checkedModules =
    require(
      std.type(modules) == 'array' && std.length(modules) > 0,
      'compose() needs a non-empty module list',
      modules
    );
  local recordsRaw = std.flattenArrays([module.panels for module in checkedModules]);
  local recordsKeyed = requireUnique(
    [record.key for record in recordsRaw],
    'panel key',
    recordsRaw
  );
  local recordsNamed = requireUnique(
    [record.outputKey for record in recordsKeyed],
    'panel outputKey',
    recordsKeyed
  );
  local records = requireUnique(
    [record.id + '' for record in recordsNamed],
    'panel id',
    recordsNamed
  );
  // Object merge (`+`) collapses same-named fields silently, so duplicates
  // must be detected on the per-module field lists, not on the merged object.
  local rowsMerged = mergedField(checkedModules, 'rows');
  local rows = requireUnique(
    std.flattenArrays([
      if has(module, 'rows') then std.objectFields(module.rows) else []
      for module in checkedModules
    ]),
    'row name',
    rowsMerged
  );
  // `rowItems` is additive: modules append GridLayoutItems to rows owned by
  // other (or their own) modules without creating rows. Extensions are folded
  // in composition order, so items appended by earlier modules come first.
  // Appended `ElementReference` names are resolved together with the row's
  // own references at render time (late key -> outputKey resolution).
  local rowItemExtensions = [
    if has(module, 'rowItems') then module.rowItems else {}
    for module in checkedModules
  ];
  // Every extension target is validated eagerly: unknown names and rows whose
  // structure cannot take items fail evaluation instead of being ignored.
  // (Object-merge fields are lazy, so validation must flow through a value
  // the dashboard actually uses.)
  local extensionTargets =
    std.flattenArrays([std.objectFields(ext) for ext in rowItemExtensions]);
  local validatedExtensions = [
    require(
      std.objectHas(rows, name),
      'unknown row extension target: ' + name,
      require(
        has(rows[name], 'kind')
          && rows[name].kind == 'RowsLayoutRow'
          && has(rows[name], 'spec')
          && has(rows[name].spec, 'layout')
          && has(rows[name].spec.layout, 'kind')
          && rows[name].spec.layout.kind == 'GridLayout'
          && has(rows[name].spec.layout, 'spec')
          && has(rows[name].spec.layout.spec, 'items'),
        'unsupported row extension target: expected a RowsLayoutRow with a '
          + 'GridLayout layout carrying spec.items',
        rows[name]
      )
    )
    for name in extensionTargets
  ];
  local extensionDigest = std.toString(validatedExtensions);
  assert std.type(extensionDigest) == 'string' : 'rowItems validation failed';
  local rowsExtended = std.foldl(
    function(acc, extensions)
      acc + {
        [name]:
          acc[name]
          {
            spec: acc[name].spec {
              layout: acc[name].spec.layout {
                spec: { items: acc[name].spec.layout.spec.items + extensions[name] },
              },
            },
          }
        for name in std.objectFields(extensions)
      },
    rowItemExtensions,
    rows
  );
  local variablesMerged = mergedField(checkedModules, 'variables');
  local variables = requireUnique(
    std.flattenArrays([
      if has(module, 'variables') then std.objectFields(module.variables) else []
      for module in checkedModules
    ]),
    'variable name',
    variablesMerged
  );
  local outputKeys = { [record.key]: record.outputKey for record in records };
  local variablesOrdered = require(
    std.length([name for name in dashboard.variableOrder if !std.objectHas(variables, name)]) == 0,
    'unknown variable(s) in variableOrder: '
      + std.join(
        ', ',
        [name for name in dashboard.variableOrder if !std.objectHas(variables, name)]
      ),
    variables
  );
  local rowsOrdered = require(
    std.length([name for name in dashboard.rowOrder if !std.objectHas(rowsExtended, name)]) == 0,
    'unknown row(s) in rowOrder: '
      + std.join(', ', [name for name in dashboard.rowOrder if !std.objectHas(rowsExtended, name)]),
    rowsExtended
  );
  local moduleTags = std.flattenArrays([
    if has(module, 'tags') then module.tags else []
    for module in checkedModules
  ]);
  local tags = std.foldl(
    function(acc, tag) if std.member(acc, tag) then acc else acc + [tag],
    dashboard.tags + moduleTags,
    []
  );
  {
    apiVersion: 'dashboard.grafana.app/v2',
    kind: 'Dashboard',
    metadata: dashboard.metadata,
    spec: dashboard.spec {
      elements: { [record.outputKey]: panel(record) for record in records },
      layout: {
        kind: 'RowsLayout',
        spec: {
          rows: [replaceReferences(rowsOrdered[name], outputKeys) for name in dashboard.rowOrder],
        },
      },
      title: dashboard.title,
      variables: [variablesOrdered[name] for name in dashboard.variableOrder],
      tags: tags,
    },
  };

{
  compose: compose,
}
