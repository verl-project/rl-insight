// Thin stable entrypoint: composes every dashboard registered in the
// production composition registry. New dashboards only need a registry entry;
// this file should not need to change.
local composer = import 'framework/composer.libsonnet';
local registry = import 'dashboard_compositions.libsonnet';

{
  [name]: composer.compose(
    registry.compositions[name].modules,
    registry.compositions[name].dashboard
  )
  for name in std.objectFields(registry.compositions)
}
