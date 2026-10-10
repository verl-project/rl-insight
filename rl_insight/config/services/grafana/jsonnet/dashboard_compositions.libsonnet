// The composition registry consumed by `dashboards.jsonnet`.
//
// The generic mechanism ships this file empty: nothing here knows about any
// concrete dashboard, so this change can be merged and started on its own while
// the runtime keeps staging the bundled static JSON dashboards.
//
// A dependent change that adds real content specializes this registry:
//
//   local myDashboard = import 'compositions/my_dashboard.libsonnet';
//   { compositions: { my_dashboard: myDashboard } }
//
// Each entry provides `modules` (the ordered modules to compose) and `dashboard`
// (metadata, title, tags, chrome `spec`, `variableOrder`, `rowOrder`).
{
  compositions: {},
}
