// The production composition registry consumed by `dashboards.jsonnet`.
//
// Each entry is one dashboard: the ordered `modules` to compose plus the
// dashboard-level config (metadata, title, tags, chrome `spec`, `variableOrder`,
// `rowOrder`). The two VERL compositions live in `compositions/`, so adding a
// dashboard means adding one composition module and one entry here — the
// entrypoint itself never changes.
local verlVllm = import 'compositions/verl_vllm.libsonnet';
local verlSglang = import 'compositions/verl_sglang.libsonnet';

{
  compositions: {
    verl_tainer_v1_with_vllm_engine_jsonnet: verlVllm,
    verl_tainer_v1_with_sglang_engine_jsonnet: verlSglang,
  },
}
