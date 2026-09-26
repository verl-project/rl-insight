// Lightweight structural verification for the semantic production modules
// and the composition registry.
local controller = import 'controller.libsonnet';
local npu = import 'npu.libsonnet';
local registry = import '../dashboard_compositions.libsonnet';
local sglang = import 'sglang.libsonnet';
local storage = import 'storage.libsonnet';
local trainer = import 'trainer.libsonnet';
local trajectory = import 'trajectory.libsonnet';
local vllm = import 'vllm.libsonnet';

local sharedModules = [trainer, controller, storage, trajectory];
local objectField(modules, field) =
  std.foldl(
    function(acc, module) acc + (if std.objectHas(module, field) then module[field] else {}),
    modules,
    {}
  );
local panels(modules) = std.flattenArrays([module.panels for module in modules]);
local sortedPanels(items) = std.sort(items, function(item) item.key);
local fingerprint(value) = std.md5(std.manifestJson(value));
local unique(items) = std.length(std.set(items)) == std.length(items);

local sharedPanels = panels(sharedModules);
local sharedRows = objectField(sharedModules, 'rows');
local sharedVariables = objectField(sharedModules, 'variables');
local vllmPanels = vllm.panels + npu.panels;
local vllmRows = vllm.rows + npu.rows;
local vllmVariables = vllm.variables + npu.variables;

assert std.length(sharedPanels) == 116 : 'shared panel count changed';
assert unique([panel.key for panel in sharedPanels]) : 'duplicate shared panel key';
assert unique([panel.outputKey for panel in sharedPanels]) : 'duplicate shared outputKey';
assert unique([panel.id for panel in sharedPanels]) : 'duplicate shared panel id';
assert fingerprint(sortedPanels(sharedPanels)) == '36bd2faec9baf08551d008c1e7e0bfeb'
       : 'shared panel content changed';
assert fingerprint(sharedRows) == '86143182d57394fa90e2477a3059b8b8'
       : 'shared row layout changed';
assert fingerprint(sharedVariables) == '9e931b598a5dbea59946ec2872a6e345'
       : 'shared variables changed';

assert std.length(vllm.panels) == 22 : 'vLLM non-NPU panel count changed';
assert std.length(npu.panels) == 8 : 'NPU panel count changed';
assert std.length(vllmPanels) == 30 : 'combined vLLM panel count changed';
assert unique([panel.key for panel in vllmPanels]) : 'duplicate vLLM/NPU panel key';
assert unique([panel.outputKey for panel in vllmPanels]) : 'duplicate vLLM/NPU outputKey';
assert unique([panel.id for panel in vllmPanels]) : 'duplicate vLLM/NPU panel id';
assert fingerprint(sortedPanels(vllmPanels)) == 'c6ff65eec14b579a8c4b725a60211b20'
       : 'vLLM/NPU panel content changed';
assert fingerprint(vllmRows) == '880ddbbd20f02e73e5dad2680a9df225'
       : 'vLLM row layout changed';
assert fingerprint(vllmVariables) == '24ddbd0b82b40730ba5dbcbf13ea58e5'
       : 'vLLM/NPU variables changed';

assert std.length(sglang.panels) == 8 : 'SGLang panel count changed';
assert fingerprint(sortedPanels(sglang.panels)) == '6648f8760711149085da7f370872aa88'
       : 'SGLang panel content changed';
assert fingerprint(sglang.rows) == 'aab69f1cfd88bbcc94190075c657f33d'
       : 'SGLang row layout changed';
assert fingerprint(sglang.variables) == '70d60ff7ccb33170c0b53f19140d5f21'
       : 'SGLang variables changed';

// The registry is the only dashboard-config entrypoint; the migrated config
// content must be the same object that used to live in
// dashboard_configs.libsonnet (same {vllm, sglang} shape and fingerprint).
local compositions = registry.compositions;
local migratedConfigs = {
  vllm: compositions.verl_tainer_v1_with_vllm_engine.dashboard,
  sglang: compositions.verl_tainer_v1_with_sglang_engine.dashboard,
};
assert fingerprint(migratedConfigs) == 'aaca26665ba894b549ca5b7b238624d5'
       : 'dashboard metadata, chrome, or ordering changed';

// Composition-level checks run over the ACTUAL aggregate module objects held
// by each registry entry (not over module-name strings): the aggregate must
// match the reassembled module lists and stay conflict-free.
local vllmAggregate = compositions.verl_tainer_v1_with_vllm_engine.modules;
local sglangAggregate = compositions.verl_tainer_v1_with_sglang_engine.modules;
local expectedVllmAggregate = sharedModules + [vllm, npu];
local expectedSglangAggregate = sharedModules + [sglang];

assert fingerprint(vllmAggregate) == fingerprint(expectedVllmAggregate)
       : 'vLLM registry entry does not aggregate the real modules';
assert fingerprint(sglangAggregate) == fingerprint(expectedSglangAggregate)
       : 'SGLang registry entry does not aggregate the real modules';

local vllmAggregatePanels = panels(vllmAggregate);
local sglangAggregatePanels = panels(sglangAggregate);

assert std.length(vllmAggregatePanels) == 146 : 'vLLM composition panel count changed';
assert unique([panel.key for panel in vllmAggregatePanels]) : 'duplicate panel key in vLLM composition';
assert unique([panel.outputKey for panel in vllmAggregatePanels]) : 'duplicate outputKey in vLLM composition';
assert unique([panel.id for panel in vllmAggregatePanels]) : 'duplicate panel id in vLLM composition';
assert fingerprint(sortedPanels(vllmAggregatePanels)) == 'f76cba9a1d086235dc6549fbe2c13bbe'
       : 'vLLM composition panel content changed';
assert fingerprint(objectField(vllmAggregate, 'rows')) == 'fc1b9ef07e0feaf9e4fa9cfcfedd138e'
       : 'vLLM composition row layout changed';
assert fingerprint(objectField(vllmAggregate, 'variables')) == '0093676e244f8223a423c21181e23f27'
       : 'vLLM composition variables changed';

assert std.length(sglangAggregatePanels) == 124 : 'SGLang composition panel count changed';
assert unique([panel.key for panel in sglangAggregatePanels]) : 'duplicate panel key in SGLang composition';
assert unique([panel.outputKey for panel in sglangAggregatePanels]) : 'duplicate outputKey in SGLang composition';
assert unique([panel.id for panel in sglangAggregatePanels]) : 'duplicate panel id in SGLang composition';
assert fingerprint(sortedPanels(sglangAggregatePanels)) == '85a89203eac31dce63d38c1e1d98cae1'
       : 'SGLang composition panel content changed';
assert fingerprint(objectField(sglangAggregate, 'rows')) == 'a0e6aa94b98f76fafd8e4188e761abd0'
       : 'SGLang composition row layout changed';
assert fingerprint(objectField(sglangAggregate, 'variables')) == 'df4c2344b6be8923078304377d0e0cad'
       : 'SGLang composition variables changed';

{
  modules: {
    trainer: { panels: std.length(trainer.panels), rows: std.length(std.objectFields(trainer.rows)), variables: std.length(std.objectFields(trainer.variables)) },
    controller: { panels: std.length(controller.panels), rows: std.length(std.objectFields(controller.rows)), variables: std.length(std.objectFields(controller.variables)) },
    storage: { panels: std.length(storage.panels), rows: std.length(std.objectFields(storage.rows)), variables: std.length(std.objectFields(storage.variables)) },
    trajectory: { panels: std.length(trajectory.panels), rows: std.length(std.objectFields(trajectory.rows)), variables: std.length(std.objectFields(trajectory.variables)) },
    vllm: { panels: std.length(vllm.panels), rows: std.length(std.objectFields(vllm.rows)), variables: std.length(std.objectFields(vllm.variables)) },
    sglang: { panels: std.length(sglang.panels), rows: std.length(std.objectFields(sglang.rows)), variables: std.length(std.objectFields(sglang.variables)) },
    npu: { panels: std.length(npu.panels), rows: std.length(std.objectFields(npu.rows)), variables: std.length(std.objectFields(npu.variables)) },
  },
  totals: {
    sharedPanels: std.length(sharedPanels),
    vllmPanels: std.length(vllmPanels),
    sglangPanels: std.length(sglang.panels),
    vllmCompositionPanels: std.length(vllmAggregatePanels),
    sglangCompositionPanels: std.length(sglangAggregatePanels),
  },
}
