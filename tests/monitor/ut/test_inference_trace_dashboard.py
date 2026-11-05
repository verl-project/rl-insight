# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the inference-trace row of the verl vLLM dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "rl_insight"
    / "config"
    / "services"
    / "grafana"
    / "dashboards"
    / "verl"
    / "verl_tainer_v1_with_vllm_engine.json"
)
_TRACE_ROW_TITLE = "vLLM request traces"
_TRACE_SPAN_NAME = "llm_request"
_ROW_VARIABLE_QUERY = (
    '{ name = "llm_request" && resource.project =~ "$project" '
    '&& resource.experiment_name =~ "$experiment_name" }'
)

_NATIVE_LATENCY_ATTRIBUTES = (
    "gen_ai.latency.e2e",
    "gen_ai.latency.time_to_first_token",
    "gen_ai.latency.time_in_queue",
    "gen_ai.latency.time_in_scheduler",
    "gen_ai.latency.time_in_model_prefill",
    "gen_ai.latency.time_in_model_forward",
    "gen_ai.latency.time_in_model_execute",
    "gen_ai.latency.time_in_model_decode",
    "gen_ai.latency.time_in_model_inference",
)
_NATIVE_COUNT_ATTRIBUTES = (
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.request.max_tokens",
)
_NATIVE_IDENTIFIER_ATTRIBUTES = (
    "gen_ai.response.model",
    "gen_ai.response.finish_reasons",
)
_NATIVE_ATTRIBUTES = (
    _NATIVE_LATENCY_ATTRIBUTES
    + _NATIVE_COUNT_ATTRIBUTES
    + _NATIVE_IDENTIFIER_ATTRIBUTES
)
# Request id is no longer a table column, but it must still stay out of
# Prometheus queries along with every other native request attribute.
_PROMETHEUS_FORBIDDEN_ATTRIBUTES = _NATIVE_ATTRIBUTES + ("gen_ai.request.id",)


def _load_spec() -> dict[str, Any]:
    with open(_DASHBOARD, encoding="utf-8") as dashboard_file:
        return json.load(dashboard_file)["spec"]


def _find_row(layout: Any, title: str) -> dict[str, Any] | None:
    if not isinstance(layout, dict):
        return None
    kind = layout.get("kind")
    row_spec = layout.get("spec") or {}
    if kind == "RowsLayout":
        for row in row_spec.get("rows") or []:
            spec = row.get("spec") or {}
            if spec.get("title") == title:
                return spec
            found = _find_row(spec.get("layout"), title)
            if found is not None:
                return found
    elif kind == "TabsLayout":
        for tab in row_spec.get("tabs") or []:
            found = _find_row((tab.get("spec") or {}).get("layout"), title)
            if found is not None:
                return found
    return None


def _element_names(layout: Any) -> list[str]:
    if not isinstance(layout, dict):
        return []
    kind = layout.get("kind")
    spec = layout.get("spec") or {}
    names: list[str] = []
    if kind in ("GridLayout", "AutoGridLayout"):
        for item in spec.get("items") or []:
            element = (item.get("spec") or {}).get("element") or {}
            if element.get("kind") == "ElementReference":
                names.append(str(element.get("name")))
    elif kind == "RowsLayout":
        for row in spec.get("rows") or []:
            names.extend(_element_names((row.get("spec") or {}).get("layout")))
    elif kind == "TabsLayout":
        for tab in spec.get("tabs") or []:
            names.extend(_element_names((tab.get("spec") or {}).get("layout")))
    return names


def _panel_specs(
    spec: dict[str, Any], element_names: list[str]
) -> list[dict[str, Any]]:
    return [spec["elements"][name]["spec"] for name in element_names]


def _query_specs(panel: dict[str, Any]) -> list[dict[str, Any]]:
    group = panel.get("data") or {}
    queries = (group.get("spec") or {}).get("queries") or []
    return [query["spec"]["query"] for query in queries]


def _trace_panels(spec: dict[str, Any]) -> list[dict[str, Any]]:
    row = _find_row(spec["layout"], _TRACE_ROW_TITLE)
    assert row is not None, f"missing dashboard row: {_TRACE_ROW_TITLE}"
    panels = _panel_specs(spec, _element_names(row.get("layout")))
    return [
        panel
        for panel in panels
        if any(query.get("group") == "tempo" for query in _query_specs(panel))
    ]


def _trace_tables(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        panel for panel in _trace_panels(spec) if panel["vizConfig"]["group"] == "table"
    ]


def test_trace_row_should_query_tempo_spans() -> None:
    panels = _trace_panels(_load_spec())

    assert panels
    queries = [query for panel in panels for query in _query_specs(panel)]
    for query in queries:
        assert query["group"] == "tempo"
        assert query["spec"]["queryType"] == "traceql"
        assert query["spec"]["tableType"] == "spans"


def test_trace_query_should_filter_by_resource_attributes() -> None:
    for panel in _trace_panels(_load_spec()):
        query = _query_specs(panel)[0]["spec"]["query"]
        assert f'name = "{_TRACE_SPAN_NAME}"' in query
        for attribute in ("project", "experiment_name"):
            assert f'resource.{attribute} =~ "$' in query
        assert 'resource.replica =~ "^($trace_replica)$"' in query


def test_trace_query_should_filter_by_correlation_attributes() -> None:
    for panel in _trace_tables(_load_spec()):
        query = _query_specs(panel)[0]["spec"]["query"]

        # -1 keeps the step filter optional; integer attributes do not support
        # regex, so the query uses a static "$step = -1" comparison.
        assert "(span.global_steps = $step || $step = -1)" in query
        # Anchor the regex so selecting "100" does not also match "1006".
        # The All option expands to ".*"; the extra static comparison keeps
        # spans without the sample attribute visible in that case.
        assert 'span.sample =~ "^($sample)$"' in query
        assert '"$sample" = ".*"' in query
        # The row-scoped Replica filter must not reuse the Prometheus vllm
        # replica variable; it is sourced from Tempo tags instead.
        assert 'resource.replica =~ "^($trace_replica)$"' in query
        assert '"$trace_replica" = ".*"' in query


def test_trace_row_should_expose_step_sample_and_replica_filters() -> None:
    row = _find_row(_load_spec()["layout"], _TRACE_ROW_TITLE)
    assert row is not None, f"missing dashboard row: {_TRACE_ROW_TITLE}"
    variables = {
        variable["spec"]["name"]: variable["spec"] for variable in row["variables"]
    }

    # Trajectory and Session have no dropdowns: the raw trace covers them. The
    # result cap is fixed in the query, so there is no limit variable.
    assert list(variables) == ["step", "trace_replica", "sample"]
    assert variables["step"]["label"] == "Step"
    assert variables["step"]["definition"] == _ROW_VARIABLE_QUERY
    assert variables["step"]["query"]["spec"]["query"] == _ROW_VARIABLE_QUERY
    assert variables["step"]["multi"] is False
    # Grafana selects All whenever includeAll is on and no value is saved, so
    # step trades the built-in All option for a trailing static All option.
    assert variables["step"]["includeAll"] is False
    assert variables["step"]["staticOptions"] == [{"text": "All", "value": "-1"}]
    assert variables["step"]["staticOptionsOrder"] == "after"
    assert variables["step"]["allValue"] == "-1"
    # No saved value: the newest step (first option after numericalDesc) is
    # selected, so expanding the row does not load every step by default.
    assert variables["step"]["current"] == {"text": "", "value": ""}
    assert variables["step"]["sort"] == "numericalDesc"
    assert variables["sample"]["label"] == "Sample"
    assert variables["sample"]["definition"] == _ROW_VARIABLE_QUERY
    assert variables["sample"]["query"]["spec"]["query"] == _ROW_VARIABLE_QUERY
    assert variables["sample"]["multi"] is True
    assert variables["sample"]["includeAll"] is True
    assert variables["sample"]["allValue"] == ".*"
    assert variables["sample"]["sort"] == "numericalAsc"
    assert variables["trace_replica"]["label"] == "Replica"
    assert variables["trace_replica"]["definition"] == _ROW_VARIABLE_QUERY
    assert variables["trace_replica"]["query"]["spec"]["query"] == _ROW_VARIABLE_QUERY
    assert variables["trace_replica"]["multi"] is True
    assert variables["trace_replica"]["includeAll"] is True
    assert variables["trace_replica"]["allValue"] == ".*"
    assert variables["trace_replica"]["sort"] == "numericalAsc"


def test_trace_query_should_select_native_vllm_attributes() -> None:
    for panel in _trace_tables(_load_spec()):
        query = _query_specs(panel)[0]["spec"]["query"]
        for attribute in _NATIVE_ATTRIBUTES:
            assert f"span.{attribute}" in query


def test_trace_query_should_select_optional_correlation_attributes() -> None:
    panels = _trace_panels(_load_spec())
    queries = [_query_specs(panel)[0]["spec"]["query"] for panel in panels]

    assert any(
        all(
            f"span.{attribute}" in query
            for attribute in ("global_steps", "turn", "sample")
        )
        for query in queries
    )
    # Trajectory, Session and the engine request id are dropped from the table;
    # they stay discoverable through the trace itself.
    for query in queries:
        for attribute in ("traj", "session", "gen_ai.request.id"):
            assert f"span.{attribute}" not in query


def test_state_timeline_should_only_query_state_interval_spans() -> None:
    # llm_request spans also carry state_lane_id; filtering only by that
    # attribute pulls request spans into the trajectory timeline.
    spec = _load_spec()
    panel = spec["elements"]["panel-40"]["spec"]
    query = _query_specs(panel)[0]["spec"]["query"]

    assert "monitor.trace_segment" in query
    assert "state_interval" in query
    # Tempo only returns attributes referenced by the query, so the lane must
    # stay in the select list for the partition transformation.
    assert "span.state_lane_id" in query

    organize = next(
        transformation["spec"]["options"]
        for transformation in panel["data"]["spec"]["transformations"]
        if transformation["group"] == "organize"
    )
    # The filter attribute must not become a second state value per lane.
    assert organize["excludeByName"]["monitor.trace_segment"] is True


def test_trace_table_should_use_a_fixed_result_limit() -> None:
    # Tempo requires a numeric limit and does not interpolate dashboard
    # variables into that field, so the row has one table with a fixed result
    # cap and no limit selector.
    spec = _load_spec()
    panels = _trace_tables(spec)
    assert len(panels) == 1
    panel = panels[0]

    limits = {query["spec"].get("limit") for query in _query_specs(panel)}
    assert limits == {20000}
    assert "trace_limit" not in json.dumps(spec)
    assert panel["vizConfig"]["spec"]["options"]["enablePagination"] is True


def test_trace_row_should_be_collapsed_by_default() -> None:
    row = _find_row(_load_spec()["layout"], _TRACE_ROW_TITLE)
    assert row is not None, f"missing dashboard row: {_TRACE_ROW_TITLE}"

    assert row["collapse"] is True


def test_trace_table_should_freeze_start_and_trace_columns() -> None:
    for panel in _trace_tables(_load_spec()):
        options = panel["vizConfig"]["spec"]["options"]

        assert options["frozenColumns"] == {"left": 2}


def test_trace_tables_should_use_gauge_cells() -> None:
    for panel in _trace_tables(_load_spec()):
        e2e_override = next(
            override
            for override in panel["vizConfig"]["spec"]["fieldConfig"]["overrides"]
            if override["matcher"]["options"] == "E2E (s)"
        )
        cell_options = next(
            prop["value"]
            for prop in e2e_override["properties"]
            if prop["id"] == "custom.cellOptions"
        )
        assert cell_options["type"] == "gauge"


def test_replica_variable_should_match_any_vllm_metric() -> None:
    spec = _load_spec()
    replica = next(
        variable
        for variable in spec["variables"]
        if variable["spec"]["name"] == "replica"
    )
    definition = replica["spec"]["definition"]

    assert "vllm:.*" in definition
    assert "request_prompt_tokens_sum" not in definition


def test_trace_table_should_keep_the_tempo_span_link() -> None:
    for panel in _trace_tables(_load_spec()):
        transformations = panel["data"]["spec"]["transformations"]
        organize = next(
            transformation
            for transformation in transformations
            if transformation["group"] == "organize"
        )
        excluded = organize["spec"]["options"]["excludeByName"]

        # The Tempo datasource attaches the trace link to the spanID field and
        # resolves the trace id through ${__data.fields.traceIdHidden}; removing
        # traceIdHidden would leave the link with an empty query. Tempo already
        # hides this field from the visualization, so it stays in the frame.
        assert "spanID" not in excluded
        assert "Span ID" not in excluded
        assert excluded.get("traceIdHidden") is not True


def test_trace_tables_should_order_columns_by_tempo_display_names() -> None:
    # Grafana's organize transformation matches indexByName/renameByName against
    # field display names, and Tempo renames time/spanID to Start time/Span ID.
    for panel in _trace_tables(_load_spec()):
        organize = next(
            transformation["spec"]["options"]
            for transformation in panel["data"]["spec"]["transformations"]
            if transformation["group"] == "organize"
        )

        assert organize["indexByName"]["Start time"] == 0
        assert organize["indexByName"]["Span ID"] == 1
        assert organize["renameByName"]["Start time"] == "Start"
        assert organize["renameByName"]["Span ID"] == "Trace"
        assert organize["excludeByName"]["project"] is True
        assert organize["excludeByName"]["experiment_name"] is True


def test_trace_table_should_expose_replica_and_sample_columns() -> None:
    for panel in _trace_tables(_load_spec()):
        query = _query_specs(panel)[0]["spec"]["query"]
        select_clause = query.split("| select(", 1)[1]
        assert "resource.replica" in select_clause

        organize = next(
            transformation["spec"]["options"]
            for transformation in panel["data"]["spec"]["transformations"]
            if transformation["group"] == "organize"
        )
        assert organize["renameByName"].get("replica") == "Replica"
        assert organize["renameByName"].get("sample") == "Sample"


def test_trace_table_should_derive_decode_throughput() -> None:
    panel = _trace_tables(_load_spec())[0]
    transformations = panel["data"]["spec"]["transformations"]
    calculate = next(
        transformation
        for transformation in transformations
        if transformation["group"] == "calculateField"
    )
    options = calculate["spec"]["options"]

    assert options["alias"] == "Decode tok/s"
    assert (
        options["binary"]["left"]["matcher"]["options"]
        == "gen_ai.usage.completion_tokens"
    )
    assert options["binary"]["operator"] == "/"
    assert (
        options["binary"]["right"]["matcher"]["options"]
        == "gen_ai.latency.time_in_model_decode"
    )

    organize = next(
        transformation["spec"]["options"]
        for transformation in transformations
        if transformation["group"] == "organize"
    )
    # Start/Trace sit first, then identity columns, then the timing columns.
    assert organize["indexByName"]["Decode tok/s"] == 11
    assert organize["indexByName"]["replica"] == 2
    assert organize["indexByName"]["global_steps"] == 3
    assert organize["indexByName"]["turn"] == 4
    assert organize["indexByName"]["sample"] == 5

    unit = next(
        prop["value"]
        for override in panel["vizConfig"]["spec"]["fieldConfig"]["overrides"]
        if override["matcher"]["options"] == "Decode tok/s"
        for prop in override["properties"]
        if prop["id"] == "unit"
    )
    assert unit == "cps"


def test_native_timing_attributes_should_not_leak_into_prometheus() -> None:
    spec = _load_spec()

    for element in spec["elements"].values():
        for query in _query_specs(element["spec"]):
            if query.get("group") != "prometheus":
                continue
            expression = query["spec"].get("expr", "")
            for attribute in _PROMETHEUS_FORBIDDEN_ATTRIBUTES:
                assert attribute not in expression, (
                    element["spec"].get("title"),
                    attribute,
                )
