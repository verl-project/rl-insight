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

from __future__ import annotations

import copy
import json
import urllib.parse
from pathlib import Path

import pytest
import yaml

from experiment.degradation_perception.prometheus_workflow import (
    PrometheusWorkflowError,
    _prepare_metric_configs,
    fetch_query_range,
    load_workflow_config,
    main,
    normalize_workflow_config,
    run_prometheus_workflow,
)
from experiment.degradation_perception.simulated_prometheus import (
    ASSOCIATED_METRICS,
    METRICS,
    TARGET_METRIC,
    generate_simulation_package,
)


EXAMPLE_CONFIG = (
    Path(__file__).parents[1] / "prometheus_workflow.example.yaml"
)


def _config() -> dict:
    metrics = {
        metric: {
            "standard_query": f"test_standard_query_{index}",
            "inference_query": f"test_inference_query_{index}",
            "abnormal_type": "UP",
            "series_policy": "exactly_one",
        }
        for index, metric in enumerate(METRICS)
    }
    return {
        "prometheus": {
            "base_url": "http://prometheus.test:9090",
            "timeout_seconds": 5,
            "bearer_token_env": "PROM_TEST_TOKEN",
            "use_environment_proxy": True,
        },
        "query_step_seconds": 10,
        "windows": {
            "standard": {
                "start": 1785301200,
                "end": 1785302390,
            },
            "inference": {
                "start": 1785303000,
                "end": 1785304790,
            },
        },
        "association_target": TARGET_METRIC,
        "metrics": metrics,
        "association": {
            "top_k": 5,
            "context_ratio": 0.2,
            "min_aligned_points": 10,
            "min_rf_samples": 30,
            "min_coverage_ratio": 0.6,
            "weights": {
                "correlation": 0.5,
                "random_forest": 0.5,
            },
            "random_forest": {
                "n_estimators": 64,
                "random_state": 42,
            },
        },
    }


def _strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_example_config_is_valid_and_uses_real_project_metric_names():
    config = load_workflow_config(EXAMPLE_CONFIG)

    assert config["association_target"] == TARGET_METRIC
    assert config["prometheus"]["use_environment_proxy"] is False
    assert list(config["metrics"]) == [
        TARGET_METRIC,
        *ASSOCIATED_METRICS,
    ]
    assert "vllm:kv_cache_usage_perc{" in config["metrics"][
        "kv_cache_usage_perc"
    ]["standard_query"]
    assert "rate(vllm:e2e_request_latency_seconds_sum{" in config[
        "metrics"
    ]["e2e_request_latency"]["inference_query"]
    assert "global_seqlen_minmax_diff" in config["metrics"][
        "global_seqlen_minimax_diff"
    ]["standard_query"]
    isolation_labels = (
        "project=",
        "experiment_name=",
        "instance=",
        "worker=",
        "replica=",
        "run_id=",
    )
    for spec in config["metrics"].values():
        assert spec["standard_query"] != spec["inference_query"]
        for phase in ("standard", "inference"):
            query = spec[f"{phase}_query"]
            assert "{" in query and "}" in query
            assert any(label in query for label in isolation_labels)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value["prometheus"].update({"base_url": "ftp://host"}),
        lambda value: value["prometheus"].update(
            {"base_url": "http://user:secret@host:9090"}
        ),
        lambda value: value["prometheus"].update(
            {"use_environment_proxy": "false"}
        ),
        lambda value: value.update({"query_step_seconds": 0}),
        lambda value: value["windows"]["standard"].update(
            {"start": 20, "end": 10}
        ),
        lambda value: value["windows"]["standard"].update(
            {"start": "2026-07-29T00:00:00"}
        ),
        lambda value: value.update({"association_target": "missing"}),
        lambda value: value["metrics"][TARGET_METRIC].pop(
            "inference_query"
        ),
        lambda value: value["metrics"][TARGET_METRIC].update(
            {"query": "legacy_global_query"}
        ),
        lambda value: value["metrics"][TARGET_METRIC].update(
            {"select_labels": {"worker": "0"}}
        ),
        lambda value: value["association"]["weights"].update(
            {"correlation": 0.8}
        ),
    ],
)
def test_workflow_config_rejects_unsafe_or_ambiguous_values(mutate):
    config = _config()
    mutate(config)

    with pytest.raises(PrometheusWorkflowError) as exc_info:
        normalize_workflow_config(config)

    assert exc_info.value.code == "invalid_workflow_config"


def test_runtime_metric_config_is_reused_without_rewriting_existing_bytes(
    tmp_path,
):
    config = _config()
    config_dir = tmp_path / "runtime-config"
    metrics = {TARGET_METRIC: config["metrics"][TARGET_METRIC]}
    first = _prepare_metric_configs(
        config_dir,
        metrics,
        TARGET_METRIC,
        config["association"],
    )[TARGET_METRIC]
    annotated = (
        "# retained user comment\n"
        + first.read_text(encoding="utf-8").replace(
            "upper_ratio: 1.15",
            "upper_ratio: 1.25",
        )
    )
    first.write_text(annotated, encoding="utf-8")

    second = _prepare_metric_configs(
        config_dir,
        metrics,
        TARGET_METRIC,
        config["association"],
    )[TARGET_METRIC]

    assert second == first
    assert second.read_text(encoding="utf-8") == annotated


def test_runtime_metric_config_conflict_is_reported_without_overwrite(tmp_path):
    config = _config()
    config_dir = tmp_path / "runtime-config"
    metrics = {TARGET_METRIC: config["metrics"][TARGET_METRIC]}
    path = _prepare_metric_configs(
        config_dir,
        metrics,
        TARGET_METRIC,
        config["association"],
    )[TARGET_METRIC]
    existing = yaml.safe_load(path.read_text(encoding="utf-8"))
    existing["abnormal_type"] = "DOWN"
    path.write_text(
        yaml.safe_dump(existing, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    before = path.read_bytes()

    with pytest.raises(PrometheusWorkflowError) as exc_info:
        _prepare_metric_configs(
            config_dir,
            metrics,
            TARGET_METRIC,
            config["association"],
        )

    assert exc_info.value.code == "existing_metric_config_conflict"
    assert exc_info.value.details == {
        "logicalMetric": TARGET_METRIC,
        "conflictFields": ["abnormal_type"],
    }
    assert path.read_bytes() == before


def test_fetch_query_range_builds_real_api_request_and_reads_strict_json(
    monkeypatch,
):
    captured = {}
    payload = {
        "status": "success",
        "data": {"resultType": "matrix", "result": []},
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            captured["limit"] = limit
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(
        "experiment.degradation_perception.prometheus_workflow."
        "urllib.request.urlopen",
        fake_urlopen,
    )

    response = fetch_query_range(
        base_url="http://prometheus.test:9090/prefix",
        query='avg(metric_name{job="worker"})',
        start="2026-07-29T00:00:00Z",
        end="2026-07-29T00:20:00Z",
        step=10.0,
        timeout_seconds=12.5,
        bearer_token="not-written-anywhere",
        use_environment_proxy=True,
    )

    request = captured["request"]
    parsed = urllib.parse.urlsplit(request.full_url)
    parameters = urllib.parse.parse_qs(parsed.query)
    assert parsed.path == "/prefix/api/v1/query_range"
    assert parameters == {
        "query": ['avg(metric_name{job="worker"})'],
        "start": ["2026-07-29T00:00:00Z"],
        "end": ["2026-07-29T00:20:00Z"],
        "step": ["10"],
    }
    assert request.get_header("Authorization") == (
        "Bearer not-written-anywhere"
    )
    assert captured["timeout"] == 12.5
    assert response == payload


def test_fetch_query_range_bypasses_environment_proxy_by_default(monkeypatch):
    payload = {
        "status": "success",
        "data": {"resultType": "matrix", "result": []},
    }
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps(payload).encode("utf-8")

    class Opener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return Response()

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(
        "experiment.degradation_perception.prometheus_workflow."
        "urllib.request.build_opener",
        fake_build_opener,
    )
    monkeypatch.setattr(
        "experiment.degradation_perception.prometheus_workflow."
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("environment proxy path was used"),
    )

    assert fetch_query_range(
        base_url="http://127.0.0.1:9090",
        query="metric_name",
        start="1",
        end="2",
        step=1,
        timeout_seconds=3,
    ) == payload
    assert len(captured["handlers"]) == 1
    assert captured["handlers"][0].proxies == {}
    assert captured["timeout"] == 3


def test_real_workflow_fetches_both_windows_and_runs_full_algorithm(
    tmp_path,
    monkeypatch,
):
    package = generate_simulation_package()
    config = _config()
    normalized_config = normalize_workflow_config(config)
    query_to_phase_metric = {
        spec[f"{phase}_query"]: (phase, metric)
        for metric, spec in config["metrics"].items()
        for phase in ("standard", "inference")
    }
    calls = []
    monkeypatch.setenv("PROM_TEST_TOKEN", "secret-token-value")

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        parameters = urllib.parse.parse_qs(
            urllib.parse.urlsplit(request.full_url).query
        )
        query = parameters["query"][0]
        start = parameters["start"][0]
        calls.append(
            {
                "query": query,
                "start": start,
                "step": parameters["step"][0],
                "timeout": timeout,
                "authorization": request.get_header("Authorization"),
            }
        )
        expected_phase = (
            "standard"
            if float(start) == 1785301200.0
            else "inference"
        )
        query_phase, metric = query_to_phase_metric[query]
        assert query_phase == expected_phase
        return Response(
            copy.deepcopy(package[expected_phase][metric]["response"])
        )

    monkeypatch.setattr(
        "experiment.degradation_perception.prometheus_workflow."
        "urllib.request.urlopen",
        fake_urlopen,
    )
    output = run_prometheus_workflow(
        config,
        tmp_path / "real output",
    )

    assert output["ok"] is True
    assert len(calls) == len(METRICS) * 2
    assert all(
        call["authorization"] == "Bearer secret-token-value" for call in calls
    )
    assert all(call["step"] == "10" for call in calls)
    output_dir = tmp_path / "real output"
    expected = {
        "prometheus_query_responses.json",
        "converted_algorithm_input.json",
        "adapter_diagnostics.json",
        "analysis_result.json",
        "top5_result.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    loaded = {name: _strict_load(output_dir / name) for name in expected}
    top5 = loaded["top5_result.json"]
    assert top5["anomalyMetric"] == TARGET_METRIC
    assert top5["anomalyDetected"] is True
    assert top5["eventCount"] == 1
    assert [item["metric"] for item in top5["events"][0]["top5"]] == list(
        ASSOCIATED_METRICS
    )
    assert sum(
        item["abnormalContribution"]
        for item in loaded["analysis_result.json"]["associationAnalysis"][
            "targets"
        ][TARGET_METRIC]["events"][0]["allAssociations"]
    ) == pytest.approx(100.0)
    assert "secret-token-value" not in json.dumps(loaded)
    assert loaded["prometheus_query_responses.json"]["source"] == (
        "prometheus_query_range"
    )
    for phase in ("standard", "inference"):
        for metric, spec in config["metrics"].items():
            assert loaded["prometheus_query_responses.json"][phase][metric][
                "query"
            ] == spec[f"{phase}_query"]
            diagnostic = loaded["adapter_diagnostics.json"]["phases"][
                phase
            ][metric]
            assert diagnostic["query"] == spec[f"{phase}_query"]
            assert diagnostic["queryWindow"] == normalized_config["windows"][
                phase
            ]
            assert diagnostic["returnedSeriesCount"] == 1
            assert diagnostic["returnedSeriesLabels"]


def test_workflow_rejects_samples_outside_requested_window(tmp_path):
    package = generate_simulation_package()
    config = _config()
    config["prometheus"]["bearer_token_env"] = None
    query_to_phase_metric = {
        spec[f"{phase}_query"]: (phase, metric)
        for metric, spec in config["metrics"].items()
        for phase in ("standard", "inference")
    }

    def fake_fetcher(**kwargs):
        expected_phase = (
            "standard"
            if float(kwargs["start"]) == 1785301200.0
            else "inference"
        )
        query_phase, metric = query_to_phase_metric[kwargs["query"]]
        assert query_phase == expected_phase
        phase = expected_phase
        response = copy.deepcopy(package[phase][metric]["response"])
        if phase == "standard" and metric == TARGET_METRIC:
            response["data"]["result"][0]["values"][0][0] = 1.0
        return response

    with pytest.raises(PrometheusWorkflowError) as exc_info:
        run_prometheus_workflow(
            config,
            tmp_path / "outside",
            fetcher=fake_fetcher,
        )

    assert exc_info.value.code == "sample_outside_requested_window"
    assert exc_info.value.details["logicalMetric"] == TARGET_METRIC


def test_cli_invalid_config_returns_one_strict_json_line(tmp_path, capsys):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump({"bad": True}), encoding="utf-8")

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_workflow_config"
