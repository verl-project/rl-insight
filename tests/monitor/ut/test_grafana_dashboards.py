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

"""Unit tests for Grafana dashboard directory staging."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rl_insight.server.runtime import _stage_grafana_dashboards


def test_stage_copies_dashboard_subdirectories(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "verl").mkdir(parents=True)
    (source / "verl" / "board.json").write_text("{}", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    conf = OmegaConf.create({"grafana": {"dashboards_dir": str(source)}})

    staged = _stage_grafana_dashboards(conf, runtime_dir)

    assert (staged / "verl" / "board.json").is_file()


DASHBOARD_FILES = sorted(
    (
        Path(__file__).resolve().parents[3]
        / "rl_insight/config/services/grafana/dashboards"
    ).glob("*/*.json")
)


def _walk_strings(node: Any) -> Generator[str, None, None]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"expr", "query", "definition"} and isinstance(value, str):
                yield value
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)


def test_every_dashboard_should_scope_queries_by_experiment_identity() -> None:
    assert DASHBOARD_FILES, "dashboard files must exist"
    for path in DASHBOARD_FILES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        spec = doc["spec"]
        variables = {
            (item.get("spec") or {}).get("name"): item.get("spec") or {}
            for item in spec.get("variables") or []
            if isinstance(item, dict)
        }
        assert "project" in variables, path.name
        experiment_spec = variables.get("experiment_name") or variables.get(
            "experiment"
        )
        assert experiment_spec, path.name
        experiment_query = (
            (experiment_spec.get("query") or {}).get("spec", {}).get("query", "")
        ) + str(experiment_spec.get("definition") or "")
        assert "$project" in experiment_query, (
            f"{path.name}: experiment variable must cascade on project"
        )

        for query in _walk_strings(spec):
            if query.startswith("label_values("):
                continue
            if "{span." in query or "span.project" in query:
                assert "span.project" in query and "span.experiment_name" in query, (
                    f"{path.name}: TraceQL must constrain span identity: {query}"
                )
            elif (
                query
                and "{" in query
                or re.search(r"[a-zA-Z_][a-zA-Z0-9_:]*\(", query)
                or query.isidentifier()
            ):
                assert 'project=~"$project"' in query and (
                    'experiment_name=~"$experiment_name"' in query
                    or 'experiment_name=~"$experiment"' in query
                ), f"{path.name}: PromQL must filter experiment identity: {query}"
