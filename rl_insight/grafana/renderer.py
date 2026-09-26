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

"""Render Grafana dashboard JSON from a Jsonnet composition config.

A composition config is a Jsonnet file that imports the generic composer and
evaluates to ``{ "<dashboard-name>": <dashboard resource> }``, where each
dashboard resource is the object returned by ``composer.compose(modules,
dashboard)`` for that dashboard.

This module is the single runtime core for that rendering: it evaluates the
config in-process through the ``rjsonnet`` binding and serializes the result
deterministically. Nothing here shells out to a Jsonnet CLI or to another
Python entry point, so importing :mod:`rl_insight.grafana.renderer` is enough
to render dashboards on every supported platform.

Determinism: the composer emits object fields in sorted order and
:func:`generated_text` uses a fixed indentation, so two runs over the same
sources always produce byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rjsonnet

#: Directory holding the generic Jsonnet framework assets that ship inside the
#: installed package (``composer.libsonnet``, ``viz.libsonnet``). Composition
#: configs import the composer from here, so there is exactly one source of
#: truth for the framework, shared by the library and by the optional CLI.
FRAMEWORK_DIR = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "services"
    / "grafana"
    / "jsonnet"
    / "framework"
)


class JsonnetRenderError(RuntimeError):
    """A composition config could not be evaluated or serialized.

    The message always names the offending config path and, for evaluation
    failures, keeps the Jsonnet stack trace produced by the evaluator.
    """


def _evaluate(config: Path) -> str:
    """Evaluate ``config`` with the in-process ``rjsonnet`` binding."""
    try:
        return rjsonnet.evaluate_file(str(config))
    except Exception as error:
        raise JsonnetRenderError(
            f"Jsonnet evaluation failed for config {config}: {error}"
        ) from error


def render_dashboards(config: Path) -> dict[str, Any]:
    """Evaluate ``config`` and return its ``{name: dashboard}`` mapping.

    Raises :class:`JsonnetRenderError` when the config is missing, fails to
    evaluate, does not produce JSON, or does not produce a non-empty object of
    dashboards.
    """
    config = Path(config)
    if not config.is_file():
        raise JsonnetRenderError(f"composition config not found: {config}")

    rendered = _evaluate(config)

    try:
        dashboards = json.loads(rendered)
    except json.JSONDecodeError as error:
        raise JsonnetRenderError(
            f"Jsonnet evaluation of {config} did not produce JSON: {error}"
        ) from error

    if not isinstance(dashboards, dict) or not dashboards:
        raise JsonnetRenderError(
            f"{config} must evaluate to a non-empty object of dashboards"
        )
    return dashboards


def generated_text(dashboard: Any) -> str:
    """Serialize one dashboard deterministically (stable bytes across runs)."""
    return json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n"


def materialize_dashboards(config: Path, output_dir: Path) -> list[Path]:
    """Render ``config`` and write one ``<dashboard-name>.json`` per dashboard.

    Returns the written paths in the order the config declares them. The
    output directory is created when missing.
    """
    dashboards = render_dashboards(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, dashboard in dashboards.items():
        path = output_dir / f"{name}.json"
        path.write_text(generated_text(dashboard), encoding="utf-8")
        written.append(path)
    return written


def stale_dashboards(config: Path, expected_dir: Path) -> list[Path]:
    """Return the expected files that do not match ``config``'s rendering.

    A file is stale when it is missing or differs byte-for-byte from the
    deterministic rendering. An empty result means the checked-in dashboards
    are up to date.
    """
    dashboards = render_dashboards(config)
    expected_dir = Path(expected_dir)

    stale: list[Path] = []
    for name, dashboard in dashboards.items():
        path = expected_dir / f"{name}.json"
        if not path.exists() or path.read_text(encoding="utf-8") != generated_text(
            dashboard
        ):
            stale.append(path)
    return stale
