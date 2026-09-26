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

"""Tests for the generic Grafana dashboard composition framework.

The framework is independent of any production dashboard or committed fixture:
every test renders a minimal composition config that the test writes into
``tmp_path``.

Rendering goes through the installed-package core,
``rl_insight.grafana.renderer``. The optional CLI in
``tools/grafana/framework/generate.py`` is covered separately, only to prove it
is a thin wrapper over that same core.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rl_insight.grafana import renderer

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATE = REPO_ROOT / "tools" / "grafana" / "framework" / "generate.py"

#: The generic Jsonnet assets that must ship inside the installed package.
FRAMEWORK_DIR = renderer.FRAMEWORK_DIR
COMPOSER = FRAMEWORK_DIR / "composer.libsonnet"


def run_generate(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(GENERATE), *args], capture_output=True, text=True
    )
    assert proc.returncode == expect, (
        f"expected rc={expect}, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def write_config(tmp_path: Path, body: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "compose.jsonnet"
    path.write_text(body, encoding="utf-8")
    return path


def module_source(prefix: str, panel_id: int, tags: list[str] | None = None) -> str:
    """A minimal toy module as Jsonnet source (JSON is a Jsonnet subset)."""
    module: dict = {
        "panels": [
            {
                "key": f"{prefix}.panel",
                "outputKey": f"{prefix}-panel",
                "id": panel_id,
                "title": f"{prefix} panel",
                "queries": [{"expr": f"toy_{prefix}_metric"}],
            }
        ],
        "rows": {
            f"{prefix}-row": {
                "kind": "RowsLayoutRow",
                "spec": {
                    "title": f"{prefix} row",
                    "collapse": False,
                    "layout": {
                        "kind": "GridLayout",
                        "spec": {
                            "items": [
                                {
                                    "kind": "GridLayoutItem",
                                    "spec": {
                                        "x": 0,
                                        "y": 0,
                                        "width": 12,
                                        "height": 8,
                                        "element": {
                                            "kind": "ElementReference",
                                            "name": f"{prefix}.panel",
                                        },
                                    },
                                }
                            ]
                        },
                    },
                },
            }
        },
        "variables": {
            f"zone_{prefix}": {
                "kind": "ConstantVariable",
                "spec": {
                    "name": f"zone_{prefix}",
                    "label": "Zone",
                    "value": "z1",
                    "hide": "dontHide",
                },
            }
        },
    }
    if tags is not None:
        module["tags"] = tags
    return f"local {prefix} = " + json.dumps(module) + ";"


def compose_config(
    tmp_path: Path,
    modules: list[tuple[str, int, list[str] | None]],
    dashboard: str,
    replacements: list[tuple[str, str]] | None = None,
) -> Path:
    """Write a config composing the given toy modules into ``tmp_path``."""
    body = ""
    for prefix, panel_id, tags in modules:
        source = module_source(prefix, panel_id, tags)
        for old, new in replacements or []:
            source = source.replace(old, new)
        body += source
    body += (
        "local composer = import '" + COMPOSER.as_posix() + "';\n"
        "{ compose: composer.compose(["
        + ", ".join(prefix for prefix, _, _ in modules)
        + "], "
        + dashboard
        + ") }\n"
    )
    return write_config(tmp_path, body)


def render(tmp_path: Path, config: Path, out: str) -> dict:
    """Render through the package core and read the composed dashboard back."""
    renderer.materialize_dashboards(config, tmp_path / out)
    path = tmp_path / out / "compose.json"
    return json.loads(path.read_text(encoding="utf-8"))


def extension_module_source(
    prefix: str, panel_id: int, target_row: str, panel_key: str
) -> str:
    """An additive module: one panel plus `rowItems`, owning no rows."""
    module: dict = {
        "panels": [
            {
                "key": f"{prefix}.panel",
                "outputKey": f"{prefix}-panel",
                "id": panel_id,
                "title": f"{prefix} panel",
                "queries": [{"expr": f"toy_{prefix}_metric"}],
            }
        ],
        "rowItems": {
            target_row: [
                {
                    "kind": "GridLayoutItem",
                    "spec": {
                        "x": 12,
                        "y": 0,
                        "width": 12,
                        "height": 8,
                        "element": {
                            "kind": "ElementReference",
                            "name": panel_key,
                        },
                    },
                }
            ]
        },
    }
    return f"local {prefix} = " + json.dumps(module) + ";"


DASHBOARD = (
    "{ metadata: { name: 'toy', uid: 'toy' }, title: 'Toy dashboard', "
    "tags: ['example'], spec: {}, variableOrder: %s, rowOrder: %s }"
)
TWO_MODULES: list[tuple[str, int, list[str] | None]] = [
    ("m1", 1, None),
    ("m2", 2, None),
]
FULL_DASHBOARD = DASHBOARD % ("['zone_m1']", "['m1-row', 'm2-row']")


def test_render_is_byte_deterministic(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    renderer.materialize_dashboards(config, tmp_path / "run1")
    renderer.materialize_dashboards(config, tmp_path / "run2")
    first = (tmp_path / "run1" / "compose.json").read_bytes()
    assert first == (tmp_path / "run2" / "compose.json").read_bytes()
    composed = json.loads(first)["spec"]
    assert [row["spec"]["title"] for row in composed["layout"]["spec"]["rows"]] == [
        "m1 row",
        "m2 row",
    ]


def test_check_mode_passes_when_in_sync(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    renderer.materialize_dashboards(config, tmp_path / "expected")
    assert renderer.stale_dashboards(config, tmp_path / "expected") == []


def test_check_mode_detects_stale_expected_output(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    expected_dir = tmp_path / "expected"
    renderer.materialize_dashboards(config, expected_dir)
    stale = json.loads((expected_dir / "compose.json").read_text(encoding="utf-8"))
    stale["spec"]["title"] = "tampered"
    (expected_dir / "compose.json").write_text(
        json.dumps(stale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert renderer.stale_dashboards(config, expected_dir) == [
        expected_dir / "compose.json"
    ]


def test_materialize_writes_every_dashboard_and_creates_the_output_dir(
    tmp_path: Path,
) -> None:
    config = write_config(
        tmp_path,
        module_source("m1", 1)
        + module_source("m2", 2)
        + "local composer = import '"
        + COMPOSER.as_posix()
        + "';\n"
        "{\n"
        "  first: composer.compose([m1], "
        + DASHBOARD
        % ("['zone_m1']", "['m1-row']")
        + "),\n"
        "  second: composer.compose([m2], "
        + DASHBOARD
        % ("['zone_m2']", "['m2-row']")
        + "),\n"
        "}\n",
    )
    output_dir = tmp_path / "nested" / "out"
    written = renderer.materialize_dashboards(config, output_dir)
    assert written == [output_dir / "first.json", output_dir / "second.json"]
    for path in written:
        assert json.loads(path.read_text(encoding="utf-8"))["spec"]["title"] == (
            "Toy dashboard"
        )


def test_evaluation_failure_names_the_config_and_the_jsonnet_context(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path, "{ compose: error 'toy composition failure' }\n")
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    message = str(excinfo.value)
    assert str(config) in message
    assert "runtime error: toy composition failure" in message


def test_missing_config_is_rejected_with_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jsonnet"
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(missing)
    assert str(missing) in str(excinfo.value)


def test_rendering_never_shells_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The core must evaluate in-process: no Jsonnet CLI, no subprocess, no
    # gojsonnet binding. Break every escape hatch and render anyway.
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    source = Path(renderer.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "gojsonnet" not in source

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the renderer must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert set(renderer.render_dashboards(config)) == {"compose"}


def test_framework_assets_ship_inside_the_installed_package() -> None:
    assert FRAMEWORK_DIR.is_dir()
    for name in ("composer.libsonnet", "viz.libsonnet"):
        assert (FRAMEWORK_DIR / name).is_file()
    # No second copy of the framework outside the package.
    assert not (
        REPO_ROOT / "tools" / "grafana" / "framework" / "composer.libsonnet"
    ).exists()


def test_layout_references_resolve_across_modules(tmp_path: Path) -> None:
    # The second module's layout row references the first module's panel by
    # `key`; the composer rewrites it to the panel's `outputKey` when rendering.
    m2 = module_source("m2", 2).replace('"name": "m2.panel"', '"name": "m1.panel"')
    body = (
        module_source("m1", 1)
        + m2
        + "local composer = import '"
        + COMPOSER.as_posix()
        + "';\n"
        + "{ compose: composer.compose([m1, m2], "
        + FULL_DASHBOARD
        + ") }\n"
    )
    config = write_config(tmp_path, body)
    composed = render(tmp_path, config, "out")["spec"]
    assert set(composed["elements"]) == {"m1-panel", "m2-panel"}
    second_row = composed["layout"]["spec"]["rows"][1]
    referenced = second_row["spec"]["layout"]["spec"]["items"][0]["spec"]["element"]
    assert referenced == {"kind": "ElementReference", "name": "m1-panel"}


CONFLICT_CASES = [
    ("duplicate panel key(s)", [('"key": "m2.panel"', '"key": "m1.panel"')]),
    ("duplicate panel outputKey(s)", [("m2-panel", "m1-panel")]),
    ("duplicate panel id(s)", [('"id": 2', '"id": 1')]),
    ("duplicate row name(s)", [("m2-row", "m1-row")]),
    ("duplicate variable name(s)", [("zone_m2", "zone_m1")]),
]


@pytest.mark.parametrize(("expected_error", "replacements"), CONFLICT_CASES)
def test_composition_conflicts_are_rejected(
    tmp_path: Path, expected_error: str, replacements: list[tuple[str, str]] | None
) -> None:
    config = compose_config(
        tmp_path,
        TWO_MODULES,
        DASHBOARD % ("['zone_m1']", "['m1-row']"),
        replacements=replacements,
    )
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    assert expected_error in str(excinfo.value)


def test_unknown_variable_in_variable_order_is_rejected(tmp_path: Path) -> None:
    config = compose_config(
        tmp_path, TWO_MODULES, DASHBOARD % ("['zone_m1', 'missing_var']", "['m1-row']")
    )
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    assert "unknown variable(s) in variableOrder: missing_var" in str(excinfo.value)


def test_unknown_row_in_row_order_is_rejected(tmp_path: Path) -> None:
    config = compose_config(
        tmp_path, TWO_MODULES, DASHBOARD % ("['zone_m1']", "['m1-row', 'missing_row']")
    )
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    assert "unknown row(s) in rowOrder: missing_row" in str(excinfo.value)


def test_composition_order_decides_tag_sequence(tmp_path: Path) -> None:
    forward = render(
        tmp_path,
        compose_config(
            tmp_path / "src-f",
            [("m1", 1, ["one"]), ("m2", 2, ["two"])],
            DASHBOARD % ("['zone_m1']", "['m1-row']"),
        ),
        "out-f",
    )["spec"]["tags"]
    reversed_tags = render(
        tmp_path,
        compose_config(
            tmp_path / "src-r",
            [("m2", 2, ["two"]), ("m1", 1, ["one"])],
            DASHBOARD % ("['zone_m1']", "['m1-row']"),
        ),
        "out-r",
    )["spec"]["tags"]
    assert forward == ["example", "one", "two"]
    assert reversed_tags == ["example", "two", "one"]


def test_variable_order_follows_the_composition_config(tmp_path: Path) -> None:
    def variable_names(variable_order: str, out: str) -> list[str]:
        composed = render(
            tmp_path,
            compose_config(
                tmp_path / f"src-{out}",
                TWO_MODULES,
                DASHBOARD % (variable_order, "['m1-row']"),
            ),
            f"out-{out}",
        )["spec"]
        return [variable["spec"]["name"] for variable in composed["variables"]]

    assert variable_names("['zone_m1', 'zone_m2']", "fwd") == ["zone_m1", "zone_m2"]
    assert variable_names("['zone_m2', 'zone_m1']", "rev") == ["zone_m2", "zone_m1"]


def row_item_names(composed: dict, row_name: str) -> list[str]:
    row = next(
        candidate
        for candidate in composed["spec"]["layout"]["spec"]["rows"]
        if candidate["spec"]["title"] == row_name
    )
    return [
        item["spec"]["element"]["name"]
        for item in row["spec"]["layout"]["spec"]["items"]
    ]


def test_row_items_single_extension_appends_and_resolves(tmp_path: Path) -> None:
    # The extension module owns no rows (panels + rowItems only) and appends
    # one item to the row owned by the base module; the appended
    # ElementReference resolves through key -> outputKey like any other.
    body = (
        module_source("m1", 1)
        + extension_module_source("m1x", 11, "m1-row", "m1x.panel")
        + "local composer = import '"
        + COMPOSER.as_posix()
        + "';\n"
        + "{ compose: composer.compose([m1, m1x], "
        + DASHBOARD % ("['zone_m1']", "['m1-row']")
        + ") }\n"
    )
    config = write_config(tmp_path, body)
    composed = render(tmp_path, config, "out")
    assert set(composed["spec"]["elements"]) == {"m1-panel", "m1x-panel"}
    assert row_item_names(composed, "m1 row") == ["m1-panel", "m1x-panel"]
    # rowItems never creates rows: the only row is the one m1 owns.
    assert len(composed["spec"]["layout"]["spec"]["rows"]) == 1


def test_row_items_append_in_composition_order(tmp_path: Path) -> None:
    def composed_items(module_order: list[str], out: str) -> list[str]:
        body = (
            module_source("m1", 1)
            + extension_module_source("e1", 11, "m1-row", "e1.panel")
            + extension_module_source("e2", 12, "m1-row", "e2.panel")
            + "local composer = import '"
            + COMPOSER.as_posix()
            + "';\n"
            + "{ compose: composer.compose(["
            + ", ".join(module_order)
            + "], "
            + DASHBOARD % ("['zone_m1']", "['m1-row']")
            + ") }\n"
        )
        config = write_config(tmp_path / f"src-{out}", body)
        return row_item_names(render(tmp_path, config, f"out-{out}"), "m1 row")

    assert composed_items(["m1", "e1", "e2"], "fwd") == [
        "m1-panel",
        "e1-panel",
        "e2-panel",
    ]
    assert composed_items(["m1", "e2", "e1"], "rev") == [
        "m1-panel",
        "e2-panel",
        "e1-panel",
    ]


def test_row_items_unknown_target_is_rejected(tmp_path: Path) -> None:
    body = (
        module_source("m1", 1)
        + extension_module_source("e1", 11, "missing-row", "m1.panel")
        + "local composer = import '"
        + COMPOSER.as_posix()
        + "';\n"
        + "{ compose: composer.compose([m1, e1], "
        + DASHBOARD % ("['zone_m1']", "['m1-row']")
        + ") }\n"
    )
    config = write_config(tmp_path, body)
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    assert "unknown row extension target: missing-row" in str(excinfo.value)


def test_row_items_unsupported_target_is_rejected(tmp_path: Path) -> None:
    # A target row whose layout is not a GridLayout cannot take items; the
    # composer must fail loudly instead of silently ignoring the extension.
    body = (
        module_source("m1", 1).replace('"kind": "GridLayout"', '"kind": "RowsLayout"')
        + extension_module_source("e1", 11, "m1-row", "e1.panel")
        + "local composer = import '"
        + COMPOSER.as_posix()
        + "';\n"
        + "{ compose: composer.compose([m1, e1], "
        + DASHBOARD % ("['zone_m1']", "['m1-row']")
        + ") }\n"
    )
    config = write_config(tmp_path, body)
    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)
    assert "unsupported row extension target" in str(excinfo.value)


def test_cli_writes_exactly_what_the_core_renders(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    cli_out = tmp_path / "cli-out"
    run_generate("--config", str(config), "--out-dir", str(cli_out))
    expected = renderer.generated_text(renderer.render_dashboards(config)["compose"])
    assert (cli_out / "compose.json").read_text(encoding="utf-8") == expected


def test_cli_check_matches_the_core_and_reports_stale_files(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    expected_dir = tmp_path / "expected"
    run_generate("--config", str(config), "--out-dir", str(expected_dir))
    run_generate(
        "--config", str(config), "--check", "--expected-dir", str(expected_dir)
    )
    (expected_dir / "compose.json").write_text("{}\n", encoding="utf-8")
    proc = run_generate(
        "--config",
        str(config),
        "--check",
        "--expected-dir",
        str(expected_dir),
        expect=1,
    )
    assert "compose.json" in proc.stderr


def test_cli_reports_evaluation_failures_with_the_config_path(tmp_path: Path) -> None:
    config = write_config(tmp_path, "{ compose: error 'toy composition failure' }\n")
    proc = run_generate(
        "--config", str(config), "--out-dir", str(tmp_path / "out"), expect=2
    )
    assert str(config) in proc.stderr
    assert "runtime error: toy composition failure" in proc.stderr
