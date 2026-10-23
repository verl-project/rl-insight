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

Every test renders a minimal toy composition it writes into ``tmp_path``, never
a production dashboard, through ``rl_insight.grafana.renderer``. The optional CLI
is covered only to prove it is a thin wrapper over that same core.
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
FRAMEWORK_DIR = renderer.FRAMEWORK_DIR
COMPOSER = FRAMEWORK_DIR / "composer.libsonnet"
IMPORT_COMPOSER = f"local composer = import '{COMPOSER.as_posix()}';"

#: A minimal toy module: one panel, one row, one variable. JSON is a Jsonnet
#: subset, so the templates below stay valid for any prefix.
MODULE = """local %(p)s = {
  panels: [{ key: '%(p)s.panel', outputKey: '%(p)s-panel', id: %(id)d,
             title: '%(p)s panel', queries: [{ expr: 'toy_%(p)s_metric' }] }],
  rows: { '%(p)s-row': { kind: 'RowsLayoutRow', spec: { title: '%(p)s row',
    collapse: false, layout: { kind: 'GridLayout', spec: { items: [{
      kind: 'GridLayoutItem', spec: { x: 0, y: 0, width: 12, height: 8,
        element: { kind: 'ElementReference', name: '%(p)s.panel' } } }] } } } } },
  variables: { zone_%(p)s: { kind: 'ConstantVariable',
    spec: { name: 'zone_%(p)s', label: 'Zone', value: 'z1', hide: 'dontHide' } } },
  %(tags)s
};
"""

#: An additive module: one panel plus `rowItems` extending a row it does not own.
EXTENSION = """local %(p)s = {
  panels: [{ key: '%(p)s.panel', outputKey: '%(p)s-panel', id: %(id)d,
             title: '%(p)s panel', queries: [{ expr: 'toy_%(p)s_metric' }] }],
  rowItems: { '%(target)s': [{ kind: 'GridLayoutItem', spec: { x: 12, y: 0,
    width: 12, height: 8, element: { kind: 'ElementReference', name: '%(key)s' } } }] },
};
"""

DASHBOARD = (
    "{ metadata: { name: 'toy', uid: 'toy' }, title: 'Toy dashboard', "
    "tags: ['example'], spec: {}, variableOrder: %s, rowOrder: %s }"
)
TWO_MODULES: list[tuple[str, int, list[str] | None]] = [
    ("m1", 1, None),
    ("m2", 2, None),
]
FULL_DASHBOARD = DASHBOARD % ("['zone_m1']", "['m1-row', 'm2-row']")


def module_source(prefix: str, panel_id: int, tags: list[str] | None = None) -> str:
    tags_line = "" if tags is None else "tags: " + json.dumps(tags) + ","
    return MODULE % {"p": prefix, "id": panel_id, "tags": tags_line}


def extension_module_source(
    prefix: str, panel_id: int, target_row: str, panel_key: str
) -> str:
    return EXTENSION % {
        "p": prefix,
        "id": panel_id,
        "target": target_row,
        "key": panel_key,
    }


def write_config(tmp_path: Path, body: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "compose.jsonnet"
    path.write_text(body, encoding="utf-8")
    return path


def compose_body(
    modules: list[tuple[str, int, list[str] | None]], dashboard: str
) -> str:
    return (
        "".join(module_source(*module) for module in modules)
        + IMPORT_COMPOSER
        + "\n{ compose: composer.compose(["
        + ", ".join(prefix for prefix, _, _ in modules)
        + "], "
        + dashboard
        + ") }\n"
    )


def compose_config(
    tmp_path: Path,
    modules: list[tuple[str, int, list[str] | None]],
    dashboard: str,
    replacements: list[tuple[str, str]] | None = None,
) -> Path:
    """Write a config composing the given toy modules into ``tmp_path``."""
    body = compose_body(modules, dashboard)
    for old, new in replacements or []:
        body = body.replace(old, new)
    return write_config(tmp_path, body)


def two_dashboard_config(tmp_path: Path) -> Path:
    """A config rendering two dashboards, ``first`` before ``second``."""
    body = "".join(module_source(*module) for module in TWO_MODULES) + IMPORT_COMPOSER
    body += "\n{\n"
    for name, prefix in (("first", "m1"), ("second", "m2")):
        body += (
            f"  {name}: composer.compose([{prefix}], "
            + DASHBOARD % (f"['zone_{prefix}']", f"['{prefix}-row']")
            + "),\n"
        )
    return write_config(tmp_path, body + "}\n")


def render(tmp_path: Path, config: Path, out: str) -> dict:
    """Render through the package core and read the composed dashboard back."""
    renderer.materialize_dashboards(config, tmp_path / out)
    return json.loads((tmp_path / out / "compose.json").read_text(encoding="utf-8"))


def run_generate(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(GENERATE), *args], capture_output=True, text=True
    )
    assert proc.returncode == expect, f"{proc.returncode=}\n{proc.stderr}"
    return proc


def test_render_is_byte_deterministic(tmp_path: Path) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)

    renderer.materialize_dashboards(config, tmp_path / "run1")
    renderer.materialize_dashboards(config, tmp_path / "run2")

    assert (tmp_path / "run1" / "compose.json").read_bytes() == (
        tmp_path / "run2" / "compose.json"
    ).read_bytes()


def test_empty_registry_renders_no_dashboard(tmp_path: Path) -> None:
    # The registry ships empty: an empty composition set is valid, renders zero
    # dashboards and writes nothing, so the runtime can keep its static ones.
    config = write_config(tmp_path, "{}\n")
    output_dir = tmp_path / "out"

    assert renderer.render_dashboards(config) == {}
    assert renderer.materialize_dashboards(config, output_dir) == []
    assert list(output_dir.glob("*.json")) == []


def test_check_mode_passes_when_in_sync_and_reports_stale_files(
    tmp_path: Path,
) -> None:
    config = compose_config(tmp_path, TWO_MODULES, FULL_DASHBOARD)
    expected_dir = tmp_path / "expected"
    renderer.materialize_dashboards(config, expected_dir)
    assert renderer.stale_dashboards(config, expected_dir) == []

    stale = json.loads((expected_dir / "compose.json").read_text(encoding="utf-8"))
    stale["spec"]["title"] = "tampered"
    (expected_dir / "compose.json").write_text(
        json.dumps(stale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    assert renderer.stale_dashboards(config, expected_dir) == [
        expected_dir / "compose.json"
    ]


def test_materialize_writes_dashboards_and_honours_overwrite(tmp_path: Path) -> None:
    config = two_dashboard_config(tmp_path)
    output_dir = tmp_path / "nested" / "out"

    # Additive mode writes every dashboard and creates the output directory.
    written = renderer.materialize_dashboards(config, output_dir, overwrite=False)
    assert written == [output_dir / "first.json", output_dir / "second.json"]
    for path in written:
        assert json.loads(path.read_text(encoding="utf-8"))["spec"]["title"] == (
            "Toy dashboard"
        )

    # The default stays a plain write that replaces the target path.
    for path in written:
        path.write_text("old\n", encoding="utf-8")
    assert renderer.materialize_dashboards(config, output_dir) == written
    assert "Toy dashboard" in written[0].read_text(encoding="utf-8")


def test_materialize_without_overwrite_rejects_existing_targets(
    tmp_path: Path,
) -> None:
    # Every target is checked before the first write: a collision on the second
    # dashboard leaves the first one unwritten and both paths are reported.
    config = two_dashboard_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in ("first", "second"):
        (output_dir / f"{name}.json").write_text("pre-existing\n", encoding="utf-8")

    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.materialize_dashboards(config, output_dir, overwrite=False)

    message = str(excinfo.value)
    assert str(output_dir / "first.json") in message
    assert str(output_dir / "second.json") in message
    assert (output_dir / "first.json").read_text(encoding="utf-8") == "pre-existing\n"


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [(None, "composition config not found"), ("[1, 2, 3]\n", "object of dashboards")],
)
def test_invalid_config_is_rejected_with_its_path(
    tmp_path: Path, body: str | None, expected_error: str
) -> None:
    config = (
        tmp_path / "absent.jsonnet" if body is None else write_config(tmp_path, body)
    )

    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)

    assert expected_error in str(excinfo.value)
    assert str(config) in str(excinfo.value)


def test_evaluation_failure_names_the_config_and_the_jsonnet_context(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path, "{ compose: error 'toy composition failure' }\n")

    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)

    message = str(excinfo.value)
    assert str(config) in message
    assert "runtime error: toy composition failure" in message


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
    config = compose_config(
        tmp_path,
        TWO_MODULES,
        FULL_DASHBOARD,
        replacements=[("name: 'm2.panel'", "name: 'm1.panel'")],
    )

    composed = render(tmp_path, config, "out")["spec"]

    assert set(composed["elements"]) == {"m1-panel", "m2-panel"}
    second_row = composed["layout"]["spec"]["rows"][1]
    referenced = second_row["spec"]["layout"]["spec"]["items"][0]["spec"]["element"]
    assert referenced == {"kind": "ElementReference", "name": "m1-panel"}


CONFLICT_CASES = [
    ("duplicate panel key(s)", [("key: 'm2.panel'", "key: 'm1.panel'")]),
    ("duplicate panel outputKey(s)", [("m2-panel", "m1-panel")]),
    ("duplicate panel id(s)", [("id: 2,", "id: 1,")]),
    ("duplicate row name(s)", [("m2-row", "m1-row")]),
    ("duplicate variable name(s)", [("zone_m2", "zone_m1")]),
]


@pytest.mark.parametrize(("expected_error", "replacements"), CONFLICT_CASES)
def test_composition_conflicts_are_rejected(
    tmp_path: Path, expected_error: str, replacements: list[tuple[str, str]]
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


@pytest.mark.parametrize(
    ("dashboard", "expected_error"),
    [
        (
            DASHBOARD % ("['zone_m1', 'missing_var']", "['m1-row']"),
            "unknown variable(s) in variableOrder: missing_var",
        ),
        (
            DASHBOARD % ("['zone_m1']", "['m1-row', 'missing_row']"),
            "unknown row(s) in rowOrder: missing_row",
        ),
    ],
)
def test_unknown_order_entries_are_rejected(
    tmp_path: Path, dashboard: str, expected_error: str
) -> None:
    config = compose_config(tmp_path, TWO_MODULES, dashboard)

    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)

    assert expected_error in str(excinfo.value)


def test_composition_order_decides_tag_and_variable_order(tmp_path: Path) -> None:
    def composed(
        modules: list[tuple[str, int, list[str] | None]],
        variable_order: str,
        out: str,
    ) -> dict:
        config = compose_config(
            tmp_path / f"src-{out}",
            modules,
            DASHBOARD % (variable_order, "['m1-row']"),
        )
        return render(tmp_path, config, f"out-{out}")["spec"]

    tagged: list[tuple[str, int, list[str] | None]] = [
        ("m1", 1, ["one"]),
        ("m2", 2, ["two"]),
    ]
    forward = composed(tagged, "['zone_m1', 'zone_m2']", "fwd")
    reversed_modules = composed(
        [("m2", 2, ["two"]), ("m1", 1, ["one"])], "['zone_m2', 'zone_m1']", "rev"
    )

    assert forward["tags"] == ["example", "one", "two"]
    assert reversed_modules["tags"] == ["example", "two", "one"]
    assert [item["spec"]["name"] for item in forward["variables"]] == [
        "zone_m1",
        "zone_m2",
    ]
    assert [item["spec"]["name"] for item in reversed_modules["variables"]] == [
        "zone_m2",
        "zone_m1",
    ]


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


def extension_config(
    tmp_path: Path, module_order: list[str], extra_sources: str
) -> Path:
    """A config whose base module ``m1`` owns the row the extensions append to."""
    body = (
        module_source("m1", 1)
        + extra_sources
        + IMPORT_COMPOSER
        + "\n{ compose: composer.compose(["
        + ", ".join(module_order)
        + "], "
        + DASHBOARD % ("['zone_m1']", "['m1-row']")
        + ") }\n"
    )
    return write_config(tmp_path, body)


def test_row_items_single_extension_appends_and_resolves(tmp_path: Path) -> None:
    # The extension module owns no rows (panels + rowItems only) and appends one
    # item to the row owned by the base module; the appended ElementReference
    # resolves through key -> outputKey like any other.
    config = extension_config(
        tmp_path,
        ["m1", "m1x"],
        extension_module_source("m1x", 11, "m1-row", "m1x.panel"),
    )

    composed = render(tmp_path, config, "out")

    assert set(composed["spec"]["elements"]) == {"m1-panel", "m1x-panel"}
    assert row_item_names(composed, "m1 row") == ["m1-panel", "m1x-panel"]
    # rowItems never creates rows: the only row is the one m1 owns.
    assert len(composed["spec"]["layout"]["spec"]["rows"]) == 1


def test_row_items_append_in_composition_order(tmp_path: Path) -> None:
    sources = extension_module_source("e1", 11, "m1-row", "e1.panel") + (
        extension_module_source("e2", 12, "m1-row", "e2.panel")
    )

    forward = render(
        tmp_path,
        extension_config(tmp_path / "src-fwd", ["m1", "e1", "e2"], sources),
        "out-fwd",
    )
    reversed_order = render(
        tmp_path,
        extension_config(tmp_path / "src-rev", ["m1", "e2", "e1"], sources),
        "out-rev",
    )

    assert row_item_names(forward, "m1 row") == ["m1-panel", "e1-panel", "e2-panel"]
    assert row_item_names(reversed_order, "m1 row") == [
        "m1-panel",
        "e2-panel",
        "e1-panel",
    ]


@pytest.mark.parametrize(
    ("target", "break_layout", "expected_error"),
    [
        ("missing-row", False, "unknown row extension target: missing-row"),
        ("m1-row", True, "unsupported row extension target"),
    ],
)
def test_row_items_invalid_targets_are_rejected(
    tmp_path: Path, target: str, break_layout: bool, expected_error: str
) -> None:
    # A target row that does not exist, or whose layout is not a GridLayout,
    # must fail loudly instead of silently ignoring the extension.
    base = module_source("m1", 1)
    if break_layout:
        base = base.replace("kind: 'GridLayout'", "kind: 'RowsLayout'")
    body = (
        base
        + extension_module_source("e1", 11, target, "m1.panel")
        + IMPORT_COMPOSER
        + "\n{ compose: composer.compose([m1, e1], "
        + DASHBOARD % ("['zone_m1']", "['m1-row']")
        + ") }\n"
    )
    config = write_config(tmp_path, body)

    with pytest.raises(renderer.JsonnetRenderError) as excinfo:
        renderer.render_dashboards(config)

    assert expected_error in str(excinfo.value)


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
