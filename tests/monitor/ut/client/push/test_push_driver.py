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

"""Unit tests for dotted-path sink driver loading."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from omegaconf import OmegaConf

from rl_insight.client.push import driver as driver_module
from rl_insight.client.push.sinks import LoggingSink

LOGGING_DRIVER = "rl_insight.client.push.sinks.logging:create_sink"


def _install_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch):
    created: list[str] = []

    def install(name: str, **attrs: Any) -> types.ModuleType:
        created.append(name)
        return _install_module(name, **attrs)

    yield install
    for name in created:
        sys.modules.pop(name, None)


def test_load_sink_factory_should_resolve_builtin_logging_driver() -> None:
    factory = driver_module.load_sink_factory(LOGGING_DRIVER)
    sink = factory(OmegaConf.create({"prefix": "p"}))
    assert isinstance(sink, LoggingSink)
    assert sink.prefix == "p"


@pytest.mark.parametrize("bad", ["", "nocolon", ":attr", "mod:", "mod :"])
def test_load_sink_factory_should_reject_malformed_path(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid sink driver"):
        driver_module.load_sink_factory(bad)


def test_load_sink_factory_should_raise_on_missing_module() -> None:
    with pytest.raises(ImportError):
        driver_module.load_sink_factory("no_such_module_xyz:factory")


def test_load_sink_factory_should_raise_on_missing_attribute(fake_modules) -> None:
    fake_modules("fake_drv_missing_attr_mod")
    with pytest.raises(ImportError, match="has no attribute"):
        driver_module.load_sink_factory("fake_drv_missing_attr_mod:factory")


def test_load_sink_factory_should_reject_non_callable(fake_modules) -> None:
    fake_modules("fake_drv_not_callable_mod", factory=42)
    with pytest.raises(TypeError, match="not callable"):
        driver_module.load_sink_factory("fake_drv_not_callable_mod:factory")


def test_build_sink_should_instantiate_logging_sink() -> None:
    conf = OmegaConf.create({"name": "lg", "driver": LOGGING_DRIVER, "prefix": "root"})
    sink = driver_module.build_sink(conf)
    assert isinstance(sink, LoggingSink)
    assert sink.prefix == "root"


def test_build_sink_should_return_none_when_driver_missing() -> None:
    assert driver_module.build_sink(OmegaConf.create({"name": "x"})) is None


def test_build_sink_should_return_none_when_module_unimportable() -> None:
    conf = OmegaConf.create({"driver": "no_such_module_xyz:factory"})
    assert driver_module.build_sink(conf) is None


def test_build_sink_should_return_none_when_factory_returns_none(fake_modules) -> None:
    fake_modules("fake_drv_none_mod", factory=lambda conf: None)
    conf = OmegaConf.create({"name": "n", "driver": "fake_drv_none_mod:factory"})
    assert driver_module.build_sink(conf) is None


def test_build_sink_should_return_none_for_non_sink_result(fake_modules) -> None:
    fake_modules("fake_drv_wrong_mod", factory=lambda conf: object())
    conf = OmegaConf.create({"name": "w", "driver": "fake_drv_wrong_mod:factory"})
    assert driver_module.build_sink(conf) is None


def test_build_sink_should_swallow_factory_exception(fake_modules) -> None:
    def boom(conf: Any) -> Any:
        raise RuntimeError("kaboom")

    fake_modules("fake_drv_boom_mod", factory=boom)
    conf = OmegaConf.create({"name": "b", "driver": "fake_drv_boom_mod:factory"})
    assert driver_module.build_sink(conf) is None
