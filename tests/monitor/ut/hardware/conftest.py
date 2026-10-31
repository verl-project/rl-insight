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

"""Shared fixtures for hardware monitoring unit tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from rl_insight.hardware.base import COLLECTOR_REGISTRY


@pytest.fixture(autouse=True)
def restore_collector_registry() -> Generator[None, None, None]:
    """Keep fake collector registrations from leaking between tests."""
    snapshot = dict(COLLECTOR_REGISTRY)
    yield
    COLLECTOR_REGISTRY.clear()
    COLLECTOR_REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def clean_device_visibility_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the host's device-visibility variables out of collector tests.

    The GPU/NPU collectors scope their devices to ``CUDA_VISIBLE_DEVICES`` /
    ``ASCEND_RT_VISIBLE_DEVICES``; tests that need a scoped view set the
    variable themselves, so clearing it here keeps the others hermetic
    regardless of the machine running the suite.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
