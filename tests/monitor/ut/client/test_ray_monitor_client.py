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

"""Unit tests for the Ray monitor client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from rl_insight.client import ray_monitor_client as client_module
from rl_insight.utils.constants import MonitorRayActor

_JOB_ACTOR_NAME = f"{MonitorRayActor.NAME}_test123"


def _patch_actor_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_current_job_actor_name`` to return a predictable job-scoped name."""
    monkeypatch.setattr(
        client_module,
        "_current_job_actor_name",
        lambda: _JOB_ACTOR_NAME,
    )


def test_create_ray_monitor_client_should_return_none_when_ray_is_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module.ray, "is_initialized", lambda: False)

    assert client_module.create_ray_monitor_client(OmegaConf.create({})) is None


def test_get_or_create_monitor_hub_should_reuse_actor_when_actor_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_actor_name(monkeypatch)

    actor = MagicMock()
    get_actor = MagicMock(return_value=actor)
    monkeypatch.setattr(client_module.ray, "get_actor", get_actor)

    assert client_module.get_or_create_monitor_hub(OmegaConf.create({})) is actor
    get_actor.assert_called_once_with(
        _JOB_ACTOR_NAME, namespace=MonitorRayActor.NAMESPACE
    )


def test_get_or_create_monitor_hub_should_create_actor_when_actor_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_actor_name(monkeypatch)

    conf = OmegaConf.create({"server": {"url": "http://server"}})
    actor = MagicMock()
    remote = MagicMock(return_value=actor)
    options = MagicMock(return_value=MagicMock(remote=remote))
    monkeypatch.setattr(
        client_module.ray, "get_actor", MagicMock(side_effect=ValueError)
    )
    monkeypatch.setattr(client_module, "MonitorHubActor", MagicMock(options=options))

    assert client_module.get_or_create_monitor_hub(conf) is actor
    options.assert_called_once_with(
        name=_JOB_ACTOR_NAME,
        namespace=MonitorRayActor.NAMESPACE,
    )
    remote.assert_called_once_with(conf)


def test_get_or_create_monitor_hub_should_reuse_winner_when_creation_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_actor_name(monkeypatch)

    conf = OmegaConf.create({"server": {"url": "http://server"}})
    winner = MagicMock()
    get_actor = MagicMock(side_effect=[winner])
    remote = MagicMock(side_effect=ValueError)
    monkeypatch.setattr(client_module.ray, "get_actor", get_actor)
    monkeypatch.setattr(
        client_module,
        "MonitorHubActor",
        MagicMock(options=MagicMock(return_value=MagicMock(remote=remote))),
    )

    assert client_module.get_or_create_monitor_hub(conf) is winner
    get_actor.assert_called_with(_JOB_ACTOR_NAME, namespace=MonitorRayActor.NAMESPACE)


def test_apply_event_should_submit_without_waiting_when_client_has_actor() -> None:
    actor = MagicMock()
    client = client_module.MonitorRayClient(actor)
    event = {"kind": "counter", "name": "steps", "value": 1}

    client.apply_event(event)

    actor.apply_event.remote.assert_called_once_with(event)
