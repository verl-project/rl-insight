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

"""Unit tests for built-in rollout rule packs and the pack/inline loader."""

from __future__ import annotations

from omegaconf import OmegaConf

from rl_insight.client.push.client import _load_pack, _load_rollout_rules


def _sources(rules: list) -> list[str]:
    return [str(OmegaConf.select(r, "source")) for r in rules]


def test_load_vllm_pack_tags_group_and_keeps_source() -> None:
    rules = _load_pack("vllm")
    assert rules
    assert all(str(OmegaConf.select(r, "group")) == "vllm" for r in rules)
    sources = _sources(rules)
    assert "vllm:num_requests_running" in sources
    assert "vllm:generation_tokens_total" in sources
    assert "vllm:time_to_first_token_seconds" in sources


def test_load_tq_pack_tags_group() -> None:
    rules = _load_pack("tq")
    assert rules
    assert all(str(OmegaConf.select(r, "group")) == "tq" for r in rules)
    assert "tq_controller_request_total" in _sources(rules)
    assert "tq_partition_production_progress" in _sources(rules)


def test_load_rollout_rules_merges_packs_and_inline() -> None:
    rollout_conf = OmegaConf.create(
        {
            "packs": ["vllm", "tq"],
            "metrics": [
                {"source": "eng:custom", "type": "gauge", "name": "custom"}
            ],
        }
    )
    rules = _load_rollout_rules(rollout_conf)
    assert rules is not None
    sources = _sources(rules)
    assert "eng:custom" in sources
    assert "vllm:num_requests_running" in sources
    assert "tq_controller_request_total" in sources


def test_load_rollout_rules_inline_only_when_no_packs() -> None:
    rollout_conf = OmegaConf.create(
        {"metrics": [{"source": "eng:x", "type": "gauge", "name": "x"}]}
    )
    rules = _load_rollout_rules(rollout_conf)
    assert rules is not None
    assert _sources(rules) == ["eng:x"]


def test_load_rollout_rules_skips_unknown_pack_and_keeps_inline() -> None:
    rollout_conf = OmegaConf.create(
        {
            "packs": ["does-not-exist"],
            "metrics": [{"source": "eng:x", "type": "gauge", "name": "x"}],
        }
    )
    rules = _load_rollout_rules(rollout_conf)
    assert rules is not None
    assert _sources(rules) == ["eng:x"]


def test_load_rollout_rules_empty_returns_none() -> None:
    assert _load_rollout_rules(OmegaConf.create({})) is None
