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

"""Unit tests for environment-variable tag derivation."""

from __future__ import annotations

from omegaconf import OmegaConf

from rl_insight.client.push.tags import collect_env_tags


def _rules(raw):
    return OmegaConf.create(raw)


def test_collect_env_tags_should_use_first_nonempty_env_in_fallback_list() -> None:
    rules = _rules([{"tag": "idx", "env": ["A", "B", "C"]}])
    tags = collect_env_tags(rules, environ={"B": "7", "C": "9"})
    assert tags == {"idx": "7"}


def test_collect_env_tags_should_accept_single_env_string() -> None:
    rules = _rules([{"tag": "trial", "env": "TRIAL"}])
    assert collect_env_tags(rules, environ={"TRIAL": "t-1"}) == {"trial": "t-1"}


def test_collect_env_tags_should_skip_when_unset_without_default() -> None:
    rules = _rules([{"tag": "role", "env": ["ROLE"]}])
    assert collect_env_tags(rules, environ={}) == {}


def test_collect_env_tags_should_treat_empty_string_as_unset() -> None:
    rules = _rules([{"tag": "role", "env": ["ROLE"], "default": "0"}])
    assert collect_env_tags(rules, environ={"ROLE": ""}) == {"role": "0"}


def test_collect_env_tags_should_apply_default_and_format_template() -> None:
    rules = _rules(
        [
            {
                "tag": "xgpt_psm",
                "env": ["PSM"],
                "default": "none",
                "format": "inf.ray.serve_{value}",
            }
        ]
    )
    assert collect_env_tags(rules, environ={}) == {"xgpt_psm": "inf.ray.serve_none"}
    assert collect_env_tags(rules, environ={"PSM": "foo"}) == {
        "xgpt_psm": "inf.ray.serve_foo"
    }


def test_collect_env_tags_should_ignore_rules_without_tag_name() -> None:
    rules = _rules([{"env": ["ROLE"]}])
    assert collect_env_tags(rules, environ={"ROLE": "worker"}) == {}


def test_collect_env_tags_should_return_empty_for_none_rules() -> None:
    assert collect_env_tags(None) == {}
