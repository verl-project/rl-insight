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

from types import SimpleNamespace

from rl_insight.experimental.prom_export import traj_reward, traj_success


def test_traj_reward_uses_upstream_value():
    assert traj_reward(SimpleNamespace(reward_score=0.25)) == 0.25


def test_traj_reward_does_not_invent_value_when_missing():
    assert traj_reward(SimpleNamespace(reward_score=None)) is None


def test_traj_success_matches_upstream_reward_semantics():
    assert traj_success(SimpleNamespace(reward_score=1.0)) is True
    assert traj_success(SimpleNamespace(reward_score=0.0)) is False
    assert traj_success(SimpleNamespace(reward_score=-1.0)) is False
    assert traj_success(SimpleNamespace(reward_score=None)) is False
