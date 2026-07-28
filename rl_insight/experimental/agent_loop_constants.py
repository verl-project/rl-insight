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

"""Shared Agent Loop constants (no SampleRecord imports — safe for py3.9 server)."""

SERVICE_NAME_VALUE = "agent-loop-poc"

# Dedicated Agent Loop dashboard (do NOT mutate verl_trainer_v1_with_sglang_engine).
GRAFANA_DASHBOARD_FILE = "agent_loop_trajectory.json"
GRAFANA_DASHBOARD_UID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
GRAFANA_DASHBOARD_SLUG = "agent-loop-trajectory"
GRAFANA_DASHBOARD_TITLE = "agent_loop_trajectory"

DEFAULT_GRAFANA_BASE = "http://127.0.0.1:3000"
