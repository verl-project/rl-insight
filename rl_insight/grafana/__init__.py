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

"""Generic Grafana dashboard rendering shipped with the ``rl_insight`` package.

The public API lives in :mod:`rl_insight.grafana.renderer`; this package only
re-exports it so callers can ``from rl_insight.grafana import render_dashboards``.
"""

from rl_insight.grafana.renderer import (
    FRAMEWORK_DIR,
    JsonnetRenderError,
    generated_text,
    materialize_dashboards,
    render_dashboards,
    stale_dashboards,
)

__all__ = [
    "FRAMEWORK_DIR",
    "JsonnetRenderError",
    "generated_text",
    "materialize_dashboards",
    "render_dashboards",
    "stale_dashboards",
]
