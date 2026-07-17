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

"""Online monitoring APIs.

Public symbols are loaded lazily so ``rl-insight server ...`` can start without
importing trainer-side optional dependencies such as Ray or OpenTelemetry.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "finish": ".api",
    "init": ".api",
    "metric_count": ".api",
    "metric_gauge": ".api",
    "metric_histogram": ".api",
    "trace_op": ".api",
    "trace_state": ".api",
    "update_prometheus_config": ".utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
