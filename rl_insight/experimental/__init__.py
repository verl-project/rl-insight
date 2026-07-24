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

"""Experiment trajectory pipeline (PR#120).

Imports are lazy so the py3.9 ``rl-insight-server`` can load Agent Loop Rebuild
helpers without evaluating SampleRecord type annotations.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseSample",
    "TrajectoryBuilder",
    "FileSampleRecord",
    "SampleRecord",
    "SampleTag",
    "SessionRecord",
    "SessionTag",
    "Step",
    "ToolResult",
    "ToolStatus",
    "TrajectoryRecord",
    "TrajectoryTag",
    "TrainingStatus",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BaseSample": ("rl_insight.experimental.samples.base", "BaseSample"),
    "FileSampleRecord": (
        "rl_insight.experimental.samples.file_sample",
        "FileSampleRecord",
    ),
    "SampleRecord": ("rl_insight.experimental.samples.sample", "SampleRecord"),
    "SampleTag": ("rl_insight.experimental.samples.sample", "SampleTag"),
    "SessionRecord": ("rl_insight.experimental.samples.sample", "SessionRecord"),
    "SessionTag": ("rl_insight.experimental.samples.sample", "SessionTag"),
    "Step": ("rl_insight.experimental.samples.sample", "Step"),
    "ToolResult": ("rl_insight.experimental.samples.sample", "ToolResult"),
    "ToolStatus": ("rl_insight.experimental.samples.sample", "ToolStatus"),
    "TrajectoryRecord": (
        "rl_insight.experimental.samples.sample",
        "TrajectoryRecord",
    ),
    "TrajectoryTag": ("rl_insight.experimental.samples.sample", "TrajectoryTag"),
    "TrainingStatus": ("rl_insight.experimental.samples.sample", "TrainingStatus"),
    "TrajectoryBuilder": ("rl_insight.experimental.builder", "TrajectoryBuilder"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    mod = importlib.import_module(module_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
