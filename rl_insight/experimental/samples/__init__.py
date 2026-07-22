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

"""Sample implementations for trajectory data storage.

- ``BaseSample``: Protocol defining the six-method CRUD interface.
- ``SampleRecord``: In-memory Pydantic model.
- ``FileSampleRecord``: Filesystem-backed, one JSON per trajectory.
"""

from rl_insight.experimental.samples.base import BaseSample
from rl_insight.experimental.samples.file_sample import FileSampleRecord
from rl_insight.experimental.samples.sample import (
    SampleRecord,
    SampleTag,
    SessionRecord,
    SessionTag,
    Step,
    ToolResult,
    ToolStatus,
    TrajectoryRecord,
    TrajectoryTag,
    TrainingStatus,
)

__all__ = [
    "BaseSample",
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
