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

"""Recursive conversion to JSON-native values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np


def to_json_serializable(value: Any) -> Any:
    """Recursively convert supported Python and NumPy values to JSON-native types."""

    # NumPy floating scalars and IntEnum can also satisfy native isinstance
    # checks on some Python/NumPy versions, so handle all wrappers first.
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_json_serializable(value.tolist())
    if isinstance(value, np.generic):
        return to_json_serializable(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_json_serializable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_serializable(asdict(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(to_json_serializable(key)): to_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_json_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_json_serializable(item) for item in value]
        return sorted(converted, key=repr)
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")
