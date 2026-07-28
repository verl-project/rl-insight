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

"""Normalization interface for degradation perception."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def normalize_data(
    values: Sequence[float], config: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Normalize values without changing their business scale.

    Only identity normalization is currently specified. The explicit interface
    prevents an unsupported Z-score substitution and provides a local extension
    point when an approved formula becomes available.
    """

    mode = str((config or {}).get("type", "identity")).lower()
    if mode != "identity":
        raise ValueError(f"Unsupported normalization type: {mode!r}")
    return np.asarray(values, dtype=float).copy()
