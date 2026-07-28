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

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum

import numpy as np
import pytest

from experiment.degradation_perception.serialization import to_json_serializable


class TextState(Enum):
    READY = "ready"


class NumericState(IntEnum):
    READY = 1


@dataclass
class Payload:
    scalar: object
    values: object


def test_recursive_serialization_converts_all_required_types():
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = {
        np.int64(7): Payload(
            scalar=np.float32(1.5),
            values=(
                np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                {TextState.READY, "other"},
            ),
        ),
        "when": instant,
        "bool": np.bool_(True),
        "state": NumericState.READY,
    }
    converted = to_json_serializable(value)
    assert converted["7"]["scalar"] == pytest.approx(1.5)
    assert converted["7"]["values"][0] == [[1, 2], [3, 4]]
    assert sorted(converted["7"]["values"][1]) == ["other", "ready"]
    assert converted["when"] == instant.isoformat()
    assert converted["bool"] is True
    assert converted["state"] == 1
    assert type(converted["state"]) is int
    json.dumps(converted, allow_nan=False)


def test_zero_dimensional_ndarray_is_converted_to_scalar():
    converted = to_json_serializable(np.asarray(3))
    assert converted == 3
    assert type(converted) is int


def test_non_string_mapping_keys_become_strings():
    assert to_json_serializable({(1, 2): "value"}) == {"[1, 2]": "value"}


def test_frozenset_is_a_deterministic_list():
    assert to_json_serializable(frozenset({3, 1, 2})) == [1, 2, 3]


def test_unsupported_value_raises_instead_of_stringifying_silently():
    with pytest.raises(TypeError, match="Unsupported JSON value type"):
        to_json_serializable(object())

