# Copyright (c) 2025 verl-project authors.
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

"""Integration tests for data module."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from rl_insight.data.base import BaseData, register_data_cls, get_data_cls
from rl_insight.data.enums import DataEnum
from rl_insight.data.rules import PathExistsRule


@register_data_cls()
@dataclass
class SampleData(BaseData):
    path: Path

    _data_type = DataEnum.UNKNOWN
    _rules = [PathExistsRule()]
    _content: str = ""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def load(self):
        with open(self.path, "r") as f:
            self._content = f.read()

    @property
    def content(self):
        if not self._content:
            self.load()
        return self._content


@pytest.fixture
def sample_data(tmp_path):
    # Create a temporary file for testing
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("This is a test file.")
    return SampleData(path=test_file)


class TestSampleData:
    def test_registration(self):
        data_cls = get_data_cls(DataEnum.UNKNOWN)
        assert data_cls is SampleData

    def test_path_not_exists(self, tmp_path=Path("/tmp")):
        data = SampleData(path=tmp_path / "non_existent_file.txt")
        try:
            SampleData.check(data_type=DataEnum.UNKNOWN, data=data)
        except Exception as e:
            assert isinstance(e, Exception)
            assert "Data validation failed" in str(e)
            assert "path does not exist" in str(e)

    def test_enum_mismatch(self, sample_data):
        try:
            SampleData.check(data_type=DataEnum.MULTI_JSON, data=sample_data)
        except Exception as e:
            assert isinstance(e, Exception)
            assert "Data type mismatch" in str(e)

    def test_valid_data(self, sample_data):
        try:
            SampleData.check(data_type=DataEnum.UNKNOWN, data=sample_data)
            assert sample_data.content == "This is a test file."
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
