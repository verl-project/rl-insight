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

import pytest

from rl_insight.data.data_checker import DataChecker, DataEnum
from rl_insight.data.rules import DataValidationError


def test_data_checker_multi_json_path_exists(tmp_path):
    checker = DataChecker(data_type=DataEnum.MULTI_JSON, data=str(tmp_path))
    checker.run()


def test_data_checker_multi_json_path_missing():
    checker = DataChecker(
        data_type=DataEnum.MULTI_JSON, data="C:/definitely/not/exist/path"
    )
    with pytest.raises(DataValidationError) as exc_info:
        checker.run()
    assert "Data validation failed" in str(exc_info.value)


def test_data_checker_summary_event_has_no_rule_with_dict_data():
    checker = DataChecker(data_type=DataEnum.SUMMARY_EVENT, data={"k": "v"})
    checker.run()
