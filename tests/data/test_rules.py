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

from rl_insight.data.rules import DataValidationError, PathExistsRule


def test_path_exists_rule_accepts_existing_directory(tmp_path):
    rule = PathExistsRule()
    assert rule.check(str(tmp_path)) is True


def test_path_exists_rule_rejects_non_string_input():
    rule = PathExistsRule()
    assert rule.check({"path": "x"}) is False
    assert "not a path" in rule.error_message


def test_path_exists_rule_rejects_missing_directory():
    rule = PathExistsRule()
    assert rule.check("C:/definitely/not/exist/path") is False


def test_data_validation_error_string_includes_error_details():
    err = DataValidationError("Data validation failed", ["line1", "line2"])
    text = str(err)
    assert "Data validation failed" in text
    assert "line1" in text
    assert "line2" in text
