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
from pathlib import Path
import pandas as pd
import os

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]


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


def test_data_checker_summary_event_success_with_valid_data_type():
    json_path = (
        PROJECT_ROOT
        / "data"
        / "summary_event_data"
        / "summary_event_dataframe_sample.json"
    )

    # Verify that the file exists (test precondition)
    assert os.path.exists(json_path), f"sample file {json_path} does not exist"

    df = pd.read_json(json_path, orient="records")
    checker = DataChecker(data_type=DataEnum.SUMMARY_EVENT, data=df)
    checker.run()


def test_data_checker_summary_event_fails_with_invalid_data_type():
    checker = DataChecker(data_type=DataEnum.SUMMARY_EVENT, data={"k": "v"})
    with pytest.raises(DataValidationError) as exc_info:
        checker.run()
    assert "Data validation failed" in str(exc_info.value)


def test_data_checker_summary_event_fails_with_empty_data():
    checker = DataChecker(data_type=DataEnum.SUMMARY_EVENT, data={})
    with pytest.raises(DataValidationError) as exc_info:
        checker.run()
    assert "Data validation failed" in str(exc_info.value)


def test_summary_event_raises_error_when_missing_required_columns():
    """
    Test that the validation rule raises ValueError when SUMMARY_EVENT DataFrame is missing required columns.
    """
    # Create an invalid DataFrame missing mandatory columns (domain, end_time_ms)
    invalid_df = pd.DataFrame(
        {
            "start_time_ms": [1773285888698.7263183594],
            "name": "agent_loop_rollout_replica_0",
            "end_time_ms": 1773285890928.7919921875,
            "rank_id": 1,
        }
    )

    checker = DataChecker(data_type=DataEnum.SUMMARY_EVENT, data=invalid_df)
    with pytest.raises(DataValidationError) as exc_info:
        checker.run()
    assert "Data validation failed" in str(exc_info.value)
