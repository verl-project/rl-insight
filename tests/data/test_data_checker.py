import pytest

from rl_insight.data.data_checker import DataChecker, DataEnum
from rl_insight.data.rules import DataValidationError


def test_data_checker_multi_json_path_exists(tmp_path):
    checker = DataChecker(type=DataEnum.MULTI_JSON, data=str(tmp_path))
    checker.run()


def test_data_checker_multi_json_path_missing():
    checker = DataChecker(type=DataEnum.MULTI_JSON, data="C:/definitely/not/exist/path")
    with pytest.raises(DataValidationError) as exc_info:
        checker.run()
    assert "Data validation failed" in str(exc_info.value)


def test_data_checker_summary_event_has_no_rule_with_dict_data():
    checker = DataChecker(type=DataEnum.SUMMARY_EVENT, data={"k": "v"})
    checker.run()
