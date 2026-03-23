import pytest

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
    err = DataValidationError("Data validation failed", [["line1", "line2"]])
    text = str(err)
    assert "Data validation failed" in text
    assert "line1" in text
    assert "line2" in text
