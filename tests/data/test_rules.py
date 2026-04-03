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
from rl_insight.data.verl_log_rules import VerlLogExistRule, VerlLogKeyParamsRule


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


def test_verl_log_exist_accepts_verl_named_log(tmp_path):
    log = tmp_path / "train_verl_worker.log"
    log.write_text("verl job\n", encoding="utf-8")
    rule = VerlLogExistRule()
    assert rule.check(str(log)) is True


def test_verl_log_exist_rejects_directory(tmp_path):
    rule = VerlLogExistRule()
    assert rule.check(str(tmp_path)) is False
    assert "not a directory" in rule.error_message


def test_verl_log_exist_rejects_non_log_extension(tmp_path):
    log = tmp_path / "run_verl.txt"
    log.write_text("verl job\n", encoding="utf-8")
    rule = VerlLogExistRule()
    assert rule.check(str(log)) is False


def test_verl_log_key_params_requires_keywords(tmp_path):
    log = tmp_path / "debug_verl.log"
    log.write_text(
        "\n".join(
            [
                "python3 -m verl.trainer.main_ppo",
                "Training Progress:   0%|          | 1/100 [00:01<00:00,  1.00s/it]",
                "(TaskRunner pid=1) step=0 - training/global_step:1 - training/epoch:0",
                "(TaskRunner pid=1) 'critic/score/mean': 0.1",
                "(TaskRunner pid=1) 'actor/loss': 0.2",
                "(TaskRunner pid=1) 'critic/rewards/mean': 0.3",
                "(TaskRunner pid=1) 'response_length/mean': 128.0",
                "(TaskRunner pid=1) 'actor/grad_norm': 0.99",
                "(TaskRunner pid=1) 'actor/lr': 1e-06",
                "(TaskRunner pid=1) 'actor/entropy': 0.5",
            ]
        ),
        encoding="utf-8",
    )
    rule = VerlLogKeyParamsRule(
        required_keywords=VerlLogKeyParamsRule.DEFAULT_REQUIRED_KEYWORDS
    )
    assert rule.check(str(log)) is True


def test_verl_log_key_params_fails_when_missing_keyword(tmp_path):
    log = tmp_path / "x_verl.log"
    log.write_text(
        "\n".join(
            [
                "python3 -m verl.trainer.main_ppo",
                "(TaskRunner pid=1) 'critic/score/mean': 0.1",
                "(TaskRunner pid=1) 'actor/loss': 0.2",
            ]
        ),
        encoding="utf-8",
    )
    rule = VerlLogKeyParamsRule(
        required_keywords=VerlLogKeyParamsRule.DEFAULT_REQUIRED_KEYWORDS
    )
    assert rule.check(str(log)) is False
    assert "response_length/mean" in rule.error_message
