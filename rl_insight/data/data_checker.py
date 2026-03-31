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

"""Base data definitions for RL-Insight."""

from typing import Any, List

from enum import Enum

from loguru import logger

from .rules import (
    DataValidationError,
    ParserOutputValidatorRule,
    PathExistsRule,
    ValidationRule,
)
from .verl_log_rules import VerlLogExistRule, VerlLogKeyParamsRule


class DataEnum(Enum):
    """Enum for data types in RL-Insight."""

    # input data type of parser
    MULTI_JSON = "multi_json"
    VERL_LOG = "verl_log"
    # output data type of parser, input data type of visualizer
    SUMMARY_EVENT = "summary_event"
    # other data type
    UNKNOWN = "unknown"


class DataChecker:
    """Base data class for RL-Insight."""

    rules: dict[DataEnum, List[ValidationRule]] = {
        DataEnum.MULTI_JSON: [PathExistsRule()],
        DataEnum.VERL_LOG: [VerlLogExistRule(), VerlLogKeyParamsRule()],
        DataEnum.SUMMARY_EVENT: [
            ParserOutputValidatorRule(
                domains=["role", "name", "rank_id", "start_time_ms", "end_time_ms"]
            )
        ],
        DataEnum.UNKNOWN: [],
    }

    def __init__(self, data_type: DataEnum, data: Any):
        self.data_type = data_type
        self.data = data

    def run(self):
        """Validate the data"""
        errors = []
        if self.data_type not in self.rules:
            raise ValueError(f"Invalid data type: {self.data_type}")
        rules = self.rules[self.data_type]
        for rule in rules:
            if not rule.check(self.data):
                errors.append(rule.error_message)
        if errors:
            raise DataValidationError("Data validation failed", errors)
        logger.info(f"Data validation passed for {self.data_type}")
