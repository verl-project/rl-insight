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


from typing import List
from .rules import ValidationRule, PathExistsRule, DataValidationError
from enum import Enum
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataEnum(Enum):
    """Enum for data types in RL-Insight."""
    # input data type of parser
    MULTI_JSON = "multi_json"
    VERL_LOG = "verl_log"
    # output data type of parser, input data type of visualizer
    SUMMARY_EVENT = "summary_event"
    # other data type
    UNKNOWN = "unknown"


class DataChecker():
    """Base data class for RL-Insight."""
    rules: dict[DataEnum, List[ValidationRule]] = {
        DataEnum.MULTI_JSON: [PathExistsRule()],
        DataEnum.SUMMARY_EVENT: [],
    }

    def __init__(self, type: DataEnum, data: str|dict):
        self.type = type
        self.data = data

    def run(self):
        """Validate the data"""
        errors = []
        rules = self.rules[self.type]
        for rule in rules:
            if not rule.check(self.data):
                errors.append(rule.error_message)
        if errors:
            raise DataValidationError("Data validation failed", errors)
        logger.info(f"Data validation passed for {self.type}")
