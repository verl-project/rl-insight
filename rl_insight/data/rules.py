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

from typing import List, Any
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd


class DataValidationError(Exception):
    """Exception raised when data validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []

    def __str__(self) -> str:
        if self.errors:
            return f"{super().__str__()}\n  - " + "\n  - ".join(self.errors)
        return super().__str__()


class ValidationRule(ABC):
    """Validation rule base class"""

    def __init__(self):
        self._error_message: str = ""

    @abstractmethod
    def check(self, data) -> bool:
        pass

    @property
    def error_message(self) -> str:
        return self._error_message


class PathExistsRule(ValidationRule):
    def check(self, data: Any) -> bool:
        if not isinstance(data, str):
            self._error_message = "Data object is not a path"
            return False
        try:
            path = Path(data)
            if not path.is_dir():
                self._error_message = (
                    f"Source path is not a directory or does not exist: {data}"
                )
                return False
            return True
        except TypeError as e:
            self._error_message = f"Error checking path {data}: {e}"
            return False


class ParserOutputValidatorRule(ValidationRule):
    def __init__(self, domains: List[str]):
        super().__init__()
        self.domains = set(domains)

    def check(self, data: Any) -> bool:
        """
        Parser output key information validator
        Only verify whether the key fields are included and the data is not empty
        """

        # 1. Check if it's a DataFrame
        if not isinstance(data, pd.DataFrame):
            self._error_message = f"Parsing result must be a DataFrame, got {type(data).__name__} instead."
            return False

        # 2. Check if data is not empty
        if data.empty:
            self._error_message = (
                "Parsing result validation failed: The DataFrame is empty."
            )
            return False

        # 3. Check if all key columns exist
        missing_cols = self.domains - set(data.columns)
        if missing_cols:
            # Sort for consistent error messages
            self._error_message = f"Parsing result validation failed: Missing key columns - {sorted(list(missing_cols))}"
            return False
        return True
