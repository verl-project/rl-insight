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

from typing import List
from abc import ABC, abstractmethod 
from pathlib import Path
from typing import Optional
import pandas as pd



class DataValidationError(Exception):
    """Exception raised when data validation fails."""

    def __init__(self, message: str, errors: Optional[List[List[str]]] = None):
        super().__init__(message)
        self.errors = errors or []

    def __str__(self) -> str:
        if self.errors:
            return f"{super().__str__()}\n  - " + "\n  - ".join(
                ["\n    ".join(err) for err in self.errors]
            )
        return super().__str__()


class ValidationRule(ABC):
    """Validation rule base class"""

    @abstractmethod
    def check(self, data) -> bool:
        pass

    @property
    @abstractmethod
    def error_message(self) -> List[str]:
        pass


class PathExistsRule(ValidationRule):
    _error_message: str
    def check(self, data: str|dict|pd.DataFrame) -> bool:
        if not isinstance(data, str):
            self._error_message = "Data object is not a path"
            return False
        try:
            path = Path(data)
            if not path.is_dir():
                self._error_message = f"Source path does not exist: {data}"
                return False
            return True
        except Exception as e:
            self._error_message = f"Source path does not exist: {data}"
            return False

    @property
    def error_message(self) -> List[str]:
        return self._error_message
