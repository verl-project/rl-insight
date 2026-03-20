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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Optional, Type
from rl_insight.data.enums import DataEnum

REGISTERED_DATA_ENUM: dict["DataEnum", type["BaseData"]] = {}


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
    def check(self, data: "BaseData") -> bool:
        pass

    @property
    @abstractmethod
    def error_message(self) -> List[str]:
        pass


class BaseData(ABC):
    """Base data class for RL-Insight."""

    _rules: List[ValidationRule] = []
    """Validation rules for this data class. Should be set by subclasses."""
    _data_type: DataEnum
    """Data type for this data class. Should be set by subclasses."""

    @classmethod
    def check(cls, data_type: DataEnum | list[DataEnum], data: "BaseData") -> bool:
        """Validate the data"""

        if isinstance(data_type, list):
            if cls._data_type not in data_type:
                raise DataValidationError(
                    f"Data type mismatch: expected one of {data_type}, got {cls._data_type}"
                )
        else:
            if data_type != cls._data_type:
                raise DataValidationError(
                    f"Data type mismatch: expected {cls._data_type}, got {data_type}"
                )

        errors: List[List[str]] = []
        for rule in cls._rules:
            if not rule.check(data):
                errors.append(rule.error_message)
        if errors:
            raise DataValidationError("Data validation failed", errors)
        return True

    @property
    def data_type(self) -> DataEnum:
        return self._data_type

    @abstractmethod
    def load(self):
        """Load the data from source. Should be implemented by subclasses."""
        pass


def register_data_cls() -> Callable[[type[BaseData]], type[BaseData]]:
    def decorator(cls: type[BaseData]) -> type[BaseData]:
        enum = cls._data_type
        if enum in REGISTERED_DATA_ENUM:
            raise ValueError(
                f"Data enum {enum} already registered for {REGISTERED_DATA_ENUM[enum]}"
            )
        REGISTERED_DATA_ENUM[enum] = cls
        return cls

    return decorator


def get_data_cls(data_enum: DataEnum) -> Type[BaseData]:
    if data_enum not in REGISTERED_DATA_ENUM:
        raise ValueError(
            f"Unsupported data enum: {data_enum}. Supported enums are: {list(REGISTERED_DATA_ENUM.keys())}"
        )
    return REGISTERED_DATA_ENUM[data_enum]
