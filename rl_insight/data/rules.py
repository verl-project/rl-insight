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

from typing import Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path


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


# Markers that must appear in VeRL training logs (Hydra/OmegaConf dumps, etc.).
# Adjust this tuple when your pipeline requires different keys.
VERL_REQUIRED_LOG_KEYWORDS: Tuple[str, ...] = (
    "critic/score/mean",
    "actor/loss",
    "response_length/mean",
    "actor/grad_norm",
    "critic/rewards/mean",
)

_VERL_NAME_HINT = "verl"
_READ_CHUNK_BYTES = 64 * 1024
_MAX_READ_PER_FILE = 2 * 1024 * 1024
_MAX_LOG_FILES = 32


def _path_str(data: Any) -> Optional[Path]:
    if not isinstance(data, str):
        return None
    try:
        return Path(data)
    except TypeError:
        return None


def _non_empty_log_files(root: Path) -> List[Path]:
    """Collect *.log files under root (or root itself if it is a .log file)."""
    if root.is_file():
        if root.suffix.lower() == ".log" and root.stat().st_size > 0:
            return [root]
        return []
    if not root.is_dir():
        return []
    found: List[Path] = []
    for p in root.rglob("*.log"):
        try:
            if p.is_file() and p.stat().st_size > 0:
                found.append(p)
        except OSError:
            continue
    return found


def _looks_like_verl_log(path: Path) -> bool:
    """Filename contains 'verl' or file body mentions VeRL in the first chunk."""
    if _VERL_NAME_HINT in path.name.lower():
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(_READ_CHUNK_BYTES)
        text = chunk.decode("utf-8", errors="ignore").lower()
        return _VERL_NAME_HINT in text
    except OSError:
        return False


def _verl_log_files_for_validation(root: Path) -> List[Path]:
    """Non-empty .log files that are considered VeRL logs for presence / keyword checks."""
    candidates = _non_empty_log_files(root)
    verl_specific = [p for p in candidates if _looks_like_verl_log(p)]
    return verl_specific if verl_specific else candidates


def _read_logs_for_keywords(paths: List[Path]) -> str:
    parts: List[str] = []
    for p in paths[:_MAX_LOG_FILES]:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f.read(_MAX_READ_PER_FILE))
        except OSError:
            continue
    return "\n".join(parts)


class VerlLogPresentRule(ValidationRule):
    """Ensure the path points to VeRL log data: a non-empty .log file or directory containing one."""

    def check(self, data: Any) -> bool:
        root = _path_str(data)
        if root is None:
            self._error_message = "Data object is not a path string"
            return False
        if not root.exists():
            self._error_message = f"VeRL log path does not exist: {data}"
            return False

        candidates = _non_empty_log_files(root)
        if not candidates:
            self._error_message = (
                f"No non-empty *.log file found under VeRL log path: {data}"
            )
            return False

        verl_logs = [p for p in candidates if _looks_like_verl_log(p)]
        if not verl_logs:
            self._error_message = (
                "Found .log file(s) but none identified as VeRL logs "
                "(expect filename containing 'verl' or log text mentioning 'verl')"
            )
            return False
        return True


class VerlLogKeyParamsRule(ValidationRule):
    """Ensure concatenated VeRL log text contains required configuration keywords."""

    def __init__(self, required_keywords: Tuple[str, ...] = VERL_REQUIRED_LOG_KEYWORDS):
        super().__init__()
        self._required_keywords = required_keywords

    def check(self, data: Any) -> bool:
        root = _path_str(data)
        if root is None:
            self._error_message = "Data object is not a path string"
            return False
        if not root.exists():
            self._error_message = f"VeRL log path does not exist: {data}"
            return False

        log_paths = _verl_log_files_for_validation(root)
        if not log_paths:
            self._error_message = (
                f"No readable non-empty *.log files for keyword check: {data}"
            )
            return False

        blob = _read_logs_for_keywords(log_paths).lower()
        missing = [kw for kw in self._required_keywords if kw.lower() not in blob]
        if missing:
            self._error_message = (
                "VeRL log is missing required parameter markers: "
                + ", ".join(missing)
            )
            return False
        return True
