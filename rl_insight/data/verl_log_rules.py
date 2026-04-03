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

"""Validation rules for a single VeRL training log file (.log)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .rules import ValidationRule

_READ_CHUNK_BYTES = 64 * 1024
_MAX_READ_FOR_KEYWORDS = 2 * 1024 * 1024
_VERL_NAME_HINT = "verl"


def _parse_log_path(data: object) -> Optional[Path]:
    if not isinstance(data, str):
        return None
    try:
        return Path(data)
    except TypeError:
        return None


def _validate_verl_log_file(
    data: object,
) -> tuple[Optional[Path], Optional[str]]:
    """Shared path check for VeRL log rules: exist, is file, non-empty .log, VeRL-identifiable."""
    root = _parse_log_path(data)
    if root is None:
        return None, "Data object is not a path string"
    if not root.exists():
        return None, f"VeRL log path does not exist: {data}"
    if not root.is_file():
        return (
            None,
            f"VeRL log path must be a single .log file, not a directory: {data}",
        )
    if root.suffix.lower() != ".log":
        return None, f"VeRL log path must be a .log file: {data}"
    try:
        if root.stat().st_size <= 0:
            return None, f"VeRL log file is empty: {data}"
    except OSError as e:
        return None, f"Cannot read VeRL log file {data}: {e}"

    if not _looks_like_verl_log(root):
        return (
            None,
            "Log file is not identified as VeRL "
            "(expect filename containing 'verl' or log text mentioning 'verl')",
        )
    return root, None


def _looks_like_verl_log(path: Path) -> bool:
    if _VERL_NAME_HINT in path.name.lower():
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(_READ_CHUNK_BYTES)
        text = chunk.decode("utf-8", errors="ignore").lower()
        return _VERL_NAME_HINT in text
    except OSError:
        return False


def _read_log_for_keywords(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(_MAX_READ_FOR_KEYWORDS)
    except OSError:
        return ""


class VerlLogExistRule(ValidationRule):
    """Ensure ``data`` is a path to one non-empty VeRL-identifiable .log file."""

    def check(self, data: str) -> bool:
        _, err = _validate_verl_log_file(data)
        if err is not None:
            self._error_message = err
            return False
        return True


class VerlLogKeyParamsRule(ValidationRule):
    """Ensure the VeRL log text contains required metric / config markers."""

    # Common VeRL training log substrings (tensorboard-style names); adjust per recipe.
    DEFAULT_REQUIRED_KEYWORDS: Tuple[str, ...] = (
        "verl",
        "actor/loss",
        "critic/score/mean",
        "critic/rewards/mean",
        "response_length/mean",
        "actor/grad_norm",
        "training/global_step",
        "training/epoch",
        "actor/lr",
        "actor/entropy",
        "Training Progress:",
    )

    def __init__(
        self,
        required_keywords: Tuple[str, ...] | None = None,
    ):
        super().__init__()
        self._required_keywords = (
            required_keywords
            if required_keywords is not None
            else self.DEFAULT_REQUIRED_KEYWORDS
        )

    def check(self, data: str) -> bool:
        path, err = _validate_verl_log_file(data)
        if err is not None or path is None:
            self._error_message = err or "Invalid VeRL log path"
            return False

        blob = _read_log_for_keywords(path).lower()
        if not blob:
            self._error_message = (
                f"No readable text in VeRL log for keyword check: {data}"
            )
            return False

        missing: List[str] = [
            kw for kw in self._required_keywords if kw.lower() not in blob
        ]
        if missing:
            self._error_message = (
                "VeRL log is missing required parameter markers: " + ", ".join(missing)
            )
            return False
        return True
