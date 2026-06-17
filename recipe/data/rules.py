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

import gzip
import json
from typing import Any, List, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


def _coerce_path(data: Any) -> Optional[Path]:
    if isinstance(data, Path):
        return data
    if isinstance(data, str):
        return Path(data)
    return None


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
        path = _coerce_path(data)
        if path is None:
            self._error_message = "Data object is not a path"
            return False
        try:
            if not path.is_dir():
                self._error_message = (
                    f"Source path is not a directory or does not exist: {path}"
                )
                return False
            return True
        except TypeError as e:
            self._error_message = f"Error checking path {path}: {e}"
            return False


class MstxJsonFileExistsRule(ValidationRule):
    """valid Mstx trace_view.json and profiler_info_*.json files is existed in "ASCEND_PROFILER_OUTPUT" path"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False

            ascend_profiler_output = "ASCEND_PROFILER_OUTPUT"
            trace_view_filename = "trace_view.json"
            profiler_info_filename = "profiler_info_*.json"

            # get all *_ascend_pt path
            ascend_pt_folders = list(root_path.glob("*/*_ascend_pt"))

            if not ascend_pt_folders:
                self._error_message = f"No *_ascend_pt path in {root_path}"
                return False

            for ascend_pt_path in ascend_pt_folders:
                if not ascend_pt_path.is_dir():
                    continue

                # get trace_view.json file path
                trace_view_path = (
                    ascend_pt_path / ascend_profiler_output / trace_view_filename
                )
                if not trace_view_path.exists():
                    self._error_message = f"trace_view.json does not exist in: {ascend_pt_path}/ASCEND_PROFILER_OUTPUT"
                    return False

                # get profiler_info_*.json file path
                profiler_files = list(ascend_pt_path.glob(profiler_info_filename))

                if not profiler_files:
                    self._error_message = (
                        f"profiler_info_*.json does not exist in: {ascend_pt_path}"
                    )
                    return False
            return True
        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


class MstxJsonFieldValidRule(ValidationRule):
    """valid Mstx trace_view.json and profiler_info_*.json files JSON format"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False

            # get all *_ascend_pt path
            ascend_pt_folders = list(root_path.glob("*/*_ascend_pt"))

            if not ascend_pt_folders:
                self._error_message = f"No *_ascend_pt path in {root_path}"
                return False

            for ascend_pt_path in ascend_pt_folders:
                # valid trace_view.json format
                trace_view_path = (
                    ascend_pt_path / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
                )
                if not trace_view_path.exists():
                    self._error_message = (
                        f"Missing trace_view.json in: {trace_view_path.parent}"
                    )
                    return False
                if trace_view_path.stat().st_size == 0:
                    self._error_message = f"File is empty: {trace_view_path}"
                    return False
                try:
                    with open(trace_view_path, "r", encoding="utf-8") as f:
                        trace_view_data = json.load(f)
                except Exception as exc:
                    self._error_message = (
                        f"Failed to parse JSON file {trace_view_path}: {exc}"
                    )
                    return False

                if len(trace_view_data) == 0:
                    self._error_message = f"File is empty: {trace_view_path}"
                    return False

                required_keys = {"ph", "name", "pid", "tid"}
                for row in trace_view_data:
                    missing_keys = required_keys - row.keys()
                    if missing_keys:
                        self._error_message = f"File field is missing: {missing_keys} in FilePath: {trace_view_path}"
                        return False

                # valid profiler_info_*.json format
                profiler_info_files = list(ascend_pt_path.glob("profiler_info_*.json"))
                if not profiler_info_files:
                    self._error_message = (
                        f"profiler_info_*.json does not exist in: {ascend_pt_path}"
                    )
                    return False
                for file_path in profiler_info_files:
                    if file_path.stat().st_size == 0:
                        self._error_message = f"File is empty: {file_path}"
                        return False
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            profiler_info_data = json.load(f)
                    except Exception as exc:
                        self._error_message = (
                            f"Failed to parse JSON file {file_path}: {exc}"
                        )
                        return False
                    if len(profiler_info_data) == 0:
                        self._error_message = f"File is empty: {file_path}"
                        return False
                    required_keys = {
                        "config",
                        "start_info",
                        "end_info",
                        "torch_npu_version",
                        "cann_version",
                        "rank_id",
                    }
                    missing_keys = required_keys - set(profiler_info_data.keys())
                    if missing_keys:
                        self._error_message = f"File field is missing: {missing_keys} in FilePath: {file_path}"
                        return False
            return True
        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


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
            self._error_message = (
                "Parsing result validation failed: Missing key columns - "
                f"{sorted(list(missing_cols))}"
            )
            return False
        return True


class TorchJsonFileExistsRule(ValidationRule):
    """valid Torch *.json.gz files is existed in 'torch_profile' sub path"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            is_success = True
            sub_dirs_no_json: List = []

            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False
            for subdir in root_path.iterdir():
                if subdir.is_dir():
                    gz_files = list(subdir.glob("*.json.gz"))
                    if not gz_files:
                        sub_dirs_no_json.append(str(subdir))
                        is_success = False
            if len(sub_dirs_no_json) > 0:
                paths = "; ".join(sub_dirs_no_json)
                self._error_message = f"The path '{paths}' has no prof_*.json.gz file"
            return is_success

        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


class TorchJsonFieldValidRule(ValidationRule):
    """valid torch *.json.gz files JSON format"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False
            for item_path in root_path.iterdir():
                if item_path.is_dir():
                    for json_gz_file in item_path.glob("*.json.gz"):
                        with gzip.open(json_gz_file, "rt", encoding="utf-8") as f:
                            json_data = json.load(f)
                        if len(json_data) == 0:
                            self._error_message = f"File is empty: {json_gz_file}"
                            return False

                        distributed_info = json_data.get("distributedInfo", {})
                        required_keys = {"rank", "world_size", "backend"}
                        missing_keys = required_keys - distributed_info.keys()
                        if missing_keys:
                            self._error_message = (
                                f"The 'distributedInfo' field missing: {missing_keys} in FilePath: "
                                f"{json_gz_file}"
                            )
                            return False
                        trace_events = json_data.get("traceEvents", [])
                        trace_valid = (
                            isinstance(trace_events, list) and len(trace_events) > 0
                        )
                        if not trace_valid:
                            self._error_message = f"The 'traceEvents' field is empty in FilePath: {json_gz_file}"
                            return False

                        required_keys = {"ph", "name", "pid", "tid", "ts"}

                        for event in trace_events:
                            missing_keys = required_keys - event.keys()
                            if missing_keys:
                                self._error_message = (
                                    f"The 'traceEvents' field missing: {missing_keys} in FilePath: "
                                    f"{json_gz_file}"
                                )
                                return False
            return True

        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


class NvtxJsonFileExistsRule(ValidationRule):
    """valid worker_process.*.*.jsonl files is existed in 'nvtx_profile' sub path"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False

            profiler_info_filename = "worker_process_*.*.jsonl"

            worker_files = list(root_path.glob(profiler_info_filename))

            if not worker_files:
                self._error_message = (
                    f"No worker_process_*.*.jsonl file found in: {root_path}"
                )
                return False

            return True
        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


class NvtxJsonFieldValidRule(ValidationRule):
    """valid nvtx worker_process_*.*.jsonl files JSON format"""

    def check(self, data) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        self._error_message = ""
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False

            profiler_info_filename = "worker_process_*.*.jsonl"

            worker_files = list(root_path.glob(profiler_info_filename))

            required_for_event = {"start", "end", "textId"}

            for worker_file in worker_files:
                worker_file_obj = Path(worker_file)

                if worker_file_obj.stat().st_size == 0:
                    self._error_message = f"JSONL file is empty: {worker_file}"
                    return False

                start_time_is_exist = False
                specific_event_type_is_exist = False
                missing_keys = []

                with open(worker_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        json_data = json.loads(line)

                        if "startTime" in json_data:
                            start_time_is_exist = True

                        if json_data.get("eventType") == 60:
                            specific_event_type_is_exist = True
                            for key in required_for_event:
                                if key not in json_data:
                                    missing_keys.append(key)
                            if missing_keys:
                                self._error_message = (
                                    f"File field is missing: {missing_keys} in FilePath: "
                                    f"{worker_file}"
                                )
                                return False

                if not start_time_is_exist:
                    self._error_message = f"No 'startTime' found in file: {worker_file}"
                    return False
                if not specific_event_type_is_exist:
                    self._error_message = f"No 'eventType' which equals to 60 found in file: {worker_file}"
                    return False

            return True

        except Exception as e:
            self._error_message = f"Error checking path {root_path}: {e}"
            return False

    @property
    def error_message(self) -> str:
        return self._error_message


class GmmDataRule(ValidationRule):
    """Validation rule for GMM data."""

    def check(self, data: Any) -> bool:
        root_path = _coerce_path(data)
        if root_path is None:
            self._error_message = "Data object is not a path"
            return False
        try:
            if not root_path.exists():
                self._error_message = f"Source path does not exist: {root_path}"
                return False

            group_list_files = list(root_path.rglob("*group_list.pt"))
            if not group_list_files:
                self._error_message = f"No group_list.pt files found in: {root_path}"
                return False

            valid_files = [f for f in group_list_files if "dump_tensor_data" in f.parts]
            if not valid_files:
                self._error_message = (
                    "No group_list.pt files found in dump_tensor_data directories "
                    f"under: {root_path}"
                )
                return False

            return True
        except Exception as e:
            self._error_message = f"Error checking GMM data: {e}"
            return False
