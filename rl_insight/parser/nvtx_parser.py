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

import json
from loguru import logger
import os
import re
from collections import defaultdict

from .parser import BaseClusterParser, register_cluster_parser
from rl_insight.utils.schema import Constant, DataMap, EventRow
from rl_insight.data import DataEnum


@register_cluster_parser("nvtx")
class NvtxClusterParser(BaseClusterParser):
    input_type: DataEnum = DataEnum.MULTI_JSON_NVTX

    def __init__(self, params) -> None:
        super().__init__(params)

    def parse_analysis_data(
        self, profiler_data_path: str, rank_id: int, role: str
    ) -> list[EventRow]:
        events: list[EventRow] = []
        string_map: dict[int, str] = {}
        raw_events = []
        global_start_time = None

        # define regular expression for rank info search
        rank_pattern = re.compile(r'^RANK="?(\d+)"?$', re.IGNORECASE)

        # process id can obtain from file name directly
        process_id = os.path.basename(profiler_data_path).split(".")[-3].split("_")[-1]

        with open(profiler_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data: dict = json.loads(line)
                # id -> value
                if (
                    data.get("table") == "StringIds"
                    and "id" in data
                    and "value" in data
                ):
                    string_map[data["id"]] = data["value"]
                # get the rank info
                if data.get(
                    "table"
                ) == "META_DATA_CAPTURE" and "ENVIRONMENT_VARIABLE" in data.get(
                    "name", ""
                ):
                    value = data.get("value", "").strip()
                    match = rank_pattern.match(value)
                    if match:
                        rank_id = int(match.group(1))
                # get the global start time
                if data.get("table") == "ANALYSIS_DETAILS":
                    global_start_time = data.get("startTime", None)
                # get role, start_ids, end_ids
                if data.get("eventType") == 60:
                    raw_events.append(data)

        if rank_id < 0:
            logger.warning(f"Path {profiler_data_path}: No valid rank for Analysis")
            return events

        if global_start_time is None:
            logger.warning(
                f"Path {profiler_data_path}: No valid global start time for Analysis"
            )
            return events

        us_to_ms = Constant.US_TO_MS
        ns_to_us = Constant.NS_TO_US

        for raw_event in raw_events:
            text_id = raw_event.get("textId", -1)
            role = string_map.get(text_id, "")
            start_ids = raw_event.get("start")
            end_ids = raw_event.get("end")

            if not role:
                logger.warning(f"Path {profiler_data_path}: No valid role for Analysis")
                return events

            if start_ids is None or end_ids is None:
                logger.warning(
                    f"Path {profiler_data_path}: No valid timing for Analysis"
                )
                return events

            # Convert to milliseconds
            start_time_ms = (start_ids + global_start_time) / us_to_ms / ns_to_us
            duration_ms = (end_ids - start_ids) / us_to_ms / ns_to_us
            end_time_ms = start_time_ms + duration_ms

            event_data: EventRow = {
                "name": role,
                "role": role,
                "domain": "default",
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "duration_ms": duration_ms,
                "rank_id": rank_id,
                "tid": process_id,
            }
            events.append(event_data)

        return events

    def allocate_prof_data(self, input_path: str) -> list[DataMap]:
        """Allocate and process profiling data maps from input path."""
        nsight_dirs = []
        for root, dirs, files in os.walk(input_path):
            for file_name in files:
                file_name_parts = file_name.split(".")
                if (
                    file_name.endswith(Constant.NV_PROFILER_SUFFIX)
                    and len(file_name_parts) == 3
                ):
                    path = os.path.join(root, file_name)
                    nsight_dirs.append({"role": file_name_parts[1], "path": path})
        data_map = self._get_data_map(nsight_dirs)
        data_maps = self._get_rank_path_with_role(data_map)
        return data_maps

    def _get_data_map(self, nsight_dirs) -> dict[str, list[str]]:
        data_map = {}
        role_map = defaultdict(list)

        for path_info in nsight_dirs:
            role = path_info.get("role")
            file_name = path_info.get("path")

            # For RL Analysis
            role_map[role].append(file_name)

        for map_key, file_list in role_map.items():
            data_map[map_key] = file_list

        return data_map

    def _get_rank_path_with_role(self, data_map) -> list[DataMap]:
        """Get json path information for all ranks.

        This function is intentionally decoupled from class state; pass required
        dependencies in via arguments.
        """

        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        roles = list(data_map.keys())
        data_paths: list[DataMap] = []
        for task_role in roles:
            file_list = data_map[task_role]

            for profiler_data_path in file_list:
                data_path_dict: DataMap = {
                    Constant.RANK_ID: -1,  # rank_id will be loaded from jsonl file.
                    Constant.ROLE: task_role,  # real role name will be loaded from jsonl file
                    Constant.PROFILER_DATA_PATH: "",
                }

                if os.path.exists(profiler_data_path):
                    data_path_dict[Constant.PROFILER_DATA_PATH] = profiler_data_path
                    data_paths.append(data_path_dict)
                else:
                    logger.warning(
                        f"Profiler data file not found, role: {task_role}, data path: {profiler_data_path}."
                    )
        return data_paths
