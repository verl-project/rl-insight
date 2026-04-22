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

"""
Unit tests for nvtx_parser module.

Tests cover:
- NvtxClusterParser registration
- parse_analysis_data method
- allocate_prof_data method
- _get_data_map method
- _get_rank_path_with_role method
"""

import json
import os
import pytest
from unittest.mock import patch

from rl_insight.parser.nvtx_parser import NvtxClusterParser
from rl_insight.utils.schema import Constant
from rl_insight.data import DataEnum


def create_test_nvtx_jsonl(tmp_path, filename: str = "worker_process_1234.5.jsonl"):
    """
    Helper function to create a temporary valid JSONL test file
    Contains StringIds, RANK environment variable, ANALYSIS_DETAILS, and eventType=60
    """
    file_path = os.path.join(tmp_path, filename)

    # Core test data lines
    test_lines = [
        # String ID mapping table
        json.dumps({"id": 107, "table": "StringIds", "value": "compute_values"}),
        json.dumps({"id": 61, "table": "StringIds", "value": "test_func"}),
        # RANK environment variable
        json.dumps(
            {
                "name": "PROCESS_0:ENVIRONMENT_VARIABLE",
                "table": "META_DATA_CAPTURE",
                "value": 'RANK="0"',
            }
        ),
        # Global start time
        json.dumps({"table": "ANALYSIS_DETAILS", "startTime": 1000000}),
        # Key NVTX event with eventType=60
        json.dumps(
            {
                "domainId": 0,
                "end": 2487364321,
                "eventType": 60,
                "globalTid": 282747880941798,
                "rangeId": 2,
                "start": 18624241,
                "table": "NVTX_EVENTS",
                "textId": 107,
            }
        ),
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    return file_path


class TestNvtxClusterParser:
    """Unit test suite for NvtxClusterParser"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fixture: Initialize parser before each test case"""
        self.params = {"rank_list": "all"}
        self.parser = NvtxClusterParser(self.params)

    def test_class_attributes(self):
        """Test basic class attributes are correctly defined"""
        assert self.parser.input_type == DataEnum.MULTI_JSON_NVTX

    def test_parse_analysis_data_normal_case(self, tmp_path):
        """
        Test normal parsing flow:
        Build string map, extract RANK, resolve textId for eventType=60, compute time fields
        """
        # Create valid test file
        file_path = create_test_nvtx_jsonl(tmp_path)

        # Execute parsing
        events = self.parser.parse_analysis_data(
            profiler_data_path=file_path, rank_id=-1, role="test_role"
        )

        # Validate results
        assert len(events) == 1
        event = events[0]

        assert event["rank_id"] == 0
        assert event["role"] == "compute_values"
        assert event["name"] == "compute_values"
        assert event["tid"] == "1234"  # Extracted from filename
        assert event["start_time_ms"] > 0
        assert event["end_time_ms"] > event["start_time_ms"]
        assert event["duration_ms"] > 0

    def test_parse_no_rank(self, tmp_path):
        """Test parsing returns empty list when RANK info is missing"""
        file_path = os.path.join(tmp_path, "worker_process_1234.5.jsonl")
        test_lines = [
            json.dumps({"id": 107, "table": "StringIds", "value": "compute_values"}),
            json.dumps({"table": "ANALYSIS_DETAILS", "startTime": 1000000}),
            json.dumps({"eventType": 60, "textId": 107, "start": 100, "end": 200}),
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines))

        events = self.parser.parse_analysis_data(file_path, -1, "test")
        assert len(events) == 0

    def test_parse_no_role_textid(self, tmp_path):
        """Test parsing returns empty list when textId has no matching StringIds entry"""
        file_path = os.path.join(tmp_path, "worker_process_1234.5.jsonl")
        test_lines = [
            json.dumps(
                {"table": "META_DATA_CAPTURE", "name": "...", "value": "RANK=0"}
            ),
            json.dumps({"table": "ANALYSIS_DETAILS", "startTime": 1000000}),
            json.dumps({"eventType": 60, "textId": 9999, "start": 100, "end": 200}),
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines))

        events = self.parser.parse_analysis_data(file_path, -1, "test")
        assert len(events) == 0

    def test_parse_missing_time(self, tmp_path):
        """Test parsing returns empty list when start/end timestamps are missing"""
        file_path = os.path.join(tmp_path, "worker_process_1234.5.jsonl")
        test_lines = [
            json.dumps({"id": 107, "table": "StringIds", "value": "a"}),
            json.dumps({"table": "META_DATA_CAPTURE", "value": "RANK=0"}),
            json.dumps({"eventType": 60, "textId": 107}),
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_lines))

        events = self.parser.parse_analysis_data(file_path, -1, "test")
        assert len(events) == 0

    def test_allocate_prof_data(self, tmp_path):
        """Test data allocation: filter valid profiler files with correct naming format"""
        # Create test directory structure
        sub_dir = os.path.join(tmp_path, "test_dir")
        os.makedirs(sub_dir, exist_ok=True)

        # Create valid and invalid test files
        valid_file = os.path.join(sub_dir, "worker_process.123.jsonl")
        invalid_file1 = os.path.join(sub_dir, "worker.jsonl")
        invalid_file2 = os.path.join(sub_dir, "worker_process_123.jsonl")

        for f in [valid_file, invalid_file1, invalid_file2]:
            with open(f, "w") as fobj:
                fobj.write("{}")

        # Execute method
        data_maps = self.parser.allocate_prof_data(str(tmp_path))

        # Verify valid files are detected
        assert len(data_maps) >= 1

    def test_get_rank_path_with_role(self):
        """
        Test building rank-path mapping with role info.
        Mock file existence to avoid dependency on real files.
        """
        # Mock file paths (do not need to exist physically)
        test_map = {
            "actor": ["/tmp/test1.jsonl", "/tmp/test2.jsonl"],
            "learner": ["/tmp/test3.jsonl"],
        }

        # Mock os.path.exists to always return True for unit test
        with patch("os.path.exists", return_value=True):
            paths = self.parser._get_rank_path_with_role(test_map)

        # Validate output structure and content
        assert len(paths) == 3
        assert paths[0][Constant.ROLE] == "actor"
        assert paths[1][Constant.ROLE] == "actor"
        assert paths[2][Constant.ROLE] == "learner"

        assert paths[0][Constant.PROFILER_DATA_PATH] == "/tmp/test1.jsonl"
        assert paths[1][Constant.PROFILER_DATA_PATH] == "/tmp/test2.jsonl"
        assert paths[2][Constant.PROFILER_DATA_PATH] == "/tmp/test3.jsonl"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
