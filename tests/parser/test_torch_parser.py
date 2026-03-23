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
Unit tests for torch_parser module.

Tests cover:
- TorchClusterParser registration
- parse_analysis_data method
- allocate_prof_data method
- _get_data_map method
- _get_rank_path_with_role method
"""

import gzip
import json
from pathlib import Path

import pytest

from rl_insight.parser import (
    CLUSTER_PARSER_REGISTRY,
    get_cluster_parser_cls,
)
from rl_insight.utils.schema import Constant
from rl_insight.parser import TorchClusterParser


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_torch_profiler_data():
    """Sample torch profiler data with traceEvents."""
    return {
        "distributedInfo": {
            "rank": 0,
            "world_size": 2,
        },
        "traceEvents": [
            {
                "ph": "X",
                "pid": 12345,
                "tid": 1,
                "ts": 1000000,
                "dur": 500000,
                "name": "forward",
                "cat": "python",
            },
            {
                "ph": "X",
                "pid": 12345,
                "tid": 1,
                "ts": 1600000,
                "dur": 300000,
                "name": "backward",
                "cat": "python",
            },
        ],
    }


@pytest.fixture
def sample_torch_profiler_data_no_distributed_info():
    """Sample torch profiler data without distributedInfo."""
    return {
        "traceEvents": [
            {
                "ph": "X",
                "pid": 12345,
                "tid": 1,
                "ts": 1000000,
                "dur": 500000,
                "name": "forward",
                "cat": "python",
            },
        ],
    }


@pytest.fixture
def sample_torch_profiler_data_empty_events():
    """Sample torch profiler data with empty traceEvents."""
    return {
        "distributedInfo": {
            "rank": 0,
            "world_size": 2,
        },
        "traceEvents": [],
    }


@pytest.fixture
def sample_torch_profiler_data_invalid_timing():
    """Sample torch profiler data with invalid timing values."""
    return {
        "distributedInfo": {
            "rank": 0,
            "world_size": 2,
        },
        "traceEvents": [
            {
                "ph": "X",
                "pid": 12345,
                "tid": 1,
                "ts": -1,
                "dur": 500000,
                "name": "forward",
                "cat": "python",
            },
        ],
    }


@pytest.fixture
def mock_torch_profiler_structure(tmp_path, sample_torch_profiler_data):
    """
    Create mock torch profiler directory structure.

    Structure:
    tmp_path/
    └── actor_train/
        └── trace.json.gz
    """
    role_dir = tmp_path / "actor_train"
    role_dir.mkdir()

    trace_file = role_dir / "trace.json.gz"
    with gzip.open(trace_file, "wt", encoding="utf-8") as f:
        json.dump(sample_torch_profiler_data, f)

    return str(tmp_path)


@pytest.fixture
def mock_torch_profiler_with_async_llm(tmp_path, sample_torch_profiler_data):
    """
    Create mock torch profiler directory structure with async_llm file that should be excluded.
    """
    role_dir = tmp_path / "actor_train"
    role_dir.mkdir()

    trace_file = role_dir / "trace.json.gz"
    with gzip.open(trace_file, "wt", encoding="utf-8") as f:
        json.dump(sample_torch_profiler_data, f)

    async_llm_file = role_dir / "async_llm_trace.json.gz"
    with gzip.open(async_llm_file, "wt", encoding="utf-8") as f:
        json.dump(sample_torch_profiler_data, f)

    return str(tmp_path)


# =============================================================================
# Parser Registration Tests
# =============================================================================


class TestTorchParserRegistry:
    """Tests for torch parser registration."""

    def test_torch_parser_registered(self):
        """Test that torch parser is registered in the registry."""
        assert "torch" in CLUSTER_PARSER_REGISTRY
        assert CLUSTER_PARSER_REGISTRY["torch"] == TorchClusterParser

    def test_get_torch_parser_cls(self):
        """Test getting torch parser class from registry."""
        parser_cls = get_cluster_parser_cls("torch")
        assert parser_cls == TorchClusterParser


# =============================================================================
# TorchClusterParser Tests
# =============================================================================


class TestTorchClusterParser:
    """Tests for TorchClusterParser implementation."""

    def test_parse_analysis_data_success(self, mock_torch_profiler_structure):
        """Test parsing analysis data from valid json.gz file."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_structure,
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = (
            Path(mock_torch_profiler_structure) / "actor_train" / "trace.json.gz"
        )
        events = parser.parse_analysis_data(str(trace_file), -1, "actor_train")

        assert len(events) == 1
        event = events[0]
        assert event["name"] == "actor_train"
        assert event["role"] == "actor_train"
        assert event["domain"] == "default"
        assert event["rank_id"] == 0
        assert event["tid"] == 12345
        assert event["start_time_ms"] == pytest.approx(1000.0)
        assert event["end_time_ms"] == pytest.approx(1900.0)
        assert event["duration_ms"] == pytest.approx(900.0)

    def test_parse_analysis_data_no_distributed_info(
        self, tmp_path, sample_torch_profiler_data_no_distributed_info
    ):
        """Test parsing analysis data without distributedInfo."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(sample_torch_profiler_data_no_distributed_info, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 0

    def test_parse_analysis_data_empty_events(
        self, tmp_path, sample_torch_profiler_data_empty_events
    ):
        """Test parsing analysis data with empty traceEvents."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(sample_torch_profiler_data_empty_events, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 0

    def test_parse_analysis_data_invalid_timing(
        self, tmp_path, sample_torch_profiler_data_invalid_timing
    ):
        """Test parsing analysis data with invalid timing values."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(sample_torch_profiler_data_invalid_timing, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 0

    def test_parse_analysis_data_empty_file(self, tmp_path):
        """Test parsing analysis data from empty json.gz file."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump({}, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 0

    def test_allocate_prof_data_success(self, mock_torch_profiler_structure):
        """Test allocating profiler data from directory structure."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_structure,
                Constant.RANK_LIST: "all",
            }
        )

        data_maps = parser.allocate_prof_data(mock_torch_profiler_structure)

        assert len(data_maps) == 1
        assert data_maps[0]["rank_id"] == -1
        assert data_maps[0]["role"] == "actor_train"
        assert "trace.json.gz" in data_maps[0]["profiler_data_path"]

    def test_allocate_prof_data_excludes_async_llm(
        self, mock_torch_profiler_with_async_llm
    ):
        """Test that async_llm files are excluded from allocation."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_with_async_llm,
                Constant.RANK_LIST: "all",
            }
        )

        data_maps = parser.allocate_prof_data(mock_torch_profiler_with_async_llm)

        assert len(data_maps) == 1
        assert "async_llm" not in data_maps[0]["profiler_data_path"]

    def test_allocate_prof_data_empty_directory(self, tmp_path):
        """Test allocating profiler data from empty directory."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))

        assert len(data_maps) == 0

    def test_allocate_prof_data_non_all_rank_list(self, mock_torch_profiler_structure):
        """Test allocating profiler data with non-'all' rank list."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_structure,
                Constant.RANK_LIST: "0,1",
            }
        )

        data_maps = parser.allocate_prof_data(mock_torch_profiler_structure)

        assert len(data_maps) == 0

    def test_get_data_map(self):
        """Test building data map from file list."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: "/tmp",
                Constant.RANK_LIST: "all",
            }
        )

        nv_files = [
            {"role": "actor_train", "path": "/tmp/actor_train/trace.json.gz"},
            {"role": "actor_train", "path": "/tmp/actor_train/trace2.json.gz"},
            {"role": "rollout_generate", "path": "/tmp/rollout_generate/trace.json.gz"},
        ]

        data_map = parser._get_data_map(nv_files)

        assert "actor_train" in data_map
        assert "rollout_generate" in data_map
        assert len(data_map["actor_train"]) == 2
        assert len(data_map["rollout_generate"]) == 1

    def test_get_data_map_empty(self):
        """Test building data map from empty file list."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: "/tmp",
                Constant.RANK_LIST: "all",
            }
        )

        data_map = parser._get_data_map([])

        assert len(data_map) == 0

    def test_get_rank_path_with_role_success(self, mock_torch_profiler_structure):
        """Test getting rank paths with role from data map."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_structure,
                Constant.RANK_LIST: "all",
            }
        )

        trace_file = (
            Path(mock_torch_profiler_structure) / "actor_train" / "trace.json.gz"
        )
        data_map = {
            "actor_train": [str(trace_file)],
        }

        data_paths = parser._get_rank_path_with_role(data_map)

        assert len(data_paths) == 1
        assert data_paths[0]["rank_id"] == -1
        assert data_paths[0]["role"] == "actor_train"
        assert str(trace_file) == data_paths[0]["profiler_data_path"]

    def test_get_rank_path_with_role_missing_file(self, tmp_path):
        """Test getting rank paths when file does not exist."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        non_existent_file = str(tmp_path / "non_existent.json.gz")
        data_map = {"actor_train": [non_existent_file]}

        data_paths = parser._get_rank_path_with_role(data_map)

        assert len(data_paths) == 0

    def test_full_pipeline(self, mock_torch_profiler_structure):
        """Test full parsing pipeline from directory to events."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: mock_torch_profiler_structure,
                Constant.RANK_LIST: "all",
            }
        )

        data_maps = parser.allocate_prof_data(mock_torch_profiler_structure)

        assert len(data_maps) == 1

        all_events = []
        for data_map in data_maps:
            events = parser.parse_analysis_data(
                data_map["profiler_data_path"],
                data_map["rank_id"],
                data_map["role"],
            )
            all_events.extend(events)

        assert len(all_events) == 1
        assert all_events[0]["role"] == "actor_train"
        assert all_events[0]["rank_id"] == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestTorchClusterParserEdgeCases:
    """Tests for edge cases in TorchClusterParser."""

    def test_parse_with_missing_pid(self, tmp_path):
        """Test parsing when traceEvents have missing pid."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        data = {
            "distributedInfo": {"rank": 0},
            "traceEvents": [
                {
                    "ph": "X",
                    "tid": 1,
                    "ts": 1000000,
                    "dur": 500000,
                    "name": "forward",
                },
            ],
        }

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 1
        assert events[0]["tid"] == -1

    def test_parse_with_negative_duration(self, tmp_path):
        """Test parsing when traceEvents have negative duration."""
        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        data = {
            "distributedInfo": {"rank": 0},
            "traceEvents": [
                {
                    "ph": "X",
                    "pid": 12345,
                    "tid": 1,
                    "ts": 1000000,
                    "dur": -500000,
                    "name": "forward",
                },
            ],
        }

        trace_file = tmp_path / "trace.json.gz"
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        events = parser.parse_analysis_data(str(trace_file), -1, "test_role")

        assert len(events) == 0

    def test_parse_multiple_roles(self, tmp_path, sample_torch_profiler_data):
        """Test parsing with multiple roles in directory structure."""
        actor_dir = tmp_path / "actor_train"
        actor_dir.mkdir()
        actor_trace = actor_dir / "trace.json.gz"
        with gzip.open(actor_trace, "wt", encoding="utf-8") as f:
            json.dump(sample_torch_profiler_data, f)

        rollout_dir = tmp_path / "rollout_generate"
        rollout_dir.mkdir()
        rollout_trace = rollout_dir / "trace.json.gz"
        with gzip.open(rollout_trace, "wt", encoding="utf-8") as f:
            json.dump(sample_torch_profiler_data, f)

        parser = TorchClusterParser(
            {
                Constant.INPUT_PATH: str(tmp_path),
                Constant.RANK_LIST: "all",
            }
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))

        assert len(data_maps) == 2
        roles = {dm["role"] for dm in data_maps}
        assert roles == {"actor_train", "rollout_generate"}
