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

import csv
import json
import pytest

from recipe.parser import MemoryClusterParser, get_cluster_parser_cls
from recipe.parser.parser import CLUSTER_PARSER_REGISTRY
from recipe.utils.schema import Constant


def _write_trace_view_json(path, events):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f)


def _write_operator_memory_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Name",
                "Size(KB)",
                "Allocation Time(us)",
                "Release Time(us)",
                "Active Release Time(us)",
                "Duration(us)",
                "Allocation Total Allocated(MB)",
                "Allocation Total Reserved(MB)",
                "Allocation Total Active(MB)",
                "Release Total Allocated(MB)",
                "Release Total Reserved(MB)",
                "Release Total Active(MB)",
                "Stream Ptr",
                "Device Type",
            ]
        )
        for row in rows:
            writer.writerow(row)


def _create_ascend_profile_dir(
    tmp_path, role="actor_update", rank_id=0, trace_events=None, memory_rows=None
):
    role_dir = tmp_path / role
    role_dir.mkdir(exist_ok=True)
    ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
    ascend_pt_dir.mkdir()
    profiler_info = ascend_pt_dir / f"profiler_info_{rank_id}.json"
    profiler_info.write_text("{}")
    metadata = ascend_pt_dir / "profiler_metadata.json"
    metadata.write_text(json.dumps({"role": role}))

    output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
    output_dir.mkdir()

    if trace_events is not None:
        _write_trace_view_json(str(output_dir / "trace_view.json"), trace_events)
    if memory_rows is not None:
        _write_operator_memory_csv(str(output_dir / "operator_memory.csv"), memory_rows)

    return str(tmp_path)


SAMPLE_TRACE_EVENTS = [
    {
        "ph": "X",
        "name": "aten::empty",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "1000000.0",
        "dur": 500.0,
        "args": {"Call stack": "fsdp2.py(112): train_batch;\r\nmodel.py(50): forward"},
    },
    {
        "ph": "X",
        "name": "aten::matmul",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "2000000.0",
        "dur": 1000.0,
        "args": {"Call stack": "model.py(60): forward;\r\nlayer.py(30): __call__"},
    },
    {
        "ph": "X",
        "name": "aten::empty",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "3000000.0",
        "dur": 300.0,
        "args": {"Call stack": "fsdp2.py(120): train_batch;\r\nmodel.py(55): forward"},
    },
    {
        "ph": "X",
        "name": "aten::relu",
        "cat": "kernel",
        "pid": 1,
        "tid": 1,
        "ts": "4000000.0",
        "dur": 200.0,
    },
    {
        "ph": "X",
        "name": "aten::cumsum",
        "cat": "cpu_op",
        "pid": 1,
        "tid": 1,
        "ts": "5000000.0",
        "dur": 800.0,
    },
]

SAMPLE_MEMORY_ROWS = [
    [
        "aten::empty",
        1024.0,
        "1000100.0",
        "",
        "",
        "",
        100.0,
        200.0,
        50.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::empty",
        2048.0,
        "3000050.0",
        "",
        "",
        "",
        200.0,
        300.0,
        100.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::matmul",
        4096.0,
        "2000500.0",
        "",
        "",
        "",
        300.0,
        400.0,
        150.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::unknown",
        512.0,
        "6000000.0",
        "",
        "",
        "",
        50.0,
        100.0,
        25.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
    [
        "aten::empty",
        -1024.0,
        "7000000.0",
        "",
        "",
        "",
        100.0,
        200.0,
        50.0,
        "",
        "",
        "",
        "123",
        "NPU:0",
    ],
]


# =============================================================================
# Parser Registration Tests
# =============================================================================


class TestMemoryParserRegistry:
    def test_memory_parser_registered(self):
        assert "memory" in CLUSTER_PARSER_REGISTRY
        assert CLUSTER_PARSER_REGISTRY["memory"] == MemoryClusterParser

    def test_get_memory_parser_cls(self):
        parser_cls = get_cluster_parser_cls("memory")
        assert parser_cls == MemoryClusterParser


# =============================================================================
# _build_call_stack_index Tests
# =============================================================================


class TestBuildCallStackIndex:
    def test_filters_cpu_op_only(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op2",
                "cat": "kernel",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert "op1" in index
        assert "op2" not in index

    def test_filters_events_without_call_stack(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op2",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {},
            },
            {"ph": "X", "name": "op3", "cat": "cpu_op", "ts": "3000.0", "dur": 300.0},
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert "op1" in index
        assert "op2" not in index
        assert "op3" not in index

    def test_groups_by_name(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "aten::empty",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "aten::empty",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
            {
                "ph": "X",
                "name": "aten::matmul",
                "cat": "cpu_op",
                "ts": "3000.0",
                "dur": 300.0,
                "args": {"Call stack": "stack3"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert len(index["aten::empty"]["entries"]) == 2
        assert len(index["aten::matmul"]["entries"]) == 1

    def test_sorted_by_ts(self, tmp_path):
        events = [
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "3000.0",
                "dur": 300.0,
                "args": {"Call stack": "stack3"},
            },
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "1000.0",
                "dur": 100.0,
                "args": {"Call stack": "stack1"},
            },
            {
                "ph": "X",
                "name": "op1",
                "cat": "cpu_op",
                "ts": "2000.0",
                "dur": 200.0,
                "args": {"Call stack": "stack2"},
            },
        ]
        _write_trace_view_json(str(tmp_path / "trace_view.json"), events)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        ts_values = index["op1"]["ts_list"]
        assert ts_values == [1000.0, 2000.0, 3000.0]

    def test_empty_json(self, tmp_path):
        _write_trace_view_json(str(tmp_path / "trace_view.json"), [])

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        index = parser._build_call_stack_index(str(tmp_path / "trace_view.json"))

        assert len(index) == 0


# =============================================================================
# _match_call_stack Tests
# =============================================================================


class TestMatchCallStack:
    def setup_method(self):
        self.parser = MemoryClusterParser(
            {Constant.INPUT_PATH: "/tmp", Constant.RANK_LIST: "all"}
        )
        self.index = {
            "aten::empty": {
                "entries": [
                    {
                        "ts": 1000.0,
                        "dur": 500.0,
                        "call_stack": "fsdp2.py(10): func1;\r\nmodel.py(20): func2",
                    },
                    {
                        "ts": 3000.0,
                        "dur": 300.0,
                        "call_stack": "fsdp2.py(30): func3;\r\nmodel.py(40): func4",
                    },
                ],
                "ts_list": [1000.0, 3000.0],
            },
            "aten::matmul": {
                "entries": [
                    {
                        "ts": 5000.0,
                        "dur": 1000.0,
                        "call_stack": "model.py(50): forward",
                    },
                ],
                "ts_list": [5000.0],
            },
        }

    def test_match_found(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 1100.0, self.index
        )
        assert call_stack == "fsdp2.py(10): func1;\r\nmodel.py(20): func2"
        assert call_stack_top == "fsdp2.py(10): func1"

    def test_match_closest_ts(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 3500.0, self.index
        )
        assert call_stack == "fsdp2.py(30): func3;\r\nmodel.py(40): func4"
        assert call_stack_top == "fsdp2.py(30): func3"

    def test_name_not_found(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::unknown", 1000.0, self.index
        )
        assert call_stack == ""
        assert call_stack_top == ""

    def test_all_ts_greater_than_allocation_time(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 500.0, self.index
        )
        assert call_stack == ""
        assert call_stack_top == ""

    def test_exact_ts_match(self):
        call_stack, call_stack_top = self.parser._match_call_stack(
            "aten::empty", 3000.0, self.index
        )
        assert call_stack == "fsdp2.py(30): func3;\r\nmodel.py(40): func4"


# =============================================================================
# _parse_operator_memory Tests
# =============================================================================


class TestParseOperatorMemory:
    def test_parse_basic(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [
                    {
                        "ts": 1000000.0,
                        "dur": 500.0,
                        "call_stack": "fsdp2.py(10): func;\r\nmodel.py(20): func2",
                    },
                ],
                "ts_list": [1000000.0],
            },
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        row = results[0]
        assert row["name"] == "aten::empty"
        assert row["size_kb"] == 1024.0
        assert row["start_time_ms"] == pytest.approx(1000.1)
        assert row["call_stack"] == "fsdp2.py(10): func;\r\nmodel.py(20): func2"
        assert row["call_stack_top"] == "fsdp2.py(10): func"
        assert row["role"] == "actor_update"
        assert row["rank_id"] == 0
        assert row["device_type"] == "NPU:0"

    def test_parse_negative_size(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    -1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [
                    {"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"},
                ],
                "ts_list": [1000000.0],
            },
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        assert results[0]["size_kb"] == -1024.0

    def test_parse_duration_conversion(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "5000.0",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["duration_ms"] == pytest.approx(5.0)

    def test_parse_empty_duration(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    1024.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    100.0,
                    200.0,
                    50.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["duration_ms"] == 0.0

    def test_parse_unmatched_call_stack(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::unknown",
                    512.0,
                    "1000100.0",
                    "",
                    "",
                    "",
                    50.0,
                    100.0,
                    25.0,
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {}

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert results[0]["call_stack"] == ""
        assert results[0]["call_stack_top"] == ""

    def test_parse_empty_allocation_total_fields(self, tmp_path):
        csv_path = str(tmp_path / "operator_memory.csv")
        _write_operator_memory_csv(
            csv_path,
            [
                [
                    "aten::empty",
                    -1024.0,
                    "1000100.0",
                    "2000100.0",
                    "2000100.0",
                    "1000.0",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "123",
                    "NPU:0",
                ],
            ],
        )

        index = {
            "aten::empty": {
                "entries": [{"ts": 1000000.0, "dur": 500.0, "call_stack": "stack"}],
                "ts_list": [1000000.0],
            }
        }

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )
        results = parser._parse_operator_memory(
            csv_path, index, rank_id=0, role="actor_update"
        )

        assert len(results) == 1
        assert results[0]["total_allocated_mb"] == 0.0
        assert results[0]["total_reserved_mb"] == 0.0
        assert results[0]["total_active_mb"] == 0.0


# =============================================================================
# _extract_timestamp_key Tests
# =============================================================================


class TestExtractTimestampKey:
    def test_timestamp_format(self):
        assert (
            MemoryClusterParser._extract_timestamp_key(
                "/data/actor/20250101_120000_ascend_pt"
            )
            == "20250101_120000"
        )

    def test_sort_order(self):
        paths = [
            "/data/role/20250102_010000_ascend_pt",
            "/data/role/20250101_230000_ascend_pt",
            "/data/role/20250101_120000_ascend_pt",
        ]
        sorted_paths = sorted(paths, key=MemoryClusterParser._extract_timestamp_key)
        assert sorted_paths[0].endswith("20250101_120000_ascend_pt")
        assert sorted_paths[1].endswith("20250101_230000_ascend_pt")
        assert sorted_paths[2].endswith("20250102_010000_ascend_pt")


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestMemoryParserEndToEnd:
    def test_full_pipeline(self, tmp_path):
        input_path = _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: input_path, Constant.RANK_LIST: "all"}
        )

        data_maps = parser.allocate_prof_data(input_path)
        assert len(data_maps) == 1
        assert data_maps[0]["role"] == "actor_update"
        assert data_maps[0]["rank_id"] == 0

        events = parser.parse_analysis_data(
            data_maps[0]["profiler_data_path"],
            data_maps[0]["rank_id"],
            data_maps[0]["role"],
        )

        assert len(events) == 5

        aten_empty_events = [e for e in events if e["name"] == "aten::empty"]
        assert len(aten_empty_events) == 3

        first_empty = [e for e in aten_empty_events if e["size_kb"] == 1024.0][0]
        assert first_empty["call_stack"] != ""
        assert "fsdp2.py(112)" in first_empty["call_stack_top"]

        matmul_events = [e for e in events if e["name"] == "aten::matmul"]
        assert len(matmul_events) == 1
        assert matmul_events[0]["call_stack"] != ""

        unknown_events = [e for e in events if e["name"] == "aten::unknown"]
        assert len(unknown_events) == 1
        assert unknown_events[0]["call_stack"] == ""
        assert unknown_events[0]["call_stack_top"] == ""

        release_events = [e for e in events if e["size_kb"] < 0]
        assert len(release_events) == 1

    def test_multiple_roles(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )
        _create_ascend_profile_dir(
            tmp_path,
            role="rollout_generate",
            rank_id=1,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))
        assert len(data_maps) == 2

        roles = {dm["role"] for dm in data_maps}
        assert roles == {"actor_update", "rollout_generate"}

    def test_missing_trace_view(self, tmp_path):
        role_dir = tmp_path / "actor"
        role_dir.mkdir()
        ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
        ascend_pt_dir.mkdir()
        (ascend_pt_dir / "profiler_info_0.json").write_text("{}")
        (ascend_pt_dir / "profiler_metadata.json").write_text(
            json.dumps({"role": "actor"})
        )
        output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
        output_dir.mkdir()
        _write_operator_memory_csv(
            str(output_dir / "operator_memory.csv"), SAMPLE_MEMORY_ROWS
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data(str(output_dir), 0, "actor")
        assert len(events) == 0

    def test_missing_operator_memory(self, tmp_path):
        role_dir = tmp_path / "actor"
        role_dir.mkdir()
        ascend_pt_dir = role_dir / "20250101_120000_ascend_pt"
        ascend_pt_dir.mkdir()
        (ascend_pt_dir / "profiler_info_0.json").write_text("{}")
        (ascend_pt_dir / "profiler_metadata.json").write_text(
            json.dumps({"role": "actor"})
        )
        output_dir = ascend_pt_dir / "ASCEND_PROFILER_OUTPUT"
        output_dir.mkdir()
        _write_trace_view_json(str(output_dir / "trace_view.json"), SAMPLE_TRACE_EVENTS)

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data(str(output_dir), 0, "actor")
        assert len(events) == 0

    def test_empty_profiler_data_path(self):
        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: "/tmp", Constant.RANK_LIST: "all"}
        )

        events = parser.parse_analysis_data("", 0, "actor")
        assert len(events) == 0

    def test_non_all_rank_list(self, tmp_path):
        _create_ascend_profile_dir(
            tmp_path,
            role="actor_update",
            rank_id=0,
            trace_events=SAMPLE_TRACE_EVENTS,
            memory_rows=SAMPLE_MEMORY_ROWS,
        )

        parser = MemoryClusterParser(
            {Constant.INPUT_PATH: str(tmp_path), Constant.RANK_LIST: "0,1"}
        )

        data_maps = parser.allocate_prof_data(str(tmp_path))
        assert len(data_maps) == 0
