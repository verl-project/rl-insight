# Copyright (c) 2026 verl-project authors.
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

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from rl_insight import cli
from rl_insight.data_inspection import (
    ExperimentSummary,
    SourceRange,
    _inspect_tempo,
    _parse_openmetrics,
    _prometheus_sample_count,
    _tempo_pairs,
    format_summaries,
    inspect_data_directory,
)


def _display_width(text: str) -> int:
    return sum(
        0
        if unicodedata.combining(character)
        else 2
        if unicodedata.east_asian_width(character) in {"F", "W"}
        else 1
        for character in text
    )


def test_parse_openmetrics_extracts_project_experiment_and_timestamp() -> None:
    line = (
        'rl_insight_monitor_agent_loop_run_info{experiment_name="exp-1",'
        'global_steps="1",project="proj-1"} 1 1788319580.703'
    )

    parsed = list(_parse_openmetrics(line))

    assert len(parsed) == 1
    pair, timestamp = parsed[0]
    assert pair == ("proj-1", "exp-1")
    assert timestamp == datetime(2026, 9, 2, 3, 26, 20, 703000, tzinfo=timezone.utc)


def test_tempo_pairs_extracts_nested_span_attributes() -> None:
    row = {
        "rs": [
            {
                "Resource": {"Attrs": []},
                "ss": [
                    {
                        "Spans": [
                            {
                                "Attrs": [
                                    {"Key": "project", "Value": ["proj-1"]},
                                    {"Key": "experiment_name", "Value": ["exp-1"]},
                                ]
                            }
                        ]
                    }
                ],
            }
        ]
    }

    assert _tempo_pairs(row) == {("proj-1", "exp-1")}


def test_inspect_tempo_reads_scalar_attribute_values(tmp_path) -> None:
    tempo_dir = tmp_path / "tempo" / "traces" / "single-tenant" / "test"
    tempo_dir.mkdir(parents=True)
    attribute_type = pa.struct(
        [
            pa.field("Key", pa.string()),
            pa.field("Value", pa.string()),
        ]
    )
    resource_spans_type = pa.struct(
        [
            pa.field(
                "Resource",
                pa.struct([pa.field("Attrs", pa.list_(attribute_type))]),
            )
        ]
    )
    table = pa.table(
        {
            "StartTimeUnixNano": pa.array(
                [1788319580000000000], type=pa.uint64()
            ),
            "EndTimeUnixNano": pa.array(
                [1788319640000000000], type=pa.uint64()
            ),
            "rs": pa.array(
                [
                    [
                        {
                            "Resource": {
                                "Attrs": [
                                    {"Key": "project", "Value": "proj-1"},
                                    {
                                        "Key": "experiment_name",
                                        "Value": "exp-1",
                                    },
                                ]
                            }
                        }
                    ]
                ],
                type=pa.list_(resource_spans_type),
            ),
        }
    )
    pq.write_table(table, tempo_dir / "data.parquet")

    ranges = _inspect_tempo(tmp_path)

    assert ranges == {
        ("proj-1", "exp-1"): SourceRange(
            start=datetime(2026, 9, 2, 3, 26, 20, tzinfo=timezone.utc),
            end=datetime(2026, 9, 2, 3, 27, 20, tzinfo=timezone.utc),
        )
    }


def test_prometheus_sample_count_sums_block_metadata(tmp_path) -> None:
    block_a = tmp_path / "block-a"
    block_b = tmp_path / "block-b"
    block_a.mkdir()
    block_b.mkdir()
    (block_a / "meta.json").write_text(
        '{"stats":{"numSamples":100}}', encoding="utf-8"
    )
    (block_b / "meta.json").write_text(
        '{"stats":{"numSamples":23}}', encoding="utf-8"
    )

    assert _prometheus_sample_count(tmp_path) == 123


def test_inspect_data_directory_returns_empty_for_empty_directory(tmp_path) -> None:
    assert inspect_data_directory(tmp_path) == []


def test_data_inspect_reports_no_data_without_table(tmp_path, capsys) -> None:
    args = cli._build_parser().parse_args(
        ["data", "inspect", "--log-dir", str(tmp_path)]
    )

    assert cli._handle_data_inspect(args) == 0
    assert capsys.readouterr().out.strip() == "Data not found."


def test_parser_accepts_log_dir(tmp_path) -> None:
    args = cli._build_parser().parse_args(
        ["data", "inspect", "--log-dir", str(tmp_path)]
    )

    assert args.log_dir == tmp_path
    assert args.func.__name__ == "_handle_data_inspect"


def test_format_summaries_renders_project_experiment_and_ranges() -> None:
    summary = ExperimentSummary(
        project="proj-1",
        experiment="exp-1",
        prometheus=SourceRange(
            start=datetime(2026, 9, 2, 3, 26, 20, tzinfo=timezone.utc),
            end=datetime(2026, 9, 2, 3, 27, 20, tzinfo=timezone.utc),
        ),
        tempo=None,
    )

    output = format_summaries([summary])

    assert "proj-1" in output
    assert "exp-1" in output
    assert "2026-09-02 03:26～03:27" in output
    assert "-" in output


def test_format_summaries_renders_full_dates_when_range_crosses_days() -> None:
    summary = ExperimentSummary(
        project="proj-1",
        experiment="exp-1",
        prometheus=SourceRange(
            start=datetime(2026, 9, 2, 2, 26, tzinfo=timezone.utc),
            end=datetime(2026, 9, 3, 5, 12, tzinfo=timezone.utc),
        ),
        tempo=None,
    )

    output = format_summaries([summary])

    lines = output.splitlines()
    assert any("2026-09-02 02:26～" in line for line in lines)
    assert any("2026-09-03 05:12" in line for line in lines)


def test_format_summaries_keeps_table_aligned_with_multiline_ranges() -> None:
    summary = ExperimentSummary(
        project="proj-1",
        experiment="exp-1",
        prometheus=SourceRange(
            start=datetime(2026, 9, 2, 2, 26, tzinfo=timezone.utc),
            end=datetime(2026, 9, 3, 5, 12, tzinfo=timezone.utc),
        ),
        tempo=SourceRange(
            start=datetime(2026, 9, 2, 3, 26, tzinfo=timezone.utc),
            end=datetime(2026, 9, 2, 3, 27, tzinfo=timezone.utc),
        ),
    )

    output = format_summaries([summary])
    separator_indexes = {
        tuple(
            _display_width(line[:index])
            for index, char in enumerate(line)
            if char == "|"
        )
        for line in output.splitlines()
        if line.startswith("|")
    }

    assert len(separator_indexes) == 1


def test_format_summaries_merges_project_cells_and_separates_rows() -> None:
    summaries = [
        ExperimentSummary(
            project="proj-1",
            experiment="exp-1",
            prometheus=SourceRange(
                start=datetime(2026, 9, 2, 3, 26, tzinfo=timezone.utc),
                end=datetime(2026, 9, 2, 3, 27, tzinfo=timezone.utc),
            ),
            tempo=None,
        ),
        ExperimentSummary(
            project="proj-1",
            experiment="exp-2",
            prometheus=SourceRange(
                start=datetime(2026, 9, 2, 3, 28, tzinfo=timezone.utc),
                end=datetime(2026, 9, 2, 3, 29, tzinfo=timezone.utc),
            ),
            tempo=None,
        ),
        ExperimentSummary(
            project="proj-2",
            experiment="exp-1",
            prometheus=SourceRange(
                start=datetime(2026, 9, 2, 3, 30, tzinfo=timezone.utc),
                end=datetime(2026, 9, 2, 3, 31, tzinfo=timezone.utc),
            ),
            tempo=None,
        ),
    ]

    output = format_summaries(summaries)

    assert output.count("proj-1") == 1
    assert output.count("proj-2") == 1
    assert any(line.startswith("|         +") for line in output.splitlines())
