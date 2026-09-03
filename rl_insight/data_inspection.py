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
"""Offline inspection of persisted Prometheus and Tempo data."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from prometheus_client.parser import text_string_to_metric_families
from pyarrow import parquet
from tqdm import tqdm

from .utils.constants import MonitorPaths

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class SourceRange:
    """Time range for one data source."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class ExperimentSummary:
    """One project/experiment pair found in persisted data."""

    project: str
    experiment: str
    prometheus: SourceRange | None
    tempo: SourceRange | None


def inspect_data_directory(
    data_dir: str | Path,
    *,
    promtool_bin: str | Path | None = None,
) -> list[ExperimentSummary]:
    """Return project/experiment summaries from a persisted RL-Insight data directory."""
    resolved_dir = Path(data_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {resolved_dir}")

    prometheus_ranges = _inspect_prometheus(resolved_dir, promtool_bin=promtool_bin)
    tempo_ranges = _inspect_tempo(resolved_dir)
    keys = sorted(set(prometheus_ranges) | set(tempo_ranges))
    return [
        ExperimentSummary(
            project=project,
            experiment=experiment,
            prometheus=prometheus_ranges.get((project, experiment)),
            tempo=tempo_ranges.get((project, experiment)),
        )
        for project, experiment in keys
    ]


def format_summaries(summaries: Sequence[ExperimentSummary]) -> str:
    """Render experiment summaries as a terminal table."""
    headers = ["Project", "Experiment", "Prometheus", "Tempo"]
    rows = [
        [
            summary.project,
            summary.experiment,
            _format_range(
                summary.prometheus.start if summary.prometheus else None,
                summary.prometheus.end if summary.prometheus else None,
            ),
            _format_range(
                summary.tempo.start if summary.tempo else None,
                summary.tempo.end if summary.tempo else None,
            ),
        ]
        for summary in summaries
    ]
    return _format_table(headers, rows)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a grouped table for experiment summaries."""
    rendered_rows = [[_cell_lines(cell) for cell in row] for row in rows]
    widths = [_display_width(header) for header in headers]
    for row in rendered_rows:
        for idx, cell_lines in enumerate(row):
            widths[idx] = max(
                [widths[idx], *(_display_width(line) for line in cell_lines)]
            )

    def _line(skip_first: bool = False) -> str:
        if skip_first:
            return (
                "|"
                + " " * (widths[0] + 2)
                + "+"
                + "+".join("-" * (width + 2) for width in widths[1:])
                + "+"
            )
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _row(values: Sequence[Sequence[str]], *, show_first: bool = True) -> str:
        row_cells = list(values)
        if not show_first:
            row_cells[0] = [""]
        lines = []
        height = max(len(cell_lines) for cell_lines in row_cells)
        for line_idx in range(height):
            cells = []
            for idx, cell_lines in enumerate(row_cells):
                text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                cells.append(_pad(text, widths[idx]))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    group_starts = [True]
    group_starts.extend(
        previous[0][0] != current[0][0]
        for previous, current in zip(rendered_rows, rendered_rows[1:])
    )

    lines = [_line(), _row([[header] for header in headers]), _line()]
    for idx, row in enumerate(rendered_rows):
        lines.append(_row(row, show_first=group_starts[idx]))
        if idx < len(rendered_rows) - 1:
            same_group = rendered_rows[idx + 1][0][0] == row[0][0]
            lines.append(_line(skip_first=same_group))
    lines.append(_line())
    return "\n".join(lines)


def _cell_lines(cell: Any) -> list[str]:
    lines = str(cell).splitlines()
    return lines if lines else [""]


def _display_width(text: str) -> int:
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        if unicodedata.east_asian_width(character) in {"F", "W"}:
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _display_width(text))


def _inspect_prometheus(
    data_dir: Path,
    *,
    promtool_bin: str | Path | None,
) -> dict[tuple[str, str], SourceRange]:
    prometheus_dir = data_dir / "prometheus"
    if not prometheus_dir.is_dir():
        return {}

    ranges: dict[tuple[str, str], SourceRange] = {}
    for line in _promtool_lines(prometheus_dir, promtool_bin=promtool_bin):
        for pair, timestamp in _parse_openmetrics(line):
            _update_range(ranges, pair, timestamp)
    return ranges


def _promtool_lines(
    prometheus_dir: Path,
    *,
    promtool_bin: str | Path | None,
) -> Iterator[str]:
    binary = _find_promtool(promtool_bin)
    estimated_total = _prometheus_sample_count(prometheus_dir)
    with tempfile.TemporaryDirectory(
        prefix=".rl-insight-promtool-",
        dir=prometheus_dir.parent,
    ) as sandbox:
        command = [
            str(binary),
            "--experimental",
            "tsdb",
            "dump-openmetrics",
            f"--sandbox-dir-root={sandbox}",
            str(prometheus_dir),
        ]
        with tempfile.TemporaryFile(mode="w+") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
            )
            progress = tqdm(
                total=estimated_total,
                desc="Prometheus samples",
                unit="sample",
                disable=False,
            )
            try:
                for line in process.stdout:
                    progress.update(1)
                    yield line
                if progress.n > progress.total:
                    progress.total = progress.n
            finally:
                progress.close()
                if process.stdout is not None:
                    process.stdout.close()
                process.wait()
                stderr_file.seek(0)
                stderr = stderr_file.read()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    stderr=stderr,
                )


def _prometheus_sample_count(prometheus_dir: Path) -> int:
    return sum(
        json.loads(meta_file.read_text(encoding="utf-8"))
        .get("stats", {})
        .get("numSamples", 0)
        for meta_file in prometheus_dir.glob("*/meta.json")
    )


def _find_promtool(explicit: str | Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"promtool binary does not exist: {path}")
        return path

    found = shutil.which("promtool")
    if found:
        return Path(found).resolve()

    service_root = MonitorPaths.STATE_ROOT / "services" / "prometheus"
    candidates = sorted(
        path for path in service_root.rglob("promtool") if path.is_file()
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "promtool was not found on PATH or under ~/.rl-insight/services/prometheus"
    )


def _parse_openmetrics(output: str) -> Iterator[tuple[tuple[str, str], datetime]]:
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        raw_timestamp = float(line.rsplit(None, 1)[-1])
        for family in text_string_to_metric_families(line):
            for sample in family.samples:
                project = sample.labels.get("project")
                experiment = sample.labels.get("experiment_name")
                if not project or not experiment:
                    continue
                yield (project, experiment), _timestamp_to_datetime(raw_timestamp)


def _inspect_tempo(data_dir: Path) -> dict[tuple[str, str], SourceRange]:
    tempo_dir = data_dir / "tempo"
    if not tempo_dir.is_dir():
        return {}

    ranges: dict[tuple[str, str], SourceRange] = {}
    columns = ["StartTimeUnixNano", "EndTimeUnixNano", "rs"]
    parquet_files = sorted(tempo_dir.rglob("data.parquet"))
    total_rows = sum(
        parquet.ParquetFile(parquet_file).metadata.num_rows
        for parquet_file in parquet_files
    )
    with tqdm(
        total=total_rows,
        desc="Tempo traces",
        unit="trace",
        disable=False,
    ) as progress:
        for parquet_file in parquet_files:
            parquet_file_handle = parquet.ParquetFile(parquet_file)
            for batch in parquet_file_handle.iter_batches(
                batch_size=256,
                columns=columns,
            ):
                for row in batch.to_pylist():
                    start = _nanoseconds_to_datetime(row["StartTimeUnixNano"])
                    end = _nanoseconds_to_datetime(row["EndTimeUnixNano"])
                    for pair in _tempo_pairs(row):
                        _update_range(ranges, pair, start, end=end)
                progress.update(batch.num_rows)
    return ranges


def _tempo_pairs(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for resource_spans in row.get("rs") or []:
        resource_attributes = resource_spans.get("Resource", {}).get("Attrs") or []
        pairs.update(_attributes_to_pairs(resource_attributes))
        for scope_spans in resource_spans.get("ss") or []:
            for span in scope_spans.get("Spans") or []:
                pairs.update(_attributes_to_pairs(span.get("Attrs") or []))
    return pairs


def _attributes_to_pairs(
    attributes: Iterable[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    values: dict[str, str] = {}
    for attribute in attributes:
        key = attribute.get("Key")
        value = attribute.get("Value")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if key and value:
            values[str(key)] = str(value)
    project = values.get("project")
    experiment = values.get("experiment_name")
    return {(project, experiment)} if project and experiment else set()


def _update_range(
    ranges: dict[tuple[str, str], SourceRange],
    pair: tuple[str, str],
    start: datetime,
    *,
    end: datetime | None = None,
) -> None:
    finish = end if end is not None else start
    existing = ranges.get(pair)
    if existing is None:
        ranges[pair] = SourceRange(start=start, end=finish)
        return
    ranges[pair] = SourceRange(
        start=min(existing.start, start),
        end=max(existing.end, finish),
    )


def _timestamp_to_datetime(timestamp: float) -> datetime:
    return _EPOCH + timedelta(seconds=timestamp)


def _nanoseconds_to_datetime(timestamp: int) -> datetime:
    return _EPOCH + timedelta(microseconds=timestamp // 1_000)


def _format_range(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "-"
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}～{end:%H:%M}"
    return f"{start:%Y-%m-%d %H:%M}～\n{end:%Y-%m-%d %H:%M}"
