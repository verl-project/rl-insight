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

"""Terminal display helpers for RL-Insight server commands."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .dependencies import ServiceStatus


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a compact ASCII table."""
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _row(values: Sequence[Any]) -> str:
        return (
            "| "
            + " | ".join(
                str(value).ljust(widths[idx]) for idx, value in enumerate(values)
            )
            + " |"
        )

    lines = [_line(), _row(headers), _line()]
    lines.extend(_row(row) for row in rendered_rows)
    lines.append(_line())
    return "\n".join(lines)


def format_logo() -> str:
    """Render the RL-Insight startup logo."""
    return "\n".join(
        [
            "      ____  __     ____           _       __    __",
            "     / __ \\/ /    /  _/___  _____(_)___ _/ /_  / /_",
            "    / /_/ / /     / // __ \\/ ___/ / __ `/ __ \\/ __/",
            "   / _, _/ /_____/ // / / (__  ) / /_/ / / / / /_",
            "  /_/ |_/_____/___/_/ /_/____/_/\\__, /_/ /_/\\__/",
            "                               /____/",
            "  RL-Insight Server",
        ]
    )


def format_panel(title: str, rows: Sequence[tuple[str, Any]]) -> str:
    """Render a compact key-value panel for startup status."""
    label_width = max([len(label) for label, _ in rows] + [0])
    value_width = max([len(str(value)) for _, value in rows] + [0])
    width = max(len(title), label_width + value_width + 5)
    border = "-" * (width + 2)
    lines = [
        f"+{border}+",
        f"| {title.ljust(width)} |",
        f"+{border}+",
    ]
    for label, value in rows:
        text = f"{label.ljust(label_width)}  {value}"
        lines.append(f"| {text.ljust(width)} |")
    lines.append(f"+{border}+")
    return "\n".join(lines)


def active_state_rows(state: dict[str, Any]) -> list[list[Any]]:
    services = state.get("services", [])
    return [
        [
            service.get("name", "unknown"),
            service.get("pid", ""),
            service.get("log_file", ""),
        ]
        for service in services
    ]


def dependency_rows(statuses: Iterable[ServiceStatus]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for status in statuses:
        if not status.enabled:
            state = "disabled"
        elif status.binary is None:
            state = "missing"
        elif status.detail.startswith("version <"):
            state = "too old"
        elif status.detail == "version unknown":
            state = "unknown"
        else:
            state = "ok"

        version = status.version or status.detail or "-"
        if status.current_version and status.min_version:
            version = f"{status.current_version} (need >= {status.min_version})"

        rows.append(
            [
                status.name,
                state,
                status.source or "-",
                version,
                str(status.binary) if status.binary else "-",
            ]
        )
    return rows
