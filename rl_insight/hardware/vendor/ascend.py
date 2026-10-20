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

"""Ascend NPU collector backed by the ``npu-smi`` command line tool."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Callable

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    present_samples,
    register_collector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["AscendNpuCollector", "parse_npu_smi_output"]

_NPU_SMI_BINARY = "npu-smi"
_NPU_SMI_TIMEOUT_SECONDS = 5.0
_VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"

# ``npu-smi info`` table rows are pipe-delimited column groups, e.g.
# ``| 0       910B1 | OK | 67.8   42   0 / 0 |`` (NPU row) and
# ``| 0       910B1 | 0000:C1:00.0 | 60   855 / 15171 |`` (chip row).
_BUS_ID_PATTERN = re.compile(
    r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$"
)
_VERSION_PATTERN = re.compile(r"npu-smi\s+([0-9][0-9.]*)")
_MB_TO_BYTES = 1024 * 1024

# Runner signature: (binary, timeout_seconds) -> stdout text, or None on failure.
NpuSmiRunner = Callable[[str, float], "str | None"]


def _run_npu_smi(binary: str, timeout: float) -> str | None:
    """Run ``npu-smi info`` once and return its stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            [binary, "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _split_row(line: str) -> list[str] | None:
    """Split one ``npu-smi`` table row into stripped column groups."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    groups = [group.strip() for group in stripped[1:-1].split("|")]
    return groups if any(groups) else None


def _first_number(text: str) -> float | None:
    """Parse the first token of ``text`` as a float, or ``None``."""
    tokens = text.split()
    if not tokens:
        return None
    try:
        return float(tokens[0])
    except ValueError:
        return None


def _hbm_megabytes(text: str) -> tuple[float, float] | None:
    """Parse the HBM ``used / total`` megabyte pair from one column group.

    Newer ``npu-smi`` releases (e.g. 26.1.1) merge several ``x / y`` fields
    into one chip column: ``AICore(%) Memory-Usage(MB) HBM-Usage(MB)``.
    The HBM pair is always the last one, so the last match wins; on older
    releases with a single pair, that pair is the HBM usage.
    """
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    if not matches:
        return None
    used, total = matches[-1]
    return float(used), float(total)


def parse_npu_smi_output(text: str) -> list[Sample]:
    """Parse ``npu-smi info`` output into per-chip samples.

    Rows are paired into (NPU row, chip row) per device; rows that do not
    pair up, and fields that fail to parse, are skipped so one malformed
    line never drops the other devices.

    Args:
        text: Raw ``npu-smi info`` stdout.

    Returns:
        Samples for every successfully parsed device (possibly empty).
    """
    version_match = _VERSION_PATTERN.search(text)
    version = version_match.group(1) if version_match else ""

    rows: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        groups = _split_row(line)
        if not groups or len(groups) < 3:
            continue
        head = groups[0].split()
        if not head or not head[0].isdigit():
            continue  # header or separator row
        kind = "chip" if _BUS_ID_PATTERN.match(groups[1]) else "npu"
        rows.append((kind, groups))

    samples: list[Sample] = []
    for position, (kind, groups) in enumerate(rows):
        if kind != "npu":
            continue
        chip = _paired_chip(rows, position)
        if chip is None:
            continue
        # The NPU number and product name live in the NPU row; the chip row
        # carries the per-chip number, which is always 0 on single-chip
        # devices and must not be used as the device index.
        npu_head = groups[0].split()
        index = npu_head[0]
        chip_name = " ".join(npu_head[1:])
        labels = {"device_vendor": "ascend", "device_index": index}

        power = _first_number(groups[2])
        tail = groups[2].split()
        temperature = _first_number(tail[1]) if len(tail) >= 2 else None
        aicore = _first_number(chip[2])
        hbm = _hbm_megabytes(chip[2])

        samples.extend(
            present_samples(
                [
                    optional_sample(
                        "rlinsight_npu_utilization_percent",
                        aicore,
                        labels,
                        "Ascend NPU AICore utilization percent",
                    ),
                    optional_sample(
                        "rlinsight_npu_power_watts",
                        power,
                        labels,
                        "Ascend NPU power draw in watts",
                    ),
                    optional_sample(
                        "rlinsight_npu_temperature_celsius",
                        temperature,
                        labels,
                        "Ascend NPU temperature in celsius",
                    ),
                    optional_sample(
                        "rlinsight_npu_memory_used_bytes",
                        None if hbm is None else hbm[0] * _MB_TO_BYTES,
                        labels,
                        "Ascend NPU HBM used in bytes",
                    ),
                    optional_sample(
                        "rlinsight_npu_memory_total_bytes",
                        None if hbm is None else hbm[1] * _MB_TO_BYTES,
                        labels,
                        "Ascend NPU HBM total in bytes",
                    ),
                    optional_sample(
                        "rlinsight_npu_memory_utilization_percent",
                        None if hbm is None or not hbm[1] else hbm[0] / hbm[1] * 100.0,
                        labels,
                        "Ascend NPU HBM utilization percent",
                    ),
                ]
            )
        )

        # Keep the label dimension fixed across devices: a device whose
        # product name is missing must still carry the key (with an empty
        # value), otherwise ``rlinsight_npu_info`` would register two label
        # sets and the hub's registry would reject the batch as a duplicate
        # time series, dropping every sample after the offending one.
        info_labels = dict(labels)
        info_labels["chip_name"] = chip_name
        info_labels["npu_smi_version"] = version
        samples.append(
            Sample(
                name="rlinsight_npu_info",
                value=1.0,
                labels=info_labels,
                documentation="Ascend NPU identity (always 1; see labels)",
            )
        )
    return samples


def _paired_chip(rows: list[tuple[str, list[str]]], position: int) -> list[str] | None:
    """Return the chip row immediately following the NPU row at ``position``."""
    if position + 1 >= len(rows) or rows[position + 1][0] != "chip":
        return None
    return rows[position + 1][1]


def _visible_device_indexes() -> set[str] | None:
    """Device indexes this process is allowed to use, or ``None`` unrestricted.

    ``npu-smi`` reports every chip on the node regardless of the caller's
    environment, so a job pinned to a subset (``ASCEND_RT_VISIBLE_DEVICES``)
    would otherwise publish samples for cards it cannot even reach. Honouring
    the variable keeps the job-scoped in-process metrics limited to the cards
    the job actually uses. An unset or empty value means no restriction.
    """
    raw = os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()} or None


class AscendNpuCollector(HardwareCollector):
    """Per-chip Ascend NPU metrics through the ``npu-smi`` CLI.

    ``npu-smi`` is a host-side management tool that may be absent inside
    training containers; ``discover`` only checks that the binary is on
    ``PATH`` so missing tooling degrades to an unavailable status instead
    of failing. All subprocess execution is bounded by a timeout.

    Reporting is scoped to ``ASCEND_RT_VISIBLE_DEVICES`` when that variable
    is set, so a job using a subset of the node's cards only publishes
    samples for those cards.
    """

    name = "npu"

    def __init__(
        self,
        binary: str = _NPU_SMI_BINARY,
        runner: NpuSmiRunner | None = None,
    ) -> None:
        """
        Args:
            binary: ``npu-smi`` executable name or path.
            runner: Injected subprocess runner for tests; defaults to one
                bounded ``npu-smi info`` invocation per round.
        """
        self._binary = binary
        self._runner: NpuSmiRunner = runner or _run_npu_smi

    def discover(self) -> bool:
        """Report whether ``npu-smi`` is usable, without probing devices.

        Only tooling availability is checked: a node whose binary exists but
        reports no cards still counts as available, and ``collect`` then
        yields no samples. ``rlinsight_hardware_collector_available``
        therefore reflects whether the collector set up its tooling, not
        whether the node has NPUs. Checking the binary only also keeps a
        transient ``npu-smi`` failure from permanently disabling the
        collector before the first round.
        """
        if shutil.which(self._binary) is None:
            logger.info(
                "[rl-insight] npu-smi was not found on PATH; Ascend NPU "
                "metrics are unavailable on this node."
            )
            return False
        return True

    def collect(self) -> list[Sample]:
        output = self._runner(self._binary, _NPU_SMI_TIMEOUT_SECONDS)
        if output is None:
            return []
        samples = parse_npu_smi_output(output)
        visible = _visible_device_indexes()
        if visible is None:
            return samples
        return [
            sample
            for sample in samples
            if str(sample.labels.get("device_index", "")) in visible
        ]


register_collector(AscendNpuCollector.name, AscendNpuCollector)
