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

"""Ascend NPU collector backed by a single ``npu-smi info`` invocation."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    drop_missing_samples,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["AscendNpuCollector", "list_chip_identities", "parse_npu_smi_output"]

_NPU_SMI_BINARY = "npu-smi"
_NPU_SMI_TIMEOUT_SECONDS = 5.0
# The per-chip usage query only serves the HBM bandwidth field; bound it so one
# slow call cannot stall the round.
_NPU_SMI_USAGES_TIMEOUT_SECONDS = 2.0
# Query chips concurrently; cap the pool to avoid one process per chip on large
# nodes.
_USAGES_MAX_WORKERS = 8
_VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"
_HBM_BANDWIDTH_LABEL = "hbm bandwidth usage rate"

# ``npu-smi info`` table rows are pipe-delimited column groups, e.g.
# ``| 0       910B1 | OK | 67.8   42   0 / 0 |`` (NPU row) and
# ``| 0       910B1 | 0000:C1:00.0 | 60   855 / 15171 |`` (chip row).
_VERSION_PATTERN = re.compile(r"npu-smi\s+([0-9][0-9.]*)")
_HEALTH_VALUES = frozenset({"OK", "WARNING", "ALARM", "CRITICAL", "UNKNOWN"})

# Runner signature: (binary, timeout_seconds) -> stdout text, or None on failure.
NpuSmiRunner = Callable[[str, float], str | None]
# Runner signature: (binary, card, chip) -> stdout text, or None on failure.
NpuUsagesRunner = Callable[[str, str, str], str | None]


def _run_npu_smi(binary: str, args: list[str], timeout: float) -> str | None:
    """Run one ``npu-smi`` command and return its stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("[rl-insight] npu-smi %s failed: %s", " ".join(args), exc)
        return None
    return result.stdout


def _run_npu_smi_info(binary: str, timeout: float) -> str | None:
    """Run ``npu-smi info`` once and return its stdout."""
    return _run_npu_smi(binary, ["info"], timeout)


def _run_npu_smi_usages(binary: str, card: str, chip: str) -> str | None:
    """Run ``npu-smi info -t usages`` for one chip and return its stdout."""
    return _run_npu_smi(
        binary,
        ["info", "-t", "usages", "-i", card, "-c", chip],
        _NPU_SMI_USAGES_TIMEOUT_SECONDS,
    )


def _split_row(line: str) -> list[str] | None:
    """Split one ``npu-smi`` table row into stripped column groups."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    groups = [group.strip() for group in stripped[1:-1].split("|")]
    return groups if any(groups) else None


def _parse_number(token: str) -> float | None:
    """Parse one token as a float, or ``None`` when it is not numeric."""
    try:
        return float(token)
    except ValueError:
        return None


def _first_number(text: str) -> float | None:
    """Parse the first token of ``text`` as a float, or ``None``."""
    tokens = text.split()
    if not tokens:
        return None
    return _parse_number(tokens[0])


def _power_and_temperature(text: str) -> tuple[float | None, float | None]:
    """Parse the power and temperature columns of an NPU row.

    Atlas A3/A5 reports ``-`` for the power of a second chip while temperature
    remains valid, so temperature is the first numeric field after power.
    """
    tokens = text.split()
    if not tokens:
        return None, None
    power = _parse_number(tokens[0])
    temperature = None
    for token in tokens[1:]:
        temperature = _parse_number(token)
        if temperature is not None:
            break
    return power, temperature


def _hbm_megabytes(text: str) -> tuple[float, float] | None:
    """Parse the HBM ``used / total`` megabyte pair from one column group.

    Newer ``npu-smi`` releases merge several ``x / y`` fields into one chip
    column; the HBM pair is the last one. Older releases have a single pair.
    """
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    if not matches:
        return None
    used, total = matches[-1]
    return float(used), float(total)


def _health_value(text: str) -> float | None:
    """Map the summary table's health state to 1 (healthy) or 0 (unhealthy)."""
    state = text.strip().upper()
    if state == "OK":
        return 1.0
    if state in _HEALTH_VALUES:
        return 0.0
    return None


def _is_npu_row(groups: list[str]) -> bool:
    """Return whether a summary row is an NPU row rather than a chip row.

    NPU rows carry the health state in the second column; chip rows carry a
    PCI bus id or ``NA``. The documented health values distinguish the two.
    """
    return groups[1].strip().upper() in _HEALTH_VALUES


def _field_number(text: str, label: str) -> float | None:
    """Return the numeric value of a ``label : value`` line, or ``None``."""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower().startswith(label):
            return _parse_number(value.strip())
    return None


def _pair_chips(text: str) -> Iterator[tuple[list[str], int, list[str], str]]:
    """Pair each NPU row with its chip rows.

    Yields ``(npu_columns, chip_position, chip_columns, device_index)`` per chip.
    """
    # Only ``Chip Phy-ID`` is a global device ordinal. Older Atlas A2 releases
    # expose ``Chip Device``, which is not, so they use the enumeration index.
    has_phy_id = "Phy-ID" in text

    rows: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        groups = _split_row(line)
        # Summary rows have three groups; the process table has more.
        if groups is None or len(groups) != 3:
            continue
        head = groups[0].split()
        if not head or not head[0].isdigit():
            continue  # header or separator row
        rows.append(("npu" if _is_npu_row(groups) else "chip", groups))

    flat_index = 0
    position = 0
    while position < len(rows):
        kind, groups = rows[position]
        if kind != "npu":
            position += 1
            continue

        chip_rows: list[list[str]] = []
        next_position = position + 1
        while next_position < len(rows) and rows[next_position][0] == "chip":
            chip_rows.append(rows[next_position][1])
            next_position += 1

        for chip_position, chip in enumerate(chip_rows):
            chip_head = chip[0].split()
            # ``Chip Phy-ID`` is present on A3/A5 and is the visible device
            # ordinal. A2 exposes ``Chip`` or ``Chip Device`` instead, neither
            # of which is global, so those layouts use the enumeration index.
            phy_id = (
                chip_head[1]
                if has_phy_id and len(chip_head) > 1 and chip_head[1].isdigit()
                else None
            )
            index = phy_id if phy_id is not None else str(flat_index)
            flat_index += 1
            yield groups, chip_position, chip, index

        position = next_position


def list_chip_identities(text: str) -> list[tuple[str, str, str]]:
    """Return ``(card, chip, device_index)`` for every chip of a summary table.

    ``card`` and ``chip`` are the identifiers accepted by the targeted
    ``npu-smi info -t usages -i <card> -c <chip>`` query, and ``device_index``
    is the ordinal used by the metric labels.
    """
    identities: list[tuple[str, str, str]] = []
    for npu_columns, _chip_position, chip_columns, index in _pair_chips(text):
        chip_head = chip_columns[0].split()
        identities.append((npu_columns[0].split()[0], chip_head[0], index))
    return identities


def parse_npu_smi_output(text: str) -> list[Sample]:
    """Parse ``npu-smi info`` output into per-chip samples.

    Supports Atlas A2 (one chip per NPU) and A3/A5 (several chips per NPU with
    a ``Chip Phy-ID`` column). The physical ID becomes ``device_index`` because
    it matches ``ASCEND_RT_VISIBLE_DEVICES``. Malformed rows are skipped.
    """
    version_match = _VERSION_PATTERN.search(text)
    version = version_match.group(1) if version_match else ""

    samples: list[Sample] = []
    for npu_columns, chip_position, chip, index in _pair_chips(text):
        npu_head = npu_columns[0].split()
        chip_name = " ".join(npu_head[1:])
        power, temperature = _power_and_temperature(npu_columns[2])
        health = _health_value(npu_columns[1])

        labels = {"device_vendor": "ascend", "device_index": index}
        aicore = _first_number(chip[2])
        hbm = _hbm_megabytes(chip[2])
        hbm_used = hbm_total = hbm_utilization = None
        if hbm is not None:
            hbm_used, hbm_total = hbm
            hbm_utilization = hbm_used / hbm_total * 100.0 if hbm_total else None
        # On multi-chip cards the NPU row reports the card power once;
        # attributing it to every chip would over-count it.
        chip_power = power if chip_position == 0 else None

        samples.extend(
            drop_missing_samples(
                [
                    optional_sample(
                        "rlinsight_npu_chip_info_utilization",
                        aicore,
                        labels,
                        "Ascend NPU AICore utilization percent",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_power",
                        chip_power,
                        labels,
                        "Ascend NPU power draw in watts",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_temperature",
                        temperature,
                        labels,
                        "Ascend NPU temperature in celsius",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_health_status",
                        health,
                        labels,
                        "Ascend NPU health (1=healthy, 0=unhealthy)",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_hbm_used_memory",
                        hbm_used,
                        labels,
                        "Ascend NPU HBM used memory in megabytes",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_hbm_utilization",
                        hbm_utilization,
                        labels,
                        "Ascend NPU HBM utilization percent",
                    ),
                ]
            )
        )

        # Keep the label keys identical across devices: a missing product name
        # still carries ``chip_name`` so the hub sees one label schema.
        info_labels = dict(labels)
        info_labels["chip_name"] = chip_name
        info_labels["npu_smi_version"] = version
        samples.append(
            Sample(
                name="rlinsight_npu_chip_info_name",
                value=1.0,
                labels=info_labels,
                documentation="Ascend NPU identity (constant 1; see labels)",
            )
        )
    return samples


def _visible_device_indexes() -> set[str] | None:
    """Return the device indexes allowed by ``ASCEND_RT_VISIBLE_DEVICES``.

    ``npu-smi`` reports every chip on the node, so this filter keeps a job's
    metrics limited to the cards it can use. ``None`` means unrestricted.
    """
    raw = os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()} or None


class AscendNpuCollector(HardwareCollector):
    """Per-chip Ascend NPU metrics through the ``npu-smi`` CLI.

    Each round issues one bounded ``npu-smi info`` call and parses the summary
    table, so process cost does not grow with the card count. HBM bandwidth
    needs one ``npu-smi info -t usages`` call per chip; a failed call only drops
    that chip's bandwidth sample. Missing ``npu-smi`` marks the collector
    unavailable. Reporting is scoped to ``ASCEND_RT_VISIBLE_DEVICES``.
    """

    name = "npu"

    def __init__(
        self,
        binary: str = _NPU_SMI_BINARY,
        runner: NpuSmiRunner | None = None,
        usages_runner: NpuUsagesRunner | None = None,
    ) -> None:
        """``runner`` and ``usages_runner`` are injectable for tests."""
        self._binary = binary
        self._runner: NpuSmiRunner = runner or _run_npu_smi_info
        self._usages_runner: NpuUsagesRunner = usages_runner or _run_npu_smi_usages
        # Log only the first failure to avoid repeating it every round.
        self._failure_logged = False

    def discover(self) -> bool:
        """Return whether ``npu-smi`` is on ``PATH``.

        Only tooling availability is checked; a node without cards stays
        available and ``collect`` returns no samples. This avoids disabling the
        collector on a transient device failure before the first round.
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
            if not self._failure_logged:
                self._failure_logged = True
                logger.warning(
                    "[rl-insight] npu-smi info failed; skipping this NPU sampling "
                    "round. Further failures are logged at debug level."
                )
            return []
        samples = parse_npu_smi_output(output)
        self._failure_logged = False
        visible = _visible_device_indexes()
        samples.extend(self._collect_hbm_bandwidth(output, visible))
        if visible is None:
            return samples
        return [
            sample
            for sample in samples
            if str(sample.labels.get("device_index", "")) in visible
        ]

    def _collect_hbm_bandwidth(
        self, summary: str, visible: set[str] | None
    ) -> list[Sample]:
        """Sample HBM bandwidth with one ``npu-smi info -t usages`` call per chip.

        Chips are queried concurrently. A failed, slow, or timed-out call only
        drops that chip's bandwidth sample.
        """
        identities = [
            (card, chip, index)
            for card, chip, index in list_chip_identities(summary)
            if visible is None or index in visible
        ]
        if not identities:
            return []

        workers = min(_USAGES_MAX_WORKERS, len(identities))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rl-insight-npu-usages"
        ) as pool:
            # ``map`` keeps the sample order aligned with ``npu-smi`` output.
            samples = list(pool.map(self._sample_hbm_bandwidth, identities))
        return [sample for sample in samples if sample is not None]

    def _sample_hbm_bandwidth(self, identity: tuple[str, str, str]) -> Sample | None:
        """Query one chip's usage table and build its bandwidth sample."""
        card, chip, index = identity
        try:
            text = self._usages_runner(self._binary, card, chip)
        except Exception:
            # Contain a raising runner to this chip.
            logger.debug(
                "[rl-insight] npu-smi usages query failed for card %s chip %s",
                card,
                chip,
                exc_info=True,
            )
            return None
        if text is None:
            return None
        value = _field_number(text, _HBM_BANDWIDTH_LABEL)
        if value is None:
            return None
        return Sample(
            name="rlinsight_npu_chip_info_hbm_bandwidth_utilization",
            value=value,
            labels={"device_vendor": "ascend", "device_index": index},
            documentation="Ascend NPU HBM bandwidth utilization percent",
        )
