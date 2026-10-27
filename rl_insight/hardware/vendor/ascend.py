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
from typing import Callable, Iterator

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    present_samples,
    register_collector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["AscendNpuCollector", "list_chip_identities", "parse_npu_smi_output"]

_NPU_SMI_BINARY = "npu-smi"
_NPU_SMI_TIMEOUT_SECONDS = 5.0
# The per-chip usage query only serves the HBM bandwidth field, so a slow or
# hung query must not hold up the rest of the round.
_NPU_SMI_USAGES_TIMEOUT_SECONDS = 2.0
_VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"
_HBM_BANDWIDTH_LABEL = "hbm bandwidth usage rate"

# ``npu-smi info`` table rows are pipe-delimited column groups, e.g.
# ``| 0       910B1 | OK | 67.8   42   0 / 0 |`` (NPU row) and
# ``| 0       910B1 | 0000:C1:00.0 | 60   855 / 15171 |`` (chip row).
_VERSION_PATTERN = re.compile(r"npu-smi\s+([0-9][0-9.]*)")
_HEALTH_VALUES = frozenset({"OK", "WARNING", "ALARM", "CRITICAL", "UNKNOWN"})
_HEALTHY_STATES = frozenset({"OK"})

# Runner signature: (binary, timeout_seconds) -> stdout text, or None on failure.
NpuSmiRunner = Callable[[str, float], "str | None"]
# Runner signature: (binary, card, chip) -> stdout text, or None on failure.
NpuUsagesRunner = Callable[[str, str, str], "str | None"]


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
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("[rl-insight] npu-smi info failed: %s", exc)
        return None
    return result.stdout


def _run_npu_smi_usages(binary: str, card: str, chip: str) -> str | None:
    """Run ``npu-smi info -t usages`` for one chip and return its stdout."""
    try:
        result = subprocess.run(
            [binary, "info", "-t", "usages", "-i", card, "-c", chip],
            capture_output=True,
            text=True,
            timeout=_NPU_SMI_USAGES_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("[rl-insight] npu-smi usages failed: %s", exc)
        return None
    return result.stdout


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
    """Parse the ``Power(W) Temp(C)`` columns of an NPU row.

    Atlas A3/A5 reports ``-`` for the power of the second chip of a card
    while the temperature remains valid, so the temperature is taken from the
    first numeric field after the power field rather than from a fixed index.
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

    Newer ``npu-smi`` releases (e.g. 26.1.1) merge several ``x / y`` fields
    into one chip column: ``AICore(%) Memory-Usage(MB) HBM-Usage(MB)``.
    The HBM pair is always the last one, so the last match wins; on older
    releases with a single pair, that pair is the HBM usage. Atlas A3 chips
    use the same layout, while older releases may omit the spaces around
    ``/``.
    """
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    if not matches:
        return None
    used, total = matches[-1]
    return float(used), float(total)


def _health_value(text: str) -> float | None:
    """Map the summary table's health state to 1 (healthy) or 0 (unhealthy)."""
    state = text.strip().upper()
    if state in _HEALTHY_STATES:
        return 1.0
    if state in _HEALTH_VALUES:
        return 0.0
    return None


def _is_npu_row(groups: list[str]) -> bool:
    """Whether a three-column summary row is an NPU row rather than a chip row.

    NPU rows carry the health state in the second column. Chip rows carry the
    PCI bus id there, or ``NA`` on releases that do not expose it. The health
    states are limited to the documented values so that ``NA`` is classified
    as a chip row. Relying on the health column also covers the Atlas A3/A5
    layout, where several NPU rows share one NPU number and each is followed
    by a chip row.
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
    """Pair every NPU row with its chip rows in an ``npu-smi info`` table.

    Args:
        text: Raw ``npu-smi info`` stdout.

    Yields:
        ``(npu_columns, chip_position, chip_columns, device_index)`` for every
        chip row. ``npu_columns`` is the NPU row that owns the chip, and
        ``chip_position`` is the ordinal of the chip within that NPU row.
    """
    # Only ``Chip Phy-ID`` is a globally unique device ordinal. Older Atlas A2
    # releases expose ``Chip Device`` in that position, whose value is not a
    # global chip index, so those layouts use the enumeration index instead.
    has_phy_id = "Phy-ID" in text

    rows: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        groups = _split_row(line)
        # Summary rows have exactly three column groups; the trailing process
        # table has more and must not produce samples.
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
            # ``Chip Phy-ID`` is present on A3/A5 and is the globally visible
            # device ordinal. A2 exposes ``Chip`` (or ``Chip Device`` on older
            # 310B releases), neither of which is a global ordinal, so the
            # enumeration index is used there.
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

    The parser supports the summary layouts of Atlas A2, A3 and A5:

    - A2 exposes one chip per NPU; the chip row carries ``Chip`` only.
    - A3 and A5 may expose several chips per NPU and add ``Chip Phy-ID`` to
      the chip row. The physical ID is used as ``device_index`` because it
      matches ``ASCEND_RT_VISIBLE_DEVICES`` and the NPU Exporter numbering.
    - Rows that do not pair up, and fields that fail to parse, are skipped so
      that one malformed line never drops the other devices.

    Args:
        text: Raw ``npu-smi info`` stdout.

    Returns:
        Samples for every successfully parsed device (possibly empty).
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
        # On multi-chip cards the NPU row reports the card power once;
        # attributing it to every chip would over-count it.
        chip_power = power if chip_position == 0 else None

        samples.extend(
            present_samples(
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
                        None if hbm is None else hbm[0],
                        labels,
                        "Ascend NPU HBM used memory in megabytes",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_hbm_total_memory",
                        None if hbm is None else hbm[1],
                        labels,
                        "Ascend NPU HBM total memory in megabytes",
                    ),
                    optional_sample(
                        "rlinsight_npu_chip_info_hbm_utilization",
                        None if hbm is None or not hbm[1] else hbm[0] / hbm[1] * 100.0,
                        labels,
                        "Ascend NPU HBM utilization percent",
                    ),
                ]
            )
        )

        # Keep the label dimension fixed across devices: a device whose product
        # name is missing must still carry the key (with an empty value),
        # otherwise ``rlinsight_npu_chip_info_name`` would register two label
        # sets and the hub's registry would reject the batch as a duplicate
        # time series.
        info_labels = dict(labels)
        info_labels["chip_name"] = chip_name
        info_labels["npu_smi_version"] = version
        samples.append(
            Sample(
                name="rlinsight_npu_chip_info_name",
                value=1.0,
                labels=info_labels,
                documentation="Ascend NPU identity (always 1; see labels)",
            )
        )
    return samples


def _visible_device_indexes() -> set[str] | None:
    """Device indexes this process is allowed to use, or ``None`` unrestricted.

    ``npu-smi`` reports every chip on the node regardless of the caller's
    environment, so a job pinned to a subset (``ASCEND_RT_VISIBLE_DEVICES``)
    would otherwise publish samples for cards it cannot even reach. Honouring
    the variable keeps the job-scoped metrics limited to the cards the job
    actually uses. An unset or empty value means no restriction.
    """
    raw = os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()
    if not raw:
        return None
    return {token.strip() for token in raw.split(",") if token.strip()} or None


class AscendNpuCollector(HardwareCollector):
    """Per-chip Ascend NPU metrics through the ``npu-smi`` CLI.

    One bounded ``npu-smi info`` call is made per sampling round; the summary
    table is parsed directly and no helper process is kept alive between
    rounds. The summary table is parsed instead of issuing targeted per-chip
    subcommands, so the process cost of a round stays constant regardless of
    the number of cards. HBM bandwidth utilization is the exception: the
    summary does not carry it, so one bounded ``npu-smi info -t usages`` call
    is made per chip and failures only drop that chip's sample. The parser
    supports the Atlas A2, A3 and A5 summary layouts.
    ``npu-smi`` may be absent inside training containers, so ``discover`` only
    checks that the binary is on ``PATH`` and missing tooling degrades to an
    unavailable status instead of failing.

    Reporting is scoped to ``ASCEND_RT_VISIBLE_DEVICES`` when that variable
    is set, so a job using a subset of the node's cards only publishes
    samples for those cards.
    """

    name = "npu"

    def __init__(
        self,
        binary: str = _NPU_SMI_BINARY,
        runner: NpuSmiRunner | None = None,
        usages_runner: NpuUsagesRunner | None = None,
    ) -> None:
        """
        Args:
            binary: ``npu-smi`` executable name or path.
            runner: Injected subprocess runner for tests; defaults to one
                bounded ``npu-smi info`` invocation per round.
            usages_runner: Injected per-chip usage runner for tests; defaults
                to one bounded ``npu-smi info -t usages`` call per chip.
        """
        self._binary = binary
        self._runner: NpuSmiRunner = runner or _run_npu_smi
        self._usages_runner: NpuUsagesRunner = usages_runner or _run_npu_smi_usages
        # A persistently failing npu-smi must not spam the log every round.
        self._failure_logged = False

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
            if not self._failure_logged:
                self._failure_logged = True
                logger.warning(
                    "[rl-insight] npu-smi info failed; skipping this NPU sampling "
                    "round. Further failures are logged at debug level."
                )
            return []
        samples = parse_npu_smi_output(output)
        samples.extend(self._collect_hbm_bandwidth(output))
        visible = _visible_device_indexes()
        if visible is None:
            return samples
        return [
            sample
            for sample in samples
            if str(sample.labels.get("device_index", "")) in visible
        ]

    def _collect_hbm_bandwidth(self, summary: str) -> list[Sample]:
        """Sample the per-chip HBM bandwidth usage rate.

        The summary table does not expose this field, so one bounded
        ``npu-smi info -t usages`` call is made per chip. A failed or slow
        call only drops that chip's bandwidth sample, never the rest of the
        round.
        """
        visible = _visible_device_indexes()
        samples: list[Sample] = []
        for card, chip, index in list_chip_identities(summary):
            if visible is not None and index not in visible:
                continue
            text = self._usages_runner(self._binary, card, chip)
            if text is None:
                continue
            value = _field_number(text, _HBM_BANDWIDTH_LABEL)
            if value is None:
                continue
            samples.append(
                Sample(
                    name="rlinsight_npu_chip_info_hbm_bandwidth_utilization",
                    value=value,
                    labels={"device_vendor": "ascend", "device_index": index},
                    documentation="Ascend NPU HBM bandwidth utilization percent",
                )
            )
        return samples


register_collector(AscendNpuCollector.name, AscendNpuCollector)
