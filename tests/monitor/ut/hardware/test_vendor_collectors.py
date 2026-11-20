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

"""Unit tests for the NVIDIA and Ascend vendor adapters (all device
tooling is faked: no NVML driver or ``npu-smi`` binary is required)."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from rl_insight.hardware.base import HardwareCollector, Sample
from rl_insight.hardware.vendor import DeviceCollector
from rl_insight.hardware.vendor.ascend import (
    AscendNpuCollector,
    list_chip_identities,
    parse_npu_smi_output,
)
from rl_insight.hardware.vendor.nvidia import NvidiaGpuCollector

_GIB = 1024**3

# ---------------------------------------------------------------------------
# NVIDIA NVML adapter.
# ---------------------------------------------------------------------------


class _FakeRates:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _FakeMemory:
    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


class _FakeAccelerator(HardwareCollector):
    """Vendor backend double used by the device collector tests."""

    def __init__(self, name: str, available: bool, samples: list[Sample]) -> None:
        self.name = name
        self._available = available
        self._samples = samples

    def discover(self) -> bool:
        return self._available

    def collect(self) -> list[Sample]:
        return list(self._samples)


class _ExplodingAccelerator(HardwareCollector):
    """Vendor backend double whose probe fails."""

    name = "broken"

    def discover(self) -> bool:
        raise RuntimeError("probe failed")

    def collect(self) -> list[Sample]:
        return []


class FakePynvml:
    """pynvml double exposing two uniform devices."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, device_count: int = 2) -> None:
        self._device_count = device_count

    def nvmlInit(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return self._device_count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle-{index}"

    def nvmlDeviceGetName(self, _handle: str) -> bytes:
        return b"NVIDIA A100"  # bytes, like older pynvml releases

    def nvmlSystemGetDriverVersion(self) -> str:
        return "550.54.15"

    def nvmlDeviceGetUtilizationRates(self, _handle: str) -> _FakeRates:
        return _FakeRates(42)

    def nvmlDeviceGetMemoryInfo(self, _handle: str) -> _FakeMemory:
        return _FakeMemory(10 * _GIB, 40 * _GIB)

    def nvmlDeviceGetTemperature(self, _handle: str, _sensor: int) -> int:
        return 65

    def nvmlDeviceGetPowerUsage(self, _handle: str) -> int:
        return 250_000  # milliwatts


def test_device_collector_should_auto_detect_available_accelerators() -> None:
    npu_sample = Sample(
        "rlinsight_npu_chip_info_utilization", 1.0, {"device_vendor": "ascend"}
    )
    gpu_sample = Sample(
        "rlinsight_gpu_utilization_percent", 1.0, {"device_vendor": "nvidia"}
    )
    collector = DeviceCollector(
        [
            _FakeAccelerator("npu", True, [npu_sample]),
            _FakeAccelerator("gpu", False, [gpu_sample]),
        ]
    )
    assert collector.discover() is True
    assert collector.collect() == [npu_sample]

    both = DeviceCollector(
        [
            _FakeAccelerator("npu", True, [npu_sample]),
            _FakeAccelerator("gpu", True, [gpu_sample]),
            _ExplodingAccelerator(),
        ]
    )
    assert both.discover() is True
    assert both.collect() == [npu_sample, gpu_sample]

    unavailable = DeviceCollector(
        [
            _FakeAccelerator("npu", False, []),
            _FakeAccelerator("gpu", False, []),
        ]
    )
    assert unavailable.discover() is False
    assert unavailable.collect() == []


def test_gpu_collector_should_report_metrics_scope_devices_and_require_a_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = NvidiaGpuCollector(pynvml_module=FakePynvml()).collect()

    by_name: dict[str, list[Any]] = {}
    for sample in samples:
        by_name.setdefault(sample.name, []).append(sample)
    assert len(samples) == 12  # 6 metrics x 2 devices
    assert by_name["rlinsight_gpu_utilization_percent"][0].value == 42.0
    assert by_name["rlinsight_gpu_memory_used_bytes"][0].value == 10 * _GIB
    assert by_name["rlinsight_gpu_memory_utilization_percent"][0].value == 25.0
    assert by_name["rlinsight_gpu_power_watts"][0].value == pytest.approx(250.0)

    info = by_name["rlinsight_gpu_info"][0]
    assert info.labels["device_name"] == "NVIDIA A100"
    assert info.labels["driver_version"] == "550.54.15"

    # Integer CUDA_VISIBLE_DEVICES entries map to NVML indexes.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 2")
    visible = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=4)).collect()
    assert {sample.labels["device_index"] for sample in visible} == {1, 2}

    # UUID / MIG / PCI-id forms cannot be mapped, so every device stays visible.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    visible = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=4)).collect()
    assert {sample.labels["device_index"] for sample in visible} == {0, 1, 2, 3}

    # Missing or unusable NVML disables discovery instead of raising.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert NvidiaGpuCollector().discover() is False
    assert (
        NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=0)).discover() is False
    )

    class BrokenInit(FakePynvml):
        def nvmlInit(self) -> None:
            raise RuntimeError("no driver")

    assert NvidiaGpuCollector(pynvml_module=BrokenInit()).discover() is False


# ---------------------------------------------------------------------------
# Ascend npu-smi adapter.
# ---------------------------------------------------------------------------

# Atlas A2: one chip per NPU, so the chip row carries only ``Chip``.
_A2_OUTPUT = """
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B3               | OK            | 97.2                 49                      0    / 0                |
| 0                         | 0000:C1:00.0  | 10                   0    / 0                3416 / 65536            |
+===========================+===============+======================================================================+
| 1     910B3               | OK            | 91.4                 48                      0    / 0                |
| 0                         | 0000:C2:00.0  | 20                   0    / 0                3412 / 65536            |
+===========================+===============+======================================================================+
"""

# Atlas A3: two chips per NPU. The second chip of a card reports ``-`` power.
_A3_OUTPUT = """
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip  Phy-ID              | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910C                | OK            | 162.4                33                      0    / 0                |
| 0     0                   | 0000:9D:00.0  | 10                   0    / 0                2964 / 65536            |
+------------------------------------------------------------------------------------------------------------------+
| 0     910C                | OK            | -                    36                      0    / 0                |
| 1     1                   | 0000:9F:00.0  | 20                   0    / 0                2882 / 65536            |
+===========================+===============+======================================================================+
| 1     910C                | OK            | 154.1                33                      0    / 0                |
| 0     2                   | 0000:99:00.0  | 30                   0    / 0                2909 / 65536            |
+------------------------------------------------------------------------------------------------------------------+
| 1     910C                | OK            | -                    33                      0    / 0                |
| 1     3                   | 0000:9B:00.0  | 40                   0    / 0                2870 / 65536            |
+===========================+===============+======================================================================+
"""

# Atlas A5 / Ascend 950: one chip per NPU and a physical chip id.
_A5_OUTPUT = """
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip  Phy-ID              | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     Ascend950PR         | OK            | 203.3                62                      0    / 0                |
| 0     0                   | 0000:AF:00.0  | 37                   0    / 0                5242 / 131072           |
+===========================+===============+======================================================================+
| 1     Ascend950PR         | OK            | 198.0                63                      0    / 0                |
| 0     1                   | 0000:B1:00.0  | 45                   0    / 0                5243 / 131072           |
+===========================+===============+======================================================================+
"""


# Older Atlas A2 releases label the second chip column ``Device`` instead of
# ``Phy-ID``; its value is not a global ordinal and must not be used as the
# device index.
_A2_310B_OUTPUT = """
+--------------------------------------------------------------------------------------------------------+
| npu-smi 23.0.0                                       Version: 23.0.0                                   |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B1                 | OK              | 10.4          66                15    / 15           |
| 0       0                     | NA              | 0            957   / 10810                           |
+===============================+=================+======================================================+
| 1       310B1                 | OK              | 10.0          65                15    / 15           |
| 0       0                     | NA              | 0            900   / 10810                           |
+===============================+=================+======================================================+
"""


_USAGES_OUTPUT = """
        NPU ID                         : 0
        Chip ID                        : 0
        HBM Capacity(MB)               : 65536
        HBM Usage Rate(%)              : 5
        HBM Bandwidth Usage Rate(%)    : 37
"""


def _by_index(samples: list[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        result.setdefault(str(sample.labels["device_index"]), {})[sample.name] = sample
    return result


@pytest.mark.parametrize(
    ("output", "expected", "expected_ids"),
    [
        (
            _A2_OUTPUT,
            {
                "0": ("910B3", 10.0, 97.2, 49.0, 3416.0, 65536.0),
                "1": ("910B3", 20.0, 91.4, 48.0, 3412.0, 65536.0),
            },
            [("0", "0", "0"), ("1", "0", "1")],
        ),
        (
            _A3_OUTPUT,
            {
                "0": ("910C", 10.0, 162.4, 33.0, 2964.0, 65536.0),
                "1": ("910C", 20.0, None, 36.0, 2882.0, 65536.0),
                "2": ("910C", 30.0, 154.1, 33.0, 2909.0, 65536.0),
                "3": ("910C", 40.0, None, 33.0, 2870.0, 65536.0),
            },
            [("0", "0", "0"), ("0", "1", "1"), ("1", "0", "2"), ("1", "1", "3")],
        ),
        (
            _A5_OUTPUT,
            {
                "0": ("Ascend950PR", 37.0, 203.3, 62.0, 5242.0, 131072.0),
                "1": ("Ascend950PR", 45.0, 198.0, 63.0, 5243.0, 131072.0),
            },
            [("0", "0", "0"), ("1", "0", "1")],
        ),
    ],
    ids=["a2", "a3", "a5"],
)
def test_parse_should_support_atlas_layouts(
    output: str,
    expected: dict[str, tuple[str, float, float | None, float, float, float]],
    expected_ids: list[tuple[str, str, str]],
) -> None:
    samples = parse_npu_smi_output(output)
    indexed = _by_index(samples)

    assert set(indexed) == set(expected)
    for index, (name, util, power, temp, used, total) in expected.items():
        metrics = indexed[index]
        assert metrics["rlinsight_npu_chip_info_utilization"].value == util
        assert metrics["rlinsight_npu_chip_info_temperature"].value == temp
        assert metrics["rlinsight_npu_chip_info_health_status"].value == 1.0
        # NPU memory metrics use megabytes, matching NPU Exporter.
        assert metrics["rlinsight_npu_chip_info_hbm_used_memory"].value == used
        assert metrics[
            "rlinsight_npu_chip_info_hbm_utilization"
        ].value == pytest.approx(used / total * 100)
        if power is None:
            assert "rlinsight_npu_chip_info_power" not in metrics
        else:
            assert metrics["rlinsight_npu_chip_info_power"].value == power
        assert metrics["rlinsight_npu_chip_info_name"].labels["chip_name"] == name
        assert (
            metrics["rlinsight_npu_chip_info_name"].labels["npu_smi_version"]
            == "26.1.1"
        )

    assert list_chip_identities(output) == expected_ids


def test_parse_should_handle_old_a2_device_column_and_malformed_output() -> None:
    # Older Atlas A2 releases use a non-ordinal "Device" column, but the rows
    # still map to sequential per-NPU device indexes.
    samples = parse_npu_smi_output(_A2_310B_OUTPUT)
    assert {sample.labels["device_index"] for sample in samples} == {"0", "1"}
    assert list_chip_identities(_A2_310B_OUTPUT) == [
        ("0", "0", "0"),
        ("1", "0", "1"),
    ]

    assert parse_npu_smi_output("") == []
    assert parse_npu_smi_output("totally unrelated text\n") == []
    # NPU rows without their chip rows cannot produce device samples.
    output = "\n".join(line for line in _A2_OUTPUT.splitlines() if "0000:" not in line)
    assert parse_npu_smi_output(output) == []


def test_npu_collector_should_collect_hbm_bandwidth_and_scope_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = AscendNpuCollector(
        runner=lambda _binary, _timeout: _A2_OUTPUT,
        usages_runner=lambda _binary, _card, _chip: _USAGES_OUTPUT,
    )
    indexed = _by_index(collector.collect())
    assert len(indexed) == 2
    for index in ("0", "1"):
        assert (
            indexed[index]["rlinsight_npu_chip_info_hbm_bandwidth_utilization"].value
            == 37.0
        )

    # ASCEND_RT_VISIBLE_DEVICES scopes collection to the listed chips.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    visible = collector.collect()
    assert {sample.labels["device_index"] for sample in visible} == {"1"}
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")

    # A missing or exploding per-chip query only drops that chip's bandwidth.
    for failure in ("returns-none", "raises"):

        def usages(
            binary: str, card: str, chip: str, _failure: str = failure
        ) -> str | None:
            _ = (binary, chip)
            if card != "0":
                return _USAGES_OUTPUT
            if _failure == "raises":
                raise RuntimeError("npu-smi crashed")
            return None

        indexed = _by_index(
            AscendNpuCollector(
                runner=lambda _binary, _timeout: _A2_OUTPUT, usages_runner=usages
            ).collect()
        )
        assert "rlinsight_npu_chip_info_hbm_bandwidth_utilization" not in indexed["0"]
        assert (
            indexed["1"]["rlinsight_npu_chip_info_hbm_bandwidth_utilization"].value
            == 37.0
        ), failure


def test_npu_collector_should_handle_tooling_and_runner_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which", lambda _name: None
    )
    assert AscendNpuCollector().discover() is False

    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which",
        lambda _name: "/usr/local/sbin/npu-smi",
    )
    assert AscendNpuCollector().discover() is True
    assert AscendNpuCollector(runner=lambda _binary, _timeout: None).collect() == []
