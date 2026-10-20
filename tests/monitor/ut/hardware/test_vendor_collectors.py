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

import re
import sys
from typing import Any

import pytest

from rl_insight.hardware.base import COLLECTOR_REGISTRY
from rl_insight.hardware.vendor.ascend import (
    AscendNpuCollector,
    parse_npu_smi_output,
)
from rl_insight.hardware.vendor.nvidia import NvidiaGpuCollector

_GIB = 1024**3
_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# NVIDIA NVML adapter.
# ---------------------------------------------------------------------------


class _FakeRates:
    """Stand-in for ``nvmlDeviceGetUtilizationRates`` return value."""

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _FakeMemory:
    """Stand-in for ``nvmlDeviceGetMemoryInfo`` return value."""

    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


class FakePynvml:
    """pynvml double exposing two uniform devices."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, device_count: int = 2) -> None:
        self._device_count = device_count
        self.init_calls = 0

    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return self._device_count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle-{index}"

    def nvmlDeviceGetName(self, _handle: str) -> bytes | str:
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


def test_gpu_collector_should_register_under_config_name() -> None:
    assert COLLECTOR_REGISTRY["gpu"] is NvidiaGpuCollector


def test_gpu_discover_should_follow_capability_and_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pynvml = FakePynvml(device_count=2)
    assert NvidiaGpuCollector(pynvml_module=pynvml).discover() is True
    assert pynvml.init_calls == 1

    # A None entry makes ``import pynvml`` raise ImportError deterministically.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert NvidiaGpuCollector().discover() is False

    class BrokenInit(FakePynvml):
        def nvmlInit(self) -> None:
            raise RuntimeError("no driver")

    assert NvidiaGpuCollector(pynvml_module=BrokenInit()).discover() is False
    assert (
        NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=0)).discover() is False
    )


def test_gpu_collect_should_report_all_fields_per_device() -> None:
    collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=2))

    samples = collector.collect()

    by_name: dict[str, list[Any]] = {}
    for sample in samples:
        by_name.setdefault(sample.name, []).append(sample)
    assert len(samples) == 14  # 7 metrics x 2 devices
    assert sorted(
        sample.labels["device_index"]
        for sample in by_name["rlinsight_gpu_utilization_percent"]
    ) == [0, 1]
    assert by_name["rlinsight_gpu_utilization_percent"][0].value == 42.0
    assert by_name["rlinsight_gpu_memory_used_bytes"][0].value == 10 * _GIB
    assert by_name["rlinsight_gpu_memory_total_bytes"][0].value == 40 * _GIB
    assert by_name["rlinsight_gpu_memory_utilization_percent"][
        0
    ].value == pytest.approx(25.0)
    assert by_name["rlinsight_gpu_temperature_celsius"][0].value == 65.0
    assert by_name["rlinsight_gpu_power_watts"][0].value == pytest.approx(250.0)

    info = by_name["rlinsight_gpu_info"][0]
    assert info.value == 1.0
    assert info.labels["device_vendor"] == "nvidia"
    assert info.labels["device_name"] == "NVIDIA A100"
    assert info.labels["driver_version"] == "550.54.15"


def test_gpu_collect_should_decode_bytes_and_accept_str_names() -> None:
    class StrNamePynvml(FakePynvml):
        def nvmlDeviceGetName(self, _handle: str) -> str:
            return "H100"  # str, like newer pynvml releases

    samples = NvidiaGpuCollector(pynvml_module=StrNamePynvml()).collect()

    info = next(sample for sample in samples if sample.name == "rlinsight_gpu_info")
    assert info.labels["device_name"] == "H100"


def test_gpu_collect_should_omit_unsupported_fields_but_keep_the_rest() -> None:
    class NoTemperaturePynvml(FakePynvml):
        def nvmlDeviceGetTemperature(self, _handle: str, _sensor: int) -> int:
            raise RuntimeError("not supported")

    samples = NvidiaGpuCollector(pynvml_module=NoTemperaturePynvml()).collect()

    names = {sample.name for sample in samples}
    assert "rlinsight_gpu_temperature_celsius" not in names
    assert "rlinsight_gpu_utilization_percent" in names
    assert len(samples) == 12  # 6 metrics x 2 devices


def test_gpu_collect_should_skip_one_unhealthy_device_but_keep_others() -> None:
    class OneBrokenHandlePynvml(FakePynvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            if index == 1:
                raise RuntimeError("device lost")
            return f"handle-{index}"

    samples = NvidiaGpuCollector(pynvml_module=OneBrokenHandlePynvml()).collect()

    indexes = {sample.labels["device_index"] for sample in samples}
    assert indexes == {0}
    assert len(samples) == 7


def test_gpu_info_should_keep_a_fixed_label_dimension() -> None:
    # One device resolves no name while the other does: both info samples must
    # expose the same label keys (empty value for the unknown one) so the
    # family keeps a single label dimension.
    class OneUnnamedPynvml(FakePynvml):
        def nvmlDeviceGetName(self, handle: str) -> str:
            if handle == "handle-1":
                raise RuntimeError("name unavailable")
            return "NVIDIA A100"

    samples = NvidiaGpuCollector(pynvml_module=OneUnnamedPynvml()).collect()

    info = [sample for sample in samples if sample.name == "rlinsight_gpu_info"]
    assert len(info) == 2
    assert len({tuple(sorted(sample.labels)) for sample in info}) == 1
    by_index = {sample.labels["device_index"]: sample for sample in info}
    assert by_index[0].labels["device_name"] == "NVIDIA A100"
    assert by_index[1].labels["device_name"] == ""


def test_gpu_collect_should_scope_to_integer_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``CUDA_VISIBLE_DEVICES`` uses plain indexes here, which map onto NVML
    # indexes, so a job pinned to a subset only reports those cards.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 2")
    collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=4))

    samples = collector.collect()

    assert {sample.labels["device_index"] for sample in samples} == {1, 2}


def test_gpu_collect_should_not_scope_on_non_integer_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # UUID / MIG / PCI-id forms cannot be mapped to NVML indexes; report every
    # device instead of silently dropping all of them.
    for value in ("GPU-abc123", "MIG-abc", "0,GPU-xyz"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=2))

        samples = collector.collect()

        assert {sample.labels["device_index"] for sample in samples} == {0, 1}


# ---------------------------------------------------------------------------
# Ascend npu-smi adapter.
# ---------------------------------------------------------------------------

# Captured verbatim from ``npu-smi info`` (26.1.1) on an Atlas 910B node.
# The chip row merges AICore / DDR memory / HBM into one column group, the
# NPU number and product name live in the NPU row, and the output ends with
# a per-process table that must not produce samples.
_SAMPLE_OUTPUT = """
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B3               | OK            | 97.2                 49                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                3416 / 65536            |
+===========================+===============+======================================================================+
| 1     910B3               | OK            | 91.4                 48                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                3412 / 65536            |
+===========================+===============+======================================================================+
| 2     910B3               | OK            | 86.9                 48                      0    / 0                |
| 0                         | 0000:81:00.0  | 0                    0    / 0                3412 / 65536            |
+===========================+===============+======================================================================+
| 3     910B3               | OK            | 95.4                 48                      0    / 0                |
| 0                         | 0000:82:00.0  | 0                    0    / 0                3412 / 65536            |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 1                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 2                                                                              |
+===========================+===============+======================================================================+
| No running processes found in NPU 3                                                                              |
+===========================+===============+======================================================================+
"""


def _by_name_and_index(samples: list[Any]) -> dict[tuple[str, str], Any]:
    """Index samples by (metric name, device index) for convenient lookup."""
    return {
        (sample.name, str(sample.labels["device_index"])): sample for sample in samples
    }


def test_npu_collector_should_register_under_config_name() -> None:
    assert COLLECTOR_REGISTRY["npu"] is AscendNpuCollector


def test_parse_should_report_all_fields_per_npu() -> None:
    samples = parse_npu_smi_output(_SAMPLE_OUTPUT)

    # 7 metrics x 4 NPUs; the trailing process table must add nothing.
    assert len(samples) == 28
    indexed = _by_name_and_index(samples)

    # The NPU number (not the per-chip number) is the device index.
    assert {index for _, index in indexed} == {"0", "1", "2", "3"}

    device_0 = {name: indexed[(name, "0")] for name, index in indexed if index == "0"}
    assert device_0["rlinsight_npu_utilization_percent"].value == 0.0
    assert device_0["rlinsight_npu_power_watts"].value == 97.2
    assert device_0["rlinsight_npu_temperature_celsius"].value == 49.0
    assert device_0["rlinsight_npu_memory_used_bytes"].value == 3416 * _MB
    assert device_0["rlinsight_npu_memory_total_bytes"].value == 65536 * _MB
    assert device_0["rlinsight_npu_memory_utilization_percent"].value == (
        pytest.approx(3416 / 65536 * 100)
    )

    info = device_0["rlinsight_npu_info"]
    assert info.value == 1.0
    assert info.labels["device_vendor"] == "ascend"
    assert info.labels["chip_name"] == "910B3"
    assert info.labels["npu_smi_version"] == "26.1.1"

    assert indexed[("rlinsight_npu_power_watts", "1")].value == 91.4
    assert indexed[("rlinsight_npu_power_watts", "2")].value == 86.9
    assert indexed[("rlinsight_npu_power_watts", "3")].value == 95.4


def test_parse_should_return_empty_for_empty_or_malformed_output() -> None:
    assert parse_npu_smi_output("") == []
    assert parse_npu_smi_output("totally unrelated text\n") == []


def test_parse_should_skip_devices_without_a_chip_row() -> None:
    # NPU rows without their chip rows: cannot pair, so no samples at all.
    output = "\n".join(
        line for line in _SAMPLE_OUTPUT.splitlines() if "0000:" not in line
    )

    assert parse_npu_smi_output(output) == []


def test_parse_should_omit_unparseable_fields_but_keep_the_device() -> None:
    output = _SAMPLE_OUTPUT.replace("| 97.2 ", "| N/A ")
    output = output.replace(
        "| 0                    0    / 0                3416 / 65536",
        "| N/A                    0    / 0                3416 / 65536",
    )

    samples = parse_npu_smi_output(output)
    indexed = _by_name_and_index(samples)

    assert ("rlinsight_npu_power_watts", "0") not in indexed
    assert ("rlinsight_npu_utilization_percent", "0") not in indexed
    assert ("rlinsight_npu_temperature_celsius", "0") in indexed
    assert ("rlinsight_npu_power_watts", "1") in indexed


def test_npu_info_should_keep_a_fixed_label_dimension() -> None:
    # Device 1 reports no product name: both info samples must expose the same
    # label keys (empty value for the unknown name) so the family keeps a
    # single label dimension and the hub registry accepts every device.
    output = re.sub(r"\| 1\s+910B3", "| 1", _SAMPLE_OUTPUT, count=1)

    samples = parse_npu_smi_output(output)

    info = [sample for sample in samples if sample.name == "rlinsight_npu_info"]
    assert len(info) == 4
    assert len({tuple(sorted(sample.labels)) for sample in info}) == 1
    by_index = {sample.labels["device_index"]: sample for sample in info}
    assert by_index["0"].labels["chip_name"] == "910B3"
    assert by_index["1"].labels["chip_name"] == ""


def test_npu_discover_should_follow_binary_availability(
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


def test_npu_collect_should_parse_runner_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    collector = AscendNpuCollector(runner=lambda _binary, _timeout: _SAMPLE_OUTPUT)

    samples = collector.collect()

    assert len(samples) == 28
    assert all(sample.labels["device_vendor"] == "ascend" for sample in samples)


def test_npu_collect_should_scope_to_the_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``npu-smi`` lists every chip on the node, so a job pinned to a subset
    # must only publish samples for the cards it can actually use.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2, 3")
    collector = AscendNpuCollector(runner=lambda _binary, _timeout: _SAMPLE_OUTPUT)

    samples = collector.collect()

    assert len(samples) == 14  # 7 metrics x 2 visible NPUs
    assert {str(sample.labels["device_index"]) for sample in samples} == {"2", "3"}


def test_npu_collect_should_treat_an_empty_visible_list_as_unrestricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", " , ")
    collector = AscendNpuCollector(runner=lambda _binary, _timeout: _SAMPLE_OUTPUT)

    assert len(collector.collect()) == 28


def test_npu_collect_should_return_empty_when_runner_fails() -> None:
    collector = AscendNpuCollector(runner=lambda _binary, _timeout: None)

    assert collector.collect() == []
