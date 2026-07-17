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

from pathlib import Path

from recipe.data.rules import (
    DataValidationError,
    PathExistsRule,
    MstxJsonFileExistsRule,
    MstxJsonFieldValidRule,
    TorchJsonFileExistsRule,
    TorchJsonFieldValidRule,
    NvtxJsonFileExistsRule,
    NvtxJsonFieldValidRule,
    AscendMemoryFileExistsRule,
    AscendMemoryFieldValidRule,
)
from recipe.data.verl_log_rules import VerlLogExistRule, VerlLogKeyParamsRule
from tests.recipe.data.test_paths import (
    MSTX_PROFILE_PATH,
    NVTX_PROFILE_PATH,
    TORCH_PROFILE_PATH,
    MEMORY_DATA_PATH,
)


def test_path_exists_rule_accepts_existing_directory():
    rule = PathExistsRule()
    assert rule.check(str(MSTX_PROFILE_PATH)) is True


def test_path_exists_rule_rejects_non_string_input():
    rule = PathExistsRule()
    assert rule.check({"path": "x"}) is False
    assert "not a path" in rule.error_message


def test_path_exists_rule_rejects_missing_directory():
    rule = PathExistsRule()
    assert rule.check("C:/definitely/not/exist/path") is False


def test_data_validation_error_string_includes_error_details():
    err = DataValidationError("Data validation failed", ["line1", "line2"])
    text = str(err)
    assert "Data validation failed" in text
    assert "line1" in text
    assert "line2" in text


def test_mstx_jsonfile_exists():
    path_rule = PathExistsRule()
    file_rule = MstxJsonFileExistsRule()
    assert path_rule.check(str(MSTX_PROFILE_PATH)) is True
    assert file_rule.check(str(MSTX_PROFILE_PATH)) is True


def test_mstx_jsonfile_exists_accepts_path_object():
    file_rule = MstxJsonFileExistsRule()
    assert file_rule.check(MSTX_PROFILE_PATH) is True


def test_mstx_jsonfile_exists_with_fake_path():
    file_rule = MstxJsonFileExistsRule()
    fake_path = "fake_path"
    assert file_rule.check(fake_path) is False


def test_mstx_json_fields_valid():
    path_rule = PathExistsRule()
    filed_rule = MstxJsonFieldValidRule()
    assert path_rule.check(str(MSTX_PROFILE_PATH)) is True
    assert filed_rule.check(str(MSTX_PROFILE_PATH)) is True


def test_mstx_json_fields_valid_with_fake_path():
    filed_rule = MstxJsonFieldValidRule()
    fake_path = "fake_path"
    assert filed_rule.check(fake_path) is False


def test_verl_log_exist_accepts_verl_named_log(tmp_path):
    log = tmp_path / "train_verl_worker.log"
    log.write_text("verl job\n", encoding="utf-8")
    rule = VerlLogExistRule()
    assert rule.check(str(log)) is True


def test_verl_log_exist_rejects_directory(tmp_path):
    rule = VerlLogExistRule()
    assert rule.check(str(tmp_path)) is False
    assert "not a directory" in rule.error_message


def test_verl_log_exist_rejects_non_log_extension(tmp_path):
    log = tmp_path / "run_verl.txt"
    log.write_text("verl job\n", encoding="utf-8")
    rule = VerlLogExistRule()
    assert rule.check(str(log)) is False


def test_verl_log_key_params_requires_keywords(tmp_path):
    log = tmp_path / "debug_verl.log"
    log.write_text(
        "\n".join(
            [
                "python3 -m verl.trainer.main_ppo",
                "Training Progress:   0%|          | 1/100 [00:01<00:00,  1.00s/it]",
                "(TaskRunner pid=1) step=0 - training/global_step:1 - training/epoch:0",
                "(TaskRunner pid=1) 'critic/score/mean': 0.1",
                "(TaskRunner pid=1) 'actor/loss': 0.2",
                "(TaskRunner pid=1) 'critic/rewards/mean': 0.3",
                "(TaskRunner pid=1) 'response_length/mean': 128.0",
                "(TaskRunner pid=1) 'actor/grad_norm': 0.99",
                "(TaskRunner pid=1) 'actor/lr': 1e-06",
                "(TaskRunner pid=1) 'actor/entropy': 0.5",
            ]
        ),
        encoding="utf-8",
    )
    rule = VerlLogKeyParamsRule(
        required_keywords=VerlLogKeyParamsRule.DEFAULT_REQUIRED_KEYWORDS
    )
    assert rule.check(str(log)) is True


def test_verl_log_key_params_fails_when_missing_keyword(tmp_path):
    log = tmp_path / "x_verl.log"
    log.write_text(
        "\n".join(
            [
                "python3 -m verl.trainer.main_ppo",
                "(TaskRunner pid=1) 'critic/score/mean': 0.1",
                "(TaskRunner pid=1) 'actor/loss': 0.2",
            ]
        ),
        encoding="utf-8",
    )
    rule = VerlLogKeyParamsRule(
        required_keywords=VerlLogKeyParamsRule.DEFAULT_REQUIRED_KEYWORDS
    )
    assert rule.check(str(log)) is False
    assert "response_length/mean" in rule.error_message


def test_torch_jsonfile_exists():
    path_rule = PathExistsRule()
    file_rule = TorchJsonFileExistsRule()
    assert path_rule.check(str(TORCH_PROFILE_PATH)) is True
    assert file_rule.check(str(TORCH_PROFILE_PATH)) is True


def test_torch_jsonfile_exists_accepts_path_object():
    file_rule = TorchJsonFileExistsRule()
    assert file_rule.check(TORCH_PROFILE_PATH) is True


def test_torch_jsonfile_exists_with_fake_path():
    file_rule = TorchJsonFileExistsRule()
    fake_path = "fake_path"
    assert file_rule.check(fake_path) is False


def test_torch_json_fields_valid():
    path_rule = PathExistsRule()
    filed_rule = TorchJsonFieldValidRule()
    assert path_rule.check(str(TORCH_PROFILE_PATH)) is True
    assert filed_rule.check(str(TORCH_PROFILE_PATH)) is True


def test_nvtx_jsonfile_exists():
    path_rule = PathExistsRule()
    file_rule = NvtxJsonFileExistsRule()
    assert path_rule.check(str(NVTX_PROFILE_PATH)) is True
    assert file_rule.check(str(NVTX_PROFILE_PATH)) is True


def test_nvtx_jsonfile_exists_accepts_path_object():
    file_rule = NvtxJsonFileExistsRule()
    assert file_rule.check(NVTX_PROFILE_PATH) is True


def test_nvtx_jsonfile_exists_with_fake_path():
    file_rule = NvtxJsonFileExistsRule()
    fake_path = "fake_path"
    assert file_rule.check(fake_path) is False


def test_gmm_data_rule_accepts_path_object():
    from recipe.data.rules import GmmDataRule

    rule = GmmDataRule()
    assert rule.check(Path("data/recipe/gmm_data")) is True


def test_nvtx_json_fields_valid():
    path_rule = PathExistsRule()
    field_rule = NvtxJsonFieldValidRule()
    assert path_rule.check(str(NVTX_PROFILE_PATH)) is True
    assert field_rule.check(str(NVTX_PROFILE_PATH)) is True


# ---------------------------------------------------------------------------
# AscendMemory rules
# ---------------------------------------------------------------------------

_MEMORY_CSV_COLUMNS = [
    "Name",
    "Size(KB)",
    "Allocation Time(us)",
    "Release Time(us)",
    "Active Release Time(us)",
    "Duration(us)",
    "Active Duration(us)",
    "Allocation Total Allocated(MB)",
    "Allocation Total Reserved(MB)",
    "Allocation Total Active(MB)",
    "Release Total Allocated(MB)",
    "Release Total Reserved(MB)",
    "Release Total Active(MB)",
    "Stream Ptr",
    "Device Type",
]


def _write_memory_csv(path, rows=None):
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_MEMORY_CSV_COLUMNS)
        for r in rows or [
            [
                "aten::empty",
                "1.0",
                "1000.0",
                "",
                "",
                "",
                "",
                "10.0",
                "20.0",
                "10.0",
                "",
                "",
                "",
                "6",
                "NPU:0",
            ]
        ]:
            writer.writerow(r)


def _create_memory_layout(tmp_path, with_csv=True, with_trace=True, rank_id=0):
    import json

    ascend_pt = tmp_path / "20250101_120000_ascend_pt"
    ascend_pt.mkdir(parents=True)
    (ascend_pt / f"profiler_info_{rank_id}.json").write_text(
        json.dumps({"rank_id": str(rank_id)}), encoding="utf-8"
    )
    (ascend_pt / "profiler_metadata.json").write_text(
        json.dumps({"ENV_VARIABLES": {}}), encoding="utf-8"
    )
    output_dir = ascend_pt / "ASCEND_PROFILER_OUTPUT"
    output_dir.mkdir()
    if with_csv:
        _write_memory_csv(str(output_dir / "operator_memory.csv"))
    if with_trace:
        with open(output_dir / "trace_view.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "cat": "cpu_op",
                        "name": "aten::empty",
                        "ts": "1000",
                        "dur": 0,
                        "args": {"Call stack": "foo()"},
                    }
                ],
                f,
            )
    return tmp_path


def test_ascend_memory_file_exists_on_sample_data():
    rule = AscendMemoryFileExistsRule()
    assert rule.check(str(MEMORY_DATA_PATH)) is True


def test_ascend_memory_file_exists_accepts_path_object():
    rule = AscendMemoryFileExistsRule()
    assert rule.check(MEMORY_DATA_PATH) is True


def test_ascend_memory_file_exists_rejects_fake_path():
    rule = AscendMemoryFileExistsRule()
    assert rule.check("fake_path") is False


def test_ascend_memory_file_exists_rejects_missing_csv(tmp_path):
    root = _create_memory_layout(tmp_path, with_csv=False)
    rule = AscendMemoryFileExistsRule()
    assert rule.check(str(root)) is False
    assert "operator_memory.csv" in rule.error_message


def test_ascend_memory_file_exists_rejects_missing_trace(tmp_path):
    root = _create_memory_layout(tmp_path, with_trace=False)
    rule = AscendMemoryFileExistsRule()
    assert rule.check(str(root)) is False
    assert "trace_view.json" in rule.error_message


def test_ascend_memory_field_valid_on_sample_data():
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(MEMORY_DATA_PATH)) is True


def test_ascend_memory_field_valid_accepts_path_object():
    rule = AscendMemoryFieldValidRule()
    assert rule.check(MEMORY_DATA_PATH) is True


def test_ascend_memory_field_valid_rejects_fake_path():
    rule = AscendMemoryFieldValidRule()
    assert rule.check("fake_path") is False


def test_ascend_memory_field_valid_rejects_bad_profiler_info(tmp_path):
    root = _create_memory_layout(tmp_path)
    ascend_pt = root / "20250101_120000_ascend_pt"
    (ascend_pt / "profiler_info_0.json").write_text("{not json", encoding="utf-8")
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "profiler_info" in rule.error_message


def test_ascend_memory_field_valid_rejects_profiler_info_without_rank_id(tmp_path):
    import json

    root = _create_memory_layout(tmp_path)
    ascend_pt = root / "20250101_120000_ascend_pt"
    (ascend_pt / "profiler_info_0.json").write_text(
        json.dumps({"config": {}}), encoding="utf-8"
    )
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "rank_id" in rule.error_message


def test_ascend_memory_field_valid_rejects_bad_metadata(tmp_path):
    root = _create_memory_layout(tmp_path)
    ascend_pt = root / "20250101_120000_ascend_pt"
    (ascend_pt / "profiler_metadata.json").write_text("{not json", encoding="utf-8")
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "profiler_metadata.json" in rule.error_message


def test_ascend_memory_field_valid_rejects_csv_missing_columns(tmp_path):
    root = _create_memory_layout(tmp_path)
    output_dir = root / "20250101_120000_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    # overwrite csv with only one column
    (output_dir / "operator_memory.csv").write_text(
        "Name\naten::empty\n", encoding="utf-8"
    )
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "operator_memory.csv" in rule.error_message


def test_ascend_memory_field_valid_rejects_empty_csv(tmp_path):
    root = _create_memory_layout(tmp_path)
    output_dir = root / "20250101_120000_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    import csv

    with open(
        output_dir / "operator_memory.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(_MEMORY_CSV_COLUMNS)  # header only, no data rows
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "no data rows" in rule.error_message


def test_ascend_memory_field_valid_rejects_empty_trace_view(tmp_path):
    root = _create_memory_layout(tmp_path)
    output_dir = root / "20250101_120000_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    (output_dir / "trace_view.json").write_text("[]", encoding="utf-8")
    rule = AscendMemoryFieldValidRule()
    assert rule.check(str(root)) is False
    assert "trace_view.json" in rule.error_message
