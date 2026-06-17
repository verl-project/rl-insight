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

from recipe.parser.gmm_parser import GmmParser
from recipe.utils.schema import Constant


def test_gmm_path_parsing_is_cross_platform():
    parser = GmmParser({Constant.RANK_LIST: "all"})

    windows_style_path = (
        r"C:\workspace\gmm_dump\step_1\actor_update\rank0\dump_tensor_data"
        r"\NPU.npu_grouped_matmul.0.forward.kwargs.group_list.pt"
    )

    assert parser._extract_rank_id_from_path(windows_style_path) == 0
    assert parser._extract_step_from_path(windows_style_path) == 1
    assert parser._training_step_from_path(windows_style_path) == 1


def test_gmm_normalize_path_text_returns_posix():
    parser = GmmParser({Constant.RANK_LIST: "all"})
    assert parser._normalize_path_text(r"C:\workspace\gmm_dump\step_1") == (
        "C:/workspace/gmm_dump/step_1"
    )
