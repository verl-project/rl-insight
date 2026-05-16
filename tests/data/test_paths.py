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


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
MSTX_PROFILE_PATH = PROJECT_ROOT / "data/mstx_data/mstx_profile"
NVTX_PROFILE_PATH = PROJECT_ROOT / "data/nvtx_data/nvtx_profile"
TORCH_PROFILE_PATH = PROJECT_ROOT / "data/torch_data/torch_profile"
