# Copyright (c) 2025 verl-project authors.
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

from typing import List

from rl_insight.data.base import ValidationRule


class PathExistsRule(ValidationRule):
    _error_message: List[str] = []

    def check(self, data) -> bool:
        if not hasattr(data, "path"):
            self._error_message = ["Data object does not have 'path' attribute"]
            return False
        if not data.path.exists():
            self._error_message = [f"Source path does not exist: {data.path}"]
            return False
        return True

    @property
    def error_message(self) -> List[str]:
        return self._error_message
