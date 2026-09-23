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

"""Direct-emit (push) monitor backend.

The push backend forwards metric/trace events in-process to one or more
:class:`~rl_insight.client.push.sinks.base.PushSink` drivers, without a Ray
collector actor or a self-hosted server. Sinks are loaded from configuration via
dotted-path factories, so all deployment-specific logic stays outside this
package.
"""

from __future__ import annotations
