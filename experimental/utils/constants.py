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

"""String constants for monitor backends and event kinds."""

from __future__ import annotations


class MonitorBackend:
    """Supported monitor transport implementations (``create_monitor_client`` dispatches on ``type``)."""

    RAY = "ray"


class MonitorEventKind:
    """String ``kind`` field on events sent through ``MonitorHubActor.apply_event``."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TRACE = "trace"
