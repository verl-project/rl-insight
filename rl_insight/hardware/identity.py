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

"""Best-effort node and run identity for hardware monitoring labels."""

from __future__ import annotations

import socket

__all__ = ["detect_node_label", "detect_run_id_label", "identity_labels"]


def detect_node_label() -> str:
    """Best-effort node identity: Ray node IP in a Ray job, else hostname."""
    try:
        import ray

        if ray.is_initialized():
            return str(ray.util.get_node_ip_address())
    except Exception:
        pass
    return socket.gethostname()


def detect_run_id_label() -> str:
    """Best-effort run identity: the current Ray job id, or an empty string."""
    try:
        import ray

        if ray.is_initialized():
            job_id = ray.get_runtime_context().get_job_id()
            return job_id if isinstance(job_id, str) else job_id.hex()
    except Exception:
        pass
    return ""


def identity_labels() -> dict[str, str]:
    """Return the node and run ids detected on the current process."""
    labels: dict[str, str] = {}
    node = detect_node_label()
    if node:
        labels["node"] = node
    run_id = detect_run_id_label()
    if run_id:
        labels["run_id"] = run_id
    return labels
