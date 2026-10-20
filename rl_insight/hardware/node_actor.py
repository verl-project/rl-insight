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

"""Per-node Ray actor that runs one hardware monitor on its scheduled node."""

from __future__ import annotations

import logging
from typing import Any

import ray
from omegaconf import DictConfig

from .coordinator import HardwareMonitor
from .identity import identity_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["NodeActorSet", "NodeHardwareActor"]


@ray.remote
class NodeHardwareActor:
    """Runs one host-level hardware monitor on the node it is scheduled to.

    Exactly one actor is created per Ray node (via a hard ``NodeAffinity``
    scheduling strategy and a node-scoped actor name), so host samples are
    collected once per node no matter how many processes called
    ``rl_insight.init``. Samples are forwarded through the configured monitor
    client: the job's Ray hub for ``server.backend=ray``, or the configured
    push sinks for ``server.backend=push``.
    """

    def __init__(
        self,
        conf: DictConfig,
        labels: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """
        Args:
            conf: Merged monitor config used to create the configured monitor
                client (the job's Ray hub for ``ray``, push sinks for ``push``).
            labels: Init-level labels (project, experiment name); the ``node``
                and ``run_id`` labels are detected here, on the actor's own
                node, so multi-node jobs get one distinct series per node.
            params: Resolved hardware params: ``interval_seconds``,
                ``collector_names``, and optional ``collector_options``.
        """
        # Imported here so importing this module never pulls the client stack.
        from ..client import create_monitor_client

        actor_labels = {**labels, **identity_labels()}
        node = actor_labels.get("node", "")

        client = create_monitor_client(conf)
        if client is None:
            logger.warning(
                "[rl-insight] hardware monitor on node %r is disabled: the "
                "configured monitor backend produced no client.",
                node or "<unknown>",
            )
            return

        self._monitor = HardwareMonitor(
            client=client,
            labels=actor_labels,
            interval_seconds=params["interval_seconds"],
            collector_names=params["collector_names"],
            collector_options=params.get("collector_options"),
        )
        self._monitor.start()


class NodeActorSet:
    """Stop-able handle over the per-node hardware actors of one job.

    Mirrors the ``HardwareMonitor`` lifecycle surface (``is_running`` /
    ``stop``) so ``rl_insight.finish`` can treat both modes uniformly.
    """

    def __init__(self, handles: list[Any]) -> None:
        """
        Args:
            handles: Ray actor handles for the created node actors.
        """
        self._handles = list(handles)

    @property
    def is_running(self) -> bool:
        """Whether any node actor is still being tracked."""
        return bool(self._handles)

    @property
    def node_count(self) -> int:
        """Number of tracked node actors."""
        return len(self._handles)

    def stop(self) -> None:
        """Kill every tracked node actor; best effort per actor."""
        for handle in self._handles:
            try:
                ray.kill(handle)
            except Exception:
                logger.warning(
                    "[rl-insight] failed to stop a hardware node actor.",
                    exc_info=True,
                )
        self._handles.clear()
