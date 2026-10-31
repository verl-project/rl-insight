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
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import ray
from omegaconf import DictConfig

from .base import HardwareSettings, identity_labels
from .coordinator import HardwareMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["NodeActorSet", "NodeHardwareActor"]

# Reconcile pending actors for scheduling/initialization at this cadence, and
# warn once when an actor stays pending longer than the warning threshold.
_READY_WAIT_TIMEOUT_SECONDS = 0.1
_PENDING_WARNING_SECONDS = 30.0
_STOP_JOIN_TIMEOUT_SECONDS = 5.0


@ray.remote
class NodeHardwareActor:
    """Run one host-level hardware monitor on the node it is scheduled to.

    One actor is created per job and node through node affinity and a
    job/node-scoped actor name, so host samples are collected once per node
    regardless of how many processes call ``rl_insight.init``. Samples go
    through the configured client: the job's Ray hub or the configured push
    sinks.
    """

    def __init__(
        self,
        conf: DictConfig,
        labels: dict[str, Any],
        settings: HardwareSettings,
    ) -> None:
        """``node`` and ``run_id`` are detected on the actor's own node."""
        # Keep the client stack out of this module's import path.
        from ..client import create_monitor_client

        actor_labels = {
            # Keep the label keys identical for every node actor: a worker
            # whose init omitted the project name would otherwise register a
            # second Prometheus schema and drop the batch as a duplicate.
            "project": "",
            "experiment_name": "",
            "node": "",
            "run_id": "",
            **{str(key): str(value) for key, value in labels.items()},
            **identity_labels(),
        }

        # The actor is owned by the process that created it, so Ray reclaims it
        # when that process exits.
        self._monitor: HardwareMonitor | None = None
        client = create_monitor_client(conf)
        if client is None:
            logger.warning(
                "[rl-insight] hardware monitor on node %r is disabled: the "
                "configured monitor backend produced no client.",
                actor_labels.get("node") or "<unknown>",
            )
            return

        self._monitor = HardwareMonitor(
            client=client,
            labels=actor_labels,
            interval_seconds=settings.interval_seconds,
            collector_names=settings.collector_names,
        )
        self._monitor.start()


class NodeActorSet:
    """Keep the per-node hardware actors of one job alive.

    The node set is a fixed startup snapshot; nodes that join later are not
    added. Missing actors in that set are recreated, and actors that fail
    creation or readiness are retried on the next pass. Ready actors are probed
    only once. Exposes the same ``stop`` interface as ``HardwareMonitor`` so
    ``rl_insight.finish`` handles both modes uniformly.
    """

    def __init__(
        self,
        handles: Mapping[str, Any],
        *,
        create_actor: Callable[[str], Any],
        node_ids: Iterable[str],
        reconcile_interval_seconds: float = 30.0,
    ) -> None:
        """``handles`` and ``node_ids`` form the fixed startup snapshot."""
        self._handles: dict[str, Any] = dict(handles)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._create_actor = create_actor
        # Keeping the snapshot immutable removes periodic State API queries.
        self._node_ids = set(node_ids)
        self._reconcile_interval_seconds = max(1.0, float(reconcile_interval_seconds))
        self._pending_since = {node_id: time.monotonic() for node_id in self._handles}
        self._warned_pending: set[str] = set()
        # Nodes with an unresolved failure; each failure mode is logged once
        # until the actor becomes ready again.
        self._warned_unhealthy: set[str] = set()
        self._warned_poll_failure = False
        self._reconcile_thread: threading.Thread | None = None
        if self._node_ids:
            thread = threading.Thread(
                target=self._reconcile_loop,
                name="rl-insight-hardware-nodes",
                daemon=True,
            )
            self._reconcile_thread = thread
            thread.start()

    @property
    def node_ids(self) -> list[str]:
        """Sorted node ids with a tracked actor."""
        with self._lock:
            return sorted(self._handles)

    def stop(self) -> None:
        """Stop reconciliation and release the local actor handles.

        The actors are shared by every process in the job and carry no explicit
        shutdown call here; Ray reclaims each actor when the process that
        created it exits. Failures are logged with their traceback so
        ``finish()`` can continue releasing the monitor client.
        """
        try:
            self._stop_event.set()
            thread = self._reconcile_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
            with self._lock:
                self._handles.clear()
                self._pending_since.clear()
                self._warned_pending.clear()
                self._warned_unhealthy.clear()
        except Exception:
            logger.error(
                "[rl-insight] failed to stop hardware monitoring.", exc_info=True
            )

    def _reconcile_loop(self) -> None:
        """Keep the fixed node set ready until ``stop`` is called."""
        while not self._stop_event.wait(self._reconcile_interval_seconds):
            try:
                self._reconcile_once()
            except Exception:
                logger.warning(
                    "[rl-insight] hardware node actor refresh failed; retrying "
                    "on the next round.",
                    exc_info=True,
                )

    def _reconcile_once(self) -> None:
        """Recreate missing actors in the initial node set and poll readiness."""
        if self._stop_event.is_set():
            return

        desired = self._node_ids
        with self._lock:
            current = set(self._handles)

        for node_id in sorted(desired - current):
            if self._stop_event.is_set():
                return
            try:
                handle = self._create_actor(node_id)
            except Exception:
                with self._lock:
                    first_failure = node_id not in self._warned_unhealthy
                    self._warned_unhealthy.add(node_id)
                if first_failure:
                    logger.warning(
                        "[rl-insight] failed to create the hardware actor on node "
                        "%s during reconciliation; retrying every %.0fs.",
                        node_id,
                        self._reconcile_interval_seconds,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "[rl-insight] hardware actor on node %s is still unavailable.",
                        node_id,
                        exc_info=True,
                    )
                continue
            with self._lock:
                stopped = self._stop_event.is_set()
                if not stopped:
                    self._handles[node_id] = handle
                    self._pending_since[node_id] = time.monotonic()
            if stopped:
                self._kill_handle(handle)
                return

        self._check_ready()

    def _check_ready(self) -> None:
        """Poll readiness only for actors that have not become ready yet."""
        with self._lock:
            pending = [
                (node_id, self._handles[node_id])
                for node_id in self._pending_since
                if node_id in self._handles
            ]
        if not pending:
            return

        refs: dict[Any, str] = {}
        for node_id, handle in pending:
            try:
                # ``__ray_ready__`` is an actor method, so it must be called
                # through ``remote``; a direct call raises TypeError.
                refs[handle.__ray_ready__.remote()] = node_id
            except Exception:
                # The handle cannot submit the readiness call; treat it as dead.
                with self._lock:
                    first_failure = node_id not in self._warned_unhealthy
                    self._warned_unhealthy.add(node_id)
                if first_failure:
                    logger.warning(
                        "[rl-insight] readiness probe failed for the hardware "
                        "actor on node %s; it will be recreated.",
                        node_id,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "[rl-insight] hardware actor on node %s is still unreachable.",
                        node_id,
                        exc_info=True,
                    )
                self._drop_handle(node_id)
        if not refs:
            return

        try:
            ready_refs, _ = ray.wait(list(refs), timeout=_READY_WAIT_TIMEOUT_SECONDS)
        except Exception:
            with self._lock:
                first_failure = not self._warned_poll_failure
                self._warned_poll_failure = True
            if first_failure:
                logger.warning(
                    "[rl-insight] failed to poll hardware actor readiness; "
                    "retrying on the next round.",
                    exc_info=True,
                )
            else:
                logger.debug(
                    "[rl-insight] hardware actor readiness polling is still failing.",
                    exc_info=True,
                )
            return
        with self._lock:
            self._warned_poll_failure = False

        now = time.monotonic()
        for ref in ready_refs:
            node_id = refs[ref]
            try:
                ray.get(ref, timeout=_READY_WAIT_TIMEOUT_SECONDS)
            except TimeoutError:
                # Not collected within the grace period; check again next round.
                continue
            except Exception:
                with self._lock:
                    first_failure = node_id not in self._warned_unhealthy
                    self._warned_unhealthy.add(node_id)
                if first_failure:
                    logger.warning(
                        "[rl-insight] hardware actor on node %s exited; it will "
                        "be recreated if the node is still part of the job.",
                        node_id,
                    )
                else:
                    logger.debug(
                        "[rl-insight] hardware actor on node %s exited again.",
                        node_id,
                    )
                self._drop_handle(node_id)
                continue
            with self._lock:
                self._pending_since.pop(node_id, None)
                self._warned_pending.discard(node_id)
                self._warned_unhealthy.discard(node_id)

        with self._lock:
            for node_id, since in list(self._pending_since.items()):
                if (
                    now - since >= _PENDING_WARNING_SECONDS
                    and node_id not in self._warned_pending
                ):
                    logger.warning(
                        "[rl-insight] hardware actor on node %s is still pending "
                        "after %.0fs; check Ray scheduling and node resources.",
                        node_id,
                        now - since,
                    )
                    self._warned_pending.add(node_id)

    def _drop_handle(self, node_id: str) -> None:
        """Forget one tracked actor and kill it if it is still alive."""
        with self._lock:
            handle = self._handles.pop(node_id, None)
            self._pending_since.pop(node_id, None)
            self._warned_pending.discard(node_id)
        if handle is not None:
            self._kill_handle(handle)

    @staticmethod
    def _kill_handle(handle: Any) -> None:
        """Kill an actor handle, ignoring actors that already exited."""
        try:
            ray.kill(handle)
        except Exception:
            logger.debug(
                "[rl-insight] failed to kill a hardware node actor.",
                exc_info=True,
            )
