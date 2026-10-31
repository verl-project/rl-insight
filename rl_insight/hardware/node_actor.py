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
from collections.abc import Callable, Mapping
from typing import Any

import ray
from omegaconf import DictConfig

from .coordinator import HardwareMonitor
from .identity import identity_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["NodeActorSet", "NodeHardwareActor"]

# Reconcile pending actors for scheduling/initialization at this cadence, and
# warn once when an actor stays pending longer than the warning threshold.
_READY_WAIT_TIMEOUT_SECONDS = 0.1
_PENDING_WARNING_SECONDS = 30.0
_STOP_JOIN_TIMEOUT_SECONDS = 5.0
# A detached node actor outlives the process that created it, so it must notice
# on its own when the job ends; poll the job state at this cadence.
_JOB_STATE_POLL_SECONDS = 60.0
_JOB_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})


@ray.remote
class NodeHardwareActor:
    """Runs one host-level hardware monitor on the node it is scheduled to.

    One actor is created per job and node (via a hard ``NodeAffinity``
    scheduling strategy and a job/node-scoped actor name), so host samples are
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
            params: Resolved hardware params: ``interval_seconds`` and
                ``collector_names``.
        """
        # Imported here so importing this module never pulls the client stack.
        from ..client import create_monitor_client

        actor_labels = {
            # Keep the label keys identical for every node actor of the job: a
            # worker whose ``init`` omitted the project name would otherwise
            # register a second Prometheus schema for the same metric name,
            # and the hub would reject the whole batch as a duplicate series.
            "project": "",
            "experiment_name": "",
            "node": "",
            "run_id": "",
            **{str(key): str(value) for key, value in labels.items()},
            **identity_labels(),
        }
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
        self._start_job_watchdog()

    def _start_job_watchdog(self) -> None:
        """Stop this detached actor once the job that created it has ended."""
        thread = threading.Thread(
            target=self._job_watchdog,
            name="rl-insight-hardware-job-watchdog",
            daemon=True,
        )
        thread.start()

    def _job_watchdog(self) -> None:
        """Exit the actor when its Ray job reaches a terminal state.

        The actor is detached so that dropping the last handle (for example
        when a short-lived worker calls ``finish``) no longer destroys it
        mid-job. Detached actors are not reclaimed with their job, so the
        state of the creating job is polled and the actor exits by itself.
        Failed polls keep the actor alive: only an explicit terminal state
        ends it.
        """
        job_id = ray.get_runtime_context().get_job_id()
        job_id = job_id if isinstance(job_id, str) else job_id.hex()
        from ray.util.state import get_job

        while True:
            time.sleep(_JOB_STATE_POLL_SECONDS)
            try:
                state = get_job(job_id=job_id)
            except Exception:
                continue
            status = str(getattr(state, "status", "") or "").rsplit(".", 1)[-1]
            if status in _JOB_TERMINAL_STATES:
                logger.info(
                    "[rl-insight] job %s finished; stopping its hardware actor.",
                    job_id,
                )
                ray.actor.exit_actor()
                return


class NodeActorSet:
    """Tracks the per-node hardware actors of one job and reconciles changes.

    The set adds actors for job nodes discovered after ``init`` (for example
    nodes of workers that start later) and removes actors whose node is no
    longer used by the job. It also polls actor readiness so a scheduler-pending
    actor is reported instead of being silently counted as running.

    Mirrors the ``HardwareMonitor`` lifecycle surface (``is_running`` /
    ``stop``) so ``rl_insight.finish`` can treat both modes uniformly.
    """

    def __init__(
        self,
        handles: Mapping[str, Any] | list[Any],
        *,
        create_actor: Callable[[str], Any] | None = None,
        node_ids_provider: Callable[[], list[str] | None] | None = None,
        reconcile_interval_seconds: float = 30.0,
    ) -> None:
        """
        Args:
            handles: Initial actor handles, keyed by Ray node id. A plain list
                is accepted for simple lifecycle tests.
            create_actor: Callback that creates one actor for a node id.
            node_ids_provider: Callback returning the nodes currently used by
                the job, or ``None`` when discovery is temporarily unavailable.
            reconcile_interval_seconds: Delay between reconciliation rounds.
        """
        if isinstance(handles, Mapping):
            self._handles: dict[str, Any] = dict(handles)
        else:
            self._handles = {
                f"node-{index}": handle for index, handle in enumerate(handles)
            }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._create_actor = create_actor
        self._node_ids_provider = node_ids_provider
        self._reconcile_interval_seconds = max(1.0, float(reconcile_interval_seconds))
        self._pending_since = {node_id: time.monotonic() for node_id in self._handles}
        self._warned_pending: set[str] = set()
        self._reconcile_thread: threading.Thread | None = None
        if create_actor is not None and node_ids_provider is not None:
            thread = threading.Thread(
                target=self._reconcile_loop,
                name="rl-insight-hardware-nodes",
                daemon=True,
            )
            self._reconcile_thread = thread
            thread.start()

    @property
    def is_running(self) -> bool:
        """Whether any node actor is still being tracked."""
        with self._lock:
            return bool(self._handles)

    @property
    def node_count(self) -> int:
        """Number of tracked node actors."""
        with self._lock:
            return len(self._handles)

    @property
    def node_ids(self) -> list[str]:
        """Sorted node ids with a tracked actor."""
        with self._lock:
            return sorted(self._handles)

    def stop(self) -> None:
        """Stop reconciliation and forget the tracked node actors.

        The actors are shared by every process of the job (they are named per
        job and node), so killing them here would stop hardware sampling for
        the whole job whenever any single process calls ``rl_insight.finish``.
        They are left to Ray, which reaps them when the job exits.
        """
        self._stop_event.set()
        thread = self._reconcile_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
        with self._lock:
            self._handles.clear()
            self._pending_since.clear()
            self._warned_pending.clear()

    def _reconcile_loop(self) -> None:
        """Reconcile the node set until ``stop`` is called."""
        while not self._stop_event.wait(self._reconcile_interval_seconds):
            try:
                self._reconcile_once()
            except Exception:
                logger.warning(
                    "[rl-insight] hardware node reconciliation failed; retrying "
                    "on the next round.",
                    exc_info=True,
                )

    def _reconcile_once(self) -> None:
        """Add new job nodes, drop removed ones, and poll actor readiness."""
        provider = self._node_ids_provider
        create_actor = self._create_actor
        if provider is None or create_actor is None or self._stop_event.is_set():
            return

        node_ids = provider()
        if self._stop_event.is_set():
            return
        if node_ids is None:
            # Discovery failed: keep the current node set, but still poll
            # actor readiness so a dead actor can be recovered later.
            self._check_ready()
            return
        desired = set(node_ids)
        with self._lock:
            current = set(self._handles)

        for node_id in sorted(desired - current):
            if self._stop_event.is_set():
                return
            try:
                handle = create_actor(node_id)
            except Exception:
                logger.warning(
                    "[rl-insight] failed to create the hardware actor on node "
                    "%s during reconciliation.",
                    node_id,
                    exc_info=True,
                )
                continue
            if handle is None:
                continue
            with self._lock:
                if self._stop_event.is_set():
                    should_stop = True
                else:
                    self._handles[node_id] = handle
                    self._pending_since[node_id] = time.monotonic()
                    should_stop = False
            if should_stop:
                self._kill_handle(handle)
                return

        # A successful discovery round is authoritative: actors on nodes no
        # longer used by the job are stopped. A failed discovery returns
        # ``None`` above and never reaches this point.
        for node_id in sorted(current - desired):
            self._drop_handle(node_id)

        self._check_ready()

    def _check_ready(self) -> None:
        """Poll actor readiness and drop actors that have died."""
        with self._lock:
            handles = list(self._handles.items())

        refs: dict[Any, str] = {}
        for node_id, handle in handles:
            try:
                # ``__ray_ready__`` is a regular actor method, so it must be
                # invoked through ``remote``; calling the attribute directly
                # raises TypeError even for a healthy actor.
                refs[handle.__ray_ready__.remote()] = node_id
            except Exception:
                # The handle cannot even submit the readiness call; treat it as
                # dead and let the next reconciliation recreate it.
                logger.warning(
                    "[rl-insight] readiness probe failed for the hardware actor "
                    "on node %s.",
                    node_id,
                    exc_info=True,
                )
                self._drop_handle(node_id)
        if not refs:
            return

        try:
            ready_refs, _ = ray.wait(list(refs), timeout=_READY_WAIT_TIMEOUT_SECONDS)
        except Exception:
            logger.warning(
                "[rl-insight] failed to poll hardware actor readiness.",
                exc_info=True,
            )
            return

        now = time.monotonic()
        for ref in ready_refs:
            node_id = refs[ref]
            try:
                ray.get(ref, timeout=_READY_WAIT_TIMEOUT_SECONDS)
            except TimeoutError:
                # The readiness call was not collected within the grace
                # period; keep the actor and check again next round.
                continue
            except Exception:
                logger.warning(
                    "[rl-insight] hardware actor on node %s exited; it will be "
                    "recreated if the node is still part of the job.",
                    node_id,
                )
                self._drop_handle(node_id)
                continue
            with self._lock:
                self._pending_since.pop(node_id, None)
                self._warned_pending.discard(node_id)

        with self._lock:
            for node_id, since in list(self._pending_since.items()):
                if node_id not in self._handles:
                    continue
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
        """Best-effort actor kill; dead actors are expected during reconcile."""
        try:
            ray.kill(handle)
        except Exception:
            logger.debug(
                "[rl-insight] failed to kill a hardware node actor.",
                exc_info=True,
            )
