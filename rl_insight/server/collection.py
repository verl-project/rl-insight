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

"""Bounded, framework-independent collection of leased metric endpoints."""

from __future__ import annotations

import http.client
import ipaddress
import ssl
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from types import SimpleNamespace
from urllib.parse import urlsplit

from prometheus_client import CollectorRegistry, Gauge, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from ..protocol import RunError, proof_headers

logger = logging.getLogger(__name__)


class Collection:
    def __init__(self, registry, timeout=9.0):
        self.registry = registry
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="insight-scrape"
        )
        self._capacity = threading.BoundedSemaphore(16)
        self._run_locks: dict[str, threading.Lock] = {}
        self._run_reads: dict[str, int] = {}
        self._lock = threading.Lock()

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _read(self, item, deadline):
        address = urlsplit(item["metrics_url"])
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Scrape deadline exceeded")
        connection = http.client.HTTPConnection(
            address.hostname, address.port or (443 if address.scheme == "https" else 80)
        )
        family = (
            socket.AF_INET6
            if ipaddress.ip_address(address.hostname).version == 6
            else socket.AF_INET
        )
        connection.sock = socket.socket(family, socket.SOCK_STREAM)
        connection.sock.settimeout(min(2, remaining))
        timer = None
        try:

            def abort():
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except (OSError, AttributeError):
                    pass

            timer = threading.Timer(remaining, abort)
            timer.daemon = True
            timer.start()
            connection.sock.connect((address.hostname, connection.port))
            if address.scheme == "https":
                connection.sock = ssl.create_default_context().wrap_socket(
                    connection.sock,
                    server_hostname=address.hostname,
                    do_handshake_on_connect=False,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("TLS handshake deadline exceeded")
                connection.sock.settimeout(min(2, remaining))
                connection.sock.do_handshake()
            connection.request(
                "GET",
                address.path or "/metrics",
                headers={
                    "Accept": "text/plain; version=0.0.4",
                    "Accept-Encoding": "identity",
                },
            )
            response = connection.getresponse()
            context = self.registry.get(item["run_id"])
            if response.status != 200 or any(
                response.getheader(key) != value
                for key, value in proof_headers(context, item["incarnation"]).items()
            ):
                raise RunError(
                    "Exporter response identity does not match its publisher", 409
                )
            body = bytearray()
            while chunk := response.read(65536):
                body.extend(chunk)
                if len(body) > 4 * 1024 * 1024 or time.monotonic() >= deadline:
                    raise RunError("Exporter response exceeds collection budget", 502)
            return body.decode("utf-8")
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()

    def _acquire(self, run_id):
        with self._lock:
            count = self._run_reads.get(run_id, 0)
            if count >= 4 or not self._capacity.acquire(blocking=False):
                return False
            self._run_reads[run_id] = count + 1
            return True

    def _release(self, run_id):
        with self._lock:
            count = self._run_reads[run_id] - 1
            if count:
                self._run_reads[run_id] = count
            else:
                self._run_reads.pop(run_id)
            self._capacity.release()

    def verify(self, description):
        if not self._acquire(description["run_id"]):
            raise RunError("Collection capacity exhausted", 503)
        try:
            self._read(description, time.monotonic() + min(3, self.timeout))
        except RunError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise RunError("Exporter response could not be verified", 502) from exc
        finally:
            self._release(description["run_id"])

    def _submit(self, item, deadline):
        if not self._acquire(item["run_id"]):
            return None

        def fetch():
            try:
                text = self._read(item, deadline)
                if not self.registry.endpoint_valid(item):
                    raise RunError("Endpoint lease changed during collection", 409)
                families = []
                seen = set()
                count = 0
                context = self.registry.get(item["run_id"])
                for family in text_string_to_metric_families(text):
                    families.append(family)
                    count += len(family.samples)
                    if count > 100000 or time.monotonic() >= deadline:
                        raise RunError("Endpoint parsing budget exceeded", 502)
                    if family.name.startswith("rl_insight_proxy_"):
                        raise RunError(
                            "Exporter uses reserved collection metric prefix", 409
                        )
                    for sample in family.samples:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Endpoint parsing deadline exceeded")
                        labels = {**item["labels"], **sample.labels}
                        labels = context.labels(labels)
                        for key, value in {
                            "component": item["component"],
                            "endpoint_id": item["endpoint_id"],
                        }.items():
                            if key in labels and str(labels[key]) != value:
                                raise RunError(
                                    "Exporter metric identity conflicts with registration",
                                    409,
                                )
                            labels[key] = value
                        labels.setdefault(
                            "instance", urlsplit(item["metrics_url"]).netloc
                        )
                        key = (sample.name, tuple(sorted(labels.items())))
                        if key in seen:
                            raise RunError("Exporter has duplicate samples", 409)
                        seen.add(key)
                        sample.labels.clear()
                        sample.labels.update(labels)
                return families, len(text.encode())
            finally:
                self._release(item["run_id"])

        try:
            return self._pool.submit(fetch)
        except Exception:
            self._release(item["run_id"])
            raise

    def collect(self, run_id, timeout=None):
        self.registry.get(run_id)
        with self._lock:
            lock = self._run_locks.setdefault(run_id, threading.Lock())
            if not lock.acquire(blocking=False):
                raise RunError("This Run already has a collection request", 503)
        try:
            return self._collect(run_id, min(self.timeout, timeout or self.timeout))
        finally:
            with self._lock:
                lock.release()
                self._run_locks.pop(run_id, None)

    def _collect(self, run_id, timeout):
        items = self.registry.endpoints(run_id)
        deadline = time.monotonic() + timeout
        sources = iter(items)
        pending = {}
        merged = {}
        results = []
        total_bytes = 0
        total_samples = 0
        successful = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 4:
                item = next(sources, None)
                if item is None:
                    exhausted = True
                    break
                future = self._submit(item, deadline)
                if future is None:
                    logger.warning(
                        "Run %s endpoint %s cannot acquire collection capacity",
                        run_id,
                        item["endpoint_id"],
                    )
                else:
                    pending[future] = item
            if not pending:
                continue
            done, _ = wait(
                pending,
                timeout=max(0, deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                logger.warning(
                    "Run %s collection deadline exceeded with %d reads pending",
                    run_id,
                    len(pending),
                )
                break
            for future in done:
                item = pending.pop(future)
                try:
                    families, size = future.result()
                    total_bytes += size
                    total_samples += sum(len(family.samples) for family in families)
                    if total_bytes > 16 * 1024 * 1024 or total_samples > 200000:
                        raise RunError("Aggregated sample budget exceeded", 503)
                    if time.monotonic() >= deadline or not self.registry.endpoint_valid(
                        item
                    ):
                        raise TimeoutError("Collection lease or deadline expired")
                    results.append((item, families))
                except (
                    RunError,
                    OSError,
                    ValueError,
                    http.client.HTTPException,
                ) as exc:
                    logger.warning(
                        "Run %s endpoint %s collection failed: %s",
                        run_id,
                        item["lease_id"],
                        exc,
                    )
        definitions = {}
        for _item, families in results:
            for family in families:
                definitions.setdefault(family.name, set()).add(
                    (family.type, family.documentation)
                )
        conflicts = {
            name for name, variants in definitions.items() if len(variants) > 1
        }
        for item, families in results:
            if any(
                family.name in conflicts for family in families
            ) or not self.registry.endpoint_valid(item):
                logger.warning(
                    "Run %s endpoint %s has conflicting metadata or a changed lease",
                    run_id,
                    item["endpoint_id"],
                )
                continue
            for family in families:
                if family.name in merged:
                    merged[family.name].samples.extend(family.samples)
                else:
                    merged[family.name] = family
            successful.add(item["lease_id"])
        metrics = CollectorRegistry()
        status = Gauge(
            "rl_insight_proxy_endpoint_up",
            "Endpoint collected successfully",
            ["run_id", "endpoint_id", "component"],
            registry=metrics,
        )
        for item in items:
            status.labels(run_id, item["endpoint_id"], item["component"]).set(
                item["lease_id"] in successful
            )
        for name, value in {
            "endpoints": len(items),
            "failures": len(items) - len(successful),
            "empty": not items,
        }.items():
            Gauge(
                "rl_insight_proxy_" + name,
                "Run collection " + name,
                ["run_id"],
                registry=metrics,
            ).labels(run_id).set(value)
        body = generate_latest(
            SimpleNamespace(collect=lambda: merged.values())
        ) + generate_latest(metrics)
        if len(body) > 16 * 1024 * 1024:
            raise RunError("Aggregated response budget exceeded", 503)
        return (503 if items and not successful else 200), body
