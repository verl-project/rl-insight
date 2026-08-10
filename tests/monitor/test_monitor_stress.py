#!/usr/bin/env python3
# Copyright (c) 2026 verl-project authors.
#
"""RL-Insight stress / concurrency API test.

Tests all combinations of process count × thread count:
  - PROCESS_LEVELS  = [1, 2, 4, 8, 16]
  - THREAD_LEVELS   = [10, 20, 40, 80]

Per combination:
  - Client submission latency (avg, p50, p95, queue = p95-p50)
  - Hub delivery failure rate (submitted vs events_applied)
  - Track exact count and sum of all submitted values

Checkpoint: after each combination, results are written to disk. On restart,
already-completed combinations are skipped so the test can resume.

Post-test verification:
  - Stress aggregate: Prometheus delta == submitted total (count + sum)
  - Grafana frontend health + panel data
  - End-to-end consistency: known values -> Prometheus exact match

Run from the repo root:
    python tests/monitor/test_monitor_stress.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import threading
import multiprocessing as mp
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import ray
import rl_insight as insight
import rl_insight.api as _api
from rl_insight.client.ray_monitor_client import (
    _current_job_actor_name,
    MonitorRayClient,
)
from rl_insight.utils.constants import MonitorRayActor
from rl_insight.utils import MonitorEventKind


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRESS_DURATION_S = 1
PROCESS_LEVELS = [1, 2, 4]
THREAD_LEVELS = [10, 20]

# Extra single-dimension stress: push one axis while holding the other fixed
EXTRA_1P_THREADS = [40]
EXTRA_10T_PROCS = [8]
FAILURE_THRESHOLD = 0.05
PROMETHEUS_PORT = int(os.environ.get("RL_INSIGHT_PROMETHEUS_PORT", "9090"))
GRAFANA_PORT = 3000
SCRAPE_WAIT_S = 15
AGGREGATE_WAIT_S = 30
NS = "rl_insight_monitor"
CHECKPOINT_FILE = os.environ.get(
    "RL_INSIGHT_STRESS_CHECKPOINT", "stress_checkpoint.json"
)


def _server_ip() -> str:
    return os.environ.get("RL_INSIGHT_SERVICE_IP", "127.0.0.1").strip() or "127.0.0.1"


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------


def _load_checkpoint() -> dict[str, Any]:
    if not os.path.exists(CHECKPOINT_FILE):
        return {"prometheus_before": {}, "results": {}, "completed": []}
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)


def _save_checkpoint(data: dict[str, Any]) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _combo_key(api_name: str, num_procs: int, num_threads: int) -> str:
    return f"{api_name}:{num_procs}p:{num_threads}t"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    avg: float = 0.0
    queue_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls()
        s = sorted(samples)
        p50 = _pct(s, 50)
        p95 = _pct(s, 95)
        return cls(
            p50=p50,
            p95=p95,
            p99=_pct(s, 99),
            avg=statistics.mean(s),
            queue_ms=max(0.0, (p95 - p50) * 1000),
        )


@dataclass
class ConcurrencyResult:
    api_name: str
    num_procs: int
    num_threads: int
    submitted: int
    submitted_sum: float = 0.0
    hub_delta: int = 0
    failure_rate: float = 0.0
    throughput_per_sec: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    k = (p / 100.0) * (len(data) - 1)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)


# ---------------------------------------------------------------------------
# Exact-sum helper for histogram values
# ---------------------------------------------------------------------------


def _histogram_sum_for(n: int) -> float:
    """Exact sum of N histogram values: each call emits 200 + (seq % 100)."""
    k, r = divmod(n, 100)
    # sum of i%100 for i in [0, n-1] = k * sum(0..99) + sum(0..r-1)
    cycle_sum = k * 4950 + r * (r - 1) // 2
    return float(n * 200 + cycle_sum)


# ---------------------------------------------------------------------------
# Hub / Prometheus helpers
# ---------------------------------------------------------------------------


def _hub_events_count() -> int:
    """Query the hub actor's events_applied counter.

    Tries the job-scoped actor name first (post-#112), then falls back
    to the bare constant name for backwards compatibility.
    """
    namespace = MonitorRayActor.NAMESPACE
    for actor_name in (_current_job_actor_name(), MonitorRayActor.NAME):
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
            status = ray.get(actor.get_status.remote())
            return int(status.get("events_applied", 0))
        except ValueError:
            continue
        except Exception:
            return -1
    return -1


def _promql_value(service_ip: str, metric: str) -> float | None:
    import urllib.parse

    url = (
        f"http://{service_ip}:{PROMETHEUS_PORT}/api/v1/query"
        f"?query={urllib.parse.quote(metric)}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("status") != "success":
            return None
        results = data.get("data", {}).get("result", [])
        if not results:
            return None
        return float(results[0].get("value", [None, 0])[1])
    except Exception:
        return None


def _prometheus_has_data(service_ip: str, query: str) -> bool:
    import urllib.parse

    url = (
        f"http://{service_ip}:{PROMETHEUS_PORT}/api/v1/query"
        f"?query={urllib.parse.quote(query)}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return data.get("status") == "success" and bool(
            data.get("data", {}).get("result")
        )
    except Exception:
        return False


def _grafana_healthy(service_ip: str) -> bool:
    try:
        url = f"http://{service_ip}:{GRAFANA_PORT}/api/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Process worker (runs in each spawned child process)
# ---------------------------------------------------------------------------


def _process_worker(
    api_name: str,
    num_threads: int,
    duration_s: float,
    track_sum: bool,
    result_queue: mp.Queue,
    hub_actor_name: str,
) -> None:
    """Entry point for each child process. Connects to the existing hub, spawns threads.

    Unlike the main process, child processes do NOT create their own hub
    (that would collide on the Prometheus metrics port).  Instead they
    look up the hub actor created by the main process and submit events
    directly through ``MonitorRayClient``.
    """
    import os as _os
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    import ray as _ray

    # Connect to Ray (shares the cluster with the main process)
    try:
        _ray.init(
            address="auto", namespace="rl-insight-monitor", ignore_reinit_error=True
        )
    except ConnectionError:
        _ray.init(namespace="rl-insight-monitor", ignore_reinit_error=True)

    # Look up the main process's hub — do NOT create a new one
    try:
        hub_handle = _ray.get_actor(hub_actor_name, namespace=MonitorRayActor.NAMESPACE)
        client = MonitorRayClient(hub_handle)
    except Exception:
        result_queue.put({"submitted": 0, "submitted_sum": 0.0, "latencies": []})
        return

    _process_id = str(_os.getpid())

    # Build emitter using direct client.apply_event (bypasses rl_insight API
    # so we never call init() which would try to create a second hub).
    if api_name == "counter":

        def _emit(seq: int) -> None:
            client.apply_event(
                {
                    "kind": MonitorEventKind.COUNTER,
                    "name": "train_step_total",
                    "documentation": "Counter: total training steps",
                    "value": 1.0,
                    "labels": {
                        "project": "verl",
                        "experiment_name": "ppo-stress-test",
                        "worker": "stress",
                    },
                }
            )

    elif api_name == "gauge":

        def _emit(seq: int) -> None:
            client.apply_event(
                {
                    "kind": MonitorEventKind.GAUGE,
                    "name": "reward_mean",
                    "documentation": "Gauge: mean reward value",
                    "value": float(seq % 1000),
                    "labels": {
                        "project": "verl",
                        "experiment_name": "ppo-stress-test",
                        "worker": "stress",
                    },
                }
            )

    elif api_name == "histogram":

        def _emit(seq: int) -> None:
            client.apply_event(
                {
                    "kind": MonitorEventKind.HISTOGRAM,
                    "name": "step_latency_ms",
                    "documentation": "Histogram: step latency in ms",
                    "value": float(200 + seq % 100),
                    "labels": {
                        "project": "verl",
                        "experiment_name": "ppo-stress-test",
                        "worker": "stress",
                    },
                }
            )

    elif api_name == "trace":

        def _emit(seq: int) -> None:
            now_ns = _time.time_ns()
            client.apply_event(
                {
                    "kind": MonitorEventKind.TRACE,
                    "name": "rollout_generate",
                    "start_time_ns": now_ns,
                    "end_time_ns": now_ns,
                    "attributes": {
                        "process_id": _process_id,
                        "project": "verl",
                        "experiment_name": "ppo-stress-test",
                        "monitor.trace_segment": "state_interval",
                        "state_name": "rollout_generate",
                        "state_lane_id": "stress",
                        "step": seq,
                    },
                }
            )

    else:
        result_queue.put({"submitted": 0, "submitted_sum": 0.0, "latencies": []})
        return

    # Histogram sum helper (local copy for child process)
    if track_sum:

        def _hist_sum(n: int) -> float:
            k, r = divmod(n, 100)
            cycle_sum = k * 4950 + r * (r - 1) // 2
            return float(n * 200 + cycle_sum)

    # Thread workers
    stop_event = threading.Event()
    worker_counts: list[int] = [0] * num_threads
    all_latencies: list[float] = []
    lat_lock = threading.Lock()

    def _thread_worker(wid: int) -> None:
        seq = 0
        local_lat: list[float] = []
        while not stop_event.is_set():
            t0 = _time.perf_counter()
            try:
                _emit(seq)
            except Exception:
                pass
            else:
                local_lat.append(_time.perf_counter() - t0)
            seq += 1
        worker_counts[wid] = len(local_lat)
        with lat_lock:
            all_latencies.extend(local_lat)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_thread_worker, i) for i in range(num_threads)]
        _time.sleep(duration_s)
        stop_event.set()
        for f in futures:
            f.result(timeout=10)

    submitted = sum(worker_counts)
    submitted_sum = 0.0
    if track_sum:
        submitted_sum = sum(_hist_sum(n) for n in worker_counts)

    result_queue.put(
        {
            "submitted": submitted,
            "submitted_sum": submitted_sum,
            "latencies": all_latencies,
        }
    )


# ---------------------------------------------------------------------------
# Stress runner
# ---------------------------------------------------------------------------


def run_concurrency_test(
    api_name: str,
    num_procs: int,
    num_threads: int,
    hub_actor_name: str,
    track_sum: bool = False,
) -> ConcurrencyResult:
    hub_before = _hub_events_count()

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes: list[mp.context.SpawnProcess] = []

    for _ in range(num_procs):
        p = ctx.Process(
            target=_process_worker,
            args=(
                api_name,
                num_threads,
                STRESS_DURATION_S,
                track_sum,
                result_queue,
                hub_actor_name,
            ),
        )
        processes.append(p)
        p.start()

    # Collect results from all child processes
    all_latencies: list[float] = []
    total_submitted = 0
    total_sum = 0.0
    for _ in range(len(processes)):
        try:
            result = result_queue.get(timeout=STRESS_DURATION_S + 30)
            total_submitted += result["submitted"]
            total_sum += result["submitted_sum"]
            all_latencies.extend(result["latencies"])
        except Exception:
            pass  # Process may have died; results from survivors are still valid

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    hub_after = _hub_events_count()
    hub_delta = hub_after - hub_before if hub_before >= 0 and hub_after >= 0 else -1

    if hub_delta >= 0 and total_submitted > 0:
        fail_rate = max(0.0, 1.0 - hub_delta / total_submitted)
    else:
        fail_rate = 1.0

    return ConcurrencyResult(
        api_name=api_name,
        num_procs=num_procs,
        num_threads=num_threads,
        submitted=total_submitted,
        submitted_sum=total_sum,
        hub_delta=hub_delta,
        failure_rate=fail_rate,
        throughput_per_sec=total_submitted / max(STRESS_DURATION_S, 0.001),
        latency=LatencyStats.from_samples(all_latencies),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_lat(lat: LatencyStats) -> str:
    return (
        f"avg={lat.avg * 1000:.1f}ms  "
        f"p50={lat.p50 * 1000:.1f}ms  "
        f"p95={lat.p95 * 1000:.1f}ms  "
        f"queue={lat.queue_ms:.0f}ms"
    )


def print_header() -> None:
    print(
        f"\n{'API':<12} {'P':>3} {'T':>3}  {'Submitted':>10} {'HubRcvd':>8}  {'Fail%':>6}  {'Avg(ms)':>8}  {'Queue(ms)':>10}  {'Thru/s':>10}"
    )
    print("-" * 104)


def print_row(r: ConcurrencyResult) -> None:
    tag = "" if r.failure_rate <= FAILURE_THRESHOLD else " <<<"
    print(
        f"{r.api_name:<12} {r.num_procs:>3} {r.num_threads:>3}  "
        f"{r.submitted:>10} {r.hub_delta:>8}  "
        f"{r.failure_rate:>5.1%}  {r.latency.avg * 1000:>7.1f}ms  "
        f"{r.latency.queue_ms:>9.0f}  {r.throughput_per_sec:>10.0f}{tag}"
    )


def print_api_summary(all_results: list[ConcurrencyResult]) -> None:
    print(f"\n{'=' * 70}")
    print("  Per-API Latency Breakdown (highest load combination)")
    print(f"{'=' * 70}")
    print(
        f"{'API':<14} {'P':>3} {'T':>3} {'Avg(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} {'Queue(ms)':>10}  {'Thru/s':>10}"
    )
    print("-" * 79)
    for api_name in ["counter", "gauge", "histogram", "trace"]:
        api_results = [r for r in all_results if r.api_name == api_name]
        if not api_results:
            continue
        # Pick the result with highest total concurrency (procs × threads)
        best = max(api_results, key=lambda r: r.num_procs * r.num_threads)
        lat = best.latency
        print(
            f"{api_name:<14} {best.num_procs:>3} {best.num_threads:>3} "
            f"{lat.avg * 1000:>8.1f} {lat.p50 * 1000:>8.1f} "
            f"{lat.p95 * 1000:>8.1f} {lat.queue_ms:>9.0f}  {best.throughput_per_sec:>10.0f}"
        )


def print_analysis(all_results: list[ConcurrencyResult]) -> int:
    """Print analysis summary and return the number of failing combinations."""
    print(f"\n{'=' * 70}")
    print("  Analysis")
    print(f"{'=' * 70}")
    mq = max(
        (r.latency.queue_ms for r in all_results if r.latency.queue_ms > 0), default=0
    )
    total_combos = len(all_results)
    failures = sum(1 for r in all_results if r.failure_rate > FAILURE_THRESHOLD)
    print(f"""
  {total_combos} combinations tested, {failures} failures.
  Queue time = p95 - p50: wait time in Ray actor mailbox.
  Max observed queue: {mq:.0f}ms.  Grows with concurrency; inherent to
  single-actor design, not a bug.  The checks below verify that
  despite queue delays, all data reaches Prometheus intact.
""")
    return failures


# ---------------------------------------------------------------------------
# Grafana
# ---------------------------------------------------------------------------


def _verify_grafana_frontend(service_ip: str) -> int:
    """Return number of failed checks."""
    failures = 0
    print(f"\n{'=' * 70}")
    print("  Grafana Frontend")
    print(f"{'=' * 70}")
    ok = _grafana_healthy(service_ip)
    print(f"  http://{service_ip}:{GRAFANA_PORT}  {'OK' if ok else 'UNREACHABLE'}")
    if not ok:
        failures += 1
        return failures
    for panel, query in [
        ("metric_count", f"{NS}_train_step_total"),
        ("metric_gauge", f"{NS}_reward_mean"),
        ("metric_histogram", f"{NS}_step_latency_ms_bucket"),
    ]:
        has = _prometheus_has_data(service_ip, query)
        if not has:
            failures += 1
        print(f"    [{'PASS' if has else 'FAIL'}] {panel}")
    return failures


# ---------------------------------------------------------------------------
# Stress aggregate: exact count + sum verification
# ---------------------------------------------------------------------------


def _verify_stress_aggregate(
    service_ip: str,
    all_results: list[ConcurrencyResult],
    prometheus_before: dict[str, float],
) -> int:
    """Return number of failed checks."""
    print(f"\n{'=' * 70}")
    print("  Stress Data Aggregate Verification  (exact match)")
    print(f"{'=' * 70}")

    counter_total = sum(r.submitted for r in all_results if r.api_name == "counter")
    hist_total = sum(r.submitted for r in all_results if r.api_name == "histogram")
    hist_sum = sum(r.submitted_sum for r in all_results if r.api_name == "histogram")

    print(f"  Counter submitted      : {counter_total}")
    print(f"  Histogram submitted     : {hist_total}")
    print(f"  Histogram expected sum  : {hist_sum:.0f}")
    print(f"  Waiting for Prometheus ({AGGREGATE_WAIT_S}s) ...")
    time.sleep(AGGREGATE_WAIT_S)

    checks: list[tuple[str, bool, str, str]] = []
    L = '{worker="stress"}'

    # Counter: exact count
    after_counter = _promql_value(service_ip, f"{NS}_train_step_total" + L)
    base_counter = prometheus_before.get("counter", 0.0)
    if after_counter is not None:
        delta = after_counter - base_counter
        ok = abs(delta - counter_total) < 0.5
        checks.append(
            ("counter delta == submitted", ok, str(counter_total), f"{delta:.0f}")
        )
    else:
        checks.append(("counter delta", False, str(counter_total), "no data"))

    # Histogram: exact count
    after_hist_count = _promql_value(service_ip, f"{NS}_step_latency_ms_count" + L)
    base_hist_count = prometheus_before.get("hist_count", 0.0)
    if after_hist_count is not None:
        delta = after_hist_count - base_hist_count
        ok = abs(delta - hist_total) < 0.5
        checks.append(
            ("histogram count == submitted", ok, str(hist_total), f"{delta:.0f}")
        )
    else:
        checks.append(("histogram count", False, str(hist_total), "no data"))

    # Histogram: exact sum
    after_hist_sum = _promql_value(service_ip, f"{NS}_step_latency_ms_sum" + L)
    base_hist_sum = prometheus_before.get("hist_sum", 0.0)
    if after_hist_sum is not None:
        delta = after_hist_sum - base_hist_sum
        expected = hist_sum
        ok = abs(delta - expected) < 1.0
        checks.append(
            ("histogram sum == expected", ok, f"{expected:.0f}", f"{delta:.0f}")
        )
    else:
        checks.append(("histogram sum", False, f"{hist_sum:.0f}", "no data"))

    failures = 0
    for label, ok, exp, act in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}: expected={exp}, actual={act}")
        if not ok:
            failures += 1

    print()
    if failures == 0:
        print("  All checks passed.  Every event accounted for.")
    else:
        print(f"  {failures} checks FAILED.")
    return failures


# ---------------------------------------------------------------------------
# Data consistency (known values -> Prometheus, count + content)
# ---------------------------------------------------------------------------


def _verify_data_consistency(
    service_ip: str, prometheus_before: dict[str, float]
) -> int:
    """Return number of failed checks."""
    print(f"\n{'=' * 70}")
    print("  End-to-End Data Consistency (known values)")
    print(f"{'=' * 70}")

    checks: list[tuple[str, bool, str, str]] = []
    LC = '{worker="consistency_check"}'

    # --- Counter: delta AND exact value ---
    base = prometheus_before.get(
        "counter_ck", _promql_value(service_ip, f"{NS}_train_step_total" + LC) or 0.0
    )
    N, AMOUNT = 5, 1
    for _ in range(N):
        insight.metric_count(
            "train_step_total",
            amount=AMOUNT,
            documentation="Counter: total training steps",
            worker="consistency_check",
        )
    time.sleep(SCRAPE_WAIT_S)
    after = _promql_value(service_ip, f"{NS}_train_step_total" + LC)
    if after is not None:
        delta = after - base
        checks.append(
            ("counter +" + str(N), abs(delta - N) < 0.5, str(N), f"{delta:.1f}")
        )
        expected = base + N * AMOUNT
        checks.append(
            (
                "counter value = " + str(int(expected)),
                abs(after - expected) < 0.5,
                str(int(expected)),
                f"{after:.0f}",
            )
        )
    else:
        checks.append(("counter", False, str(N), "no data"))

    # --- Gauge ---
    GAUGE_VALUES = [1.23, 4.56, 7.89]
    for v in GAUGE_VALUES:
        insight.metric_gauge(
            "reward_mean",
            value=v,
            documentation="Gauge: mean reward value",
            worker="consistency_check",
        )
    time.sleep(SCRAPE_WAIT_S)
    after = _promql_value(service_ip, f"{NS}_reward_mean" + LC)
    expected = GAUGE_VALUES[-1]
    ok = after is not None and abs(after - expected) < 0.01
    checks.append(
        (
            "gauge = " + str(expected),
            ok,
            str(expected),
            f"{after:.4f}" if after is not None else "no data",
        )
    )

    # --- Histogram: count AND sum ---
    base_count = prometheus_before.get(
        "hist_ck_count",
        _promql_value(service_ip, f"{NS}_step_latency_ms_count" + LC) or 0.0,
    )
    base_sum = prometheus_before.get(
        "hist_ck_sum",
        _promql_value(service_ip, f"{NS}_step_latency_ms_sum" + LC) or 0.0,
    )
    HIST_VALUES = [100.0, 200.0, 300.0]
    for v in HIST_VALUES:
        insight.metric_histogram(
            "step_latency_ms",
            value=v,
            documentation="Histogram: step latency in ms",
            worker="consistency_check",
        )
    time.sleep(SCRAPE_WAIT_S)
    after_count = _promql_value(service_ip, f"{NS}_step_latency_ms_count" + LC)
    after_sum_val = _promql_value(service_ip, f"{NS}_step_latency_ms_sum" + LC)
    M = len(HIST_VALUES)

    if after_count is not None:
        delta = after_count - base_count
        checks.append(
            ("histogram count +" + str(M), abs(delta - M) < 0.5, str(M), f"{delta:.1f}")
        )
    else:
        checks.append(("histogram count", False, str(M), "no data"))

    if after_sum_val is not None:
        expected_sum = base_sum + sum(HIST_VALUES)
        ok = abs(after_sum_val - expected_sum) < 1.0
        checks.append(
            (
                "histogram sum = " + str(int(expected_sum)),
                ok,
                str(int(expected_sum)),
                f"{after_sum_val:.0f}",
            )
        )
    else:
        checks.append(("histogram sum", False, str(int(sum(HIST_VALUES))), "no data"))

    failures = 0
    for label, ok, exp, act in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}: expected={exp}, actual={act}")
        if not ok:
            failures += 1

    print()
    if failures == 0:
        print("  All checks passed.  Count AND content exact match.")
    else:
        print(f"  {failures} checks FAILED.")
    return failures


# ---------------------------------------------------------------------------
# Full results dump
# ---------------------------------------------------------------------------


def print_full_results(all_results: list[ConcurrencyResult]) -> None:
    """Reprint every result for record-keeping."""
    print(f"\n{'=' * 70}")
    print("  Full Results (all combinations)")
    print(f"{'=' * 70}")
    for api_name in ["counter", "gauge", "histogram", "trace"]:
        api_results = [r for r in all_results if r.api_name == api_name]
        if not api_results:
            continue
        print(f"\n  -- {api_name} --")
        hdr2 = f"  {'P':>3} {'T':>3}  {'Submitted':>10} {'HubRcvd':>8}  {'Fail%':>6}  {'Avg(ms)':>8} {'p50(ms)':>8} {'p95(ms)':>8} {'Queue(ms)':>10}  {'Thru/s':>10}"
        print(hdr2)
        print("  " + "-" * 100)
        for r in api_results:
            lat = r.latency
            print(
                f"  {r.num_procs:>3} {r.num_threads:>3}  "
                f"{r.submitted:>10} {r.hub_delta:>8}  "
                f"{r.failure_rate:>5.1%}  {lat.avg * 1000:>7.1f} {lat.p50 * 1000:>7.1f} "
                f"{lat.p95 * 1000:>7.1f} {lat.queue_ms:>9.0f}  {r.throughput_per_sec:>10.0f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    service_ip = _server_ip()

    # -- Load checkpoint (support resume after interruption) --
    ckpt = _load_checkpoint()
    completed: set[str] = set(ckpt.get("completed", []))
    all_results: list[ConcurrencyResult] = []
    # Rebuild results from checkpoint
    for api_name, entries in ckpt.get("results", {}).items():
        for entry in entries:
            all_results.append(
                ConcurrencyResult(
                    api_name=api_name,
                    num_procs=entry["num_procs"],
                    num_threads=entry["num_threads"],
                    submitted=entry["submitted"],
                    submitted_sum=entry.get("submitted_sum", 0.0),
                    hub_delta=entry.get("hub_delta", 0),
                    failure_rate=entry.get("failure_rate", 0.0),
                    throughput_per_sec=entry.get("throughput_per_sec", 0.0),
                    latency=LatencyStats(**entry["latency"]),
                )
            )

    api_order = ["counter", "gauge", "histogram", "trace"]
    total_combos = len(api_order) * len(PROCESS_LEVELS) * len(THREAD_LEVELS)
    extra_combos = len(EXTRA_1P_THREADS) + len(EXTRA_10T_PROCS)
    total_combos = len(api_order) * (
        len(PROCESS_LEVELS) * len(THREAD_LEVELS) + extra_combos
    )
    skipped = len(completed)

    print("=" * 70)
    print("  RL-Insight Monitor  --  Stress Test Report")
    print("=" * 70)
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}    server: {service_ip}")
    print(f"  processes {PROCESS_LEVELS}  x  threads {THREAD_LEVELS}")
    print(f"  extra: 1p × {EXTRA_1P_THREADS}, {EXTRA_10T_PROCS} × 10t")
    print(f"  {STRESS_DURATION_S}s per combination, {total_combos} total")
    if skipped:
        print(f"  Resuming: {skipped}/{total_combos} already completed")
    print()

    # -- Setup --
    try:
        ray.init(
            address="auto", namespace="rl-insight-monitor", ignore_reinit_error=True
        )
    except ConnectionError:
        ray.init(namespace="rl-insight-monitor", ignore_reinit_error=True)

    os.environ.setdefault("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080")

    insight.init(
        project="verl",
        experiment_name="ppo-stress-test",
        config={"server": {"service_ip": service_ip}},
    )
    if not _api._STATE.enabled:
        print("  FAIL: monitoring not enabled.")
        return 1

    # Remember the job-scoped actor name so child processes can connect
    # to the *same* hub instead of trying to create their own (which
    # would collide on the Prometheus metrics port).
    hub_actor_name = _current_job_actor_name()
    hub_events = _hub_events_count()
    print(f"  Ray: OK  |  Monitor: OK  |  Hub events: {hub_events}")

    # -- Prometheus baseline snapshot (before stress) --
    # Use saved baseline from checkpoint if resuming, otherwise capture fresh
    LS = '{worker="stress"}'
    LC = '{worker="consistency_check"}'
    prom_before = ckpt.get("prometheus_before", {})
    if not prom_before:
        prom_before = {
            "counter": _promql_value(service_ip, f"{NS}_train_step_total" + LS) or 0.0,
            "hist_count": _promql_value(service_ip, f"{NS}_step_latency_ms_count" + LS)
            or 0.0,
            "hist_sum": _promql_value(service_ip, f"{NS}_step_latency_ms_sum" + LS)
            or 0.0,
            "counter_ck": _promql_value(service_ip, f"{NS}_train_step_total" + LC)
            or 0.0,
            "hist_ck_count": _promql_value(
                service_ip, f"{NS}_step_latency_ms_count" + LC
            )
            or 0.0,
            "hist_ck_sum": _promql_value(service_ip, f"{NS}_step_latency_ms_sum" + LC)
            or 0.0,
        }
        ckpt["prometheus_before"] = prom_before
        _save_checkpoint(ckpt)

    # -- Stress tests (all process × thread × api combinations) --
    print(f"\n{'=' * 70}")
    print("  Stress Test Results  (P=processes, T=threads, queue=p95-p50)")
    print(f"{'=' * 70}")

    total_failures = 0

    for api_name in api_order:
        track = api_name == "histogram"
        print(f"\n  -- {api_name} --")
        print_header()

        for num_procs in PROCESS_LEVELS:
            for num_threads in THREAD_LEVELS:
                key = _combo_key(api_name, num_procs, num_threads)
                if key in completed:
                    existing = [
                        r
                        for r in all_results
                        if r.api_name == api_name
                        and r.num_procs == num_procs
                        and r.num_threads == num_threads
                    ]
                    if existing:
                        print_row(existing[0])
                    continue

                result = run_concurrency_test(
                    api_name,
                    num_procs,
                    num_threads,
                    hub_actor_name=hub_actor_name,
                    track_sum=track,
                )
                all_results.append(result)
                completed.add(key)
                print_row(result)

                if api_name not in ckpt["results"]:
                    ckpt["results"][api_name] = []
                ckpt["results"][api_name].append(
                    {
                        "num_procs": result.num_procs,
                        "num_threads": result.num_threads,
                        "submitted": result.submitted,
                        "submitted_sum": result.submitted_sum,
                        "hub_delta": result.hub_delta,
                        "failure_rate": result.failure_rate,
                        "throughput_per_sec": result.throughput_per_sec,
                        "latency": {
                            "avg": result.latency.avg,
                            "p50": result.latency.p50,
                            "p95": result.latency.p95,
                            "p99": result.latency.p99,
                            "queue_ms": result.latency.queue_ms,
                        },
                    }
                )
                ckpt["completed"] = list(completed)
                _save_checkpoint(ckpt)
                time.sleep(1)

        # -- Extra: 1-process high-thread stress --
        print(f"\n  -- {api_name} (extra: 1p, high threads) --")
        for num_threads in EXTRA_1P_THREADS:
            key = _combo_key(api_name, 1, num_threads)
            if key in completed:
                existing = [
                    r
                    for r in all_results
                    if r.api_name == api_name
                    and r.num_procs == 1
                    and r.num_threads == num_threads
                ]
                if existing:
                    print_row(existing[0])
                continue

            result = run_concurrency_test(
                api_name,
                1,
                num_threads,
                hub_actor_name=hub_actor_name,
                track_sum=track,
            )
            all_results.append(result)
            completed.add(key)
            print_row(result)

            if api_name not in ckpt["results"]:
                ckpt["results"][api_name] = []
            ckpt["results"][api_name].append(
                {
                    "num_procs": result.num_procs,
                    "num_threads": result.num_threads,
                    "submitted": result.submitted,
                    "submitted_sum": result.submitted_sum,
                    "hub_delta": result.hub_delta,
                    "failure_rate": result.failure_rate,
                    "throughput_per_sec": result.throughput_per_sec,
                    "latency": {
                        "avg": result.latency.avg,
                        "p50": result.latency.p50,
                        "p95": result.latency.p95,
                        "p99": result.latency.p99,
                        "queue_ms": result.latency.queue_ms,
                    },
                }
            )
            ckpt["completed"] = list(completed)
            _save_checkpoint(ckpt)
            time.sleep(1)

        # -- Extra: 10-thread high-process stress --
        print(f"\n  -- {api_name} (extra: 10t, high processes) --")
        for num_procs in EXTRA_10T_PROCS:
            key = _combo_key(api_name, num_procs, 10)
            if key in completed:
                existing = [
                    r
                    for r in all_results
                    if r.api_name == api_name
                    and r.num_procs == num_procs
                    and r.num_threads == 10
                ]
                if existing:
                    print_row(existing[0])
                continue

            result = run_concurrency_test(
                api_name,
                num_procs,
                10,
                hub_actor_name=hub_actor_name,
                track_sum=track,
            )
            all_results.append(result)
            completed.add(key)
            print_row(result)

            if api_name not in ckpt["results"]:
                ckpt["results"][api_name] = []
            ckpt["results"][api_name].append(
                {
                    "num_procs": result.num_procs,
                    "num_threads": result.num_threads,
                    "submitted": result.submitted,
                    "submitted_sum": result.submitted_sum,
                    "hub_delta": result.hub_delta,
                    "failure_rate": result.failure_rate,
                    "throughput_per_sec": result.throughput_per_sec,
                    "latency": {
                        "avg": result.latency.avg,
                        "p50": result.latency.p50,
                        "p95": result.latency.p95,
                        "p99": result.latency.p99,
                        "queue_ms": result.latency.queue_ms,
                    },
                }
            )
            ckpt["completed"] = list(completed)
            _save_checkpoint(ckpt)
            time.sleep(1)

    # -- Summary --
    print_api_summary(all_results)

    # -- Analysis --
    stress_failures = print_analysis(all_results)
    total_failures += stress_failures

    # -- Stress aggregate (exact count + sum) --
    total_failures += _verify_stress_aggregate(service_ip, all_results, prom_before)

    # -- Grafana --
    total_failures += _verify_grafana_frontend(service_ip)

    # -- Data consistency (known values) --
    total_failures += _verify_data_consistency(service_ip, prom_before)

    # -- Full results dump --
    print_full_results(all_results)

    print(f"\n{'=' * 70}")
    if total_failures == 0:
        print("  Done — all checks passed.")
    else:
        print(f"  Done — {total_failures} check(s) FAILED.")
    print(f"{'=' * 70}\n")
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
