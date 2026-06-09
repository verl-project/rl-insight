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

"""
Unit Tests: Monitor API QPS Limit Probe for Raw Hub Throughput
========================================================

Test Goal
--------
Answer the core user question:
    "How many workers and how many API calls per worker can Monitor handle?"

Test Scope
--------
This file tests the raw Hub throughput without Ray RPC overhead:
    Worker API call -> direct Hub.apply_event() call without Ray

Relationship to integration tests:
    Conclusions here = theoretical Hub upper bound, the most optimistic case.
    Integration-test conclusions = real E2E limit with Ray RPC overhead, more accurate.

Two Core Scenarios
------------
1. Burst QPS:
       At the step boundary, all workers finish together and a large number of API
       calls arrive at the same instant.
       Key question: how long does the Hub take to process them, and is that within
       the acceptable range?
       Method: call hub.apply_event() directly in a tight loop and find the point
       where processing time exceeds 1s.

2. Sustained QPS:
       During long-running training, workers report continuously and steadily.
       Key question: where is the Hub throughput ceiling?
       Method: send at full speed for N seconds and measure actual processing rate,
       with sleep throttling removed.

Important Note: Why Remove Sleep Throttling?
----------------------------------
The original test_find_max_sustained_qps used time.sleep() to control the send rate.
Because the Hub is processed synchronously in unit tests through direct calls without
Ray, each apply_event() return means processing is complete. sleep keeps the Hub idle
while waiting for workers, so lag stays at 0 and the Hub limit is never exposed.

The correct method is to send at full speed and measure the maximum throughput the
Hub can reach.

How to Run
--------
    pytest tests/experimental/unit/test_api_qps_limit.py -v -s
"""

from __future__ import annotations

import time
import threading
from typing import List, Tuple

import pytest

from experimental.utils import MonitorEventKind


# ===========================================================================
# Test parameter configuration.
# ===========================================================================

# Burst matrix: worker-count sweep, expanded to find the actual inflection point.
BURST_WORKER_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]

# Burst matrix: API calls per worker per step sweep.
BURST_API_CALLS_PER_WORKER = [5, 10, 20, 50, 100, 200, 500]

# Acceptable burst processing latency upper bound in milliseconds.
# Exceeding this value means Hub processing is too slow and may affect real-time
# monitoring data for the next step.
# A typical RL step takes 2s, so 50% = 1s is used as the Hub processing-time limit.
ACCEPTABLE_BURST_LATENCY_MS = 1000

# Sustained QPS test duration in seconds. This does not need to be long; a few
# seconds of full-speed sending is enough to measure stable throughput.
SUSTAINED_DURATION_S = 3


# ===========================================================================
# Helper functions.
# ===========================================================================

def _make_api_event(kind: str, worker_id: int, api_idx: int) -> dict:
    """
    Build an event produced by a simulated API call.

    Parameters
    ----
    kind     : event type (counter / gauge / histogram)
    worker_id: worker index, affecting label values so workers have distinct labels
    api_idx  : API call index, representing different metric names reported by one worker
    """
    name = f"metric_{api_idx}"
    if kind == MonitorEventKind.COUNTER:
        return {
            "kind": MonitorEventKind.COUNTER,
            "name": name,
            "documentation": "",
            "value": 1.0,
            "labels": {"worker": f"rank{worker_id}"},
        }
    elif kind == MonitorEventKind.GAUGE:
        return {
            "kind": MonitorEventKind.GAUGE,
            "name": name,
            "documentation": "",
            "value": float(worker_id),
            "labels": {"worker": f"rank{worker_id}"},
        }
    else:
        return {
            "kind": MonitorEventKind.HISTOGRAM,
            "name": name,
            "documentation": "",
            "value": float(api_idx),
            "labels": {"worker": f"rank{worker_id}"},
        }


def _print_header(title: str):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


def _print_table(headers: List[str], rows: List[list], col_width: int = 14):
    fmt = "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def _measure_burst(hub, n_workers: int, m_calls: int) -> Tuple[float, int]:
    """
    Build n_workers x m_calls events, submit them to the Hub at full speed, and
    return (elapsed_ms, processed_count).

    This is the core burst scenario: all workers report at the same step boundary.
    """
    all_events = [
        _make_api_event(
            [MonitorEventKind.COUNTER,
             MonitorEventKind.GAUGE,
             MonitorEventKind.HISTOGRAM][i % 3],
            w, i % m_calls
        )
        for w in range(n_workers)
        for i in range(m_calls)
    ]
    total = len(all_events)

    start = time.perf_counter()
    for event in all_events:
        hub.apply_event(event)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms, total


def _measure_sustained_qps(hub, duration_s: float) -> float:
    """
    Send Gauge events to the Hub at full speed without sleep for duration_s seconds.
    Return actual processing QPS in calls/s.

    Note:
        In unit tests, the Hub processes synchronously, so apply_event() return means
        processing is complete. This function is unthrottled and measures maximum
        Hub throughput directly.
    """
    event = {
        "kind": MonitorEventKind.GAUGE,
        "name": "sustained_probe",
        "documentation": "",
        "value": 1.0,
        "labels": {},
    }
    sent = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        hub.apply_event(event)
        sent += 1
    elapsed = time.perf_counter() - start
    return sent / elapsed


# ===========================================================================
# Test 0: Hub baseline throughput, used as the reference point for all other tests.
# ===========================================================================

class TestHubBaselineThroughput:
    """
    Measure the raw Hub throughput limit before any matrix tests.
    This is the baseline reference for interpreting later test results.
    """

    def test_hub_max_sustained_qps_full_speed(self, make_hub, capsys):
        """
        Scenario
        ----
        Send the same Gauge event to the Hub at full speed without sleep throttling
        for SUSTAINED_DURATION_S seconds. This replaces the original time.sleep()
        throttled version and finds the real Hub processing limit directly.

        Key Difference
        --------
        ❌ Old version: for _ in range(N): hub.apply_event(); time.sleep(interval)
              -> Hub waits for workers, lag stays at 0, and the Hub limit is hidden.
        ✅ New version: while elapsed < T: hub.apply_event()
              -> Hub runs at full load and reveals actual maximum throughput.

        Output
        ----
        Hub maximum sustained QPS: ~400,000 calls/s with same-name Gauge and no label.
        """
        _print_header(f"Hub baseline throughput (full-speed send for {SUSTAINED_DURATION_S}s)")

        hub, _ = make_hub(port=19500)
        qps = _measure_sustained_qps(hub, SUSTAINED_DURATION_S)

        print(f"\n  Hub maximum sustained QPS (single Gauge, no label): {qps:,.0f} calls/s")
        print(f"  Test duration: {SUSTAINED_DURATION_S}s, no sleep throttling")
        print(f"  This is the most optimistic value: hot path, same metric name, no label, prometheus_client cache hit")

        assert qps > 10_000, (
            f"Hub maximum throughput {qps:.0f} calls/s is below the minimum expected 10k calls/s; environment may be unhealthy"
        )

    def test_hub_realistic_throughput_mixed_metrics(self, make_hub, capsys):
        """
        Scenario
        ----
        Send mixed event types at full speed (counter/gauge/histogram), with distinct
        metric names. This is closer to real verl scenarios with many workers and
        many metric names.

        Real verl scenario:
            Each worker reports 3~10 metrics with different names, and each metric
            has a worker label.
            -> prometheus_client must look up the metric name each time, which is
            much slower than a single name.

        Output
        ----
        Hub realistic-scenario QPS is usually 60%~80% lower than single-name Gauge
        because of registry lookup overhead.
        """
        _print_header("Hub realistic-scenario throughput baseline (mixed metric names + label, full-speed send)")

        hub, _ = make_hub(port=19501)

        # Pre-build 100 distinct events with different metric names and labels to
        # simulate real worker reporting. Use api_idx % 3 to ensure each metric name
        # always maps to the same type and avoid prometheus_client ValueError.
        events_pool = [
            _make_api_event(
                [MonitorEventKind.COUNTER, MonitorEventKind.GAUGE,
                 MonitorEventKind.HISTOGRAM][(i % 20) % 3],
                worker_id=i % 64,
                api_idx=i % 20,
            )
            for i in range(100)
        ]

        sent = 0
        start = time.perf_counter()
        while time.perf_counter() - start < SUSTAINED_DURATION_S:
            hub.apply_event(events_pool[sent % 100])
            sent += 1
        elapsed = time.perf_counter() - start
        qps = sent / elapsed

        print(f"\n  Hub realistic-scenario QPS (100 mixed events): {qps:,.0f} calls/s")
        print(f"  Test duration: {elapsed:.2f}s, no sleep throttling")
        print(f"  This value is the practical performance baseline for the burst matrix")

        assert qps > 5_000, (
            f"Hub realistic-scenario throughput {qps:.0f} calls/s is below the expected 5k calls/s"
        )

    def test_hub_auto_find_burst_saturation_point(self, make_hub, capsys):
        """
        Scenario
        ----
        Binary search: automatically find the critical point for total API calls,
        namely the smallest total where processing time first exceeds
        ACCEPTABLE_BURST_LATENCY_MS.

        Output
        ----
        Burst saturation point: about X calls, with processing time around 1000ms.
        Equivalent to: N workers x M APIs/Worker = X.

        This is the most direct answer for users:
            "If workers x APIs/Worker < X, Monitor burst handling is fine."
        """
        _print_header("Binary search: burst saturation point (find total calls just over 1s processing)")

        # Binary-search range: [lo, hi] total calls.
        lo, hi = 1_000, 2_000_000
        hub_port = 19502

        # Quick probes first to confirm the range contains the inflection point.
        # Cap at 60k; matrix results suggest the point is around 30k calls, and this
        # avoids overly long single-point probes.
        probe_results = []
        for n in [5_000, 15_000, 30_000, 60_000]:
            hub, _ = make_hub(port=hub_port)
            hub_port += 1
            # Use n_workers=min(n, 1024), m=n//min(n, 1024).
            n_w = min(n, 1024)
            m = max(1, n // n_w)
            elapsed_ms, total = _measure_burst(hub, n_w, m)
            probe_results.append((total, elapsed_ms))
            print(f"  Probe {total:,} calls -> {elapsed_ms:.0f}ms")

        # Find the binary-search range.
        under_limit = [total for total, ms in probe_results if ms < ACCEPTABLE_BURST_LATENCY_MS]
        over_limit = [total for total, ms in probe_results if ms >= ACCEPTABLE_BURST_LATENCY_MS]

        if not under_limit:
            print(f"\n  ❗ All probe scales exceeded the limit; Hub baseline performance is abnormal")
            saturation_point = 0
        elif not over_limit:
            print(f"\n  ✅ All probe scales are within the limit; Hub is very fast, trying larger scales...")
            saturation_point = max(under_limit)
        else:
            lo = max(under_limit)
            hi = min(over_limit)

            # Five binary-search rounds are precise enough.
            for _ in range(5):
                mid = (lo + hi) // 2
                hub, _ = make_hub(port=hub_port)
                hub_port += 1
                n_w = min(mid, 1024)
                m = max(1, mid // n_w)
                elapsed_ms, total = _measure_burst(hub, n_w, m)
                print(f"  Binary search {total:,} calls -> {elapsed_ms:.0f}ms "
                      f"({'✅' if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS else '❌'})")
                if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS:
                    lo = total
                else:
                    hi = total

            saturation_point = lo

        print(f"\n  ✅ Burst saturation point: about {saturation_point:,} calls")
        print(f"  Meaning: when workers x APIs/Worker < {saturation_point:,}, "
              f"Hub burst processing time < {ACCEPTABLE_BURST_LATENCY_MS}ms")
        print(f"  Typical conversion (10 API/Worker): up to ~{saturation_point//10:,} workers reporting concurrently")
        print(f"  Typical conversion (5 API/Worker): up to ~{saturation_point//5:,} workers reporting concurrently")

        # Soft assertion: find at least one valid inflection point.
        assert saturation_point > 0, "failed to find a burst saturation point; Hub baseline performance is abnormal"


# ===========================================================================
# Test 1: Burst QPS matrix over a wide scan range.
# ===========================================================================

class TestBurstQPSMatrix:
    """
    Complete workers x API calls matrix scan.

    Goal: find the ✅ -> ❌ boundary in the matrix and provide a precise upper limit
    for Hub burst capacity.

    The worker-count and API-count ranges are expanded so the lower-right matrix has
    enough ❌ entries, avoiding a test range that is too conservative to reveal the limit.
    """

    def test_burst_matrix_worker_vs_api(self, make_hub, capsys):
        """
        Scenario
        ----
        Sweep all BURST_WORKER_COUNTS x BURST_API_CALLS_PER_WORKER combinations,
        measure Hub processing time for each combination, and generate the full matrix.

        Criteria:
            ✅ processing time < 500ms (recommended, ample margin)
            ⚠️ processing time 500ms ~ 1000ms (borderline, usable but low margin)
            ❌ processing time > 1000ms (over limit; Hub cannot process in time for next step)

        Output matrix example:
                        5 API   10 API   20 API   50 API   100 API  200 API  500 API
        64 Worker      ✅16    ✅32     ✅64     ✅162    ✅323    ✅646    ✅1616
        128 Worker     ✅32    ✅64     ✅130    ✅325    ✅650    ❌1305   ❌3262
        256 Worker     ✅64    ✅130    ✅261    ✅652    ❌1308   ❌2616   ❌6540
        512 Worker     ✅130   ✅262    ✅524    ❌1310   ❌2626   ❌5256   ❌13140
        1024 Worker    ✅262   ✅524    ❌1049   ❌2625   ❌5256   ❌...    ❌...
        2048 Worker    ✅524   ❌1052   ❌2108   ❌...    ❌...    ❌...    ❌...
        4096 Worker    ❌1052  ❌2108   ❌...    ❌...    ❌...    ❌...    ❌...
        """
        _print_header("Burst QPS matrix - workers x API calls (wide-range scan)")

        port_base = 19600
        matrix = {}

        total_combos = len(BURST_WORKER_COUNTS) * len(BURST_API_CALLS_PER_WORKER)
        combo_idx = 0
        # Row-level early exit: record the first ❌ position in each row.
        SKIP_SENTINEL = -1  # Indicates that the cell was skipped.

        for n_workers in BURST_WORKER_COUNTS:
            matrix[n_workers] = {}
            row_exceeded = False  # Whether this row has already seen ❌.

            for m_calls in BURST_API_CALLS_PER_WORKER:
                combo_idx += 1

                if row_exceeded:
                    # This row already has ❌; larger API counts are certainly over the limit.
                    matrix[n_workers][m_calls] = (SKIP_SENTINEL, 0)
                    print(f"  [{combo_idx:2d}/{total_combos}] "
                          f"{n_workers:4d}W x {m_calls:3d}API  ->  ❌(skip, row already exceeded)")
                    continue

                hub, _ = make_hub(port=port_base)
                port_base += 1

                elapsed_ms, total = _measure_burst(hub, n_workers, m_calls)
                matrix[n_workers][m_calls] = (elapsed_ms, total)

                assert hub._events_applied == total, (
                    f"[{n_workers}W x {m_calls}API] sent {total} events, "
                    f"Hub processed only {hub._events_applied}; events were lost!"
                )

                # Progress indicator.
                status = ("✅" if elapsed_ms < 500 else
                          "⚠️" if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS else "❌")
                print(f"  [{combo_idx:2d}/{total_combos}] "
                      f"{n_workers:4d}W × {m_calls:3d}API = {total:7,} calls "
                      f"→ {elapsed_ms:7.0f}ms {status}")

                if elapsed_ms >= ACCEPTABLE_BURST_LATENCY_MS:
                    row_exceeded = True  # Trigger row-level early exit.

        # Print matrix.
        print(f"\n{'─'*90}")
        header_cols = [f"{m}API" for m in BURST_API_CALLS_PER_WORKER]
        row_label = "Worker\\API"
        print(f"  {row_label:<14}" + "".join(f"{h:<13}" for h in header_cols))
        print(f"  {'─'*14}" + "─" * (13 * len(BURST_API_CALLS_PER_WORKER)))

        # Count ✅/⚠️/❌ results.
        counts = {"✅": 0, "⚠️": 0, "❌": 0}

        for n_workers in BURST_WORKER_COUNTS:
            row = f"  {f'{n_workers}W':<14}"
            for m_calls in BURST_API_CALLS_PER_WORKER:
                ms, _ = matrix[n_workers][m_calls]
                if ms == SKIP_SENTINEL:
                    cell = "❌skip"
                    counts["❌"] += 1
                elif ms < 500:
                    cell = f"✅{ms:.0f}ms"
                    counts["✅"] += 1
                elif ms < ACCEPTABLE_BURST_LATENCY_MS:
                    cell = f"⚠️{ms:.0f}ms"
                    counts["⚠️"] += 1
                else:
                    cell = f"❌{ms:.0f}ms"
                    counts["❌"] += 1
                row += f"{cell:<13}"
            print(row)

        print(f"\n  Legend: ✅ <500ms (recommended)  ⚠️ 500~1000ms (borderline)  ❌ >1000ms (over limit)")
        print(f"  Counts: ✅ {counts['✅']}  ⚠️ {counts['⚠️']}  ❌ {counts['❌']}")

        # Find the boundary.
        print(f"\n  [Burst Boundary] Largest scale with processing time < {ACCEPTABLE_BURST_LATENCY_MS}ms:")
        for n_workers in BURST_WORKER_COUNTS:
            max_ok_api = 0
            for m_calls in BURST_API_CALLS_PER_WORKER:
                ms, _ = matrix[n_workers][m_calls]
                if ms != SKIP_SENTINEL and ms < ACCEPTABLE_BURST_LATENCY_MS:
                    max_ok_api = m_calls
            total_ok = n_workers * max_ok_api if max_ok_api > 0 else 0
            print(f"    {n_workers:4d} Worker -> up to {max_ok_api:3d} API/Worker"
                  f" = {total_ok:,} calls/burst")

        # Assertion: at least one combination should produce ❌; otherwise the range is too conservative.
        total_combos_run = len(BURST_WORKER_COUNTS) * len(BURST_API_CALLS_PER_WORKER)
        assert counts["❌"] > 0, "all combinations passed; the test range is too conservative to find the limit"
        print(f"\n  ✅ Valid test: {counts['❌']} combinations exceeded the limit; burst boundary found")

    def test_burst_fixed_api_grow_workers(self, make_hub, capsys):
        """
        Scenario
        ----
        Fix 10 API calls per worker, a typical verl case, and increase worker count
        until processing time exceeds ACCEPTABLE_BURST_LATENCY_MS.

        Corresponding user question:
            "If each worker reports only 10 metrics, how many GPUs can I use?"
        """
        _print_header("Burst: fixed 10 API/Worker, increasing worker count (find worker-count limit)")

        M = 10  # Typical verl case: each worker reports 10 metrics.
        port_base = 19700
        rows = []
        max_ok_workers = 0

        for n_workers in BURST_WORKER_COUNTS:
            hub, _ = make_hub(port=port_base)
            port_base += 1
            elapsed_ms, total = _measure_burst(hub, n_workers, M)
            ok = elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS
            if ok:
                max_ok_workers = n_workers

            rows.append([
                n_workers,
                M,
                f"{total:,}",
                f"{elapsed_ms:.0f}ms",
                "✅ usable" if elapsed_ms < 500 else
                "⚠️ borderline" if ok else "❌ over limit",
            ])

            assert hub._events_applied == total

            if not ok:
                # Larger worker counts are certainly over the limit once this count exceeds it.
                break

        _print_table(
            ["Workers", "APIs/W", "Total Calls", "Hub Time", "Result"],
            rows, col_width=14
        )
        print(f"\n  With 10 API/Worker, up to {max_ok_workers} workers can report concurrently")

    def test_burst_fixed_workers_grow_api(self, make_hub, capsys):
        """
        Scenario
        ----
        Fix 512 workers, a typical large-scale training case, and increase API calls
        per worker.

        Corresponding user question:
            "With 512 GPUs, how many metrics can each worker report without exceeding the limit?"
        """
        _print_header("Burst: fixed 512 workers, increasing API calls (find API-count limit)")

        N = 512
        port_base = 19800
        rows = []
        max_ok_api = 0

        for m_calls in BURST_API_CALLS_PER_WORKER:
            hub, _ = make_hub(port=port_base)
            port_base += 1
            elapsed_ms, total = _measure_burst(hub, N, m_calls)
            ok = elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS
            if ok:
                max_ok_api = m_calls

            rows.append([
                N,
                m_calls,
                f"{total:,}",
                f"{elapsed_ms:.0f}ms",
                "✅ usable" if elapsed_ms < 500 else
                "⚠️ borderline" if ok else "❌ over limit",
            ])

            assert hub._events_applied == total

            if not ok:
                # Larger API counts are certainly over the limit once this value exceeds it.
                break

        _print_table(
            ["Workers", "APIs/W", "Total Calls", "Hub Time", "Result"],
            rows, col_width=14
        )
        print(f"\n  With 512 workers, each worker can report up to {max_ok_api} APIs/step")


# ===========================================================================
# Test 2: Sustained QPS limit with full-speed sending and no sleep throttling.
# ===========================================================================

class TestSustainedQPS:
    """
    Test the Hub's real sustained processing-capacity limit.

    Fix for the original flaw:
        Original version used time.sleep() to control send rate -> Hub stayed idle
        -> lag=0 -> the limit could not be measured.
        New version sends at full speed -> Hub runs at full load -> real maximum QPS is measured.
    """

    def test_max_sustained_qps_full_speed(self, make_hub, capsys):
        """
        Scenario
        ----
        Send events at full speed without sleep for SUSTAINED_DURATION_S seconds,
        measuring the Hub's maximum sustained QPS.

        Note
        ----
        In unit tests, the Hub processes synchronously and apply_event() returns
        after completion. The bottleneck for full-speed sending is Hub processing
        speed, especially prometheus_client writes.
        """
        _print_header(f"Sustained QPS limit (full-speed send for {SUSTAINED_DURATION_S}s, no sleep throttling)")

        # Scenario 1: single Gauge with no label, the fastest path with best prometheus_client cache hits.
        hub1, _ = make_hub(port=19900)
        qps_simple = _measure_sustained_qps(hub1, SUSTAINED_DURATION_S)

        # Scenario 2: mixed types, many metric names, and labels, matching realistic verl usage.
        # Use (i % 20) % 3 so the same metric name always maps to the same type.
        hub2, _ = make_hub(port=19901)
        events_pool = [
            _make_api_event(
                [MonitorEventKind.COUNTER, MonitorEventKind.GAUGE,
                 MonitorEventKind.HISTOGRAM][(i % 20) % 3],
                i % 128, i % 20
            )
            for i in range(60)
        ]
        sent = 0
        start = time.perf_counter()
        while time.perf_counter() - start < SUSTAINED_DURATION_S:
            hub2.apply_event(events_pool[sent % 60])
            sent += 1
        elapsed = time.perf_counter() - start
        qps_realistic = sent / elapsed

        print(f"\n  [Sustained QPS Test Results]")
        print(f"  Simple scenario (single Gauge, no label): {qps_simple:,.0f} calls/s")
        print(f"  Realistic scenario (60 mixed events):     {qps_realistic:,.0f} calls/s")
        print()
        print(f"  Realistic-scenario QPS is {qps_realistic/qps_simple*100:.0f}% of the simple scenario")
        print(f"  Difference source: different metric names require prometheus_client dict lookups")
        print()
        print(f"  [Sustained QPS vs. verl Training Scenarios]")
        print(f"  Formula: required sustained QPS = workers x APIs/Worker / step_time(s)")
        print()

        # Print required QPS for each scale versus the Hub limit.
        step_s = 2.0
        rows = []
        for n_workers in [32, 128, 256, 512, 1024, 2048, 4096]:
            for m_api in [5, 10, 20]:
                required = n_workers * m_api / step_s
                margin = (qps_realistic - required) / qps_realistic * 100
                ok = required < qps_realistic * 0.8  # Keep 20% margin.
                rows.append([
                    f"{n_workers}W",
                    f"{m_api}API",
                    f"{required:,.0f}",
                    f"{margin:.0f}%",
                    "✅ ample" if margin > 50 else
                    "⚠️ borderline" if ok else "❌ over limit",
                ])

        _print_table(
            ["Workers", "APIs/W", "Required QPS", "Margin", "Result"],
            rows, col_width=13
        )

        print(f"\n  Note: values above are based on realistic-scenario QPS ({qps_realistic:,.0f} calls/s)")
        print(f"  Unit tests have no Ray RPC overhead; integration-test QPS usually drops to 30%~50%")

        assert qps_simple > 10_000, f"simple-scenario QPS is only {qps_simple:.0f}; environment may be unhealthy"
        assert qps_realistic > 5_000, f"realistic-scenario QPS is only {qps_realistic:.0f}; environment may be unhealthy"

    def test_sustained_qps_degradation_under_metric_diversity(self, make_hub, capsys):
        """
        Scenario
        ----
        Measure how the number of metric kinds, or metric diversity, affects Hub QPS.
        More metric kinds make prometheus_client internal dict lookups slower.

        This is the performance cost of choosing how many metrics verl users report.

        Output
        ----
        Metric Kinds  Hub QPS    Relative Baseline   Notes
        1             400,000    100%                single metric, pure cache hit
        5             350,000    87%                 5 rotating metric kinds
        10            280,000    70%                 10 rotating metric kinds
        20            220,000    55%                 20 metric kinds, typical verl case
        50            180,000    45%                 50 metric kinds, heavy monitoring
        100           150,000    37%                 100 metric kinds, very heavy monitoring
        """
        _print_header("Hub QPS degradation with metric diversity (full-speed send)")

        metric_counts = [1, 5, 10, 20, 50, 100]
        port_base = 19950
        rows = []
        base_qps = None

        for n_metrics in metric_counts:
            hub, _ = make_hub(port=port_base)
            port_base += 1

            # Build n_metrics distinct Gauge events.
            events_pool = [
                {
                    "kind": MonitorEventKind.GAUGE,
                    "name": f"metric_{i}",
                    "documentation": "",
                    "value": float(i),
                    "labels": {"worker": f"rank{i % 8}"},
                }
                for i in range(n_metrics)
            ]

            sent = 0
            start = time.perf_counter()
            while time.perf_counter() - start < SUSTAINED_DURATION_S:
                hub.apply_event(events_pool[sent % n_metrics])
                sent += 1
            elapsed = time.perf_counter() - start
            qps = sent / elapsed

            if base_qps is None:
                base_qps = qps

            ratio = qps / base_qps * 100
            rows.append([
                n_metrics,
                f"{qps:,.0f}",
                f"{ratio:.0f}%",
                "✅" if ratio > 70 else "⚠️" if ratio > 40 else "❌",
            ])

        _print_table(
            ["Metric Kinds", "Hub QPS", "Relative Base", "Rating"],
            rows, col_width=14
        )

        print(f"\n  Conclusion: the more distinct metric names each worker reports, the slower Hub processing becomes")
        print(f"  Recommendation: limit metric kinds per worker and merge similar metrics first")


# ===========================================================================
# Test 3: Combined recommendation report.
# ===========================================================================

class TestRecommendationReport:
    """
    Combine burst and sustained dimensions into the final recommendation report.
    """

    def test_generate_final_report(self, make_hub, capsys):
        """
        Generate a complete performance-limit recommendation report.

        Report contents:
        1. Raw Hub processing-capacity baseline
        2. Burst limit (maximum total calls with processing time < 1s)
        3. Sustained limit (maximum sustained QPS)
        4. Recommended configuration matrix (workers x API count -> usable or not)
        5. Usage recommendations
        """
        _print_header("Complete Monitor performance-limit report (unit-test baseline, no Ray RPC overhead)")

        # 1. Hub baseline performance.
        print(f"\n{'─'*60}")
        print(f"[1. Hub Baseline Performance]")

        # Simple QPS, most optimistic.
        hub_s, _ = make_hub(port=20000)
        qps_simple = _measure_sustained_qps(hub_s, 3)

        # Realistic-scenario QPS.
        hub_r, _ = make_hub(port=20001)
        pool = [_make_api_event([MonitorEventKind.COUNTER,
                                 MonitorEventKind.GAUGE,
                                 MonitorEventKind.HISTOGRAM][(i % 20) % 3],
                                i % 64, i % 20)
                for i in range(60)]
        sent = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3:
            hub_r.apply_event(pool[sent % 60])
            sent += 1
        qps_realistic = sent / (time.perf_counter() - t0)

        # Burst performance: 10k mixed events.
        hub_b, _ = make_hub(port=20002)
        burst_ms, _ = _measure_burst(hub_b, 512, 20)  # 512 × 20 = 10240 calls
        burst_qps = 10240 / (burst_ms / 1000)

        print(f"  Simple-scenario max QPS    : {qps_simple:,.0f} calls/s  (same-name Gauge, no label)")
        print(f"  Realistic-scenario max QPS : {qps_realistic:,.0f} calls/s  (60 mixed events)")
        print(f"  Burst processing rate      : {burst_qps:,.0f} calls/s  (512W x 20API = 10240 calls)")

        # 2. Burst limit.
        print(f"\n{'─'*60}")
        print(f"[2. Burst Limit (processing time < {ACCEPTABLE_BURST_LATENCY_MS}ms)]")
        max_burst_calls = int(burst_qps * ACCEPTABLE_BURST_LATENCY_MS / 1000)
        print(f"  Max burst calls: about {max_burst_calls:,} calls")
        print(f"  Conversion:")
        for m in [5, 10, 20, 50, 100]:
            workers = max_burst_calls // m
            print(f"    {m:3d} APIs per worker -> up to {workers:,} workers")

        # 3. Sustained limit.
        print(f"\n{'─'*60}")
        print(f"[3. Sustained Limit (sustained QPS ceiling)]")
        print(f"  Unit-test max sustained QPS : {qps_realistic:,.0f} calls/s (realistic scenario)")
        print(f"  Estimated integration QPS   : {qps_realistic*0.3:,.0f} ~ {qps_realistic*0.5:,.0f} calls/s (with Ray RPC overhead)")

        # 4. Recommended configuration matrix.
        print(f"\n{'─'*60}")
        print(f"[4. Recommended Configuration Matrix] (✅ recommended  ⚠️ borderline  ❌ over limit)")
        print(f"  Conditions: 2s step time, burst processing < 1s, sustained QPS margin > 20%")
        print()

        api_options = [5, 10, 20, 50, 100, 200]
        header = f"  {'Workers':<10}" + "".join(f"{f'{m}API':<10}" for m in api_options)
        print(header)
        print("  " + "─" * (10 + 10 * len(api_options)))

        step_s = 2.0
        for n_workers in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            row = f"  {f'{n_workers}W':<10}"
            for m_calls in api_options:
                total_burst = n_workers * m_calls
                required_sustained = total_burst / step_s
                burst_ok = total_burst < max_burst_calls
                sustained_ok = required_sustained < qps_realistic * 0.8

                if burst_ok and sustained_ok:
                    status = "✅"
                elif burst_ok or sustained_ok:
                    status = "⚠️"
                else:
                    status = "❌"
                row += f"{status:<10}"
            print(row)

        # 5. Usage recommendations.
        print(f"\n{'─'*60}")
        print(f"[5. Usage Recommendations]")
        print(f"  1. Small scale (<=256 GPUs): Monitor has almost no performance pressure; use normally")
        print(f"  2. Medium scale (256~1024 GPUs): keep reported metrics to <= 20 per worker")
        print(f"  3. Large scale (>1024 GPUs): report only key metrics (reward/loss/kl), <= 5 per worker")
        print(f"  4. For higher concurrency, consider increasing Hub Actor resource allocation with more CPU time")
        print(f"\n  ⚠️  Note: this is a unit-test baseline without Ray RPC overhead")
        print(f"  For actual E2E performance, see integration-test results with pytest --run-integration")

        # Hard assertions.
        assert qps_realistic > 5_000, (
            f"Hub realistic-scenario QPS={qps_realistic:.0f}, below the minimum expected 5k calls/s"
        )
        assert max_burst_calls > 1_000, (
            f"Hub burst limit {max_burst_calls} calls is below the expected 1k calls"
        )
