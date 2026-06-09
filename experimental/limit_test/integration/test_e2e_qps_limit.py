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
Integration Tests: Monitor End-to-End API QPS Limit with Real Ray RPC Overhead
====================================================================

Test Goal
--------
Answer the core user question with real Ray RPC overhead included:
    "How many workers and how many API calls per worker can Monitor handle end to end?"

Relationship to Unit Tests
----------------
Unit tests: direct Hub calls without Ray, measuring the theoretical upper bound.
This file: real Ray Actor plus fire-and-forget calls, measuring the practical E2E limit.

End-to-End Path
----------
Worker threads that simulate Ray workers
    ↓ actor.apply_event.remote()  ← Ray RPC with serialization plus network/IPC
    ↓
Hub Actor, a Ray Actor that processes events serially
    ↓ hub._events_applied counter increases
    ✅ processing complete

Key Optimizations
--------
1. Row-level early exit:
       After the first ❌ in a matrix row, larger API counts in that row are marked
       as ❌(skip) directly. This avoids long waits for combinations that clearly
       exceed the limit.

2. Shorter sustained-test duration:
       Reduced from 30s to 10s, while still collecting 5 lag snapshots for each
       worker scale. Early exit also stops testing larger worker counts after lag ❌.

3. Larger test scale with guardrails:
       WORKER_COUNTS is extended to 1024 and API_CALLS_PER_WORKER to 200.
       Row-level early exit prevents unbounded waits at very large scales.

How to Run
--------
    pytest tests/experimental/integration/test_e2e_qps_limit.py \
        -v -s --run-integration
"""

from __future__ import annotations

import time
import threading
from typing import List, Optional
from unittest.mock import patch

import pytest

ray = pytest.importorskip("ray", reason="ray must be installed")

from experimental.utils import MonitorEventKind


# ===========================================================================
# Test parameters.
# ===========================================================================

# Expanded scale; row-level early exit keeps the suite from timing out.
WORKER_COUNTS = [8, 32, 64, 128, 256, 512, 1024]
API_CALLS_PER_WORKER = [5, 10, 20, 50, 100, 200]

# Acceptable burst processing latency in milliseconds.
# Same as the unit tests: Hub burst processing time < 1s, or 50% of a typical step.
ACCEPTABLE_BURST_LATENCY_MS = 1000

# Sustained-test duration in seconds, reduced from 30s to 10s while still collecting
# 5 lag snapshots per scale.
SUSTAINED_DURATION_S = 10

# Acceptable lag growth threshold; lag growth below this value within 10s is stable.
ACCEPTABLE_LAG_GROWTH = 500

# Skip marker for matrix cells skipped by early exit.
SKIP_SENTINEL = -1.0


# ===========================================================================
# Fixture: Ray cluster.
# ===========================================================================

@pytest.fixture(scope="module")
def ray_local():
    """Start local Ray and shut it down after the tests."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            logging_level="WARNING",
        )
    yield
    ray.shutdown()


# ===========================================================================
# Helper functions.
# ===========================================================================

def _create_ray_hub(name: str, port: int):
    """Create a real Ray Hub Actor with the HTTP server and Prometheus reload patched out."""
    from experimental.collector.ray_monitor_hub import MonitorHubActor

    conf = {
        "namespace": f"test_qps_{port}",
        "prometheus": {
            "metrics_report_port": port,
            "reload": {"mode": "none"},
        },
        "otel": {"traces_endpoint": ""},
    }

    with patch("experimental.collector.ray_monitor_hub.start_metrics_http_server"), \
         patch("experimental.collector.ray_monitor_hub.update_prometheus_config"):
        actor = (
            MonitorHubActor
            .options(name=name, namespace="test-qps", lifetime="detached")
            .remote(conf)
        )
    return actor


def _cleanup(actor):
    try:
        ray.kill(actor, no_restart=True)
    except Exception:
        pass


def _make_event(worker_id: int, api_idx: int) -> dict:
    """
    Build an event that simulates an API call.

    Use api_idx % 3 to choose the type, ensuring the same metric name always maps
    to the same type and avoiding ValueError from registering one metric name with
    two prometheus_client types.
    """
    kind = [
        MonitorEventKind.COUNTER,
        MonitorEventKind.GAUGE,
        MonitorEventKind.HISTOGRAM,
    ][api_idx % 3]
    return {
        "kind": kind,
        "name": f"metric_{api_idx}",
        "documentation": "",
        "value": float(worker_id),
        "labels": {"worker": f"rank{worker_id}"},
    }


def _wait_hub_processed(actor, expected: int, timeout: float = 60.0) -> tuple:
    """
    Wait for the Hub to process expected events and return (processed_count, wait_s).
    On timeout, return the current processed count without raising.
    """
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        status = ray.get(actor.get_status.remote())
        if status["events_applied"] >= expected:
            return status["events_applied"], time.perf_counter() - start
        time.sleep(0.05)
    status = ray.get(actor.get_status.remote())
    return status["events_applied"], time.perf_counter() - start


def _burst_e2e(actor, n_workers: int, m_calls: int) -> tuple:
    """
    Have N worker threads send M APIs concurrently using fire-and-forget, then wait
    for the Hub to finish processing.

    Return (send_ms, hub_wait_ms, e2e_ms, processed, total).
    """
    total = n_workers * m_calls
    barrier = threading.Barrier(n_workers)

    def worker_fn(wid):
        barrier.wait()
        for idx in range(m_calls):
            actor.apply_event.remote(_make_event(wid, idx))

    threads = [threading.Thread(target=worker_fn, args=(wid,)) for wid in range(n_workers)]

    send_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    send_ms = (time.perf_counter() - send_start) * 1000

    processed, wait_s = _wait_hub_processed(actor, total, timeout=60.0)
    hub_wait_ms = wait_s * 1000
    e2e_ms = send_ms + hub_wait_ms

    return send_ms, hub_wait_ms, e2e_ms, processed, total


def _print_header(title: str):
    print(f"\n{'='*74}")
    print(f"  {title}")
    print(f"{'='*74}")


def _print_table(headers: List[str], rows: List[list], col_width: int = 14):
    fmt = "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ===========================================================================
# Test 1: End-to-end burst QPS.
# ===========================================================================

@pytest.mark.integration
class TestE2EBurstQPS:
    """
    End-to-end burst test: N worker threads send M API calls concurrently using
    fire-and-forget. Measure the real E2E time from send completion to Hub completion.
    """

    def test_burst_by_worker_count(self, ray_local, capsys):
        """
        Scenario
        ----
        Fix 10 API calls per worker, a typical verl case, and increase the worker count.
        Stop after the first ❌ because larger worker counts can be inferred.

        Corresponding user question:
            "If each worker reports 10 metrics, how many GPUs can be supported?"

        Output Report
        --------
        Workers  APIs  Total API Calls  Send Time(ms)  Hub Time(ms)  E2E Time(ms)  Result
        8        10    80               12.3           45.2          57.5          ✅ usable
        32       10    320              18.4           89.3          107.7         ✅ usable
        128      10    1280             45.2           312.4         357.6         ✅ usable
        256      10    2560             89.4           634.2         723.6         ⚠️ borderline
        512      10    5120             178.3          1284.5        1462.8        ❌ over limit (stop)
        """
        M = 10
        _print_header(f"E2E Burst - fixed {M} API/Worker, increasing worker count")

        rows = []
        max_ok_workers = 0

        for i, n_workers in enumerate(WORKER_COUNTS):
            actor = _create_ray_hub(f"hub_burst_w{n_workers}", port=21000 + i)

            try:
                send_ms, hub_ms, e2e_ms, processed, total = _burst_e2e(
                    actor, n_workers, M
                )
                ok = e2e_ms < ACCEPTABLE_BURST_LATENCY_MS
                if ok:
                    max_ok_workers = n_workers

                rows.append([
                    n_workers, M, total,
                    f"{send_ms:.0f}",
                    f"{hub_ms:.0f}",
                    f"{e2e_ms:.0f}",
                    "✅ usable" if e2e_ms < 500 else
                    "⚠️ borderline" if ok else "❌ over limit",
                ])

                assert processed == total, (
                    f"{n_workers}W: sent {total}, Hub processed {processed}, "
                    f"lost {total - processed} events!"
                )

            finally:
                _cleanup(actor)

            if not ok:
                print(f"  -> {n_workers}W exceeded the limit; stopping because larger scales can be inferred")
                break

        _print_table(
            ["Workers", "APIs", "Total", "Send(ms)", "Hub(ms)", "E2E(ms)", "Result"],
            rows, col_width=13
        )
        print(f"\n  E2E burst limit (10 API/W): up to {max_ok_workers} workers")

    def test_burst_matrix_e2e(self, ray_local, capsys):
        """
        Scenario
        ----
        Sweep WORKER_COUNTS x API_CALLS_PER_WORKER and generate an E2E matrix.
        Row-level early exit marks larger API counts in the same row as ❌(skip)
        after the first ❌.

        This is the final lookup table for users, including real Ray RPC overhead.

        Output matrix example:
                    5API      10API     20API     50API     100API    200API
        8W         ✅28ms    ✅54ms    ✅108ms   ✅271ms   ✅542ms   ⚠️921ms
        32W        ✅112ms   ✅218ms   ✅436ms   ⚠️891ms   ❌1784ms  ❌skip
        128W       ✅448ms   ⚠️872ms   ❌1744ms  ❌skip    ❌skip    ❌skip
        256W       ⚠️896ms   ❌1744ms  ❌skip    ❌skip    ❌skip    ❌skip
        512W       ❌1792ms  ❌skip    ❌skip    ❌skip    ❌skip    ❌skip
        """
        _print_header("E2E burst matrix - workers x API calls (final user lookup table)")

        port_base = 21100
        matrix = {}
        total_combos = len(WORKER_COUNTS) * len(API_CALLS_PER_WORKER)
        combo_idx = 0

        for n_workers in WORKER_COUNTS:
            matrix[n_workers] = {}
            row_exceeded = False

            for m_calls in API_CALLS_PER_WORKER:
                combo_idx += 1

                if row_exceeded:
                    matrix[n_workers][m_calls] = (SKIP_SENTINEL, 0, 0)
                    print(f"  [{combo_idx:2d}/{total_combos}] "
                          f"{n_workers:4d}W x {m_calls:3d}API  ->  ❌(skip, row already exceeded)")
                    continue

                actor = _create_ray_hub(
                    f"hub_mx_w{n_workers}_m{m_calls}", port=port_base
                )
                port_base += 1

                try:
                    send_ms, hub_ms, e2e_ms, processed, total = _burst_e2e(
                        actor, n_workers, m_calls
                    )
                    matrix[n_workers][m_calls] = (e2e_ms, processed, total)

                    status = ("✅" if e2e_ms < 500 else
                              "⚠️" if e2e_ms < ACCEPTABLE_BURST_LATENCY_MS else "❌")
                    print(f"  [{combo_idx:2d}/{total_combos}] "
                          f"{n_workers:4d}W × {m_calls:3d}API = {total:7,} calls "
                          f"->  send {send_ms:.0f}ms + Hub {hub_ms:.0f}ms "
                          f"= {e2e_ms:.0f}ms {status}")

                    assert processed == total, (
                        f"[{n_workers}W x {m_calls}API] sent {total}, "
                        f"processed {processed}, lost {total-processed} events!"
                    )

                    if e2e_ms >= ACCEPTABLE_BURST_LATENCY_MS:
                        row_exceeded = True

                finally:
                    _cleanup(actor)

        # Print matrix.
        print(f"\n{'─'*80}")
        header_cols = [f"{m}API" for m in API_CALLS_PER_WORKER]
        row_label = "Worker\\API"
        print(f"  {row_label:<12}" + "".join(f"{h:<12}" for h in header_cols))
        print(f"  {'─'*12}" + "─" * (12 * len(API_CALLS_PER_WORKER)))

        counts = {"✅": 0, "⚠️": 0, "❌": 0}

        for n_workers in WORKER_COUNTS:
            row = f"  {f'{n_workers}W':<12}"
            for m_calls in API_CALLS_PER_WORKER:
                e2e_ms, _, _ = matrix[n_workers][m_calls]
                if e2e_ms == SKIP_SENTINEL:
                    cell = "❌skip"
                    counts["❌"] += 1
                elif e2e_ms < 500:
                    cell = f"✅{e2e_ms:.0f}ms"
                    counts["✅"] += 1
                elif e2e_ms < ACCEPTABLE_BURST_LATENCY_MS:
                    cell = f"⚠️{e2e_ms:.0f}ms"
                    counts["⚠️"] += 1
                else:
                    cell = f"❌{e2e_ms:.0f}ms"
                    counts["❌"] += 1
                row += f"{cell:<12}"
            print(row)

        print(f"\n  Legend: ✅ <500ms (recommended)  ⚠️ 500~1000ms (borderline)  ❌ >1000ms (over limit)")
        print(f"  Counts: ✅ {counts['✅']}  ⚠️ {counts['⚠️']}  ❌ {counts['❌']}")

        # Boundary summary.
        print(f"\n  [E2E Burst Boundary] Largest scale with processing time < {ACCEPTABLE_BURST_LATENCY_MS}ms:")
        for n_workers in WORKER_COUNTS:
            max_ok_api = 0
            for m_calls in API_CALLS_PER_WORKER:
                e2e_ms, _, _ = matrix[n_workers][m_calls]
                if e2e_ms != SKIP_SENTINEL and e2e_ms < ACCEPTABLE_BURST_LATENCY_MS:
                    max_ok_api = m_calls
            total_ok = n_workers * max_ok_api if max_ok_api > 0 else 0
            print(f"    {n_workers:4d} Worker -> up to {max_ok_api:3d} API/Worker"
                  f" = {total_ok:,} calls/burst")

        assert counts["❌"] > 0, "all combinations passed; the test range is too conservative to find the limit"
        print(f"\n  ✅ Valid test: {counts['❌']} combinations exceeded the limit; E2E burst boundary found")


# ===========================================================================
# Test 2: End-to-end sustained QPS limit.
# ===========================================================================

@pytest.mark.integration
class TestE2ESustainedQPS:
    """
    End-to-end sustained QPS test: multiple worker threads send continuously at
    the step cadence and measure the Hub lag trend.

    Difference from sustained-QPS unit tests:
        Unit tests: send at full speed without sleep to find maximum Hub throughput.
        Integration tests: keep step-interval sleep to simulate real verl training
                  cadence and test whether lag is controlled in real scenarios.
    """

    def test_sustained_lag_by_worker_count(self, ray_local, capsys):
        """
        Scenario
        ----
        Fix 10 API calls per worker with a 2s step interval, increase worker count,
        run for SUSTAINED_DURATION_S seconds, and observe Hub lag trend.

        Stop after the first lag ❌ with rapid growth.

        Output Report
        --------
        Workers  Required QPS  Hub QPS  Lag Growth  Result
        32       160           950      +0          ✅ stable
        128      640           950      +12         ✅ stable
        256      1280          950      +148        ⚠️ slow growth
        512      2560          950      +1240       ❌ rapid growth (stop)
        """
        _print_header(
            f"E2E sustained QPS - fixed 10 API/W, 2s step, {SUSTAINED_DURATION_S}s duration"
        )

        M = 10
        STEP_S = 2.0
        rows = []
        max_ok_workers = 0

        for i, n_workers in enumerate(WORKER_COUNTS):
            required_qps = n_workers * M / STEP_S
            actor = _create_ray_hub(f"hub_sus_w{n_workers}", port=21300 + i)

            try:
                stop_flag = [False]
                sent_count = [0]
                lock = threading.Lock()

                def worker_continuous(wid, _actor=actor):
                    """Report a batch of M APIs every step_s seconds to simulate real verl cadence."""
                    while not stop_flag[0]:
                        for api_idx in range(M):
                            if stop_flag[0]:
                                break
                            _actor.apply_event.remote(_make_event(wid, api_idx))
                            with lock:
                                sent_count[0] += 1
                        time.sleep(STEP_S)

                threads = [
                    threading.Thread(target=worker_continuous, args=(wid,), daemon=True)
                    for wid in range(n_workers)
                ]
                for t in threads:
                    t.start()

                # Sample lag every 2s.
                lag_samples = []
                for _ in range(SUSTAINED_DURATION_S // 2):
                    time.sleep(2.0)
                    status = ray.get(actor.get_status.remote())
                    lag = sent_count[0] - status["events_applied"]
                    lag_samples.append(lag)

                stop_flag[0] = True
                for t in threads:
                    t.join(timeout=5)

                # Analyze lag trend.
                if len(lag_samples) >= 2:
                    lag_growth = lag_samples[-1] - lag_samples[0]
                else:
                    lag_growth = 0

                if lag_growth < 100:
                    trend = f"stable(+{lag_growth:.0f})"
                    sustainable = True
                    max_ok_workers = n_workers
                elif lag_growth < ACCEPTABLE_LAG_GROWTH:
                    trend = f"slow growth(+{lag_growth:.0f})"
                    sustainable = False
                else:
                    trend = f"rapid growth(+{lag_growth:.0f})"
                    sustainable = False

                final = ray.get(actor.get_status.remote())
                hub_qps = final["events_applied"] / SUSTAINED_DURATION_S

                rows.append([
                    n_workers,
                    f"{required_qps:.0f}",
                    f"{hub_qps:.0f}",
                    trend,
                    "✅ stable" if lag_growth < 100 else
                    "⚠️ borderline" if lag_growth < ACCEPTABLE_LAG_GROWTH else "❌ unsustainable",
                ])

            finally:
                _cleanup(actor)

            if not sustainable:
                print(f"  -> {n_workers}W lag growth exceeded the limit; stopping because larger scales can be inferred")
                break

        _print_table(
            ["Workers", "Required QPS", "Hub QPS", "Lag Trend", "Result"],
            rows, col_width=18
        )
        print(f"\n  E2E sustained QPS limit: {max_ok_workers} Worker x {M} API, step {STEP_S}s")
        print(f"  Criterion: lag growth < 100 events within {SUSTAINED_DURATION_S}s")


# ===========================================================================
# Test 3: Final end-to-end recommendation report.
# ===========================================================================

@pytest.mark.integration
class TestE2EFinalReport:
    """
    Combine burst and sustained QPS into a complete E2E recommendation report,
    including the Ray RPC reduction factor.
    """

    def test_final_recommendation_report(self, ray_local, capsys):
        """
        Combine burst and sustained QPS into the final recommendation report.

        Report contents:
        1. E2E performance baseline (send QPS + Hub QPS + RPC reduction factor)
        2. Burst limit (maximum total calls with processing time < 1s)
        3. Sustained QPS limit
        4. Recommended configuration matrix (GPU count x API count -> usable or not)
        5. Usage recommendations
        """
        _print_header("Complete E2E recommendation report - verl training scale supported by Monitor with Ray RPC overhead")

        # 1. E2E performance baseline.
        actor_base = _create_ray_hub("hub_report_base", port=21500)

        try:
            # Send 1000 events sequentially from one thread to measure fire-and-forget send rate.
            N = 1_000
            event = {
                "kind": MonitorEventKind.GAUGE,
                "name": "benchmark",
                "documentation": "",
                "value": 1.0,
                "labels": {"worker": "w0"},
            }
            t0 = time.perf_counter()
            for _ in range(N):
                actor_base.apply_event.remote(event)
            send_elapsed = time.perf_counter() - t0
            send_qps = N / send_elapsed

            processed, wait_s = _wait_hub_processed(actor_base, N, timeout=30.0)
            hub_qps = N / (send_elapsed + wait_s)

        finally:
            _cleanup(actor_base)

        print(f"\n[1. E2E Performance Baseline]")
        print(f"  fire-and-forget send rate : {send_qps:,.0f} calls/s  (single-thread sequential send)")
        print(f"  E2E throughput (send+process): {hub_qps:,.0f} calls/s")
        print(f"  Send time                 : {send_elapsed*1000:.1f}ms ({N} events)")
        print(f"  Hub processing wait       : {wait_s*1000:.1f}ms")

        # 2. Burst limit.
        actor_burst = _create_ray_hub("hub_report_burst", port=21501)

        try:
            # Use 64W x 20API = 1280 calls as the burst-rate baseline point.
            send_ms, hub_ms, e2e_ms, _, _ = _burst_e2e(actor_burst, 64, 20)
            burst_e2e_qps = 1280 / (e2e_ms / 1000)
        finally:
            _cleanup(actor_burst)

        max_burst_calls = int(burst_e2e_qps * ACCEPTABLE_BURST_LATENCY_MS / 1000)

        print(f"\n[2. Burst Limit (processing time < {ACCEPTABLE_BURST_LATENCY_MS}ms)]")
        print(f"  64W x 20API burst time : send {send_ms:.0f}ms + Hub {hub_ms:.0f}ms = {e2e_ms:.0f}ms")
        print(f"  Estimated burst rate   : {burst_e2e_qps:,.0f} calls/s")
        print(f"  Max burst calls        : about {max_burst_calls:,} calls")
        print(f"  Conversion:")
        for m in [5, 10, 20]:
            workers = max_burst_calls // m
            print(f"    {m:2d} APIs per worker -> up to ~{workers:,} workers")

        # 3. Sustained QPS notes.
        print(f"\n[3. Sustained QPS Notes]")
        print(f"  E2E sustained QPS at real cadence:")
        print(f"    The Hub processes serially, so practical sustained QPS is roughly the Hub processing rate ({hub_qps:,.0f} calls/s)")
        print(f"    Lag will not grow as long as workers x APIs/worker / step_time < Hub QPS")

        # 4. Recommended configuration matrix.
        print(f"\n[4. Recommended Configuration Matrix] (✅ recommended  ⚠️ borderline  ❌ over limit)")
        print(f"  Conditions: 2s step time, burst processing < 1s, sustained QPS margin > 20%")
        print()

        api_options = [5, 10, 20, 50, 100]
        col_w = 10
        header = f"  {'GPUs':<10}" + "".join(f"{f'{m}API':<{col_w}}" for m in api_options)
        print(header)
        print("  " + "─" * (10 + col_w * len(api_options)))

        step_s = 2.0
        for n_gpus in [8, 32, 64, 128, 256, 512, 1024]:
            row = f"  {f'{n_gpus}GPU':<10}"
            for m_calls in api_options:
                total_burst = n_gpus * m_calls
                required_sustained = total_burst / step_s
                burst_ok = total_burst < max_burst_calls
                sustained_ok = required_sustained < hub_qps * 0.8

                if burst_ok and sustained_ok:
                    status = "✅"
                elif burst_ok or sustained_ok:
                    status = "⚠️"
                else:
                    status = "❌"
                row += f"{status:<{col_w}}"
            print(row)

        # 5. Usage recommendations.
        print(f"\n[5. Usage Recommendations]")
        print(f"  1. Prioritize core metrics: reward / kl / loss (<= 5 APIs/Worker)")
        print(f"  2. Keep custom metrics within 10 per worker to avoid Hub burst backlog")
        print(f"  3. For more than 256 GPUs, contact the team to evaluate Hub scaling options")
        print(f"\n  ⚠️  Notes:")
        print(f"  - This report is based on a single-node Hub Actor with serial Ray processing")
        print(f"  - Unit-test QPS without Ray is about 3~10x this test and is the theoretical upper bound")
        print(f"  - For larger scales, consider multiple Hub Actor shards, fewer API calls, or a longer step interval")

        # Hard assertions.
        assert hub_qps > 100, (
            f"E2E QPS={hub_qps:.0f}, below the minimum expected 100 calls/s; Ray environment may be unhealthy"
        )
        assert max_burst_calls > 100, (
            f"E2E burst limit {max_burst_calls} calls is below expectation"
        )
