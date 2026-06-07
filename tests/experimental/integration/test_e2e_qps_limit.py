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
Integration Tests: Monitor 端到端 API QPS 上限（含真实 Ray RPC 开销）
====================================================================

测试目标
--------
回答用户核心问题（含真实 Ray RPC 开销的准确答案）：
    "同时有多少个 Worker，每个 Worker 调多少个 API，Monitor 端到端能扛住？"

与单元测试的关系
----------------
单元测试：Hub 直接调用（无 Ray），测理论上限（乐观值）
本文件：  真实 Ray Actor + fire-and-forget，测实际端到端上限（准确值）

端到端链路
----------
Worker 线程（模拟 Ray Worker）
    ↓ actor.apply_event.remote()  ← Ray RPC（序列化 + 网络/IPC）
    ↓
Hub Actor（Ray Actor，串行处理）
    ↓ hub._events_applied 计数增加
    ✅ 处理完成

关键优化
--------
1. 行级早退（Early Exit）：
       矩阵扫描中，某行第一个 ❌ 出现后，后续更大 API 数直接标记为 ❌(跳过)。
       避免对明显超限的组合做无意义的长时等待。

2. 持续测试时长收短：
       从 30s 降至 10s，每个 Worker 规模仍能采集 5 次 lag 快照。
       同时加早退：lag ❌ 后不再测试更大 Worker 数。

3. 测试规模扩大但有保护：
       WORKER_COUNTS 扩展到 1024，API_CALLS_PER_WORKER 扩展到 200。
       行级早退确保不会对超大规模做无限等待。

运行方式
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

ray = pytest.importorskip("ray", reason="需要安装 ray")

from experimental.utils import MonitorEventKind


# ===========================================================================
# 测试参数
# ===========================================================================

# 扩大规模，行级早退保证不超时
WORKER_COUNTS = [8, 32, 64, 128, 256, 512, 1024]
API_CALLS_PER_WORKER = [5, 10, 20, 50, 100, 200]

# 可接受的瞬时处理延迟（ms）
# 同单元测试：Hub Burst 处理时间 < 1s = 典型 step 耗时的 50%
ACCEPTABLE_BURST_LATENCY_MS = 1000

# 持续测试时长（秒）—— 从 30s 降至 10s，每规模仍采集 5 次 lag 快照
SUSTAINED_DURATION_S = 10

# lag 增长可接受阈值（10s 内 lag 增长 < 此值视为稳定）
ACCEPTABLE_LAG_GROWTH = 500

# 跳过标志（矩阵中被早退跳过的格子）
SKIP_SENTINEL = -1.0


# ===========================================================================
# Fixture：Ray 集群
# ===========================================================================

@pytest.fixture(scope="module")
def ray_local():
    """启动本地 Ray，测试结束后关闭。"""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            logging_level="WARNING",
        )
    yield
    ray.shutdown()


# ===========================================================================
# 工具函数
# ===========================================================================

def _create_ray_hub(name: str, port: int):
    """创建真实 Ray Hub Actor，patch 掉 HTTP server 和 Prometheus reload。"""
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
    构造一条模拟 API 调用的 event。

    用 api_idx % 3 决定类型，保证同一 metric 名始终对应同一类型，
    避免 prometheus_client 因同名 metric 注册两种类型而抛 ValueError。
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
    等待 Hub 处理完 expected 条事件，返回 (实际处理数, 等待耗时s)。
    超时则返回当前处理数（不抛异常）。
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
    N 个 Worker 线程同时发送 M 个 API（fire-and-forget），等待 Hub 处理完。

    返回 (send_ms, hub_wait_ms, e2e_ms, processed, total)。
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
# 测试一：端到端瞬时峰值 QPS
# ===========================================================================

@pytest.mark.integration
class TestE2EBurstQPS:
    """
    端到端瞬时峰值测试：N 个 Worker 线程同时发送 M 个 API 调用（fire-and-forget）。
    测量从全部发出到 Hub 全部处理完的真实端到端耗时。
    """

    def test_burst_by_worker_count(self, ray_local, capsys):
        """
        场景
        ----
        固定每 Worker 调用 10 个 API（verl 典型场景），梯度增加 Worker 数量。
        第一个 ❌ 出现后停止（更大 Worker 数结果可推算）。

        对应用户问题：
            "我每个 Worker 上报 10 个指标，最多能支持多少 GPU？"

        输出报告
        --------
        Worker数  API数  总API调用  发送耗时(ms)  Hub处理(ms)  端到端(ms)  评估
        8         10     80         12.3          45.2          57.5        ✅ 可用
        32        10     320        18.4          89.3          107.7       ✅ 可用
        128       10     1280       45.2          312.4         357.6       ✅ 可用
        256       10     2560       89.4          634.2         723.6       ⚠️ 临界
        512       10     5120       178.3         1284.5        1462.8      ❌ 超限（停止）
        """
        M = 10
        _print_header(f"端到端 Burst —— 固定 {M} API/Worker，梯度增加 Worker 数")

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
                    "✅ 可用" if e2e_ms < 500 else
                    "⚠️ 临界" if ok else "❌ 超限",
                ])

                assert processed == total, (
                    f"{n_workers}W: 发送 {total}，Hub 处理 {processed}，"
                    f"丢失 {total - processed} 条！"
                )

            finally:
                _cleanup(actor)

            if not ok:
                print(f"  → {n_workers}W 已超限，停止（更大规模结果可推算）")
                break

        _print_table(
            ["Worker数", "API数", "总调用数", "发送(ms)", "Hub处理(ms)", "端到端(ms)", "评估"],
            rows, col_width=13
        )
        print(f"\n  端到端 Burst 上限（10 API/W）：最多支持 {max_ok_workers} 个 Worker")

    def test_burst_matrix_e2e(self, ray_local, capsys):
        """
        场景
        ----
        遍历 WORKER_COUNTS × API_CALLS_PER_WORKER 组合，生成端到端矩阵。
        行级早退：同一行第一个 ❌ 后，后续更大 API 数标记 ❌(跳过)。

        这是给用户的最终查询表（含 Ray RPC 真实开销）。

        输出矩阵（示例）：
                    5API      10API     20API     50API     100API    200API
        8W         ✅28ms    ✅54ms    ✅108ms   ✅271ms   ✅542ms   ⚠️921ms
        32W        ✅112ms   ✅218ms   ✅436ms   ⚠️891ms   ❌1784ms  ❌skip
        128W       ✅448ms   ⚠️872ms   ❌1744ms  ❌skip    ❌skip    ❌skip
        256W       ⚠️896ms   ❌1744ms  ❌skip    ❌skip    ❌skip    ❌skip
        512W       ❌1792ms  ❌skip    ❌skip    ❌skip    ❌skip    ❌skip
        """
        _print_header("端到端 Burst 矩阵 —— Worker数 × API调用数（用户最终查询表）")

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
                          f"{n_workers:4d}W × {m_calls:3d}API  →  ❌(跳过，行内已超限)")
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
                          f"→  发送 {send_ms:.0f}ms + Hub {hub_ms:.0f}ms "
                          f"= {e2e_ms:.0f}ms {status}")

                    assert processed == total, (
                        f"[{n_workers}W×{m_calls}API] 发送 {total}，"
                        f"处理 {processed}，丢失 {total-processed} 条！"
                    )

                    if e2e_ms >= ACCEPTABLE_BURST_LATENCY_MS:
                        row_exceeded = True

                finally:
                    _cleanup(actor)

        # ── 打印矩阵 ──
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

        print(f"\n  图例：✅ <500ms（推荐）  ⚠️ 500~1000ms（临界）  ❌ >1000ms（超限）")
        print(f"  统计：✅ {counts['✅']} 个  ⚠️ {counts['⚠️']} 个  ❌ {counts['❌']} 个")

        # ── 边界线汇总 ──
        print(f"\n  【端到端 Burst 边界线】处理时间 < {ACCEPTABLE_BURST_LATENCY_MS}ms 的最大规模：")
        for n_workers in WORKER_COUNTS:
            max_ok_api = 0
            for m_calls in API_CALLS_PER_WORKER:
                e2e_ms, _, _ = matrix[n_workers][m_calls]
                if e2e_ms != SKIP_SENTINEL and e2e_ms < ACCEPTABLE_BURST_LATENCY_MS:
                    max_ok_api = m_calls
            total_ok = n_workers * max_ok_api if max_ok_api > 0 else 0
            print(f"    {n_workers:4d} Worker → 最多 {max_ok_api:3d} API/Worker"
                  f" = {total_ok:,} calls/burst")

        assert counts["❌"] > 0, "所有组合均通过，测试范围太保守，未找到上限"
        print(f"\n  ✅ 测试有效：{counts['❌']} 个组合超限，成功找到端到端 Burst 上限边界")


# ===========================================================================
# 测试二：端到端持续 QPS 上限
# ===========================================================================

@pytest.mark.integration
class TestE2ESustainedQPS:
    """
    端到端持续 QPS 测试：多 Worker 线程按 step 节奏持续发送，测 Hub lag 趋势。

    与单元测试持续 QPS 的区别：
        单元测试：全速发送（去掉 sleep），找 Hub 最大吞吐
        集成测试：按 step 间隔发送（保留 sleep），模拟真实 verl 训练节奏，
                  测在真实场景下 lag 是否可控
    """

    def test_sustained_lag_by_worker_count(self, ray_local, capsys):
        """
        场景
        ----
        固定每 Worker 调用 10 个 API，每 step 间隔 2s，
        梯度增加 Worker 数量，持续 SUSTAINED_DURATION_S 秒，
        观察 Hub lag 变化趋势。

        第一个 lag ❌（快速增长）后停止测试。

        输出报告
        --------
        Worker数  需求QPS  Hub处理QPS  lag增长  评估
        32        160      950         +0       ✅ 稳定
        128       640      950         +12      ✅ 稳定
        256       1280     950         +148     ⚠️ 缓慢增长
        512       2560     950         +1240    ❌ 快速增长（停止）
        """
        _print_header(
            f"端到端持续 QPS —— 固定 10 API/W，step 2s，持续 {SUSTAINED_DURATION_S}s"
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
                    """每隔 step_s 秒上报一批 M 个 API（模拟真实 verl 训练节奏）。"""
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

                # 每 2s 采样一次 lag
                lag_samples = []
                for _ in range(SUSTAINED_DURATION_S // 2):
                    time.sleep(2.0)
                    status = ray.get(actor.get_status.remote())
                    lag = sent_count[0] - status["events_applied"]
                    lag_samples.append(lag)

                stop_flag[0] = True
                for t in threads:
                    t.join(timeout=5)

                # 分析 lag 趋势
                if len(lag_samples) >= 2:
                    lag_growth = lag_samples[-1] - lag_samples[0]
                else:
                    lag_growth = 0

                if lag_growth < 100:
                    trend = f"稳定(+{lag_growth:.0f})"
                    sustainable = True
                    max_ok_workers = n_workers
                elif lag_growth < ACCEPTABLE_LAG_GROWTH:
                    trend = f"缓慢增长(+{lag_growth:.0f})"
                    sustainable = False
                else:
                    trend = f"快速增长(+{lag_growth:.0f})"
                    sustainable = False

                final = ray.get(actor.get_status.remote())
                hub_qps = final["events_applied"] / SUSTAINED_DURATION_S

                rows.append([
                    n_workers,
                    f"{required_qps:.0f}",
                    f"{hub_qps:.0f}",
                    trend,
                    "✅ 稳定" if lag_growth < 100 else
                    "⚠️ 临界" if lag_growth < ACCEPTABLE_LAG_GROWTH else "❌ 不可持续",
                ])

            finally:
                _cleanup(actor)

            if not sustainable:
                print(f"  → {n_workers}W lag 增长超限，停止（更大规模可推算）")
                break

        _print_table(
            ["Worker数", "需求QPS", "Hub处理QPS", "lag增长趋势", "评估"],
            rows, col_width=18
        )
        print(f"\n  端到端持续 QPS 上限：{max_ok_workers} Worker × {M} API，step {STEP_S}s")
        print(f"  判断标准：持续 {SUSTAINED_DURATION_S}s 内 lag 增长 < 100 条")


# ===========================================================================
# 测试三：端到端最终推荐报告
# ===========================================================================

@pytest.mark.integration
class TestE2EFinalReport:
    """
    综合 Burst 和持续 QPS，生成端到端完整推荐报告，含 Ray RPC 折减系数。
    """

    def test_final_recommendation_report(self, ray_local, capsys):
        """
        综合瞬时峰值和持续 QPS，输出最终推荐报告。

        报告内容：
        1. 端到端性能基准（send QPS + Hub 处理 QPS + RPC 折减系数）
        2. Burst 上限（处理时间 < 1s 的最大总调用数）
        3. 持续 QPS 上限
        4. 推荐配置矩阵（GPU数 × API数 → 是否可用）
        5. 使用建议
        """
        _print_header("端到端完整推荐报告 —— Monitor 支持的 verl 训练规模（含 Ray RPC 开销）")

        # ── 1. 端到端性能基准 ──
        actor_base = _create_ray_hub("hub_report_base", port=21500)

        try:
            # 单线程顺序发送 1000 条，测 fire-and-forget 发送速率
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

        print(f"\n【1. 端到端性能基准】")
        print(f"  fire-and-forget 发送速率 : {send_qps:,.0f} calls/s  （单线程顺序发）")
        print(f"  端到端吞吐（发+处理）    : {hub_qps:,.0f} calls/s")
        print(f"  发送耗时                 : {send_elapsed*1000:.1f}ms（{N}条）")
        print(f"  Hub 处理等待             : {wait_s*1000:.1f}ms")

        # ── 2. Burst 上限 ──
        actor_burst = _create_ray_hub("hub_report_burst", port=21501)

        try:
            # 用 64W × 20API = 1280 calls 测 Burst 速率作为基准点
            send_ms, hub_ms, e2e_ms, _, _ = _burst_e2e(actor_burst, 64, 20)
            burst_e2e_qps = 1280 / (e2e_ms / 1000)
        finally:
            _cleanup(actor_burst)

        max_burst_calls = int(burst_e2e_qps * ACCEPTABLE_BURST_LATENCY_MS / 1000)

        print(f"\n【2. Burst 上限（处理时间 < {ACCEPTABLE_BURST_LATENCY_MS}ms）】")
        print(f"  64W×20API Burst 耗时   : 发送 {send_ms:.0f}ms + Hub {hub_ms:.0f}ms = {e2e_ms:.0f}ms")
        print(f"  推算 Burst 处理速率    : {burst_e2e_qps:,.0f} calls/s")
        print(f"  最大可 Burst 调用数    : 约 {max_burst_calls:,} calls")
        print(f"  换算：")
        for m in [5, 10, 20]:
            workers = max_burst_calls // m
            print(f"    每 Worker {m:2d} 个 API → 最多支持 ~{workers:,} 个 Worker")

        # ── 3. 持续 QPS 说明 ──
        print(f"\n【3. 持续 QPS 说明】")
        print(f"  端到端持续 QPS（真实节奏）：")
        print(f"    Hub 串行处理，实际可持续 QPS 约等于 Hub 处理速率（{hub_qps:,.0f} calls/s）")
        print(f"    只要 Worker数 × API数/Worker ÷ step耗时 < Hub QPS，lag 就不会增长")

        # ── 4. 推荐配置矩阵 ──
        print(f"\n【4. 推荐配置矩阵】（✅ 推荐  ⚠️ 临界  ❌ 超限）")
        print(f"  条件：step 耗时 2s，Burst 处理 < 1s，持续 QPS 余量 > 20%")
        print()

        api_options = [5, 10, 20, 50, 100]
        col_w = 10
        header = f"  {'GPU数':<10}" + "".join(f"{f'{m}API':<{col_w}}" for m in api_options)
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

        # ── 5. 使用建议 ──
        print(f"\n【5. 使用建议】")
        print(f"  1. 优先监控核心指标：reward / kl / loss（≤ 5 个 API/Worker）")
        print(f"  2. 自定义指标控制在 10 个以内/Worker，避免 Hub Burst 积压")
        print(f"  3. 超过 256 GPU 建议联系团队评估 Hub 扩容方案")
        print(f"\n  ⚠️  说明：")
        print(f"  - 本报告基于单节点 Hub Actor（Ray 串行处理）")
        print(f"  - 单元测试 QPS（无 Ray）约为本测试的 3~10x，是理论上限")
        print(f"  - 如需更大规模，考虑：多 Hub Actor 分片 / 减少 API 调用数 / 增大 step 间隔")

        # 硬性断言
        assert hub_qps > 100, (
            f"端到端 QPS={hub_qps:.0f}，低于最低预期 100 calls/s，Ray 环境有问题"
        )
        assert max_burst_calls > 100, (
            f"端到端 Burst 上限 {max_burst_calls} calls，低于预期"
        )
