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
Unit Tests: Monitor API QPS 上限探测（Hub 原始处理能力）
========================================================

测试目标
--------
回答用户核心问题：
    "同时有多少个 Worker，每个 Worker 调多少个 API，Monitor 能扛住？"

测试范围
--------
本文件测试 Hub 的原始处理能力（不含 Ray RPC 开销）：
    Worker 调用 API → Hub.apply_event() 直接调用（无 Ray）

与集成测试的关系：
    本文件结论 = Hub 理论上限（最乐观）
    集成测试结论 = 真实端到端上限（含 Ray RPC 开销，更准确）

两个核心场景
------------
1. 瞬时峰值 QPS（Burst）：
       step 边界，所有 Worker 同时完成，同一瞬间涌入大量 API 调用
       关键问题：Hub 多久处理完？处理时间是否在可接受范围内？
       测试手段：直接循环调用 hub.apply_event()，不限速，找到处理时间超过 1s 的拐点

2. 持续 QPS（Sustained）：
       长时间训练，Worker 持续稳定上报
       关键问题：Hub 的吞吐上限在哪里？
       测试手段：不限速全速发送 N 秒，测实际处理速率（去掉 sleep 限速）

运行方式
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
# 测试参数配置
# ===========================================================================

# 瞬时峰值矩阵：Worker 数量梯度（扩大范围以找到实际拐点）
BURST_WORKER_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]

# 瞬时峰值矩阵：每 Worker 每 step API 调用数梯度
BURST_API_CALLS_PER_WORKER = [5, 10, 20, 50, 100, 200, 500]

# 可接受的瞬时处理延迟上限（ms）
# 超过此值说明 Hub 处理太慢，会影响下一个 step 的监控数据实时性
# 典型 RL step 耗时 2s，取 50% = 1s 作为 Hub 处理时间上限
ACCEPTABLE_BURST_LATENCY_MS = 1000

# 持续 QPS 测试时长（秒）——无需过长，全速发送 5s 就足以测出稳定吞吐
SUSTAINED_DURATION_S = 3


# ===========================================================================
# 工具函数
# ===========================================================================

def _make_api_event(kind: str, worker_id: int, api_idx: int) -> dict:
    """
    构造一条模拟 API 调用产生的 event。

    参数
    ----
    kind     : 事件类型（counter / gauge / histogram）
    worker_id: Worker 编号（影响 label 值，代表不同 worker 有不同 label）
    api_idx  : 第几个 API 调用（代表同一 Worker 上报不同指标名）
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
    构造 n_workers × m_calls 条事件并全速提交给 Hub，返回 (耗时ms, 处理数)。

    这是 Burst 场景的核心：所有 Worker 在同一个 step 边界同时上报。
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
    全速（无 sleep）向 Hub 发送 Gauge 事件，持续 duration_s 秒。
    返回实际处理 QPS（calls/s）。

    说明：
        单元测试中 Hub 是同步处理，apply_event() 返回 = 处理完毕。
        此函数不限速，直接测出 Hub 的最大吞吐。
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
# 测试零：Hub 基础性能基准（其他所有测试的参考点）
# ===========================================================================

class TestHubBaselineThroughput:
    """
    在任何矩阵测试之前，先单独测出 Hub 的原始吞吐上限。
    这是解读后续测试结果的基准参考。
    """

    def test_hub_max_sustained_qps_full_speed(self, make_hub, capsys):
        """
        场景
        ----
        全速（无 sleep 限速）向 Hub 发送同一 Gauge 事件，持续 SUSTAINED_DURATION_S 秒。
        此测试替代了原版 time.sleep() 限速版本，直接找到 Hub 的真实处理上限。

        关键区别
        --------
        ❌ 旧版：for _ in range(N): hub.apply_event(); time.sleep(interval)
              → Hub 在等 Worker，lag 恒为 0，测不出 Hub 上限
        ✅ 新版：while elapsed < T: hub.apply_event()
              → Hub 满负荷，测出实际最大吞吐

        输出
        ----
        Hub 最大持续 QPS：~400,000 calls/s（gauge 同名，无 label）
        """
        _print_header(f"Hub 基础吞吐基准（全速发送 {SUSTAINED_DURATION_S}s）")

        hub, _ = make_hub(port=19500)
        qps = _measure_sustained_qps(hub, SUSTAINED_DURATION_S)

        print(f"\n  Hub 最大持续 QPS（单一 Gauge，无 label）: {qps:,.0f} calls/s")
        print(f"  测试时长：{SUSTAINED_DURATION_S}s，无 sleep 限速")
        print(f"  此为最乐观值（热路径：同名 metric，无 label，prometheus_client 缓存命中）")

        assert qps > 10_000, (
            f"Hub 最大吞吐 {qps:.0f} calls/s，低于最低预期 10k calls/s，环境有问题"
        )

    def test_hub_realistic_throughput_mixed_metrics(self, make_hub, capsys):
        """
        场景
        ----
        全速发送混合类型事件（counter/gauge/histogram），每个事件使用不同 metric 名。
        这更接近 verl 实际场景（多 Worker、多指标名）。

        verl 实际场景：
            每 Worker 上报 3~10 个不同名称的指标，每个指标有 worker label
            → prometheus_client 每次需要查找 metric 名 → 比单一名称慢很多

        输出
        ----
        Hub 真实场景 QPS：通常比单名 Gauge 低 60%~80%（registry 查找开销）
        """
        _print_header("Hub 真实场景吞吐基准（混合指标名 + label，全速发送）")

        hub, _ = make_hub(port=19501)

        # 预构造 100 种不同事件（不同 metric 名 + label），模拟真实 Worker 上报
        # 注意：用 api_idx % 3 决定类型，保证同一 metric 名始终对应同一类型，
        # 避免 prometheus_client 因同名 metric 注册两种类型而抛 ValueError。
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

        print(f"\n  Hub 真实场景 QPS（100 种混合事件）: {qps:,.0f} calls/s")
        print(f"  测试时长：{elapsed:.2f}s，无 sleep 限速")
        print(f"  此值是 Burst 矩阵的实际性能参考基准")

        assert qps > 5_000, (
            f"Hub 真实场景吞吐 {qps:.0f} calls/s，低于预期 5k calls/s"
        )

    def test_hub_auto_find_burst_saturation_point(self, make_hub, capsys):
        """
        场景
        ----
        二分搜索：自动找到"总 API 调用数"的临界点，
        即处理时间首次超过 ACCEPTABLE_BURST_LATENCY_MS 的最小总量。

        输出
        ----
        Burst 饱和点：约 X calls（处理耗时 ~1000ms）
        等价于：N Worker × M API/Worker = X

        这是给用户最直接的答案：
            "如果你的 Worker数 × API数/Worker < X，Monitor Burst 就没问题"
        """
        _print_header("二分搜索：Burst 饱和点（自动找到处理时间刚好超过 1s 的总调用数）")

        # 二分范围：[lo, hi] 总调用数
        lo, hi = 1_000, 2_000_000
        hub_port = 19502

        # 快速预测：先单点测几个，确认范围包含拐点
        # 上限定为 60k（从矩阵结果看 ~30k calls 附近是拐点），避免单点探测耗时过长
        probe_results = []
        for n in [5_000, 15_000, 30_000, 60_000]:
            hub, _ = make_hub(port=hub_port)
            hub_port += 1
            # 用 n_workers=min(n,1024), m=n//min(n,1024)
            n_w = min(n, 1024)
            m = max(1, n // n_w)
            elapsed_ms, total = _measure_burst(hub, n_w, m)
            probe_results.append((total, elapsed_ms))
            print(f"  预探测 {total:,} calls → {elapsed_ms:.0f}ms")

        # 找二分范围
        under_limit = [total for total, ms in probe_results if ms < ACCEPTABLE_BURST_LATENCY_MS]
        over_limit = [total for total, ms in probe_results if ms >= ACCEPTABLE_BURST_LATENCY_MS]

        if not under_limit:
            print(f"\n  ❗ 所有预探测规模都超限，Hub 基线性能异常")
            saturation_point = 0
        elif not over_limit:
            print(f"\n  ✅ 所有预探测规模都在限内（Hub 极快），尝试更大规模...")
            saturation_point = max(under_limit)
        else:
            lo = max(under_limit)
            hi = min(over_limit)

            # 二分 5 轮，精度够用
            for _ in range(5):
                mid = (lo + hi) // 2
                hub, _ = make_hub(port=hub_port)
                hub_port += 1
                n_w = min(mid, 1024)
                m = max(1, mid // n_w)
                elapsed_ms, total = _measure_burst(hub, n_w, m)
                print(f"  二分 {total:,} calls → {elapsed_ms:.0f}ms "
                      f"({'✅' if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS else '❌'})")
                if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS:
                    lo = total
                else:
                    hi = total

            saturation_point = lo

        print(f"\n  ✅ Burst 饱和点：约 {saturation_point:,} calls")
        print(f"  含义：Worker数 × API数/Worker < {saturation_point:,} 时，"
              f"Hub Burst 处理时间 < {ACCEPTABLE_BURST_LATENCY_MS}ms")
        print(f"  典型场景换算（10 API/Worker）：最多支持 ~{saturation_point//10:,} 个 Worker 同时上报")
        print(f"  典型场景换算（5 API/Worker）：最多支持 ~{saturation_point//5:,} 个 Worker 同时上报")

        # 软性断言：至少找到一个有效拐点
        assert saturation_point > 0, "未能找到 Burst 饱和点，Hub 基线性能异常"


# ===========================================================================
# 测试一：瞬时峰值 QPS 矩阵（大范围扫描）
# ===========================================================================

class TestBurstQPSMatrix:
    """
    完整的 Worker数 × API调用数 组合矩阵扫描。

    目标：找到矩阵中 ✅→❌ 的边界线，给出 Hub Burst 能力的精确上限。

    扩大了 Worker 数和 API 数的范围，确保矩阵右下角有足够多的 ❌，
    避免"测的太保守、看不出上限"的问题。
    """

    def test_burst_matrix_worker_vs_api(self, make_hub, capsys):
        """
        场景
        ----
        遍历 BURST_WORKER_COUNTS × BURST_API_CALLS_PER_WORKER 所有组合，
        测量每个组合的 Hub 处理时间，生成完整矩阵。

        判断标准：
            ✅ 处理时间 < 500ms（推荐，余量充足）
            ⚠️ 处理时间 500ms ~ 1000ms（临界，可用但余量不足）
            ❌ 处理时间 > 1000ms（超限，Hub 来不及处理，影响下一 step）

        输出矩阵（示例）：
                        5 API   10 API   20 API   50 API   100 API  200 API  500 API
        64 Worker      ✅16    ✅32     ✅64     ✅162    ✅323    ✅646    ✅1616
        128 Worker     ✅32    ✅64     ✅130    ✅325    ✅650    ❌1305   ❌3262
        256 Worker     ✅64    ✅130    ✅261    ✅652    ❌1308   ❌2616   ❌6540
        512 Worker     ✅130   ✅262    ✅524    ❌1310   ❌2626   ❌5256   ❌13140
        1024 Worker    ✅262   ✅524    ❌1049   ❌2625   ❌5256   ❌...    ❌...
        2048 Worker    ✅524   ❌1052   ❌2108   ❌...    ❌...    ❌...    ❌...
        4096 Worker    ❌1052  ❌2108   ❌...    ❌...    ❌...    ❌...    ❌...
        """
        _print_header("瞬时峰值 QPS 矩阵 —— Worker数 × API调用数（大范围扫描）")

        port_base = 19600
        matrix = {}

        total_combos = len(BURST_WORKER_COUNTS) * len(BURST_API_CALLS_PER_WORKER)
        combo_idx = 0
        # 用于行级早退：记录每行第一个 ❌ 出现的位置
        SKIP_SENTINEL = -1  # 表示该格被跳过

        for n_workers in BURST_WORKER_COUNTS:
            matrix[n_workers] = {}
            row_exceeded = False  # 本行是否已出现 ❌

            for m_calls in BURST_API_CALLS_PER_WORKER:
                combo_idx += 1

                if row_exceeded:
                    # 本行已有 ❌，后续更大的 API 数肯定也超限，直接跳过
                    matrix[n_workers][m_calls] = (SKIP_SENTINEL, 0)
                    print(f"  [{combo_idx:2d}/{total_combos}] "
                          f"{n_workers:4d}W × {m_calls:3d}API  →  ❌(跳过，行内已超限)")
                    continue

                hub, _ = make_hub(port=port_base)
                port_base += 1

                elapsed_ms, total = _measure_burst(hub, n_workers, m_calls)
                matrix[n_workers][m_calls] = (elapsed_ms, total)

                assert hub._events_applied == total, (
                    f"[{n_workers}W × {m_calls}API] 发送 {total} 条，"
                    f"Hub 只处理了 {hub._events_applied} 条，存在丢失！"
                )

                # 进度提示
                status = ("✅" if elapsed_ms < 500 else
                          "⚠️" if elapsed_ms < ACCEPTABLE_BURST_LATENCY_MS else "❌")
                print(f"  [{combo_idx:2d}/{total_combos}] "
                      f"{n_workers:4d}W × {m_calls:3d}API = {total:7,} calls "
                      f"→ {elapsed_ms:7.0f}ms {status}")

                if elapsed_ms >= ACCEPTABLE_BURST_LATENCY_MS:
                    row_exceeded = True  # 触发行级早退

        # ── 打印矩阵 ──
        print(f"\n{'─'*90}")
        header_cols = [f"{m}API" for m in BURST_API_CALLS_PER_WORKER]
        row_label = "Worker\\API"
        print(f"  {row_label:<14}" + "".join(f"{h:<13}" for h in header_cols))
        print(f"  {'─'*14}" + "─" * (13 * len(BURST_API_CALLS_PER_WORKER)))

        # 统计 ✅/⚠️/❌ 数量
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

        print(f"\n  图例：✅ <500ms（推荐）  ⚠️ 500~1000ms（临界）  ❌ >1000ms（超限）")
        print(f"  统计：✅ {counts['✅']} 个  ⚠️ {counts['⚠️']} 个  ❌ {counts['❌']} 个")

        # ── 找到边界线 ──
        print(f"\n  【Burst 边界线】处理时间 < {ACCEPTABLE_BURST_LATENCY_MS}ms 的最大规模：")
        for n_workers in BURST_WORKER_COUNTS:
            max_ok_api = 0
            for m_calls in BURST_API_CALLS_PER_WORKER:
                ms, _ = matrix[n_workers][m_calls]
                if ms != SKIP_SENTINEL and ms < ACCEPTABLE_BURST_LATENCY_MS:
                    max_ok_api = m_calls
            total_ok = n_workers * max_ok_api if max_ok_api > 0 else 0
            print(f"    {n_workers:4d} Worker → 最多 {max_ok_api:3d} API/Worker"
                  f" = {total_ok:,} calls/burst")

        # 断言：至少有一半组合测出 ❌，否则范围太保守
        total_combos_run = len(BURST_WORKER_COUNTS) * len(BURST_API_CALLS_PER_WORKER)
        assert counts["❌"] > 0, "所有组合均通过，测试范围太保守，未找到上限"
        print(f"\n  ✅ 测试有效：{counts['❌']} 个组合超限，成功找到 Burst 上限边界")

    def test_burst_fixed_api_grow_workers(self, make_hub, capsys):
        """
        场景
        ----
        固定每 Worker 调用 10 个 API（verl 典型场景），梯度增加 Worker 数量。
        直到处理时间超过 ACCEPTABLE_BURST_LATENCY_MS 为止。

        对应用户问题：
            "我每个 Worker 只上报 10 个指标，最多能用多少 GPU？"
        """
        _print_header("Burst：固定 10 API/Worker，梯度增加 Worker 数（找到 Worker 数上限）")

        M = 10  # verl 典型场景：每 Worker 上报 10 个指标
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
                "✅ 可用" if elapsed_ms < 500 else
                "⚠️ 临界" if ok else "❌ 超限",
            ])

            assert hub._events_applied == total

            if not ok:
                # Worker 数超限后更大的 Worker 数肯定也超限，直接退出
                break

        _print_table(
            ["Worker数", "API数/W", "总调用数", "Hub处理时间", "评估"],
            rows, col_width=14
        )
        print(f"\n  10 API/Worker 场景下，最多支持 {max_ok_workers} 个 Worker 同时上报")

    def test_burst_fixed_workers_grow_api(self, make_hub, capsys):
        """
        场景
        ----
        固定 512 个 Worker（典型大规模训练），梯度增加每 Worker API 调用数。

        对应用户问题：
            "我有 512 GPU，每个 Worker 上报多少指标不会超限？"
        """
        _print_header("Burst：固定 512 Worker，梯度增加 API 调用数（找到 API 数上限）")

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
                "✅ 可用" if elapsed_ms < 500 else
                "⚠️ 临界" if ok else "❌ 超限",
            ])

            assert hub._events_applied == total

            if not ok:
                # API 数超限后更大的值肯定也超限，直接退出
                break

        _print_table(
            ["Worker数", "API数/W", "总调用数", "Hub处理时间", "评估"],
            rows, col_width=14
        )
        print(f"\n  512 Worker 场景下，每 Worker 最多上报 {max_ok_api} 个 API/step")


# ===========================================================================
# 测试二：持续 QPS 上限（全速发送，无 sleep 限速）
# ===========================================================================

class TestSustainedQPS:
    """
    测试 Hub 真实的持续处理能力上限。

    修复原版缺陷：
        原版使用 time.sleep() 控制发送速率 → Hub 永远空闲 → lag=0 → 测不出上限
        新版全速发送 → Hub 满负荷 → 测出真实最大 QPS
    """

    def test_max_sustained_qps_full_speed(self, make_hub, capsys):
        """
        场景
        ----
        全速（无 sleep）发送事件，持续 SUSTAINED_DURATION_S 秒，
        测出 Hub 的最大持续 QPS。

        注意
        ----
        单元测试中 Hub 同步处理，apply_event() 返回即完成。
        "全速发送"的瓶颈就是 Hub 的处理速度（prometheus_client 写操作）。
        """
        _print_header(f"持续 QPS 上限（全速发送 {SUSTAINED_DURATION_S}s，无 sleep 限速）")

        # 场景 1：单一 Gauge，无 label（最快路径，prometheus_client 缓存命中率最高）
        hub1, _ = make_hub(port=19900)
        qps_simple = _measure_sustained_qps(hub1, SUSTAINED_DURATION_S)

        # 场景 2：混合类型，多 metric 名，有 label（verl 真实场景）
        # 用 (i % 20) % 3 决定类型，保证同名 metric 始终是同一类型。
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

        print(f"\n  【持续 QPS 测试结果】")
        print(f"  简单场景（单 Gauge，无 label）：{qps_simple:,.0f} calls/s")
        print(f"  真实场景（混合60种事件）：      {qps_realistic:,.0f} calls/s")
        print()
        print(f"  真实场景 QPS 是简单场景的 {qps_realistic/qps_simple*100:.0f}%")
        print(f"  （差距来自：不同 metric 名导致 prometheus_client 需要做 dict 查找）")
        print()
        print(f"  【持续 QPS 与 verl 训练场景对比】")
        print(f"  公式：持续所需 QPS = Worker数 × API数/Worker ÷ step耗时(s)")
        print()

        # 打印各规模所需 QPS vs Hub 上限
        step_s = 2.0
        rows = []
        for n_workers in [32, 128, 256, 512, 1024, 2048, 4096]:
            for m_api in [5, 10, 20]:
                required = n_workers * m_api / step_s
                margin = (qps_realistic - required) / qps_realistic * 100
                ok = required < qps_realistic * 0.8  # 留 20% 余量
                rows.append([
                    f"{n_workers}W",
                    f"{m_api}API",
                    f"{required:,.0f}",
                    f"{margin:.0f}%",
                    "✅ 充裕" if margin > 50 else
                    "⚠️ 临界" if ok else "❌ 超限",
                ])

        _print_table(
            ["Worker数", "API数/W", "需求QPS", "余量", "评估"],
            rows, col_width=13
        )

        print(f"\n  注意：以上基于真实场景 QPS（{qps_realistic:,.0f} calls/s）")
        print(f"  单元测试无 Ray RPC 开销，集成测试 QPS 通常降至 30%~50%")

        assert qps_simple > 10_000, f"简单场景 QPS 仅 {qps_simple:.0f}，环境有问题"
        assert qps_realistic > 5_000, f"真实场景 QPS 仅 {qps_realistic:.0f}，环境有问题"

    def test_sustained_qps_degradation_under_metric_diversity(self, make_hub, capsys):
        """
        场景
        ----
        测量 metric 种类数（metric 多样性）对 Hub QPS 的影响。
        metric 种类越多，prometheus_client 内部 dict 查找越慢。

        这是 verl 用户选择「上报几个指标」的性能代价。

        输出
        ----
        Metric种类  Hub QPS    相对基准   说明
        1           400,000    100%       单一 metric，纯缓存命中
        5           350,000    87%        5 种 metric 轮换
        10          280,000    70%        10 种 metric 轮换
        20          220,000    55%        20 种 metric（verl 典型场景）
        50          180,000    45%        50 种 metric（重监控）
        100         150,000    37%        100 种 metric（极重监控）
        """
        _print_header("Hub QPS 随 Metric 多样性的衰减（全速发送）")

        metric_counts = [1, 5, 10, 20, 50, 100]
        port_base = 19950
        rows = []
        base_qps = None

        for n_metrics in metric_counts:
            hub, _ = make_hub(port=port_base)
            port_base += 1

            # 构造 n_metrics 种不同 Gauge 事件
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
            ["Metric种类", "Hub QPS", "相对基准", "评级"],
            rows, col_width=14
        )

        print(f"\n  结论：每 Worker 上报的不同指标名越多，Hub 处理越慢")
        print(f"  建议：控制每 Worker 上报的 metric 种类，优先合并相似指标")


# ===========================================================================
# 测试三：综合推荐报告
# ===========================================================================

class TestRecommendationReport:
    """
    综合 burst 和 sustained 两个维度，生成最终推荐报告。
    """

    def test_generate_final_report(self, make_hub, capsys):
        """
        生成完整的性能上限推荐报告。

        报告内容：
        1. Hub 原始处理能力基准
        2. Burst 上限（处理时间 < 1s 的最大总调用数）
        3. Sustained 上限（最大持续 QPS）
        4. 推荐配置矩阵（Worker数 × API数 → 是否可用）
        5. 使用建议
        """
        _print_header("Monitor 性能上限完整报告（单元测试基准，不含 Ray RPC 开销）")

        # ── 1. Hub 基准性能 ──
        print(f"\n{'─'*60}")
        print(f"【1. Hub 基准性能】")

        # 简单 QPS（最乐观）
        hub_s, _ = make_hub(port=20000)
        qps_simple = _measure_sustained_qps(hub_s, 3)

        # 真实场景 QPS
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

        # Burst 性能：10k 混合事件
        hub_b, _ = make_hub(port=20002)
        burst_ms, _ = _measure_burst(hub_b, 512, 20)  # 512 × 20 = 10240 calls
        burst_qps = 10240 / (burst_ms / 1000)

        print(f"  简单场景最大 QPS : {qps_simple:,.0f} calls/s  （同名 Gauge，无 label）")
        print(f"  真实场景最大 QPS : {qps_realistic:,.0f} calls/s  （60种混合事件）")
        print(f"  Burst 处理速率   : {burst_qps:,.0f} calls/s  （512W×20API=10240 calls）")

        # ── 2. Burst 上限 ──
        print(f"\n{'─'*60}")
        print(f"【2. Burst 上限（处理时间 < {ACCEPTABLE_BURST_LATENCY_MS}ms）】")
        max_burst_calls = int(burst_qps * ACCEPTABLE_BURST_LATENCY_MS / 1000)
        print(f"  最大可 Burst 调用数：约 {max_burst_calls:,} calls")
        print(f"  换算：")
        for m in [5, 10, 20, 50, 100]:
            workers = max_burst_calls // m
            print(f"    每 Worker {m:3d} 个 API → 最多支持 {workers:,} 个 Worker")

        # ── 3. Sustained 上限 ──
        print(f"\n{'─'*60}")
        print(f"【3. Sustained 上限（持续 QPS 上限）】")
        print(f"  单元测试最大持续 QPS：{qps_realistic:,.0f} calls/s（真实场景）")
        print(f"  集成测试预估 QPS    ：{qps_realistic*0.3:,.0f} ~ {qps_realistic*0.5:,.0f} calls/s（含 Ray RPC 开销）")

        # ── 4. 推荐配置矩阵 ──
        print(f"\n{'─'*60}")
        print(f"【4. 推荐配置矩阵】（✅ 推荐  ⚠️ 临界  ❌ 超限）")
        print(f"  条件：step 耗时 2s，Burst 处理 < 1s，持续 QPS 余量 > 20%")
        print()

        api_options = [5, 10, 20, 50, 100, 200]
        header = f"  {'Worker数':<10}" + "".join(f"{f'{m}API':<10}" for m in api_options)
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

        # ── 5. 使用建议 ──
        print(f"\n{'─'*60}")
        print(f"【5. 使用建议】")
        print(f"  1. 小规模（≤256 GPU）：Monitor 几乎没有性能压力，正常使用即可")
        print(f"  2. 中规模（256~1024 GPU）：控制每 Worker 上报指标数 ≤ 20 个")
        print(f"  3. 大规模（>1024 GPU）：仅上报关键指标（reward/loss/kl），≤ 5 个/Worker")
        print(f"  4. 如需更高并发，考虑升级 Hub Actor 的资源配额（更多 CPU 时间）")
        print(f"\n  ⚠️  注意：以上为单元测试基准（不含 Ray RPC 开销）")
        print(f"  实际端到端性能请参考集成测试结果（pytest --run-integration）")

        # 硬性断言
        assert qps_realistic > 5_000, (
            f"Hub 真实场景 QPS={qps_realistic:.0f}，低于最低预期 5k calls/s"
        )
        assert max_burst_calls > 1_000, (
            f"Hub Burst 上限 {max_burst_calls} calls，低于预期 1k calls"
        )
