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

"""Unit tests for Prometheus -> sink translation rules."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from rl_insight.client.push.translate import (
    OP_COUNTER,
    OP_STORE,
    OP_TIMER,
    ScrapeState,
)

_RULES = OmegaConf.create(
    [
        {
            "source": "vllm:prompt_tokens_total",
            "type": "counter_delta",
            "name": "token_throughput",
            "tags": {"stage": "prompt"},
        },
        {
            "source": "vllm:time_to_first_token_seconds",
            "type": "summary_mean_us",
            "name": "TTFT",
        },
        {"source": "vllm:num_requests_running", "type": "gauge", "name": "running"},
        {
            "numerator": "vllm:gpu_cache_usage",
            "denominator": "vllm:gpu_cache_max",
            "type": "ratio",
            "name": "kv_cache_ratio",
        },
    ]
)

_SCRAPE_1 = """
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 100.0
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total_created{model_name="m"} 1.0
# TYPE vllm:time_to_first_token_seconds summary
vllm:time_to_first_token_seconds_sum{model_name="m"} 0.5
vllm:time_to_first_token_seconds_count{model_name="m"} 5.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3.0
# TYPE vllm:gpu_cache_usage gauge
vllm:gpu_cache_usage 160.0
# TYPE vllm:gpu_cache_max gauge
vllm:gpu_cache_max 200.0
"""

_SCRAPE_2 = """
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 150.0
vllm:prompt_tokens_total_created{model_name="m"} 1.0
# TYPE vllm:time_to_first_token_seconds summary
vllm:time_to_first_token_seconds_sum{model_name="m"} 0.8
vllm:time_to_first_token_seconds_count{model_name="m"} 8.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 4.0
# TYPE vllm:gpu_cache_usage gauge
vllm:gpu_cache_usage 180.0
# TYPE vllm:gpu_cache_max gauge
vllm:gpu_cache_max 200.0
"""


def _by_name(emissions, name):
    return [e for e in emissions if e.name == name]


def test_first_scrape_only_baselines_for_delta_rules() -> None:
    state = ScrapeState()
    out = state.translate(_SCRAPE_1, _RULES)
    names = {e.name for e in out}
    assert "token_throughput" not in names  # counter needs two scrapes
    assert "TTFT" not in names  # summary needs two scrapes
    assert "running" in names  # gauge emits immediately
    assert "kv_cache_ratio" in names  # ratio emits immediately


def test_gauge_emits_current_value_with_sample_labels() -> None:
    state = ScrapeState()
    out = state.translate(_SCRAPE_1, _RULES)
    running = _by_name(out, "running")
    assert len(running) == 1
    assert running[0].op == OP_STORE
    assert running[0].value == 3.0
    assert running[0].tags["model_name"] == "m"


def test_gauge_timer_delivers_gauge_value_as_timer() -> None:
    state = ScrapeState()
    rules = OmegaConf.create(
        [
            {
                "source": "vllm:num_requests_running",
                "type": "gauge_timer",
                "name": "decode_batch_size",
                "tags": {"pending_status": "infer"},
            }
        ]
    )
    out = state.translate(_SCRAPE_1, rules)
    assert len(out) == 1
    assert out[0].name == "decode_batch_size"
    assert out[0].op == OP_TIMER
    assert out[0].value == 3.0
    assert out[0].tags["model_name"] == "m"
    assert out[0].tags["pending_status"] == "infer"


def test_ratio_emits_numerator_over_denominator() -> None:
    state = ScrapeState()
    out = state.translate(_SCRAPE_1, _RULES)
    ratio = _by_name(out, "kv_cache_ratio")
    assert ratio[0].op == OP_STORE
    assert ratio[0].value == 0.8


def test_second_scrape_emits_counter_delta_and_rule_tags() -> None:
    state = ScrapeState()
    state.translate(_SCRAPE_1, _RULES)
    out = state.translate(_SCRAPE_2, _RULES)
    counters = _by_name(out, "token_throughput")
    assert len(counters) == 1
    assert counters[0].op == OP_COUNTER
    assert counters[0].value == 50.0
    assert counters[0].tags["stage"] == "prompt"
    assert counters[0].tags["model_name"] == "m"


def test_second_scrape_emits_summary_mean_in_microseconds() -> None:
    state = ScrapeState()
    state.translate(_SCRAPE_1, _RULES)
    out = state.translate(_SCRAPE_2, _RULES)
    ttft = _by_name(out, "TTFT")
    assert len(ttft) == 1
    assert ttft[0].op == OP_TIMER
    # (0.8 - 0.5) / (8 - 5) seconds = 0.1s = 100_000 us
    assert ttft[0].value == pytest.approx(100_000.0)


def test_missing_source_is_skipped() -> None:
    state = ScrapeState()
    rules = OmegaConf.create(
        [{"source": "absent_metric", "type": "gauge", "name": "x"}]
    )
    assert state.translate(_SCRAPE_1, rules) == []


def test_counter_reset_is_rebaselined_without_negative_delta() -> None:
    state = ScrapeState()
    state.translate(_SCRAPE_1, _RULES)
    text = """
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 10.0
"""
    out = state.translate(text, [_RULES[0]])
    assert _by_name(out, "token_throughput") == []


def test_zero_denominator_ratio_is_skipped() -> None:
    state = ScrapeState()
    text = """
# TYPE a gauge
a 1.0
# TYPE b gauge
b 0.0
"""
    rules = OmegaConf.create(
        [{"numerator": "a", "denominator": "b", "type": "ratio", "name": "r"}]
    )
    assert state.translate(text, rules) == []
