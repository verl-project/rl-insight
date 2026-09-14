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

"""Translate Prometheus text exposition into sink operations.

One :class:`ScrapeState` instance is kept per scrape target so monotonic
counters can be reported as inter-poll deltas and Prometheus ``summary``
values can be reported as mean latency in microseconds
(``delta_sum / delta_count * 1e6``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf
from prometheus_client.parser import text_string_to_metric_families

# Sink operation names.
OP_COUNTER = "counter"
OP_STORE = "store"
OP_TIMER = "timer"

# Supported translation rule types.
RULE_COUNTER_DELTA = "counter_delta"
RULE_SUMMARY_MEAN_US = "summary_mean_us"
RULE_GAUGE = "gauge"
RULE_GAUGE_TIMER = "gauge_timer"
RULE_RATIO = "ratio"

_SECONDS_TO_US = 1_000_000.0


@dataclass(frozen=True)
class Emission:
    """One translated sink operation (name is relative to the sink prefix)."""

    name: str
    op: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)


def build_family_map(text: str) -> dict[str, Any]:
    """Parse Prometheus text exposition into ``{name: family}``.

    Counter family names have a trailing ``_total`` stripped by the parser, so
    both the stripped name and the original ``..._total`` sample name are
    registered to keep configuration matching the wire metric name intuitive.
    """
    families: dict[str, Any] = {}
    for family in text_string_to_metric_families(text):
        families[family.name] = family
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            families.setdefault(sample.name, family)
    return families


def _rule(conf: Any, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(conf, key, default=default)
    return default if value is None else value


def _rule_tags(conf: Any) -> dict[str, str]:
    raw = OmegaConf.select(conf, "tags")
    if raw is None:
        return {}
    return {str(k): str(v) for k, v in dict(raw).items()}


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _counter_samples(family: Any):
    """Yield a counter family's value samples, excluding ``_created``."""
    for sample in family.samples:
        if sample.name.endswith("_created"):
            continue
        if sample.name == family.name or sample.name == f"{family.name}_total":
            yield sample


def _summary_pair(family: Any):
    """Yield ``(labels, sum_value, count_value)`` for a summary family."""
    sums: dict[tuple, tuple[dict[str, str], float]] = {}
    counts: dict[tuple, float] = {}
    for sample in family.samples:
        labels = dict(sample.labels)
        if sample.name.endswith("_sum"):
            sums[_label_key(labels)] = (labels, float(sample.value))
        elif sample.name.endswith("_count"):
            counts[_label_key(labels)] = float(sample.value)
    for key, (labels, sum_value) in sums.items():
        if key in counts:
            yield labels, sum_value, counts[key]


class ScrapeState:
    """Per-target previous-scrape values and the translation entry point."""

    def __init__(self) -> None:
        # key -> previous numeric value(s)
        self._prev: dict[tuple[str, tuple], Any] = {}

    def translate(self, text: str, rules: Any) -> list[Emission]:
        """Translate one scrape ``text`` according to ``rules``.

        The first scrape only establishes baselines: ``counter_delta`` and
        ``summary_mean_us`` rules emit nothing until the second scrape.
        """
        families = build_family_map(text)
        emissions: list[Emission] = []
        for rule in rules or []:
            rtype = str(_rule(rule, "type"))
            if rtype == RULE_COUNTER_DELTA:
                emissions.extend(self._counter_delta(families, rule))
            elif rtype == RULE_SUMMARY_MEAN_US:
                emissions.extend(self._summary_mean_us(families, rule))
            elif rtype == RULE_GAUGE:
                emissions.extend(self._gauge(families, rule))
            elif rtype == RULE_GAUGE_TIMER:
                emissions.extend(self._gauge(families, rule, OP_TIMER))
            elif rtype == RULE_RATIO:
                emissions.extend(self._ratio(families, rule))
        return emissions

    def _tags(self, rule: Any, sample_labels: dict[str, str] | None) -> dict[str, str]:
        merged = {str(k): str(v) for k, v in (sample_labels or {}).items()}
        merged.update(_rule_tags(rule))
        return merged

    def _counter_delta(self, families: dict[str, Any], rule: Any) -> list[Emission]:
        source = str(_rule(rule, "source"))
        name = str(_rule(rule, "name"))
        family = families.get(source)
        if family is None:
            return []
        out: list[Emission] = []
        for sample in _counter_samples(family):
            key = (source, _label_key(dict(sample.labels)))
            current = float(sample.value)
            previous = self._prev.get(key)
            self._prev[key] = current
            if previous is None:
                continue  # first scrape: baseline only
            delta = current - previous
            if delta < 0:  # counter reset after restart: re-baseline, skip
                continue
            out.append(
                Emission(name, OP_COUNTER, delta, self._tags(rule, dict(sample.labels)))
            )
        return out

    def _summary_mean_us(self, families: dict[str, Any], rule: Any) -> list[Emission]:
        source = str(_rule(rule, "source"))
        name = str(_rule(rule, "name"))
        family = families.get(source)
        if family is None:
            return []
        out: list[Emission] = []
        for labels, sum_value, count_value in _summary_pair(family):
            key = (source, _label_key(labels))
            previous = self._prev.get(key)
            self._prev[key] = (sum_value, count_value)
            if previous is None:
                continue
            prev_sum, prev_count = previous
            delta_count = count_value - prev_count
            if delta_count <= 0:
                continue
            mean_seconds = (sum_value - prev_sum) / delta_count
            out.append(
                Emission(
                    name,
                    OP_TIMER,
                    mean_seconds * _SECONDS_TO_US,
                    self._tags(rule, labels),
                )
            )
        return out

    def _gauge(
        self, families: dict[str, Any], rule: Any, op: str = OP_STORE
    ) -> list[Emission]:
        source = str(_rule(rule, "source"))
        name = str(_rule(rule, "name"))
        family = families.get(source)
        if family is None:
            return []
        return [
            Emission(
                name,
                op,
                float(sample.value),
                self._tags(rule, dict(sample.labels)),
            )
            for sample in family.samples
        ]

    def _ratio(self, families: dict[str, Any], rule: Any) -> list[Emission]:
        numerator_source = str(_rule(rule, "numerator"))
        denominator_source = str(_rule(rule, "denominator"))
        name = str(_rule(rule, "name"))
        num_family = families.get(numerator_source)
        den_family = families.get(denominator_source)
        if num_family is None or den_family is None:
            return []
        if not num_family.samples or not den_family.samples:
            return []
        denominator = float(den_family.samples[0].value)
        if denominator == 0:
            return []
        value = float(num_family.samples[0].value) / denominator
        return [Emission(name, OP_STORE, value, _rule_tags(rule))]
