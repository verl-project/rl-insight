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

import math
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
RULE_HISTOGRAM = "histogram"
RULE_COUNTER_RATIO = "counter_ratio"

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


def _value_samples(family: Any) -> dict[tuple, Any]:
    """Map each label set of ``family`` to its single value sample.

    Only families that expose one value sample per label set are usable as a
    ratio operand; a label set seen twice (summary ``_sum``/``_count`` pairs,
    histogram buckets) is ambiguous and therefore dropped instead of being
    resolved by picking an arbitrary one.
    """
    names = {family.name, f"{family.name}_total"}
    samples = [
        sample
        for sample in family.samples
        if not sample.name.endswith(("_created", "_bucket"))
    ]
    preferred = [sample for sample in samples if sample.name in names]
    out: dict[tuple, Any] = {}
    ambiguous: set[tuple] = set()
    for sample in preferred or samples:
        key = _label_key(dict(sample.labels))
        if key in out:
            ambiguous.add(key)
        out[key] = sample
    for key in ambiguous:
        out.pop(key, None)
    return out


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
        self._prev: dict[tuple[Any, ...], Any] = {}

    def translate(self, text: str, rules: Any) -> list[Emission]:
        """Translate one scrape ``text`` according to ``rules``.

        The first scrape only establishes baselines: ``counter_delta`` and
        ``summary_mean_us`` rules emit nothing until the second scrape.
        """
        families = build_family_map(text)
        emissions: list[Emission] = []
        for rule in rules or []:
            presence = _rule(rule, "presence")
            if presence:
                sources = (
                    [_rule(rule, "source")]
                    if _rule(rule, "source")
                    else [_rule(rule, "numerator"), _rule(rule, "denominator")]
                )
                present = all(
                    families.get(str(source)) is not None for source in sources
                )
                emissions.append(
                    Emission(
                        str(presence),
                        OP_STORE,
                        float(present),
                        {"metric": str(_rule(rule, "name"))},
                    )
                )
            rtype = str(_rule(rule, "type"))
            if rtype == RULE_COUNTER_DELTA:
                emissions.extend(self._counter_delta(families, rule))
            elif rtype == RULE_SUMMARY_MEAN_US:
                emissions.extend(self._summary_mean_us(families, rule))
            elif rtype == RULE_GAUGE:
                emissions.extend(self._gauge(families, rule))
            elif rtype == RULE_GAUGE_TIMER:
                emissions.extend(self._gauge(families, rule, OP_TIMER))
            elif rtype == RULE_HISTOGRAM:
                emissions.extend(self._histogram(families, rule))
            elif rtype == RULE_COUNTER_RATIO:
                emissions.extend(self._counter_ratio(families, rule))
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
        samples = list(_counter_samples(family))
        if not samples:
            samples = [sample for sample in family.samples if sample.name == source]
        for sample in samples:
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
        """Divide numerator by denominator per source label set.

        Families are joined on identical labels so a multi-series source (for
        example one series per ``model_name``) yields one ratio per series
        instead of dividing an arbitrary numerator series by an arbitrary
        denominator series. A denominator that exposes a single series is used
        as the shared divisor for every numerator series; a series whose own
        denominator is zero is skipped without discarding the others.
        """
        numerator_source = str(_rule(rule, "numerator"))
        denominator_source = str(_rule(rule, "denominator"))
        name = str(_rule(rule, "name"))
        num_family = families.get(numerator_source)
        den_family = families.get(denominator_source)
        if num_family is None or den_family is None:
            return []
        numerators = _value_samples(num_family)
        denominators = _value_samples(den_family)
        if not numerators or not denominators:
            return []
        fallback = next(iter(denominators.values())) if len(denominators) == 1 else None
        out = []
        for key, numerator in numerators.items():
            denominator = denominators.get(key, fallback)
            if denominator is None:
                continue
            divisor = float(denominator.value)
            if divisor == 0:
                continue
            out.append(
                Emission(
                    name,
                    OP_STORE,
                    float(numerator.value) / divisor,
                    self._tags(rule, dict(numerator.labels)),
                )
            )
        return out

    def _histogram(self, families: dict[str, Any], rule: Any) -> list[Emission]:
        """Preserve interval bucket/sum/count increments and derive per-label quantiles.

        Quantiles are gauges over the latest scrape interval, not samples drawn
        from a distribution of interval means. Consumers must never average
        quantiles across replicas or time; raw bucket counters are also emitted
        so a histogram-capable backend can aggregate the distribution instead.
        """
        source = str(_rule(rule, "source"))
        name = str(_rule(rule, "name"))
        family = families.get(source)
        if family is None:
            return []
        scale = float(_rule(rule, "scale", 1.0))
        quantiles = [float(q) for q in _rule(rule, "quantiles", [0.5, 0.9, 0.95, 0.99])]
        if (
            not math.isfinite(scale)
            or scale <= 0
            or any(not 0 <= q <= 1 for q in quantiles)
        ):
            raise ValueError(
                "histogram requires a finite positive scale and quantiles in [0, 1]"
            )
        buckets: dict[tuple, dict[float, float]] = {}
        for sample in family.samples:
            if sample.name != source + "_bucket":
                continue
            labels = dict(sample.labels)
            upper = float(labels.pop("le"))
            key = _label_key(labels)
            buckets.setdefault(key, {})[upper] = float(sample.value)
        out = []
        for labels, total, count in _summary_pair(family):
            key = _label_key(labels)
            current = buckets.get(key, {})
            state_key = ("histogram", source, name, key)
            previous = self._prev.get(state_key)
            self._prev[state_key] = (dict(current), total, count)
            if previous is None:
                continue
            old, old_sum, old_count = previous
            if (
                not math.isfinite(total)
                or not math.isfinite(count)
                or current.keys() != old.keys()
                or count < old_count
                or total < old_sum
            ):
                continue
            increments = sorted(
                (upper, value - old[upper]) for upper, value in current.items()
            )
            if (
                not increments
                or any(math.isnan(upper) or upper < 0 for upper, _ in increments)
                or not math.isinf(increments[-1][0])
            ):
                continue
            values = [value for _, value in increments]
            delta_count = count - old_count
            if (
                any(not math.isfinite(v) or v < 0 for v in values)
                or any(value < values[i] for i, value in enumerate(values[1:]))
                or not math.isclose(values[-1], delta_count)
            ):
                continue
            tags = self._tags(rule, labels)
            for upper, value in increments:
                out.append(
                    Emission(
                        name + ".bucket", OP_COUNTER, value, {**tags, "le": str(upper)}
                    )
                )
            out.append(Emission(name + ".sum", OP_COUNTER, total - old_sum, tags))
            out.append(Emission(name + ".count", OP_COUNTER, delta_count, tags))
            if delta_count <= 0:
                continue
            out.append(
                Emission(
                    name + ".mean",
                    OP_STORE,
                    (total - old_sum) / delta_count * scale,
                    tags,
                )
            )
            for q in quantiles:
                rank = q * delta_count
                lower, before = 0.0, 0.0
                for upper, cumulative in increments:
                    if cumulative >= rank and cumulative > before:
                        # Prometheus classic histogram linear interpolation;
                        # observations in +Inf resolve to the last finite bound.
                        value = (
                            lower
                            if math.isinf(upper)
                            else (
                                lower
                                + (upper - lower)
                                * (rank - before)
                                / (cumulative - before)
                            )
                        )
                        out.append(
                            Emission(
                                name + ".quantile",
                                OP_STORE,
                                value * scale,
                                {**tags, "quantile": str(q)},
                            )
                        )
                        break
                    lower, before = upper, cumulative
        return out

    def _counter_ratio(self, families: dict[str, Any], rule: Any) -> list[Emission]:
        """Join counter deltas by labels; never divide unrelated targets/series."""
        numerator = families.get(str(_rule(rule, "numerator")))
        denominator = families.get(str(_rule(rule, "denominator")))
        if numerator is None or denominator is None:
            return []
        nums = {_label_key(dict(s.labels)): s for s in _counter_samples(numerator)}
        dens = {_label_key(dict(s.labels)): s for s in _counter_samples(denominator)}
        name = str(_rule(rule, "name"))
        out = []
        for key in nums.keys() & dens.keys():
            num, den = float(nums[key].value), float(dens[key].value)
            state_key = ("counter_ratio", name, key)
            previous = self._prev.get(state_key)
            self._prev[state_key] = (num, den)
            if previous is None or num < previous[0] or den <= previous[1]:
                continue
            value = (num - previous[0]) / (den - previous[1])
            value = value * float(_rule(rule, "scale", 1.0)) + float(
                _rule(rule, "offset", 0.0)
            )
            if math.isfinite(value):
                out.append(
                    Emission(
                        name, OP_STORE, value, self._tags(rule, dict(nums[key].labels))
                    )
                )
        return out
