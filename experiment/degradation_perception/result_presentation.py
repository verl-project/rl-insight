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

"""Small, tester-facing view of the full association-analysis response."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any


def build_top5_result(
    analysis_result: Mapping[str, Any],
    target_metric: str,
) -> dict[str, Any]:
    """Return a strict-JSON summary while preserving every target event."""

    association = analysis_result.get("associationAnalysis", {})
    targets = association.get("targets", {}) if isinstance(association, Mapping) else {}
    target = targets.get(target_metric, {}) if isinstance(targets, Mapping) else {}
    target_status = (
        str(target.get("status", "association_unavailable"))
        if isinstance(target, Mapping)
        else "association_unavailable"
    )
    raw_events = target.get("events", []) if isinstance(target, Mapping) else []
    events: list[dict[str, Any]] = []
    if isinstance(raw_events, Sequence) and not isinstance(
        raw_events, (str, bytes, bytearray)
    ):
        for event_index, raw_event in enumerate(raw_events, start=1):
            if not isinstance(raw_event, Mapping):
                continue
            raw_top = raw_event.get("topAssociations", [])
            top5: list[dict[str, Any]] = []
            if isinstance(raw_top, Sequence) and not isinstance(
                raw_top, (str, bytes, bytearray)
            ):
                for item in raw_top[:5]:
                    if not isinstance(item, Mapping):
                        continue
                    contribution = item.get("abnormalContribution")
                    top5.append(
                        {
                            "rank": int(item["rank"]),
                            "metric": str(item["metric"]),
                            "abnormalContribution": round(float(contribution), 2),
                            "correlationDirection": item.get(
                                "correlationDirection"
                            ),
                        }
                    )
            raw_range = raw_event.get("targetAbnormalRange")
            simple_range = (
                {
                    key: copy.deepcopy(raw_range[key])
                    for key in (
                        "startTime",
                        "endTime",
                        "abnormalType",
                        "abnormalPointCount",
                    )
                    if key in raw_range
                }
                if isinstance(raw_range, Mapping)
                else None
            )
            events.append(
                {
                    "event": event_index,
                    "status": str(raw_event.get("status", target_status)),
                    "abnormalTimeRange": simple_range,
                    "top5": top5,
                }
            )

    abnormal_ranges = analysis_result.get("abnormalTimeRange", {})
    target_ranges = (
        abnormal_ranges.get(target_metric, [])
        if isinstance(abnormal_ranges, Mapping)
        else []
    )
    result = {
        "formatVersion": 1,
        "status": target_status,
        "anomalyMetric": target_metric,
        "anomalyDetected": bool(target_ranges),
        "eventCount": len(events),
        "contributionUnit": "percent",
        "events": events,
        "notice": "异常贡献度是关联排名分数，不是因果概率。",
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
