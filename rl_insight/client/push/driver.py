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

"""Configuration-driven sink driver loading.

A driver is referenced from YAML as a dotted path ``"pkg.module:factory"``; the
factory is called with the sink config node and must return a
:class:`~rl_insight.client.push.sinks.base.PushSink` (or ``None`` to disable
that sink). No deployment-specific package name appears anywhere in this code.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from omegaconf import OmegaConf

from .sinks.base import PushSink

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

SinkFactory = Callable[[Any], "PushSink | None"]


def load_sink_factory(dotted_path: str) -> SinkFactory:
    """Resolve ``"pkg.module:attr"`` to a callable factory.

    Args:
        dotted_path: Colon-separated module and attribute path.

    Returns:
        The resolved factory callable.

    Raises:
        ValueError: Malformed path (missing module, colon, or attribute).
        ImportError: Module cannot be imported or attribute is missing.
    """
    module_path, separator, attr_path = str(dotted_path).partition(":")
    if not separator or not module_path.strip() or not attr_path.strip():
        raise ValueError(
            f"Invalid sink driver {dotted_path!r}; expected 'package.module:factory'"
        )
    module = importlib.import_module(module_path.strip())
    try:
        factory = getattr(module, attr_path.strip())
    except AttributeError as exc:
        raise ImportError(
            f"Sink driver module {module_path!r} has no attribute {attr_path!r}"
        ) from exc
    if not callable(factory):
        raise TypeError(f"Sink driver {dotted_path!r} is not callable")
    return factory


def build_sink(sink_conf: Any) -> PushSink | None:
    """Instantiate one sink from its config node, never raising.

    Reads ``driver`` (dotted path) and calls the factory with the whole sink
    config node. Any failure (bad path, import error, factory error, wrong
    return type) logs one warning and returns ``None`` so one broken sink never
    disables monitoring for the others or training itself.
    """
    dotted = OmegaConf.select(sink_conf, "driver")
    name = str(OmegaConf.select(sink_conf, "name") or (dotted or "sink"))
    if not dotted:
        logger.warning("[rl-insight] push sink %r is missing 'driver'; skipping.", name)
        return None
    try:
        factory = load_sink_factory(str(dotted))
        sink = factory(sink_conf)
    except (ValueError, ImportError, TypeError) as exc:
        logger.warning("[rl-insight] push sink %r driver unavailable: %s", name, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - driver errors must not break training
        logger.warning("[rl-insight] push sink %r factory failed: %s", name, exc)
        return None

    if sink is None:
        logger.info("[rl-insight] push sink %r disabled itself (factory returned None).", name)
        return None
    if not isinstance(sink, PushSink):
        logger.warning(
            "[rl-insight] push sink %r returned %r, not a PushSink; skipping.",
            name,
            type(sink).__name__,
        )
        return None
    return sink
