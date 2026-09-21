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

"""Generic mechanism for deriving low-cardinality tags from environment variables.

Only the *mechanism* lives here; which variables map to which tags is pure
configuration supplied by the deployment profile (e.g. ``tags_from_env``).
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from omegaconf import OmegaConf

MISSING = object()


def _select(rule: Any, key: str) -> Any:
    return OmegaConf.select(rule, key, default=MISSING)


def collect_env_tags(
    rules: Any, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build a tag map from a list of ``tags_from_env`` rule objects.

    Each rule supports:
      * ``tag``: output tag name (required);
      * ``env``: one env var name or a fallback list; first non-empty value wins;
      * ``default``: value used when every listed variable is absent/empty; if
        omitted, the tag is skipped entirely;
      * ``format``: optional ``str.format`` template receiving ``{value}``.

    Args:
        rules: Config list/sequence of rule mappings (or ``None``).
        environ: Environment mapping (defaults to :data:`os.environ`).

    Returns:
        Mapping of ``tag -> string value``.
    """
    env_source = os.environ if environ is None else environ
    tags: dict[str, str] = {}
    if rules is None:
        return tags

    for rule in rules or []:
        tag = _select(rule, "tag")
        if tag is MISSING or tag is None:
            continue
        candidates = _select(rule, "env")
        if candidates is MISSING or candidates is None:
            candidates = []
        elif isinstance(candidates, str):
            candidates = [candidates]
        else:
            candidates = list(candidates)

        value: Any = MISSING
        for var in candidates:
            raw = env_source.get(str(var))
            if raw is not None and str(raw) != "":
                value = raw
                break

        if value is MISSING:
            default = _select(rule, "default")
            if default is MISSING or default is None:
                continue
            value = default

        text = str(value)
        fmt = _select(rule, "format")
        if fmt is not MISSING and fmt is not None:
            text = str(fmt).format(value=text)
        tags[str(tag)] = text
    return tags
