# Copyright (c) 2025 verl-project authors.
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

from typing import Any, Union

from omegaconf import DictConfig


def get_config_value(
    config: Union[DictConfig, dict], key: str, default: Any = None
) -> Any:
    """Retrieve a value from a DictConfig or dict using a dot-separated key.

    Supports both nested access (``output.path``) and flat key fallback
    (``output_path``) for backward compatibility.

    Args:
        config: Configuration object (DictConfig or plain dict).
        key: Dot-separated key path, e.g. ``"output.path"``.
        default: Value returned when the key is not found.

    Returns:
        The resolved value, or *default* if the key does not exist.
    """
    if isinstance(config, DictConfig):
        flat_key = key.replace(".", "_")
        if hasattr(config, flat_key):
            return getattr(config, flat_key)
        parts = key.split(".")
        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value

    flat_key = key.replace(".", "_")
    if flat_key in config:
        return config.get(flat_key)
    if key in config:
        return config.get(key)

    last_part = key.split(".")[-1]
    if last_part in config:
        return config.get(last_part)

    nested_parts = key.split(".")
    if len(nested_parts) > 1:
        value = config
        for part in nested_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    return default
