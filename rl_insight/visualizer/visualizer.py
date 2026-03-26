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

from abc import ABC, abstractmethod
from typing import Callable

from rl_insight.data import DataEnum


class BaseVisualizer(ABC):
    input_type: DataEnum = DataEnum.SUMMARY_EVENT

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def run(self, data):
        raise NotImplementedError


CLUSTER_VISUALIZER_REGISTRY: dict[str, type[BaseVisualizer]] = {}


def register_cluster_visualizer(
    name: str,
) -> Callable[[type[BaseVisualizer]], type[BaseVisualizer]]:
    def decorator(cls: type[BaseVisualizer]) -> type[BaseVisualizer]:
        CLUSTER_VISUALIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_cluster_visualizer_cls(name: str):
    if name not in CLUSTER_VISUALIZER_REGISTRY:
        raise ValueError(
            f"Unsupported cluster visualizer: {name}. Supported cls are: {list(CLUSTER_VISUALIZER_REGISTRY.keys())}"
        )
    return CLUSTER_VISUALIZER_REGISTRY[name]
