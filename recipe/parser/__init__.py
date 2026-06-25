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

from .memory_parser import MemoryClusterParser
from .mstx_parser import MstxClusterParser
from .torch_parser import TorchClusterParser
from .nvtx_parser import NvtxClusterParser
from .parser import BaseClusterParser, get_cluster_parser_cls as _get_cluster_parser_cls


def get_cluster_parser_cls(name):
    if name == "gmm":
        from . import gmm_parser  # noqa: F401
    return _get_cluster_parser_cls(name)


def __getattr__(name):
    if name == "GmmParser":
        from .gmm_parser import GmmParser

        return GmmParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseClusterParser",
    "get_cluster_parser_cls",
    "register_parser_specific_args",
    "MemoryClusterParser",
    "MstxClusterParser",
    "TorchClusterParser",
    "NvtxClusterParser",
    "GmmParser",
]
