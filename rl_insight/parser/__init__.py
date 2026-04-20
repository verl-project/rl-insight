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

import argparse

from .mstx_parser import MstxClusterParser
from .torch_parser import TorchClusterParser
from .nvtx_parser import NvtxClusterParser
from .parser import BaseClusterParser, get_cluster_parser_cls as _get_cluster_parser_cls


def get_cluster_parser_cls(name):
    if name == "gmm":
        # Lazy import keeps optional gmm dependency off non-gmm paths.
        from . import gmm_parser  # noqa: F401
    return _get_cluster_parser_cls(name)


def register_parser_specific_args(arg_parser: argparse.ArgumentParser) -> None:
    """Register optional parser CLI flags (additive). Safe for non-GMM runs; extras are ignored."""
    gmm_group = arg_parser.add_argument_group("GMM parser parameters")
    gmm_group.add_argument(
        "--step",
        type=str,
        help="Step filter for GMM parser, e.g. '1' or '1,2'",
    )
    gmm_group.add_argument(
        "--role",
        type=str,
        help="Role filter for GMM parser",
    )


def __getattr__(name):
    if name == "GmmParser":
        from .gmm_parser import GmmParser

        return GmmParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseClusterParser",
    "get_cluster_parser_cls",
    "register_parser_specific_args",
    "MstxClusterParser",
    "TorchClusterParser",
    "NvtxClusterParser",
    "GmmParser",
]
