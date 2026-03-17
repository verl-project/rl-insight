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

"""
Cluster scheduling analysis and visualization for RL workloads.

This package exposes:

- ``main.main``: CLI entry point
- ``mstx_parser.MstxClusterParser``: parser for Ascend MSTX traces
"""

from .main import main  # noqa: F401

from .parser import mstx_parser
from .parser import torch_parser

__all__ = ["mstx_parser", "torch_parser"]
