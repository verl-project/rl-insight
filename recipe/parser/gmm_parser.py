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

from loguru import logger
import re
from pathlib import Path
from typing import Any, List, Optional, Union
import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig

from recipe.config import get_config_value
from recipe.parser.parser import BaseClusterParser, register_cluster_parser
from recipe.utils.schema import DataMap
from recipe.data import DataEnum


@register_cluster_parser("gmm")
class GmmParser(BaseClusterParser):
    input_type = DataEnum.GMM_DATA

    def __init__(self, params: Union[DictConfig, dict]) -> None:
        super().__init__(params)
        self.events_summary: Optional[pd.DataFrame] = None
        rank_list = get_config_value(params, "input.rank_list", "all")
        self._rank_list = (
            rank_list
            if rank_list == "all"
            else [
                int(rank.strip())
                for rank in rank_list.split(",")
                if rank.strip().isdigit()
            ]
        )
        step = get_config_value(params, "heatmap.parser.step", None)
        if step is None:
            self._step_list: Optional[list[int]] = None
        elif isinstance(step, int):
            self._step_list = [step]
        else:
            step_tokens = str(step).split(",")
            self._step_list = [
                int(token.strip()) for token in step_tokens if token.strip().isdigit()
            ]
            if not self._step_list:
                logger.warning(
                    f"Invalid step filter: {step!r}. Expected int or comma-separated ints. "
                    "Will process all steps."
                )
                self._step_list = None
        self._role = get_config_value(params, "heatmap.parser.role", None)

    @staticmethod
    def _normalize_path_text(path_value: str | Path) -> str:
        return str(path_value).replace("\\", "/")

    @classmethod
    def _extract_rank_id_from_path(cls, path_value: str | Path) -> int:
        normalized = cls._normalize_path_text(path_value)
        m_rank = re.search(r"(?:^|/)rank(\d+)(?:/|$)", normalized)
        return int(m_rank.group(1)) if m_rank else -1

    @classmethod
    def _extract_step_from_path(cls, path_value: str | Path) -> int:
        normalized = cls._normalize_path_text(path_value)
        m_step = re.search(r"(?:^|/)step_(\d+)(?:/|$)", normalized)
        return int(m_step.group(1)) if m_step else -1

    def allocate_prof_data(self, input_path: str) -> List[DataMap]:
        """Allocate and organize GMM profiling data from the input path."""
        data_maps: List[DataMap] = []
        root = Path(input_path)

        if not root.is_dir():
            logger.warning(f"Input path is not a directory: {input_path}")
            return data_maps

        # Find all group_list.pt files
        group_list_files = list(root.rglob("*group_list.pt"))
        logger.info(f"Found {len(group_list_files)} group_list.pt files")

        for file_path in group_list_files:
            # Skip files not in dump_tensor_data directory
            if "dump_tensor_data" not in file_path.parts:
                continue

            # Parse rank, step, stage from path
            parts = file_path.parts
            rank_id = self._extract_rank_id_from_path(file_path)
            if rank_id < 0:
                continue
            step = self._extract_step_from_path(file_path)
            if step < 0:
                continue

            # Extract stage
            stage = None
            for i, p in enumerate(parts):
                if p.startswith("step_") and i + 1 < len(parts):
                    stage = parts[i + 1]
                    break
            if stage is None or not stage or stage.startswith("step"):
                continue

            # Check if rank is in the specified rank list
            if self._rank_list != "all" and rank_id not in self._rank_list:
                continue

            # Check if step matches the specified step filter(s)
            if self._step_list is not None and step not in self._step_list:
                continue

            # Check if role matches the specified role
            if self._role is not None and stage != self._role:
                continue

            data_map: DataMap = {
                "rank_id": rank_id,
                "role": stage,
                "step": step,
                "profiler_data_path": str(file_path),
            }
            data_maps.append(data_map)

        logger.info(f"Allocated {len(data_maps)} data maps for GMM parsing")
        return data_maps

    def _load_group_list(self, file_path: str) -> np.ndarray:
        """Load a group_list.pt file into a numpy array."""
        try:
            obj = torch.load(file_path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError):
            obj = torch.load(file_path, map_location="cpu")
        if torch.is_tensor(obj):
            arr = obj.detach().float().cpu().numpy().ravel()
        elif isinstance(obj, np.ndarray):
            arr = obj.astype(np.float64).ravel()
        else:
            raise ValueError(f"Unexpected object in {file_path}: {type(obj)}")
        return arr

    @staticmethod
    def _training_step_from_path(profiler_data_path: str) -> int:
        step = GmmParser._extract_step_from_path(profiler_data_path)
        return step if step >= 0 else 0

    def parse_analysis_data(
        self, profiler_data_path: str, rank_id: int, role: str
    ) -> list[dict[str, Any]]:
        """Parse GMM profiling data for a specific rank and return GMM row information."""
        step = self._training_step_from_path(profiler_data_path)
        events: list[dict[str, Any]] = []
        try:
            # Load group_list data
            group_list = self._load_group_list(profiler_data_path)
            logger.info(
                f"Loaded group_list with {len(group_list)} experts from {profiler_data_path}"
            )

            # Extract op index (stage) from file name
            file_name = Path(profiler_data_path).name
            m_op = re.search(
                r"npu_grouped_matmul\.(\d+)\.forward\.kwargs\.group_list\.pt$",
                file_name,
            )
            op_index = int(m_op.group(1)) if m_op else 0
            stage_idx = op_index  # Use op_index as stage index

            # Create GmmRow for each expert
            for expert_idx, load in enumerate(group_list):
                event: dict[str, Any] = {
                    "role": role,
                    "rank_id": rank_id,
                    "step": step,
                    "stage": stage_idx,
                    "expert_index": expert_idx,
                    "load": load,
                }
                events.append(event)
            logger.info(
                f"Created {len(events)} GmmRow entries for {profiler_data_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to parse {profiler_data_path}: {e}")

        return events

    def reducer_func(self, mapper_res):
        """Process data collected from all ranks"""
        # Flatten valid results from all ranks
        reduce_results: list[dict] = []
        for result in mapper_res:
            if not result:
                continue
            if isinstance(result, list):
                # GmmRow is a TypedDict, which is already a dict
                for event in result:
                    if isinstance(event, dict):
                        reduce_results.append(event)
                    elif hasattr(event, "__dict__"):
                        reduce_results.append(event.__dict__)
                    else:
                        reduce_results.append(event)
            else:
                raise TypeError(
                    f"parse_analysis_data must return list[GmmRow] or None, got {type(result)}"
                )

        if not reduce_results:
            logger.warning("No valid data collected from any rank")
            return

        self.events_summary = pd.DataFrame(reduce_results)
        if not self.events_summary.empty:
            self.events_summary.sort_values(
                ["step", "role", "rank_id", "stage", "expert_index", "load"],
                inplace=True,
            )
        logger.info(f"Sorted {len(reduce_results)} events")
        logger.info(f"Created DataFrame with {len(self.events_summary)} rows")
        if not self.events_summary.empty:
            logger.info(f"DataFrame columns: {list(self.events_summary.columns)}")
            logger.info(f"Sample data: {self.events_summary.head()}")

    def get_data(self) -> pd.DataFrame:
        """Return the parsed DataFrame"""
        ## debug print pd.DataFrame to excel
        return self.events_summary
