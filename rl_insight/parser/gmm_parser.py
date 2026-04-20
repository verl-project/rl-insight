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
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit("This parser requires PyTorch: pip install torch") from e

from rl_insight.parser.parser import BaseClusterParser, register_cluster_parser
from rl_insight.utils.schema import DataMap, EventRow, Constant
from rl_insight.data import DataEnum


@register_cluster_parser("gmm")
class GmmParser(BaseClusterParser):
    input_type = DataEnum.GMM_DATA
    
    def __init__(self, params) -> None:
        super().__init__(params)
        self.events_summary: Optional[pd.DataFrame] = None
        rank_list = params.get(Constant.RANK_LIST, "all")
        self._rank_list = (
            rank_list
            if rank_list == "all"
            else [int(rank) for rank in rank_list.split(",") if rank.isdigit()]
        )
        # Get step and role filters if provided
        self._step = params.get('step', None)
        self._role = params.get('role', None)
    
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
            text = str(file_path)
            
            # Extract rank
            m_rank = re.search(r"/rank(\d+)/", text)
            if not m_rank:
                continue
            rank_id = int(m_rank.group(1))
            
            # Extract step
            m_step = re.search(r"/step_(\d+)/", text)
            if not m_step:
                continue
            step = int(m_step.group(1))
            
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
            
            # Check if step matches the specified step
            if self._step is not None and step != self._step:
                continue
            
            # Check if role matches the specified role
            if self._role is not None and stage != self._role:
                continue
            
            data_map: DataMap = {
                "rank_id": rank_id,
                "role": stage,
                "step": step,
                "profiler_data_path": str(file_path)
            }
            data_maps.append(data_map)
        
        logger.info(f"Allocated {len(data_maps)} data maps for GMM parsing")
        return data_maps
    
    def _load_group_list(self, file_path: str) -> np.ndarray:
        """Load a group_list.pt file into a numpy array."""
        try:
            obj = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(file_path, map_location="cpu")
        if torch.is_tensor(obj):
            arr = obj.detach().float().cpu().numpy().ravel()
        elif isinstance(obj, np.ndarray):
            arr = obj.astype(np.float64).ravel()
        else:
            raise ValueError(f"Unexpected object in {file_path}: {type(obj)}")
        return arr
    
    def parse_analysis_data(self, profiler_data_path: str, rank_id: int, role: str, step: int = 0) -> List[EventRow]:
        """Parse GMM profiling data for a specific rank and return event information."""
        events = []
        try:
            # Load group_list data
            group_list = self._load_group_list(profiler_data_path)
            logger.info(f"Loaded group_list with {len(group_list)} experts from {profiler_data_path}")
            
            # Extract op index (stage) from file name
            file_name = Path(profiler_data_path).name
            m_op = re.search(r"npu_grouped_matmul\.(\d+)\.forward\.kwargs\.group_list\.pt$", file_name)
            op_index = int(m_op.group(1)) if m_op else 0
            stage_idx = op_index  # Use op_index as stage index
            
            # Create events for each expert
            for expert_idx, load in enumerate(group_list):
                # Generate a timestamp based on op index and expert index
                # This is a placeholder since GMM data doesn't have actual timestamps
                timestamp = op_index * 1000 + expert_idx
                
                # Create a dictionary directly (EventRow is a TypedDict)
                event = {
                    "name": "gmm_expert_load",
                    "role": role,
                    "domain": "gmm",
                    "start_time_ms": timestamp,
                    "end_time_ms": timestamp + 1,  # Placeholder duration
                    "duration_ms": 1,  # Placeholder duration
                    "rank_id": rank_id,
                    "tid": 0,  # Placeholder thread ID
                    "step": step,
                    "stage": stage_idx,
                    "expert_index": expert_idx,
                    "load": load
                }
                events.append(event)
            logger.info(f"Created {len(events)} events for {profiler_data_path}")
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
                # EventRow is a TypedDict, which is already a dict
                for event in result:
                    if isinstance(event, dict):
                        reduce_results.append(event)
                    elif hasattr(event, "__dict__"):
                        reduce_results.append(event.__dict__)
                    else:
                        reduce_results.append(event)
            else:
                raise TypeError(
                    f"parse_analysis_data must return list[EventRow] or None, got {type(result)}"
                )

        if not reduce_results:
            logger.warning("No valid data collected from any rank")
            return

        # Sort by step, then role, then rank_id, then stage, then expert_index, then load
        reduce_results.sort(key=lambda x: (x.get("step", 0), x.get("role", ""), x.get("rank_id", 0), x.get("stage", 0), x.get("expert_index", 0), x.get("load", 0)))
        logger.info(f"Sorted {len(reduce_results)} events by step, role, rank_id, stage, expert_index, load")
        self.events_summary = pd.DataFrame(reduce_results)
        logger.info(f"Created DataFrame with {len(self.events_summary)} rows")
        if not self.events_summary.empty:
            logger.info(f"DataFrame columns: {list(self.events_summary.columns)}")
            logger.info(f"Sample data: {self.events_summary.head()}")

    def get_data(self) -> pd.DataFrame:
        """Return the parsed DataFrame"""
        ## debug print pd.DataFrame to excel
        return self.events_summary
    
    def _mapper_func(self, data_map: DataMap) -> list[EventRow]:
        """Collect GMM data from a single rank"""
        profiler_data_path = data_map.get("profiler_data_path", "")
        rank_id = data_map.get("rank_id", -1)
        role = data_map.get("role", "")
        step = data_map.get("step", 0)

        if not profiler_data_path:
            logger.warning(f"Rank {rank_id}: profiler_data_path not found")
            return []

        return self.parse_analysis_data(profiler_data_path, rank_id, role, step)