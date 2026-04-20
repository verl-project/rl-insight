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

from rl_insight.data import DataChecker, DataEnum
from rl_insight.parser import get_cluster_parser_cls
from rl_insight.utils.schema import Constant
from rl_insight.visualizer import get_cluster_visualizer_cls


class OfflineInsightPipeline:
    def __init__(self, config):
        self.config = config

        # init data
        self.input_data_type = DataEnum(self.config.input_type)

        # parser related
        parser_config = self._prepare_parser_config()
        parser_cls = get_cluster_parser_cls(self.config.profiler_type)
        self.parser = parser_cls(parser_config)

        # visualizer related
        visualizer_config = self._prepare_visualizer_config()
        visualizer_cls = get_cluster_visualizer_cls(self.config.vis_type)
        self.visualizer = visualizer_cls(visualizer_config)

    def _prepare_parser_config(self):
        config = {
            Constant.RANK_LIST: self.config.rank_list,
        }
        
        # Add GMM-specific parameters only for GMM parser
        if self.config.profiler_type == "gmm":
            if hasattr(self.config, 'step') and self.config.step is not None:
                config['step'] = self.config.step
            if hasattr(self.config, 'role') and self.config.role is not None:
                config['role'] = self.config.role
        
        return config

    def _prepare_visualizer_config(self):
        config = {"output": self.config.output_path}
        
        # Add general parameters
        if hasattr(self.config, 'rank'):
            config['rank'] = self.config.rank
        if hasattr(self.config, 'dpi'):
            config['dpi'] = self.config.dpi
        if hasattr(self.config, 'cmap'):
            config['cmap'] = self.config.cmap
        
        # Add GMM-specific parameters only for GMM visualizer
        if self.config.vis_type == "gmm_heatmap":
            if hasattr(self.config, 'step') and self.config.step is not None:
                config['step'] = self.config.step
            if hasattr(self.config, 'role') and self.config.role is not None:
                config['role'] = self.config.role
        
        return config

    def run(self):
        if self.input_data_type != self.parser.input_type:
            raise ValueError(
                f"Input data type {self.input_data_type} does not match parser input type {self.parser.input_type}"
            )
        # validate input data
        DataChecker(self.input_data_type, self.config.input_path).run()

        output_data = self.parser.run(self.config.input_path)

        # validate output data
        DataChecker(self.visualizer.input_type, output_data).run()

        self.visualizer.run(output_data)