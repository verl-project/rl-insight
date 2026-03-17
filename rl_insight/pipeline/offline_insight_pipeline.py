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

from rl_insight.parser import get_cluster_parser_cls
from rl_insight.schema import Constant
from rl_insight.visualizer import RLTimelineVisualizer


class OfflineInsightPipeline:
    def __init__(self, config):
        self.input_path = config.input_path
        self.profiler_type = config.profiler_type
        self.output_path = config.output_path
        self.vis_type = config.vis_type
        self.rank_list = config.rank_list
        self.parser_config = self._prepare_parser_config()
        self.visualizer_config = self._prepare_visualizer_config()
        self.parser_cls = get_cluster_parser_cls(self.profiler_type)
        self.parser = self.parser_cls(self.parser_config)
        self.visualizer = RLTimelineVisualizer(self.visualizer_config)

    def _prepare_parser_config(self):
        return {
            Constant.INPUT_PATH: self.input_path,
            Constant.RANK_LIST: self.rank_list,
        }

    def _prepare_visualizer_config(self):
        return {
            "output_path": self.output_path,
            "vis_type": self.vis_type,
        }

    def run(self):
        data = self.parser.run()
        self.visualizer.run(data)
