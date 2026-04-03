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

import pytest
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch

from rl_insight.data import DataEnum
from rl_insight.visualizer import RLTimelinePNGVisualizer


@pytest.fixture
def valid_test_data():
    """Valid standard input DataFrame"""
    return pd.DataFrame(
        {
            "role": ["worker", "worker", "ps", "ps"],
            "name": ["train", "eval", "save", "load"],
            "rank_id": [0, 0, 1, 1],
            "start_time_ms": [100, 200, 150, 300],
            "end_time_ms": [180, 250, 220, 400],
        }
    )


@pytest.fixture
def empty_data():
    """Empty DataFrame for edge case testing"""
    return pd.DataFrame()


@pytest.fixture
def missing_column_data():
    """Data missing required columns for exception testing"""
    return pd.DataFrame(
        {
            "role": ["worker"],
            "name": ["train"],
        }
    )


@pytest.fixture
def short_event_data():
    """Data with short duration events for merge testing"""
    return pd.DataFrame(
        {
            "Role": ["worker", "worker"],
            "Name": ["train", "train"],
            "Rank ID": [0, 0],
            "Start": [100, 110],
            "Finish": [105, 115],
            "Duration": [5, 5],
            "Start_rel": [0, 10],
            "End_rel": [5, 15],
        }
    )


@pytest.fixture
def oversized_data():
    """Large dataset exceeding max points for downsampling test"""
    data = []
    for i in range(4000):
        data.append(
            {
                "role": "worker",
                "name": f"task_{i % 5}",
                "rank_id": i % 3,
                "start_time_ms": i * 10,
                "end_time_ms": i * 10 + 20,
            }
        )
    return pd.DataFrame(data)


class TestRLTimelinePNGVisualizer:
    @pytest.fixture
    def visualizer(self):
        """Initialize visualizer instance"""
        config = {"output_path": "test_output"}
        return RLTimelinePNGVisualizer(config)

    def test_init(self, visualizer):
        """Test constructor with given config"""
        assert visualizer.output_path == "test_output"
        assert visualizer.width == 2000
        assert visualizer.scale == 2
        assert visualizer.input_type == DataEnum.SUMMARY_EVENT

    def test_init_default_config(self):
        """Test constructor with empty default config"""
        vis = RLTimelinePNGVisualizer({})
        assert vis.output_path is None

    def test_load_and_preprocess_valid(self, visualizer, valid_test_data):
        """Test preprocessing with valid input data"""
        df, t0 = visualizer.load_and_preprocess(valid_test_data)

        required = [
            "Role",
            "Name",
            "Rank ID",
            "Start",
            "Finish",
            "Duration",
            "Start_rel",
            "End_rel",
        ]
        assert all(col in df.columns for col in required)
        assert len(df) > 0
        assert (df["Finish"] > df["Start"]).all()
        assert t0 == df["Start"].min()
        assert (df["Start_rel"] >= 0).all()

    def test_load_and_preprocess_none(self, visualizer):
        """Test error when input_data is None"""
        with pytest.raises(ValueError, match="input_data is None or empty!"):
            visualizer.load_and_preprocess(None)

    def test_load_and_preprocess_empty(self, visualizer, empty_data):
        """Test error when input DataFrame is empty"""
        with pytest.raises(ValueError, match="input_data is None or empty!"):
            visualizer.load_and_preprocess(empty_data)

    def test_load_and_preprocess_missing_columns(self, visualizer, missing_column_data):
        """Test error when required columns are missing"""
        with pytest.raises(ValueError, match="Required column missing"):
            visualizer.load_and_preprocess(missing_column_data)

    def test_merge_short_events_no_short(self, visualizer, valid_test_data):
        """Test no merging when no short events exist"""
        df, _ = visualizer.load_and_preprocess(valid_test_data)
        original_len = len(df)
        merged_df = visualizer.merge_short_events(df)
        assert len(merged_df) == original_len

    def test_downsample_not_needed(self, visualizer, valid_test_data):
        """Test no downsampling when data size is acceptable"""
        df, _ = visualizer.load_and_preprocess(valid_test_data)
        result_df = visualizer.downsample_if_needed(df)
        assert len(result_df) == len(df)

    def test_downsample_oversized(self, visualizer, oversized_data):
        """Test downsampling for large datasets exceeding max points"""
        df, _ = visualizer.load_and_preprocess(oversized_data)
        result_df = visualizer.downsample_if_needed(df, max_points=3000)
        assert len(result_df) <= 3000

    def test_build_y_mappings(self, visualizer, valid_test_data):
        """Test Y-axis label and position mapping generation"""
        df, _ = visualizer.load_and_preprocess(valid_test_data)
        y_map, y_step = visualizer.build_y_mappings(df)

        assert "positions" in y_map
        assert "bar_height" in y_map
        assert "labels" in y_map
        assert y_step == 50
        assert len(y_map["labels"]) > 0

    def test_build_traces(self, visualizer, valid_test_data):
        """Test Plotly bar trace generation"""
        df, _ = visualizer.load_and_preprocess(valid_test_data)
        y_map, _ = visualizer.build_y_mappings(df)
        traces = visualizer.build_traces(df, y_map)

        assert isinstance(traces, list)
        assert len(traces) > 0
        assert isinstance(traces[0], go.Bar)

    def test_assemble_static_figure(self, visualizer, valid_test_data):
        """Test layout and styling of the static timeline figure"""
        df, t0 = visualizer.load_and_preprocess(valid_test_data)
        y_map, y_step = visualizer.build_y_mappings(df)
        traces = visualizer.build_traces(df, y_map)
        fig = visualizer.assemble_static_figure(traces, df, t0, y_map, y_step)

        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 2000
        assert "Time (ms)" in fig.layout.xaxis.title.text

    @patch("rl_insight.visualizer.timeline_visualizer.to_image")
    @patch("rl_insight.visualizer.timeline_visualizer.os.makedirs")
    @patch("rl_insight.visualizer.timeline_visualizer.open", create=True)
    def test_save_png(self, mock_open, mock_makedirs, mock_to_image, visualizer):
        """Test PNG export with mocked file I/O"""
        fig = go.Figure()
        mock_to_image.return_value = b"fake_png_data"

        visualizer.save_png(fig, "test_output", "test.png")

        mock_makedirs.assert_called_once_with("test_output", exist_ok=True)
        mock_open.assert_called_once()
        mock_to_image.assert_called_once()

    @patch.object(RLTimelinePNGVisualizer, "generate_rl_timeline_png")
    def test_run(self, mock_generate, visualizer, valid_test_data):
        """Test the main run entry point"""
        visualizer.run(valid_test_data)
        mock_generate.assert_called_once_with(valid_test_data)

    @patch.object(RLTimelinePNGVisualizer, "save_png")
    def test_generate_rl_timeline_png_full(
        self, mock_save, visualizer, valid_test_data
    ):
        """Test full timeline PNG generation pipeline"""
        fig = visualizer.generate_rl_timeline_png(valid_test_data)
        assert isinstance(fig, go.Figure)
        mock_save.assert_called_once()
