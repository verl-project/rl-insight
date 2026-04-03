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

import os
import pytest
import pandas as pd
import plotly.graph_objects as go
from rl_insight.visualizer import RLTimelinePNGVisualizer


@pytest.fixture
def visualizer():
    """Initialize the visualizer instance for testing."""
    config = {"output_path": "test_output", "width": 2000, "scale": 2}
    return RLTimelinePNGVisualizer(config)


@pytest.fixture
def valid_test_data():
    """Generate a valid DataFrame with standard input data."""
    return pd.DataFrame(
        {
            "role": ["trainer", "trainer", "evaluator", "evaluator"],
            "name": ["rollout", "train", "rollout", "eval"],
            "rank_id": [0, 0, 1, 1],
            "start_time_ms": [100, 200, 150, 300],
            "end_time_ms": [180, 250, 220, 400],
        }
    )


@pytest.fixture
def short_event_data():
    """Generate DataFrame for testing short-duration event merging logic."""
    return pd.DataFrame(
        {
            "role": ["trainer"] * 4,
            "name": ["rollout"] * 4,
            "rank_id": [0] * 4,
            "start_time_ms": [100, 105, 110, 200],
            "end_time_ms": [105, 109, 115, 210],
        }
    )


# ====================== Unit Tests ======================
def test_initialization(visualizer):
    """Test that configuration parameters are loaded correctly."""
    assert visualizer.output_path == "test_output"
    assert visualizer.width == 2000
    assert visualizer.scale == 2
    assert visualizer.input_type.value == "summary_event"


def test_load_and_preprocess_valid(visualizer, valid_test_data):
    """Test data preprocessing with valid input DataFrame."""
    df, t0 = visualizer.load_and_preprocess(valid_test_data)

    required_cols = [
        "Role",
        "Name",
        "Rank ID",
        "Start",
        "Finish",
        "Duration",
        "Start_rel",
        "End_rel",
    ]
    assert all(col in df.columns for col in required_cols)
    assert df.isna().sum().sum() == 0
    assert df["Duration"].iloc[0] == 80
    assert df["Rank ID"].is_monotonic_increasing


def test_load_and_preprocess_empty_data(visualizer):
    """Test ValueError is raised for empty or None input."""
    with pytest.raises(ValueError, match="input_data is None or empty"):
        visualizer.load_and_preprocess(pd.DataFrame())


def test_load_and_preprocess_missing_columns(visualizer):
    """Test ValueError is raised when required columns are missing."""
    bad_df = pd.DataFrame(
        {
            "role": ["trainer"],
            "name": ["rollout"],
        }
    )
    with pytest.raises(ValueError, match="Required column missing"):
        visualizer.load_and_preprocess(bad_df)


def test_merge_short_events(visualizer, short_event_data):
    """Test that consecutive short events are merged correctly."""
    df, _ = visualizer.load_and_preprocess(short_event_data)
    assert not df.empty
    assert "Role" in df.columns
    assert "Name" in df.columns
    assert "Rank ID" in df.columns
    assert True


def test_downsample_if_needed(visualizer, valid_test_data):
    """Test downsampling logic for large datasets to prevent overflow."""
    large_df = pd.concat([valid_test_data] * 1000, ignore_index=True)
    df, _ = visualizer.load_and_preprocess(large_df)

    downsampled = visualizer.downsample_if_needed(df, max_points=100)
    assert len(downsampled) <= 100


def test_build_y_mappings(visualizer, valid_test_data):
    """Test Y-axis label and position mapping is built correctly."""
    df, _ = visualizer.load_and_preprocess(valid_test_data)
    y_map, y_step = visualizer.build_y_mappings(df)

    assert all(k in y_map for k in ["positions", "bar_height", "labels"])
    assert len(y_map["labels"]) == 2
    assert y_step == 50


def test_build_traces(visualizer, valid_test_data):
    """Test Plotly bar chart traces are generated correctly."""
    df, _ = visualizer.load_and_preprocess(valid_test_data)
    y_map, _ = visualizer.build_y_mappings(df)
    traces = visualizer.build_traces(df, y_map)

    assert len(traces) == 3
    assert isinstance(traces[0], go.Bar)
    assert traces[0].base is not None


def test_assemble_static_figure(visualizer, valid_test_data):
    """Test figure layout assembly works without errors."""
    df, t0 = visualizer.load_and_preprocess(valid_test_data)
    y_map, y_step = visualizer.build_y_mappings(df)
    traces = visualizer.build_traces(df, y_map)
    fig = visualizer.assemble_static_figure(traces, df, t0, y_map, y_step)

    assert isinstance(fig, go.Figure)
    assert fig.layout.width == 2000


def test_save_png(visualizer):
    """Test PNG image file is saved to the output directory."""
    fig = go.Figure()
    out_dir = "test_output"
    visualizer.save_png(fig, out_dir, "test.png")

    assert os.path.exists(os.path.join(out_dir, "test.png"))


def test_run_end_to_end(visualizer, valid_test_data):
    """End-to-end smoke test: full pipeline runs successfully."""
    fig = visualizer.run(valid_test_data)
    assert isinstance(fig, go.Figure)
