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

import dataclasses
import inspect
import sys
from pathlib import Path
from typing import Any, List, Optional, get_type_hints

from omegaconf import DictConfig, MISSING, OmegaConf

from .config import AppConfig


class _HelpRenderer:
    """Generate CLI help text from structured config dataclasses."""

    @staticmethod
    def render(supported_presets: set[str]) -> str:
        sections = [
            _HelpRenderer._header(supported_presets),
            _HelpRenderer._config_keys(),
            _HelpRenderer._examples(),
        ]
        return "\n".join(sections)

    @staticmethod
    def _header(supported_presets: set[str]) -> str:
        return (
            "Usage: python -m recipe.main [key=value] ... [config_path=PATH] [preset=NAME]\n"
            "\n"
            "RL Insight - Cluster scheduling visualization for RL training\n"
            "\n"
            "Special keys:\n"
            f"  {'config_path':<30s} Path to YAML config file\n"
            f"  {'preset':<30s} Preset name: {', '.join(sorted(supported_presets))}\n"
            "\n"
            "Configuration keys (key=value):\n"
        )

    @staticmethod
    def _config_keys() -> str:
        lines: list[str] = []
        _HelpRenderer._format_group(lines, AppConfig, prefix="")
        return "\n".join(lines)

    @staticmethod
    def _examples() -> str:
        return (
            "\n"
            "Examples:\n"
            "  python -m recipe.main input.path=./data/recipe/mstx_data/mstx_profile\n"
            "  python -m recipe.main preset=heatmap input.path=./data/recipe/gmm_data\n"
            "  python -m recipe.main config_path=my_config.yaml heatmap.visualizer.dpi=300\n"
            "  python -m recipe.main preset=timeline timeline.visualizer.type=png\n"
            "  python -m recipe.main preset=timeline timeline.parser.type=torch\n"
            "  python -m recipe.main preset=memory input.path=./data/recipe/memory_data\n"
            "  python -m recipe.main memory.parser.type=memory memory.visualizer.type=memory_html input.path=./data/recipe/memory_data\n"
        )

    @staticmethod
    def _format_group(lines: list[str], cls_type: Any, prefix: str) -> None:
        group_name = cls_type.__doc__.strip() if cls_type.__doc__ else cls_type.__name__
        lines.append(f"  [{group_name}]")

        hints = get_type_hints(cls_type)
        source_lines = inspect.getsource(cls_type).split("\n")

        for f in dataclasses.fields(cls_type):
            if dataclasses.is_dataclass(f.type):
                continue

            full_key = f"{prefix}{f.name}" if prefix else f.name
            type_name = _HelpRenderer._type_name(hints.get(f.name, f.type))
            default_str = _HelpRenderer._default_str(f.default)
            comment = _HelpRenderer._field_comment(f.name, source_lines)

            lines.append(
                f"    {full_key + ' (' + type_name + ')':<38s} "
                f"{default_str:<25s}{comment}"
            )

        for f in dataclasses.fields(cls_type):
            if dataclasses.is_dataclass(f.type):
                sub_prefix = f"{prefix}{f.name}." if prefix else f"{f.name}."
                lines.append("")
                _HelpRenderer._format_group(lines, f.type, sub_prefix)

    @staticmethod
    def _type_name(hint) -> str:
        return hint.__name__ if hasattr(hint, "__name__") else str(hint)

    @staticmethod
    def _default_str(default) -> str:
        if default is dataclasses.MISSING or default is MISSING:
            return "REQUIRED"
        if default is None:
            return "null"
        return repr(default)

    @staticmethod
    def _field_comment(field_name: str, source_lines: list[str]) -> str:
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith(f"{field_name}:") and "#" in stripped:
                return "  " + stripped.split("#", 1)[1].strip()
        return ""


class ConfigLoader:
    PRESETS_DIR = Path(__file__).parent
    SUPPORTED_PRESETS = {"timeline", "heatmap", "memory"}

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        preset: Optional[str] = None,
        cli_args: Optional[List[str]] = None,
    ) -> DictConfig:
        cfg = OmegaConf.structured(AppConfig)

        if preset:
            cfg = cls._merge_preset(cfg, preset)

        if config_path:
            cfg = cls._merge_yaml(cfg, config_path)

        if cli_args:
            cli_cfg = OmegaConf.from_cli(cli_args)
            cfg = OmegaConf.merge(cfg, cli_cfg)

        OmegaConf.resolve(cfg)
        return cfg

    @classmethod
    def load_from_cli(cls, argv: Optional[List[str]] = None) -> DictConfig:
        if argv is None:
            argv = sys.argv[1:]

        if "--help" in argv or "-h" in argv:
            print(_HelpRenderer.render(cls.SUPPORTED_PRESETS))
            sys.exit(0)

        config_path, preset, remaining = cls._parse_special_args(argv)

        if preset is None and config_path is None:
            preset = cls._infer_preset_from_args(remaining) or "timeline"

        return cls.load(
            config_path=config_path,
            preset=preset,
            cli_args=remaining or None,
        )

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> DictConfig:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        return OmegaConf.load(path)

    @classmethod
    def save_to_yaml(cls, cfg: DictConfig, yaml_path: str) -> None:
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, path)

    @classmethod
    def get_default_config(cls) -> DictConfig:
        return OmegaConf.structured(AppConfig)

    @classmethod
    def _merge_preset(cls, cfg: DictConfig, preset: str) -> DictConfig:
        if preset not in cls.SUPPORTED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Supported presets: {', '.join(sorted(cls.SUPPORTED_PRESETS))}"
            )
        preset_path = cls.PRESETS_DIR / f"{preset}.yaml"
        if preset_path.exists():
            cfg = OmegaConf.merge(cfg, OmegaConf.load(preset_path))
        return cfg

    @classmethod
    def _merge_yaml(cls, cfg: DictConfig, config_path: str) -> DictConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return OmegaConf.merge(cfg, OmegaConf.load(path))

    @staticmethod
    def _parse_special_args(
        argv: list[str],
    ) -> tuple[Optional[str], Optional[str], list[str]]:
        config_path: Optional[str] = None
        preset: Optional[str] = None
        remaining: list[str] = []

        for arg in argv:
            if arg.startswith("config_path="):
                config_path = arg.split("=", 1)[1]
            elif arg.startswith("preset="):
                preset = arg.split("=", 1)[1]
            else:
                remaining.append(arg)

        return config_path, preset, remaining

    @staticmethod
    def _infer_preset_from_args(args: list[str]) -> Optional[str]:
        for arg in args:
            if arg.startswith("heatmap."):
                return "heatmap"
            if arg.startswith("memory."):
                return "memory"
        return None
