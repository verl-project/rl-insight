# Copyright (c) 2026 verl-project authors.
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

"""Command-line entry point for RL-Insight."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .server.commands import ServerCommands
from .utils.constants import MonitorPaths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``rl-insight``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    logger = logging.getLogger(__name__)
    logger.info("rl-insight v%s", __version__)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    """Construct the root argument parser."""
    parser = argparse.ArgumentParser(prog="rl-insight")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rl-insight v{__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_server_parser(subparsers)
    _add_data_parser(subparsers)
    return parser


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    commands = ServerCommands()
    server = subparsers.add_parser(
        "server",
        help="Install and manage the RL-Insight server stack.",
    )
    server_subparsers = server.add_subparsers(dest="server_command", required=True)

    install = server_subparsers.add_parser(
        "install",
        help="Download missing Prometheus, Tempo, and Grafana binaries.",
    )
    _add_common_config_args(install)
    install.add_argument(
        "--install-dir",
        type=Path,
        default=None,
        help="Managed install directory used by this installer; default is ~/.rl-insight/services.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Download and reinstall enabled services even when binaries exist.",
    )
    install.add_argument(
        "--local-archive",
        type=Path,
        default=None,
        help="Directory with pre-downloaded .tar.gz archives; skip download when archive matches.",
    )

    install.set_defaults(func=commands.install)

    start = server_subparsers.add_parser(
        "start",
        help="Start the RL-Insight server stack.",
    )
    _add_common_config_args(start)
    mode_group = start.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--detach",
        action="store_true",
        help="Start in background and return immediately.",
    )
    mode_group.add_argument(
        "--attach-logs",
        action="store_true",
        help="Run in foreground and stream service logs.",
    )
    start.set_defaults(func=commands.start)

    stop = server_subparsers.add_parser(
        "stop",
        help="Stop the RL-Insight server stack.",
    )
    _add_common_config_args(stop)
    stop.set_defaults(func=commands.stop)

    targets = server_subparsers.add_parser(
        "targets",
        help="Manage Prometheus scrape targets.",
    )
    target_subparsers = targets.add_subparsers(dest="targets_command", required=True)
    add_targets = target_subparsers.add_parser(
        "add",
        help="Add scrape targets from a YAML file.",
    )
    add_targets.add_argument(
        "target_file",
        type=Path,
        help="YAML file containing Prometheus jobs and targets.",
    )
    _add_common_config_args(add_targets)
    add_targets.set_defaults(func=commands.add_targets)


def _add_data_parser(subparsers: argparse._SubParsersAction) -> None:
    data = subparsers.add_parser(
        "data",
        help="Inspect persisted RL-Insight data.",
    )
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    inspect = data_subparsers.add_parser(
        "inspect",
        help="List projects, experiments, and their time ranges.",
    )
    inspect.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="RL-Insight data directory; defaults to ~/.rl-insight/data.",
    )
    inspect.add_argument(
        "--promtool-bin",
        type=Path,
        default=None,
        help="Path to promtool; auto-detected when omitted.",
    )
    inspect.set_defaults(func=_handle_data_inspect)


def _handle_data_inspect(args: argparse.Namespace) -> int:
    """Inspect persisted data and print the result."""
    from .data_inspection import format_summaries, inspect_data_directory

    data_dir = _resolve_data_dir(args)
    try:
        summaries = inspect_data_directory(data_dir, promtool_bin=args.promtool_bin)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Data inspection failed: {exc}", file=sys.stderr)
        return 1

    if not summaries:
        print("Data not found.")
        return 0

    print(format_summaries(summaries))
    return 0


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.log_dir is None:
        return (MonitorPaths.STATE_ROOT / "data").resolve()
    return args.log_dir.expanduser().resolve()


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--config`` shared by subcommands that read stack YAML."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Server YAML; default is bundled rl_insight/config/config.yaml.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
