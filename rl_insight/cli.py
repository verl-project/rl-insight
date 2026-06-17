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
from pathlib import Path
from typing import Sequence

from .server.commands import ServerCommands


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``rl-insight``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
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

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_server_parser(subparsers)
    return parser


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    commands = ServerCommands()
    server = subparsers.add_parser(
        "server",
        help="Install and manage Prometheus, Tempo, and Grafana services.",
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
    install.set_defaults(func=commands.install)

    start = server_subparsers.add_parser(
        "start",
        help="Start Prometheus, Tempo, and Grafana.",
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
        help="Stop Prometheus, Tempo, and Grafana.",
    )
    _add_common_config_args(stop)
    stop.set_defaults(func=commands.stop)


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
