#!/usr/bin/env python3

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

"""Shim → ``generate_data.py … --tempo`` (map to Tempo only; no Rebuild).

Preferred::

    python rl_insight/experimental/generate_data.py /tmp/trajs --samples 8 --seed 42 --tempo

Then in Grafana open ``agent_loop_trajectory`` and click **Rebuild Agent Loop**.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    argv = list(sys.argv[1:])
    if not argv or argv[0].startswith("-"):
        argv = ["/tmp/rl-insight-agent-loop-demo", *argv]
    if "--tempo" not in argv:
        argv.append("--tempo")

    print(
        "Forwarding to generate_data.py --tempo (Tempo map only; Rebuild in Grafana).\n"
        f"  → python rl_insight/experimental/generate_data.py {' '.join(argv)}\n"
    )
    sys.argv = ["generate_data.py", *argv]
    from rl_insight.experimental.generate_data import main as generate_main

    generate_main()


if __name__ == "__main__":
    main()
