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
"""Check that no relative image or link references exist in docs, README, etc.

Relative references break on PyPI (README) and readthedocs (docs).
All image/link URLs must be absolute.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Patterns that indicate a relative URL in markdown or HTML
RELATIVE_PATTERNS = [
    # Markdown image: ![alt](./foo.png) or ![alt](../foo.png)
    (r"!\[.*?\]\(\.\.?/", "markdown image with relative path"),
    # HTML image: <img src="./foo.png"> or <img src="../foo.png">
    (r'<img[^>]+src=["\']\.\.?/', "HTML <img> with relative path"),
    # Markdown link: [text](./foo) or [text](../foo) — skip http/https
    # We allow relative links to other .md files (docs navigation), so only flag image-like extensions
    (
        r"\[.*?\]\(\.\.?/[^)]*\.(png|svg|jpg|jpeg|gif|webp)\)",
        "markdown link to image with relative path",
    ),
]

# Files to scan
GLOB_PATTERNS = [
    "**/*.md",
    "**/*.rst",
    "**/*.html",
]

# Directories to exclude
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "_build",
    "dist",
    "*.egg-info",
}


def main() -> int:
    errors: list[str] = []
    files_checked = 0

    for pattern in GLOB_PATTERNS:
        for filepath in REPO_ROOT.glob(pattern):
            # Skip excluded directories
            parts = set(filepath.parts)
            if parts & EXCLUDE_DIRS:
                continue

            # Read file
            try:
                content = filepath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            files_checked += 1
            lines = content.split("\n")

            for i, line in enumerate(lines, start=1):
                for pat, desc in RELATIVE_PATTERNS:
                    if re.search(pat, line):
                        rel_path = filepath.relative_to(REPO_ROOT)
                        errors.append(f"{rel_path}:{i}: {desc}: {line.strip()[:120]}")

    if errors:
        print(f"Found {len(errors)} relative image/link reference(s):\n")
        for err in errors:
            print(f"  {err}")
        print(f"\n{'=' * 60}")
        print(
            "All image/link URLs must be absolute so they work on PyPI and readthedocs."
        )
        print("Use full URLs like:")
        print(
            "  https://raw.githubusercontent.com/verl-project/rl-insight/main/assets/foo.png"
        )
        print("instead of relative paths like ./assets/foo.png or ../assets/foo.png")
        return 1

    print(
        f"OK: {files_checked} files checked, no relative image/link references found."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
