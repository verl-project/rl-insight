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

"""
This script provides functionality to validate all URLs found within the project's documentation files.
It recursively scans the '/docs' directory, extracts all HTTP/HTTPS links, and verifies their availability
by sending network requests. A pytest test case is included to ensure all documentation links are valid.

The main features include:
- Recursive file traversal of the documentation directory.
- URL extraction using regular expressions.
- HTTP HEAD request validation with timeout handling.
- Comprehensive pytest assertion for CI/CD integration.
"""

import os
import re
from pathlib import Path

import requests

# Configuration Constants
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DOCS_FOLDER = PROJECT_ROOT / "docs"
URL_PATTERN = re.compile(r"https?://[^\s)+\"'>]+")
TIMEOUT = 5  # request timeout
TEMPLATE_MARKERS = ("<", "${")
TEXT_SUFFIXES = {".md", ".rst", ".txt", ".py"}


def get_all_files_in_docs() -> list[Path]:
    """
    Recursively get all files in the docs directory.
    """
    files = []
    for root, _, filenames in os.walk(DOCS_FOLDER):
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix in TEXT_SUFFIXES:
                files.append(file_path)
    return files


def extract_urls_from_file(file_path: Path) -> list[str]:
    """
    Extract all URLs from a given file using regex pattern.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return URL_PATTERN.findall(content)
    except Exception as e:
        print(f"Could not read file {file_path}: {str(e)}")
        return []


def is_url_valid(url: str) -> bool:
    """
    Check if a URL is reachable (status code 200 ~ 399).
    """
    try:
        response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        if 200 <= response.status_code < 400:
            return True
        if response.status_code in {403, 405, 429}:
            with requests.get(
                url,
                timeout=TIMEOUT,
                allow_redirects=True,
                stream=True,
            ) as response:
                return 200 <= response.status_code < 400
        return False
    except requests.exceptions.RequestException:
        return False


def is_template_url(url: str) -> bool:
    """
    Skip documentation placeholders, such as http://<server-ip> or shell
    template URLs containing ${VERSION}. They are examples, not reachable URLs.
    """
    return any(marker in url for marker in TEMPLATE_MARKERS)


def test_docs_folder_all_urls_are_valid():
    """
    Test that all URLs inside docs directory files are valid and reachable.
    """
    all_files = get_all_files_in_docs()
    invalid_links = []

    for file in all_files:
        urls = extract_urls_from_file(file)
        for url in urls:
            if is_template_url(url):
                continue
            if not is_url_valid(url):
                invalid_links.append(f"{file} -> {url}")

    # Assert no invalid links exist
    assert len(invalid_links) == 0, (
        f"Found {len(invalid_links)} invalid URLs in docs:\n" + "\n".join(invalid_links)
    )
