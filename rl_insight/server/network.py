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

"""Network helpers for RL-Insight server endpoints."""

from __future__ import annotations

import ipaddress
import socket
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse


@lru_cache(maxsize=1)
def local_addresses() -> dict[str, str]:
    """Detect local IPv4/IPv6 addresses once, preferring IPv4 when both exist."""
    ipv4 = ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            if host and not ipaddress.ip_address(host).is_loopback:
                ipv4 = host
    except (OSError, ValueError):
        pass

    if not ipv4:
        try:
            host = socket.gethostbyname(socket.gethostname())
            if host and not ipaddress.ip_address(host).is_loopback:
                ipv4 = host
        except (OSError, ValueError):
            pass

    ipv6 = ""

    def usable_ipv6(host: str) -> bool:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return bool(
            address.version == 6
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_unspecified
        )

    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
            sock.connect(("2001:4860:4860::8888", 80))
            host = sock.getsockname()[0]
            if usable_ipv6(host):
                ipv6 = host
    except OSError:
        pass

    if not ipv6:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6):
                candidate = info[4][0]
                if not isinstance(candidate, str):
                    continue
                if usable_ipv6(candidate):
                    ipv6 = candidate
                    break
        except OSError:
            pass

    use_ipv6 = bool(ipv6 and not ipv4)
    return {
        "ipv4": ipv4,
        "ipv6": ipv6,
        "host": ipv4 or ipv6,
        "bind": "::" if use_ipv6 else "0.0.0.0",
        # TODO: support pure IPv6 loopback
        "loopback": "127.0.0.1",
        "family": "ipv6" if use_ipv6 else "ipv4",
    }


def is_ipv6_address(value: str) -> bool:
    try:
        ipaddress.IPv6Address(value)
        return True
    except ValueError:
        return False


def format_host_port(host: str, port: Any) -> str:
    """Return host:port, using [ipv6]:port when needed."""
    value = str(host).strip()
    raw = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    if is_ipv6_address(raw) and not value.startswith("["):
        value = f"[{value}]"
    return f"{value}:{int(port)}"


def service_url_from_server_url(
    server_url: str,
    port: Any,
    path: str = "",
) -> str:
    """Build a service URL using the host from the RL-Insight server URL."""
    if not port:
        return ""
    parsed = urlparse(str(server_url))
    host = parsed.hostname or ""
    if not host:
        return ""
    scheme = parsed.scheme or "http"
    normalized_path = path if not path or path.startswith("/") else f"/{path}"
    return f"{scheme}://{format_host_port(host, port)}{normalized_path}"
