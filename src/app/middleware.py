"""Cross-cutting HTTP and MCP middleware.

The HTTP layer rejects abusive clients before MCP parsing or upstream work.
The MCP layer keeps a second, process-wide ceiling for shared DART/KRX quotas.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network
from math import ceil
from time import monotonic
from typing import Callable, Iterable

from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

IpNetwork = IPv4Network | IPv6Network


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_override(value: int | None, name: str, default: int) -> int:
    resolved = _positive_int(name, default) if value is None else value
    if resolved <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return resolved


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _blocked_networks(value: str | Iterable[str]) -> tuple[IpNetwork, ...]:
    entries = value.split(",") if isinstance(value, str) else value
    networks: list[IpNetwork] = []
    for entry in entries:
        candidate = entry.strip()
        if candidate:
            networks.append(ip_network(candidate, strict=False))
    return tuple(networks)


@dataclass
class _ClientState:
    requests: deque[float] = field(default_factory=deque)
    violations: deque[float] = field(default_factory=deque)


class AbuseProtectionMiddleware:
    """Bounded, process-wide abuse protection for the public HTTP transport.

    Railway terminates public traffic at its edge and supplies ``X-Real-IP``.
    That header is trusted only on Railway (or with an explicit opt-in) so a
    direct client cannot rotate identities with a spoofed forwarding header.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_window: int | None = None,
        health_requests_per_window: int | None = None,
        window_seconds: int | None = None,
        strikes_to_block: int | None = None,
        strike_window_seconds: int | None = None,
        block_seconds: int | None = None,
        max_clients: int | None = None,
        blocked_networks: str | Iterable[str] | None = None,
        trust_x_real_ip: bool | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.app = app
        self.requests_per_window = _positive_override(
            requests_per_window, "ABUSE_RATE_LIMIT_REQUESTS", 30
        )
        self.health_requests_per_window = _positive_override(
            health_requests_per_window,
            "ABUSE_HEALTH_RATE_LIMIT_REQUESTS",
            120,
        )
        self.window_seconds = _positive_override(
            window_seconds, "ABUSE_RATE_LIMIT_WINDOW_SECONDS", 60
        )
        self.strikes_to_block = _positive_override(
            strikes_to_block, "ABUSE_STRIKES_TO_BLOCK", 3
        )
        self.strike_window_seconds = _positive_override(
            strike_window_seconds, "ABUSE_STRIKE_WINDOW_SECONDS", 600
        )
        self.block_seconds = _positive_override(
            block_seconds, "ABUSE_BLOCK_SECONDS", 3600
        )
        self.max_clients = _positive_override(
            max_clients, "ABUSE_MAX_CLIENTS", 10_000
        )
        configured_networks = (
            os.environ.get("ABUSE_BLOCKED_IPS", "")
            if blocked_networks is None
            else blocked_networks
        )
        self.blocked_networks = _blocked_networks(configured_networks)
        on_railway = bool(
            os.environ.get("RAILWAY_PROJECT_ID")
            or os.environ.get("RAILWAY_ENVIRONMENT_ID")
        )
        self.trust_x_real_ip = (
            _env_bool("ABUSE_TRUST_X_REAL_IP", on_railway)
            if trust_x_real_ip is None
            else trust_x_real_ip
        )
        self.clock = clock
        self._clients: OrderedDict[str, _ClientState] = OrderedDict()
        self._blocked_until: OrderedDict[str, float] = OrderedDict()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        now = self.clock()
        client_id = self._client_id(scope)
        request_limit = (
            self.health_requests_per_window
            if scope.get("path") == "/health"
            else self.requests_per_window
        )
        state_id = f"{'health' if scope.get('path') == '/health' else 'http'}:{client_id}"

        if self._is_manually_blocked(client_id):
            await self._send_rejection(send, 403, "forbidden")
            return

        blocked_until = self._blocked_until.get(client_id)
        if blocked_until is not None:
            if blocked_until > now:
                self._blocked_until.move_to_end(client_id)
                await self._send_rejection(
                    send,
                    429,
                    "temporarily_blocked",
                    ceil(blocked_until - now),
                    request_limit,
                )
                return
            self._blocked_until.pop(client_id, None)

        state = self._clients.get(state_id)
        if state is None:
            if len(self._clients) >= self.max_clients:
                self._clients.popitem(last=False)
            state = _ClientState()
            self._clients[state_id] = state
        else:
            self._clients.move_to_end(state_id)

        self._discard_before(state.requests, now - self.window_seconds)
        self._discard_before(state.violations, now - self.strike_window_seconds)

        if len(state.requests) >= request_limit:
            state.violations.append(now)
            retry_after = max(
                1,
                ceil(self.window_seconds - (now - state.requests[0])),
            )
            if len(state.violations) >= self.strikes_to_block:
                self._clients.pop(state_id, None)
                self._remember_block(client_id, now + self.block_seconds)
                retry_after = self.block_seconds
            await self._send_rejection(
                send,
                429,
                "rate_limit_exceeded",
                retry_after,
                request_limit,
            )
            return

        state.requests.append(now)
        await self.app(scope, receive, send)

    def _client_id(self, scope: Scope) -> str:
        if self.trust_x_real_ip:
            real_ip_values = [
                value
                for key, value in scope.get("headers", [])
                if key.lower() == b"x-real-ip"
            ]
            if len(real_ip_values) == 1:
                try:
                    return str(ip_address(real_ip_values[0].decode("ascii").strip()))
                except (UnicodeDecodeError, ValueError):
                    pass

        client = scope.get("client")
        if client:
            try:
                return str(ip_address(client[0]))
            except ValueError:
                return client[0]
        return "unknown"

    def _is_manually_blocked(self, client_id: str) -> bool:
        try:
            client_ip = ip_address(client_id)
        except ValueError:
            return False
        return any(client_ip in network for network in self.blocked_networks)

    def _remember_block(self, client_id: str, blocked_until: float) -> None:
        if len(self._blocked_until) >= self.max_clients:
            self._blocked_until.popitem(last=False)
        self._blocked_until[client_id] = blocked_until

    @staticmethod
    def _discard_before(timestamps: deque[float], cutoff: float) -> None:
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

    async def _send_rejection(
        self,
        send: Send,
        status: int,
        error: str,
        retry_after: int | None = None,
        request_limit: int | None = None,
    ) -> None:
        body = json.dumps(
            {"error": error},
            separators=(",", ":"),
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]
        if retry_after is not None:
            headers.append((b"retry-after", str(retry_after).encode("ascii")))
            headers.append(
                (
                    b"x-ratelimit-limit",
                    str(request_limit or self.requests_per_window).encode("ascii"),
                )
            )
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_rate_limiter() -> RateLimitingMiddleware:
    """Keep a global ceiling for shared upstream quotas after HTTP filtering."""
    return RateLimitingMiddleware(
        max_requests_per_second=float(os.environ.get("RATE_LIMIT_RPS", "5")),
        burst_capacity=int(os.environ.get("RATE_LIMIT_BURST", "15")),
        global_limit=True,
    )
