"""Cross-cutting HTTP and MCP middleware.

The HTTP layer rejects abusive clients before MCP parsing or upstream work.
The MCP layer keeps a second, process-wide ceiling for shared DART/KRX quotas.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from ipaddress import ip_address
from math import ceil
from time import monotonic
from typing import Callable

from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

def _positive_override(value: int | None, default: int, label: str) -> int:
    resolved = default if value is None else value
    if resolved <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return resolved


@dataclass
class _ClientState:
    requests: deque[float] = field(default_factory=deque)
    violations: deque[float] = field(default_factory=deque)


class AbuseProtectionMiddleware:
    """Bounded, process-wide abuse protection for the public HTTP transport.

    Railway requests are recognized by their injected request ID, then the
    edge-provided ``X-Real-IP`` is used automatically. No deployment variables
    or operator-managed denylist are required.
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
        trust_x_real_ip: bool | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.app = app
        self.requests_per_window = _positive_override(
            requests_per_window, 30, "requests_per_window"
        )
        self.health_requests_per_window = _positive_override(
            health_requests_per_window,
            120,
            "health_requests_per_window",
        )
        self.window_seconds = _positive_override(
            window_seconds, 60, "window_seconds"
        )
        self.strikes_to_block = _positive_override(
            strikes_to_block, 3, "strikes_to_block"
        )
        self.strike_window_seconds = _positive_override(
            strike_window_seconds, 600, "strike_window_seconds"
        )
        self.block_seconds = _positive_override(
            block_seconds, 3600, "block_seconds"
        )
        self.max_clients = _positive_override(
            max_clients, 10_000, "max_clients"
        )
        self.trust_x_real_ip = trust_x_real_ip
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

        blocked_until = self._blocked_until.get(client_id)
        if blocked_until is not None:
            if blocked_until > now:
                self._blocked_until.move_to_end(client_id)
                await self._send_rejection(
                    send,
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
                "rate_limit_exceeded",
                retry_after,
                request_limit,
            )
            return

        state.requests.append(now)
        await self.app(scope, receive, send)

    def _client_id(self, scope: Scope) -> str:
        headers = scope.get("headers", [])
        trust_x_real_ip = self.trust_x_real_ip
        if trust_x_real_ip is None:
            trust_x_real_ip = any(
                key.lower() == b"x-railway-request-id" for key, _ in headers
            )

        if trust_x_real_ip:
            real_ip_values = [
                value
                for key, value in headers
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
        error: str,
        retry_after: int,
        request_limit: int,
    ) -> None:
        body = json.dumps(
            {"error": error},
            separators=(",", ":"),
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
            (b"retry-after", str(retry_after).encode("ascii")),
            (b"x-ratelimit-limit", str(request_limit).encode("ascii")),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": 429,
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
