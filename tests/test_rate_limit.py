"""Baseline hardening: global rate limiting on the MCP layer."""

import httpx
import pytest
from fastmcp import Client, FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.middleware import AbuseProtectionMiddleware, build_rate_limiter
from app.server import mcp


async def test_server_has_rate_limiter_registered():
    assert any(type(m).__name__ == "RateLimitingMiddleware" for m in mcp.middleware)


async def test_http_app_has_abuse_protection_registered():
    from app.main import app

    assert any(
        middleware.cls is AbuseProtectionMiddleware
        for middleware in app.user_middleware
    )


async def test_burst_beyond_capacity_is_rejected(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("RATE_LIMIT_BURST", "3")

    toy = FastMCP("toy")
    toy.add_middleware(build_rate_limiter())

    @toy.tool
    def noop() -> str:
        return "ok"

    async with Client(toy) as client:
        with pytest.raises(Exception, match="[Rr]ate limit"):
            for _ in range(10):
                await client.call_tool("noop", {})


def _protected_app(**options):
    calls = {"count": 0}

    async def endpoint(request: Request) -> JSONResponse:
        calls["count"] += 1
        return JSONResponse({"ok": True})

    inner = Starlette(
        routes=[
            Route("/mcp", endpoint, methods=["POST"]),
            Route("/health", endpoint, methods=["GET"]),
        ]
    )
    return AbuseProtectionMiddleware(inner, **options), calls


async def test_repeat_offender_is_blocked_before_handler_execution():
    app, calls = _protected_app(
        requests_per_window=1,
        health_requests_per_window=10,
        window_seconds=60,
        strikes_to_block=2,
        strike_window_seconds=600,
        block_seconds=120,
        trust_x_real_ip=False,
    )
    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 1234))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        assert (await client.post("/mcp")).status_code == 200
        assert (await client.post("/mcp")).status_code == 429
        blocked = await client.post("/mcp")
        still_blocked = await client.post("/mcp")

    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "120"
    assert still_blocked.json() == {"error": "temporarily_blocked"}
    assert calls["count"] == 1


async def test_railway_real_ip_isolates_clients():
    app, calls = _protected_app(
        requests_per_window=1,
        health_requests_per_window=10,
        window_seconds=60,
        strikes_to_block=1,
        strike_window_seconds=600,
        block_seconds=120,
        trust_x_real_ip=True,
    )
    transport = httpx.ASGITransport(app=app, client=("10.0.0.1", 1234))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/mcp", headers={"x-real-ip": "203.0.113.10"})
        blocked = await client.post("/mcp", headers={"x-real-ip": "203.0.113.10"})
        other = await client.post("/mcp", headers={"x-real-ip": "203.0.113.11"})

    assert [first.status_code, blocked.status_code, other.status_code] == [
        200,
        429,
        200,
    ]
    assert calls["count"] == 2


async def test_manual_cidr_block_applies_before_every_http_route():
    app, calls = _protected_app(
        requests_per_window=10,
        health_requests_per_window=10,
        window_seconds=60,
        strikes_to_block=3,
        strike_window_seconds=600,
        block_seconds=120,
        blocked_networks="203.0.113.0/24",
        trust_x_real_ip=False,
    )
    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 1234))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        denied = await client.post("/mcp")
        health = await client.get("/health")

    assert denied.status_code == 403
    assert denied.json() == {"error": "forbidden"}
    assert health.status_code == 403
    assert calls["count"] == 0


async def test_untrusted_real_ip_header_cannot_rotate_identity():
    app, calls = _protected_app(
        requests_per_window=1,
        health_requests_per_window=10,
        window_seconds=60,
        strikes_to_block=2,
        strike_window_seconds=600,
        block_seconds=120,
        trust_x_real_ip=False,
    )
    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 1234))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/mcp", headers={"x-real-ip": "198.51.100.1"})
        rotated = await client.post("/mcp", headers={"x-real-ip": "198.51.100.2"})

    assert first.status_code == 200
    assert rotated.status_code == 429
    assert calls["count"] == 1


async def test_health_route_has_a_separate_limit_instead_of_a_bypass():
    app, calls = _protected_app(
        requests_per_window=10,
        health_requests_per_window=1,
        window_seconds=60,
        strikes_to_block=2,
        strike_window_seconds=600,
        block_seconds=120,
        trust_x_real_ip=False,
    )
    transport = httpx.ASGITransport(app=app, client=("203.0.113.10", 1234))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        limited = await client.get("/health")
        mcp = await client.post("/mcp")

    assert [health.status_code, limited.status_code, mcp.status_code] == [
        200,
        429,
        200,
    ]
    assert calls["count"] == 2
