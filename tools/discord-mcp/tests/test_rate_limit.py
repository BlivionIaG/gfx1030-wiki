import asyncio

from discord_mcp.rate_limit import (
    RateLimiter,
    is_global_limit,
    major_param,
    parse_retry_after,
    route_key,
)


def test_major_param_channel_and_guild() -> None:
    assert major_param("/channels/123/messages") == "channels:123"
    assert major_param("/guilds/456/channels") == "guilds:456"
    assert major_param("/users/@me/guilds") is None


def test_route_key_splits_channels() -> None:
    a = route_key("GET", "/channels/1/messages")
    b = route_key("GET", "/channels/2/messages")
    assert a != b
    assert "{id}" in a


def test_parse_retry_after_prefers_json_float() -> None:
    wait = parse_retry_after(
        {"Retry-After": "65"},
        {"retry_after": 64.57, "global": False},
    )
    assert wait == 64.57


def test_parse_retry_after_not_capped_at_ten() -> None:
    assert parse_retry_after({"Retry-After": "65"}, None) == 65.0


def test_is_global_limit() -> None:
    assert is_global_limit({"X-RateLimit-Global": "true"}, {})
    assert is_global_limit({"X-RateLimit-Scope": "global"}, {})
    assert is_global_limit({}, {"global": True})
    assert not is_global_limit({"X-RateLimit-Scope": "shared"}, {"global": False})


def test_observe_updates_remaining() -> None:
    limiter = RateLimiter()
    limiter.observe(
        "GET",
        "/channels/1/messages",
        status=200,
        headers={
            "X-RateLimit-Limit": "5",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset-After": "1.5",
            "X-RateLimit-Bucket": "abcd",
        },
    )
    bucket = limiter._bucket_for("GET", "/channels/1/messages")
    assert bucket.remaining == 0
    assert bucket.hash == "abcd"


def test_global_429_blocks_all_routes() -> None:
    clock = {"t": 100.0}

    def now() -> float:
        return clock["t"]

    limiter = RateLimiter(clock=now)
    wait = limiter.observe(
        "GET",
        "/users/@me",
        status=429,
        headers={"X-RateLimit-Global": "true", "Retry-After": "2"},
        body={"retry_after": 2.0, "global": True},
    )
    assert wait >= 2.0
    assert limiter._global_until > clock["t"]


def test_acquire_sleeps_when_bucket_empty(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    limiter = RateLimiter(global_rps=1000)
    limiter.observe(
        "GET",
        "/channels/9/messages",
        status=200,
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset-After": "0.4",
            "X-RateLimit-Bucket": "msg",
        },
    )

    asyncio.run(limiter.acquire("GET", "/channels/9/messages"))
    assert slept
    assert slept[0] >= 0.4
