"""Discord REST rate-limit tracking.

Discord's published rules (https://docs.discord.com/developers/topics/rate-limits):

- Do **not** hard-code per-route quotas; parse ``X-RateLimit-*`` headers.
- Per-route buckets are identified by ``X-RateLimit-Bucket`` and calculated
  independently per top-level resource (channel_id / guild_id).
- Global cap is 50 authenticated requests per second.
- On HTTP 429, wait ``retry_after`` (JSON) or ``Retry-After`` (header).
- 401 / 403 / 429 count toward Cloudflare's 10_000 invalid requests / 10 min
  (except 429 with ``X-RateLimit-Scope: shared``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

log = logging.getLogger("discord_mcp.rate_limit")

# Discord documents 50 req/s global. Stay a little under so header lag does not
# push us over. This is a ceiling, not a substitute for header parsing.
GLOBAL_RPS = 45
# MCP hosts time out; wait Discord's retry_after fully up to this, then fail.
MAX_WAIT_SECONDS = 120.0
SAFETY_PAD = 0.05

_MAJOR_RE = re.compile(
    r"^/(?P<kind>channels|guilds|webhooks)/(?P<id>\d+)(?:/|$)",
    re.IGNORECASE,
)


def major_param(path: str) -> str | None:
    """Top-level resource Discord uses to split per-route buckets."""
    match = _MAJOR_RE.match(path)
    if not match:
        return None
    return f"{match.group('kind')}:{match.group('id')}"


def route_key(method: str, path: str) -> str:
    major = major_param(path) or "_"
    template = _MAJOR_RE.sub(
        lambda m: f"/{m.group('kind')}/{{id}}",
        path.split("?", 1)[0],
        count=1,
    )
    return f"{method.upper()}:{major}:{template}"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is not None:
            return str(value)
        lower = name.lower()
        for key, val in headers.items():
            if str(key).lower() == lower:
                return str(val)
    return None


def parse_retry_after(headers: Mapping[str, str], body: Any) -> float:
    """Seconds Discord wants us to wait after a 429.

    Prefer JSON ``retry_after`` (sub-second precision), then ``Retry-After``.
    """
    if isinstance(body, dict) and body.get("retry_after") is not None:
        try:
            return max(0.0, float(body["retry_after"]))
        except (TypeError, ValueError):
            pass
    raw = _header(headers, "Retry-After") or _header(headers, "X-RateLimit-Reset-After")
    if raw is None:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def is_global_limit(headers: Mapping[str, str], body: Any) -> bool:
    if _header(headers, "X-RateLimit-Global"):
        return True
    scope = (_header(headers, "X-RateLimit-Scope") or "").lower()
    if scope == "global":
        return True
    return bool(isinstance(body, dict) and body.get("global") is True)


@dataclass
class Bucket:
    remaining: int | None = None
    reset_after: float = 0.0
    reset_monotonic: float = 0.0
    limit: int | None = None
    hash: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimiter:
    def __init__(self, *, global_rps: float = GLOBAL_RPS, clock=time.monotonic) -> None:
        self.global_rps = max(1.0, global_rps)
        self._clock = clock
        self._buckets: dict[str, Bucket] = {}
        self._route_bucket: dict[str, str] = {}
        self._hits: deque[float] = deque()
        self._global_until = 0.0
        self._global_lock = asyncio.Lock()

    def _bucket_for(self, method: str, path: str) -> Bucket:
        key = route_key(method, path)
        mapped = self._route_bucket.get(key, key)
        bucket = self._buckets.get(mapped)
        if bucket is None:
            bucket = Bucket()
            self._buckets[mapped] = bucket
        return bucket

    async def acquire(self, method: str, path: str) -> None:
        await self._wait_global()
        bucket = self._bucket_for(method, path)
        async with bucket.lock:
            await self._wait_bucket(bucket)

    async def _wait_global(self) -> None:
        async with self._global_lock:
            while True:
                now = self._clock()
                if now < self._global_until:
                    wait = min(self._global_until - now + SAFETY_PAD, MAX_WAIT_SECONDS)
                    log.info("Discord global rate limit: sleeping %.3fs", wait)
                    await asyncio.sleep(wait)
                    continue
                window_start = now - 1.0
                while self._hits and self._hits[0] < window_start:
                    self._hits.popleft()
                if len(self._hits) < int(self.global_rps):
                    self._hits.append(now)
                    return
                wait = 1.0 - (now - self._hits[0]) + SAFETY_PAD
                await asyncio.sleep(max(wait, 0.01))

    async def _wait_bucket(self, bucket: Bucket) -> None:
        if bucket.remaining is None:
            return
        if bucket.remaining > 0:
            bucket.remaining -= 1
            return
        wait = bucket.reset_monotonic - self._clock()
        if wait > 0:
            wait = min(wait + SAFETY_PAD, MAX_WAIT_SECONDS)
            log.info("Discord bucket %s exhausted: sleeping %.3fs", bucket.hash, wait)
            await asyncio.sleep(wait)
        bucket.remaining = None

    def observe(
        self,
        method: str,
        path: str,
        *,
        status: int,
        headers: Mapping[str, str],
        body: Any = None,
    ) -> float:
        """Update buckets from a response. Returns extra seconds to sleep on 429."""
        key = route_key(method, path)
        bucket_hash = _header(headers, "X-RateLimit-Bucket")
        if bucket_hash:
            mapped = f"{bucket_hash}:{major_param(path) or '_'}"
            self._route_bucket[key] = mapped
            bucket = self._buckets.get(mapped)
            if bucket is None:
                bucket = Bucket(hash=bucket_hash)
                self._buckets[mapped] = bucket
            else:
                bucket.hash = bucket_hash
        else:
            bucket = self._bucket_for(method, path)

        remaining = _header(headers, "X-RateLimit-Remaining")
        reset_after = _header(headers, "X-RateLimit-Reset-After")
        limit = _header(headers, "X-RateLimit-Limit")
        if remaining is not None:
            try:
                bucket.remaining = int(float(remaining))
            except ValueError:
                pass
        if limit is not None:
            try:
                bucket.limit = int(float(limit))
            except ValueError:
                pass
        if reset_after is not None:
            try:
                bucket.reset_after = float(reset_after)
                bucket.reset_monotonic = self._clock() + bucket.reset_after
            except ValueError:
                pass

        if status != 429:
            return 0.0

        wait = parse_retry_after(headers, body) + SAFETY_PAD
        if is_global_limit(headers, body):
            self._global_until = max(self._global_until, self._clock() + wait)
            log.warning("Discord GLOBAL 429: retry_after=%.3fs", wait)
        else:
            scope = (_header(headers, "X-RateLimit-Scope") or "user").lower()
            bucket.remaining = 0
            bucket.reset_after = wait
            bucket.reset_monotonic = self._clock() + wait
            log.warning("Discord 429 scope=%s bucket=%s retry_after=%.3fs", scope, bucket.hash, wait)
        return wait
