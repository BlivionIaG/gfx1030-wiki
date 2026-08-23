"""Minimal Discord REST client (bot token, read-only)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from discord_mcp.rate_limit import MAX_WAIT_SECONDS, RateLimiter, parse_retry_after

API_BASE = "https://discord.com/api/v10"
USER_AGENT = "DiscordMCP (https://github.com/local/discord-mcp, 1.0.0)"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DiscordAPIError(RuntimeError):
    def __init__(self, status: int, body: Any, path: str) -> None:
        self.status = status
        self.body = body
        self.path = path
        detail = body
        if isinstance(body, dict):
            detail = body.get("message") or body
        super().__init__(f"Discord API {status} on {path}: {detail}")


def load_env_files() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.home() / ".config" / "discord-mcp" / ".env")


def allowed_guild_ids() -> frozenset[str]:
    """Guild IDs this MCP is allowed to read. Empty means all guilds the bot is in."""
    load_env_files()
    raw = os.environ.get("DISCORD_ALLOWED_GUILDS") or os.environ.get("DISCORD_GUILD_ID") or ""
    return frozenset(part.strip() for part in raw.replace(",", " ").split() if part.strip())


def load_token() -> str:
    load_env_files()
    token = (
        os.environ.get("DISCORD_BOT_TOKEN")
        or os.environ.get("DISCORD_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "Missing DISCORD_BOT_TOKEN. Set it as a Cursor Cloud Agent secret "
            "(or write it to .env locally)."
        )
    if token.lower().startswith("bot "):
        token = token[4:].strip()
    return token


class DiscordClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or load_token()
        self._limiter = RateLimiter()
        self._http = httpx.AsyncClient(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bot {self.token}",
                "User-Agent": USER_AGENT,
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str]] | dict[str, Any] | None = None,
        retries: int = 5,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries):
            await self._limiter.acquire(method, path)
            response = await self._http.request(method, path, params=params)
            payload = _safe_json(response) if response.content else None
            wait = self._limiter.observe(
                method,
                path,
                status=response.status_code,
                headers=response.headers,
                body=payload,
            )
            if response.status_code == 429:
                if wait > MAX_WAIT_SECONDS:
                    raise DiscordAPIError(
                        429,
                        (
                            f"Rate limited for {parse_retry_after(response.headers, payload):.1f}s "
                            f"(over {MAX_WAIT_SECONDS:.0f}s cap). Retry later."
                        ),
                        path,
                    )
                await asyncio.sleep(wait)
                continue
            if response.status_code == 202:
                wait_index = 1.0
                if isinstance(payload, dict):
                    wait_index = float(payload.get("retry_after") or 1)
                if attempt < retries - 1:
                    await asyncio.sleep(min(max(wait_index, 0.5), 10.0))
                    continue
                raise DiscordAPIError(202, payload, path)
            if response.status_code == 401:
                raise DiscordAPIError(
                    401,
                    "Bot token was rejected. Reset it in the Developer Portal "
                    "(Bot → Reset Token) and re-run the setup wizard.",
                    path,
                )
            if response.status_code >= 400:
                raise DiscordAPIError(response.status_code, payload, path)
            if response.status_code == 204 or not response.content:
                return None
            return payload
        if last_error:
            raise last_error
        raise DiscordAPIError(429, "rate limited", path)

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def me(self) -> dict[str, Any]:
        return await self.get("/users/@me")

    async def guilds(self) -> list[dict[str, Any]]:
        items = await self.get("/users/@me/guilds", params={"with_counts": "true"})
        allowed = allowed_guild_ids()
        if allowed:
            items = [g for g in items if str(g.get("id")) in allowed]
        return items

    async def guild(self, guild_id: str) -> dict[str, Any]:
        return await self.get(f"/guilds/{guild_id}")

    async def guild_channels(self, guild_id: str) -> list[dict[str, Any]]:
        return await self.get(f"/guilds/{guild_id}/channels")

    async def channel(self, channel_id: str) -> dict[str, Any]:
        return await self.get(f"/channels/{channel_id}")

    async def messages(
        self,
        channel_id: str,
        *,
        limit: int = 50,
        before: str | None = None,
        after: str | None = None,
        around: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {"limit": str(max(1, min(limit, 100)))}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        if around:
            params["around"] = around
        return await self.get(f"/channels/{channel_id}/messages", params=params)

    async def message(self, channel_id: str, message_id: str) -> dict[str, Any]:
        return await self.get(f"/channels/{channel_id}/messages/{message_id}")

    async def pins(self, channel_id: str) -> Any:
        try:
            return await self.get(f"/channels/{channel_id}/messages/pins")
        except DiscordAPIError as exc:
            if exc.status not in {404, 405}:
                raise
            return await self.get(f"/channels/{channel_id}/pins")

    async def active_threads(self, guild_id: str) -> dict[str, Any]:
        return await self.get(f"/guilds/{guild_id}/threads/active")

    async def archived_threads(self, channel_id: str) -> dict[str, Any]:
        return await self.get(f"/channels/{channel_id}/threads/archived/public")

    async def search_messages(
        self,
        guild_id: str,
        *,
        content: str | None = None,
        channel_ids: list[str] | None = None,
        author_id: str | None = None,
        limit: int = 25,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        has: str | None = None,
    ) -> dict[str, Any]:
        params: list[tuple[str, str]] = [
            ("limit", str(max(1, min(limit, 25)))),
            ("sort_by", sort_by),
            ("sort_order", sort_order),
        ]
        if content:
            params.append(("content", content))
        if author_id:
            params.append(("author_id", author_id))
        if has:
            params.append(("has", has))
        for cid in channel_ids or []:
            params.append(("channel_id", cid))
        return await self.get(f"/guilds/{guild_id}/messages/search", params=params)


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


_client: DiscordClient | None = None


def client() -> DiscordClient:
    global _client
    if _client is None:
        _client = DiscordClient()
    return _client
