"""Read-only Discord MCP server."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from discord_mcp.client import DiscordAPIError, allowed_guild_ids, client
from discord_mcp.format import (
    READABLE_CHANNEL_TYPES,
    format_channel,
    format_guild,
    format_message,
    parse_discord_ref,
)

mcp = MCPServer(
    "discord",
    instructions=(
        "Read-only Discord access, locked to the gfx1030 server when "
        "DISCORD_GUILD_ID is set. Typical flow: list_channels → get_messages "
        "or search_messages. Channel IDs, #names, and Discord message URLs "
        "are accepted. Other Discord servers are rejected. The bot can only "
        "see channels it has View Channel + Read Message History on."
    ),
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, destructive_hint=False)


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _error(exc: Exception) -> str:
    if isinstance(exc, DiscordAPIError):
        if exc.status == 403:
            return (
                f"{exc}\nThe bot is missing View Channel or Read Message History "
                "on this channel. For private channels, add the bot's role in "
                "the channel's permissions."
            )
        if exc.status == 404:
            return f"{exc}\nCheck the server/channel/message ID. The bot must be in that server."
    return str(exc)


def _guild_lock_error(guild_id: str | None = None) -> RuntimeError:
    allowed = ", ".join(sorted(allowed_guild_ids())) or "(none)"
    extra = f" Requested {guild_id}." if guild_id else ""
    return RuntimeError(
        f"This MCP is locked to the configured guild(s) (allowed IDs: {allowed}).{extra}"
    )


def _ensure_guild_allowed(guild_id: str | None) -> str:
    if not guild_id:
        raise RuntimeError("Missing server ID.")
    allowed = allowed_guild_ids()
    if allowed and guild_id not in allowed:
        raise _guild_lock_error(guild_id)
    return guild_id


async def _resolve_guild_id(server: str | None) -> str:
    discord = client()
    guilds = await discord.guilds()
    if not guilds:
        allowed = allowed_guild_ids()
        if allowed:
            raise RuntimeError(
                "The bot is not in the allowed server. Invite it, then add its "
                "role on the channels you want to read."
            )
        raise RuntimeError(
            "The bot is not in any servers. Invite it with the URL from "
            "scripts/setup-discord-bot.sh."
        )
    if server is None:
        if len(guilds) == 1:
            return _ensure_guild_allowed(str(guilds[0]["id"]))
        names = ", ".join(f"{g.get('name')} ({g.get('id')})" for g in guilds)
        raise RuntimeError(f"Pass server_id or server_name. Available: {names}")
    ref = parse_discord_ref(server)
    if ref.get("guild"):
        return _ensure_guild_allowed(ref["guild"])
    if ref.get("id"):
        wanted = ref["id"]
        _ensure_guild_allowed(wanted)
        if any(str(g.get("id")) == wanted for g in guilds):
            return wanted
        raise RuntimeError(f"The bot is not in server {wanted}.")
    needle = (ref.get("name") or server).lower().lstrip("#")
    matches = [g for g in guilds if (g.get("name") or "").lower() == needle]
    if len(matches) == 1:
        return _ensure_guild_allowed(str(matches[0]["id"]))
    contains = [g for g in guilds if needle in (g.get("name") or "").lower()]
    if len(contains) == 1:
        return _ensure_guild_allowed(str(contains[0]["id"]))
    names = ", ".join(f"{g.get('name')} ({g.get('id')})" for g in guilds)
    raise RuntimeError(f"Could not uniquely match server {server!r}. Available: {names}")


async def _resolve_channel_id(
    channel: str,
    server: str | None = None,
) -> tuple[str, dict[str, Any]]:
    discord = client()
    ref = parse_discord_ref(channel)
    if ref.get("guild"):
        _ensure_guild_allowed(ref["guild"])
    if ref.get("channel"):
        cid = ref["channel"]
        ch = await discord.channel(cid)
        _ensure_guild_allowed(str(ch.get("guild_id") or ""))
        return cid, ch
    if ref.get("id"):
        cid = ref["id"]
        try:
            ch = await discord.channel(cid)
            _ensure_guild_allowed(str(ch.get("guild_id") or ""))
            return cid, ch
        except DiscordAPIError:
            pass
    guild_id = await _resolve_guild_id(ref.get("guild") or server)
    channels = await discord.guild_channels(guild_id)
    needle = (ref.get("name") or channel).lower().lstrip("#")
    readable = [c for c in channels if c.get("type") in READABLE_CHANNEL_TYPES]
    exact = [c for c in readable if (c.get("name") or "").lower() == needle]
    if len(exact) == 1:
        return str(exact[0]["id"]), exact[0]
    contains = [c for c in readable if needle in (c.get("name") or "").lower()]
    if len(contains) == 1:
        return str(contains[0]["id"]), contains[0]
    options = ", ".join(f"#{c.get('name')} ({c.get('id')})" for c in (exact or contains or readable)[:30])
    raise RuntimeError(f"Could not uniquely match channel {channel!r}. Candidates: {options}")


@mcp.tool(annotations=READ_ONLY)
async def whoami() -> str:
    """Show the connected Discord bot's identity. Use this to verify the token works."""
    try:
        me = await client().me()
        return _dump(
            {
                "id": str(me.get("id")),
                "username": me.get("username"),
                "application_id": str(me.get("id")),
                "bot": True,
            }
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def list_servers() -> str:
    """List Discord servers this MCP may read. Locked to gfx1030 when DISCORD_GUILD_ID is set."""
    try:
        guilds = await client().guilds()
        return _dump({"servers": [format_guild(g) for g in guilds], "count": len(guilds)})
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def list_channels(
    server: Annotated[
        str | None,
        Field(description="Server ID or name. Optional if the bot is in only one server."),
    ] = None,
    query: Annotated[
        str | None,
        Field(description="Optional substring to filter channel names."),
    ] = None,
    include_categories: Annotated[
        bool,
        Field(description="Include category channels in the listing."),
    ] = False,
) -> str:
    """List channels in a server the bot can see. Returns IDs needed by get_messages."""
    try:
        guild_id = await _resolve_guild_id(server)
        discord = client()
        channels = await discord.guild_channels(guild_id)
        categories = {
            str(c["id"]): c.get("name") or ""
            for c in channels
            if c.get("type") == 4
        }
        wanted_types = set(READABLE_CHANNEL_TYPES)
        if include_categories:
            wanted_types.add(4)
        items = [c for c in channels if c.get("type") in wanted_types]
        if query:
            needle = query.lower().lstrip("#")
            items = [c for c in items if needle in (c.get("name") or "").lower()]
        items.sort(key=lambda c: (c.get("position") is None, c.get("position") or 0, c.get("id") or ""))
        formatted = [format_channel(c, categories) for c in items]
        return _dump({"server_id": guild_id, "channels": formatted, "count": len(formatted)})
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def get_messages(
    channel: Annotated[
        str,
        Field(
            description=(
                "Channel ID, #name, or a Discord channel/message URL "
                "(https://discord.com/channels/server/channel)."
            )
        ),
    ],
    server: Annotated[
        str | None,
        Field(description="Server ID or name. Needed when channel is a name shared across servers."),
    ] = None,
    limit: Annotated[
        int,
        Field(ge=1, le=500, description="How many messages to fetch (newest first from Discord, returned oldest-first)."),
    ] = 50,
    before: Annotated[
        str | None,
        Field(description="Fetch messages older than this message ID (pagination)."),
    ] = None,
    after: Annotated[
        str | None,
        Field(description="Fetch messages newer than this message ID."),
    ] = None,
    around: Annotated[
        str | None,
        Field(description="Fetch messages around this message ID. Mutually exclusive with before/after."),
    ] = None,
) -> str:
    """Read recent message history from a text channel, thread, or voice-chat channel."""
    try:
        ref = parse_discord_ref(channel)
        channel_id, channel_obj = await _resolve_channel_id(channel, server)
        around = around or ref.get("message")
        if around:
            before = None
            after = None
        discord = client()
        collected: list[dict[str, Any]] = []
        remaining = limit
        page_before = before
        page_after = after
        while remaining > 0:
            page_size = min(remaining, 100)
            page = await discord.messages(
                channel_id,
                limit=page_size,
                before=page_before,
                after=page_after,
                around=around,
            )
            if not page:
                break
            collected.extend(page)
            remaining -= len(page)
            if around or len(page) < page_size:
                break
            if after:
                page_after = str(page[0]["id"])
            else:
                page_before = str(page[-1]["id"])
        # Discord returns newest-first; present oldest-first for reading.
        collected.sort(key=lambda m: int(m.get("id") or 0))
        missing = sum(
            1
            for m in collected
            if not (m.get("content") or m.get("attachments") or m.get("embeds"))
        )
        payload: dict[str, Any] = {
            "channel": format_channel(channel_obj),
            "count": len(collected),
            "messages": [format_message(m) for m in collected],
        }
        if collected:
            payload["oldest_id"] = str(collected[0]["id"])
            payload["newest_id"] = str(collected[-1]["id"])
            payload["hint"] = "Pass before=oldest_id to page further back, or after=newest_id for newer messages."
        if missing and missing == len(collected):
            payload["warning"] = (
                "Every message had empty content. Enable Message Content Intent "
                "(Developer Portal → Bot → Privileged Gateway Intents) and save."
            )
        return _dump(payload)
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def get_message(
    channel: Annotated[
        str,
        Field(description="Channel ID, #name, or Discord message URL."),
    ],
    message_id: Annotated[
        str | None,
        Field(description="Message ID. Optional when channel is a full Discord message URL."),
    ] = None,
    server: Annotated[
        str | None,
        Field(description="Server ID or name, if channel is a name."),
    ] = None,
) -> str:
    """Fetch a single Discord message by ID or by discord.com/channels/... URL."""
    try:
        ref = parse_discord_ref(channel)
        mid = message_id or ref.get("message")
        if not mid:
            raise RuntimeError("Pass message_id, or a URL that includes the message ID.")
        channel_id, _ = await _resolve_channel_id(channel, server)
        msg = await client().message(channel_id, mid)
        return _dump(format_message(msg))
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def search_messages(
    query: Annotated[
        str,
        Field(description="Text to search for in message content."),
    ],
    server: Annotated[
        str | None,
        Field(description="Server ID or name. Optional if the bot is in only one server."),
    ] = None,
    channel: Annotated[
        str | None,
        Field(description="Optional channel ID or #name to restrict the search."),
    ] = None,
    author_id: Annotated[
        str | None,
        Field(description="Optional Discord user ID to restrict to one author."),
    ] = None,
    has: Annotated[
        Literal["image", "video", "file", "embed", "link", "sticker", "poll"] | None,
        Field(description="Optional filter for messages that include this kind of attachment/embed."),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=25, description="Max results (Discord caps at 25).")] = 25,
) -> str:
    """Search message history in a server. Requires Message Content Intent on the bot."""
    try:
        guild_id = await _resolve_guild_id(server)
        channel_ids: list[str] | None = None
        if channel:
            cid, _ = await _resolve_channel_id(channel, guild_id)
            channel_ids = [cid]
        result = await client().search_messages(
            guild_id,
            content=query,
            channel_ids=channel_ids,
            author_id=author_id,
            limit=limit,
            has=has,
        )
        nested = result.get("messages") or []
        flat: list[dict[str, Any]] = []
        for group in nested:
            if isinstance(group, list):
                flat.extend(group)
            elif isinstance(group, dict):
                flat.append(group)
        return _dump(
            {
                "server_id": guild_id,
                "total_results": result.get("total_results"),
                "count": len(flat),
                "messages": [format_message(m) for m in flat],
            }
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def list_threads(
    server: Annotated[
        str | None,
        Field(description="Server ID or name. Optional if the bot is in only one server."),
    ] = None,
    channel: Annotated[
        str | None,
        Field(description="If set, also list archived public threads in this parent channel."),
    ] = None,
) -> str:
    """List active threads in a server. Forum posts are threads — use get_messages on the thread ID."""
    try:
        guild_id = await _resolve_guild_id(server)
        discord = client()
        active = await discord.active_threads(guild_id)
        threads = list(active.get("threads") or [])
        archived: list[dict[str, Any]] = []
        if channel:
            cid, _ = await _resolve_channel_id(channel, guild_id)
            archived_payload = await discord.archived_threads(cid)
            archived = list(archived_payload.get("threads") or [])
        return _dump(
            {
                "server_id": guild_id,
                "active": [format_channel(t) for t in threads],
                "archived": [format_channel(t) for t in archived],
                "active_count": len(threads),
                "archived_count": len(archived),
            }
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(annotations=READ_ONLY)
async def get_pinned_messages(
    channel: Annotated[str, Field(description="Channel ID, #name, or Discord channel URL.")],
    server: Annotated[str | None, Field(description="Server ID or name, if channel is a name.")] = None,
) -> str:
    """List pinned messages in a channel."""
    try:
        channel_id, channel_obj = await _resolve_channel_id(channel, server)
        payload = await client().pins(channel_id)
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("messages") or []
            messages = [i.get("message", i) if isinstance(i, dict) else i for i in items]
        else:
            messages = payload or []
        return _dump(
            {
                "channel": format_channel(channel_obj),
                "count": len(messages),
                "messages": [format_message(m) for m in messages if isinstance(m, dict)],
            }
        )
    except Exception as exc:
        return _error(exc)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
