"""Helpers for Discord snowflakes, channel types, and compact message output."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

CHANNEL_TYPES: dict[int, str] = {
    0: "text",
    1: "dm",
    2: "voice",
    3: "group_dm",
    4: "category",
    5: "announcement",
    10: "announcement_thread",
    11: "public_thread",
    12: "private_thread",
    13: "stage",
    14: "directory",
    15: "forum",
    16: "media",
}

READABLE_CHANNEL_TYPES = {0, 2, 5, 10, 11, 12, 13, 15, 16}

DISCORD_URL_RE = re.compile(
    r"(?:https?://)?(?:(?:ptb|canary)\.)?discord(?:app)?\.com/channels/"
    r"(?P<guild>\d+|@me)/(?P<channel>\d+)(?:/(?P<message>\d+))?",
    re.IGNORECASE,
)


def channel_type_name(type_id: int | None) -> str:
    if type_id is None:
        return "unknown"
    return CHANNEL_TYPES.get(int(type_id), f"unknown_{type_id}")


def parse_discord_ref(value: str) -> dict[str, str]:
    """Extract guild/channel/message IDs from a Discord URL, snowflake, or #name."""
    raw = value.strip()
    match = DISCORD_URL_RE.search(raw)
    if match:
        out = {k: v for k, v in match.groupdict().items() if v and v != "@me"}
        return out
    parsed = urlparse(raw)
    if parsed.scheme and "discord" in (parsed.netloc or ""):
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "channels":
            out: dict[str, str] = {}
            if parts[1] != "@me":
                out["guild"] = parts[1]
            out["channel"] = parts[2]
            if len(parts) >= 4:
                out["message"] = parts[3]
            return out
    if raw.startswith("#"):
        return {"name": raw[1:]}
    if raw.isdigit():
        return {"id": raw}
    return {"name": raw.lstrip("#")}


def author_label(user: dict[str, Any] | None) -> str:
    if not user:
        return "unknown"
    name = user.get("global_name") or user.get("username") or "unknown"
    uid = user.get("id")
    return f"{name} ({uid})" if uid else name


def summarize_attachment(att: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": att.get("id"),
        "filename": att.get("filename"),
        "content_type": att.get("content_type"),
        "size": att.get("size"),
        "url": att.get("url") or att.get("proxy_url"),
    }
    return {k: v for k, v in out.items() if v is not None}


def summarize_embed(embed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("title", "description", "url", "type"):
        if embed.get(key):
            out[key] = embed[key]
    if embed.get("author", {}).get("name"):
        out["author"] = embed["author"]["name"]
    if embed.get("fields"):
        out["fields"] = [
            {"name": f.get("name"), "value": f.get("value")}
            for f in embed["fields"]
            if f.get("name") or f.get("value")
        ]
    return out


def format_message(msg: dict[str, Any]) -> dict[str, Any]:
    reactions = []
    for reaction in msg.get("reactions") or []:
        emoji = reaction.get("emoji") or {}
        label = emoji.get("name") or ""
        if emoji.get("id"):
            label = f"{label}:{emoji['id']}"
        reactions.append({"emoji": label, "count": reaction.get("count", 0)})

    content = msg.get("content") or ""
    attachments = [summarize_attachment(a) for a in msg.get("attachments") or []]
    embeds = [summarize_embed(e) for e in msg.get("embeds") or [] if e]
    embeds = [e for e in embeds if e]

    out: dict[str, Any] = {
        "id": str(msg.get("id", "")),
        "channel_id": str(msg.get("channel_id", "")),
        "author": author_label(msg.get("author")),
        "author_id": str((msg.get("author") or {}).get("id") or ""),
        "timestamp": msg.get("timestamp"),
        "edited_timestamp": msg.get("edited_timestamp"),
        "content": content,
        "pinned": bool(msg.get("pinned")),
        "type": msg.get("type", 0),
    }
    if msg.get("webhook_id"):
        out["webhook_id"] = str(msg["webhook_id"])
    if attachments:
        out["attachments"] = attachments
    if embeds:
        out["embeds"] = embeds
    if reactions:
        out["reactions"] = reactions
    if msg.get("mention_everyone"):
        out["mention_everyone"] = True
    mentions = msg.get("mentions") or []
    if mentions:
        out["mentions"] = [author_label(u) for u in mentions]
    ref = msg.get("message_reference") or {}
    if ref.get("message_id"):
        out["reply_to_message_id"] = str(ref["message_id"])
    referenced = msg.get("referenced_message")
    if referenced:
        out["reply_to"] = {
            "id": str(referenced.get("id", "")),
            "author": author_label(referenced.get("author")),
            "content": (referenced.get("content") or "")[:240],
        }
    thread = msg.get("thread")
    if thread:
        out["thread"] = {
            "id": str(thread.get("id", "")),
            "name": thread.get("name"),
        }
    stickers = msg.get("sticker_items") or msg.get("stickers") or []
    if stickers:
        out["stickers"] = [s.get("name") for s in stickers if s.get("name")]
    if not content and not attachments and not embeds:
        out["content_missing"] = (
            "Message content is empty. If this is unexpected, enable Message "
            "Content Intent on the bot in the Discord Developer Portal."
        )
    return {k: v for k, v in out.items() if v is not None and v != ""}


def format_channel(ch: dict[str, Any], categories: dict[str, str] | None = None) -> dict[str, Any]:
    parent_id = ch.get("parent_id")
    out: dict[str, Any] = {
        "id": str(ch.get("id", "")),
        "name": ch.get("name"),
        "type": channel_type_name(ch.get("type")),
        "type_id": ch.get("type"),
        "topic": ch.get("topic") or None,
        "nsfw": bool(ch.get("nsfw")),
        "position": ch.get("position"),
        "parent_id": str(parent_id) if parent_id else None,
        "parent_name": (categories or {}).get(str(parent_id)) if parent_id else None,
        "last_message_id": str(ch["last_message_id"]) if ch.get("last_message_id") else None,
    }
    if ch.get("guild_id"):
        out["server_id"] = str(ch["guild_id"])
    meta = ch.get("thread_metadata") or {}
    if meta:
        out["archived"] = bool(meta.get("archived"))
        out["locked"] = bool(meta.get("locked"))
    if ch.get("message_count") is not None:
        out["message_count"] = ch["message_count"]
    return {k: v for k, v in out.items() if v is not None}


def format_guild(guild: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(guild.get("id", "")),
        "name": guild.get("name"),
        "owner": bool(guild.get("owner")),
        "approximate_member_count": guild.get("approximate_member_count"),
    }
