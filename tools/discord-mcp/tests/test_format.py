from discord_mcp.format import (
    format_channel,
    format_message,
    parse_discord_ref,
)


def test_parse_message_url() -> None:
    ref = parse_discord_ref(
        "https://discord.com/channels/111/222/333"
    )
    assert ref == {"guild": "111", "channel": "222", "message": "333"}


def test_parse_channel_url() -> None:
    ref = parse_discord_ref("https://discord.com/channels/111/222")
    assert ref == {"guild": "111", "channel": "222"}


def test_parse_hash_name() -> None:
    assert parse_discord_ref("#general") == {"name": "general"}


def test_parse_snowflake() -> None:
    assert parse_discord_ref("123456789012345678") == {"id": "123456789012345678"}


def test_format_message_compact() -> None:
    formatted = format_message(
        {
            "id": "1",
            "channel_id": "2",
            "content": "hello",
            "timestamp": "2026-01-01T00:00:00.000000+00:00",
            "author": {"id": "9", "username": "alice", "global_name": "Alice"},
            "attachments": [{"filename": "a.png", "url": "https://cdn.example/a.png"}],
            "reactions": [{"count": 2, "emoji": {"name": "\ud83d\udd25"}}],
        }
    )
    assert formatted["author"] == "Alice (9)"
    assert formatted["content"] == "hello"
    assert formatted["attachments"][0]["filename"] == "a.png"
    assert formatted["reactions"][0]["emoji"] == "\ud83d\udd25"
    assert "content_missing" not in formatted


def test_format_message_empty_content_warns() -> None:
    formatted = format_message(
        {
            "id": "1",
            "channel_id": "2",
            "content": "",
            "author": {"id": "9", "username": "bot"},
        }
    )
    assert "content_missing" in formatted


def test_format_channel_includes_parent_name() -> None:
    formatted = format_channel(
        {
            "id": "10",
            "name": "general",
            "type": 0,
            "parent_id": "99",
            "guild_id": "1",
        },
        {"99": "chat"},
    )
    assert formatted["type"] == "text"
    assert formatted["parent_name"] == "chat"
    assert formatted["server_id"] == "1"
