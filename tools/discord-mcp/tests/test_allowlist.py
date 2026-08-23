from discord_mcp.client import allowed_guild_ids
from discord_mcp.server import _ensure_guild_allowed


ALLOWED = "111111111111111111"
OTHER = "222222222222222222"


def test_allowed_guild_ids_parses_single(monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_GUILD_ID", ALLOWED)
    monkeypatch.delenv("DISCORD_ALLOWED_GUILDS", raising=False)
    assert allowed_guild_ids() == frozenset({ALLOWED})


def test_allowed_guild_ids_parses_csv(monkeypatch) -> None:
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)
    monkeypatch.setenv("DISCORD_ALLOWED_GUILDS", f"{ALLOWED}, {OTHER}")
    assert allowed_guild_ids() == frozenset({ALLOWED, OTHER})


def test_ensure_guild_allowed_rejects_other_server(monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_GUILD_ID", ALLOWED)
    monkeypatch.delenv("DISCORD_ALLOWED_GUILDS", raising=False)
    try:
        _ensure_guild_allowed(OTHER)
        raise AssertionError("expected lock error")
    except RuntimeError as exc:
        assert "locked" in str(exc).lower()
        assert OTHER in str(exc)


def test_ensure_guild_allowed_accepts_gfx(monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_GUILD_ID", ALLOWED)
    monkeypatch.delenv("DISCORD_ALLOWED_GUILDS", raising=False)
    assert _ensure_guild_allowed(ALLOWED) == ALLOWED
