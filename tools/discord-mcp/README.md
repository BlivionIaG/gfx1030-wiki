# Discord MCP (read-only)

Stdio MCP server. Locked to the guild in `DISCORD_GUILD_ID`. Token from env only.

```bash
export DISCORD_BOT_TOKEN=...   # never commit
export DISCORD_GUILD_ID=111111111111111111  # example
python3 -m discord_mcp.server
```
