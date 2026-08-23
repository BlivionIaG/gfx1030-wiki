# Agent notes

## Cursor Cloud specific instructions

- Discord MCP is stdio: `bash tools/discord-mcp/run.sh`
- Token comes from the Cloud Agent secret `DISCORD_BOT_TOKEN`. Never write it to the repo.
- Reads are locked to guild `DISCORD_GUILD_ID` from `.cursor/mcp.json`.
- Private Discord channels return Missing Access until the bot role is added on that channel.
- Do not vendor `.env` files. Use `.env.example` only.
