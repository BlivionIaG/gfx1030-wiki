# gfx1030-wiki

Private gfx1030 / RDNA2 inference notes.

## Discord MCP (Cursor Cloud Agents)

Read-only Discord tools live in `tools/discord-mcp`. They only read the **gfx1030** server. The bot token is **not** in this repo.

1. [cursor.com/dashboard/cloud-agents](https://cursor.com/dashboard/cloud-agents) → **Secrets**
2. Add `DISCORD_BOT_TOKEN` (the Discord bot token). Do not commit it.
3. [cursor.com/agents](https://cursor.com/agents) → gfx1030-wiki → **MCP** dropdown → enable **discord** (or add the stdio server from `.cursor/mcp.json` if it is not listed)
4. New cloud agent run: *List gfx1030 channels, then the last 20 messages in #General.*

`.cursor/environment.json` installs the package on Builds. `tools/discord-mcp/run.sh` also installs deps on first MCP start if needed.
