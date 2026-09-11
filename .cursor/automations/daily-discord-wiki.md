# Daily Discord → wiki (Cursor Automation)

This file is the **paste-ready prompt** and dashboard checklist for a scheduled
Cloud Agent. Cursor does not load `.cursor/automations/` as config — create the
automation at [cursor.com/automations/new](https://cursor.com/automations/new)
(or `/automate` in a **local** Agents Window), then Save and activate.

There is no public API to create automations. A cloud agent run cannot flip
**enabled** for you.

## Dashboard fields

| Field | Value |
|---|---|
| **Name** | Daily Discord wiki digest |
| **Trigger** | Scheduled — daily. Cron: `0 9 * * *` (09:00 UTC). Change if you prefer another hour. Cron may run late, never early. |
| **Repository** | **Required:** `BlivionIaG/gfx1030-wiki`, branch `master`. Scheduled triggers default to *no repo* — without this, the agent cannot edit the wiki or open a PR. |
| **Tools** | Pull request creation **on**; Memories **on**; MCP → **discord** (this repo's stdio server). Computer use optional. |
| **Environment** | The gfx1030-wiki Cloud Agent environment (mdBook + Discord MCP install). |
| **Secrets** | Runtime: `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID` (already used by Cloud Agents). |
| **Model** | Your Cloud Agent default, or a high-context model. Automations always use max context. |
| **Permissions** | Private until the first few runs look good. |

### MCP / secrets checklist

1. [cursor.com/dashboard/cloud-agents](https://cursor.com/dashboard/cloud-agents) → Secrets: `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`.
2. [cursor.com/agents](https://cursor.com/agents) → this repo → MCP dropdown → **discord** enabled.
3. On the automation: attach the **same** Discord MCP. Connecting MCP grants every tool on that server (this one is read-only).
4. Private Discord channels need the bot role or the run gets Missing Access (the skill skips those).

## Prompt (paste into the automation)

```text
You maintain the gfx1030 (RDNA2 / Navi 21) mdBook wiki in this repository.

Follow the repo skill `.cursor/skills/discord-wiki-digest/SKILL.md` end to end.
Also follow AGENTS.md and src/meta/contributing.md (especially Privacy).

## Goal
Once per day, read new gfx1030 Discord traffic via the Discord MCP (guild locked
by DISCORD_GUILD_ID) and update the wiki if there is durable, wiki-worthy
technical knowledge that is not already documented.

## Tools
- Discord MCP: list_channels, get_messages, list_threads, search_messages.
- Memories: checkpoint last_successful_run_utc, last_pr_url, missing-access
  channel names. Never store Discord message bodies or usernames in Memories.
- Open a pull request when the quality bar is met. Do not merge.

## Quality bar — open a draft PR only if all are true
- At least one factual, non-ephemeral finding (setup, ROCm, vLLM, llama.cpp,
  tuning, troubleshooting, public image/tag/fork changes).
- Grounded in Discord; cite channel names in the PR body, not message URLs.
- Change is wiki-shaped. `mdbook build` exits 0. No book/ commit.
- Privacy table passed: no Discord usernames, @mentions, message IDs/URLs,
  homes, hostnames, PCI BDFs, tokens, or guild/channel IDs.

## Decision rules
- Look back ~36 hours (48 hours on a first run). Prefer updating existing src/
  pages; add SUMMARY.md entries for new pages.
- If an open PR titled like `wiki: discord digest` exists, update that branch.
- Nothing new / already documented / MCP auth failure: no PR, no empty commit.
  On auth failure, stop; do not invent Discord content.
- Skip #memes, #marketplace, moderation, and off-arch channels unless the
  finding clearly applies to gfx1030/V620.
- Do not merge. Leave the PR draft for a human.

## Output
- PR title: `wiki: discord digest YYYY-MM-DD`
- PR body: UTC window, channels scanned, Missing Access, what changed, privacy
  check, mdbook build result.
```

## After the first run

- Confirm the run appears under [cursor.com/agents](https://cursor.com/agents) with source Automations.
- Empty Discord day → no PR is success.
- Wiki edits → draft PR; merge only after a human privacy + accuracy pass.
- Paste the automation UUID into an issue or this file later if you want agents
  to call `get-automation` and confirm it is enabled.
