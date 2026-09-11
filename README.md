# gfx1030-wiki

A focused, markdown-based documentation site for running LLM inference on AMD **gfx1030** (RDNA2 /
Navi 21) GPUs — with a bias toward the Radeon PRO V620. Covers V620/gfx1030 **tuning** (power cap, PCIe
P2P) and **vLLM** via the `blivioniag/vllm-rdna` Docker images and the official
`opengfx1030/vllm-rdna` extras fork.

Powered by **[mdBook](https://rust-lang.github.io/mdBook/)** and automatically deployed to GitHub Pages
via GitHub Actions. Anyone is welcome to contribute — just open a pull request.

**Live site:** https://blivioniag.github.io/gfx1030-wiki/ (after the first Pages deployment)

**Community Discord (gfx1030 club):** https://discord.gg/mESex2aBp — where most tuning tips and fork
updates land first (`#vllm-rdna`, `#llamacpp`, `#general`, `#benchmarks`).

## Contributing

You can contribute either directly on GitHub (edit a file and open a PR) or locally with mdBook so you
can preview your changes. The basics:

- Create/edit a Markdown file in `src/`.
- Add or update its entry in `src/SUMMARY.md` (this drives the sidebar).
- Optionally run `mdbook build` / `mdbook serve` to preview.
- Open a pull request against `master`.

See [`src/contributing.md`](./src/meta/contributing.md) for full guidelines.

## Local preview

mdBook is a single static binary with no runtime dependencies:

```sh
# Install mdBook (prebuilt binary; Rust not required)
mkdir -p "$HOME/.local/bin"
MDBOOK_VERSION=v0.5.4
curl -sL "https://github.com/rust-lang/mdBook/releases/download/${MDBOOK_VERSION}/mdbook-${MDBOOK_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
  | tar -xz -C "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

# Build & preview
mdbook build      # static HTML in ./book
mdbook serve      # live preview at http://localhost:3000
```

If you have a Rust toolchain, `cargo install mdbook` works too.

## Project structure

```
gfx1030-wiki/
├── book/                     # Generated static site (git-ignored, output of `mdbook build`)
├── src/                      # Markdown sources
│   ├── SUMMARY.md            # Sidebar / table of contents
│   └── *.md                  # Wiki pages
├── book.toml                 # mdBook configuration
├── tools/discord-mcp/        # Read-only Discord MCP for research (see below)
└── .github/workflows/
    └── mdbook.yml            # Build + deploy to GitHub Pages
```

## CI / Deployment

`.github/workflows/mdbook.yml` builds the book and publishes `./book` to GitHub Pages on every push to
`master`. Enable **Settings → Pages → Source: GitHub Actions** once, and every merge deploys
automatically.

## Discord MCP (Cursor Cloud Agents)

Read-only Discord tools live in `tools/discord-mcp`. They only read the **gfx1030** server. The bot token is **not** in this repo.

1. [cursor.com/dashboard/cloud-agents](https://cursor.com/dashboard/cloud-agents) → **Secrets**
2. Add **`DISCORD_BOT_TOKEN`** (bot token) and **`DISCORD_GUILD_ID`** (gfx1030 server ID). Do not
   commit either value — `.cursor/mcp.json` only points at `tools/discord-mcp/run.sh`.
3. [cursor.com/agents](https://cursor.com/agents) → gfx1030-wiki → **MCP** dropdown → enable **discord** (or add the stdio server from `.cursor/mcp.json` if it is not listed)
4. New cloud agent run: *List gfx1030 channels, then the last 20 messages in #general.*

`.cursor/environment.json` installs the package on Builds. `tools/discord-mcp/run.sh` also installs deps on first MCP start if needed.

### Daily Discord → wiki automation

A Cursor Automation can run that Discord pass on a schedule and open a **draft**
PR instead of waiting for a manual “update the wiki from Discord” prompt.

1. Create it at [cursor.com/automations/new](https://cursor.com/automations/new)
   (cloud agents cannot enable the schedule for you).
2. Use the checklist and prompt in
   [`.cursor/automations/daily-discord-wiki.md`](./.cursor/automations/daily-discord-wiki.md).
3. The agent follows
   [`.cursor/skills/discord-wiki-digest/SKILL.md`](./.cursor/skills/discord-wiki-digest/SKILL.md)
   (privacy rules, channel list, when to no-op).

Attach **this repo** on the automation (scheduled jobs default to no repo) and
leave Discord MCP + `DISCORD_BOT_TOKEN` / `DISCORD_GUILD_ID` Secrets as for other
Cloud Agents. Merge digest PRs only after a human check.
