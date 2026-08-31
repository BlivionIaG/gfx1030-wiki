# AGENTS.md

## Cursor Cloud specific instructions

This repository has two parts:

- A **documentation-only wiki** for AMD gfx1030 (RDNA2 / Navi 21) GPUs, built with
  [mdBook](https://rust-lang.github.io/mdBook/) — the "product" is a static HTML site generated from
  Markdown under `src/`.
- A small **Discord MCP** tool under `tools/discord-mcp/` used to pull gfx1030 Discord conversations into
  the agent for wiki research.

### Wiki (mdBook)

There is **no application server, database, backend, or automated test/lint suite** for the wiki.

#### Layout
- `src/*.md` — the wiki pages (Markdown).
- `src/SUMMARY.md` — the table of contents; it drives the sidebar. A page only appears in the site if it
  is listed here.
- `book.toml` — mdBook configuration.
- `.github/workflows/mdbook.yml` — builds the book and deploys `./book` to GitHub Pages on push to
  `master`.
- `book/` — build output; git-ignored, do not commit.

#### Tooling
- The only tool needed is the `mdbook` binary. The startup/update script installs it to
  `/usr/local/cargo/bin/mdbook`, which is already on `PATH` (no `~/.bashrc` changes required). If for
  some reason `mdbook` is missing, re-run the update script or grab a prebuilt binary from
  https://github.com/rust-lang/mdBook/releases .
- This is pinned to **mdBook 0.5.x**. Note two 0.5.x gotchas discovered during setup:
  - `book.toml` must **not** contain the `multilingual` field (removed in 0.5; it errors the build).
  - Search assets are emitted with hashed filenames (e.g. `searchindex-<hash>.js`), not the old
    `searchindex.js` — this is expected, search still works.

#### Build / serve / "test"
- Build: `mdbook build` → static site in `./book`. This is also the closest thing to a lint/test:
  it exits non-zero on config errors and reports broken internal links.
- Pull requests to `master` run the same build in CI (`.github/workflows/mdbook-check.yml`);
  merge is blocked only if you enable the **mdBook build check** required status in branch protection.
- Preview: `mdbook serve --hostname 127.0.0.1 --port 3000` then open http://127.0.0.1:3000/ .
  `mdbook serve` live-reloads on file changes. Run it in a tmux session for the browser preview.
- There is no separate lint command and no unit tests; `mdbook build` succeeding cleanly is the
  validation signal.

#### Deployment
- Pushes to `master` auto-deploy via `.github/workflows/mdbook.yml` to
  `https://blivioniag.github.io/gfx1030-wiki/`.
- The workflow sets `enablement: true` on `configure-pages` so the first deploy can turn on
  GitHub Pages without a manual **Settings → Pages → Source: GitHub Actions** step. If auto-enable
  fails (repo permissions), set that source once by hand.

### Discord MCP (`tools/discord-mcp/`)

- Discord MCP is stdio: `bash tools/discord-mcp/run.sh`
- Token and guild lock come from Cloud Agent **Secrets** (`DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`).
  Never write either value to the repo.
- Reads are locked to the guild in `DISCORD_GUILD_ID` when that secret is set.
- Private Discord channels return Missing Access until the bot role is added on that channel.
- Do not vendor `.env` files. Use `.env.example` only.
