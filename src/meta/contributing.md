# Contributing

This is a community wiki for the gfx1030 (RDNA2 / Navi 21) GPU ecosystem. Contributions of all sizes are
welcome — fixing a typo, correcting an out-of-date command, or adding a whole new guide.

## Two ways to contribute

### 1. Directly on GitHub (no command line)

1. Open the page you want to edit on GitHub and click the **edit** (pencil) button, or use the
   "Suggest an edit" link on the mdBook page. You'll be prompted to fork the repo — do so.
2. Make your changes, keeping the markdown clean (see [style](#markdown-style) below).
3. Commit with a descriptive message.
4. Open a pull request targeting the `master` branch.

### 2. Locally with mdBook (lets you preview)

```sh
git clone https://github.com/blivioniag/gfx1030-wiki.git
cd gfx1030-wiki

# Install mdBook (prebuilt binary, no Rust required)
mkdir -p "$HOME/.local/bin"
MDBOOK_VERSION=v0.5.4
curl -sL "https://github.com/rust-lang/mdBook/releases/download/${MDBOOK_VERSION}/mdbook-${MDBOOK_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
  | tar -xz -C "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

mdbook serve   # live preview at http://localhost:3000
```

## Adding a new page

1. Read [Wiki structure](./structure.md) — pick the right folder and naming convention.
2. Create a markdown file under `src/<section>/`, e.g. `src/vllm/my-topic.md`.
3. Add a link in `src/SUMMARY.md` and the section `overview.md` if one exists.
4. Run `mdbook build` to confirm it renders and there are no broken links.
5. Update [Verification status](../reference/verification.md) for benchmarks or community claims.

## Markdown style

- Use ATX headings (`#`, `##`, …) and start each page with a single `#` title.
- Wrap shell commands in fenced code blocks with a language hint (```` ```sh ````).
- Prefer relative links between wiki pages (e.g. `../vllm/overview.md`) so they work locally and deployed.
- Keep lines readable; hard-wrapping around ~100 columns is fine but not required.

## Accuracy

RDNA2 tooling moves quickly. When you add a command, note the ROCm / library version you tested it with
if it might matter, and prefer linking to official docs over pasting version-specific numbers that will
age.

For pages sourced from Discord or fork release notes, add or update entries in
[Verification status](../reference/verification.md) so readers know what is **solid**, **fork-source**,
**community-reported**, or **needs verify**. Mark unconfirmed throughput claims as community-reported
rather than presenting them as wiki-tested facts.

## Privacy (Discord → wiki)

This wiki is public on GitHub Pages. When turning Discord messages into docs:

| Do | Don't |
|---|---|
| Summarize operational facts (env vars, topology, bench numbers) | Paste message URLs, message IDs, or `@mentions` |
| Attribute with channel names (`#vllm-rdna`) or "community report" | Copy Discord usernames, display names, or avatars |
| Use `/path/to/model`, `./build/bin/…`, `-hf org/model` | Copy someone's home directory, hostnames, or internal IPs |
| Link public repos (GitHub, Hugging Face, lunnova.dev) | Commit bot tokens, guild IDs, or channel IDs |
| Generalize hardware ("Ice Lake 4× V620 host") | Quote forum thread titles that include a member's name |

**Never commit:** `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `.env` files, or screenshots of private
channels. Cloud Agents should set token + guild ID in **Secrets** only (see repo `README.md`).

## Deployment

Merges to `master` are built and published to GitHub Pages automatically by the
`.github/workflows/mdbook.yml` workflow — no manual deploy step is needed.
