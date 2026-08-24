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

1. Create a markdown file under `src/`, e.g. `src/my_topic.md`.
2. Add a link to it in `src/SUMMARY.md` under the appropriate section — the sidebar is generated from
   this file.
3. Run `mdbook build` (or `mdbook serve`) to confirm it renders and there are no broken links.
4. Open a pull request against `master`.

## Markdown style

- Use ATX headings (`#`, `##`, …) and start each page with a single `#` title.
- Wrap shell commands in fenced code blocks with a language hint (```` ```sh ````).
- Prefer relative links between wiki pages (e.g. `./reference.md`) so they work both locally and once
  deployed.
- Keep lines readable; hard-wrapping around ~100 columns is fine but not required.

## Accuracy

RDNA2 tooling moves quickly. When you add a command, note the ROCm / library version you tested it with
if it might matter, and prefer linking to official docs over pasting version-specific numbers that will
age.

For pages sourced from Discord or fork release notes, add or update entries in
[Verification status](./verification.md) so readers know what is **solid**, **fork-source**,
**community-reported**, or **needs verify**. Mark unconfirmed throughput claims as community-reported
rather than presenting them as wiki-tested facts.

## Deployment

Merges to `master` are built and published to GitHub Pages automatically by the
`.github/workflows/mdbook.yml` workflow — no manual deploy step is needed.
