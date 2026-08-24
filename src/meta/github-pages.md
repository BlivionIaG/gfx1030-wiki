# GitHub Pages setup

The wiki is published with **mdBook** and **GitHub Actions**. The live URL (once deployed):

**https://blivioniag.github.io/gfx1030-wiki/**

## One-time setup (repo admin)

You only need to do this once per repository.

### 1. Enable GitHub Pages with Actions as the source

1. Open **https://github.com/BlivionIaG/gfx1030-wiki/settings/pages**
2. Under **Build and deployment → Source**, choose **GitHub Actions** (not “Deploy from a branch”).
3. Save. You do not need to pick a branch or `/docs` folder — the workflow builds `./book` from mdBook.

> If **GitHub Actions** is not listed, confirm you have admin access to the repo. Organization repos may
> need an org admin to allow Pages.

### 2. Run the deploy workflow

After Pages is enabled, either:

- **Push to `master`** — `.github/workflows/mdbook.yml` runs automatically, or
- **Actions → “Deploy mdBook site to Pages” → Run workflow** (manual dispatch).

The workflow uses `actions/configure-pages@v5` with `enablement: true`, which can register Pages on first
run if step 1 was skipped — but choosing **GitHub Actions** in Settings is still the reliable path.

### 3. Confirm deployment

1. **Actions** tab — latest “Deploy mdBook site to Pages” run should be green.
2. **Settings → Pages** — shows the site URL and last deployment time.
3. Open **https://blivioniag.github.io/gfx1030-wiki/** — you should see the Introduction page.

First deploy can take 1–2 minutes after the workflow finishes.

## How it works

| Piece | Role |
|---|---|
| `book.toml` | mdBook config; `site-url = "/gfx1030-wiki/"` fixes links on the project site |
| `src/` + `SUMMARY.md` | Markdown sources and sidebar |
| `.github/workflows/mdbook.yml` | Builds on push/PR to `master`; deploys only on `master` push |
| `./book/` | Generated HTML (git-ignored); uploaded as the Pages artifact |

## Troubleshooting

### Workflow fails at “Setup Pages” / `Get Pages site failed`

**Cause:** Pages not enabled, or Source is not **GitHub Actions**.

**Fix:** Complete [step 1](#1-enable-github-pages-with-actions-as-the-source) above, then re-run the workflow.

### Site loads but CSS / links are broken

**Cause:** Missing or wrong `site-url` in `book.toml`.

**Fix:** Must be `site-url = "/gfx1030-wiki/"` (trailing slash, repo name matches GitHub repo).

### Changes not visible after merge

- Hard-refresh the browser (Ctrl+Shift+R).
- Check **Actions** — latest deploy succeeded after your merge.
- Pages can cache briefly; wait a minute and retry.

### Pull requests

The workflow **builds** on PRs (validates `mdbook build`) but does **not** deploy. Only pushes to `master`
update the live site.

## Custom domain (optional)

1. Add a `CNAME` file or configure the domain under **Settings → Pages**.
2. Update `site-url` in `book.toml` to your domain (e.g. `https://wiki.example.com/`).
3. Re-run the deploy workflow.

For the default `*.github.io` URL, no custom domain setup is required.
