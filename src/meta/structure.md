# Wiki structure

How this mdBook is organized — for editors adding or moving pages.

## Directory layout

```text
src/
  intro.md                 # Landing page (WIP banner)
  SUMMARY.md               # Sidebar navigation — edit this when adding pages

  setup/                   # Hardware and ROCm install
    hardware.md
    getting-started.md
    installing-rocm.md
    hsa-override.md

  tuning/                  # V620-specific host tuning
    power.md
    p2p.md

  vllm/                    # vLLM serving on RDNA
    overview.md            # Section hub — start here
    running.md             # Images, docker run, quick start
    configuration.md       # Env vars, CUDA graphs, compose
    quantization.md        # GPTQ/AWQ, KV, MTP, INT4
    fork.md                # rdna2_extras kernels
    images.md              # Building Docker images

  llama-cpp/               # llama.cpp and RDNA2 fork
    overview.md            # Section hub
    building.md            # Stock llama.cpp build
    rdna2-overview.md      # Fork intro, build, requirements
    rdna2-benchmarks.md    # Author-reported numbers
    rdna2-speculative.md   # DFlash2, MTP, ngram
    rdna2-serving.md         # Launch, Docker, limits

  reference/               # Lookup tables
    env-vars.md            # Environment variable cheat-sheet
    verification.md        # What's verified vs community-reported

  troubleshooting/         # Problem → fix, by stack
    index.md               # Hub
    general.md
    vllm.md
    llama-cpp.md

  meta/                    # Wiki maintenance
    resources.md           # External links
    contributing.md
    github-pages.md        # Pages deploy setup
    structure.md           # This page
```

## Conventions

| Rule | Example |
|---|---|
| **One topic per file** — split when a page exceeds ~200 lines or mixes concerns | `vllm/configuration.md` vs `vllm/quantization.md` |
| **Section `overview.md`** — first page in each major folder links to sub-pages | `vllm/overview.md`, `llama-cpp/overview.md` |
| **kebab-case filenames** | `getting-started.md`, not `getting_started.md` |
| **Relative links** — use `../` to cross folders | From `vllm/running.md` to setup: `../setup/hsa-override.md` |
| **WIP banner** — on pages with Discord/community claims | See `vllm/overview.md` |
| **Verification row** — add/update in `reference/verification.md` for new benchmarks | Status: solid / fork-source / community / needs verify |

## Adding a new page

1. Pick the folder (or create one if it's a new major topic).
2. Create `src/<folder>/<topic>.md` with a `#` title and optional WIP banner.
3. Add an entry to `src/SUMMARY.md` under the right section.
4. Link from the section `overview.md` if one exists.
5. Add verification rows if the page contains benchmarks or unconfirmed claims.
6. Run `mdbook build` — fix broken links before merging.

## Splitting an existing page

1. Move shared intro to `overview.md` or trim the original.
2. Create focused sub-pages; link between them at the bottom ("Next steps").
3. Update `SUMMARY.md` to nest sub-pages under the section.
4. Grep for old filename links: `rg 'old-name\.md' src/`
5. Update `reference/verification.md` section headers to match new paths.

## What goes where?

| Content type | Location |
|---|---|
| Install ROCm, hardware list | `setup/` |
| Power cap, P2P enablement | `tuning/` |
| `docker run`, compose, env vars for vLLM | `vllm/` |
| Kernel source, fork branches | `vllm/fork.md` |
| Docker image build (vllm-rdna-docker) | `vllm/images.md` |
| Stock llama.cpp cmake build | `llama-cpp/building.md` |
| Fork benchmarks, DFlash2, TP serve | `llama-cpp/rdna2-*.md` |
| Env var tables | `reference/env-vars.md` |
| Claim audit / WIP status | `reference/verification.md` |
| Error messages and fixes | `troubleshooting/` |
| External repos, Discord, PR links | `meta/resources.md` |
