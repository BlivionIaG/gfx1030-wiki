# Building the Images

The [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) and
[`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) images are produced by the
[`vllm-rdna-docker`](https://github.com/blivioniag/vllm-rdna-docker) repo. It's deliberately small: two
Dockerfiles, one bake graph, one CI workflow — no Python layer, no custom linter.

## Layout

| File | Purpose |
|---|---|
| `Dockerfile.base` | The ROCm + PyTorch + Triton base (`rocm-rdna`). |
| `Dockerfile.vllm` | Clones and builds vLLM on top of a base image (`vllm-rdna`). |
| `docker-bake.hcl` | **Source of truth** — defines every target, tag, and build arg. |
| `patches/*.patch` | RDNA-specific fixes applied after the vLLM clone. |
| `.github/workflows/build.yml` | CI: builds on tag push (`v*`) and manual dispatch. |

The seven RDNA archs `gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201` are baked into every image
via `PYTORCH_ROCM_ARCH`.

## Build locally

Everything runs through stock `docker buildx bake`:

```bash
# Build everything (bases + vLLM images)
docker buildx bake --file docker-bake.hcl all

# Just the base images, or just the vLLM images
docker buildx bake --file docker-bake.hcl all-bases
docker buildx bake --file docker-bake.hcl all-vllm

# A single target
docker buildx bake --file docker-bake.hcl vllm-0271-rocm720

# Print the plan without building
docker buildx bake --file docker-bake.hcl --print all
```

Targets are named `vllm-<source>-<base>`, e.g. `vllm-0271-rocm720`, `vllm-0271-rocm720-extras`. Groups:
`all`, `all-bases`, `all-vllm`.

## Key build arguments (`Dockerfile.vllm`)

Set per-target in `docker-bake.hcl`:

| ARG | Purpose |
|---|---|
| `BASE_IMAGE` | Published base image, e.g. `blivioniag/rocm-rdna:7.2.0`. |
| `VLLM_REPOSITORY` / `VLLM_REF` | vLLM clone URL and git ref (tag or branch). |
| `VLLM_COMMIT` | Full 40-char commit; the build fails if `HEAD` doesn't match (reproducibility). |
| `VLLM_VARIANT` | `upstream` or `extras-fork` (recorded as an image label). |
| `PYTORCH_ROCM_ARCH` | Semicolon-joined gfx targets. |
| `TORCH_BACKEND` | `uv --torch-backend` value, e.g. `rocm7.2`. |
| `FLASH_ATTENTION_INSTALL` | `base` \| `vllm` \| `none`, plus `FLASH_ATTENTION_REPO` / `_REF`. |
| `USE_SCCACHE` | `1` to wrap HIP compilation in sccache (base must also be built with it). |
| `VLLM_PATCH_FILE` | A `.patch` in `patches/` applied after the clone (empty = none). |

## The `patches/` directory

vLLM occasionally needs small RDNA fixes (e.g. platform detection so consumer Radeon cards are picked up).
The repo keeps these as versioned diffs, for example:

- `patches/v0.26.0-rocm-platforms.patch`
- `patches/v0.27.1-rocm-platform-detect.patch`
- `patches/v0.27.1-amdsmi-wrapper-guard.patch`
- `patches/v0.27.1-extras-*.patch`

To add one for a new vLLM release: reproduce the fix against a fresh clone, `git diff` the changed
file(s) into `patches/`, verify it applies against a clean checkout of that tag, then point the target's
`VLLM_PATCH_FILE` at it.

## Adding a new base or vLLM source

- **New base:** copy a `target "base-<id>"` block, set the ROCm/PyTorch/Triton versions and index URL,
  and add the id to `all-bases` + `all-vllm`.
- **New vLLM source:** add a `target "vllm-<source>-<base>"` block per base with `VLLM_REPOSITORY`,
  `VLLM_REF`, `VLLM_COMMIT`, `VLLM_VARIANT`, `IMAGE_TAG`, and add the ids to `all-vllm`.

See the [`vllm-rdna-docker` README](https://github.com/blivioniag/vllm-rdna-docker) for the full,
authoritative instructions.

## CI

`.github/workflows/build.yml` builds and pushes to `docker.io/blivioniag/` on tag push (`v*`), and
supports manual dispatch of a single target (optionally pushing). It uses stock `docker/setup-buildx`,
`docker/login`, `docker/metadata`, and `docker/bake` actions with GitHub Actions cache.
