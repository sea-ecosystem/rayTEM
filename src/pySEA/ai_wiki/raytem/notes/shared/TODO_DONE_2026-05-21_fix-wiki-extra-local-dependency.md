# TODO: Fix wiki extra local dependency resolution

## Problem

`uv run --extra wiki pysea-gen-ai-index` fails in rayTEM because the `wiki`
extra depends on `sea-ecosystem`, but rayTEM does not tell `uv` that
`sea-ecosystem` is a local sibling checkout. `uv` therefore tries to resolve
`sea-ecosystem` from the package registry and reports the extra as
unsatisfiable.

The failure is made noisier by rayTEM's broad Python range
`>=3.8,<3.18`, which makes `uv` evaluate future Python/platform resolution
splits. The local source is the primary missing piece; the Python range should
also match the currently supported ecosystem range unless rayTEM is explicitly
tested outside it.

## Proposed steps

- [x] Add a `[tool.uv.sources]` entry that points `sea-ecosystem` to the local
  parent checkout, matching the other sibling repos.
- [x] Narrow `requires-python` to the supported ecosystem range so `uv` does
  not try unsupported future Python resolution splits.
- [x] Verify the intended workflow succeeds from rayTEM:
  `uv run --extra wiki pysea-gen-ai-index`.
- [x] Re-run scaffold hygiene checks after regeneration.

## Notes

This is a repo-local fix for rayTEM. If another sibling repo hits the same
failure, compare its `pyproject.toml` against sea-eco, PoseiTEM, sea-pearl,
sea-pearl-mcp, and sea-sand, which already declare local `uv` sources for
ecosystem dependencies.
