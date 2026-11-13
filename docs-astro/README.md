# rayTEM docs

This repository's HTML docs are built with the shared pySEA documentation
site in `sea-ecosystem/docs-site`. Only the settings live here; write pages in
`docs/` (MyST Markdown and RST, as before) and docstrings in the source.

| File | What it is |
|---|---|
| `site.json` | title, package, GitHub repo, edit branch; optional `theme`, `extraLinks` |
| `pages/` (optional) | this repo's own `.astro` guides |
| `dist/` | the built site (git-ignored) |

From the repository root, with sea-ecosystem as an editable sibling checkout
(`pip install -e ../sea-ecosystem`), Node 22+ and uv:

```bash
pysea-build-docs            # → docs-astro/dist
pysea-build-docs --preview  # and serve it at http://localhost:4321
```

Editing pages, `.astro` guides, checking a build against Sphinx:
`sea-ecosystem/docs-site/README.md`.
