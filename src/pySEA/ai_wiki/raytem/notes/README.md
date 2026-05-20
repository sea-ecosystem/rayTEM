# AI notes — rayTEM

This directory holds AI-maintained notes for rayTEM-specific work: design decisions,
active TODOs, status snapshots, and working notes for contributors.

Ecosystem-level concerns (cross-repo work, scaffolding across repos) belong in
`sea-ecosystem/src/pySEA/ai_wiki/ecosystem/notes/`.

## Note types

- `STATUS` — current state of a feature or component
- `DESIGN` — settled architectural decisions
- `TODO` — prioritised work items
- `WORKFLOW` — step-by-step procedures
- `INTENT` — high-level direction from a session

## Liveness markers (in filename)

| Marker | Meaning |
|--------|---------|
| `ACTIVE` | Current — read and follow |
| `PAUSED` | Valid but deprioritised |
| `SUPERSEDED` | Replaced or abandoned — do not follow |
| `ARCHIVED` | Completed — history only |
| `DONE` | TODO fully checked off |
| `DRAFT` | Not yet authoritative |

Filename format: `TYPE_LIVENESS_YYYY-MM-DD_slug.md`

## Scanning for active TODOs

When starting work in this repo, scan active TODO filenames — slugs are descriptive:

```bash
ls src/pySEA/ai_wiki/raytem/notes/shared/TODO_ACTIVE_*.md
```

Open any whose slug matches your task. Check items off as you complete them;
rename ACTIVE → DONE once all items in a file are done.

## Active notes

*(none yet)*
