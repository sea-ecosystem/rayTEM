# rayTEM Agent Notes

This repository is a ray-tracing electron optics simulator for TEM instruments.
Read this file before adding elements, changing the ray representation, or touching fitting or calibration logic.

## Read first (always)

1. `src/pySEA/ai_wiki/raytem/index.md` — orientation and mental model
2. `src/pySEA/ai_wiki/raytem/layer-map.md` — module responsibilities and invariants
3. `src/pySEA/ai_wiki/raytem/wiki/<relevant-module>.md` — method-level detail for the file you are about to edit

For ecosystem context (what other repos exist and how they connect), see
`src/pySEA/ai_wiki/ecosystem/CLAUDE.md` (installed via the `sea-ecosystem` wiki dependency — `pip install -e ".[dev,wiki]"`).

## Navigation

Two-step lookup for any class or method:

1. Search `src/pySEA/ai_wiki/raytem/method-index.json` by `class` + `method` name
   → get `wiki_path` and `wiki_lineno`
2. Read `wiki_path` at `offset=wiki_lineno` — lands directly in the right section

If `wiki_method_lineno` is non-null, use it instead for a tighter read.

## Freshness loop

**Before editing a `.py` file:**
Read `src/pySEA/ai_wiki/raytem/wiki/<module>.md` (use method-index.json to find the path).

**After editing a `.py` file:**
Update the corresponding wiki doc — revise changed method entries, add cross-links for new relationships. Commit the doc alongside the code change.

**After editing any wiki doc:**
Run `pysea-refresh-wiki` to update TOC headers, wiki line numbers, and the ecosystem index:
```bash
pysea-refresh-wiki
```

**After structural changes** (new modules, renamed files, moved functions):
```bash
pysea-refresh-wiki
```
This refreshes repo-map.md, symbol-index.json, method-index.json, all TOC headers,
and the ecosystem index.md.

## Contributor notes

Before writing any note, run `git config user.name` to identify the current contributor.
Write working notes to `src/pySEA/ai_wiki/raytem/notes/<name>/`.
Shared architectural decisions go in `notes/shared/`.

## Core invariants

- **Ray columns are fixed**: [x, xθ, y, yθ, z, I, E, R] — index 0–7 as returned by `columnByName()`. R is cumulative Larmor rotation accumulated by thick lenses. Do not reorder without updating every Element and all `columnByName` callers.
- **Transfer matrices are 8×8**: Produced by `fix_mat_dims()`. Never pass a raw 2×2 matrix directly to `propagate_ray`.
- **Element → MicroscopeSection → Microscope hierarchy**: Rays propagate strictly bottom-up. Microscope never reaches inside an Element; MicroscopeSection never reaches inside a Microscope.
- **seashells is the sea_eco seam**: All serialization goes through `seashells.SEASerializable`, which gracefully degrades if sea_eco is absent. Do not import from sea_eco directly in framework code.

## High-risk areas

- `elements.py` — `columnByName()` and `fix_mat_dims()` are referenced everywhere; changes break all Elements
- `seashells.py` — conditional import logic; changing import path or attribute names breaks sea_eco round-trips
- `AS2.py` — talks to a live instrument; errors here can send bad values to real hardware

## Style rules

- NumPy docstring style for public callables.
- Prefer explicit classes over dataclasses unless clearly justified.
- Keep method names action-oriented.
- `microscopes/` subdirectories contain instrument-specific scripts, not general framework code. Do not import from them in framework modules.

## AI workflow

- Prefer small, targeted changes.
- Update or add tests when framework behavior changes.
- Keep wiki docs synchronized with module behavior.
- If a change modifies the ray representation or transfer matrix conventions, explain the change explicitly in the wiki doc.

## Accessing wiki content programmatically

From any Python environment with `raytem` installed:

```python
from importlib.resources import files
index = (files("pySEA") / "ai_wiki" / "raytem" / "index.md").read_text()
method_index = (files("pySEA") / "ai_wiki" / "raytem" / "method-index.json").read_text()
```
