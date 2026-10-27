# TODO — Schema type letter `S` means sextupole

**Branch/worktree:** `claude/u200-emulator-comparison-tables-z0vgol` in rayTEM and
sea-pearl. The session's tooling assigns this branch and does not allow renaming it;
the work matches the slug `sextupole-schema-letter`.
**Plan:** none. Context: Eric's gun-section review of the U200 against the emulator
(2026-10-27), which decided "quadrupoles are for stigmation; use `S` for sextupoles".

**Goal:** `S` in the column naming schema names a sextupole, not a stigmator; a
quadrupole, stigmating or not, is `Q`.
**Why:** the U200's 200 kV schematic labels gun-lens slots `GL_4Sa`, `GL_6Sa`,
`GL_7Sb`, `GL_9Sb`, which are sextupoles. Under the old meaning they would be read as
stigmators and validated as quadrupoles.

- [x] rayTEM `schema.py`: `S` → "sextupole", realized by a `Sextupole` class (none
      exists yet, so nothing in rayTEM is typed `S` until one is added); module docs
- [x] rayTEM tests: `test_schema.py` — `Quadrapole` names only `Q`; parse tests keep `PSa`
      as a sextupole name
- [x] rayTEM `examples/01_basicRays.ipynb`: the stigmator pair `PSa`/`PSb` → `PQa`/`PQb`
- [x] rayTEM wiki `schema.md`; run the refresh
- [ ] sea-pearl scope-map schema `intents.md` (SM-3) and `scope_templates.py` docstring:
      `S` = sextupole; behaviour unchanged (two-axis `focus`)
- [ ] tests pass in both repos
