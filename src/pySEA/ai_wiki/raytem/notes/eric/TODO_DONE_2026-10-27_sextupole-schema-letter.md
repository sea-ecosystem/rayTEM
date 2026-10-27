# TODO — Schema type letter `S` means sextupole

**Branch/worktree:** `claude/u200-emulator-comparison-tables-z0vgol` in rayTEM and
sea-pearl. The session's tooling assigns this branch and does not allow renaming it;
the work matches the slug `sextupole-schema-letter`.
**Plan:** none. Context: Eric's gun-section review of the U200 against the emulator
(2026-10-27), which decided "quadrupoles are for stigmation; use `S` for sextupoles".

**Goal:** `S` in the column naming schema names a sextupole, not a stigmator; a
quadrupole, stigmating or not, is `Q`.
**Why:** a gun lens's multipole stack can hold sextupole slots. Under the old meaning
they would be read as
stigmators and validated as quadrupoles.

- [x] rayTEM `schema.py`: `S` → "sextupole", realized by a `Sextupole` class (none
      exists yet, so nothing in rayTEM is typed `S` until one is added); module docs
- [x] rayTEM tests: `test_schema.py` — `Quadrapole` names only `Q`; parse tests keep `PSa`
      as a sextupole name
- [x] rayTEM `examples/01_basicRays.ipynb`: the stigmator pair `PSa`/`PSb` → `PQa`/`PQb`
- [x] rayTEM wiki `schema.md`; run the refresh
- [x] sea-pearl scope-map schema `intents.md` (SM-3) and `scope_templates.py` docstring:
      `S` = sextupole; behaviour unchanged (two-axis `focus`)
- [x] tests pass in both repos

**Outcome:** done. rayTEM `b5fd790`, sea-pearl `scope-map` intents and message. rayTEM full
suite: one failure, `test_rotated_quadrupole`, which fails on the parent commit too. The
saved U200, emulator and emulator_column scopes validate with no errors (none of them
names anything with `S`). Not run: the example notebook (no Jupyter here); only the two
element names and their echoes in its saved output changed.
