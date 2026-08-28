# TODO — Terminology page + audit items 1–4

**Branch/worktree:** `Signal_and_propagation_additions_new` (rayTEM)
**Plan:** none

Eric's assignment: build the docs terminology page answering issue #8, and
perform the four queued audit items, with tests, wiki sync, and a rebuilt
docs site.

- [x] Terminology page (`docs/terminology.md`, wired into `index.rst`):
      strength vs focal length vs focal power (Brown 1983 refs), aberration
      powers, `transfer_block`/`body_block`, moments, `wave_kind` accuracy,
      body and screen. Answers issue #8.
- [x] Item 1 — section-level `Aberrations`: aberrations declared on a
      `MicroscopeSection` as a whole, not just per element.
- [x] Item 2 — skew quadrupole: give `Quadrapole` a rotation so a 45°
      stigmator is representable (4×4 coupled block).
- [x] Item 3 — example 06 cleanups: mixed tabs/spaces, magic plane index.
- [x] Item 4 — `Microscope.index` raises instead of printing `ERROR:` and
      returning `None`.
- [x] Tests green, wiki refreshed, docs site rebuilt and committed, pushed.
