# TODO — Aperture transmission: current the traced rays do not sample

**Branch/worktree:** `claude/u200-emulator-comparison-tables-z0vgol` (assigned by the session's
tooling; slug `aperture-transmission`).
**Plan:** part of the gun-section work; sea-pearl `notes/eric/TODO_ACTIVE_2026-10-27_gun-section.md`.

**Goal:** an aperture can pass a stated share of the current inside its radius.
**Why:** a gun aperture must take a µA tip emission down to a nA column current. Clipping rays can't
do it: so few rays would survive that the trace below would be useless.

- [x] `Aperture(transmission=...)`, `(0, 1]`, default 1; ray intensity, wave screen (amplitude √T),
      `transmitted_fraction`
- [x] tests: ray current, wave flux, refused values, a saved column and a file without the field
- [x] wiki `elements.md`; refresh

**Outcome:** done (`be3b2bd`). Full suite: one failure, `test_rotated_quadrupole`, which fails
without the change too.
