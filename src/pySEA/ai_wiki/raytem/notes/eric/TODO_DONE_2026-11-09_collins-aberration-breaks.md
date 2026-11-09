# TODO — Collins runs break at aberrated elements, which are walked through

**Branch/worktree:** `dev` (rayTEM; sea-pearl follows for the probe)
**Plan:** none. Decision D2 of sea-pearl's `notes/eric/TODO_ACTIVE_2026-11-05_contracted-section-transfers.md`.

Eric (2026-11-09), on D2: one aberration function per run ("doesn't it matter where the aberrations
originate along the column? ... if it is not real then no") — it is exact only to the traced order and
for field-independent aberrations, so no. Option 2: "It might be slower but we only will assign
aberrations in a few places so its just a few breaks in the contractions."

How: an aberrated element ends the contracted run before it and starts the next after it; the element
itself is walked as the scaled walker walks a medium (`MEDIUM_SLICES` = 16: half a slice of the bare
body, 1/16 of its aberration phase at the local ray heights, the other half), each half slice a Collins
step on the same wave, the phase evaluated at the samples' true positions in the column. A thin lens
or an `AberrationScreen` applies its whole phase at its plane. No conversion to the scaled frame.

- [x] `CollinsTransfer`: aberrated elements (lens or plate with a power) become walked steps; supplied phase screens still refused
- [x] tests: thin lens with C30 vs a brute-force grid; thick lens with C30 vs the hybrid walker; AberrationScreen; `is_current` on a coefficient change
- [x] docs (propagation_modes, wiki), notes

Done 2026-11-09. `("walk", element, length)` steps in `collins.CollinsTransfer` (`_walk`, `_slab`,
`_aberration_power`); an aberrated element with an alignment is refused (its aberration would need its
own axis). Measured: thin C30 lens vs the fixed path on 2048² — 0.27 % of the peak (C30 moves the image
~30 %); thick C30 lens (f = 1.8 mm, 11 mrad, C30 = 2 mm) vs the hybrid walker at and just before focus —
4-8 % L1, the walker's resampling (C30 changes the image 33-110 %); an `AberrationScreen` after an ideal
lens equals the aberrated thin lens to 1e-9.
