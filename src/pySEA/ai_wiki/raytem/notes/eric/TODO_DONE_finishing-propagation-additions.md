# TODO — finishing propagation additions

**Branch/worktree:** `finishing_propagation_additions` (rayTEM), merged into
`dev` on every commit; this branch is kept up to date with `dev`. The old
`Signal_and_propagation_additions_new` branch is deleted and must not be
recreated.
**Plan:** none

- [x] Branch created from `dev`; docs CI restricted to `dev`/`main` so the
      rebuild bot cannot diverge the work branch from `dev`.
- [x] Thick-lens focal-power semantics (Eric's call): `focal_power` restored
      to the EFL power K·sin(KL) — the pupil-angle number aberrations consume;
      Thomas's measured `focal_length` stays as the back-focal distance.
      Regression test pins the split; terminology page gains EFL-vs-BFD.
- [x] Example 06 re-derived: F_OL 2→3 mm (was stale, deflating panel E by
      (2/3)²), C30 10→4.5 µm (Strehl 0.632, the intended regime), delivered
      fraction now MEASURED in panel E (0.969 — OL1 is nearly thin now).
- [x] Merge-fallout repairs in examples 01/02/03/05 (renamed
      calibrated_strength, Rays-based plot2D/plot3D/convert signatures).
