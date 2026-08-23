# TODO — thick quadrupole: symplecticity + wave-side follow-on

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Issue:** [sea-ecosystem/rayTEM#3](https://github.com/sea-ecosystem/rayTEM/issues/3)
**Plan:** [PLAN_2026-08-22_thick-quadrupole-symplecticity.md](PLAN_2026-08-22_thick-quadrupole-symplecticity.md)

**Unblocked 2026-08-23.** Eric settled the convention: **K > 0 focuses x and
defocuses y**, in both the thin and thick branches. Steps 1-2 in progress.
Skew (the quad angle) is acknowledged as needed but is a separate feature —
the class has no angle parameter, and a rotated quadrupole couples x and y, so
it cannot be expressed as two independent 2x2 blocks. Step 3 stays queued.

- [x] mitigation: `Element.transfer_xblock` mirrors `transfer_matrix` exactly
      (defect included) so plane finding never diverges from ray tracing
- [x] mitigation: symplecticity guard in `Microscope._accumulate_xblocks`
      refuses a non-symplectic body, naming the determinant
- [x] issue #3 filed with reproducer, plan inlined
- [ ] step 1 — thick block symplectic (defocusing axis -> cosh/sinh);
      det == 1, halves compose, thin limit, emittance invariant
- [ ] step 1b — the B term uses signed `1/K`, so it goes *negative* for K < 0
      (a drift-like term must not); use k = |K| in the block
- [ ] step 2 — one axis convention for thin AND thick (K > 0 focuses x),
      stated in the docstring; removes the thin `X,Y` swap
- [ ] step 2b — one private body-block helper feeding `transfer_matrix`,
      `transfer_xblock` and `focal_powers`, so the three cannot drift apart
- [ ] step 3 — generalize the segment propagator (element-agnostic name,
      per-axis strengths) and add `Quadrapole.scaled_segment()`; wave line foci
      match ray/analytic to ~1e-9; guard stops firing on its own
- [x] step 3b — **dropped.** Tried and reverted 2026-08-23: an aperture is not
      expressible as a phase (exp(i*chi) is unimodular), but it is the only
      element that needs an amplitude seam, so the overrides stay. See
      TODO_REVERTED_wave-seam-cleanup.md.
- [ ] step 4 (new, deferred) — skew/rotated quadrupole: needs an angle
      parameter and x-y coupling, so it is a feature, not a sign fix
