# TODO — thick quadrupole: symplecticity + wave-side follow-on

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Issue:** [sea-ecosystem/rayTEM#3](https://github.com/sea-ecosystem/rayTEM/issues/3)
**Plan:** [PLAN_2026-08-22_thick-quadrupole-symplecticity.md](PLAN_2026-08-22_thick-quadrupole-symplecticity.md)

Blocked on a convention decision (which axis focuses for K > 0) and on
agreement to change quadrupole ray physics — see the plan's open questions.
The mitigation below is already done; the fix itself is not started.

- [x] mitigation: `Element.transfer_xblock` mirrors `transfer_matrix` exactly
      (defect included) so plane finding never diverges from ray tracing
- [x] mitigation: symplecticity guard in `Microscope._accumulate_xblocks`
      refuses a non-symplectic body, naming the determinant
- [x] issue #3 filed with reproducer, plan inlined
- [ ] step 1 — thick block symplectic (defocusing axis -> cosh/sinh);
      det == 1, halves compose, thin limit, emittance invariant
- [ ] step 2 — one axis convention for thin AND thick, stated in the docstring
- [ ] step 3 — generalize the segment propagator (element-agnostic name,
      per-axis strengths) and add `Quadrapole.scaled_segment()`; wave line foci
      match ray/analytic to ~1e-9; guard stops firing on its own
- [ ] step 3b — amplitude/mask declaration seam so `Source`/`Aperture`/`Prism`
      stop overriding `propagate_wave` (Eric's architectural point)
