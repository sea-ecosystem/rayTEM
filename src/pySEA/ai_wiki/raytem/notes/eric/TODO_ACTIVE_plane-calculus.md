# TODO — one plane calculus: wave image planes, covariance waists, reference planes

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-21_matrix-conjugate-planes.md](PLAN_2026-08-21_matrix-conjugate-planes.md)
(ray-side note) · [PLAN_2026-08-22_aberrated-focal-surfaces.md](PLAN_2026-08-22_aberrated-focal-surfaces.md)
(item 4, for Eric + the other contributor)

Eric's four asks. 1-3 share one mechanism: a single per-element walk of the
accumulated 2x2, which is simultaneously the wave frame's own arithmetic
(the frame IS a reference ray) and the ray matrices — so one implementation
answers all three and the three methods must agree.

- [ ] `Element.transfer_xblock(dz, axis)` seam: rotating-frame 2x2 for a
      partial or full length (base = drift + thin kick; `Lens` = cos/sin)
- [ ] (1) wave image planes: same walk with the conjugate seed -> `B = 0`.
      The frame update *is* the 2x2, so this must reproduce the analytic and
      ray numbers exactly; `wavefield_at` then reconstructs the field there
- [ ] (3) `reference=` : accumulate from a named element instead of the
      column entrance, so "planes conjugate to the sample" and "conjugate to
      the condenser aperture" are different, correct answers; report both
      absolute z and offset from the reference
- [ ] (2) covariance: `beam_waists(axis)` from `Sigma_12 = 0` per segment,
      with the width there — the finite-emittance analog of a crossover, and
      the honest minimum-width plane (offset from the geometric crossover is
      the emittance-driven focal shift)
- [ ] tests: frame/matrix vs ray agreement; reference= reproduces the
      entrance case; waists bracket the crossovers; thick-lens interiors
- [ ] (4) write the aberrated-focal-surface plan; docs/wiki; protocol finish
