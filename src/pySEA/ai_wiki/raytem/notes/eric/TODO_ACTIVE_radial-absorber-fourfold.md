# TODO — radial absorbing boundary (fix the fourfold fringe pattern)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (diagnosis + fix recorded in LOG entry; c4 table below)

Eric: all downstream planes carry a fourfold, pixel-aligned fringe pattern
inside the disc, with and without the band limit. Diagnosis (fourfold harmonic
c4 of interior intensity, suspects toggled in-memory): the square separable
`boundary_window` is the source — its corners sit √2 farther than its edges,
so the aperture halo is clipped azimuthally anisotropically at every τ
sub-step and the fourfold-modulated survivor interferes back into the disc.
Radial window kills c4 to 0.0000 at sample and detector; band limit and
beam-support policy toggles change nothing (c4 0.0008–0.0015 all).

- [ ] `boundary_window` → radially symmetric raised cosine (inscribed circle)
- [ ] tests: radial-symmetry unit test + c4 < 5e-4 at the sample plane
- [ ] suite green; re-measure energy thresholds if tripped
- [ ] A/B figure rerun → send Eric; docs/wiki sync; protocol finish
