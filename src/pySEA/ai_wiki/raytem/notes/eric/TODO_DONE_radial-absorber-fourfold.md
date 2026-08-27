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

- [x] `boundary_window` → radially symmetric raised cosine (inscribed circle)
- [x] tests: radial-symmetry unit test + c4 < 5e-4 at the sample AND detector
- [x] suite green (67); interior ring-contrast thresholds re-measured
      (sample 0.012 → bound 0.015; detector 0.023 → bound 0.03, flat_min 0.85)
- [x] A/B figure rerun → sent Eric (c4 sample 0.00095→0.00004, detector
      0.00145→0.00005; only concentric Fresnel rings remain); docs/wiki synced
