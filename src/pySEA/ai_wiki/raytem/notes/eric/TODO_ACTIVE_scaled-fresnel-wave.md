# TODO — Scaled Fresnel propagation (`propagate_wave_scaled`)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-19_scaled-fresnel-wave.md](PLAN_2026-08-19_scaled-fresnel-wave.md)

- [ ] Step 1 — `Element.phase_shift(dimensions, wavelength, scaled=False, s=1)`:
      explicit per-element χ (Lens round, Quadrapole saddle, Dipole tilt, Drift
      kernel), space-tagged phase Signals, waveoptics χ builders +
      `include_carrier` kwarg; refactor fixed `propagate_wave` to consume it;
      test 0 (defs + domain tags + scaled split + fixed-path regression)
- [ ] Step 2 — scaled Signal seam (`make_scaled_wavefield_signal`/
      `read_scaled_wavefield` + fallback); `factor_wave` +
      `reconstruct_physical_wave` (matching grids); identity test
- [ ] Step 3 — scaled free propagation, constant s (R=∞); equivalence vs
      ordinary propagator
- [ ] Step 4 — linear s(z) + Δτ (closed form verified vs numeric integral);
      curved-R equivalence, grid-scaling and normalization tests
- [ ] Step 5 — `phase_shift(scaled=True)` consumption (lens→R absorption,
      quad/dipole→phase on U with sampling guard); thin-lens comparison;
      aperture→free→lens→free system incl. `Source.aperture_field`;
      electron-scale invariants (200 kV, 20 µm, f=45 mm) + s_min guard
- [ ] Step 6 — Fourier band-limited target-grid reconstruction (Eq 44);
      entrance-plane equivalence test
- [ ] Step 7 — `propagate_wave_scaled` drivers + `.wave_scaled` SignalSet +
      `Microscope.wavefield_at` + dispatcher `kind="wave-scaled"`; column test;
      full suite green
- [ ] Step 8 — docs (`wave-optics-sampling.md` update), wiki refresh +
      hand-edits, crossover chart-switching follow-up issue; finish protocol
