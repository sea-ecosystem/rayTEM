# TODO — wave seam cleanup: no element owns a propagation method (P1)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (Eric's architectural direction, recorded in
[PLAN_2026-08-22_thick-quadrupole-symplecticity.md](PLAN_2026-08-22_thick-quadrupole-symplecticity.md) §5)
**Related issue:** [#3](https://github.com/sea-ecosystem/rayTEM/issues/3) (its step 3
needs the propagator rename done here)

Eric: an element should declare its physics and let the generic propagator
consume it — as the ray side does with `transfer_matrix` + `propagate_ray`.
Today three elements still override `propagate_wave` itself.

- [ ] (D3) `phase_shift(dimensions, wavelength, kind='fixed'|'scaled', s=1.0)`
      dispatching to per-element `_phase_shift_fixed` / `_phase_shift_scaled`
- [ ] amplitude/mask declaration (`amplitude_mask`) so an aperture states its
      mask instead of owning a propagation method
- [ ] remove the `propagate_wave` overrides on `Source` (transparent + `wave()`
      is the seed), `Aperture` (now a mask declaration), `Prism` (declares
      "unsupported" instead of overriding propagation)
- [ ] rename `waveoptics.propagate_thick_lens_scaled` -> element-agnostic
      (it is a homogeneous quadratic-index *segment*, not a lens) and make it
      per-axis capable, so a quadrupole can use it once #3 lands
- [ ] tests: identical results before/after (pure refactor), no element defines
      a propagate_* method, mask seam covered
