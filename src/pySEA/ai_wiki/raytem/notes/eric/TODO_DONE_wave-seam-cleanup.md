# TODO — wave seam cleanup: no element owns a propagation method (P1)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (Eric's architectural direction, recorded in
[PLAN_2026-08-22_thick-quadrupole-symplecticity.md](PLAN_2026-08-22_thick-quadrupole-symplecticity.md) §5)
**Related issue:** [#3](https://github.com/sea-ecosystem/rayTEM/issues/3) (its step 3
needs the propagator rename done here)

Eric: an element should declare its physics and let the generic propagator
consume it — as the ray side does with `transfer_matrix` + `propagate_ray`.
Today three elements still override `propagate_wave` itself.

- [x] (D3) `phase_shift(dimensions, wavelength, kind='fixed'|'scaled', s=1.0)`
      dispatching to per-element `_phase_shift_fixed` / `_phase_shift_scaled`
- [x] amplitude/mask declaration (`amplitude_mask`) so an aperture states its
      mask instead of owning a propagation method
- [x] remove the `propagate_wave` overrides on `Source` (transparent + `wave()`
      is the seed), `Aperture` (now a mask declaration), `Prism` (declares
      "unsupported" instead of overriding propagation)
- [x] rename `waveoptics.propagate_thick_lens_scaled` -> element-agnostic
      (it is a homogeneous quadratic-index *segment*, not a lens).
      `propagate_quadratic_segment_scaled`; `scaled_delta_tau_lens` ->
      `scaled_delta_tau_quadratic`
- [x] per-axis capability **deferred to issue #3 step 3, deliberately**: an
      anisotropic segment needs a per-axis `K` *and* a separable (dtau_x,
      dtau_y) kernel, and its only consumer is the thick quadrupole whose
      transfer block is not yet symplectic. Building it here would be
      speculative and unverifiable; the isotropic-frame refusal already names
      the limit actionably. The rename — which is what actually unblocks #3 —
      is done.
- [x] tests: 86 green (was 84). The pre-existing regression tests that pin
      this seam — `test_fixed_path_refactor_regression` and
      `test_wave_kind_aperture_matches__aperture_wave` — still pass unchanged,
      which is the identical-results proof. New:
      `test_no_element_owns_a_wave_propagation_method` (guards the invariant),
      `test_phase_shift_kind_dispatch`, and
      `test_phase_shift_not_a_phase_elements` rewritten onto the mask seam.
      Scoped to the **wave** seam: `Source`/`Aperture` still own
      `propagate_ray`/`propagate_moments` (a seed and a hard block are not
      matrices) — noted in the test, not fixed here.
- [x] wiki: `elements.md` seam + `Aperture.amplitude_mask` sections rewritten,
      `waveoptics.md` renamed entries; refresh run
