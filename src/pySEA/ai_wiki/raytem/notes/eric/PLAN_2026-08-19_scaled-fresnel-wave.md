# PLAN — Scaled Fresnel propagation (`propagate_wave_scaled`)

**Status:** approved, in progress
**TODO:** [TODO_ACTIVE_scaled-fresnel-wave.md](TODO_ACTIVE_scaled-fresnel-wave.md)
**Branch:** `Signal_and_propagation_additions`
**Supersedes:** an earlier pilot-Gaussian/ABCD draft of this file — rejected;
the implemented design is Eric's scaled-Fresnel handoff (pure wave-phase
factorization, no ABCD construction).

## Problem

The fixed-grid wave mode cannot cross a real column: lens phases exceed the
grid Nyquist by 10–100× and the beam's transverse scale changes by ~10⁶
between planes (`docs/wave-optics-sampling.md`).

## Representation (Eric's handoff)

Factor the paraxial wave (carrier `e^{ikz}` excluded):

```
ψ(x,y,z) = (1/s)·U(ξ,η,τ)·exp[ik(x²+y²)/(2R)] ,  ξ = x/s ,  1/R = s′/s ,
τ = ∫ dz/s²
```

- Free segment: linear `s(z) = s₀[1+(z−z₀)/R₀]`, `R(z) = R₀+(z−z₀)`,
  `Δτ = Δz / (s₀²[1+Δz/R₀])` (verified algebraically + numerically in tests);
  U Fresnel-propagates over Δτ with the existing angular-spectrum kernel
  (carrier stripped).
- Ideal thin lens: `U⁺ = U⁻`, `1/R⁺ = 1/R⁻ − 1/f`, `s` continuous.
- Reconstruction at any plane: `ψ = (1/s)·U(x/s, y/s)·exp[ikr²/2R]`; physical
  pixel `Δx = |s|Δξ`. Target-grid reconstruction via Fourier band-limited
  resampling. The reconstructed physical wave (a calibrated wavefield Signal)
  is the boundary handed to any external multislice package.
- `s` must never cross 0: `|s| > s_min` guard raises an actionable error;
  crossover chart-switching is a follow-up (GitHub issue to be filed).

## Design in rayTEM

- **`Element.phase_shift(dimensions, wavelength, scaled=False, s=1)`** — each
  element class states its wave physics explicitly (no matrix derivation):
  Lens `−k(x²+y²)/2f`; Quadrapole `−k(x²−y²)/2f_q` (one axis focuses, the
  other diverges); Dipole `k(θₓx+θᵧy)`; Drift the reciprocal-space kernel.
  Returned as real phase Signals whose Dimension `space` tags the domain
  (`position` = screen, `scattering` = FFT-domain). `scaled=True` returns
  (power absorbed into R, phase applied explicitly to U at `x = s·ξ`):
  Lens → (1/f, None); Quad/Dipole → (0, full phase, sampling-guarded);
  Drift → free-segment updates. The fixed `propagate_wave` is refactored to
  consume `phase_shift(scaled=False)` (thin-lens behavior unchanged).
- **No bespoke state class** — Δξ/Δη live on the U Signal's Dimensions;
  `s, R, tau, z, wavelength` in metadata (in flight) and as companion Signals
  in the `.wave_scaled` **SignalSet** (stacked result; U grid is fixed in
  ξ,η so one stack works). New seashells factories:
  `make_scaled_wavefield_signal`/`read_scaled_wavefield`/
  `make_scaled_wave_signalset` (+ fallbacks).
- **Drivers**: `propagate_wave_scaled` on Element/Section/Microscope (same
  plane-logging + eager re-chain as other modes); `Microscope.wavefield_at`
  reconstructs the physical wave at a logged plane; dispatcher
  `kind="wave-scaled"`; `Source.scaled_field()` + `Source.aperture_field(r)`
  (hard-aperture initial wave Θ(a−r)).

## Validation

Handoff test ladder: Δτ closed form vs numeric; factor/reconstruct identity;
free-prop and thin-lens equivalence vs the ordinary propagator; the
aperture→free→lens→free system at several planes; grid scaling Δx=|s|Δξ;
normalization Σ|ψ|²ΔxΔy = Σ|U|²ΔξΔη; entrance-plane equivalence on a target
grid; 200 kV / 20 µm / f=45 mm electron-scale invariants (the case the fixed
grid cannot sample); column-driver equivalence + full regression suite.

## Out of scope (phase 1)

Partial coherence; chromatic + high-order aberrations (hook: extra phase terms
on U); anisotropic s/R (strong quads); crossover chart switching (follow-up
issue); post-specimen scaled propagation.
