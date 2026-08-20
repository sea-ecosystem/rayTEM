# PLAN — Scaled (co-moving grid) Fresnel propagation for the wave mode

**Status:** proposed — not started (no TODO yet; open one when work begins)
**Branch:** TBD (suggest `scaled-fresnel-wave`)
**Related:** `docs/wave-optics-sampling.md` (the problem statement) ·
[issue #1](https://github.com/sea-ecosystem/rayTEM/issues/1) (Collins/ABCD Gaussian
mode — shares the pilot-beam utilities built here)

## Goal

Make `propagate_wave` numerically valid through a full realistic column (e.g.
`basic_column`: 200 kV, µm beam, cm-focal lenses) by removing the quadratic
phase from the sampled field and letting the grid follow the beam. The sampled
array then carries only the *residual* structure (aperture diffraction, and
later aberrations) — which is small-bandwidth everywhere — instead of the
full lens phases that violate the grid Nyquist by 10–100×.

## Physics: pilot beam + residual field

Factor the wavefunction as

```
ψ(x, z) = G(x; q(z), μ(z)) · φ(u, ζ)          u = (x − μ(z)) / m(z)
```

- **`G` is an analytic pilot Gaussian** carried by the ABCD law
  `q' = (Aq+B)/(Cq+D)` (per transverse axis), with centroid `μ(z)` carried by
  the ordinary ray matrices. It absorbs **all quadratic and linear phase**:
  ideal lenses/quads update only `q`, dipole kicks update only `μ` — the
  residual `φ` sees the **identity** for every ideal element.
- **`φ` is the sampled residual** on the dimensionless co-moving coordinate
  `u`, with magnification `m(z) ∝ w_pilot(z)`. By the Talanov/Niederer lens
  transformation, between element planes `φ` obeys *free-space Fresnel
  propagation in scaled coordinates* with effective distance

  ```
  ζ(z) = ∫ dz' / m(z')²
  ```

  → implementable with the **existing `angular_spectrum_propagate`** on the
  u-grid (unit pitch `du`, effective distance `ζ`). No new FFT machinery.
- **Non-quadratic elements** act multiplicatively on `φ` at their plane:
  an `Aperture` of radius `r` masks at scaled radius `r/m(z)`; later,
  aberration phases `χ` are applied the same way. `Prism` stays
  `NotImplementedError`.

**Why the grid never breaks:** `m(z)` tracks the pilot width, whose *minimum is
the diffraction-limited waist* — never zero. At a crossover the co-moving grid
automatically zooms to the focused-spot scale (nm), at the detector it expands
to the image scale (100 µm), with a constant number of samples. The ~10⁶
zoom range that killed the fixed grid becomes bookkeeping in `m(z)`.

## Design

### New pieces (mostly `waveoptics.py` + a small pilot struct)

1. **Pilot-beam utilities** *(shared with issue #1 — build once)*
   - `q` initialization from `Source` (`wavelength`, `size`); per-axis
     (`qx`, `qy`) for astigmatism.
   - ABCD update from an element's `transfer_matrix()` blocks (reuse the
     block-extraction pattern from `propagate_moments`).
   - Centroid `μ` update via `propagate_ray` on a single ray.
   - Accumulated Gouy phase + `m(z) = w(z)/w₀` per axis.
2. **Scaled drift step** — `ζ` accumulation per drift/element length and a
   residual step `φ → angular_spectrum_propagate(φ, du, du, λ_eff, Δζ)`
   (equivalently fold λ into ζ; pick one convention and document it).
3. **Element mapping** in `Element.propagate_wave(..., method="scaled")`:
   quadratic elements → pilot-only update; `Aperture` → mask `φ` at `r/m`;
   thick-lens Larmor rotation → rotate the residual grid (or track a frame
   angle on the pilot, applied on reconstruction — decide during
   implementation; rotation is sampling-safe on the scaled grid).
4. **Reconstruction** — `ψ(x) = G·φ/√(mₓ·m_y)` evaluated on demand to return a
   physical-coordinate wavefield Signal at any requested plane.

### API (decision for Eric — pick one)

- **(A) `propagate_wave(method="scaled")`** with `method="fixed"` the current
  behavior *(recommended: one entry point, dispatcher untouched)*, or
- (B) a separate `kind="wave-scaled"` in the `propagate(kind=...)` dispatcher.

### Storage (decision for Eric — pick one)

The current `.wave` contract (one `(Nz,Ny,Nx)` Signal with a single shared
transverse calibration) cannot hold per-plane pixel sizes `dx(z) = m(z)·dx₀`:

- **(A) Scaled-coordinate stack** *(recommended)*: keep one `(Nz,Ny,Nx)` Signal
  of the residual (or reconstructed-in-u) field with axes in `u` (dimensionless
  or units of `w(z)`), and store `m(z)`, `μ(z)`, `q(z)`, Gouy in metadata /
  companion Signals; provide `wavefield_at(z)` to reconstruct the calibrated
  physical-coordinate 2D Signal for any plane.
- (B) A `SignalCollection` of per-plane Signals, each with its own calibration.
- (C) Resample every plane onto the detector-plane grid (lossy at crossovers).

## Implementation steps (unchecked — copy into a TODO when starting)

- [ ] Pilot utilities: `q₀` seeding, per-axis ABCD update, centroid, Gouy, `m(z)` (+ unit tests vs analytic Gaussian formulas)
- [ ] Scaled drift step: `ζ` accumulation + residual angular-spectrum step (+ test: pilot×residual reproduces direct fixed-grid propagation in a regime where the fixed grid is valid)
- [ ] Element mapping: lens/quad → pilot; dipole → centroid; aperture → scaled mask; Larmor rotation handling
- [ ] Driver + API per decision (A/B above); dispatcher passthrough
- [ ] Storage per decision (A/B/C above); `wavefield_at(z)` reconstruction
- [ ] Validation (see below); golden tests in `test_wave_and_envelope.py`
- [ ] Update `docs/wave-optics-sampling.md` (move scaled Fresnel from "roadmap" to "implemented", add the residual-bandwidth validity criterion); wiki refresh

## Validation

1. **Pure Gaussian through `basic_column`**: residual must stay ≡ 1 (machine
   precision) — every element is absorbed by the pilot; `w(z)` from the pilot
   must match `propagate_moments` seeded at the diffraction limit.
2. **Aperture diffraction**: mid-scale case (valid on a fine fixed grid) —
   scaled result vs direct fixed-grid reference; Airy-ring positions match.
3. **Full column end-to-end**: `basic_column` with the condenser aperture in —
   detector-plane |ψ|² and its qx–qy plane are finite, alias-free, energy-conserving
   (`∫|ψ|²` constant to <1e-6 across all planes).
4. **Waist behavior**: reconstructed spot at the sample plane ≈ pilot waist
   (nm-scale), where the fixed grid produced garbage.

## Risks / notes

- Astigmatism makes `mₓ ≠ m_y` (separable — handled per axis); a rotated
  residual on an anisotropic grid needs care (defer strong quad + rotation
  combined cases; test coverage decides).
- `ζ` accumulation through *thick* lenses: integrate `dz/m²` through the lens
  interior (piecewise, using the pilot's in-lens evolution) or treat the lens
  as thin for the residual with the drift split around it — start with the
  latter, document.
- Incoherence is still out of scope: this transports one coherent field;
  the covariance mode remains the tool for the full (partially coherent) beam.

## References

- Talanov, V. I., "Focusing of light in cubic media," *JETP Lett.* **11**, 199 (1970) (lens/scaling transformation).
- Niederer, U., "Maximal kinematical invariance group of the free Schrödinger equation" (1972) (the underlying symmetry).
- Krist, J., "PROPER: an optical propagation library," *Proc. SPIE* **6675** (2007) (pilot-beam / two-step Fresnel practice).
- Goodman, *Introduction to Fourier Optics*, ch. 5–6 (two-step Fresnel, sampling).
