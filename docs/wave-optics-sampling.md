# Why full-column electron wave propagation breaks fixed-grid sampling

*(rayTEM developer note — a candidate first page for the "Into the SEA-weeds"
developer docs. Numbers below use the `basic_column` template: 200 kV,
λ = 2.508 pm, beam entering the condenser with σₓ = 2.5 µm and σθ = 0.1 mrad.)*

## Context: what `propagate_wave` does

rayTEM's wave mode is a **fixed-grid, split-step paraxial propagator**: the
wavefunction lives on an `N×N` grid of pitch `dx` (the `basic_column` default is
`W = 20 µm`, `N = 256`, so `dx = 78 nm`), lenses/deflectors act as **thin phase
screens** derived from the same 6×6 transfer matrices as ray tracing, and drifts
are **angular-spectrum (Fresnel) FFT steps**. The grid is fixed once at the
source and reused at every plane.

Any sampled representation of a wave has one hard rule: the local phase must
not change by more than π between neighbouring samples — the grid Nyquist limit
`|∂φ/∂x| < π/dx`.

## The two sampling criteria

**1. Drifts (angular spectrum) — cheap.** The free-space transfer function
`H = exp(−i k⊥² Δz / 2k)` is adequately sampled while

```
Δz  ≲  N·dx² / λ
```

For the current grid that is `256 × (78 nm)² / 2.5 pm ≈ 0.62 m` — comfortably
longer than any drift in the column. **Free-space propagation is not the
problem.**

**2. Lenses (phase screens) — the killer.** A lens of focal length `f` imprints
`φ(x) = −k x² / 2f`, whose gradient at the beam edge is `k·x_max/f`. Sampling it
requires

```
dx  <  π f / (k · x_max)        (equivalently  N > W² k / (2π f) )
```

With `k = 2π/λ = 2.5×10¹² rad/m` and `x_max = 10 µm`:

| lens | f | phase gradient `k·x_max/f` | grid Nyquist `π/dx` | required `dx` | required `N` (per side) |
|---|---|---|---|---|---|
| C1 | 45 mm | 5.6×10⁸ rad/m | 4.0×10⁷ rad/m | < 5.6 nm | ≥ 3 500 |
| OL1 | 8 mm | 3.1×10⁹ rad/m | 4.0×10⁷ rad/m | < 1.0 nm | ≥ 20 000 |

The current 256² grid undersamples the lens phase by **14–78×**. An undersampled
phase screen does not degrade gracefully — it aliases, scattering intensity into
spurious diffraction orders, so everything downstream of the first strong lens
is unphysical.

## Why this is fundamental, not an implementation bug

The grid size demanded by a wave calculation is the **space–bandwidth product**
of the field it must carry:

```
N  ≳  (spatial extent) × (angular extent) / λ      (per axis)
```

Because λ at 200 kV is picometres while beams are micrometres wide and
convergence angles are milliradians, this product is enormous for a real
column:

- The **source alone** (±4σ: 20 µm × 0.8 mrad) needs ≈ 6×10³ resolvable cells
  per axis.
- After a strong lens the angular content grows to `x_max/f` (~1 mrad at OL1),
  and the focused spot shrinks toward `λf/W` (~nm) while the field envelope is
  still tens of µm — the same grid must then resolve nm structure over µm
  extents: `N ~ 10⁴–10⁵` per side *per plane*, with the natural feature size
  zooming by ~10⁶ between a crossover and an image plane.
- The beam is also not one wave: with emittance `σₓσθ = 2.5×10⁻¹⁰ m·rad` it
  contains `σₓσθ / (λ/4π) ≈ 1 250` mutually incoherent transverse modes per
  axis. A single coherent field is the wrong object for the full beam in the
  first place — the covariance mode is the correct transport for that.

This is why **no electron-optics code propagates a sampled wavefunction through
a whole column at true scale**. Field practice splits the problem:

- **Column transport** → matrices (rays / Gaussian moments), exactly what
  `propagate_ray` and `propagate_moments` do at any scale.
- **Wave physics** → applied only where λ matters and the field of view is
  small: multislice through the specimen (nm FOV), the aberration function
  χ(q) applied in the **pupil/diffraction plane** (where the support is mrad,
  not µm), probe formation over 10–100 nm windows.

## What is valid in rayTEM's wave mode today

Use the wave mode when both criteria hold on your grid:

1. every drift satisfies `Δz ≲ N·dx²/λ`, and
2. every phase screen satisfies `dx < πf/(k·x_max)` (aperture edges: the mask
   is binary, so criterion 1 near the aperture governs).

In practice: source-plane fields and their Fourier (qx–qy) planes are exact;
short, weak-lens toy columns work end-to-end (the test suite verifies focusing
at `z = f` and Fresnel Gaussian spreading); the realistic `basic_column`
template is valid at the source plane but not through its cm-focal-length
lenses.

## Remedies (roadmap)

1. **Collins / ABCD Gaussian-mode transport (grid-free, exact).** Propagate the
   complex beam parameter `1/q = 1/R − i λ/(π w²)` through the *same* transfer
   matrices via `q' = (A q + B)/(C q + D)`. No grid → no sampling limit; yields
   `w(z)`, wavefront curvature `R(z)`, Gouy phase, and the full complex Gaussian
   field analytically at any plane. Note what it adds over the existing modes:
   rays and covariance are λ-free (a covariance beam seeded at the diffraction
   limit `σₓσθ = λ/4π` reproduces the Gaussian **widths** exactly, but carries no
   phase); the q-parameter adds the wave content — curvature, Gouy phase,
   complex field — while remaining exact in any linear paraxial system.
2. **Co-moving / scaled coordinates.** Factor the quadratic reference phase out
   analytically (Fresnel scaling / two-step propagators) and sample only the
   residual field on a grid that zooms with the beam. Standard in light-optics
   codes; removes the lens-phase criterion at the cost of bookkeeping.
3. **Hybrid transport (the field-standard pattern).** Use matrices for the
   column, and attach sampled-wave patches only at planes of interest — e.g.
   apply the aberration phase in the diffraction plane and a wave calculation
   across the specimen window.

## References

- Collins, S. A., "Lens-System Diffraction Integral Written in Terms of Matrix
  Optics," *J. Opt. Soc. Am.* **60**, 1168 (1970).
- Siegman, A. E., *Lasers*, ch. 20 (ABCD law for Gaussian beams).
- Goodman, J. W., *Introduction to Fourier Optics*, ch. 5 (sampling of
  quadratic phases and the space–bandwidth product).
- Matsushima, K. & Shimobaba, T., "Band-limited angular spectrum method,"
  *Opt. Express* **17**, 19662 (2009) (the drift criterion).
