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

## Scaled Fresnel propagation (implemented): `propagate_wave(mode='scaled'|'hybrid')`

The co-moving / scaled-coordinates remedy is now implemented inside the one
wave method: `propagate_wave(..., mode='scaled')` for a single scaled frame and
`mode='hybrid'` for automatic frame switching through crossovers (dispatcher
kinds `"wave-scaled"` / `"wave-hybrid"`). The wave is factored as

```
ψ(x, y, z) = (1/s(z)) · U(ξ, η, τ) · exp[i k (x² + y²) / 2R(z)],
ξ = x/s,  η = y/s,  1/R = s′/s,  τ = ∫ dz′/s²
```

so the reduced field `U` obeys the ordinary paraxial equation in `(ξ, η, τ)`
and is propagated by the *same* angular-spectrum kernel over `Δτ`
(carrier-free). What this buys:

- **Lens phases never touch the grid.** A round lens updates only the
  curvature state, `1/R⁺ = 1/R⁻ − 1/f` (`U⁺ = U⁻`), so the lens-phase
  sampling criterion above disappears entirely — cm-focal-length electron
  lenses are exact.
- **The grid zooms with the beam.** Through a free segment `s` evolves
  linearly, `s(z) = s₀[1 + Δz/R₀]`, and the physical pixel is always
  `Δx = |s|·Δξ`, so a converging beam is followed at constant relative
  resolution. `Δτ = Δz / (s₀²[1 + Δz/R₀])` in closed form (verified against
  the numerical integral in the test suite).
- **Thick lenses are exact too, and cost U nothing.** A thick round lens is a
  quadratic-index *medium*, not a screen: in the scaled frame it is one
  segment whose scale law is sinusoidal, `s(z) = s₀cos(Kz) + (u₀/K)sin(Kz)`,
  with its own closed-form `Δτ = [tan(KL − φ) + tan φ]/(K·C²)`
  (`C² = s₀² + (u₀/K)²`, `tan φ = u₀/(K s₀)`). U takes no screen and no kick —
  it just propagates over that Δτ. Thin lenses (`length == 0`) keep the
  curvature-kick route. This matters: treating a thick lens as *drift L/2 →
  kick → drift L/2* misplaced every crossover on `basic_column` by 422–4808 µm,
  and the exact segment puts them on the ray-traced diffraction planes to
  0.0 µm. The one limitation is a crossover landing *inside* a lens body,
  where this frame is singular and the code raises rather than guessing —
  mid-element frame switching is not implemented.
- **Larmor rotation is available** (`propagate_wave(..., rotate=True)`): each
  thick lens rotates the field by `−K·L`, the same angle the ray path applies,
  via an exact unitary three-shear Fourier rotation. Off by default because it
  is analytically a no-op for rotationally symmetric fields, where it would
  only add resampling noise.
- **Non-quadratic phases still apply to U** (quadrupole saddle, dipole tilt,
  future aberrations), evaluated at physical coordinates `x = s·ξ` and
  protected by a per-pixel `|Δχ| < π` sampling guard — stigmator-scale
  strengths pass; over-strong settings fail loudly instead of aliasing.
- **Plotting through the standard `show`.**
  `scope.show(kind='wave-hybrid')` draws the |ψ(x, y=0, z)| cross-section —
  the wave analog of the ray diagram, with element and crossover overlays —
  and `show(kind='wave-hybrid', plane=...)` images one plane (a logged-plane
  index, a z in metres, a named position, or a `scope.crossovers[i]` focal
  plane) by delegating to the reconstructed wavefield Signal's own `.show()`.
  A result's z sampling is one plane per element exit plus the hybrid run's
  frame events, so `show(..., zpts=5e-3)` (or `scope.subdivided(5e-3)`)
  propagates a temporary copy with finer drift steps when a smooth
  cross-section is wanted. Crossovers need no subdivision — see below.
- **Both families of conjugate planes.** A column has two independent
  conjugate families: **diffraction** (back-focal) planes, where rays that
  entered parallel converge, and **image** planes, where rays from one on-axis
  point re-converge. The scaled frame follows exactly one — whichever its seed
  belongs to, which for the usual flat-wavefront seed (`s=1, R=∞`) is the
  diffraction family, so that is what `scope.crossovers` holds.
  `scope.conjugate_planes(axis)` returns **both** in metres (reusing the ray
  side's `findPlanes` convention on a throwaway copy), the cross-section
  annotates the image planes alongside the crossovers, and
  `zpts=scope.conjugate_planes()['image']` logs the image planes exactly for
  reconstruction. Neither family follows from one lens's `f` — the frame's
  accumulated curvature carries the whole upstream system (verified: a
  collimated beam through f=45 mm then f=30 mm at 100 mm spacing puts the
  second diffraction plane at 176 mm, the image of the first crossover, not at
  `z_L2 + f2` = 140 mm).
- **Reconstruction back to physical x, y** at any logged plane:
  `Microscope.wavefield_at(z_or_name, target_dx=..., target_shape=...)`
  (crossover planes included — the back-focal wavefield of each lens)
  returns a standard calibrated wavefield Signal — on the native `|s|·Δξ`
  grid, or band-limited-resampled onto a prescribed grid for an external
  package (e.g. multislice) to consume.

State lives in the sea_eco architecture: `Δξ/Δη` on the ξ/η `Dimension`
calibration, the frame scalars `(s, R, τ)` in metadata (single plane) and as
companion Signals in the `.wave_scaled` `SignalSet` (stacked result, sharing
the plane-z axis with the U stack).

**Crossovers — solved by frame switching (`mode='hybrid'`).** A *frame* is a
factorization choice (s, R, τ); a converging frame is singular where `s → 0`
(the beam crossover) — the reference wavefront collapses to a point while the
diffracted wave stays finite. `propagate_wave(mode='hybrid')` switches frames
automatically (`waveoptics.change_scaled_frame`, the general re-expression of
the same physical wave in another frame): the converging frame **flattens**
where its reference curvature first becomes representable on the shrinking
grid (`|R_flat| = R²/(A·s²)`, a frame invariant with a closed-form split
point), the wave crosses the real focus by ordinary carrier-free Fresnel
propagation — the crossover (back-focal / diffraction) plane is logged and
listed on `Microscope.crossovers` (the focus position is the closed-form
`z_cross = z + |R|`, not a numerical search, so the engine splits its
propagation exactly there and the focal plane is always logged exactly) — and
re-factors onto a diverging frame at
the mirror-image distance past it. One ξ/η calibration serves the entire
column while `s(z)` dips and recovers at each focus; the `basic_column`
template runs source → detector (five crossovers) with energy conserved at
every plane and the physical pixel spanning sub-nm (foci) to µm (detector).
`mode='scaled'` keeps the single-frame behavior (an actionable error before
the crossover); `s_min` remains as a backstop guard. Switch planes and the
frame-change sampling guard are measured at the **beam's support** (the
bounding half-width where |U| > 10⁻⁶·max), not the grid edge — phase applied
where the field is essentially zero is harmless, so a beam smaller than its
grid flattens earlier (larger s, larger pixels) and padded grids cannot trip
the backstop. A `crossover='jump'` policy is also available (direct
converging → diverging mirror frame change, no flat window); measured on the
electron-scale column it rides the converging frame twice as deep and loses
the beam when the focal structure (Airy/s in ξ) outruns the field of view, so
`'flat'` stays the default — use `'jump'` only for mild crossovers.

**Apertures are sampled alias-free.** The physical model stays the sharp disk
Θ(a−r) (Fresnel edge diffraction is real and kept); the grid holds its
**band-limited projection**, synthesized from the analytic J₁ spectrum
(`waveoptics.bandlimited_disk`), so the edge's above-Nyquist content no longer
folds back and propagates as a spurious axis-aligned "gridding" texture.
Mid-column `Aperture` masks use the 1-pixel area-coverage edge for the same
reason (`antialias=False` restores binary masks). The second half of that
texture was periodic-FFT wraparound of the edge halo: free segments now carry
an **absorbing boundary** (`absorb=0.1` margin, raised-cosine window applied
between τ sub-steps) — electrons that diffract out of the modeled field of
view are removed, as they are physically lost from the beam, so total |U|²
honestly decreases by what leaves. The window is **radially symmetric** (it
ends at the inscribed circle): a square window clips the halo anisotropically
— corners √2 farther out than edges — and the fourfold-modulated surviving
halo interferes back into the beam as a pixel-axis-aligned fringe pattern at
every downstream plane (fourfold harmonic c4 ≈ 1e-3 square vs ≈ 0 radial at
the basic_column sample and detector; only the isotropic concentric Fresnel
rings remain).

**Anisotropic frames (s_x ≠ s_y).** The frame generalizes per axis,
`ψ = (s_x s_y)^{-1/2}·U(x/s_x, y/s_y)·exp[ik(x²/2R_x + y²/2R_y)]`, so a
**quadrupole absorbs its (P, −P) powers into (R_x, R_y)** exactly like a
round lens absorbs one power into R — no saddle phase screen, no sampling
limit, arbitrarily strong stigmators run. Each axis flattens, crosses, and
re-diverges at its own **line focus** (hybrid tags `flatten-x`,
`crossover-y`, …; both listed on `Microscope.crossovers`), the seam stores
per-axis metadata (`s_x`/`s_y`, `R_x_m`/`R_y_m`, `tau_x`/`tau_y`) and
per-axis SignalSet companions only when the axes differ, and reconstruction
uses rectangular native pixels (dx = |s_x|Δξ, dy = |s_y|Δη). Frame
quantities travel as scalars wherever the axes agree, so round-lens columns
keep their scalar state and outputs bit-for-bit.

## Other remedies (roadmap)

1. **Collins / ABCD Gaussian-mode transport (grid-free, exact).** Propagate the
   complex beam parameter `1/q = 1/R − i λ/(π w²)` through the *same* transfer
   matrices via `q' = (A q + B)/(C q + D)`. No grid → no sampling limit; yields
   `w(z)`, wavefront curvature `R(z)`, Gouy phase, and the full complex Gaussian
   field analytically at any plane. Note what it adds over the existing modes:
   rays and covariance are λ-free (a covariance beam seeded at the diffraction
   limit `σₓσθ = λ/4π` reproduces the Gaussian **widths** exactly, but carries no
   phase); the q-parameter adds the wave content — curvature, Gouy phase,
   complex field — while remaining exact in any linear paraxial system.
   (Tracked as GitHub issue #1.)
2. **Hybrid transport (the field-standard pattern).** Use matrices for the
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
- Sziklas, E. A. & Siegman, A. E., "Mode calculations in unstable resonators
  with flowing saturable gain. 2: Fast Fourier transform method,"
  *Appl. Opt.* **14**, 1874 (1975) (the coordinate-scaling transform behind
  `propagate_wave(mode='scaled'|'hybrid')`).
