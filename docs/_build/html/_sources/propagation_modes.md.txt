# Propagation Modes

One geometry, four propagations. Every mode runs through the same
element-by-element hierarchy and the same dispatcher,
`scope.propagate(kind=...)`; the mode only changes *what* travels.

| kind | call | what travels | stored on |
|---|---|---|---|
| rays | `propagate_ray()` | geometric rays `(x, x', y, y', z, E)` | `.rays`, with `.I`/`.R` alongside |
| moments | `propagate_moments()` | mean + covariance, `Σ' = MΣMᵀ` | `.mu`, `.covariance_matrix` |
| wave | `propagate_wave()` | 2D complex wavefield on a fixed grid | `.wave` (a sea-eco `Signal`) |
| scaled wave | `propagate_wave(mode="hybrid")` | scaled-Fresnel field `ψ = U(x/s)/s` | `.wave_scaled` (a `SignalSet`) |

## Rays

The workhorse. Each element applies its 6×6 transfer matrix (plus any
nonlinear aberration kick), and one plane is logged per element — so the z
sampling of every result *is* the element list. `Microscope.subdivided(zpts)`
returns a copy with the plain drifts cut finer when you need denser sampling.

Intensity (`I`) and cumulative Larmor rotation (`R`) are not ray columns; they
travel as parallel arrays. `beam_current` (amps) is stated once on the
`Source`/`Gun` and derived everywhere else from the intensities that survive.

## Moments

The same transfer matrices transport a Gaussian envelope instead of individual
rays. Waists (`beam_waists()`) sit beside the geometric image planes,
displaced by the emittance focal shift — that displacement is physics the ray
picture cannot show. Apertures are a documented no-op in this mode.

## Fixed-grid wave

A split-step paraxial propagator on a grid fixed at the source. Exact where
its sampling holds — and full-column electron optics is far outside that
regime; see [the sampling analysis](wave-optics-sampling.md) for why. Use it
near planes of interest, not for whole-column transport.

## Scaled-Fresnel wave (`mode="hybrid"`)

The full-column wave mode. The field is factorized as `ψ = U(x/s)/s` against a
geometric reference frame `(s, R)` that follows the beam, so the grid zooms
with the envelope instead of undersampling it. The hybrid engine switches
frames analytically at crossovers and logs every one; `scope.crossovers`
lists the focal planes of the seed's conjugate family, and `wavefield_at(z)`
reconstructs the physical field at any logged plane.

## They answer for each other

The three descriptions of any solved column agree on where its conjugate
planes are: `conjugate_planes(method='frame')` (closed-form transfer blocks,
the exact reference), `method='ray'` (traced), the hybrid wave's own
crossovers, and the covariance waists. `examples/07_eightConfigurations.py`
prints that cross-check as a table with deltas, and the test suite holds it
to numerical precision.
