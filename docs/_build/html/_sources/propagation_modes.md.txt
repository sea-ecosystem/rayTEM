# Propagation Modes

One geometry, four propagations. Every mode runs through the same
element-by-element hierarchy and the same dispatcher,
`scope.propagate(kind=...)`; the mode only changes *what* travels.

| kind | call | what travels | stored on |
|---|---|---|---|
| rays | `propagate_ray()` | geometric rays `(x, x', y, y', z, E)` | `.rays`, with `.I`/`.R` alongside |
| moments | `propagate_moments()` | mean + covariance, `Σ' = MΣMᵀ` | `.mu`, `.covariance_matrix`, `.covariance_beam` |
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

Aberrations enter this mode **analytically**. Terms linear in the ray vector
(C10, aligned C12) are power changes and fold into the matrix exactly.
Everything else is a genuinely nonlinear kick, and the mode handles it by
recovering that kick as a *polynomial* (`Element.aberration_monomials`, from
the same `deflection_at` the ray path uses) and taking its moments through an
explicit **moment closure**. Every Krivanek order the pupil carries is
covered, rotated terms included — there is no per-aberration code on this path
either.

### The closure is a stated assumption, not a hidden one

A covariance carries only two moments; a nonlinear map needs more. What
supplies them is a `moments.MomentClosure` object, passed to
`propagate_moments(closure=...)` and defaulting to
`moments.GaussianMomentClosure`, which evaluates any central moment by Wick
pairing (Isserlis).

Using it does **not** make the beam Gaussian. It is exact for one aberrated
element acting on a Gaussian beam; it becomes an approximation only when a
*second* nonlinear element acts on what the first one distorted. That
approximation is auditable, because a cubic kick leaves excess kurtosis
`γ₂ = 27f²` where `f` is the aberration's share of the angular variance — the
same number that says the aberration matters says how non-Gaussian the beam
became. `examples/08_covariancePropagation.py` prints `f` at each aberrated
element rather than assuming it is small.

The closure is complete rather than truncated, which is why the result is
always a physically possible covariance: it is the exact pushforward of a
Gaussian, so positive semidefiniteness is guaranteed for any aberration
strength. A *partial* closure — keeping the in-plane terms and dropping the
cross-plane ones, as an earlier per-axis version did — loses that guarantee,
and on an x-y coupled beam the dropped terms are the same size as the kept
ones.

### Chromatic

Chromatic aberration lives here too, and it is a different kind of term. The
Krivanek `C_{n,m}` are functions of pupil coordinate alone; chromatic couples
the pupil to the **energy column the state vector already carries**, so the
kick is bilinear — `δθ = C_c P² δ x` with `δ = (E − E₀)/E₀` — and cannot be a
matrix entry, even though it is "just a power change", because the power
change differs per electron.

It is declared *inside* the aberration set, as the `'Cc'` term:
`Lens(aberrations={'C30': 4.5e-6, 'Cc': 1.2e-3})`. One declaration then carries
everything the element does beyond its matrix — it serializes with the
aberrations, is detached with them by `apply_aberrations=False`, and cannot be
left behind when they are cleared. `Element.chromatic_aberration` is a
convenience view onto it. Pair it with `Source.energy_spread`; either alone
leaves the column achromatic.

Unlike the Krivanek terms it needs no round-pupil assumption, so it is applied
**per axis** and is exact on an astigmatic element: a quadrupole's two powers
have opposite signs and each gets its own `ΔP_u = −C_c P_u² δ`.

Its covariance contribution `ΔΣ_θθ = κ²σ_δ²σ_x²` needs only a fourth moment
that factorizes when the energy spread is independent of position, so unlike
the geometric terms it is **exact rather than closed**.

### Reading the result

`scope.covariance_beam` is a `moments.CovarianceBeam` view over the stored
`.mu`/`.covariance_matrix`, and reports what a covariance is actually read
for: rms widths, the signed position-angle correlations, the emittances, and
the principal axes of the real-space, angular and transverse-momentum blocks.
Prefer emittance to width when judging aberration damage — a width says
nothing on its own, while emittance is invariant under ideal transport and
grows only where something nonlinear acted.

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
