# Plan — covariance propagation and aberration resolution example

Adapted from Eric's handoff plan of 2026-08-30. Two adaptations were directed
explicitly and are marked **[adapted]** and **[added]** below:

- **[adapted]** the source plan writes the transverse state as a 4×1 vector
  `(x, y, u_x, u_y)`. rayTEM's state is the 6-column
  `convention = ["x","xt","y","yt","z","E"]`, so the angles are `xt`/`yt` and
  the transverse block is interleaved `(x, xt, y, yt)`, not position-then-angle.
  Every equation below is restated in rayTEM's ordering. Σ is already 6×6 over
  those columns — no separate 4×4 object is introduced.
- **[added]** chromatic aberration. The source plan defers it ("before adding
  Midgley geometry, chromatic effects, ..."); Eric pulled it into scope. It
  gets its own section, its own equations, and its own cases.

Linked TODO: `TODO_ACTIVE_covariance-propagation.md`.

---

## Purpose

Add and validate a covariance-based beam propagation mode for the basic rayTEM
column.

The goal is not to reproduce individual ray trajectories. The calculation
begins from a beam described by its first and second moments, propagates those
moments through the microscope, and quantifies how pre-specimen and
post-specimen aberrations change the beam resolution.

The example uses the existing basic column and compares four configurations:

1. ideal column, no aberrations,
2. pre-specimen OL1 aberrations only,
3. post-specimen OL2 aberrations only,
4. OL1 and OL2 aberrations together,

plus **[added]** a chromatic overlay on each (see "Chromatic" below).

Midgley defocus is explicitly out of scope for this first example. Use the
ordinary focused specimen configuration so the effect of aberrations can be
isolated before adding the position–angle correlations of Midgley operation.

The example should determine what changes to the covariance implementation are
actually required rather than designing a large new framework in advance.

## Core physical representation **[adapted]**

The source plan's `η_⊥ = (x, y, u_x, u_y)ᵀ` becomes rayTEM's state vector

```
r = (x, xt, y, yt, z, E)ᵀ                                              (1)
```

where `xt`, `yt` are the paraxial angles (the source plan's `u_x`, `u_y`) and
`z`, `E` are the existing longitudinal-position and energy columns. The
transverse block is the interleaved `(x, xt, y, yt)` sub-ordering — index it
through `columnByName`, never by hard-coded integers.

The beam centroid is `μ = ⟨r⟩` (2), and the covariance is

```
Σ = ⟨(r − μ)(r − μ)ᵀ⟩                                                  (3)
```

For a centered source `μ_⊥ = 0`, so `Σ_⊥ = ⟨r_⊥ r_⊥ᵀ⟩` (4). Written out in
rayTEM's ordering, the transverse block is

```
        ⎛ σ_x²     σ_x,xt   σ_xy     σ_x,yt  ⎞
Σ_⊥  =  ⎜ σ_x,xt   σ_xt²    σ_y,xt   σ_xt,yt ⎟                         (5)
        ⎜ σ_xy     σ_y,xt   σ_y²     σ_y,yt  ⎟
        ⎝ σ_x,yt   σ_xt,yt  σ_y,yt   σ_yt²   ⎠
```

Notation rules, kept throughout:

- diagonal elements are variances, e.g. `Σ[x,x] = σ_x²`;
- off-diagonal elements are covariances, e.g. `Σ[x,xt] = σ_x,xt`;
- do **not** write an off-diagonal covariance as `σ_x,xt²`.

## Source and gun boundary condition

Begin from physically meaningful gun/source properties, not an arbitrary
covariance matrix. At minimum: beam energy, transverse source size (waist),
transverse angular spread, emittance, optional x–y asymmetry, and
**[added]** energy spread.

For an uncoupled, axisymmetric source, `Σ₀` is diagonal (6) with
`σ_x0 = σ_y0` and `σ_xt0 = σ_yt0` for the symmetric case (7). This is what
`Source.moments()` already builds from `size` and `angle`.

The transverse rms emittances are

```
ε_x² = σ_x² σ_xt² − σ_x,xt²                                            (8)
ε_y² = σ_y² σ_yt² − σ_y,yt²                                            (9)
```

At an uncorrelated waist, `ε_x = σ_x0 σ_xt0`, `ε_y = σ_y0 σ_yt0` (10).
Treat emittance as a primary source specification **and** as the invariant
that checks ideal linear transport.

Do not construct a ray fan.

## Linear transport

For every ideal linear section `r_out = M r_in` (11), so

```
Σ_out = M Σ_in Mᵀ                                                     (12)
```

is exact. This covers free propagation, ideal focusing, magnification, Larmor
rotation, and the existing paraxial thick-lens matrices. Reuse rayTEM's
existing `transfer_matrix` throughout — do not introduce a parallel set of
covariance-only optics.

The focal terminology matters here: aberration pupil scaling uses
`focal_power`; `back_focal_distance` is an exit-plane geometric distance and
must not be used as a pupil scale. See `docs/terminology.md`.

## Nonlinear aberrations

Do not add an "aberration variance" directly to Σ. An aberration changes the
transport *map*:

```
r'_i = M_ij r_j + ½ T_ijk r_j r_k + ⅙ U_ijkl r_j r_k r_l + ...        (13)
```

with the output covariance defined the usual way (14). The implementation fact
that follows: nonlinear maps require moments above second order — a quadratic
map already needs `⟨r_i r_j r_k⟩` and `⟨r_i r_j r_k r_l⟩` (15). Mean and
covariance alone therefore cannot propagate an arbitrary non-Gaussian beam
exactly through nonlinear aberrations. That is a mathematical limitation of
covariance-only propagation, not an implementation deficiency.

## Avoid hard-coding a Gaussian assumption

The covariance object must not make "Gaussian beam" an invisible assumption.
Separate the beam state `{μ, Σ}` (16) from the **moment closure model** (17).

Gaussian closure is acceptable for the first implementation. For a centered
Gaussian, `⟨r_i r_j r_k⟩ = 0` (18) and

```
⟨r_i r_j r_k r_l⟩ = Σ_ij Σ_kl + Σ_ik Σ_jl + Σ_il Σ_jk                 (19)
```

which lets nonlinear contributions be evaluated from Σ alone. Implement it as
an *explicit* closure:

```
CovarianceBeam            MomentClosure (base)
    mean                      └── GaussianMomentClosure
    covariance
    moment_closure
```

rather than making all covariance propagation intrinsically Gaussian. That
leaves room for other analytic closures or explicit higher moments later.
No ray representation is required.

**Integration constraint (rayTEM-specific):** the existing transport primitive
is `Element.propagate_moments(mu, Sigma) -> (mu, Sigma)` and the drivers on
`MicroscopeSection`/`Microscope` store `.mu` and `.covariance_matrix`. The new
abstraction must be **additive** — the `(mu, Sigma)` tuple contract keeps
working unchanged, with the closure injected as an optional argument and
defaulting to `GaussianMomentClosure`.

## Linear vs aberrated propagation

Linear propagation of Σ is exact for any underlying distribution. Gaussian
closure is needed only where a nonlinear element requires moments that are not
stored. So `linear → linear → linear` is exact at the covariance level, and
`Gaussian source → OL1 aberration → linear transport` is exact at OL1 to the
retained polynomial order if the initial Gaussian assumption is accepted.

The real approximation appears when a later nonlinear element acts on a
distribution an earlier nonlinear element already made non-Gaussian, but which
is carried only as mean + covariance:

```
source → OL1 nonlinear → (retain μ, Σ only) → OL2 nonlinear
```

Case 4 is specifically the test of whether Gaussian closure between OL1 and
OL2 is adequate.

## Chromatic **[added]**

Chromatic aberration is *not* a Krivanek `C_{n,m}` term — those are functions
of pupil coordinate alone. Chromatic couples the transverse coordinate to the
**energy column that rayTEM's state vector already carries** (`E`, index 5,
in keV, populated by `Source(voltage=)`), which makes it the natural first
non-geometric aberration for this mode and the reason it belongs here rather
than in a later phase.

Write the fractional energy deviation of a member of the ensemble as

```
δ = (E − E₀)/E₀                                                       (28)
```

A lens of chromatic coefficient `C_c` (metres) focuses an off-energy electron
at a shifted focal length, and to first order this is a **focal-power shift**

```
Δf = C_c δ          ⟹      ΔP = −P² Δf = −C_c P² δ                    (29)
```

so the extra angular kick each electron receives is

```
δ(xt) = −ΔP·x = +C_c P² δ · x,       δ(yt) = +C_c P² δ · y            (30)
```

Define `κ = C_c P²` so the map through a chromatic lens is

```
x'  = x
xt' = xt − P x + κ δ x                                                (31)
```

and likewise in y. The kick is **bilinear** — a product of two state
components (`δ` and `x`) — so unlike `C10` defocus it cannot be folded into
the 6×6 transfer matrix. It is a genuine second-order term in the sense of
Eq. (13), with `T_ijk ≠ 0` for `(i,j,k) = (xt, x, E)`.

Its covariance contribution needs the fourth moment `⟨δ² x²⟩`. For a source
whose energy spread is independent of the transverse coordinates — which is
the physical situation and the seeded `Σ₀` — Isserlis gives
`⟨δ² x²⟩ = σ_δ² σ_x²` exactly, so the leading contribution

```
ΔΣ[xt,xt] = κ² σ_δ² σ_x²,        ΔΣ[yt,yt] = κ² σ_δ² σ_y²             (32)
```

is **exact**, not a closure approximation, whenever `δ ⟂ (x, xt, y, yt)` and
the beam is centered. That is worth stating in the example: the chromatic term
is the one aberration in this study whose covariance update carries no
Gaussian-closure debt. The familiar chromatic disc of confusion is the same
quantity read at the image plane,

```
d_c ~ C_c α (ΔE/E)   ⟺   σ_x,chromatic = |κ| σ_δ σ_x · (drift lever)  (33)
```

Requirements this adds:

- `Source` gains an **energy spread** so the ensemble actually has one — it
  must seed `Σ[E,E]` for the moments path and sample the `E` column for the
  ray path, so the two can be cross-checked.
- Elements gain a **chromatic coefficient** attribute, separate from
  `aberrations` (which stays strictly Krivanek).
- The covariance update adds the bilinear term; the ray path adds the matching
  per-ray kick so a Monte-Carlo reference test is available.
- The four cases each gain a chromatic-on variant, and the example reports how
  much of the final resolution is chromatic versus geometric.

## Resolution quantities

Do not report only a single probe diameter. At selected planes report at
minimum `σ_x, σ_xt, σ_y, σ_yt` (20), the position–angle covariances
`σ_x,xt` and `σ_y,yt` (21), and the emittances `ε_x, ε_y` (22).

Also report eigenvalues and eigenvectors of the relevant blocks so the
resolution ellipse is quantified rather than collapsed to independent x and y
widths. For the real-space block

```
Σ_rr = [[σ_x², σ_xy], [σ_xy, σ_y²]]                                   (23)
```

the eigenvectors give the real-space principal axes and the square roots of
the eigenvalues the corresponding rms widths. Likewise the angular block

```
Σ_uu = [[σ_xt², σ_xt,yt], [σ_xt,yt, σ_yt²]]                           (24)
```

defines the angular-resolution ellipse, and for wavevector `k₀`

```
Σ_kk = k₀² Σ_uu                                                       (25)
```

maps the angular principal widths directly into momentum-resolution widths.
`k₀` comes from the existing `Source.wavelength`.

## Basic column example

One example, built on the existing basic rayTEM column, following the
project's normal example style rather than a special standalone demo.

One fixed source boundary condition for all cases. The only differences
between cases are which aberrations are enabled.

**Case 1 — ideal column.** OL1 and OL2 aberrations disabled. Propagate the
source covariance through the complete column. Required check:
`Σ_out = M_total Σ₀ M_totalᵀ` (26) must agree with sequential per-element
propagation, and emittance must be constant to numerical tolerance. This
establishes the baseline specimen and detector resolution.

**Case 2 — OL1 only.** Enable the pre-specimen OL1 aberrations, keep OL2
ideal. Use OL1's existing physical aberration parameters and pupil/focal-power
conventions — no covariance-specific aberration model. Chain: source → ideal
upstream → OL1 aberrated update → ideal transport to sample → ideal OL2 and
downstream → detector. Quantify the change in specimen-plane `Σ_rr` and
`Σ_uu`, the change in emittance, induced cross-correlations, the detector
covariance, and the resolution change relative to Case 1. Answers: what
resolution penalty comes specifically from pre-specimen aberrations?

**Case 3 — OL2 only.** The mirror case. It is also the clean test of the OL2
covariance-aberration machinery, because its input has not previously passed
through a nonlinear element.

**Case 4 — both.** The critical case. If only μ and Σ are retained after OL1,
the OL2 update requires a closure assumption; the example must state
explicitly that Gaussian closure is used at OL2. Compare against Cases 2 and 3
and determine whether the combined degradation is approximately additive,
strongly coupled, or dominated by one objective. Do **not** assume

```
ΔΣ_OL1+OL2 = ΔΣ_OL1 + ΔΣ_OL2                                          (27)
```

**Chromatic overlay [added].** Repeat with a nonzero source energy spread and
nonzero `C_c` on the objectives, and report the chromatic share of the final
resolution separately from the geometric share.

## What the example should plot

A small number of physically useful plots, not every covariance element:

- `σ_x(z)` and `σ_y(z)` through the column, all cases;
- `σ_xt(z)` and `σ_yt(z)`;
- `ε_x(z)` and `ε_y(z)`;
- real-space covariance ellipses at specimen and detector;
- angular (or momentum) ellipses at the same planes.

Mark OL1 and OL2 so any increase in projected rms emittance can be attributed
to the element that generated it. A useful diagnostic: whether the covariance
changes abruptly at an aberrated element and then evolves exactly under the
intervening linear transport.

## Tests derived from the example

The example is the source of focused regression tests.

- Ideal case: exact covariance transport and emittance conservation.
- OL1-only and OL2-only: zero aberration strength recovers the ideal result;
  aberration contributions scale with the expected order in aperture angle;
  Σ stays symmetric; Σ stays positive semidefinite; mean shifts caused by
  even-order aberrations are retained, not discarded.
- Combined case: disabling either objective reproduces the corresponding
  single-aberration case.
- **[added]** Chromatic: zero energy spread or zero `C_c` recovers the
  achromatic result; the `σ_xt²` contribution scales as `C_c² σ_δ² σ_x²`; the
  closure result matches a Monte-Carlo ray reference (a separate test, not
  part of covariance mode).

## Do not add rays to covariance mode

The target is a moment-based beam model. Do not add an internal Monte-Carlo
ray fan to make the aberration update easy. Validation against rays, where
useful, is a separate independent reference calculation — not part of
covariance propagation.

## Decision point after the four cases

Inspect where the covariance implementation fails or becomes unnecessarily
approximate, and only then decide whether the code needs Gaussian closure
only, explicit third/fourth moment tensors, composition of nonlinear maps
across linear sections, another analytic closure, or nothing further. The four
cases supply the evidence.

## Preferred initial implementation

```
source properties → mean + covariance + emittance → exact linear covariance
transport → explicit nonlinear aberration covariance update → Gaussian closure
only where higher moments are mathematically required → resolution metrics
```

Keep the Gaussian assumption localized and visible. Do not describe the whole
covariance beam as Gaussian merely because Gaussian identities close one
nonlinear moment calculation.

## Success criteria

One basic-column example runs all configurations from the same source
definition and reports physically interpretable resolution changes at specimen
and detector, making it possible to answer quantitatively: how much of the
final resolution is set by the source/emittance, how much by OL1, how much by
OL2, how much appears only when both aberrated objectives are present, and
**[added]** how much is chromatic.
