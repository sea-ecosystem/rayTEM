# Terminology

The lens infrastructure uses several near-synonyms — *strength*, *focal
length*, *focal power* — plus a handful of coined names (*transfer block*,
*body*, *screen*, *moments*). This page defines each one, says why it exists
instead of one of the others, and points at the literature. It exists to
answer [issue #8](https://github.com/sea-ecosystem/rayTEM/issues/8).

## Strength, focal length, focal power

Three numbers describe the same lens; each is the natural variable of a
different calculation, which is why all three survive.

**Strength `K`** (1/m) is the *field* parameter from Brown's matrix
formalism [Brown 1983, pp. 105–106]: inside a magnetic round lens the
paraxial equation of motion is harmonic, `u'' + K²u = 0`, and `K` is that
oscillation rate. It is the quantity the transfer matrix is written in
(`cos KL`, `sin KL` entries), the quantity a calibration maps a lens current
onto, and the quantity a thick lens's Larmor rotation is built from
(`R = KL`). It is a property of the *field*, not of the imaging: two lenses
with the same `K` but different lengths focus differently.

**Focal length `f`** (m) is the classical light-optics quantity: the
back-focal distance of the equivalent thin lens. It is what a microscopist
quotes and what column geometry is designed around. For a thin element the
two are tied by `1/f = sign(K)·K²`; for a thick body by Brown's focusing
relation

```
1/f = K · sin(K·L)
```

(`Lens.focal_power` implements exactly this pair). Note the thick relation
is not monotonic: past `KL = π/2` a stronger field gives a *longer* focal
length, which is why `strength_for_focal_length` in
`microscopes/basic_column.py` restricts itself to the first branch,
`f ≥ 2L/π`.

**Focal power `P = 1/f`** (1/m, the light-optics *dioptre*) is the
reciprocal, and it is not a redundant synonym — it is the quantity that
**composes**. Focal lengths of stacked thin elements do not add; powers do
(`P = P₁ + P₂` in contact, the thin-lens limit of Gullstrand's equation).
That additivity is why the framework's derived quantities are expressed as
powers:

- `Quadrapole.focal_powers` is a **per-axis pair** `(P_x, P_y)` with
  opposite signs — one number per axis where a single `f` cannot describe
  an astigmatic element at all.
- `Lens.aberration_powers` expresses low-order aberrations as **changes of
  power** (next section) — meaningful precisely because powers add.

Rule of thumb: `strength` when talking to the field or the matrix, `focal
length` when talking to the instrument, `focal power` whenever anything is
being *summed per axis*.

### Aberrations as power changes

The first-order Krivanek terms are quadratic in the pupil coordinate — the
same shape as the lens's own focusing parabola — so they are not "extra
phase" at all but a modification of the effective power:

- `C10` (defocus) is isotropic: `P → P + C10·P²` on both axes.
- Aligned `C12` (twofold astigmatism) is the same size with opposite signs
  per axis: `P_x = P + C12·P²`, `P_y = P − C12·P²`, producing two **line
  foci** at `1/(P ± C12·P²)` instead of one point focus.

So "the effective focal length of an aberrated lens" is per-axis
`f = 1/(P + ΔP)` with `ΔP = C_1m·P²` — this is what `Lens.aberration_powers`
returns, and the scaled-wave engine absorbs those terms into its quadratic
reference frame rather than wasting a sampled screen on them. Everything of
second order and above is genuinely non-quadratic and stays in the residual
screen. See Krivanek's aberration nomenclature [Krivanek 1999] for the
`C_{n,m}` indexing.

### Effective strength

`_effective_strength` is `strength` after the lens's calibration mapping
(the current-to-field conversion) is applied. It exists so the ray and wave
paths read the *same* calibrated `K` from one place instead of each applying
the calibration separately.

## `transfer_matrix`, `transfer_block`, `_body_block`

All three feed ray optics; they differ in frame and granularity.

- **`transfer_matrix()`** — the full `6×6` lab-frame matrix over the whole
  element, Larmor rotation included. The one method every element must
  provide for ray optics; `propagate_ray` consumes it.
- **`transfer_block(dz, axis)`** — the `2×2` `[[A,B],[C,D]]`
  position–angle sub-block for **one transverse axis in the rotating
  frame** (Larmor rotation divided out), accepting a **partial** length
  `dz` so a plane *inside* a body can be located exactly. This is the
  object conjugate-plane finding works with: accumulated from a reference
  plane, `B = 0` marks an image plane and `A = 0` a diffraction
  (back-focal) plane [Brown 1983, §2].
- **`Quadrapole._body_block(dz, axis)`** — private: the exact in-body law
  of a quadrupole (harmonic on the focusing axis, hyperbolic on the
  defocusing one). `transfer_matrix`, `transfer_block` and `focal_powers`
  all read it so they cannot drift apart.

`Prism.focus_matrix` / `Prism.bending_matrix` follow the same pattern —
named factors that compose into `transfer_matrix`.

## Moments

"Moments" are the **statistical moments of the ray distribution**: the mean
`μ` (first moment, the centroid ray) and the covariance matrix `Σ` (second
central moment, the beam envelope and its correlations). `propagate_moments`
transports them with the *same* transfer matrices as the rays,
`μ' = Mμ`, `Σ' = MΣMᵀ` — the sigma-matrix formalism of accelerator physics
[Brown 1983, §7]. It answers "how wide is the beam and where is its waist"
without tracing individual rays, and it carries emittance physics (the waist
sits displaced from the geometric image plane) that single-ray tracing
cannot show.

## Body and screen

The two idealizations every element reduces to:

- A **body** is a finite-length region where something acts *distributed
  along z* — a thick lens's harmonic medium, a quadrupole's
  harmonic/hyperbolic medium. Bodies have partial propagators
  (`transfer_block(dz)`), interior planes, and distributed aberration kicks.
- A **screen** is a zero-thickness plane that multiplies the wave:
  a real-valued screen is a phase `exp(iχ)`; a complex one is a transmission
  `T` (amplitude and phase — what a fabricated phase plate has). The ray
  path consumes the same `χ` as its gradient, `Δθ = (1/k)∇χ`.

A thin element is "a screen between two nothings"; a thick element is "a
body". The wave path of a thick lens is kernel–screen–kernel: half the body,
its screen at the mid-plane, the other half.

## `wave_kind`

`wave_kind` is a **generation recipe on `Source`** (`'plane'`,
`'gaussian'`, `'point'`, `'aperture'`): it tells `Source.wave()` what initial
field to synthesize, exactly as `size`/`angle` tell `Source.rays()` what fan
to generate. It is *not* encoded in the propagated field and makes no claim
about it — one element downstream, the field is whatever propagation made
it, and `wave_kind` still reads whatever the source was told at
construction. Treat it like `np_xy`: an input parameter, not a diagnostic.

## References

- K. L. Brown, "A First- and Second-Order Matrix Theory for the Design of
  Beam Transport Systems and Charged Particle Spectrometers," SLAC Report 75
  (1983 reprint). Strength `K`, the thick-lens matrices (pp. 105–106), and
  the sigma-matrix formalism.
- O. L. Krivanek, N. Dellby, A. R. Lupini, "Towards sub-Å electron beams,"
  Ultramicroscopy 78 (1999) 1–11. The `C_{n,m}` aberration nomenclature.
- P. W. Hawkes, E. Kasper, *Principles of Electron Optics*, Academic Press.
  Larmor rotation and the rotating frame for round magnetic lenses.
