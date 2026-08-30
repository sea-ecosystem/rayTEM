# Into the SEA-weeds

What exists in rayTEM, why it exists, and the invariants that protect it.
User-facing workflows live in the Guides; this section is for people changing
the framework.

## The mental model

An `Element` declares linear optics (a 6×6 `transfer_matrix`), optional
nonlinear physics (`aberrations`, applied generically — no per-aberration
code), and optional wave physics (`phase_shift`, screens). A
`MicroscopeSection` stacks elements; a `Microscope` stacks sections. All four
propagation modes walk the same hierarchy bottom-up — the Microscope never
reaches inside an Element, and a Section never reaches inside a Microscope.

## Invariants

These are the load-bearing walls (the repository `CLAUDE.md` carries the same
list for agents):

- **Ray columns are geometric**: `convention = ["x","xt","y","yt","z","E"]`.
  Intensity and Larmor rotation are *not* ray columns — they travel as
  parallel `(n_planes, n_rays)` arrays (`.I`, `.R`).
- **Transfer matrices are 6×6**, produced by `fix_mat_dims()`.
- **One plane per element**: propagation logs at element exits, so the
  element list *is* the z sampling. `repair()` therefore keeps the element
  list exactly as written by default (`combine_drifts=False`) and inserts a
  leading drift when a section's first element sits past the section start —
  both learned the hard way (a merged drift pair silently removed sampling;
  a dropped leading offset compressed everything downstream).
- **Currents are amps, stated once**: the `Source`/`Gun` states
  `beam_current`; every other element derives what arrives at it, and the
  derived values are recorded state — they survive a `.sea` round trip.
- **seashells is the sea-eco seam**: all serialization and wavefield-Signal
  construction go through it, and it degrades gracefully without sea-eco.
  On reload, `safeReinstantiate` restores the recorded `__dict__` verbatim —
  the constructor only supplies the class.
- **Wavelength**: `Source(voltage=<kV>)` sets it; `voltage=None` keeps
  ray-only behavior bit-for-bit.

## The scaled-wave engine

The full-column wave mode factorizes the field against a geometric reference
frame and switches frames analytically at crossovers. The subtle parts —
where the frame may flatten, how a crossing inside a lens body is recorded,
and how a body that ends mid-restoration hands its marker to the free-space
engine — are documented at the code (`waveoptics.py`) and pinned by tests;
the companion page [Why full-column electron wave propagation breaks
fixed-grid sampling](wave-optics-sampling.md) derives why the fixed-grid mode
cannot do this job and what the scaled factorization buys.

## Aberrations

Axial wave aberrations are Krivanek `C_{n,m}` coefficients on any element
(`aberrations.py`). One declaration drives both paths: the ray side applies
the eikonal kick `(1/k)∇χ` and the wave side the phase `exp(iχ)` — exact to
all orders and wavelength-free by construction, so no element carries
per-aberration code. Screens generalize this: a real-valued screen is a phase
χ; a complex one is a transmission `T` (amplitude and phase, what a
fabricated plate has). A *supplied* screen is stored (nothing can recompute
it); a derivable one is recomputed on demand.

Aberrations attach at three levels: an element's own `aberrations`, a
stand-alone `AberrationScreen` (a zero-thickness plate with an explicit
pupil power), and a `MicroscopeSection`'s `aberrations` — the section case
synthesizes a transient screen at its exit with the section's composite
focal power as the pupil scale. The low-order terms are power changes, not
phase: see [Terminology](terminology.md) for the strength / focal length /
focal power distinctions and the aberration-power split.

They now reach **all four propagation modes**: the ray path applies the
exact eikonal kick, the wave paths carry χ, the covariance mode closes the
kick's moments analytically on Σ, and the frame/ABCD machinery computes the
**aberrated focal surface in closed form** — `focal_surface(method='frame')`
rebuilds each aberrated element's block with its pupil zone's own power
(`Element.zone_power_shift`) and solves the zone ray's crossing analytically.
On a thin lens the frame surface matches the traced one to machine precision.

## Moments, and the closure seam

The covariance mode carries two moments; a nonlinear element needs more.
Rather than hard-code where they come from, the mode splits the beam state
from the assumption used to extend it — `moments.CovarianceBeam` holds
`{mean, covariance, moment_closure}`, and `moments.MomentClosure` is a
one-method interface (`moment(Sigma, indices)`) that a caller can replace.
`GaussianMomentClosure` answers by Wick pairing at any order.

The nonlinear kick reaches the closure as a *polynomial*, not as per-term
algebra: `Element.aberration_monomials` recovers it by sampling the existing
`deflection_at` on the unit circle and solving for the coefficients of a
homogeneous degree-`n` form, which is legitimate precisely because a Krivanek
term of order `n` deflects by a degree-`n` homogeneous polynomial and
therefore carries no length scale of its own. One consequence worth stating:
**every** order the pupil carries is now closed, rotated terms included, and
adding a term to `KRIVANEK_TERMS` reaches this path with no algebra written by
hand.

Three properties this buys, each of which the previous hand-derived C30
closure lacked:

- **Completeness.** Cross-plane terms (`<x δθ_y>`, `<δθ_x δθ_y>`) are computed,
  not dropped. They vanish for a transversely decoupled beam and are
  emphatically not negligible otherwise — through a Larmor-rotating objective
  an astigmatic source produces cross-plane terms as large as the in-plane
  ones.
- **Positive semidefiniteness by construction.** A complete Gaussian closure is
  the exact pushforward of a Gaussian, so `Σ'` is a real covariance for any
  aberration strength. It is *truncation* that breaks PSD, not strength — which
  is the argument against ever shipping a partial closure, and the reason
  `CovarianceBeam.is_positive_semidefinite` exists as a guard on custom ones.
- **A retained mean shift.** An even-order aberration moves the ensemble mean
  by `<δ(r)>` while the centroid ray, which feels `δ(μ)`, does not move at all.
  `propagate_moments` takes the affine terms from an ideal ray and adds the
  aberration's contributions explicitly, so the shift is reported rather than
  absorbed into the width.

**Chromatic** attaches here as the one non-geometric term, as the `'Cc'`
entry of an element's `Aberrations` set, paired with `Source.energy_spread`
seeding `Σ[E,E]`. Putting it *in* the set is what makes it impossible to
forget: it serializes, suspends and copies with the Krivanek terms. Keeping it
out of `names`/`items()` is what keeps the Krivanek machinery from ever seeing
a term it cannot interpret — those are functions of pupil coordinate alone,
whereas chromatic couples the pupil to the energy column, making the kick
bilinear and so not matrix-expressible even though it is a power change. Its
covariance term is exact rather than closed, because the only fourth moment it
needs factorizes under the physical assumption that energy spread is
independent of transverse position. And because it needs no round pupil, it is
applied per axis, which makes it exact on a quadrupole too.

**Pupil scale.** Everything that turns a ray height into a pupil angle goes
through `Element._pupil_scale()`: an explicit `pupil_power`, else the scalar
`focal_power`, else the geometric mean of a per-axis `focal_powers` pair. That
last case is what lets an astigmatic element carry aberrations at all — a
quadrupole states `focal_powers` and so used to resolve zero, which silently
disabled its aberrations on *every* path. The mean is a stated reference scale,
not a derived truth, because the Krivanek expansion assumes a round pupil;
`Quadrapole(pupil_power=...)` states it explicitly when the coefficients were
measured against a particular one.

**Serialization.** `Aberrations`, `MomentClosure` and `CovarianceBeam` are all
`SEASerializable`. Two things this forced: a SEA object is rebuilt by calling
its class with *no* arguments before its state is assigned, so no constructor
may have a required parameter; and `Microscope.save` is a hand-rolled JSON
writer, so it needed an explicit case for a nested SEA object carrying complex
coefficients (`Aberrations.to_metadata()`, inverted by `from_metadata`) — before
which an aberrated column could not be written to `.json` at all. A trap worth
knowing: sea-eco's reader routes a stored public `x` into `_x` whenever the
object has that name, so a private method spelled `_<public attribute>` is
silently replaced by a float on reload.

The honest limit is stated where it is measured rather than buried: a cubic
kick leaves excess kurtosis `γ₂ = 27f²`, with `f` the aberration's share of the
angular variance, so there is no regime in which the aberration matters to Σ
but the induced non-Gaussianity does not. `examples/08_covariancePropagation.py`
prints `f` per element and measures the OL1–OL2 interaction residual directly.

## High-risk seams

- `elements.py` — `columnByName()` and `fix_mat_dims()` are referenced
  everywhere.
- `seashells.py` — the conditional sea-eco import and the reload path.
- `waveoptics.py` — grid centering and the frame-switch engine; focus and
  aperture positions drift if these lose consistency.
- `AS2.py` — talks to a live instrument.

## Schema

rayTEM does not currently own a schema, nor does it formally implement
sea-eco's `pipeline-editor` or `nd-plotting` schemas — its plotting goes
through matplotlib directly and its sea-eco integration is data-level
(Signals in, Signals out) rather than a backend implementation. This section
exists so that omission is a recorded decision; if a rayTEM surface grows a
second implementation, the contract starts in the owning package's schema
first.

## Provenance and verification

- **Implementation entry points**: `elements.py` (elements, screens,
  aberration application), `assemblies.py` (sections, columns, propagation
  drivers, conjugate planes), `waveoptics.py` (paraxial and scaled-Fresnel
  primitives), `aberrations.py` (the Krivanek model), `moments.py` (the
  covariance state and the closure seam), `seashells.py` (serialization
  seam).
- **Focused tests**: `src/pySEA/rayTEM/tests/` — `test_scaled_fresnel.py`
  (wave engine, screens, aberrations, currents),
  `test_eight_configurations.py` (the operating states as executable
  claims), `test_covariance_propagation.py` (the closure against hand-written
  Isserlis, the kick polynomial against `deflection_at` at every order, and
  the aberrated and chromatic covariance updates against Monte-Carlo rays),
  `test_wave_and_envelope.py`, `test_elements_sections_microscopes.py`.
- **User guides**: Getting Started, Propagation Modes, Operating the Column.
- **Executable examples**: `examples/01`–`08`, all headless-runnable; 05 and
  07 are cross-method verification scripts, 08 is the covariance resolution
  study.
- **AI-tool artifacts**: the `ai_wiki/raytem` slice (index, layer map,
  method index), refreshed by `pysea-refresh-wiki`.
- **API coverage**: generated from the NumPy-style docstrings that every
  callable in this repository is required to carry (see API Reference).

## Building the docs

```bash
pip install -r docs/requirements.txt
sphinx-build -d docs/_build/doctrees docs docs/_build/html
```

The API Reference is generated from docstrings at build time — there is no
separate regeneration step. A clean build currently emits eleven known
"duplicate object description" warnings, all from classes re-exported across
modules; anything beyond those is a regression. The built HTML is
**committed** under `docs/_build/html/` so the rendered site travels with
the branch — rebuild and commit it alongside any docs or docstring change.
There are no notebooks to execute; the example scripts run headless via
`MPLBACKEND=Agg` and are exercised by the test suite.
