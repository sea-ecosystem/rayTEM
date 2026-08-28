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
  primitives), `aberrations.py` (the Krivanek model), `seashells.py`
  (serialization seam).
- **Focused tests**: `src/pySEA/rayTEM/tests/` — `test_scaled_fresnel.py`
  (wave engine, screens, aberrations, currents),
  `test_eight_configurations.py` (the operating states as executable
  claims), `test_wave_and_envelope.py`, `test_elements_sections_microscopes.py`.
- **User guides**: Getting Started, Propagation Modes, Operating the Column.
- **Executable examples**: `examples/01`–`07`, all headless-runnable; 05 and
  07 are cross-method verification scripts.
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
