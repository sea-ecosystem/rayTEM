# TODO — Signal-backed results and new propagation modes  (DONE — marked copy)

**Branch:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-08_signal-and-propagation-additions.md](PLAN_2026-08-08_signal-and-propagation-additions.md)
**Note:** the unmarked copy of this list + the plan also live on `rayTEM/main`
(`src/pySEA/ai_wiki/raytem/notes/eric/`). This is the marked/progress copy.

## Step 1 — Finish ray-representation refactor (geometric [x,xt,y,yt,z,E]) ✓
- [x] `elements.py`: `convention` → 6 geometric cols; matrices 6×6
- [x] `elements.py`: `Source.rays` geometric (I/R seeded by drivers: I=1, R=0)
- [x] `elements.py`: `Element.apply_intensity`/`apply_rotation` side channels; `propagate_ray` stays geometric
- [x] `elements.py`: `Aperture` scales `I` separately; `Dipole` docstring off the `I` column
- [x] `assemblies.py`: Section/Microscope store `.rays` (geometric), `.I`, `.R`; fix `beam_current`/`planes`/`show`/`save`
- [x] `postprocessing.py`: `convert_to_rotating_reference_frame`, `findPlanes`, `measureAtZ`, `plot2D`/`plot3D`, `error_*` take `R`/`I` explicitly
- [x] tests: drop `swap_columns`, regenerate 6-col goldens
- [x] `pytest` green (13 passed at this step)

## Step 2 — Wavelength + result containers ✓
- [x] `utilities.relativistic_wavelength(voltage_kV)` + test
- [x] `Source.voltage` optional → seeds `E` + `wavelength`; default `None`
- [x] `covariance_matrix`/`mu`/`wave` container attributes + eager re-chain drivers
- [x] wrap `rays` as a sea_eco `SignalSet` (`rays_signalset()`)

## Step 3 — Beam-envelope mode ✓
- [x] `Element.propagate_moments`; `Source.moments` seeds Σ0; `Aperture` non-truncating
- [x] Section/Microscope `propagate_moments` drivers → `covariance_matrix` (+ `mu`)
- [x] `beam_widths`, `emittance`; tests (MC covariance, emittance conservation, waist=focus)

## Step 4 — Wave-optics mode ✓
- [x] `seashells.make_wavefield_signal` + `read_wavefield` (present + `_Wavefield` absent path)
- [x] `waveoptics.py`: angular-spectrum + focal/tilt phase + aperture mask + field builders
- [x] `Source.field` + `Element.propagate_wave` (matrix-derived powers; Prism → NotImplementedError)
- [x] Section/Microscope `propagate_wave` → single `(Nz,Ny,Nx)` wave Signal; tests (focus, Fresnel, `.sea` round-trip)

## Step 5 — Unified dispatcher ✓
- [x] `propagate(*args, kind=..., **kwargs)` on Element/Section/Microscope + test

## Step 6 — Migrate instrument scripts + tools ✓
- [x] `microscopes/` scripts off `columnByName("I")`/`("R")` → `.I`/`.R`
- [x] add `R` arg to `findPlanes`/`plot2D`/`plot3D`; `I`/`R`/`section=` to `measureAtZ`; drop stale `axes=`/`returnObjectOnly=`
- [x] `py_compile` clean for every touched script; no residual stale patterns

## Step 7 — Wiki + docs ✓
- [x] `pysea-refresh-wiki`; hand-edited CLAUDE.md core invariants + index.md/layer-map.md

## Follow-ups (noted for later)
- **04_PRIVATE_INSTRUMENT.py:** pre-existing truncated scratch fragment (invalid Python already on `origin/main`, undefined lowercase `start/lens/drift/quad` helpers). Left as-is — not a refactor issue; needs its author to finish it.
- **sea-eco:** leftover `print(...)#FLAG` debug lines in `Dimension.__init__` (values-without-scale path). Worked around from rayTEM by passing scale/offset to unstructured z axes; the real cleanup belongs in sea-eco.
- **sea-ecosystem:** `pysea-discover-wiki` couldn't regenerate `ecosystem/index.md` in this multi-repo namespace-package checkout (resolves `pySEA` to the first path). rayTEM's own wiki refresh completed.
