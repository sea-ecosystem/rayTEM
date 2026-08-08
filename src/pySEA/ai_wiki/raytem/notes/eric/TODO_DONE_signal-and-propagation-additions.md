# TODO — Signal-backed results and new propagation modes

**Branch:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-08_signal-and-propagation-additions.md](PLAN_2026-08-08_signal-and-propagation-additions.md)

## Step 1 — Finish ray-representation refactor (geometric [x,xt,y,yt,z,E]) ✓
- [x] `elements.py`: `convention` → 6 geometric cols; matrices 6×6
- [x] `elements.py`: `Source.rays` returns geometric rays (I/R seeded by drivers: I=1, R=0)
- [x] `elements.py`: `Element` gains `apply_intensity`/`apply_rotation` side channels; `propagate_ray` stays geometric
- [x] `elements.py`: `Aperture` scales `I` separately; `Dipole` docstring no longer references `I` column
- [x] `assemblies.py`: Section/Microscope store `.rays` (geometric), `.I`, `.R`; fix `beam_current`/`planes`/`show`/`save`
- [x] `postprocessing.py`: `convert_to_rotating_reference_frame`, `findPlanes`, `measureAtZ`, `plot2D`/`plot3D`, `error_*` take `R`/`I` explicitly
- [x] tests: drop `swap_columns`, regenerate 6-col goldens, drop `columnByName("I")` in dipole test
- [x] `pytest` green (13 passed)

## Step 2 — Wavelength + result containers
- [x] `utilities.relativistic_wavelength(voltage_kV)` + test
- [x] `Source.voltage` optional → seeds `E` + `wavelength`; default `None`
- [x] `covariance_matrix`/`wave`/`mu` container attributes + eager re-chain drivers (envelope done; wave in step 4)
- [x] wrap `rays` as a sea_eco `SignalSet` (`rays_signalset()` via `seashells.make_rays_signalset`; non-invasive on-demand view)

## Step 3 — Beam-envelope mode ✓
- [x] `Element.propagate_moments`; Source seeds Σ0; Aperture intensity-only (non-truncating approx)
- [x] Section/Microscope `propagate_moments` drivers → `covariance_matrix` (+ `mu`)
- [x] `beam_widths`, `emittance`; tests (MC covariance match, emittance conservation, waist=focus)

## Step 4 — Wave-optics mode ✓
- [x] `seashells.make_wavefield_signal` + `read_wavefield` (present + `_Wavefield` absent paths)
- [x] `waveoptics.py`: angular-spectrum + focal/tilt phase + aperture mask + field builders
- [x] `Source.field` + `Element.propagate_wave` (matrix-derived focal powers; Prism → NotImplementedError)
- [x] Section/Microscope `propagate_wave` → single `(Nz,Ny,Nx)` wave Signal; tests (focus, Fresnel Gaussian, `.sea` complex round-trip)

## Step 5 — Wiki + finish ✓
- [x] `pysea-gen-ai-index` refresh; hand-edited CLAUDE.md core invariants + index.md/layer-map.md for new convention & 3 modes
- [x] Rename this file ACTIVE → DONE; LOG → `[Done]` + Outcome

## Follow-ups (noted for later)
- **sea-eco:** leftover `print(...)#FLAG` debug lines in `Dimension.__init__` (values-without-scale path). Worked around from rayTEM by passing scale/offset to unstructured z axes; the real cleanup belongs in sea-eco.
- **sea-ecosystem:** `pysea-discover-wiki` couldn't regenerate `ecosystem/index.md` in this multi-repo namespace-package checkout (resolves `pySEA` to the first path). rayTEM's own wiki refresh completed.
- **`microscopes/` scripts:** several still read `columnByName("I")`/`("R")`; migrate them to `.I`/`.R` (instrument scripts, out of scope here).
