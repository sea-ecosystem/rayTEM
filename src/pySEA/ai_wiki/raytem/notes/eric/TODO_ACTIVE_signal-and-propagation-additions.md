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
- [ ] wrap `rays` as a sea_eco `SignalSet` (deferred; see step 4 sea_eco integration)

## Step 3 — Beam-envelope mode ✓
- [x] `Element.propagate_moments`; Source seeds Σ0; Aperture intensity-only (non-truncating approx)
- [x] Section/Microscope `propagate_moments` drivers → `covariance_matrix` (+ `mu`)
- [x] `beam_widths`, `emittance`; tests (MC covariance match, emittance conservation, waist=focus)

## Step 4 — Wave-optics mode
- [ ] `seashells.make_wavefield_signal` (present + absent paths)
- [ ] `waveoptics.py`: angular-spectrum + phase/mask + field builders
- [ ] `Source.field` + `Element.propagate_wave` (Prism → NotImplementedError)
- [ ] Section/Microscope `propagate_wave` → single `(Nz,Ny,Nx)` wave Signal; golden + `.sea` round-trip

## Step 5 — Wiki + finish
- [ ] `pysea-refresh-wiki`; hand-edit wiki docs + CLAUDE.md core invariants
- [ ] Rename this file ACTIVE → DONE; LOG → `[Done]` + Outcome
