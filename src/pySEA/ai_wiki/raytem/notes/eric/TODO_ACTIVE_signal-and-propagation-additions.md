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
- [ ] `utilities.relativistic_wavelength(voltage_kV)` + test
- [ ] `Source.voltage` optional → seeds `E`; default `None`
- [ ] `rays` SignalSet, `covariance_matrix`, `wave` containers; eager re-chain drivers

## Step 3 — Beam-envelope mode
- [ ] `Element.propagate_moments`; Source seeds Σ0; Aperture intensity-only
- [ ] Section/Microscope `propagate_moments` drivers → `covariance_matrix`
- [ ] `beam_widths`, `emittance`; golden test

## Step 4 — Wave-optics mode
- [ ] `seashells.make_wavefield_signal` (present + absent paths)
- [ ] `waveoptics.py`: angular-spectrum + phase/mask + field builders
- [ ] `Source.field` + `Element.propagate_wave` (Prism → NotImplementedError)
- [ ] Section/Microscope `propagate_wave` → single `(Nz,Ny,Nx)` wave Signal; golden + `.sea` round-trip

## Step 5 — Wiki + finish
- [ ] `pysea-refresh-wiki`; hand-edit wiki docs + CLAUDE.md core invariants
- [ ] Rename this file ACTIVE → DONE; LOG → `[Done]` + Outcome
