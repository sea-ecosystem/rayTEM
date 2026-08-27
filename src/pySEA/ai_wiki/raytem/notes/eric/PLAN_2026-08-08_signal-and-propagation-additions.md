# PLAN — Signal-backed results and new propagation modes

**TODO:** [TODO_ACTIVE_signal-and-propagation-additions.md](TODO_ACTIVE_signal-and-propagation-additions.md)
**Branch (rayTEM):** `Signal_and_propagation_additions`

## Goal

Give rayTEM three interchangeable propagation modes, each with its own result
container, on top of a cleaned-up geometric ray representation:

1. `propagate_ray` — geometric ray transfer (existing), now on a purely
   geometric ray vector.
2. `propagate_moments` — analytic beam-envelope: covariance `Σ' = M Σ Mᵀ`
   through the same transfer matrices.
3. `propagate_wave` — paraxial 2D complex scalar wave optics, backed by a
   sea_eco `Signal` via the `seashells` seam.

## Why now / prerequisite that turned into scope

The intended ray-representation refactor (`[x, θx, y, θy, z, E]`, with intensity
`I` and Larmor rotation `R` pulled out of the ray columns) was started but left
half-finished: on a fresh clone the code carries `convention =
["x","xt","y","yt","z","E","I","R"]`, but the golden `.npy` fixtures and the
`swap_columns = [0,1,2,3,4,6,5,7]` hack in the tests are inconsistent with it, so
7 of 13 tests fail out of the box. Finishing that refactor is therefore step one
of this task, not a precondition owned elsewhere.

## Ray-representation refactor (step 1)

- `convention = ["x","xt","y","yt","z","E"]` — 6 purely geometric columns.
  Transfer matrices become 6×6 via the unchanged `fix_mat_dims`/`fix_ray_dims`
  (they already size off `len(convention)`).
- `I` (intensity, per ray) and `R` (cumulative Larmor rotation, per ray but
  uniform across rays) leave the ray vector and travel as **separate parallel
  arrays** threaded through `propagate_ray` and stored as `.I` / `.R` on
  `MicroscopeSection` / `Microscope`.
- `postprocessing` helpers that consumed the old columns are adapted to receive
  `R` / `I` explicitly:
  - `convert_to_rotating_reference_frame(rays, R)`
  - `findPlanes(rays, R=...)` (only needs `R` to un-rotate + to report plane `R`)
  - `measureAtZ(...)` returns `x,y,xt,yt,R,I` sourced from the separate arrays.
- Tests: drop `swap_columns`, regenerate golden `.npy` under the new convention,
  fix `columnByName("I")` usage in `test_dipole_transfer_matrix`.

## Result containers (step 2)

Each mode owns its own container on `Microscope` (mirrored per Section/Element):
- `rays` — sea_eco `SignalSet`: geometric ray table + separate `I` + `R` Signals
  (shared ray-index dim). Ray-index and requested-`z` axes are **unstructured**
  (explicit `values`, not scale+offset).
- `covariance_matrix` — its own `Signal`.
- `wave` — a **single** complex `(Nz, Ny, Nx)` `Signal`: unstructured `z` axis +
  calibrated transverse `x`/`y`. Not a per-plane collection — the
  angular-spectrum propagator preserves the transverse grid, so `dx`/`dy` are
  identical at every plane.

Drivers use **eager re-chain**: always propagate source→…→last in order,
overwriting each stage's cache from its real predecessor, so a full-instrument
result is always assembly-consistent. Input resolution for any
`propagate_*(r0=None)`: explicit `r0` → else upstream neighbor's output → else
`Source` generates it.

## Wavelength

`utilities.relativistic_wavelength(voltage_kV)`; `Source.voltage` (optional,
default `None`) seeds `E` and the wave/covariance initial conditions. Default
`None` keeps existing ray goldens unaffected.

## Envelope mode (step 3)

`Element.propagate_moments(mu, Sigma)`: `M = transfer_matrix()`, `Σ' = M Σ Mᵀ`,
`μ' = M μ` plus the same additive terms as `propagate_ray`. `Source` seeds
`(μ0, Σ0)` (diagonal from `size`/`angle`). `Aperture` attenuates intensity only
(documented linear approximation). `postprocessing`: `beam_widths`, `emittance`.

## Wave mode (step 4)

- `seashells.make_wavefield_signal(...)` — sea_eco-backed complex Signal with
  2-axis calibrated Dimensions + wavelength metadata; graceful fallback when
  sea_eco absent (matches existing seam behavior).
- New `waveoptics.py` — `angular_spectrum_propagate`, `lens_phase`,
  `aperture_mask`, plane/gaussian/point field builders.
- `Element.propagate_wave(signal)` per element (`Prism` → `NotImplementedError`);
  Section/Microscope drivers stack per-`z` wavefields into the single `wave`
  Signal.

## Verification

- Regression: `pytest src/pySEA/rayTEM/tests` green under the new convention.
- Envelope: covariance waist coincides with ray-optics image plane; Σ'=MΣMᵀ vs
  Monte-Carlo sample covariance of a traced bundle.
- Wave: plane wave → thin lens → drift=f focuses at z=f; Fresnel Gaussian waist;
  wavefield Signal `.sea` round-trip preserves complex data.
- Docs: `uv run --extra ai-wiki pysea-refresh-wiki` runs clean.

## Notes for Ondrej

This finishes the geometric-ray-vector reorder (private contributor's WIP) and builds the two
remaining paradigms on it. The ray convention changes from 8 pseudo-columns to 6
geometric columns; `I`/`R` are now separate arrays/Signals, not ray coordinates.
Any code reading `columnByName("I")` / `columnByName("R")` must move to the
separate `.I` / `.R` arrays.
