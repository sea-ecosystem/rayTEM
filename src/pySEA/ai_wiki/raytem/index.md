# rayTEM — AI wiki orientation

## What this package does

rayTEM simulates propagation through a transmission electron microscope using
transfer-matrix formalism. A simulated instrument is built as a hierarchy of
`Microscope → MicroscopeSection → Element` objects, and there are **three
interchangeable propagation modes**, all driven by the same 6×6 transfer matrices:

- `propagate_ray` — geometric ray tracing (the original mode).
- `propagate_moments` — analytic beam-envelope: propagates the ensemble mean and
  covariance via `Σ' = M Σ Mᵀ` (results on `.mu` / `.covariance_matrix`).
- `propagate_wave` — paraxial 2D complex scalar wave optics, stored as a
  calibrated sea_eco `Signal` on `.wave`.

Lens strengths can be fitted to achieve a target image-plane position or
magnification.

The package also contains instrument-specific calibration work under `microscopes/`
(PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT) and a live-control interface via `AS2.py`.

## Mental model

```
Microscope
  └─ MicroscopeSection (e.g. "illumination", "objective", "projector")
       └─ Element (Source | Drift | Lens | Dipole | Quadrupole | Aperture | ...)
            └─ transfer_matrix (6×6 numpy array)
```

Rays are represented as `(N, 6)` **purely geometric** arrays where columns are ordered:

```
[x,  xθ,  y,  yθ,  z,  E]
 0    1    2    3   4   5
```

Column indices are looked up by name via `columnByName(name)`. This indirection
means columns can be added without touching every Element.

**Intensity (`I`) and cumulative Larmor rotation (`R`) are not ray columns.** They
travel as separate parallel arrays on `MicroscopeSection`/`Microscope` — `.I` and
`.R`, each shape `(n_planes, n_rays)` — because they do not participate in the
ray-transfer matrix. Elements update them through `apply_intensity` / `apply_rotation`,
and postprocessing helpers (`convert_to_rotating_reference_frame`, `findPlanes`,
`measureAtZ`, `plot2D`) take `R`/`I` explicitly. (Historically `I`/`E`/`R` were ray
columns; that reorder is complete — code still reading `columnByName("I")`/`("R")`
is stale.)

## Read order for a new contributor

1. **This file** — orientation
2. `layer-map.md` — module responsibilities and invariants
3. `wiki/rayTEM/elements.md` — Element base class, ray conventions, `columnByName`, `fix_mat_dims`
4. `wiki/rayTEM/assemblies.md` — MicroscopeSection, Microscope, propagation and fitting
5. `wiki/rayTEM/seashells.md` — sea_eco integration seam
6. `wiki/rayTEM/AS2.md` — live instrument control (read before touching live hardware paths)

For a specific method, use the two-step lookup:

1. `method-index.json` → `wiki_path` + `wiki_lineno`
2. `Read wiki_path at offset=wiki_lineno`

## Key entry points

- `src/pySEA/rayTEM/elements.py` — Element base class, concrete element types, ray utils
- `src/pySEA/rayTEM/assemblies.py` — MicroscopeSection, Microscope
- `src/pySEA/rayTEM/seashells.py` — conditional sea_eco import (the only place sea_eco is imported)
- `src/pySEA/rayTEM/AS2.py` — AS2querier for live Nion instrument control
- `src/pySEA/rayTEM/postprocessing.py` — visualization utilities
- `src/pySEA/rayTEM/utilities.py` — shared mathematical helpers
- `src/pySEA/rayTEM/xmlNion.py` — parse AS2 XML configuration files
- `REMOVED_PRIVATE_INSTRUMENT_TREE/` — per-instrument calibration scripts (not framework code)

## Current state

- Framework code (`elements.py`, `assemblies.py`) is stable and tested
- sea_eco integration is optional — seashells.py handles conditional import gracefully
- `microscopes/` contains active calibration work for PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT, PRIVATE_INSTRUMENT
- Live AS2 control path (`AS2.py`) talks to real hardware — treat with care
- No formal fitting API yet — fitting is done ad-hoc in instrument scripts

## Ecosystem connections

- **sea_eco** — optional dependency; seashells.py wraps SEASerializable for .sea file I/O
- **sea_pearl** — no direct dependency; AS2.py provides a parallel lower-level control path
- **PoseiTEM** — no dependency declared; could consume rayTEM calibration data in the future

## Notes location

- `src/pySEA/ai_wiki/raytem/notes/shared/` — settled design decisions
- `src/pySEA/ai_wiki/raytem/notes/<contributor>/` — individual working notes
