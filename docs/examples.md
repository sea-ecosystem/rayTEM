# Example Scripts

The examples live in `examples/` at the repository root, smallest first, and
every one runs headless (`MPLBACKEND=Agg python <script>`), writing its
figures into `examples/figs/` — resolved from the script, so any working
directory works. They are scripts rather than notebooks (05 is a marimo
notebook: `marimo edit examples/05_planeComparison.py`, or
`marimo export html` to render it headless). Each is also exercised by the
test suite where it makes framework claims, so they cannot silently rot.

They draw through the framework's own renderers rather than rebuilding them:
`postprocessing.plot2D`/`plot3D` for ray diagrams,
`Microscope.show(kind=...)` for a propagation result — which delegates to the
result `Signal`'s own `.show()` for a single plane, and to the shared
scaled-wave cross-section otherwise — and `Signal.show()` directly for a
Signal or a slice of one. Raw matplotlib appears only where an example
*overlays* something on a rendered panel, or plots a quantity that is not a
Signal.

| script | what it shows |
|---|---|
| `01_basicRays.py` | build a section by hand, trace rays, the 2D/3D ray plots |
| `02_basicFitting.py` | fitting lens strengths to measured behavior |
| `03_lensRotation.py` | Larmor rotation through thick lenses |
| `04_scaledWave_basic_column.py` | the scaled-Fresnel wave through the full standard column |
| `05_planeComparison.py` | ray / matrix / wave agreement on conjugate planes |
| `06_aberratedObjective.py` | an aberrated objective: rays, wave, focal surface, Strehl |
| `07_eightConfigurations.py` | the eight canonical operating states, solved and cross-checked |
| `08_covariancePropagation.py` | where the resolution goes: four aberration cases, no rays traced |
