# Example Scripts

The examples live in `examples/` at the repository root, smallest first, and
every one runs headless (`MPLBACKEND=Agg python <script>`). They are scripts
rather than notebooks — each is also exercised by the test suite where it
makes framework claims, so they cannot silently rot.

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
