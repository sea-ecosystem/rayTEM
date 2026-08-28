# Operating the Column

A column's *geometry* is fixed; an operating *state* is a set of lens
strengths. This page covers the measurements that define a state and the
eight canonical states of the standard column.

## Measurements

```python
scope.propagate_ray()
scope.beam_current            # amps at the exit -- derived from the Source's stated current
scope.current_at(plane)       # the same, at any logged plane
scope.convergence_angle       # SEMI-angle at the sample: max total angle among live rays
scope.convergence_angle_at(z) # at any plane
scope.conjugate_planes()      # image and back-focal (diffraction) plane families
scope.beam_waists()           # the covariance mode's minimum-width planes
```

Two conventions worth knowing: currents are in **amps**, stated exactly once
(on the `Source`/`Gun`) and derived everywhere downstream; and convergence
angles are **semi-angles** of the *total* deflection — a thick lens rotates
the ray by its Larmor angle while focusing, so the x-component alone
under-reports by `cos(KL)`.

## The eight canonical states

Three binary choices, each solved from a transfer-matrix condition on the
standard column — never hand-tuned:

1. **current** — C1 images the gun crossover onto the condenser aperture
   (`high`: everything passes) or overfocuses so the diverging beam overfills
   it (`low`: most of the current is cut).
2. **probe** — the condensers alone (the objective is never retuned) form a
   convergent probe at the sample (`B(source→sample) = 0`, target 30 mrad)
   or nearly parallel illumination (`D = 0`).
3. **detector** — the projector relay puts an image of the sample
   (`B(sample→detector) = 0`) or a diffraction pattern (`A = 0`) on the
   detector.

The division of labor is a rule, not a habit: probe formation belongs to the
condensers, projection to the projectors, and the objective currents do not
change between states (a test pins this).

Each solved state ships as a settings file:

```python
scope.load_setting("basic_column - high-convergent-image")
```

`examples/07_eightConfigurations.py` re-solves all eight from scratch,
propagates each with rays, moments, and the scaled wave, and prints the
cross-method plane table. Run it with `--fast` to skip the wave runs.

## Common recovery paths

- **"no ray carries intensity at z=..."** — an upstream aperture blocked the
  whole fan; measure upstream of it, or open the aperture.
- **A probe solve reports `[limited: ...]`** — the target angle is out of
  reach with the frozen objective; the message carries the reachable maximum.
  The knobs that raise it are geometry (OL1's focal length relative to the
  sample position), not condenser strength.
- **Coarse z sampling in plots** — propagation logs one plane per element;
  use `scope.subdivided(dz)` (or `show(zpts=...)`) for denser sampling
  without touching the optics.
