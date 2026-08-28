# Getting Started

rayTEM simulates a TEM column as a list of optical elements. You describe the
instrument once — sources, lenses, apertures, deflectors, drifts — and
propagate a beam through it as geometric rays, as a beam-envelope covariance,
or as a coherent wave, all from the same geometry.

Install in editable mode while developing:

```bash
pip install -e .
```

Optional extras: `pip install -e .[dev]` for the test suite, and
`pip install -e ".[dev,ai-wiki]"` to add the shared `sea-ecosystem` wiki
tooling (`pysea-refresh-wiki`).

## The minimal column

```python
from pySEA.rayTEM.elements import Gun, Drift, Lens
from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection

column = Microscope(sections=[MicroscopeSection(name="col", elements=[
    Gun(name="G", voltage=200, beam_current=1e-9,          # 200 kV, 1 nA
        size=(2.5e-6, 2.5e-6), np_xy=(5, 5),               # 5x5 ray grid
        angle=(1e-4, 1e-4), na_xy=(3, 3)),                 # 3x3 angle fan
    Drift(length=0.05),
    Lens(name="L1", strength=700.0, length=8e-5),
    Drift(length=0.10),
])])

rays = column.propagate_ray()        # (n_planes, n_rays, 6) geometric rays
column.show()                        # the ray diagram
```

Elements stack sequentially, or by explicit `position` within their section —
gaps are filled with drifts automatically. `Gun` is a `Source` under the
microscope's name for it; either works.

## The standard column

A complete, realistic 200 kV template ships with the package: a gun with a
stated 1 nA current, three condensers, the condenser aperture CA, an
objective pair around a mid-gap sample plane, four projector lenses, and a
named detector plane.

```python
from pySEA.rayTEM.assemblies import load_microscope
from importlib.resources import files

path = files("pySEA") / "rayTEM" / "microscopes" / "basic_column.sea"
scope = load_microscope(str(path))

scope.propagate_ray()
print(scope.beam_current)            # what survives to the detector, in amps
print(scope.convergence_angle)       # semi-angle at the sample plane, radians
```

## Saving and loading

Columns round-trip through sea-eco's `.sea` format — geometry, strengths, the
stated current, and any propagation results already on the object:

```python
scope.to_sea("my_column.sea")
back = load_microscope("my_column.sea")
```

Lens *settings* (a named set of strengths, not a new file) go through the
settings mechanism:

```python
scope.save_as_setting("my-state", {"C1": "strength", "C2": "strength"})
fresh = load_microscope("my_column.sea")
fresh.load_setting("my-state")       # reads settings/my-state.json
```

## Where to go next

- [Propagation modes](propagation_modes.md) — the four ways one column
  propagates, and when to use each.
- [Operating the column](operating_the_column.md) — currents, probes,
  conjugate planes, and the eight canonical operating states.
- `examples/` — runnable scripts, smallest first.
