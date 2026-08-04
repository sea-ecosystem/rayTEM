# rayTEM — module responsibilities and invariants

## Module map

| Module | Responsibility | Imports from |
|--------|---------------|--------------|
| `elements.py` | Element base class, concrete types (Source, Drift, Lens, Dipole, Quadrupole, Aperture), ray column convention, `columnByName`, `fix_mat_dims` | `seashells`, numpy |
| `assemblies.py` | MicroscopeSection, Microscope; propagation, fitting, serialization, visualization | `elements`, `postprocessing`, `seashells`, numpy |
| `seashells.py` | Conditional sea_eco import; provides `SEASerializable` that gracefully degrades | sea_eco (optional) |
| `postprocessing.py` | Plot helpers (`plot2D`, etc.) | numpy, matplotlib |
| `utilities.py` | Shared mathematical helpers | numpy |
| `AS2.py` | `AS2querier` — HTTP client for Nion AS2 live control | requests |
| `xmlNion.py` | Parse AS2 XML configuration files into structured dicts | stdlib xml |
| `microscopes/` | Per-instrument calibration scripts; uses framework but is not imported by it | elements, assemblies, AS2 |

## Invariants

### 1. Ray column order is frozen

```
index: 0   1    2   3    4  5  6
name:  x   xθ   y   yθ   z  I  E
```

`columnByName(name)` is the only safe way to get a column index. Every Element's
`transfer_matrix` and every `propagate_ray` call relies on this order.

**Never reorder columns without updating every Element, `fix_mat_dims`, `fix_ray_dims`,
and all callers in `assemblies.py` and `microscopes/`.**

### 2. Transfer matrices are 8×8

`fix_mat_dims(m, columnNames)` inflates a smaller physics matrix (e.g. 2×2 thin lens)
into the 8×8 form. Elements must call `fix_mat_dims` — they must never store a
raw 2×2 and apply it to the full 8-column ray array.

### 3. Hierarchy is strictly bottom-up

- `Element.propagate_ray()` — single element
- `MicroscopeSection.propagate_ray()` — delegates to ordered Elements
- `Microscope.propagate_ray()` — delegates to ordered MicroscopeSections

Microscope does not reach inside Elements. MicroscopeSection does not reach inside
Microscope. Cross-level direct access is a violation.

### 4. seashells is the only sea_eco import point

All framework classes that need serialization inherit from `seashells.SEASerializable`.
Direct imports of `pySEA.sea_eco.*` from `elements.py` or `assemblies.py` are forbidden.
seashells.py owns the conditional import and provides a stub when sea_eco is absent.

### 5. microscopes/ is not imported by framework code

Scripts under `microscopes/` are instrument-specific applications; they import
framework modules but not vice versa. Framework modules (`elements.py`, `assemblies.py`,
etc.) must not import from `microscopes/`.

### 6. AS2.py talks to real hardware

`AS2querier.setLens()` sends HTTP PUT requests to a live instrument. Code paths that
call it must be clearly separated from simulation paths. Never call `setLens` inside
framework propagation logic.

## Allowed import directions

```
microscopes/   →  elements, assemblies, AS2, utilities, postprocessing
assemblies     →  elements, postprocessing, seashells
elements       →  seashells, utilities (numpy only from external)
AS2            →  (external: requests, numpy only)
seashells      →  sea_eco (optional, guarded)
postprocessing →  (external: numpy, matplotlib only)
utilities      →  (external: numpy only)
xmlNion        →  (external: stdlib xml only)
```

No upward imports. No cross-imports between `microscopes/` subdirectories unless
explicitly documented in a shared note.
