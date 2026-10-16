# Column naming schema for microscope sections and elements

**Branch/worktree:** `column-naming-schema` (rayTEM); companion edits in sea-pearl on its working tree
**Ported:** 2026-10-16 from the uncommitted `rayTEM_original` working tree onto `column-naming-schema` off `dev` in `sea-ecosystem/rayTEM` (private-instrument names redacted)
**Plan:** none (designed in-session)
**Status:** Done — pending review

## Goal

Give rayTEM a controlled vocabulary for microscope section and element names,
plus a parser and a validator, so a column can be named consistently and
checked before it is built.

## Why

There was no naming schema anywhere in rayTEM or sea-pearl. Section and element
names were free-form strings, and had already drifted:

| Source | Section names |
|---|---|
| `sea-pearl` `default_usim_twin()` | `gun`, `monochromator`, `condensers`, `corrector`, `objective`, `projectors`, `detectors` |
| `sea-pearl` `Factory*` | `CLs`, `OLs`, `PLs` |
| `microscopes/MACSTEM/builder.py` | `condenser`, `objective`, `projector` |
| private instrument builder (private microscopes tree) | intended single letters, but the build loop reassigns `name` inside the element loop, so sections end up named `MC`, `OL2`, `P2` |

## Design

A **name-pattern** convention, not a new data model. The `name` string stays
the single source of truth, so `section["CL1"]` lookup, `.sea` round-trips and
existing microscope definitions keep working with no migration.

`Element.kind` was deliberately left alone. It is load-bearing — `assemblies.py`
branches on it for `ignoreLensThickness` and in `__delitem__` — and it describes
the transfer matrix, not the position in the column.

### Vocabulary

Sections: `G` gun, `M` monochromator, `C` condenser, `A` probe corrector,
`O` objective, `P` projector, `S` spectrometer, `D` detector volume.

Element types: `L` round lens, `D` dipole/deflector, `S` stigmator,
`Q` quadrupole, `A` aperture, `P` prism, `V` drift volume.

`A` and `C` are both section and type codes; position resolves it, since the
first letter is always the section (`AA1` is the corrector's first aperture).

### Two name forms

- **Host element** — `<SECTION><TYPE>[INDEX]`: `CL1`, `PL4`, `CA1`, `SP1`,
  `PV1`, `DV`.
- **Attached multipole** — `<HOST>_<TYPE>[INDEX][AXIS]`: `CL1_D1a`, `PV1_D1b`,
  `CL2_S1a`, `DV_D1a`. A deflector is named for the element it sits on.

The multipole letter is what the element's **class** must match, not the host's:
`CL1_D1a` is a `Dipole`; the `L` only records where it sits.

The older bare-axis form (`PD1_a`, `PSa`) still parses. An axis suffix is
lower-case and an attached multipole's type letter is upper-case, so the two
cannot be confused.

The vocabulary was reverse-engineered from names already in the tree, not
invented:

- `examples/01_basicRays.ipynb` already used `PL1`, `PD1_a` and `PSa`/`PSb` —
  the last two being `Quadrapole` objects used as a stigmator pair, which is
  where the `S` type letter comes from.
- sea-pearl's AS2 control map already used `CL{i}_1Da`, `PL{i}_2Db`,
  `PV1_1Da` — which is where `V` (drift volume) and the attached-multipole idea
  come from.

**Known divergence:** AS2 puts the index before the type (`CL1_1Da`); this
schema puts the type first (`CL1_D1a`). Eric chose type-first for consistency
with the host form. `ClientControl.client_name` absorbs the difference, which
is what it exists for, but it is a transposition to watch when debugging.

### Exemptions that keep existing scopes valid

- **Validation is scoped by section.** A section whose name resolves to no code
  is not schema-managed and its elements are never name-checked, so ad-hoc
  scopes (`MicroscopeSection(name="sec1")`) and existing test fixtures stay
  valid under `strict=True`.
- **`NAME_REQUIRED_CLASSES` is narrower than `ELEMENT_TYPE_CLASSES`.** `Drift`
  *may* be named (`PV1`, `DV` via type `V`) but is never *required* to be:
  drifts are auto-inserted, and named planes (`sample`, `VOA`, `CCD`) are
  exactly what they are for. `Source` is likewise never required.
- A name that **does** parse is still held to the schema even on an exempt
  class, so a `Drift` named `PV1` inside the `C` section is an error.

### Detectors

Detector *devices* are not column elements. A detector is a role resolved by
sea-pearl's `DetectorRoleRegistry` (bound at connect), with the acquire surface
coming from `detector_acquire_section`; `MicroscopeModel.detector_aliases` is
just a `dict[str, list[str]]` alias→candidates map. What belongs in the column
is the drift volume the beam crosses to reach them — `DV`, mirroring
`PV1`/`PV2`.

An earlier draft gave `D` "container" semantics so detector devices could sit
in the column unchecked. That was removed once the devices moved out; the
concept had no other user.

### Strictness

`validate_microscope(scope, strict=...)` takes `True` (raise `SchemaError`, the
default), `"warn"` (`UserWarning`), or `False` (return the report silently).
Findings carry a severity: only `error` fails validation, `info` records an
unmanaged section or an unnamed element without failing.

## What changed

rayTEM:

- [x] `src/pySEA/rayTEM/schema.py` — vocabulary, `parse_element_name`,
      `canonical_element_name`, `section_code`, `validate_element_name`,
      `validate_section`, `validate_microscope`, `describe_schema`, and
      `register_section` / `register_element_type` for site-specific codes
- [x] `src/pySEA/rayTEM/assemblies.py` — `MicroscopeSection.validate_schema()`
      and `Microscope.validate_schema()`, both thin wrappers
- [x] `src/pySEA/rayTEM/__init__.py` — export the schema surface
- [x] `src/pySEA/rayTEM/tests/test_schema.py` — 75 tests

sea-pearl:

- [x] `backend_nion.py` — `condenser_aperture` → `CA1`; the `detectors` section
      (three `ElementModel(kind="detector")` entries) replaced by a
      `detector volume` section holding `DriftModel(name="DV")`
- [x] `tests/unit/test_usim_twin.py` — updated column order, plus tests that the
      twin holds no detector devices and conforms to the schema
- [x] `skills/instrument-control-model-builder/SKILL.md` and
      `agents/instrument-model-builder.md` — name against the schema before
      building, and validate before any transport work

`MicroscopeModel` inherits `validate_schema` from `Microscope`, so sea-pearl
needed no model-layer code change.

## Findings on the existing models

All clean after the fixes above: `default_usim_twin()`, `FacotroyScope`, and
`macstem.json` (whose `CL1-3`/`OL1-2`/`PL1-4` already conformed, with
`gun`/`VOA`/`sample`/`CCD` exempt as named planes).

The private instrument's uncalibrated JSON reports no errors only because the builder bug leaves
its sections named `MC`/`OL2`/`P2`, which resolve to no code — so `C1`, `C2`,
`Diff`, `IM`, `P1`, `P2` go unchecked.

## Open questions

- [ ] **Control-layer deflector naming.** 16 `ControlGroup`s in `backend_nion.py`
      (`CL{1-3}_1D`, `CL{1-3}_2D`, `PL{1-4}_1D`, `PL{1-4}_2D`, `PV1_1D`,
      `PV2_1D`) carry leaves named `deflect_a`/`deflect_b`. Under the schema
      these read `CL1_D1` with leaves `a`/`b`, matching the `a`/`b` idiom the
      stage and aberration groups already use. 32 leaves; not attempted here.
- [ ] **`metadata["kind"]` is now redundant.** `LensModel(name="CL1",
      metadata={"kind": "condenser_lens"})` duplicates what `CL1` already says.
      Note this is *not* a conflict with `Element.kind` (they are separate
      fields) — the real inconsistency is that a generic `ElementModel` puts the
      semantic role in the real `kind=` field while a typed class has to push it
      into metadata, because `Lens.__init__` forces `kind='Thin lens'`.
- [ ] Fix the `name` reassignment bug in
      the private instrument's uncalibrated builder (private microscopes tree).
- [ ] Multipoles above quadrupole have no rayTEM class, so no type letter was
      reserved. Note the Gatan EELS driver already uses `S*` for sextupole and
      `O*` for octupole (`backends/eels.py`), which clashes with `S`=stigmator
      and `O`=objective here; `register_element_type` covers it when a class
      lands, but the letters need thought.

## Verification

```
rayTEM    pytest test_schema.py test_elements_sections_microscopes.py   88 passed
sea-pearl pytest tests/unit/test_usim_twin.py test_control_templates.py
                 test_backend_controls.py                              48 passed
```

sea-pearl's full unit suite has ~86 pre-existing failures (daemon/job timing and
`backend_raytem` `FileNotFoundError`); a clean-baseline run at `main` confirmed
the same count, so none are attributable to this work.
