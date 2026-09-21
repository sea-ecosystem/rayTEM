# PLAN — Classing rayTEM's simulated microscopes (proposal, not implemented)

**Owner:** Eric. **TODO:** [TODO_DONE_simulation-clsids.md](TODO_DONE_simulation-clsids.md)
(rayTEM). Ecosystem context: sea-sand
`notes/eric/PLAN_2026-09-21_identity-and-container-clsids.md`.

## What is already done

The *containers* rayTEM emits are classed at the seam: `seashells.py` stamps
`SS101` (Simulation Signal) on every `Signal` and `SSS01` (Simulation
SignalSet) on every `SignalSet` it builds, via `simulation_provenance()`.
That covers wavefields, scaled wavefields, rays, covariance, and phase
screens. Nothing else in rayTEM should mint container ids.

## The open question: what class is a `Microscope`?

`assemblies.Microscope` (and `MicroscopeSection`, and every `Element`) is a
`SEASerializable`, so it serialises to `.sea` and carries a `Provenance`
slot, but nothing assigns one — today a saved column has sea-eco's
unclassified default. A `Microscope` is not a container of data; it is a
**simulated instrument**: a column model that *produces* `SS*` data the way a
real `TEM04` produces `ES*` data. Its class should say that.

### Options considered (registry checked 2026-09-21 via `read_registry('clsid_prefixes')`)

1. **A new Instrument-category prefix for simulated columns.** Prefixes are
   three Crockford characters (no `I`, `L`, `O`, `U`), so "S + TEM" does not
   fit; `STM` is taken (scanning tunneling microscope); `V1R` ("virtual
   instrument") is free but would be a fourth way of spelling "not real" next
   to the container `S` role, `TWN`, and `M0D`. Not recommended.
2. **`TWN` — Digital twin** (Container category, `proposed`, "Twin artifact
   referenced in the sea-sand usage docs"). Exactly what a column built from
   a real instrument's lens positions and AS2 strengths is: a model *of a
   specific registered instrument*.
3. **`M0D` — Model** (Software category, `proposed`, "Fittable model
   definition"). The right class for a column that models no particular
   instrument — a generic template whose parameters exist to be varied or
   fitted, not to mirror a machine.

`TWN` and `M0D` are both already registered, differ on precisely the axis
that matters (does this column stand for a real instrument or not), and
leave the Instrument category to physical machines. No new prefix needed.

## Recommendation

- **`TWN01`** for a `Microscope` derived from a real instrument (the MACSTEM
  builder; any future column read from an instrument's lens table / AS2 xml).
  The twin's metadata should also record the real instrument's CLSID
  (`TEM0x`) so the pairing is queryable; the SEAID itself stays `TWN01`.
- **`M0D01`** for generic template columns (`basic_column`,
  `objective_section`) and for any `Microscope()` built ad hoc in a script or
  test — i.e. the constructor default.
- **`SS*`** stays on every emitted field, as now. A section (`MicroscopeSection`)
  and an element are parts of one model, not instruments in their own right:
  they should inherit the parent's class (`TWN`/`M0D`) rather than mint one
  of their own, mirroring how a lens is part of `TEM04`, not a `TEM` itself.

Both `TWN` and `M0D` are still `proposed` in the registry; adopting them here
is a reason to confirm them (sea-sand `add_prefix` / status flip), not a
blocker — prefix validation is opt-in, so they mint today.

## Construction sites that would carry each class

`M0D01` (default, minted in the constructor):
- `src/pySEA/rayTEM/assemblies.py:991` — `Microscope.__init__` (class
  `Microscope` at `:980`); the one place to add a `Provenance=` default so
  every ad-hoc column is classed.
- `src/pySEA/rayTEM/microscopes/basic_column.py:279` —
  `Microscope(name="basic column", ...)` (template).
- `src/pySEA/rayTEM/microscopes/objective_section.py:125` —
  `Microscope(name="objective section", ...)` (template).

`TWN01` (explicit, at the builder):
- `src/pySEA/rayTEM/microscopes/MACSTEM/builder.py:56` —
  `microscope = Microscope(sections=sections)`, built from `lens_positions.txt`
  and `AS2restore_20260103.xml`; pass `Provenance=<TWN01 id>` and record the
  MACSTEM `TEM` CLSID in metadata.

Inherit the parent's class (no id of their own):
- `src/pySEA/rayTEM/assemblies.py:203` — `MicroscopeSection.__init__`.
- `src/pySEA/rayTEM/elements.py:843` — `Element` and subclasses.

## How it would be wired (when implemented)

Mint through the same seam, not in `assemblies.py`: extend
`seashells.simulation_provenance(clsid)` use to `"M0D01"` / `"TWN01"` and
have `Microscope.__init__` accept `Provenance=None` → `simulation_provenance("M0D01")`.
This keeps the "seashells is the only sea-eco/sea-sand seam" invariant and
the `sea_available` degrade path intact. `safeReinstantiate` copies
`__dict__` back on reload, so an id stored in a `.sea` survives a round trip
unchanged; a column re-homed from template to twin should use sea-sand's
`reclassify_seaid` and keep its date/time/random tail.
