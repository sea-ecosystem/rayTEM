# TODO — Simulation CLSIDs on rayTEM's emitted containers

**Branch/worktree:** `simulation-clsids` (rayTEM, from `dev`)
**Plan:** [PLAN_2026-09-21_simulation-clsids.md](PLAN_2026-09-21_simulation-clsids.md)
(the microscope-class proposal); ecosystem plan in sea-sand
`notes/eric/PLAN_2026-09-21_identity-and-container-clsids.md`.

Eric: every Signal / SignalSet rayTEM emits is a simulation, but they all
inherit sea-eco's default container id. Class them at the seam so a `.sea`
from rayTEM says so in its SEAID, and propose how the *microscope* itself
should be classed (it is an instrument model, not a container).

- [ ] `seashells.simulation_provenance(clsid)` — guarded sea-sand import,
      `None` when sea-sand is absent (same degrade-gracefully style as
      `sea_available`)
- [ ] pass `Provenance=simulation_provenance("SS101")` at every `_Signal(`
      and `"SSS01"` at every `_SignalSet(` in seashells.py (13 + 2 sites)
- [ ] test: a wavefield Signal and a rays SignalSet carry `SS101` / `SSS01`
- [ ] wiki seashells.md entry for the helper
- [ ] proposal note for classing `Microscope` (TWN / M0D), not implemented
- [ ] suite green before and after; push `simulation-clsids`
