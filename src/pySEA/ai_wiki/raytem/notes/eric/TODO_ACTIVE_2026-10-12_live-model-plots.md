# TODO — Display Signals for live model plots

**Branch/worktree:** `live-model-plots`, cut from `dev` and merged back into `dev`; notes commits on `dev`.
**Plan:** Poseidon `src/pySEA/ai_wiki/poseidon/notes/eric/PLAN_2026-10-12_live-model-plots.md` (the shared plan for this slug)

**Goal:** derived calibrated Signals for the ray diagram, beam envelope and scaled-wave cross-section, carrying section/element/plane markers as sea-eco annotations.

**Why:** Eric, 2026-10-12: Poseidon's Control Window β will plot the live model; rayTEM draws these only in matplotlib, and the cross-section array is computed and thrown away.

## Steps

- [x] `ray_diagram_signal` on Microscope and MicroscopeSection (rotating frame, aperture ghosting, Span/Line markers)
- [ ] `envelope_signal` from the covariance stack
- [ ] `wave_cross_section_signal`; `show(kind="wave-scaled")` draws it (figure unchanged)
- [ ] tests on the basic column
- [ ] docs, wiki, refresh
