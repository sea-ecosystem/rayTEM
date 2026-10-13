# TODO — Display Signals for live model plots

**Branch/worktree:** `live-model-plots`, cut from `dev` and merged back into `dev`; notes commits on `dev`.
**Plan:** Poseidon `src/pySEA/ai_wiki/poseidon/notes/eric/PLAN_2026-10-12_live-model-plots.md` (the shared plan for this slug)

**Goal:** derived calibrated Signals for the ray diagram, beam envelope and scaled-wave cross-section, carrying section/element/plane markers as sea-eco annotations.

**Why:** Eric, 2026-10-12: Poseidon's Control Window β will plot the live model; rayTEM draws these only in matplotlib, and the cross-section array is computed and thrown away.

## Steps

- [x] `ray_diagram_signal` on Microscope and MicroscopeSection (rotating frame, aperture ghosting, Span/Line markers)
- [x] `envelope_signal` from the covariance stack
- [x] `wave_cross_section_signal`; `show(kind="wave-scaled")` draws it (figure unchanged)
- [x] tests on the basic column
- [x] docs, wiki, refresh
- [x] (found in the live loop) `Rays.convert_to_rotating_reference_frame` vectorized: the per-ray loop deep-copied the Rays metadata through `__getitem__` and rebuilt `fix_mat_dims` per ray, ~1.5 s on U200 (57 planes × 225 rays), paid twice per ray diagram and by `plot2D`. Now ~2 ms, bit-identical on U200 and the basic column; `tests/test_rotating_frame.py` keeps the old loop as the reference. A U200 live ray export went from ~4 s to well under 1 s.
