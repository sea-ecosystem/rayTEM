# TODO — Normalised scaled axes and overlays on the wave stacks

**Branch/worktree:** `overlays-and-axis-frames` in sea-eco, rayTEM, Poseidon and sea-pearl (from `dev`, merged back into `dev`).
**Plan:** Poseidon `src/pySEA/ai_wiki/poseidon/notes/eric/PLAN_2026-10-13_overlays-and-axis-frames.md`

**Goal:** Store ξ, η in a.u. (−1…1 over the source grid) with s(z) in metres, and give `.wave` and U the column's z marks as overlays.

**Why:** Eric, 2026-10-13: the scaled unit is normalised, not a distance; and a β wave plot should be the Signal's own `.show()`, marks included.

## Steps

- [x] `make_scaled_wave_signalset`: a.u. axes, s in m, `reference_half_width_m`; readers accept legacy files
- [x] `.wave` and U carry section/element (and crossover/image) marks bound to their z axis
- [x] tests, docs, wiki
