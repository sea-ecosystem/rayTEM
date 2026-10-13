# TODO — The physical plane of a stored scaled wave

**Branch/worktree:** `scaled-wave-axis-frames` (from `dev`, merged back into `dev`).
**Plan:** Poseidon `src/pySEA/ai_wiki/poseidon/notes/eric/PLAN_2026-10-13_scaled-wave-axis-frames.md`

**Goal:** `seashells.scaled_wave_physical_plane(scaled_wave, plane)` returns
the physical wavefield ψ of one plane on its native grid (x = s·ξ, pitch
|s|·Δξ, ψ = (1/s)·U·e^{ik r²/2R}). It reads a scaled-wave SignalSet as it is
stored or exported, with no live model.

**Why:** Eric, 2026-10-13. Poseidon's scaled-wave plot should offer a
calibrated frame whose axes redraw with z. Only rayTEM knows the frame, and
`wavefield_at` needs a live model, while a viewer holds an exported `.sea`.

## Steps

- [x] `scaled_wave_physical_plane` in `seashells.py`, reusing `reconstruct_physical_wave` and `make_wavefield_signal`
- [x] tests: it matches `Microscope.wavefield_at`, isotropic and anisotropic companions, and refuses what is not a scaled wave
- [x] wiki and docs
