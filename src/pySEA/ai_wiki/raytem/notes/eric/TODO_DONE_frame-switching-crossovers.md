# TODO — Scaled-frame switching through crossovers (full-column scaled propagation)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-20_frame-switching-crossovers.md](PLAN_2026-08-20_frame-switching-crossovers.md)

- [x] Step 1 — `change_scaled_frame` (Eric's Eq 5, physical-grid-continuous,
      pointwise; optional band-limited resample for explicit s_new) +
      `min_representable_curvature`; `factor_wave`/`reconstruct_physical_wave`
      delegate as the (1,∞) special cases; frame terminology in errors;
      identity + guard tests
- [x] Step 2 — hybrid free-segment engine `propagate_free_scaled_hybrid`
      (converging → flatten at |R_flat| = R²/(A·s²) → flat through the real
      focus, crossover plane logged → re-diverge at d = A·s²); `z_cross_m`
      metadata; through-focus equivalence + electron-scale Airy tests
- [x] Step 3 — API consolidation: one `propagate_wave(..., mode=
      'fixed'|'scaled'|'hybrid')` on Element/Section/Microscope (remove
      propagate_wave_scaled), `Source.wave(mode=...)`, dispatcher kinds
      wave / wave-scaled / wave-hybrid via forced kwargs
- [x] Step 4 — driver wiring: `log` threading so flatten/crossover/rediverge
      planes land in `.wave_scaled` (with a per-plane `frame` companion +
      tags), `Microscope.crossovers`; full-column basic_column test; suite
- [x] Step 5 — full-column demo (focal-plane x-y slices), docs + wiki
      terminology sweep, close GitHub issue #2, finish protocol
