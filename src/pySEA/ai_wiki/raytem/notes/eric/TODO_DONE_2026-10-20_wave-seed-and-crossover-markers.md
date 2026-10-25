# TODO — Coherent wave seed, crossover markers, and a focus on a body's exit face

**Branch/worktree:** `wave-seed-and-crossover-markers` (rayTEM), merged into `dev`
together with sea-pearl `emulator-condenser-balance`.
**Plan:** none. Context: sea-pearl's re-laid-out emulator column (half-lens
objective, sample on OL1's exit face; 4f corrector relays), and
[TODO_ACTIVE_2026-10-15_wave-probe-at-sample.md](TODO_ACTIVE_2026-10-15_wave-probe-at-sample.md).

Method unchanged: the thick lens stays a quadratic-index segment in the scaled
frame, crossed with the hybrid flatten -> cross -> rediverge policy. No ABCD /
Collins construction (rejected in PLAN_2026-08-19_scaled-fresnel-wave.md).

- [x] `Source(wave_size=..., wave_frame_curvature=...)`: opt-in coherent seed
      width and frame curvature (defaults keep the old seed)
- [x] drop a stale crossover marker that lies beyond a converging axis's own
      crossover, in free space and inside a body
- [x] a zero on a body's exit face counts as interior (look-ahead margin)
- [x] `segment_zero` tolerance relative, not floored at 1 pm: the floor hid the
      exit-face zero once < 1 um of body remained, and the last step landed on
      the singularity (dtau ~ 1e13; with `absorb`, ~1e12 sub-steps = the hang)
- [x] test: half lens focusing onto its exit face gives the Airy zero, with and
      without the absorber; 315 tests pass
- [x] emulator check (sea-pearl): live scaled wave 4.8 s, probe FWHM 44.5 pm vs
      42.6 pm Airy at 4 pm pixels, spectrum 29.9 of 30 mrad, nothing at the edge
- [x] docs so neither the bug nor the Collins proposal recurs: Into the SEA-weeds
      (`wave-optics-sampling.md`: "A focus on a body's exit face", "Design decision:
      thick lenses stay segments"; two invariants in `dev_documentation.md`), the
      Propagation Modes guide (immersion objectives), CLAUDE.md/AGENTS.md, wiki
      `waveoptics.md`; docs rebuilt (11 known warnings only), wiki refreshed
- [x] merge into `dev` with sea-pearl's branch; close this note
