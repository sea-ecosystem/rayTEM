# TODO — honest thick-lens treatment in the scaled path (+ wave rotation)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (physics + measurement in the LOG entry and
[PLAN_2026-08-21_matrix-conjugate-planes.md](PLAN_2026-08-21_matrix-conjugate-planes.md) §5b)

A thick magnetic lens is a quadratic-index **medium**, not a phase screen: in
the scaled frame it is a *segment* whose s(z) law is sinusoidal rather than
linear, and U needs no screen and no kick — it just propagates over that
segment's own dtau. The current path splits it as drift L/2 -> thin kick
P=K sin(KL) -> drift L/2, which misplaces every crossover (measured: C1 by
422.3 um, exactly as predicted; up to 4.8 mm downstream once magnified).

- [x] `waveoptics.scaled_delta_tau_lens(dz, s0, R0, K)` — closed form
      `[tan(KL - phi) + tan(phi)]/(K C^2)`
- [x] `waveoptics.propagate_thick_lens_scaled(...)` — frame (s, R) via the
      element's own cos/sin 2x2, U over dtau; raise actionably if s -> 0
      inside the body (mid-element frame switching is a separate follow-up)
- [x] element seam: `Element.scaled_segment()` (None) / `Lens.scaled_segment()`
      (quadratic, K) so `L == 0` keeps the thin-lens path and `L > 0` takes
      the segment path — both kept
- [x] tests: thick-lens crossover now matches the ray/analytic value; thin
      lens bit-for-bit unchanged; energy; dtau vs numerical integral
- [x] wave rotation through the thick lens (Larmor -K*L), band-limited via
      Fourier shears; validate the sign against the ray path's rotation
- [x] docs/wiki + protocol finish
