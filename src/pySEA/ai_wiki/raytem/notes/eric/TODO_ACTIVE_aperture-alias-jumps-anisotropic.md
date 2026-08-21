# TODO — alias-free aperture, beam-support policy, frame jumps, anisotropic frames

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-21_aperture-alias-jumps-anisotropic.md](PLAN_2026-08-21_aperture-alias-jumps-anisotropic.md)

- [x] Step 1 — alias-free sampling of the exact hard aperture Θ(a−r): the
      "gridding" Eric spotted at the sample/detector is folded above-Nyquist
      edge content (real Fresnel fringes stay — Eric: keep the physics);
      `bandlimited_disk` from the analytic J1 spectrum (+ supersampled
      area-coverage masks for mid-column Apertures); tests + demo rerun
- [x] Step 2 — beam-support frame policy: guard + flatten/re-diverge
      thresholds measured over the beam's actual support instead of the empty
      grid edge (fixes the padded-grid s_min crash; flattens earlier)
- [ ] Step 3 — direct converging→diverging frame jumps (crossover='jump'):
      mirror R_o=−d → R_n=+d at half the flatten threshold, no flat window;
      parametrized equivalence tests; default chosen from measured error
- [ ] Step 4 — anisotropic frames (s_x, s_y, R_x, R_y, τ_x, τ_y): quadrupoles
      absorb (P, −P) into curvature like round lenses; per-axis line-focus
      crossovers; isotropic behavior bit-for-bit
- [ ] Step 5 — docs + wiki sync; finish protocol
