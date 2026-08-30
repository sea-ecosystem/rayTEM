# TODO — analytic aberrations: zone-ABCD frame + Gaussian-closure covariance

**Branch/worktree:** `finishing_propagation_additions` (rayTEM), merged into
`dev` on every commit.
**Plan:** none (Eric's direction, from the "aberrations can be analytically
added" discussion).

- [x] Element.zone_power_shift(h): per-axis power change ΔP(h) an aberrated
      element applies to the pupil zone at height h, from the existing
      analytic kick (deflection_at / h) — no new per-term code.
- [x] focal_surface(method='frame'): closed-form aberrated focal surface by
      zone-modified ABCD (each aberrated element's block rebuilt as
      drift·kick(P+ΔP(h_local))·drift about its principal planes); matches
      the traced surface on thin lenses exactly.
- [x] Covariance: Gaussian-closure aberration update in propagate_moments —
      linear kicks (C10, aligned C12) exactly; cubic (C30) via Isserlis
      closure; thick bodies at mid-body; apply_aberrations=False untouched.
- [x] Tests: closure vs Monte-Carlo ray statistics; aberrated waist shift;
      frame surface vs traced focal_surface vs closed form; idle bit-for-bit.
- [x] Wiki/docs sync; suite green; merged to dev.
