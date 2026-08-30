# TODO — covariance propagation and aberration resolution example

**Branch/worktree:** `covariance_propagation` (rayTEM), branched from `dev`;
merged into `dev` as pieces land.
**Plan:** [PLAN_2026-08-30_covariance-propagation.md](PLAN_2026-08-30_covariance-propagation.md)
— Eric's handoff plan, adapted to rayTEM's 6-column state (the plan's 4×1
`(x, y, u_x, u_y)` becomes `(x, xt, y, yt)` inside the existing 6×6 Σ) and
extended with chromatic aberration, both at Eric's direction.

- [x] `MomentClosure` / `GaussianMomentClosure` / `CovarianceBeam` — make the
      Gaussian assumption explicit and swappable instead of hard-coded inside
      `_aberration_moment_pieces`. Additive: `propagate_moments(mu, Sigma)`
      keeps working, closure injected and defaulted.
- [x] Chromatic: `Source` energy spread (seeds `Σ[E,E]` and samples the `E`
      column), an element chromatic coefficient separate from `aberrations`,
      the bilinear covariance term `ΔΣ[xt,xt] = κ²σ_δ²σ_x²`, and the matching
      per-ray kick so a Monte-Carlo reference exists.
- [x] Resolution quantities: emittance, `Σ_rr`/`Σ_uu` eigen-decomposition,
      momentum block `k₀²Σ_uu` — reported per plane, not a single probe
      diameter.
- [x] Example `08_covariancePropagation.py`: four cases (ideal / OL1 / OL2 /
      both) plus the chromatic overlay, from one fixed source definition;
      the plots the plan lists.
- [x] Tests: exact linear transport, emittance conservation, zero-aberration
      recovery, order scaling in aperture angle, symmetry, PSD, retained mean
      shifts, single-case recovery from the combined case, chromatic scaling
      and its Monte-Carlo cross-check.
- [x] Docs + wiki sync; suite green; merged to `dev`.

**Outcome.** Beyond the plan's list, three defects surfaced and were fixed:
the ensemble mean was being taken from the centroid ray (so an even-order
aberration's mean shift was silently absorbed into the width); the old
per-axis closure dropped cross-plane terms that are the same size as the
retained ones on any x-y coupled beam; and `apply_aberrations=False` did not
silence chromatic, which would have contaminated every ideal reference run.
Generalising the closure also removed the "orders above three are not closed"
limitation entirely — every Krivanek order through C56 now reaches the
covariance mode, rotated terms included.

Answer to the plan's decision point, measured on `basic_column`: Gaussian
closure is **sufficient** here and no higher-moment machinery is needed. The
aberration's share of the angular variance is `f = 2.4e-4` at OL1, so the
excess kurtosis the closure discards is `27 f^2 = 1.6e-6`, and the OL1-OL2
interaction residual is +0.45% of the summed emittance growth.
