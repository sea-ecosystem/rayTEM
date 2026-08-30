# TODO — covariance propagation and aberration resolution example

**Branch/worktree:** `covariance_propagation` (rayTEM), branched from `dev`;
merged into `dev` as pieces land.
**Plan:** [PLAN_2026-08-30_covariance-propagation.md](PLAN_2026-08-30_covariance-propagation.md)
— Eric's handoff plan, adapted to rayTEM's 6-column state (the plan's 4×1
`(x, y, u_x, u_y)` becomes `(x, xt, y, yt)` inside the existing 6×6 Σ) and
extended with chromatic aberration, both at Eric's direction.

- [ ] `MomentClosure` / `GaussianMomentClosure` / `CovarianceBeam` — make the
      Gaussian assumption explicit and swappable instead of hard-coded inside
      `_aberration_moment_pieces`. Additive: `propagate_moments(mu, Sigma)`
      keeps working, closure injected and defaulted.
- [ ] Chromatic: `Source` energy spread (seeds `Σ[E,E]` and samples the `E`
      column), an element chromatic coefficient separate from `aberrations`,
      the bilinear covariance term `ΔΣ[xt,xt] = κ²σ_δ²σ_x²`, and the matching
      per-ray kick so a Monte-Carlo reference exists.
- [ ] Resolution quantities: emittance, `Σ_rr`/`Σ_uu` eigen-decomposition,
      momentum block `k₀²Σ_uu` — reported per plane, not a single probe
      diameter.
- [ ] Example `08_covariancePropagation.py`: four cases (ideal / OL1 / OL2 /
      both) plus the chromatic overlay, from one fixed source definition;
      the plots the plan lists.
- [ ] Tests: exact linear transport, emittance conservation, zero-aberration
      recovery, order scaling in aperture angle, symmetry, PSD, retained mean
      shifts, single-case recovery from the combined case, chromatic scaling
      and its Monte-Carlo cross-check.
- [ ] Docs + wiki sync; suite green; merged to `dev`.
