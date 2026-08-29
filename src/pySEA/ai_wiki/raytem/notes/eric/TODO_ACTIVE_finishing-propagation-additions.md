# TODO — finishing propagation additions

**Branch/worktree:** `finishing_propagation_additions` (rayTEM), merged into
`dev` on every commit; this branch is kept up to date with `dev`. The old
`Signal_and_propagation_additions_new` branch is deleted and must not be
recreated.
**Plan:** none

- [x] Branch created from `dev`; docs CI restricted to `dev`/`main` so the
      rebuild bot cannot diverge the work branch from `dev`.
- [ ] Thick-lens focal-power semantics: decide (with Eric/Thomas) whether the
      aberration pupil scale uses the EFL power K·sin(KL) or stays at
      1/focal_length = K·tan(KL); implement the decision.
- [ ] Example 06: re-derive the header narrative (C30 choice, Strehl,
      delivered fraction) against the current column and the power decision.
