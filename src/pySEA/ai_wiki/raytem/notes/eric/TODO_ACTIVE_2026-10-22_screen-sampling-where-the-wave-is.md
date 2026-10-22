# Screen sampling judged where the wave is

**Branch/worktree:** `dev` (main checkout)
**Plan:** none (Eric, 2026-10-22: "make sure that the geometric and wave propagation continue to work", while the
sea-pearl emulator's corrector plate gains writable coefficients to fifth order)

The scaled path's `_check_screen_sampling` refuses a real screen whose phase steps >= pi anywhere on the grid. A
coefficient folded onto the objective is evaluated out to the grid corner, far past the beam, where a fifth-order
term (or a few um of C30) steps hundreds of radians over a field that is zero: the guard refuses a propagation
whose wave is well sampled. Judge the step only where the field has amplitude.

- [ ] `_check_screen_sampling(chi, name, field=None)`: with a field, only steps between samples where
      |U| >= 1e-3 max|U| count; the existing undersampled-screen refusal still fires
- [ ] both call sites pass the field; tests: a steep phase beyond the beam passes, one inside is refused
- [ ] objective_section's note on the grid corner updated
