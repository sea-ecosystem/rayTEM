# Screen sampling judged where the wave is

**Branch/worktree:** `dev` (main checkout)
**Plan:** none (Eric, 2026-10-22: "make sure that the geometric and wave propagation continue to work", while the
sea-pearl emulator's corrector plate gains writable coefficients to fifth order)

The scaled path's `_check_screen_sampling` refuses a real screen whose phase steps >= pi anywhere on the grid. A
coefficient folded onto the objective is evaluated out to the grid corner, far past the beam, where a fifth-order
term (or a few um of C30) steps hundreds of radians over a field that is zero: the guard refuses a propagation
whose wave is well sampled. Judge the step only where the field has amplitude.

- [x] `_check_screen_sampling(chi, name, field=None)`: with a field, only steps between samples where
      |U| >= 1e-3 max|U| count; the existing undersampled-screen refusal still fires
- [x] both call sites pass the field; tests: a steep phase beyond the beam passes, one inside is refused
- [x] objective_section's note on the grid corner updated

**Outcome:** the emulator's corrector plate carries C30 of several um and fifth order through every propagation
kind. The old refusal test (C30 1 mm at 128 px) was the corner: in the beam its steps are 2.4 rad and its focus
matches 256 px to 0.3% and 512 px to 0.7%; it now runs at 64 px, where the beam itself aliases and is refused.
Suite: 315 passed; the 5 `test_elements_sections_microscopes` failures are the pre-existing ones (they fail on HEAD).
