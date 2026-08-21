# Plan — closed-form conjugate planes from the transfer matrices

**For:** the other contributor (owner of the ray / transfer-matrix side)
**From:** Eric (+ Claude)
**Related:** `TODO_DONE_conjugate-planes.md`, LOG entry 2026-08-21 "Conjugate planes"

## 1. What the current method does (confirmed by reading the code)

`postprocessing.findPlanes` (line 392) locates both plane families by **pairwise
ray difference**, not from the matrices:

- it infers four reference rays from the initial conditions — two *diffraction*
  rays (`xt = 0`, `x = ±a`) and two *image* rays (`x = 0`, `xt = ±b`);
- for each consecutive pair of **logged planes** it calls `whereRaysCross`,
  which linearly interpolates both rays and solves
  `ma·dz + ya0 = mb·dz + yb0  ->  dz = (yb0 - ya0)/(ma - mb)`;
- it returns a **fractional plane index** `i-1+dz`, converted to metres by
  `zFromFractional`;
- magnification is a second interpolation using the *other* ray pair.

So: two rays per family, find where their difference is zero, evaluated between
sampled planes. Nothing in `postprocessing.py` uses `transfer_matrix()`.

## 2. Why the matrix form is the *same physics*

For the ± pairs the ray difference **is** a matrix element. With the accumulated
x-block `M(z) = [[A, B], [C, D]]` acting on `(x, xt)`, i.e.
`x_out = A·x_in + B·xt_in`:

- diffraction pair (`xt = 0`, `x = ±a`):  `x_a - x_b = 2a·A`  ->  zero iff **A = 0**
- image pair (`x = 0`, `xt = ±b`):        `x_a - x_b = 2b·B`  ->  zero iff **B = 0**

Identical roots — the present code is already evaluating A and B, implicitly,
through two rays. (The wave side does the same thing with one column of M
without saying so: the scaled frame's curvature is `R = -A/C`, so its crossover
`z + |R|` is exactly the `A = 0` root.)

**Notation, since it caused confusion:** `A, B, C, D` here are the ordinary
**ray-transfer (ABCD) matrix** elements — the 2x2 x-block of rayTEM's existing
6x6 `transfer_matrix()` (rows/cols 0,1 of `convention = [x, xt, y, yt, z, E]`).
Nothing wave-optical about them:

| element | meaning | units |
|---|---|---|
| `A = d x_out / d x_in`   | position -> position (magnification-like) | — |
| `B = d x_out / d xt_in`  | angle -> position (camera-length-like)    | length |
| `C = d xt_out / d x_in`  | position -> angle (power-like, `-1/f`)    | 1/length |
| `D = d xt_out / d xt_in` | angle -> angle (angular magnification)    | — |

Drift `L` = `[[1, L], [0, 1]]`; thin lens `f` = `[[1, 0], [-1/f, 1]]`.

## 3. Proposal

Compute both families in closed form from the accumulated matrix — no ray array,
no interpolation, no dependence on logged-plane density:

- **in free space** after an element exit, A and B are linear in `dz`:
  `A(dz) = A0 + dz·C0`, `B(dz) = B0 + dz·D0`, so
  `dz_diffraction = -A0/C0` and `dz_image = -B0/D0`.
  Keep roots with `0 <= dz <= L_drift`; report the rest as **virtual** planes.
- **inside a thick element** of constant `K` (round lens, or a quad's focusing
  axis) the evolution is sinusoidal, so the root is still closed-form:
  `A(dz) = A0·cos(K·dz) + (C0/K)·sin(K·dz)  ->  tan(K·dz) = -K·A0/C0`.
  On a defocusing axis `cos/sin -> cosh/sinh`, giving
  `tanh(K·dz) = -K·A0/C0` (a root exists only when `|K·A0/C0| < 1`).
- **magnification falls out at the root**: at an image plane `M = A`; at a
  diffraction plane `B` is the camera-length factor. No second interpolation.
- per axis (x and y separately) for astigmatic optics; evaluate in the rotating
  frame for magnetic round lenses, as `findPlanes` already does.

## 4. What this buys

1. **Exact positions**, independent of how densely planes were logged.
2. **Catches planes inside thick lens bodies**, where linear interpolation
   between entrance and exit is the wrong functional form. Not hypothetical:
   every `basic_column` lens is a thick `QLens` (10-20 mm), and our wave
   crossovers sit **0.4-4.8 mm** from the `findPlanes` positions there, versus
   **0 nm** agreement on a thin-lens column.
3. **Cannot miss two crossings in one segment** (the sign-change test finds one).
4. **No dependence on the caller having built the right four rays** — the
   current heuristic warns and returns empty when it cannot infer them.
5. Returns **metres**, not fractional plane indices.
6. **Invertible**: "what lens strength puts a plane at z?" becomes closed-form
   instead of the numerical search `error_dz` is built around.

## 5. Non-goals

- `findPlanes` **stays**. Rays remain the ground truth and the cross-check: on
  thin-lens columns the two must agree to ~1e-12.
- No change to the ray convention, the 6x6 matrices, or the `diff`/`image`
  naming — this reuses that vocabulary exactly.

## 6. Verification

- **Thin 2-lens compound column** (collimated in, f1 = 45 mm, 100 mm gap,
  f2 = 30 mm): diffraction planes 55.0000 and 176.0000 mm — note the second is
  the *image of the first crossover*, **not** `z_L2 + f2` = 140 mm — and the
  image plane at 150.8621 mm. Matrix, `findPlanes`, and the hybrid wave
  crossovers already agree here to 0 nm.
- **Thick-lens column**: matrix and `findPlanes` should now *differ*, with the
  matrix version matching a finely subdivided ray reference.
- **basic_column**: publish the diff/image table both ways.

## 7. Open questions for you

1. Where should this live — `postprocessing`, next to `findPlanes`, or as a
   `Microscope` method? (There is a `Microscope.conjugate_planes` now, ray-traced
   via `findPlanes`; happy to re-point it at the matrix version.)
2. Do you want the closed-form inverse wired into `error_dz` / the fitting
   helpers, or kept separate?
3. Virtual planes (roots outside the segment): returned flagged, or dropped?

## 8. Two incidental findings (left untouched)

- `assemblies.py:937` (`Microscope.focus_error`) calls `findPlanes(self.rays, "x")`
  — `"x"` lands on the `R` (rotation) parameter and `axis` falls back to `"xy"`.
  Looks like a real bug.
- `postprocessing.whereCrossesZero` is used only by `findPlanes4` (~line 627),
  not by `findPlanes`.
