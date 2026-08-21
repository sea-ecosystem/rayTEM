# Note — analytic crossover planes from the transfer matrix (A = 0 and B = 0)

**For:** the other contributor (owner of the ray / transfer-matrix side)
**From:** Eric (+ Claude)
**Scope:** how to locate the two families of special planes *analytically*, from the
accumulated transfer matrix, instead of by searching a traced ray bundle.
No code changed; no request to change the wave side.

## 1. Notation

`A, B, C, D` are the ordinary ray-transfer (ABCD) elements — the 2x2 x-block of
the existing 6x6 `transfer_matrix()` (rows/cols 0,1 of
`convention = [x, xt, y, yt, z, E]`), **accumulated** from the entrance plane to
the plane of interest, so that

```
x_out  = A*x_in + B*xt_in
xt_out = C*x_in + D*xt_in
```

| element | meaning | units |
|---|---|---|
| `A` | position -> position (magnification-like) | — |
| `B` | angle -> position (camera-length-like) | length |
| `C` | position -> angle (power-like; `-1/f` for a thin lens) | 1/length |
| `D` | angle -> angle (angular magnification) | — |

Drift `L` = `[[1, L], [0, 1]]`; thin lens `f` = `[[1, 0], [-1/f, 1]]`.

## 2. The criterion

Read straight off `x_out = A*x_in + B*xt_in`:

- **`A = 0` -> diffraction (back-focal / reciprocal) plane.** Output position
  no longer depends on input position, so rays that entered parallel converge.
- **`B = 0` -> image plane.** Output position no longer depends on input angle,
  so all rays from one object point reconverge.

These are the same planes the current pairwise-ray search finds — for the `+-`
reference pairs the ray *difference* is literally the matrix element
(`x_a - x_b = 2a*A` for the parallel pair, `2b*B` for the on-axis pair) — so the
existing method is already evaluating A and B, implicitly, through two rays.
The gain here is closed-form roots instead of interpolating between sampled
planes.

## 3. Closed-form roots

**In free space** downstream of an element exit, both are linear in `dz`:

```
A(dz) = A0 + dz*C0        ->   dz_diffraction = -A0/C0
B(dz) = B0 + dz*D0        ->   dz_image       = -B0/D0
```

Keep roots with `0 <= dz <= L_drift`; the rest are **virtual** planes (report or
drop, see Q3).

**Inside a thick element** of constant `K` the evolution is sinusoidal, so the
root is still closed-form. Composing the interior matrix with the accumulated
one, `A(dz) = A0*cos(K*dz) + (C0/K)*sin(K*dz)`, hence

```
focusing axis:    tan(K*dz)  = -K*A0/C0
defocusing axis:  tanh(K*dz) = -K*A0/C0     (root only if |K*A0/C0| < 1)
```

and the same with `(B0, D0)` for the image family. Take the first root in
`[0, L_element]`.

This is the case that matters most here: linear interpolation between a thick
lens's entrance and exit is the *wrong functional form* inside the body, and
every `basic_column` lens is a thick `QLens` (10-20 mm).

## 4. Free extras at the root

- **Magnification** needs no second pass: at an image plane `M = A`; at a
  diffraction plane `B` is the camera-length factor.
- **Sanity identities** for cross-checking against textbook formulas — for an
  accumulated system matrix,

```
1/f_system = -C
BFD (back focal distance)  = -A/C
FFD (front focal distance) = -D/C
```

  so `dz_diffraction = -A0/C0` is exactly the back focal distance, and the
  familiar doublet combination `1/f12 = 1/f1 + 1/f2 - d/(f1*f2)` is just `-C`
  for two thin lenses separated by `d`. Worked check: `f1 = 45 mm`,
  `d = 100 mm`, `f2 = 30 mm` gives `f12 = -54 mm` and
  `BFD = f12*(1 - d/f1) = 66 mm`, i.e. a crossover 66 mm past the second lens.

- **Inversion**: because the roots are closed form, "what lens strength puts a
  plane at z?" can be solved directly rather than by the numerical search
  `error_dz` is built around.

## 5. Verification cases

- **Thin 2-lens compound column**, collimated in, `f1 = 45 mm`, 100 mm gap,
  `f2 = 30 mm`, lens 1 at z = 10 mm: diffraction planes at **55.0000 mm** and
  **176.0000 mm** (the second is the *image of the first crossover*, i.e.
  `10 + 100 + BFD`, **not** `z_L2 + f2 = 140 mm`), image plane at
  **150.8621 mm**. `findPlanes` and the analytic values agree here to 0 nm.
- **Thick-lens column**: analytic and `findPlanes` should now *differ*, with the
  analytic version matching a finely subdivided ray reference.
- **basic_column**: publish the diff/image table both ways. For reference, the
  current ray values (x axis, metres) are
  diff `[0.17458, 0.30519, 0.50249, 0.72929, 0.91715]`,
  image `[0.19828, 0.49380, 0.53089, 0.73033, 0.91963]`.

## 6. Open questions

1. Where should it live — `postprocessing`, next to `findPlanes`, or as a
   `Microscope` method?
2. Wire the closed-form inversion into `error_dz` / the fitting helpers, or keep
   it separate?
3. Virtual planes (roots outside the segment): returned flagged, or dropped?

`findPlanes` should stay regardless — a traced bundle is the natural ground
truth, and on thin-lens columns the two must agree to ~1e-12.

## 7. Two incidental findings (left untouched)

- `assemblies.py:937` (`Microscope.focus_error`) calls
  `findPlanes(self.rays, "x")` — `"x"` lands on the `R` (rotation) parameter and
  `axis` falls back to `"xy"`. Looks like a real bug.
- `postprocessing.whereCrossesZero` is used only by `findPlanes4` (~line 627),
  not by `findPlanes`.
