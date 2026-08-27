# Plan — thick quadrupole: symplecticity, axis convention, and the wave-side follow-on

**Issue:** [sea-ecosystem/rayTEM#3](https://github.com/sea-ecosystem/rayTEM/issues/3)
**TODO:** [TODO_ACTIVE_thick-quadrupole-symplecticity.md](TODO_ACTIVE_thick-quadrupole-symplecticity.md)
**For:** Eric + the other contributor (quadrupole ray physics is not ours to change silently)
**Status:** proposal; the mitigation (guard only) is already on
`Signal_and_propagation_additions`, no quadrupole physics touched.

## 1. The two defects

**(a) The thick y-block is not symplectic.** `Quadrapole.transfer_matrix`, thick
branch (`length > 0`), builds `Y = [[C, S/K], [+K S, C]]` with `C = cos|KL|`,
`S = sin|KL|`, so `det Y = C² − S² = cos(2|KL|)`. For
`Quadrapole(strength=12, length=0.03)` that is **0.751806** — about 25% of the
phase-space area lost over a 30 mm body, in violation of Liouville. `det X = 1`
on the focusing axis, so only one axis is affected. The defocusing axis obeys
`u'' = +K²u` and is therefore hyperbolic:
`[[cosh, sinh/K], [K sinh, cosh]]`, `det = 1`. The method's own docstring
already says this ("for K<0 `C = cosh`... `S = (1/√|K|) sinh`"); the code does
not do it.

Symptoms: emittance is not conserved through a thick quad; the block's halves
do not compose (`M(L/2)² ≠ M(L)`, residual 6.4e-2); anything that integrates
*along* the body — plane finding, envelope transport, wave propagation — is
invalid on that axis.

**(b) Thin and thick disagree on which axis focuses.** The thin branch swaps
`X`/`Y` for `K > 0`; the thick branch never swaps. So a quadrupole of a given
sign focuses one axis when thin and the other when thick. The thin swap's
comment justifies itself by reference to a since-removed instrument script, so
the intended convention needs restating rather than reverse-engineering.

## 2. What is already in place (mitigation, not a fix)

- `Element.transfer_block(dz, axis)` — the rotating-frame 2×2 at a **partial**
  length. `Quadrapole`'s override mirrors `transfer_matrix` *exactly*,
  defect included, so plane finding never silently diverges from ray tracing.
- `Microscope._accumulate_blocks` guards `det == 1` on any element with a body
  and raises, naming the determinant, rather than reporting untrustworthy
  planes. This is a **general** check, so it will stop firing by itself once
  the matrix is fixed — no follow-up edit needed.
- `basic_column` is unaffected: all its quadrupoles are `Thin quad`.

## 3. Step 1 — make the thick block symplectic

With `k = |K|`, `s = length`:

```
focusing axis:    [[ cos(k s),  sin(k s)/k ], [ -k sin(k s),  cos(k s) ]]
defocusing axis:  [[ cosh(k s), sinh(k s)/k ], [  k sinh(k s), cosh(k s) ]]
```

Acceptance:

- `det == 1` on both axes across a sweep of `K` and `L`;
- `M(L/2) · M(L/2) == M(L)` to ~1e-12 on both axes (a homogeneous body must
  compose);
- the `k·s → 0` limit reproduces the thin block, and `focal_powers()` agrees in
  sign with the thick block's off-diagonal in that limit;
- `propagate_moments` conserves `sqrt(det Σ)` through a thick quad.

## 4. Step 2 — settle the axis convention

Pick which axis focuses for `K > 0`, apply it in **both** branches, and state it
in the docstring. Acceptance: a thin quad and a short thick quad of equal
strength focus the same axis.

## 5. Step 3 — the wave-side follow-on (why this matters beyond rays)

A thick round lens is now carried **exactly** in the scaled wave path as a
quadratic-index *segment* — no thin-kick approximation — which moved
`basic_column`'s crossovers from 422–4808 µm off the ray planes to **0.0 µm**.
A thick quadrupole should get the same treatment, but cannot yet:

1. the segment propagator is currently named and scoped for a lens
   (`waveoptics.propagate_quadratic_segment_scaled` (renamed from
   `propagate_thick_lens_scaled` in the P1 wave-seam cleanup)) and **refuses anisotropic
   frames**, because a round lens is isotropic by construction. A quadrupole
   needs per-axis strengths `(K, −K)`, i.e. one focusing and one defocusing
   medium in the same element;
2. there is no correct per-axis body law to mirror until step 1 lands.

So the order is: fix the matrix, then generalize the propagator (element-
agnostic name + per-axis strengths), then add `Quadrapole._scaled_segment()`
mirroring `Lens._scaled_segment()`. Acceptance: a thick-quad column's wave line
foci match the ray-traced and analytic planes to ~1e-9, and the symplecticity
guard no longer fires.

**Related architectural point (Eric):** no element should own a propagation
method — the ray side is the model, where an element declares
`transfer_matrix()` and the generic `propagate_ray` consumes it. The wave side
mostly follows this (`phase_shift`, `_scaled_segment` are declarations consumed
by the generic `propagate_wave`), but three elements still override
`propagate_wave` itself: `Source`, `Aperture`, `Prism`. Those overrides exist
because their wave action is not a phase — a seed and an amplitude mask — so
the missing piece is an **amplitude/mask declaration** alongside
`phase_shift`, after which the overrides can be deleted. Worth doing in the
same pass as the propagator rename, since both are about the same seam.

## 6. Open questions

1. Convention: which axis focuses for `K > 0`? (Needed before step 2 can be
   written.)
2. Should `Quadrapole` support a skew (45°) orientation? The class has no axis
   parameter today, which is why `basic_column` fakes a skew pair with
   `(+K, −K)`; if skew is coming, the body law should be written with a rotation
   in mind rather than bolted on later.
3. `phase_shift(scaled=bool)` — keep the single method, or split into
   `phase_fixed` / `phase_scaled`? Eric raised this; it is a naming decision on
   a core contract, so it should be agreed rather than assumed.
