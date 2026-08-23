# Plan — when planes become surfaces (aberrations and multipoles)

**For:** Eric + the other contributor
**Status:** proposal only, nothing implemented
**Related:** `TODO_DONE_plane-calculus.md` (the paraxial machinery this extends),
`PLAN_2026-08-21_matrix-conjugate-planes.md` (the `A = 0` / `B = 0` criterion)

## 1. What breaks

Everything we have rests on one fact: in the **paraxial linear** regime the
conjugate condition is a scalar equation in `z` alone. `A(z) = 0` and
`B(z) = 0` do not know about the ray's height or azimuth, so their roots are
planes.

Add aberrations or multipoles and the transfer stops being linear. The
focusing distance acquires a dependence on where in the beam you look:

| term | dependence | what "the plane" becomes |
|---|---|---|
| spherical (Cs) | aperture angle `alpha` | longitudinal caustic; z shifts as `alpha^2` |
| field curvature | field radius `r` | a curved surface of revolution (Petzval) |
| astigmatism | `r` and `cos 2phi` | **two** surfaces (sagittal / tangential) |
| N-fold multipole | `cos N phi` | an N-lobed surface |
| chromatic (Cc) | energy spread | a z spread, not a surface |

So the object we are looking for is `z_focus(r, phi; alpha)` — a surface — and
"the image plane" becomes a *choice of criterion* on that surface rather than a
root.

## 2. The proposal in one line

Keep the paraxial planes as the **reference surface** (they are the exact
linear term), and describe aberration as a *departure* from it — measured by a
least-confusion criterion, and fitted to a small set of coefficients.

## 3. Ray side: `focal_surface(...)`

Replace "solve for a root" with "minimize a spot size", which degenerates back
to the exact paraxial root when aberrations vanish (a useful self-test).

1. Choose a reference: the existing
   `conjugate_planes(reference=..., method='frame')` gives `z_0`, the paraxial
   answer, and the family (`image` or `diff`).
2. Sample the beam: a grid of field points `r_i` x azimuths `phi_j` (and,
   for the diffraction family, aperture angles `alpha_k`).
3. For each sample, trace a small bundle and find the z of **least confusion**
   — minimum RMS radius. In a drift the RMS radius is quadratic in z, so this
   is a closed-form vertex, not a search; inside a body use the element's own
   partial block (`Element.transfer_block`, already in place).
4. Return `z_focus(r, phi)` samples plus a fit. Suggested basis, because it
   maps onto how these are measured and corrected:
   `z(r, phi) = z_0 + c_20 r^2 + c_22 r^2 cos 2(phi - phi_22) + c_31 r^3 cos(phi - phi_31) + ...`
   i.e. defocus / field curvature, two-fold astigmatism, coma, then the N-fold
   terms a multipole introduces.
5. Report both the coefficients and a scalar "surface sag" (peak-to-valley over
   the sampled field) so a user can see at a glance whether a plane is still an
   adequate description.

Deliverable shape: `Microscope.focal_surface(family='image', reference=None,
axis='xy', field=..., azimuths=..., aperture=...)` returning the samples, the
fit, and the sag — with `focal_surface` collapsing to `conjugate_planes` (to
machine precision) on a column with no aberrations. **That equivalence is the
acceptance test.**

## 4. Wave side: the frame stays paraxial — on purpose

Important asymmetry, and it is good news: **the scaled frame cannot and should
not follow an aberrated surface.** The frame absorbs exactly the *quadratic*
part of the phase (that is what `(s, R)` is), so:

- the frame's crossover stays the **paraxial reference surface**;
- every non-quadratic term stays where it belongs, as residual phase on `U`.

That is not a limitation, it is the standard aberration function. At the logged
diffraction plane the residual phase on `U` **is** `chi(q)` — the thing an
aberration corrector tunes and a CTF plots. So the wave path needs no surface
machinery at all; it needs:

1. an aberration screen applied to `U` at the pupil/diffraction plane,
   expressed in the existing `phase_shift` contract (a non-absorbable screen,
   exactly like the quadrupole saddle is today), and
2. a reader that fits `chi(q)` from the logged plane and reports the same
   coefficients as the ray-side fit.

If those two agree on a test column, the ray and wave descriptions of
aberration are consistent — a second acceptance test, and a much stronger one
than either alone.

## 5. Covariance side

`beam_waists` needs no new concept either: a waist is already defined by a
minimum, not a crossing. Aberration enters as an *emittance growth* term
(the beam is no longer a linear transform of its entrance ellipse), so the
honest extension is to report the waist plus a non-linearity indicator — e.g.
the growth of `sqrt(det Sigma)`, which is invariant under linear transport and
so is exactly zero until aberrations appear.

## 6. Suggested order

1. `focal_surface` with the least-confusion vertex, aberration-free, and the
   equivalence test against `conjugate_planes`. No new physics — pure
   scaffolding, and it locks the criterion down.
2. One aberration source (Cs on a round lens) as an element-level term, then
   the `z(r, phi)` fit and the sag report.
3. The wave-side `chi(q)` screen + reader, and the ray/wave coefficient
   comparison.
4. Multipoles (N-fold), reusing the same fit with higher N.
5. Only then revisit whether anything in the API should say "surface" instead
   of "plane" — by that point the data will say.

## 7. Open questions

1. Fit basis — the Seidel-style expansion above, or Zernike on the pupil? The
   ray side wants field coordinates; the wave side wants pupil coordinates.
   They are different expansions of the same thing and we should pick
   deliberately rather than end up with both.
2. Where do aberration coefficients live — on the `Element` (per lens, which is
   how they are measured) or on the `Microscope` (as a system total)? Per
   element composes correctly but needs a transport rule.
3. Does `crossovers` keep meaning "the frame's paraxial crossover"? I would say
   yes, and let `focal_surface` own the aberrated answer, so nothing silently
   changes meaning underneath existing code.
