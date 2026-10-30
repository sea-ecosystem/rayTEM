# TODO — Element alignment: an element's own axis, off the column's

**Branch/worktree:** `element-alignment` (rayTEM), cut from `dev`, merged back into `dev`.
**Plan:** none (Eric's request, 2026-10-30, below).

Eric (2026-10-30): "add [an affine transform] in and have the default as none so that it skips
the step, understanding that it's really just an identity transform … a physical alignment
matrix that our transfer matrix can multiply with an affine transform and that can account for
any off tilt or positioning when it does exist … include [it] in all elements. That way we can
begin to calibrate what the actual geometry is in the microscope."

Distinct from the existing `shift_x`/`tilt_x`, which are kicks the element gives (a dipole's
deflection). The alignment says where the element's own axis is: its offset and tilt at the
element's centre. Rays enter the element's frame, the element acts as it always has (matrix,
aberrations about its own axis, an aperture clips about its own centre), and leave it.

- [x] `Element.alignment` (None, or x, xt, y, yt at the centre); `affine_matrix()` (7×7 homogeneous)
- [x] ray path (and so intensity, the mean of the moments); aberrated moments use the element's frame
- [x] fixed wave path: shift and tilt the field into the element's frame and back; scaled/hybrid refuse clearly (and `Source.wave`; a gun's rays and moments leave along its axis)
- [x] tests (identity when None, displaced lens = steering, aperture clips about its centre, round trip in .sea) — `tests/test_element_alignment.py`, 11 pass; suite 342 pass, 1 pre-existing failure (`test_rotated_quadrupole`, also on `dev`)
- [x] wiki, refresh, merge into `dev`

**Not yet:** the scaled and hybrid wave paths and `Source.wave` refuse an aligned element; the section driver's display boundary ray is still clipped about the column's axis; a roll about the axis stays `Element.rotation` (quadrupoles).
