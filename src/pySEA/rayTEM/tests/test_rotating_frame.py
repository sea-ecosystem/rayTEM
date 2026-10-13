"""``Rays.convert_to_rotating_reference_frame`` is a vectorized form of its old loop.

The loop indexed rays through ``Rays.__getitem__`` (a deep copy of the Rays
metadata per ray) and rebuilt the rotation with ``fix_mat_dims`` per ray, which
cost ~1.5 s on a 57-plane, 225-ray trace -- paid by ``plot2D`` and by every
live ray diagram. The vectorized form must give the same numbers, including the
quirk that each plane's boundary rays turn by that plane's *last* ray's angle.
"""

from __future__ import annotations

import numpy as np

from pySEA.rayTEM.elements import fix_mat_dims
from pySEA.rayTEM.microscopes.basic_column import build_basic_column


def _reference_rotation(rays):
	"""Return ``(rays, boundary_ray)`` rotated by the original per-ray loop.

	Parameters
	----------
	rays : Rays
		A propagated lab-frame trace.

	Returns
	-------
	tuple[numpy.ndarray, numpy.ndarray]
		The rotated rays and boundary rays, as the pre-2026-10-12 loop computed them.
	"""

	R = np.asarray(rays.R)
	data = np.asarray(rays.rays)
	nl, nr, _ = data.shape
	converted = np.zeros(data.shape)
	boundary = np.asarray(rays.boundary_ray).copy()
	for l in range(nl):
		for r in range(nr):
			C, S = np.cos(R[l, r]), np.sin(R[l, r])
			M = fix_mat_dims(np.asarray([[C, S, 0, 0], [-S, C, 0, 0], [0, 0, C, S], [0, 0, -S, C]]), ["x", "y", "xt", "yt"])
			converted[l, r, :] = M @ data[l, r, :]
		boundary[l] = np.einsum("mn,in->im", M, boundary[l])
	return converted, boundary


def test_the_vectorized_rotation_matches_the_loop_it_replaced() -> None:
	"""Same rays, same boundary rays, same frame tag."""

	scope = build_basic_column()
	scope.propagate_ray()
	expected_rays, expected_boundary = _reference_rotation(scope.rays)
	rotated = scope.rays.convert_to_rotating_reference_frame()
	assert rotated.reference_frame == "rotating"
	assert np.allclose(np.asarray(rotated.rays), expected_rays, rtol=0.0, atol=1e-15)
	assert np.allclose(np.asarray(rotated.boundary_ray), expected_boundary, rtol=0.0, atol=1e-15)
	assert np.any(np.asarray(scope.R) != 0.0), "the basic column must rotate for this test to mean anything"


def test_an_already_rotating_trace_is_returned_unchanged() -> None:
	"""Converting twice is a no-op, as before."""

	scope = build_basic_column()
	scope.propagate_ray()
	once = scope.rays.convert_to_rotating_reference_frame()
	assert once.convert_to_rotating_reference_frame() is once
