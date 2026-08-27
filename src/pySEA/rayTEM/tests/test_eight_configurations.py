"""Tests driving examples/07_eightConfigurations.py.

The example is the specification: one column, eight operating states, every
lens strength solved from a transfer-matrix condition, and all three
propagation methods agreeing on where the conjugate planes are. These tests
import the example as a module and hold it to its own claims, so the example
cannot silently rot the way un-executed demo code does.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
_example = os.path.join(_here, "..", "..", "..", "..", "examples",
						"07_eightConfigurations.py")


def _load_example():
	"""Import the example script as a module, once per session.

	Returns
	-------
	module
		The loaded ``07_eightConfigurations`` module (its ``__main__`` block
		does not run on import).

	Raises
	------
	FileNotFoundError
		If the example script is not where the repository layout puts it.
	"""
	spec = importlib.util.spec_from_file_location("eight_configurations", _example)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


ex = _load_example()


@pytest.fixture(scope="module")
def solved_high():
	"""The high-current, convergent, imaging configuration, solved once."""
	return ex.solve_column("high", "convergent", "image")


@pytest.fixture(scope="module")
def solved_low():
	"""The low-current, convergent, imaging configuration, solved once."""
	return ex.solve_column("low", "convergent", "image")


def test_high_state_passes_the_full_current(solved_high):
	# C1 images the gun crossover onto CA: the focused spot goes through whole
	assert solved_high["predicted"]["current_fraction"] == 1.0
	scope = ex.build_column(**solved_high["strengths"])
	scope.propagate_ray()
	assert np.isclose(scope.beam_current, ex.GUN_CURRENT, rtol=1e-9)


def test_low_state_is_cut_by_the_aperture(solved_high, solved_low):
	assert solved_low["predicted"]["current_fraction"] < 0.5
	scope = ex.build_column(**solved_low["strengths"])
	scope.propagate_ray()
	high = ex.build_column(**solved_high["strengths"])
	high.propagate_ray()
	assert scope.beam_current < 0.2 * high.beam_current


def test_convergent_probe_hits_thirty_mrad(solved_high):
	# the solve's own prediction and the traced measurement both land on target
	assert not solved_high["alpha_limited"]
	assert np.isclose(solved_high["predicted"]["alpha"], ex.ALPHA_TARGET, rtol=1e-6)
	scope = ex.build_column(**solved_high["strengths"])
	scope.propagate_ray()
	assert np.isclose(scope.convergence_angle, ex.ALPHA_TARGET, rtol=1e-6)
	# and it is a probe: a crossover ON the sample plane
	Z = scope.named_positions
	B = ex.block_between(scope, 0.0, Z["sample"])[0, 1]
	assert abs(B) < 1e-12


def test_low_state_cannot_reach_thirty_mrad(solved_low):
	# CA removes phase space, not just electrons: the target is out of reach
	# and the solve says so instead of pretending
	assert solved_low["alpha_limited"]
	assert solved_low["predicted"]["alpha"] < 0.5 * ex.ALPHA_TARGET


def test_parallel_probe_is_nearly_parallel():
	sol = ex.solve_column("high", "parallel", "image")
	scope = ex.build_column(**sol["strengths"])
	Z = scope.named_positions
	# D = 0: a ray fan from one source point arrives parallel
	D = ex.block_between(scope, 0.0, Z["sample"])[1, 1]
	assert abs(D) < 1e-12
	# and the full fan's residual angle is far below the convergent state's
	assert sol["predicted"]["alpha"] < 0.05 * ex.ALPHA_TARGET


def test_detector_modes(solved_high):
	# image: B(sample->detector) = 0, so position maps to position
	assert abs(solved_high["M_det"][0, 1]) < 1e-12
	# diffraction: A = 0, so position maps to ANGLE via the camera length B
	sol = ex.solve_column("high", "convergent", "diffraction")
	A, B = sol["M_det"][0]
	assert abs(A) < 1e-10
	assert abs(B) > 0.01			# a real camera length, metres per radian


def test_conjugate_planes_agree_across_methods(solved_high):
	# rays vs accumulated matrix: the same planes to numerical exactness;
	# covariance waists: the same planes plus the emittance focal shift
	scope = ex.build_column(**solved_high["strengths"])
	scope.propagate_ray()
	frame = scope.conjugate_planes(method='frame')
	ray = scope.conjugate_planes(method='ray')
	for fam in ("image", "diff"):
		zf, zr = np.sort(frame[fam]), np.sort(ray[fam])
		# same planes both ways (the ray method may log a plane twice when a
		# crossover sits exactly on an element boundary, so sizes may differ)
		assert zf.size and zr.size
		assert all(np.min(np.abs(zr - z)) < 1e-9 for z in zf)
		assert all(np.min(np.abs(zf - z)) < 1e-9 for z in zr)
	# every geometric image plane has a covariance waist beside it, displaced
	# by the emittance focal shift -- micrometres at the demagnified planes but
	# ~7 mm at the 63x-magnified detector image, since the shift scales with
	# the local magnification squared. Being NEAR and not ON is the physics.
	waists = np.sort(scope.beam_waists()['z'])
	for z in frame["image"]:
		assert np.min(np.abs(waists - z)) < 1e-2
	# the sample and the detector are themselves conjugate planes of this state
	Z = scope.named_positions
	assert np.min(np.abs(np.asarray(frame["image"]) - Z["sample"])) < 1e-9
	assert np.min(np.abs(np.asarray(frame["image"]) - Z["detector"])) < 1e-9


def test_wave_crossovers_land_on_the_matrix_diffraction_planes(solved_high):
	# the third method: the scaled-wave frame's own crossings. A flat-phase
	# seed belongs to the diffraction family, and the hybrid run logs every
	# member of it on the way down the column.
	scope = ex.build_column(**solved_high["strengths"])
	dense = scope.subdivided(4e-3)
	dense.propagate(kind="wave-hybrid")
	frame = scope.conjugate_planes(method='frame')
	crossings = np.sort(np.asarray(dense.crossovers, float))
	for z in frame["diff"]:
		assert np.min(np.abs(crossings - z)) < 1e-9
