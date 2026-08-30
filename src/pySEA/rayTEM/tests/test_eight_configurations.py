"""Tests driving examples/07_eightConfigurations.py.

The example is the specification: the standard column (``basic_column.sea``),
eight operating states, every lens strength solved from a transfer-matrix
condition, states stored through the column's own settings mechanism, and all
three propagation methods agreeing on where the conjugate planes are. These
tests import the example as a module and hold it to its own claims, so the
example cannot silently rot the way un-executed demo code does.
"""

import importlib.util
import os

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
	"""The high-current, convergent, imaging state, solved once."""
	return ex.solve_column("high", "convergent", "image")


@pytest.fixture(scope="module")
def solved_low():
	"""The low-current, convergent, imaging state, solved once."""
	return ex.solve_column("low", "convergent", "image")


def _configured(sol):
	"""A fresh standard column carrying a solve's strengths.

	Parameters
	----------
	sol : dict
		A :func:`solve_column` result.

	Returns
	-------
	Microscope
		Freshly loaded and configured, nothing propagated yet.

	Raises
	------
	None
	"""
	return ex.apply_strengths(ex.load_base(), sol["strengths"])


def test_high_state_passes_the_full_current(solved_high):
	# C1 images the gun crossover onto CA: the focused spot goes through whole
	assert solved_high["predicted"]["current_fraction"] == 1.0
	scope = _configured(solved_high)
	scope.propagate_ray()
	stated = scope.sections[0].elements[0].beam_current
	assert stated == 1e-9							# the standard column's gun
	assert np.isclose(scope.beam_current, stated, rtol=1e-9)


def test_low_state_is_cut_by_the_aperture(solved_high, solved_low):
	# the ray-path CA is a mask: blocked rays carry zero intensity, so the
	# low state's current is the surviving fraction of the stated 1 nA
	assert solved_low["predicted"]["current_fraction"] < 0.5
	scope = _configured(solved_low)
	scope.propagate_ray()
	high = _configured(solved_high)
	high.propagate_ray()
	assert scope.beam_current < 0.5 * high.beam_current
	assert np.isclose(scope.beam_current,
					  solved_low["predicted"]["current_fraction"] * 1e-9, rtol=1e-6)


def test_convergent_probe(solved_high):
	# a probe means a crossover ON the sample plane, formed by the condensers
	# alone; and the block prediction matches the traced measurement
	scope = _configured(solved_high)
	scope.propagate_ray()
	B = ex.block_between(scope, 0.0, ex.sample_plane(scope))[0, 1]
	assert abs(B) < 1e-12
	# the traced fan's max total angle at the sample matches the block
	# prediction tightly (same linear map); convergence_angle -- the angle of
	# the outermost-BY-POSITION ray -- only approximates it at a crossover,
	# where which ray is outermost is set by residuals
	rays = scope.rays
	zs = rays[:, 0, 4]
	i = int(np.argmin(np.abs(zs - ex.sample_plane(scope))))
	traced = float(np.hypot(rays[i, :, 1], rays[i, :, 3]).max())
	assert np.isclose(traced, solved_high["predicted"]["alpha"], rtol=1e-6)
	assert np.isclose(scope.convergence_angle, solved_high["predicted"]["alpha"],
					  rtol=0.1)
	# the 30 mrad target is honest: hit exactly, or flagged as out of reach
	# (with THIS column's frozen objective the condensers cap out below it)
	if solved_high["alpha_limited"]:
		assert solved_high["predicted"]["alpha"] < ex.ALPHA_TARGET
	else:
		assert np.isclose(solved_high["predicted"]["alpha"], ex.ALPHA_TARGET,
						  rtol=1e-6)


def test_low_state_probe(solved_low):
	# still a probe: crossover on the sample, condensers only
	scope = _configured(solved_low)
	B = ex.block_between(scope, 0.0, ex.sample_plane(scope))[0, 1]
	assert abs(B) < 1e-12


def test_objective_is_never_retuned(solved_high, solved_low):
	# probe focusing belongs to the condensers and projection to the
	# projectors: no solved state may carry an objective strength
	base = ex.load_base()
	frozen = {n: base[n].strength for n in ("OL1", "OL2")}
	for sol in (solved_high, solved_low,
				ex.solve_column("high", "parallel", "diffraction")):
		assert not set(sol["strengths"]) & set(frozen)
		scope = _configured(sol)
		for n, k in frozen.items():
			assert scope[n].strength == k


def test_parallel_probe_is_nearly_parallel():
	sol = ex.solve_column("high", "parallel", "image")
	scope = _configured(sol)
	# D = 0: a ray fan from one source point arrives parallel
	D = ex.block_between(scope, 0.0, ex.sample_plane(scope))[1, 1]
	assert abs(D) < 1e-12
	# and the full fan's residual angle is far below the convergent state's.
	# "Nearly" has a floor: the finite source seen through the strong OL1
	# leaves ~x_src/f of residual angle (a couple of mrad at f = 2 mm), so
	# parallel means an order of magnitude below the probe, not zero.
	assert sol["predicted"]["alpha"] < 0.15 * ex.ALPHA_TARGET


def test_detector_modes(solved_high):
	# image: B(sample->detector) = 0, so position maps to position
	assert abs(solved_high["M_det"][0, 1]) < 1e-12
	# diffraction: A = 0, so position maps to ANGLE via the camera length B
	sol = ex.solve_column("high", "convergent", "diffraction")
	A, B = sol["M_det"][0]
	assert abs(A) < 1e-10
	assert abs(B) > 5e-3			# a real camera length, metres per radian


def test_state_survives_the_settings_round_trip(solved_high, tmp_path, monkeypatch):
	# save_as_setting writes settings/<name>.json in the CWD; a fresh column
	# plus load_setting must reproduce the solved strengths exactly
	monkeypatch.chdir(tmp_path)
	scope = _configured(solved_high)
	name = "basic_column - test-state"
	scope.save_as_setting(name, {n: "strength" for n in solved_high["strengths"]})
	fresh = ex.load_base()
	fresh.load_setting(name)
	for n, k in solved_high["strengths"].items():
		assert np.isclose(fresh[n].strength, k, rtol=1e-9)


def test_conjugate_planes_agree_across_methods(solved_high):
	# rays vs accumulated matrix: the same planes, to interpolation accuracy
	# (the ray method draws straight lines between logged planes, so a plane
	# abutting a thick body carries ~1e-5 m of interpolation error; the
	# matrix method resolves inside bodies and is the exact one)
	scope = _configured(solved_high)
	scope.propagate_ray()
	frame = scope.conjugate_planes(method='frame')
	ray = scope.conjugate_planes(method='ray')
	for fam in ("image", "diff"):
		zf, zr = np.sort(frame[fam]), np.sort(ray[fam])
		assert zf.size and zr.size
		assert all(np.min(np.abs(zr - z)) < 1e-4 for z in zf)
		assert all(np.min(np.abs(zf - z)) < 1e-4 for z in zr)
	# every geometric image plane has a covariance waist beside it, displaced
	# by the emittance focal shift -- being NEAR and not ON is the physics.
	# The plane at CA carries the largest displacement: the envelope mode
	# treats the aperture as a no-op (documented approximation), so its waist
	# ignores the cut entirely.
	waists = np.sort(scope.beam_waists()['z'])
	for z in frame["image"]:
		assert np.min(np.abs(waists - z)) < 2e-2
	# the sample and the detector are themselves conjugate planes of this state
	Z = scope.named_positions
	assert np.min(np.abs(np.asarray(frame["image"]) - ex.sample_plane(scope))) < 1e-9
	assert np.min(np.abs(np.asarray(frame["image"]) - Z["detector"])) < 1e-9


def test_wave_crossovers_land_on_the_matrix_diffraction_planes(solved_high):
	# the third method: the scaled-wave frame's own crossings. A flat-phase
	# seed belongs to the diffraction family, and the hybrid run logs every
	# member of it on the way down the column.
	scope = _configured(solved_high)
	dense = scope.subdivided(4e-3)
	dense.propagate(kind="wave-hybrid")
	frame = scope.conjugate_planes(method='frame')
	crossings = np.sort(np.asarray(dense.crossovers, float))
	for z in frame["diff"]:
		assert np.min(np.abs(crossings - z)) < 1e-9
