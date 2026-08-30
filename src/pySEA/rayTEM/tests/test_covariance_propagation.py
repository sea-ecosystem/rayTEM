"""Tests for covariance propagation, moment closure, and examples/08.

The example is the specification: one source boundary condition, four
aberration configurations through ``basic_column``, and resolution reported as
emittance and principal axes rather than as a single probe diameter. These
tests import it as a module and hold it to its own claims, then pin the
machinery underneath -- the closure, the kick polynomial, and the chromatic
coupling -- against independently computed references.

Two references are used, and neither is a fitted constant:

- **Isserlis by hand** for the closure, since a centered Gaussian's fourth and
  sixth moments have closed forms that can be written down separately from the
  code that computes them.
- **Monte-Carlo ray statistics** for the aberrated and chromatic covariance
  updates, since the per-ray kick is the same physics evaluated by a different
  route. The rays are a *reference*, never part of covariance propagation.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(1, "../../../")
from pySEA.rayTEM import Source, Lens, Drift, MicroscopeSection, Microscope, columnByName
from pySEA.rayTEM.aberrations import Aberrations
from pySEA.rayTEM.elements import (convention, suspended_aberrations,
								   _center_monomials, _gaussian_moment, _kick_moments)
from pySEA.rayTEM.postprocessing import beam_widths, emittance, resolution_ellipses
from pySEA.rayTEM.seashells import as_ndarray

_here = os.path.dirname(os.path.abspath(__file__))
_example = os.path.join(_here, "..", "..", "..", "..", "examples",
						"08_covariancePropagation.py")

IX, IXT, IY, IYT, IE = (columnByName(n) for n in ("x", "xt", "y", "yt", "E"))


def _load_example():
	"""Import the example script as a module, once per session.

	Returns
	-------
	module
		The loaded ``08_covariancePropagation`` module (its ``__main__`` block
		does not run on import, and no figure is drawn).

	Raises
	------
	FileNotFoundError
		If the example script is not where the repository layout puts it.
	"""
	spec = importlib.util.spec_from_file_location("covariance_propagation", _example)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


ex = _load_example()


def _toy(c30=0.0, chromatic=0.0, drift=0.05, focal=0.02):
	"""A Source/Lens/Drift column small enough to reason about by hand.

	Parameters
	----------
	c30 : float, optional
		Spherical aberration on the lens (m), by default 0.
	chromatic : float, optional
		Chromatic coefficient on the lens (m), by default 0.
	drift : float, optional
		Drift length after the lens (m), by default 0.05.
	focal : float, optional
		Lens focal length (m), by default 0.02.

	Returns
	-------
	assemblies.Microscope
		An unpropagated column.

	Raises
	------
	None
	"""
	terms = {}
	if c30:
		terms['C30'] = c30
	if chromatic:
		terms['Cc'] = chromatic
	lens = Lens(name="L", focal_length=focal, aberrations=Aberrations(terms))
	source = Source(name="S", size=(3e-6, 5e-6), angle=(2e-4, 2e-4),
					voltage=200, energy_spread=1e-2)
	return Microscope(sections=[MicroscopeSection(
		elements=[source, Drift(length=0.01), lens, Drift(length=drift)])])


# ---- the closure itself -------------------------------------------------

def test_gaussian_closure_reproduces_isserlis():
	# a centered Gaussian's even central moments are sums over perfect pairings;
	# the odd ones vanish. Both are checked against hand-written closed forms.
	rng = np.random.default_rng(3)
	A = rng.normal(size=(6, 6))
	S = A @ A.T
	assert _gaussian_moment(S, ()) == 1.0
	assert _gaussian_moment(S, (IX,)) == 0.0
	assert _gaussian_moment(S, (IX, IX, IX)) == 0.0
	assert np.isclose(_gaussian_moment(S, (IX, IX)), S[IX, IX])
	assert np.isclose(_gaussian_moment(S, (IX, IX, IX, IX)), 3 * S[IX, IX]**2)
	assert np.isclose(_gaussian_moment(S, (IX,) * 6), 15 * S[IX, IX]**3)
	# the general four-index identity, not just the diagonal case
	i, j, k, l = IX, IXT, IY, IYT
	assert np.isclose(_gaussian_moment(S, (i, j, k, l)),
					  S[i, j] * S[k, l] + S[i, k] * S[j, l] + S[i, l] * S[j, k])


def test_center_monomials_expands_about_the_mean():
	# (mu + s)^2 = mu^2 + 2 mu s + s^2, and is the identity for a centered beam
	mu = np.zeros(len(convention))
	assert _center_monomials([(2.0, (IX, IX))], mu) == [(2.0, (IX, IX))]
	mu[IX] = 3.0
	terms = _center_monomials([(2.0, (IX, IX))], mu)
	constant = sum(c for c, idx in terms if idx == ())
	linear = sum(c for c, idx in terms if idx == (IX,))
	quadratic = sum(c for c, idx in terms if idx == (IX, IX))
	assert np.isclose(constant, 2.0 * 9.0)
	assert np.isclose(linear, 2.0 * 2 * 3.0)
	assert np.isclose(quadratic, 2.0)


def test_kick_moments_reports_the_mean_of_an_even_order_kick():
	# <delta> for delta = g x^2 is g sigma_x^2 -- NOT zero, even though the
	# centroid ray at x = 0 feels nothing. This is the mean shift the plan
	# insists must be retained rather than absorbed into the width.
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = 4e-12
	delta_mean, C, D = _kick_moments({IXT: [(7.0, (IX, IX))]}, S, _gaussian_moment)
	assert np.isclose(delta_mean[IXT], 7.0 * 4e-12)
	assert np.isclose(C[IX, IXT], 0.0)					# <s_x s_x s_x> = 0
	assert np.isclose(D[IXT, IXT], 7.0**2 * 2 * 4e-12**2)	# 3 s^2 - s^2


# ---- the kick, as a polynomial -----------------------------------------

@pytest.mark.parametrize("name,coefficient",
						 [("C30", 1e-3), ("C32", 1e-3 + 4e-4j), ("C34", 2e-3),
						  ("C21", 5e-4), ("C41", 1e-3), ("C50", 2e-3),
						  ("C56", 1e-3 - 5e-4j), ("C12", 1e-6 + 2e-6j)])
def test_aberration_monomials_reproduce_the_kick(name, coefficient):
	# the covariance path needs the kick as an algebraic object; recovering it
	# by sampling the unit circle must agree with the ray path's own evaluation
	# at arbitrary coordinates, at every Krivanek order and for skew terms
	lens = Lens(focal_length=0.01, aberrations={name: coefficient})
	P = lens.focal_power
	monomials = lens._aberration_monomials(P, np.zeros(len(convention)))
	rng = np.random.default_rng(11)
	x, y = rng.normal(0, 2e-6, 400), rng.normal(0, 2e-6, 400)

	def evaluate(terms):
		total = np.zeros_like(x)
		for c, idx in terms:
			term = np.full_like(x, c)
			for i in idx:
				term = term * (x if i == IX else y)
			total = total + term
		return total

	from pySEA.rayTEM.elements import _split_quadratic_aberrations
	_, _, residual = _split_quadratic_aberrations(lens.aberrations, P, P, P)
	reference = (Aberrations(dict(residual.items())).deflection_at(x, y, P)
				 if residual else (np.zeros_like(x), np.zeros_like(y)))
	scale = max(np.abs(reference[0]).max(), np.abs(reference[1]).max(), 1e-30)
	assert np.abs(evaluate(monomials.get(IXT, [])) - reference[0]).max() < 1e-12 * scale
	assert np.abs(evaluate(monomials.get(IYT, [])) - reference[1]).max() < 1e-12 * scale


# ---- exactness of the linear path --------------------------------------

def test_ideal_transport_is_exact_and_conserves_emittance():
	# linear propagation of Sigma is exact for any distribution, so the
	# accumulated matrix and the element-by-element walk must agree, and the
	# emittance -- a transport invariant -- must not move
	scope = _toy()
	scope.propagate_moments()
	mu0, S0 = scope.sections[0].elements[0].moments()
	M = np.eye(len(convention))
	for element in scope.sections[0]._propagation_elements():
		M = element.transfer_matrix() @ M
	cov = as_ndarray(scope.covariance_matrix)
	assert np.allclose(cov[-1], M @ S0 @ M.T, rtol=1e-12)
	eps = emittance(cov)[:, 0]
	assert np.allclose(eps, eps[0], rtol=1e-12)


def test_zero_aberration_strength_recovers_the_ideal_result():
	# an aberration set to zero must be bit-for-bit the ideal column, not
	# merely close: the nonlinear branch has to be skipped entirely
	ideal = _toy()
	ideal.propagate_moments()
	zero = _toy(c30=0.0, chromatic=0.0)
	zero.propagate_moments()
	assert np.array_equal(as_ndarray(ideal.covariance_matrix),
						  as_ndarray(zero.covariance_matrix))


# ---- the aberrated update ----------------------------------------------

def test_aberrated_covariance_matches_monte_carlo_rays():
	# the closure is exact for one aberrated element on a Gaussian beam, so
	# the statistics of the exact per-ray kick must agree with it to MC noise
	rng = np.random.default_rng(5)
	n = 400000
	sx, sy, st = 4e-6, 4e-6, 1e-4
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = sx**2; S[IY, IY] = sy**2
	S[IXT, IXT] = st**2; S[IYT, IYT] = st**2
	mu = np.zeros(len(convention))
	lens = Lens(focal_length=0.02, aberrations={'C30': 3e3})
	_, S_out = lens.propagate_moments(mu, S)

	rays = np.zeros((n, len(convention)))
	rays[:, IX] = rng.normal(0, sx, n); rays[:, IY] = rng.normal(0, sy, n)
	rays[:, IXT] = rng.normal(0, st, n); rays[:, IYT] = rng.normal(0, st, n)
	out = lens.propagate_ray(rays)
	for a, b in ((IX, IX), (IX, IXT), (IXT, IXT), (IYT, IYT)):
		mc = np.cov(out[:, a], out[:, b])[0, 1]
		assert np.isclose(S_out[a, b], mc, rtol=0.03), (a, b, S_out[a, b], mc)


def test_cross_plane_closure_terms_appear_when_the_beam_is_coupled():
	# a cubic kick on an x-y correlated beam produces <x dtheta_y> and
	# <dtheta_x dtheta_y>, which a per-axis closure would drop. On a coupled
	# beam they are the same order as the terms that are kept.
	rho = 0.6
	sx = sy = 5e-6
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = sx**2; S[IY, IY] = sy**2
	S[IX, IY] = S[IY, IX] = rho * sx * sy
	S[IXT, IXT] = S[IYT, IYT] = 1e-8
	lens = Lens(focal_length=0.02, aberrations={'C30': 3e3})
	P = lens.focal_power
	M = lens.transfer_matrix()
	_, _, C, D = lens._aberration_moment_pieces(M, S, lens.aberrations, P)
	g = -3e3 * P**4
	# Isserlis, written out independently
	assert np.isclose(C[IX, IYT], g * 3 * S[IX, IY] * (S[IX, IX] + S[IY, IY]), rtol=1e-10)
	assert np.isclose(D[IXT, IYT],
					  g**2 * (12 * S[IX, IY]**3 + 18 * S[IX, IY] * S[IX, IX] * S[IY, IY]
							  + 15 * S[IX, IX]**2 * S[IX, IY]
							  + 15 * S[IY, IY]**2 * S[IX, IY]), rtol=1e-10)
	assert abs(C[IX, IYT]) > 0.5 * abs(C[IX, IXT])
	assert abs(D[IXT, IYT]) > 0.5 * abs(D[IXT, IXT])


def test_even_order_aberration_shifts_the_ensemble_mean():
	# a second-order aberration kicks the ensemble even though the centroid
	# ray, sitting on axis, feels nothing. Retaining that shift is the point.
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = S[IY, IY] = (6e-6)**2
	S[IXT, IXT] = S[IYT, IYT] = 1e-8
	mu = np.zeros(len(convention))
	lens = Lens(focal_length=0.02, aberrations={'C21': 2e-2})
	mu_out, _ = lens.propagate_moments(mu, S)
	with suspended_aberrations([lens]):
		mu_ideal, _ = lens.propagate_moments(mu, S)
	assert not np.allclose(mu_out, mu_ideal)
	rng = np.random.default_rng(7)
	n = 400000
	rays = np.zeros((n, len(convention)))
	rays[:, IX] = rng.normal(0, 6e-6, n); rays[:, IY] = rng.normal(0, 6e-6, n)
	rays[:, IXT] = rng.normal(0, 1e-4, n); rays[:, IYT] = rng.normal(0, 1e-4, n)
	# difference the SAME rays run aberrated and ideal, so the linear part --
	# and with it the sampling noise in the ray table's own mean -- cancels
	traced = lens.propagate_ray(rays)
	with suspended_aberrations([lens]):
		traced_ideal = lens.propagate_ray(rays)
	for i in (IXT, IYT):
		shift = mu_out[i] - mu_ideal[i]
		if shift:
			mc = (traced[:, i] - traced_ideal[:, i]).mean()
			assert np.isclose(shift, mc, rtol=0.05), (i, shift, mc)
	# and it is the kick's ensemble average, not its value at the centroid
	assert lens._aberration_kick(np.zeros((1, len(convention))))[2][0] == 0.0


def test_covariance_stays_symmetric_and_positive_semidefinite():
	# the complete Gaussian closure is the exact pushforward of a Gaussian, so
	# the result is a real covariance no matter how strong the aberration is
	rng = np.random.default_rng(13)
	for c30 in (1e2, 1e4, 1e6):
		A = rng.normal(size=(4, 4))
		block = A @ A.T * 1e-11
		S = np.zeros((len(convention),) * 2)
		S[np.ix_([IX, IXT, IY, IYT], [IX, IXT, IY, IYT])] = block
		lens = Lens(focal_length=0.02, aberrations={'C30': c30})
		_, S_out = lens.propagate_moments(np.zeros(len(convention)), S)
		assert np.allclose(S_out, S_out.T, atol=1e-30)
		values = np.linalg.eigvalsh(0.5 * (S_out + S_out.T))
		assert values.min() > -1e-12 * abs(values).max()


def test_aberration_growth_scales_as_the_square_of_the_coefficient():
	# the leading emittance growth comes from the kick's own variance, which is
	# quadratic in the coefficient
	def growth(c30):
		scope = _toy(c30=c30)
		scope.propagate_moments()
		base = _toy()
		base.propagate_moments()
		return (emittance(as_ndarray(scope.covariance_matrix))[-1, 0]**2
				- emittance(as_ndarray(base.covariance_matrix))[-1, 0]**2)
	small, large = growth(1e1), growth(2e1)
	assert np.isclose(large / small, 4.0, rtol=0.02)


# ---- chromatic ----------------------------------------------------------

def test_chromatic_needs_both_a_spread_and_a_coefficient():
	# either one alone leaves the column achromatic, bit-for-bit
	ideal = _toy(); ideal.propagate_moments()
	no_cc = _toy(chromatic=0.0); no_cc.propagate_moments()
	assert np.array_equal(as_ndarray(ideal.covariance_matrix),
						  as_ndarray(no_cc.covariance_matrix))
	mono = _toy(chromatic=2e-3)
	mono.sections[0].elements[0].energy_spread = 0.0
	mono.propagate_moments()
	flat = _toy()
	flat.sections[0].elements[0].energy_spread = 0.0
	flat.propagate_moments()
	assert np.allclose(as_ndarray(mono.covariance_matrix),
					   as_ndarray(flat.covariance_matrix), rtol=1e-12)


def test_chromatic_angular_variance_matches_the_closed_form():
	# the chromatic term needs only <delta^2 x^2>, which factorizes when the
	# energy spread is independent of position -- so it is EXACT, and equals
	# kappa^2 sigma_delta^2 sigma_x^2 with kappa = Cc P^2 / E0
	sx, sE, E0, cc = 5e-6, 2e-2, 200.0, 3e-3
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = S[IY, IY] = sx**2
	S[IXT, IXT] = S[IYT, IYT] = 1e-8
	S[IE, IE] = sE**2
	mu = np.zeros(len(convention)); mu[IE] = E0
	lens = Lens(focal_length=0.02, aberrations={'Cc': cc})
	_, out = lens.propagate_moments(mu, S)
	with suspended_aberrations([lens]):
		_, ideal = lens.propagate_moments(mu, S)
	kappa = cc * lens.focal_power**2 / E0
	assert np.isclose(out[IXT, IXT] - ideal[IXT, IXT], kappa**2 * sE**2 * sx**2,
					  rtol=1e-10)


def test_chromatic_matches_a_monte_carlo_ray_reference():
	# the ray path applies the same physics one electron at a time; the two
	# must agree. Rays are the reference here, never part of the covariance run.
	rng = np.random.default_rng(17)
	n = 400000
	sx, sE, E0, cc = 5e-6, 2e-2, 200.0, 3e-3
	S = np.zeros((len(convention),) * 2)
	S[IX, IX] = S[IY, IY] = sx**2
	S[IXT, IXT] = S[IYT, IYT] = 1e-8
	S[IE, IE] = sE**2
	mu = np.zeros(len(convention)); mu[IE] = E0
	lens = Lens(focal_length=0.02, aberrations={'Cc': cc})
	_, out = lens.propagate_moments(mu, S)
	rays = np.zeros((n, len(convention)))
	rays[:, IX] = rng.normal(0, sx, n); rays[:, IY] = rng.normal(0, sx, n)
	rays[:, IXT] = rng.normal(0, 1e-4, n); rays[:, IYT] = rng.normal(0, 1e-4, n)
	rays[:, IE] = rng.normal(E0, sE, n)
	traced = lens.propagate_ray(rays)
	assert np.isclose(out[IXT, IXT], np.var(traced[:, IXT]), rtol=0.02)


def test_chromatic_scales_with_the_energy_spread_squared():
	def growth(spread):
		scope = _toy(chromatic=3e-3)
		scope.sections[0].elements[0].energy_spread = spread
		scope.propagate_moments()
		base = _toy()
		base.sections[0].elements[0].energy_spread = spread
		base.propagate_moments()
		return (emittance(as_ndarray(scope.covariance_matrix))[-1, 0]**2
				- emittance(as_ndarray(base.covariance_matrix))[-1, 0]**2)
	assert np.isclose(growth(2e-2) / growth(1e-2), 4.0, rtol=0.02)


def test_an_ideal_reference_run_is_achromatic():
	# apply_aberrations=False must silence chromatic too, or every "ideal"
	# baseline in the repository is quietly contaminated once Cc is set
	scope = _toy(c30=1e3, chromatic=3e-3)
	scope.propagate_moments(apply_aberrations=False)
	reference = _toy()
	reference.propagate_moments()
	assert np.allclose(as_ndarray(scope.covariance_matrix),
					   as_ndarray(reference.covariance_matrix), rtol=1e-12)


def test_chromatic_survives_a_round_trip(tmp_path):
	scope = _toy(c30=1e3, chromatic=3e-3)
	base = str(tmp_path / "chromatic")
	scope.save(base)                                  # .json
	scope.to_sea(base + ".sea")
	from pySEA.rayTEM import load_microscope
	for path in (base, base + ".sea"):
		back = load_microscope(path)["L"]
		assert np.isclose(back.chromatic_aberration, 3e-3), path
		assert np.isclose(back.aberrations['C30'].real, 1e3), path


# ---- chromatic lives inside the aberration set --------------------------

def test_chromatic_is_carried_by_the_aberration_set():
	# one declaration carries everything the element does beyond its matrix,
	# so chromatic serializes, suspends and copies with the Krivanek terms
	ab = Aberrations({'C30': 1e-3, 'Cc': 1.2e-3})
	assert ab.chromatic == 1.2e-3
	assert ab['Cc'] == 1.2e-3
	assert 'Cc' in ab
	assert dict(ab.items()) == {'C30': 1e-3 + 0j}		# NOT a Krivanek term
	assert bool(Aberrations({'Cc': 1e-3}))				# chromatic alone is not ideal
	lens = Lens(focal_length=0.02, aberrations=ab)
	assert lens.chromatic_aberration == 1.2e-3
	lens.chromatic_aberration = 3e-3					# the property writes into the set
	assert lens.aberrations.chromatic == 3e-3


def test_a_chromatic_only_set_produces_no_pupil_deflection():
	# 'Cc' must never reach the Krivanek evaluators, which would not know it
	ab = Aberrations({'Cc': 2e-3})
	dx, dy = ab.deflection_at(np.array([1e-6]), np.array([2e-6]), 100.0)
	assert dx[0] == 0.0 and dy[0] == 0.0


def test_aberrations_round_trip_through_json_and_sea(tmp_path):
	# a column carrying aberrations could not be saved to JSON at all before:
	# Aberrations is a SEASerializable holding complex coefficients, and the
	# hand-rolled writer had no case for either
	scope = _toy(c30=1e3, chromatic=3e-3)
	scope["L"].aberrations['C12'] = (1e-9, 2e-9)		# complex, and json cannot hold it
	base = str(tmp_path / "ab")
	scope.save(base)
	scope.to_sea(base + ".sea")
	from pySEA.rayTEM import load_microscope
	for path in (base, base + ".sea"):
		back = load_microscope(path)["L"].aberrations
		assert np.isclose(back['C30'].real, 1e3), path
		assert np.isclose(back['C12'], 1e-9 + 2e-9j), path
		assert np.isclose(back.chromatic, 3e-3), path


def test_as_dict_is_the_flat_a_b_form_and_round_trips():
	# the storage form has no complex numbers in it, which is what lets a plain
	# JSON writer carry an aberrated column
	ab = Aberrations({'C30': 1e-3, 'C12': (2e-9, 3e-9), 'Cc': 1.2e-3})
	assert ab.as_dict() == {'C30': 1e-3, 'C12.a': 2e-9, 'C12.b': 3e-9, 'Cc': 1.2e-3}
	import json
	json.dumps(ab.as_dict())
	again = Aberrations.from_metadata(ab.as_dict())
	assert again.as_dict() == ab.as_dict()


# ---- astigmatic elements ------------------------------------------------

def test_quadrupole_aberrations_reach_every_path():
	# a quadrupole has no single focal power, so it used to resolve a pupil
	# scale of zero and its aberrations were silently ignored everywhere
	from pySEA.rayTEM import Quadrapole
	quad = Quadrapole(length=0.01, strength=30.0,
					  aberrations={'C30': 1e-3, 'Cc': 1.2e-3})
	assert quad.focal_power > 0
	rays = np.zeros((3, len(convention)))
	rays[:, IX] = [1e-5, 2e-5, 3e-5]
	rays[:, IY] = 1e-5
	rays[:, IE] = [199.9, 200.0, 200.1]
	assert quad._aberration_kick(rays) is not None
	assert np.any(quad._aberration_kick(rays)[2])
	assert np.any(quad._chromatic_deflection(rays)[0])
	assert np.any(quad.zone_power_shift(1e-5))
	S = np.eye(len(convention)) * 1e-12
	S[IE, IE] = 1e-4
	mu = np.zeros(len(convention)); mu[IE] = 200.0
	_, aberrated = quad.propagate_moments(mu, S)
	with suspended_aberrations([quad]):
		_, ideal = quad.propagate_moments(mu, S)
	assert aberrated[IXT, IXT] != ideal[IXT, IXT]


def test_chromatic_is_per_axis_on_an_astigmatic_element():
	# chromatic needs no round-pupil assumption: each axis sees its own power
	# scaled, so the kick ratio is exactly (P_x/P_y)^2
	from pySEA.rayTEM import Quadrapole
	quad = Quadrapole(length=0.01, strength=30.0, aberrations={'Cc': 1.2e-3})
	px, py = quad.focal_powers
	assert px * py < 0								# one axis focuses, one diverges
	rays = np.zeros((2, len(convention)))
	rays[:, IX] = rays[:, IY] = 1e-5
	rays[:, IE] = [199.9, 200.1]
	kx, ky = quad._chromatic_deflection(rays)
	assert np.isclose(kx[0] / ky[0], (px / py)**2)
	# and a round lens stays symmetric
	round_lens = Lens(focal_length=0.02, aberrations={'Cc': 1.2e-3})
	assert np.allclose(*round_lens._chromatic_deflection(rays))


# ---- reading the propagated covariance ----------------------------------

def test_postprocessing_reads_the_resolution_quantities():
	# the envelope mode stores a calibrated covariance Signal, and the three
	# postprocessing readers are how it is turned into resolution -- the same
	# structure the ray mode uses, results on the microscope and analysis in
	# postprocessing
	scope = _toy()
	scope.propagate_moments()
	cov = as_ndarray(scope.covariance_matrix)
	widths, eps = beam_widths(cov), emittance(cov)
	sx, st = widths[-1, 0], np.sqrt(cov[-1, IXT, IXT])
	c = cov[-1, IX, IXT]
	assert np.isclose(eps[-1, 0], np.sqrt(sx**2 * st**2 - c**2))
	real_w, real_v = resolution_ellipses(cov, 'real')
	assert np.all(real_w[:, 1] >= real_w[:, 0])
	assert np.allclose(real_v[-1] @ real_v[-1].T, np.eye(2), atol=1e-12)
	# for a round beam the principal widths are the per-axis ones
	assert np.allclose(np.sort(real_w[-1]), np.sort(widths[-1]))
	ang_w, _ = resolution_ellipses(cov, 'angular')
	assert np.isclose(ang_w[-1, 1], max(np.sqrt(cov[-1, IXT, IXT]),
										np.sqrt(cov[-1, IYT, IYT])))


def test_resolution_ellipses_rejects_an_unknown_block():
	scope = _toy()
	scope.propagate_moments()
	with pytest.raises(ValueError):
		resolution_ellipses(as_ndarray(scope.covariance_matrix), 'sideways')


def test_resolution_ellipses_finds_a_rotated_beam():
	# the point of reporting axes rather than sigma_x/sigma_y: a beam blurred
	# along a diagonal has no larger sigma_x than a round one of the same area
	cov = np.zeros((1,) + (len(convention),) * 2)
	sx, sy, rho = 3e-6, 3e-6, 0.8
	cov[0, IX, IX] = sx**2
	cov[0, IY, IY] = sy**2
	cov[0, IX, IY] = cov[0, IY, IX] = rho * sx * sy
	widths, axes = resolution_ellipses(cov, 'real')
	assert np.isclose(widths[0, 1], sx * np.sqrt(1 + rho))
	assert np.isclose(widths[0, 0], sx * np.sqrt(1 - rho))
	assert np.isclose(abs(np.degrees(np.arctan2(axes[0][1, 1], axes[0][0, 1]))), 45.0)


def test_the_covariance_is_stored_as_a_calibrated_signal():
	# the moments mode's result is a Signal, exactly as the wave mode's is --
	# there is no second container wrapping it
	scope = _toy()
	returned = scope.propagate_moments()
	assert returned is scope.covariance_matrix
	assert hasattr(scope.covariance_matrix, "dimensions")
	assert [d.name for d in scope.covariance_matrix.dimensions] == ['z', 'row', 'col']
	assert scope.covariance_matrix.metadata['components'] == list(convention)
	assert as_ndarray(scope.covariance_matrix).shape[1:] == (len(convention),) * 2


def test_a_different_closure_changes_the_answer():
	# the closure is a plain callable argument, so the Gaussian assumption is
	# visible at the call and replaceable -- a closure that returns zero for
	# every higher moment must give a different covariance
	scope_gauss = _toy(c30=1e5)
	scope_gauss.propagate_moments()
	scope_zero = _toy(c30=1e5)
	scope_zero.propagate_moments(closure=lambda Sigma, indices: (
		_gaussian_moment(Sigma, indices) if len(indices) <= 2 else 0.0))
	gauss = as_ndarray(scope_gauss.covariance_matrix)[-1, IXT, IXT]
	zero = as_ndarray(scope_zero.covariance_matrix)[-1, IXT, IXT]
	assert gauss > zero						# the discarded moments only add area


# ---- the example's own claims ------------------------------------------

def test_example_ideal_case_conserves_emittance():
	scope, planes = ex.propagate_case('ideal')
	cov = as_ndarray(scope.covariance_matrix)
	eps = emittance(cov)[:, 0]
	assert np.allclose(eps, eps[0], rtol=1e-6)
	for S in cov:									# still a physical covariance
		assert np.linalg.eigvalsh(0.5 * (S + S.T)).min() > -1e-12 * abs(S).max()


def test_example_ol2_cannot_affect_the_specimen():
	# causality: a post-specimen element must leave the specimen plane alone
	def eps_at(case, plane):
		scope, planes = ex.propagate_case(case)
		return emittance(as_ndarray(scope.covariance_matrix))[planes[plane], 0]
	ideal = eps_at('ideal', 'sample')
	assert np.isclose(eps_at('OL2', 'sample'), ideal, rtol=1e-12)
	assert eps_at('OL1', 'sample') > 1.5 * ideal


def test_example_disabling_either_objective_reproduces_the_single_case():
	both_scope = ex.build_case('both')
	with suspended_aberrations([both_scope["OL2"]]):
		both_scope.propagate_moments()
		without_ol2 = as_ndarray(both_scope.covariance_matrix)[-1]
	single = ex.build_case('OL1')
	single.propagate_moments()
	assert np.allclose(without_ol2, as_ndarray(single.covariance_matrix)[-1], rtol=1e-12)


def test_example_budget_is_dominated_by_ol1_and_nearly_additive():
	budget = ex.emittance_budget()
	assert abs(budget['OL1']) > 10 * abs(budget['OL2'])		# the pre-specimen lens dominates
	assert abs(budget['coupling']) < 0.05 * abs(budget['sum'])
	assert np.isclose(budget['both'], budget['sum'] + budget['coupling'])


def test_example_runs_at_the_stated_operating_point():
	# the whole configuration hangs on a 30 mrad nominal convergence: the
	# corrected Cs, the live chromatic term, and the balance between them are
	# only sensible there, so the emission angle is solved rather than guessed
	scope = ex.build_case('ideal')
	alpha = scope.convergence_angle_at(scope.get_element_position('sample'))
	assert np.isclose(alpha, ex.ALPHA_TARGET, rtol=1e-6)
	gun = scope.sections[0].elements[0]
	assert gun.size[0] == ex.SOURCE_SIZE
	assert gun.energy_spread == ex.ENERGY_SPREAD


def test_example_chromatic_is_a_live_term_at_this_aperture():
	# a corrected Cs and an uncorrected Cc is the real regime; chromatic must
	# be comparable to spherical here, not a rounding error
	budget = ex.emittance_budget(chromatic=True)
	assert budget['chromatic'] > 0.05 * abs(budget['OL1'])


def test_example_closure_validity_is_small_enough_to_justify_the_closure():
	# the closure asserts zero excess kurtosis; 27 f^2 is what it discards
	for name, v in ex.closure_validity('both').items():
		assert v['excess_kurtosis'] < 1e-3, (name, v)
		assert np.isclose(v['excess_kurtosis'], 27 * v['f']**2)
