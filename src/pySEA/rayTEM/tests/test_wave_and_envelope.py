# TESTS for the wavelength utility, beam-envelope (covariance) propagation, and
# wave-optics propagation. See notes/eric/PLAN_2026-08-08_signal-and-propagation-additions.md

import sys
sys.path.insert(1,"../../../")
import numpy as np
import pytest

from pySEA.rayTEM.utilities import relativistic_wavelength


# --- wavelength ------------------------------------------------------------

def test_relativistic_wavelength_known_values():
	# Standard TEM textbook values (picometres)
	assert abs(relativistic_wavelength(200)*1e12 - 2.508) < 1e-3
	assert abs(relativistic_wavelength(300)*1e12 - 1.969) < 1e-3
	# monotonic decrease with voltage
	assert relativistic_wavelength(80) > relativistic_wavelength(200) > relativistic_wavelength(300)

def test_relativistic_wavelength_rejects_nonpositive():
	with pytest.raises(ValueError):
		relativistic_wavelength(0)
	with pytest.raises(ValueError):
		relativistic_wavelength(-100)


# --- beam-envelope (covariance) propagation --------------------------------

from pySEA.rayTEM import Source,Lens,Drift,MicroscopeSection,columnByName,findPlanes
from pySEA.rayTEM import beam_widths,emittance,fix_ray_dims
from pySEA.rayTEM.postprocessing import zFromFractional

def test_moments_match_monte_carlo_covariance():
	# Analytic covariance propagation (Sigma' = M Sigma M^T) must match the sample
	# covariance of a large ray bundle traced through the same elements.
	cx,cxt,cy,cyt = [columnByName(v) for v in ["x","xt","y","yt"]]
	section = MicroscopeSection(elements=[Drift(length=1.0), Lens(strength=1.2,length=0.0), Drift(length=0.7)])

	var = {cx:0.4**2, cxt:0.05**2, cy:0.3**2, cyt:0.04**2}
	Sigma0 = np.zeros((6,6))
	for k,v in var.items():
		Sigma0[k,k] = v
	mu0 = np.zeros(6)

	# analytic
	cov = section.propagate_moments(mu0=mu0, Sigma0=Sigma0)

	# monte-carlo: sample a bundle from N(mu0, Sigma0) and trace it
	rng = np.random.default_rng(0)
	N = 200000
	r0 = np.zeros((N,6))
	for k,v in var.items():
		r0[:,k] = rng.standard_normal(N)*np.sqrt(v)
	rays = section.propagate_ray(r0)

	# compare the 4x4 transverse sub-block at the final plane
	idx = [cx,cxt,cy,cyt]
	sample_cov = np.cov(rays[-1][:,idx].T)
	analytic_cov = cov[-1][np.ix_(idx,idx)]
	# relative Frobenius error small for large N
	rel = np.linalg.norm(sample_cov-analytic_cov)/np.linalg.norm(analytic_cov)
	assert rel < 0.02, f"covariance mismatch, rel={rel}"

def test_emittance_conserved_through_symplectic_elements():
	# Drifts and thin lenses are symplectic (det M = 1), so RMS emittance is invariant.
	section = MicroscopeSection(elements=[
		Source(size=(0.3,0.2),np_xy=(9,9),angle=(0.05,0.04),na_xy=(9,9)),
		Drift(length=1.0), Lens(strength=1.5,length=0.0), Drift(length=1.3)])
	section.propagate_moments()
	eps = emittance(section.covariance_matrix)		# (n_planes, 2)
	# emittance in x and y should be constant across all planes
	assert np.ptp(eps[:,0])/eps[0,0] < 1e-9
	assert np.ptp(eps[:,1])/eps[0,1] < 1e-9

def test_envelope_waist_matches_ray_optics_focus():
	# A collimated bundle (zero angular spread) through a thin lens of focal length f
	# focuses a distance f behind the lens. The covariance beam-waist z must coincide
	# with the ray-optics diffraction plane (where the initially-parallel rays cross).
	d1, K = 2.0, np.sqrt(2.0)		# thin lens: 1/f = K^2 -> f = 0.5
	f = 1.0/K**2
	slices = [Drift(length=0.02) for _ in range(60)]	# fine z sampling after the lens
	section = MicroscopeSection(elements=[
		Source(size=(0.5,0.5),np_xy=(9,9),angle=(0.0,0.0),na_xy=(1,1)),
		Drift(length=d1), Lens(strength=K,length=0.0), *slices])

	# envelope: z of minimum x-width
	section.propagate_moments()
	widths = beam_widths(section.covariance_matrix)[:,0]
	zs_env = section.mu[:,columnByName("z")]
	z_waist = zs_env[np.argmin(widths)]

	# ray optics: diffraction plane (parallel rays crossing)
	r1 = section.propagate_ray()
	planes = findPlanes(r1, section.R, axis="x")
	zs_ray = r1[:,0,columnByName("z")]
	z_diff = zFromFractional(zs_ray, planes["x"]["diff"]["z"][0])

	assert abs(z_waist - (d1+f)) < 0.05, f"waist z={z_waist}, expected {d1+f}"
	assert abs(z_waist - z_diff) < 0.05, f"waist z={z_waist} vs ray diff plane {z_diff}"
