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


# --- wave optics -----------------------------------------------------------

import os
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.seashells import make_wavefield_signal, read_wavefield, sea_available

def test_wave_plane_wave_focuses_at_focal_length():
	# A plane wave through a thin lens (focal length f) focuses a distance f behind it:
	# the on-axis intensity is maximal at z=f, and larger there than before/after.
	N, L, lam, f = 512, 4e-3, 500e-9, 0.5
	dx = dy = L/N
	field0 = wo.plane_wave((N,N))
	focused = wo.focal_phase(field0, dx, dy, lam, 1.0/f, 1.0/f)	# power = 1/f
	c = N//2
	def central_intensity(z):
		fz = wo.angular_spectrum_propagate(focused, dx, dy, lam, z)
		return np.abs(fz[c,c])**2
	I_half, I_focus, I_double = central_intensity(0.5*f), central_intensity(f), central_intensity(1.5*f)
	# on-axis intensity peaks at the focus and is a strong concentration vs the unit input
	assert I_focus > I_half and I_focus > I_double
	fz = wo.angular_spectrum_propagate(focused, dx, dy, lam, f)
	assert np.unravel_index(np.argmax(np.abs(fz)**2), fz.shape) == (c,c)
	assert I_focus > 50.0		# strong concentration relative to the unit-amplitude plane wave

def test_wave_fresnel_gaussian_spreading():
	# A Gaussian beam spreads per w(z) = w0 sqrt(1+(z/zR)^2), zR = pi w0^2 / lambda.
	N, L, lam = 1024, 8e-3, 500e-9
	dx = dy = L/N
	sigma = 300e-6						# amplitude field exp(-r^2/(2 sigma^2))
	w0 = np.sqrt(2)*sigma				# 1/e^2 intensity radius
	zR = np.pi*w0**2/lam
	field0 = wo.gaussian_field((N,N), dx, dy, sigma, sigma)

	def rms_width_x(field):
		I = np.abs(field)**2
		Ix = I.sum(axis=0)				# collapse y -> I(x)
		x = (np.arange(N)-N//2)*dx
		xbar = (x*Ix).sum()/Ix.sum()
		return np.sqrt(((x-xbar)**2*Ix).sum()/Ix.sum())

	# at z=0 the measured RMS equals sigma/sqrt(2): the intensity marginal exp(-x^2/sigma^2)
	# has variance sigma^2/2.
	rms0_expected = sigma/np.sqrt(2)
	assert abs(rms_width_x(field0) - rms0_expected)/rms0_expected < 0.02
	# at z=zR the beam should widen by ~sqrt(2)
	fz = wo.angular_spectrum_propagate(field0, dx, dy, lam, zR)
	ratio = rms_width_x(fz)/rms_width_x(field0)
	assert abs(ratio - np.sqrt(2))/np.sqrt(2) < 0.05, f"width ratio {ratio}, expected ~{np.sqrt(2)}"

def test_wave_microscope_stack_and_sea_roundtrip(tmp_path):
	from pySEA.rayTEM import Source,Lens,Drift,MicroscopeSection,Microscope
	section1 = MicroscopeSection(elements=[
		Source(voltage=200, field_shape=(64,64), field_extent=2e-3, field_kind='gaussian'),
		Drift(length=0.05), Lens(strength=8.0, length=0.0), Drift(length=0.05)])
	section2 = MicroscopeSection(elements=[Drift(length=0.05)])
	microscope = Microscope(sections=[section1, section2])
	wave = microscope.propagate_wave()

	data, dx, dy, wavelength, z = read_wavefield(wave)
	# single stacked (Nz, Ny, Nx) complex wavefield
	assert data.ndim == 3 and data.shape[1:] == (64,64)
	assert np.iscomplexobj(data)
	assert abs(wavelength - relativistic_wavelength(200)) < 1e-16

	if sea_available:
		# wavefield Signal must round-trip through .sea preserving complex data
		from pySEA.sea_eco.architecture.base_structure import Signal
		path = str(tmp_path/"wavefield.sea")
		wave.to_sea(path)
		reloaded = Signal(); reloaded.from_sea(path)
		assert np.allclose(reloaded.data, data)
		assert reloaded.data.dtype == data.dtype

def test_wave_prism_not_implemented():
	from pySEA.rayTEM import Prism
	prism = Prism(length=0.1, strength=0.2)
	dummy = make_wavefield_signal(wo.plane_wave((16,16)), 1e-4, 1e-4, 2.5e-12, z=0.0)
	with pytest.raises(NotImplementedError):
		prism.propagate_wave(dummy)
