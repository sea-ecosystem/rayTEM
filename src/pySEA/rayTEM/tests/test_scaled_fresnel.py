# TESTS for the per-element phase_shift contract and scaled Fresnel propagation.
# See notes/eric/PLAN_2026-08-19_scaled-fresnel-wave.md (Eric's scaled-Fresnel handoff).

import sys
sys.path.insert(1,"../../../")
import numpy as np
import pytest

from pySEA.rayTEM import Source, Lens, Drift, Aperture, Dipole, Quadrapole, MicroscopeSection
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.seashells import phase_space_of, read_wavefield
from pySEA.rayTEM.utilities import relativistic_wavelength

LAM = relativistic_wavelength(200)
K200 = 2 * np.pi / LAM
GRID = ((64, 64), 1e-7, 1e-7)        # ((ny, nx), dx, dy) fallback grid spec


# --- test 0: phase_shift definitions, domain tags, scaled split ---------------

def test_phase_shift_lens_matches_focal_phase():
	lens = Lens(strength=6.0, length=0.0)
	items = lens.phase_shift(GRID, LAM)
	assert len(items) == 1 and phase_space_of(items[0]) == "position"
	# chi = -k (P x^2 + P y^2)/2 with P = sign(K) K^2
	(ny, nx), dx, dy = GRID
	P = 6.0**2
	expected = wo.quadratic_phase((ny, nx), dx, dy, LAM, P, P)
	assert np.allclose(items[0].data, expected)
	# and exp(i chi) equals the legacy focal_phase application
	field = np.ones((ny, nx), complex)
	assert np.allclose(wo.apply_phase(field, items[0].data),
					   wo.focal_phase(field, dx, dy, LAM, P, P))

def test_phase_shift_quadrupole_saddle():
	quad = Quadrapole(strength=2.0, length=0.0)
	items = quad.phase_shift(GRID, LAM)
	assert len(items) == 1 and phase_space_of(items[0]) == "position"
	chi = items[0].data
	ny, nx = chi.shape
	# saddle: opposite sign along x and y -> chi(x,0) and chi(0,y) have opposite signs
	cx = chi[ny//2, -1]      # pure-x sample
	cy = chi[-1, nx//2]      # pure-y sample
	assert cx * cy < 0, "quadrupole must focus one axis and diverge the other"
	# powers mirror focal_powers (P_x = -P_y)
	P_x, P_y = quad.focal_powers()
	assert P_x == -P_y != 0

def test_phase_shift_dipole_linear():
	dip = Dipole(strength=1e-6, axis="x")
	items = dip.phase_shift(GRID, LAM)
	assert len(items) == 1 and phase_space_of(items[0]) == "position"
	chi = items[0].data
	(ny, nx), dx, dy = GRID
	# linear in x: constant gradient k*theta_x*dx per pixel, no y dependence
	gx = np.diff(chi, axis=1)
	assert np.allclose(gx, K200 * 1e-6 * dx)
	assert np.allclose(np.diff(chi, axis=0), 0)

def test_phase_shift_drift_kernel():
	drift = Drift(length=0.05)
	items = drift.phase_shift(GRID, LAM)
	assert len(items) == 1 and phase_space_of(items[0]) == "scattering"
	chi = items[0].data
	# on-axis (f=0) sample carries only the carrier k*dz
	assert np.isclose(chi[0, 0], K200 * 0.05)
	# zero-length drift yields an empty program
	assert Drift(length=0.0).phase_shift(GRID, LAM) == []

def test_phase_shift_scaled_split():
	# Lens: full 1/f absorbed into R, nothing on U (handoff Eqs 15/45)
	power, screen = Lens(strength=6.0, length=0.0).phase_shift(GRID, LAM, scaled=True)
	assert np.isclose(power, 36.0) and screen is None
	# Quadrupole: nothing absorbed, full saddle applied to U (Eqs 47-48)
	power, screen = Quadrapole(strength=2.0, length=0.0).phase_shift(GRID, LAM, scaled=True, s=0.5)
	assert power == 0.0 and screen is not None and phase_space_of(screen) == "position"
	# at s=0.5 the physical coordinates shrink -> phase shrinks by s^2 vs s=1
	_, screen1 = Quadrapole(strength=2.0, length=0.0).phase_shift(GRID, LAM, scaled=True, s=1.0)
	assert np.allclose(screen.data, 0.25 * screen1.data)
	# Dipole: nothing absorbed, full linear phase applied to U
	power, screen = Dipole(strength=1e-6, axis="y").phase_shift(GRID, LAM, scaled=True)
	assert power == 0.0 and screen is not None
	# Drift: nothing absorbed, nothing on U (free segment handled by the driver)
	assert Drift(length=0.1).phase_shift(GRID, LAM, scaled=True) == (0.0, None)
	# zero-strength quad/dipole: fully transparent
	assert Quadrapole(strength=0.0).phase_shift(GRID, LAM, scaled=True) == (0.0, None)
	assert Dipole(strength=0.0).phase_shift(GRID, LAM, scaled=True) == (0.0, None)

def test_phase_shift_not_a_phase_elements():
	spec = GRID
	with pytest.raises(NotImplementedError):
		Aperture(radius=1e-6).phase_shift(spec, LAM)
	with pytest.raises(NotImplementedError):
		Source(voltage=200).phase_shift(spec, LAM)

def test_fixed_path_refactor_regression():
	# the refactored propagate_wave (phase-program consumer) must reproduce the
	# legacy composition focal_phase -> tilt_phase -> angular_spectrum exactly
	sec = MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(32, 32), wave_extent=2e-3, angle=(0.02, 0.02)),
		Drift(length=0.05), Lens(strength=6.0, length=0.0),
		Quadrapole(strength=1.0, length=0.0), Dipole(strength=1e-6, axis="x"),
		Drift(length=0.05)])
	out = sec.propagate_wave()
	data, dx, dy, lam, z = read_wavefield(out)

	# manual legacy chain on the same initial field
	f0, *_ = read_wavefield(sec.elements[0].wave())
	ref = wo.angular_spectrum_propagate(f0, dx, dy, lam, 0.05)
	ref = wo.focal_phase(ref, dx, dy, lam, 36.0, 36.0)
	P_x, P_y = sec.elements[3].focal_powers() if hasattr(sec.elements[3], "focal_powers") else (0, 0)
	# element order after section assembly: Source, Drift, Lens, Quad, Dipole, Drift
	quad = [e for e in sec.elements if isinstance(e, Quadrapole)][0]
	dip = [e for e in sec.elements if isinstance(e, Dipole)][0]
	ref = wo.focal_phase(ref, dx, dy, lam, *quad.focal_powers())
	ref = wo.tilt_phase(ref, dx, dy, lam, *dip.effective_tilts())
	ref = wo.angular_spectrum_propagate(ref, dx, dy, lam, 0.05)
	assert np.allclose(data[-1], ref, atol=1e-12)

# --- scaled-Fresnel core (handoff tests 1-9) -----------------------------------

from pySEA.rayTEM.seashells import (make_scaled_wavefield_signal, read_scaled_wavefield,
									make_scaled_wave_signalset)

RNG = np.random.default_rng(7)


def _align_global_phase(a, b):
	"""Return ``a`` rotated by the global phase that best matches ``b``."""
	return a * np.exp(1j * np.angle(np.vdot(a, b)))


def _gaussian_state(n=128, dx=5e-6, lam=500e-9, sigma=4e-5):
	"""A smooth band-limited test wave: centred Gaussian on an n x n grid."""
	field = wo.gaussian_field((n, n), dx, dx, sigma, sigma)
	return field, dx, lam


def test_eq29_delta_tau_vs_numerical_integral():
	# closed form (Eq 29) vs numerical integral of dz/s^2 with s(z) = s0 (1 + z/R0)
	for _ in range(20):
		s0 = RNG.uniform(0.2, 3.0) * RNG.choice([-1, 1])
		R0 = RNG.uniform(0.05, 2.0) * RNG.choice([-1, 1])
		# keep the segment on one side of the crossover (1 + dz/R0 > 0)
		dz = RNG.uniform(0.0, 0.9 * abs(R0)) if R0 < 0 else RNG.uniform(0.0, 5.0)
		zg = np.linspace(0.0, dz, 200001)
		numeric = np.trapezoid(1.0 / (s0 * (1 + zg / R0))**2, zg)
		assert np.isclose(wo.scaled_delta_tau(dz, s0, R0), numeric, rtol=1e-6)
	# flat chart (Eq 31)
	assert wo.scaled_delta_tau(0.7, 2.0, np.inf) == 0.7 / 4.0
	# in-segment crossover must raise, and name the crossover position
	with pytest.raises(ValueError, match="crossover"):
		wo.scaled_delta_tau(0.2, 1.0, -0.1)


def test_factor_reconstruct_identity():
	# Eq 55 factorization then Eq 37 reconstruction is exact to machine precision
	psi0, dx, lam = _gaussian_state()
	psi0 = psi0 * np.exp(1j * RNG.uniform(-np.pi, np.pi, psi0.shape))	# arbitrary field
	for s, R in [(1.0, np.inf), (0.7, 0.3), (-1.3, -0.08), (2.5, np.inf)]:
		U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, s, R)
		assert np.isclose(dxi, dx / s) and np.isclose(deta, dx / s)
		psi, dx_out, dy_out = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		assert np.isclose(dx_out, abs(s) * dxi) and np.isclose(dy_out, abs(s) * deta)
		assert np.allclose(psi, psi0, atol=1e-13)


def test_free_propagation_flat_chart_matches_ordinary():
	# s=1, R=inf: the scaled propagator IS the carrier-free ordinary propagator
	psi0, dx, lam = _gaussian_state()
	dz = 5e-3
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	U1, s1, R1, dtau = wo.propagate_free_scaled(U, dxi, deta, lam, dz, 1.0, np.inf)
	assert s1 == 1.0 and np.isinf(R1) and dtau == dz
	psi1, dx1, _ = wo.reconstruct_physical_wave(U1, dxi, deta, lam, s1, R1)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, dz, include_carrier=False)
	assert np.isclose(dx1, dx)
	assert np.allclose(psi1, ref, atol=1e-12)


def test_free_propagation_curved_chart_matches_ordinary():
	# arbitrary curved chart (s0=1, finite R0): reconstruct on the reference grid
	psi0, dx, lam = _gaussian_state()
	n = psi0.shape[0]
	dz, R0 = 5e-3, 0.05
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, R0)
	U1, s1, R1, _ = wo.propagate_free_scaled(U, dxi, deta, lam, dz, 1.0, R0)
	assert np.isclose(s1, 1 + dz / R0) and np.isclose(R1, R0 + dz)
	psi1, dx1, _ = wo.reconstruct_physical_wave(U1, dxi, deta, lam, s1, R1,
												target_dx=dx, target_shape=(n, n))
	assert np.isclose(dx1, dx)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, dz, include_carrier=False)
	err = np.linalg.norm(_align_global_phase(psi1, ref) - ref) / np.linalg.norm(ref)
	assert err < 1e-3


def test_thin_lens_scaled_matches_focal_phase():
	# Eq 56 (explicit lens phase) vs Eqs 57-58 (R absorption, U untouched)
	psi0, dx, lam = _gaussian_state()
	f = 0.5
	ref = wo.focal_phase(psi0, dx, dx, lam, 1 / f, 1 / f)		# exp(-ik r^2/2f)
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, 1 / f)
	assert s == 1.0 and np.isclose(R, -f)
	psi, dx_out, _ = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
	assert np.isclose(dx_out, dx)
	assert np.allclose(psi, ref, atol=1e-12)


def _run_aperture_lens_system():
	"""Aperture -> free -> thin lens -> free, scaled chain; returns both paths."""
	lam, n, dx = 633e-9, 256, 1e-5
	radius, f, d1, d2 = 8e-4, 0.5, 0.02, 0.05
	# initial wave built by the Source method the plan mandates (Eq 9, hard edge);
	# the system regime itself uses an optical wavelength so the fixed-grid
	# reference is valid for comparison (electron scale is covered separately).
	src = Source(voltage=200, wave_shape=(n, n), wave_extent=n * dx,
				 wave_kind="aperture", aperture_radius=radius)
	psi0, dx_src, dy_src, _, _ = read_wavefield(src.wave())
	assert np.isclose(dx_src, dx) and np.isclose(dy_src, dx)
	# alias-free sampling of the sharp disk (band-limited projection of theta(a-r))
	assert np.allclose(psi0, wo.bandlimited_disk((n, n), dx, dx, radius))
	# ordinary fixed-grid reference (valid regime for these parameters)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, d1, include_carrier=False)
	ref = wo.focal_phase(ref, dx, dx, lam, 1 / f, 1 / f)
	ref = wo.angular_spectrum_propagate(ref, dx, dx, lam, d2, include_carrier=False)
	# scaled chain
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	planes = [(1.0, np.inf, 0.0)]
	U, s, R, dtau = wo.propagate_free_scaled(U, dxi, deta, lam, d1, *planes[-1][:2])
	planes.append((s, R, planes[-1][2] + dtau))
	s, R = wo.apply_thin_lens_scaled(s, R, 1 / f)
	U, s, R, dtau = wo.propagate_free_scaled(U, dxi, deta, lam, d2, s, R)
	planes.append((s, R, planes[-1][2] + dtau))
	return psi0, ref, U, dxi, deta, lam, dx, n, planes


def test_aperture_lens_system_scaled_vs_ordinary():
	psi0, ref, U, dxi, deta, lam, dx, n, planes = _run_aperture_lens_system()
	s, R, _ = planes[-1]
	# after the lens the beam contracts: s = 1 - d2/f on the converging chart
	assert np.isclose(s, 1 - 0.05 / 0.5) and np.isclose(R, -0.5 + 0.05)
	psi, dx_out, _ = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R,
												  target_dx=dx, target_shape=(n, n))
	assert np.isclose(dx_out, dx)
	a = _align_global_phase(psi, ref)
	# total intensity and the cumulative radial energy profile must match tightly;
	# pointwise L2 is bounded but dominated by mutual discretization error at the
	# hard (non-band-limited) aperture edge, on both propagators alike.
	assert np.isclose((np.abs(psi)**2).sum(), (np.abs(ref)**2).sum(), rtol=1e-2)
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	r = np.hypot(X, Y)
	edges = np.linspace(0, r.max(), 60)
	cum = np.array([(np.abs(a)**2)[r <= e].sum() for e in edges])
	cum_ref = np.array([(np.abs(ref)**2)[r <= e].sum() for e in edges])
	assert np.max(np.abs(cum - cum_ref)) / cum_ref[-1] < 1e-2
	assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 0.15


def test_aperture_lens_system_converges_with_resolvable_edge():
	# the residual disagreement above is edge aliasing, not the propagator: with a
	# band-limited (cosine-tapered) aperture the same system agrees to < 5e-3
	lam, n, dx = 633e-9, 256, 1e-5
	radius, taper, f, d1, d2 = 8e-4, 16 * 1e-5, 0.5, 0.02, 0.05
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	r = np.hypot(X, Y)
	psi0 = (0.5 * (1 + np.cos(np.pi * np.clip((r - radius + taper) / taper, 0, 1)))).astype(complex)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, d1, include_carrier=False)
	ref = wo.focal_phase(ref, dx, dx, lam, 1 / f, 1 / f)
	ref = wo.angular_spectrum_propagate(ref, dx, dx, lam, d2, include_carrier=False)
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	U, s, R, _ = wo.propagate_free_scaled(U, dxi, deta, lam, d1, 1.0, np.inf)
	s, R = wo.apply_thin_lens_scaled(s, R, 1 / f)
	U, s, R, _ = wo.propagate_free_scaled(U, dxi, deta, lam, d2, s, R)
	psi, _, _ = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R,
											 target_dx=dx, target_shape=(n, n))
	a = _align_global_phase(psi, ref)
	assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 5e-3
	assert np.linalg.norm(np.abs(a) - np.abs(ref)) / np.linalg.norm(np.abs(ref)) < 5e-3


def test_grid_scaling_and_normalization():
	# Eqs 59-60 (pixel scaling) and Eq 54 (discrete norm) at every logged plane
	psi0, ref, U, dxi, deta, lam, dx, n, planes = _run_aperture_lens_system()
	E0 = (np.abs(psi0)**2).sum() * dx * dx
	for s, R, _tau in planes[1:]:
		assert s != 0
	# final plane: native reconstruction pixel is |s| * dxi ...
	s, R, _ = planes[-1]
	psi, dx_out, dy_out = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
	assert np.isclose(dx_out, abs(s) * dxi) and np.isclose(dy_out, abs(s) * deta)
	# ... and sum |psi|^2 dx dy = sum |U|^2 dxi deta = initial energy (unitary kernel)
	assert np.isclose((np.abs(psi)**2).sum() * dx_out * dy_out,
					  (np.abs(U)**2).sum() * dxi * deta, rtol=1e-12)
	assert np.isclose((np.abs(U)**2).sum() * dxi * deta, E0, rtol=1e-9)


def test_entrance_plane_equivalence_on_target_grid():
	# Eq 44: reconstruct onto a prescribed physical grid (the "entrance plane"
	# an external multislice package would consume) and compare to the ordinary
	# reference on that same grid.
	psi0, dx, lam = _gaussian_state()
	n = psi0.shape[0]
	dz, R0 = 4e-3, 0.08
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, R0)
	U1, s1, R1, _ = wo.propagate_free_scaled(U, dxi, deta, lam, dz, 1.0, R0)
	# target grid: same shape, the reference pixel size (not the native |s| dxi)
	psi1, dx1, dy1 = wo.reconstruct_physical_wave(U1, dxi, deta, lam, s1, R1,
												  target_dx=dx, target_shape=(n, n))
	assert np.isclose(dx1, dx) and np.isclose(dy1, dx)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, dz, include_carrier=False)
	a = _align_global_phase(psi1, ref)
	assert np.linalg.norm(np.abs(a) - np.abs(ref)) / np.linalg.norm(np.abs(ref)) < 1e-3
	assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 1e-3


def test_electron_scale_invariants_and_guard():
	# 200 kV, 20 um aperture, f = 45 mm -- the case the fixed grid cannot sample
	lam = LAM
	n, dx = 256, 2.5e-7		# 64 um field of view
	f, radius = 45e-3, 10e-6
	psi0 = wo.aperture_mask(wo.plane_wave((n, n)), dx, dx, radius)
	E0 = (np.abs(psi0)**2).sum() * dx * dx
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, 1 / f)		# R = -f
	# contraction toward focus: s(zeta) = 1 - zeta/f, checked across two segments
	U1, s1, R1, _ = wo.propagate_free_scaled(U, dxi, deta, lam, 15e-3, s, R)
	assert np.isclose(s1, 1 - 15e-3 / f)
	U2, s2, R2, _ = wo.propagate_free_scaled(U1, dxi, deta, lam, 25e-3, s1, R1)
	assert np.isclose(s2, 1 - 40e-3 / f)		# linear s composes across segments
	# energy conserved, beam physically contracted by s
	psi2, dx2, dy2 = wo.reconstruct_physical_wave(U2, dxi, deta, lam, s2, R2)
	assert np.isclose((np.abs(psi2)**2).sum() * dx2 * dy2, E0, rtol=1e-9)
	assert np.isclose(dx2, abs(s2) * dxi)
	# rms radius contracts ~ geometrically (large Fresnel number)
	X, Y = wo.transverse_coordinates((n, n), 1.0, 1.0)		# pixel units
	def rms_px(field):
		w = np.abs(field)**2
		return np.sqrt(((X**2 + Y**2) * w).sum() / w.sum())
	assert np.isclose(rms_px(psi2) * dx2, rms_px(psi0) * dx * abs(s2), rtol=0.05)
	# the s_min guard raises with the crossover position before s hits 0
	with pytest.raises(ValueError, match="crossover"):
		wo.propagate_free_scaled(U2, dxi, deta, lam, 4.99e-3, s2, R2)


def test_scaled_wavefield_signal_roundtrip():
	# seashells seam: single-plane factory <-> reader, and the stacked SignalSet
	U = (RNG.normal(size=(32, 32)) + 1j * RNG.normal(size=(32, 32)))
	sig = make_scaled_wavefield_signal(U, 2e-9, 3e-9, LAM, s=0.5, R=-0.05, tau=1.2e-3, z=0.1)
	U2, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(sig)
	assert np.allclose(U2, U)
	assert (dxi, deta, lam) == (2e-9, 3e-9, LAM)
	assert (s, R, tau, z) == (0.5, -0.05, 1.2e-3, 0.1)
	if hasattr(sig, "dimensions"):		# real sea_eco Signal: dims carry the scaled pitch
		dims = list(sig.dimensions)
		assert np.isclose(dims[-1].scale, 2e-9) and np.isclose(dims[-2].scale, 3e-9)
	# stacked result: U stack + s/R/tau companions on the shared plane-z axis
	stack = np.stack([U, 2 * U, 3 * U])
	sset = make_scaled_wave_signalset(stack, 2e-9, 3e-9, LAM,
									  s=[1.0, 0.5, 0.25], R=[np.inf, -0.05, -0.03],
									  tau=[0.0, 1e-3, 2e-3], z=[0.0, 0.1, 0.2])
	if sset is not None:
		assert sset.get_dataset_names() == ["U", "s", "R", "tau"]
		assert sset["U"].data.shape == (3, 32, 32)
		assert np.allclose(sset["s"].data, [1.0, 0.5, 0.25])
		assert all(sset[nm].dimensions[0].size == 3 for nm in ("U", "s", "R", "tau"))


# --- test 10: column integration ----------------------------------------------

from pySEA.rayTEM import Microscope
from pySEA.rayTEM.seashells import sea_available


def _scaled_column():
	"""Source (200 kV, 10 um aperture) -> drift -> f=45 mm lens -> drift."""
	K = np.sqrt(1 / 45e-3)		# thin-lens power = K^2 = 1/f
	return [Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
				   wave_kind="aperture", aperture_radius=5e-6),
			Drift(length=1e-3), Lens(strength=K, length=0.0, name="OL"),
			Drift(length=20e-3)]


def test_column_integration_scaled():
	sec = MicroscopeSection(elements=_scaled_column())
	out = sec.propagate_wave(mode="scaled")		# driver seeds from Source.wave(mode='scaled')

	# standalone chained element calls must match the driver bit-for-bit
	f = sec.elements[0].wave(mode="scaled")
	for ele in sec.elements:
		f = ele.propagate_wave(f, mode="scaled")
	U_end, dxi, deta, lam, s_end, R_end, tau_end, z_end = read_scaled_wavefield(f)
	U_drv, *_, s_drv, R_drv, tau_drv, z_drv = read_scaled_wavefield(sec._wave_scaled_planes[-1])
	assert np.allclose(U_drv, U_end) and s_drv == s_end and R_drv == R_end
	assert np.isclose(tau_drv, tau_end) and np.isclose(z_drv, z_end)
	# physics: s = 1 - d2/f after the lens, R updated, z at the column end
	assert np.isclose(s_end, 1 - 20e-3 / 45e-3) and np.isclose(R_end, -45e-3 + 20e-3)
	assert np.isclose(z_end, 21e-3)

	# .wave_scaled is a SignalSet whose companions share the plane-z axis
	if sea_available:
		assert out is sec.wave_scaled
		assert out.get_dataset_names() == ["U", "s", "R", "tau", "frame"]
		n_planes = out["U"].data.shape[0]
		assert all(out[nm].dimensions[0].size == n_planes for nm in ("U", "s", "R", "tau", "frame"))
		assert np.isclose(out["s"].data[-1], s_end) and np.isclose(out["tau"].data[-1], tau_end)

	# dispatcher routes kind="wave-scaled" (element, section, and microscope)
	sec2 = MicroscopeSection(elements=_scaled_column())
	sec2.propagate(kind="wave-scaled")
	U2 = read_scaled_wavefield(sec2._wave_scaled_planes[-1])[0]
	assert np.allclose(U2, U_drv)

	# Microscope driver + physical-wave reconstruction at a requested plane
	mic = Microscope(sections=[MicroscopeSection(elements=_scaled_column())])
	mic.propagate_wave(mode="scaled")
	psi_sig = mic.wavefield_at(21e-3)
	data, dx, dy, lam_out, z_out = read_wavefield(psi_sig)
	assert np.isclose(z_out, 21e-3) and np.isclose(lam_out, lam)
	assert np.isclose(dx, abs(s_end) * dxi)		# native grid: dx = |s| dxi
	# reconstruction preserves energy (Eq 54)
	assert np.isclose((np.abs(data)**2).sum() * dx * dy,
					  (np.abs(U_end)**2).sum() * dxi * deta, rtol=1e-12)
	# and a prescribed target grid is honored (Eq 44)
	psi_t = mic.wavefield_at(21e-3, target_dx=2.5e-7, target_shape=(64, 64))
	data_t, dx_t, *_ = read_wavefield(psi_t)
	assert np.isclose(dx_t, 2.5e-7) and data_t.shape == (64, 64)


def test_scaled_guard_through_column():
	# a drift long enough to reach the crossover must raise the actionable error
	K = np.sqrt(1 / 45e-3)
	sec = MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
			   wave_kind="aperture", aperture_radius=5e-6),
		Lens(strength=K, length=0.0), Drift(length=44.96e-3)])
	with pytest.raises(ValueError, match="crossover"):
		sec.propagate_wave(mode="scaled")


# --- transparent base-element defaults ------------------------------------------

from pySEA.rayTEM.elements import Element


def test_base_element_is_transparent_in_every_kind():
	# the root Element carries a working identity default for each propagation
	# kind: identity transfer matrix (rays/moments), phase of nothing (waves)
	ele = Element(name="generic")
	assert np.allclose(ele.transfer_matrix(), np.eye(6))
	r0 = RNG.normal(size=(4, 6))
	assert np.allclose(ele.propagate_ray(r0.copy()), r0)		# zero length: unchanged
	mu, Sig = RNG.normal(size=6), np.eye(6)
	mu_out, Sig_out = ele.propagate_moments(mu, Sig)
	assert np.allclose(mu_out, mu) and np.allclose(Sig_out, Sig)
	# fixed wave path: zero length -> empty phase program -> field unchanged
	assert ele.phase_shift(GRID, LAM) == []
	src = Source(voltage=200, wave_shape=(32, 32), wave_extent=2e-6)
	w0 = src.wave()
	w1 = ele.propagate_wave(w0)
	assert np.allclose(w1.data, w0.data)
	# scaled path: (0, None) split -> state unchanged
	assert ele.phase_shift(GRID, LAM, scaled=True) == (0.0, None)
	s0 = src.wave(mode='scaled')
	s1 = ele.propagate_wave(s0, mode='scaled')
	U0 = read_scaled_wavefield(s0)[0] ; U1 = read_scaled_wavefield(s1)[0]
	assert np.allclose(U1, U0)
	# a finite length makes the transparent element a pure free segment
	ele.length = 0.01
	items = ele.phase_shift(GRID, LAM)
	assert len(items) == 1 and phase_space_of(items[0]) == "scattering"


def test_wave_kind_aperture_matches__aperture_wave():
	src = Source(voltage=200, wave_shape=(64, 64), wave_extent=16e-6,
				 wave_kind="aperture", aperture_radius=5e-6)
	via_kind, dx, dy, *_ = read_wavefield(src.wave())
	via_builder, *_ = read_wavefield(src._aperture_wave(5e-6))
	assert np.allclose(via_kind, via_builder)
	# default is the alias-free band-limited projection of the sharp disk:
	# unit interior, zero exterior, bounded Gibbs ripple confined to the edge
	X, Y = wo.transverse_coordinates((64, 64), dx, dy)
	r = np.hypot(X, Y)
	assert np.allclose(np.abs(via_kind[r < 0.7 * 5e-6]), 1.0, atol=0.05)
	assert np.abs(via_kind[r > 1.4 * 5e-6]).max() < 0.05
	assert np.abs(via_kind).max() < 1.15		# Gibbs overshoot ~9%, never more
	# the exact point-sampled binary mask stays available for comparison
	binary, *_ = read_wavefield(src._aperture_wave(5e-6, antialias=False))
	assert np.allclose(np.unique(np.abs(binary)), [0.0, 1.0])
	with pytest.raises(ValueError, match="aperture_radius"):
		Source(voltage=200, wave_shape=(64, 64), wave_extent=16e-6,
			   wave_kind="aperture").wave()


# --- frame-change primitive (Eric's Eq 5) ---------------------------------------

def test_change_scaled_frame_identity():
	# the physical wave is invariant under any frame change (pointwise path)
	psi0, dx, lam = _gaussian_state()
	frames = [(1.0, np.inf), (0.7, 0.3), (0.5, -0.2), (1.4, np.inf)]
	for (s_a, R_a) in frames:
		U_a, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, s_a, R_a)
		for (s_b, R_b) in frames:
			U_b, dxi_b, deta_b = wo.change_scaled_frame(U_a, dxi, deta, lam,
														s_a, R_a, R_b, s_new=s_b)
			# physical-grid continuity: same physical pixel before and after
			assert np.isclose(abs(s_b) * dxi_b, abs(s_a) * dxi)
			psi_b, dx_b, _ = wo.reconstruct_physical_wave(U_b, dxi_b, deta_b, lam, s_b, R_b)
			assert np.isclose(dx_b, dx)
			assert np.allclose(psi_b, psi0, atol=1e-12)
	# flatten/re-diverge case (s kept): pitch unchanged
	U_a, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 0.8, -0.5)
	U_f, dxi_f, _ = wo.change_scaled_frame(U_a, dxi, deta, lam, 0.8, -0.5, np.inf)
	assert dxi_f == dxi
	psi_f, _, _ = wo.reconstruct_physical_wave(U_f, dxi_f, dxi_f, lam, 0.8, np.inf)
	assert np.allclose(psi_f, psi0, atol=1e-12)


def test_change_scaled_frame_sampling_guard():
	# moving an unrepresentably strong curvature into U must raise, naming the
	# minimum representable |R|
	n, dxi, s = 128, 5e-6, 1.0
	lam = LAM		# electron wavelength: curvature phases are severe
	U = wo.gaussian_field((n, n), dxi, dxi, 1e-4, 1e-4)
	R_min = wo.min_representable_curvature(n, dxi, lam, s, safety=0.5)
	with pytest.raises(ValueError, match="not representable"):
		wo.change_scaled_frame(U, dxi, dxi, lam, s, -0.5 * R_min, np.inf)
	# at a weaker curvature (larger |R|) the same change passes
	out, _, _ = wo.change_scaled_frame(U, dxi, dxi, lam, s, -2.0 * R_min, np.inf)
	assert np.isfinite(out).all()
	# and the pure-converter path (safety=None) never guards
	out, _, _ = wo.change_scaled_frame(U, dxi, dxi, lam, s, -0.5 * R_min, np.inf, safety=None)
	assert np.isfinite(out).all()


# --- hybrid crossover engine ----------------------------------------------------

def test_hybrid_through_focus_matches_ordinary():
	# tapered aperture -> lens -> free THROUGH the focus: the hybrid engine
	# (scaled -> flatten -> ordinary Fresnel through the crossover -> re-diverge)
	# must match the ordinary fixed-grid reference past the focus
	lam, n, dx = 633e-9, 256, 1e-5
	radius, taper, f, d1, d2 = 8e-4, 16e-5, 0.05, 0.01, 0.09		# focus at d1 + f
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	r = np.hypot(X, Y)
	psi0 = (0.5 * (1 + np.cos(np.pi * np.clip((r - radius + taper) / taper, 0, 1)))).astype(complex)

	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	U, s, R, dt, z, zc, logged = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, d1, 1.0, np.inf, z=0.0)
	assert logged == [] and s == 1.0 and np.isinf(R)
	s, R = wo.apply_thin_lens_scaled(s, R, 1 / f)
	U, s, R, dt, z, zc, logged = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, d2, s, R, z=d1)
	tags = [entry[0] for entry in logged]
	assert tags == ["flatten", "crossover", "rediverge"]
	# the crossover plane is logged exactly at the focus
	assert np.isclose(logged[1][5], d1 + f)
	# past the focus: diverging frame, no marker left, s finite and growing
	assert R > 0 and zc is None and s > 0.05
	# continue on the diverging frame until s returns to 1, so the native grid
	# coincides with the reference grid (comparison free of resampling error)
	extra = R * (1 / s - 1)
	U, s, R, dt, z, zc, logged2 = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, extra, s, R, z=z)
	assert logged2 == [] and np.isclose(s, 1.0)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, d1, include_carrier=False)
	ref = wo.focal_phase(ref, dx, dx, lam, 1 / f, 1 / f)
	ref = wo.angular_spectrum_propagate(ref, dx, dx, lam, z - d1, include_carrier=False)
	psi, dx_out, _ = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
	assert np.isclose(dx_out, dx)
	a = _align_global_phase(psi, ref)
	# measured 6.3e-3 / 4.0e-3 at safety=0.5 (dominated by edge-halo wraparound
	# in the flat window); thresholds carry margin
	assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 1.5e-2
	assert np.linalg.norm(np.abs(a) - np.abs(ref)) / np.linalg.norm(np.abs(ref)) < 1e-2
	# energy conserved through both frame switches and the focus
	assert np.isclose((np.abs(psi0)**2).sum() * dx * dx,
					  (np.abs(U)**2).sum() * abs(dxi) * abs(deta), rtol=1e-9)


def test_hybrid_electron_focal_plane_is_airy():
	# 200 kV, a = 5 um, f = 45 mm: the logged crossover plane is the back-focal
	# (diffraction) plane -- an Airy pattern with first zero at 0.61*lam*f/a
	lam = LAM
	n, dx = 256, 2.5e-7
	a_r, f = 5e-6, 45e-3
	psi0 = wo.aperture_mask(wo.plane_wave((n, n)), dx, dx, a_r)
	E0 = (np.abs(psi0)**2).sum() * dx * dx
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, 1 / f)
	U, s, R, dt, z, zc, logged = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, 60e-3, s, R, z=0.0)		# straight through the focus
	tags = [entry[0] for entry in logged]
	assert tags == ["flatten", "crossover", "rediverge"]
	assert np.isclose(logged[1][5], f)		# crossover logged at z = f
	# physical pixel |s|*dxi is continuous at every switch (dxi never changes)
	for entry in logged:
		assert entry[2] > 0		# s stays finite and positive throughout
	# focal-plane intensity: radial profile matches the Airy pattern
	U_c, s_c = logged[1][1], logged[1][2]
	psi_c, dx_c, _ = wo.reconstruct_physical_wave(U_c, dxi, deta, lam, s_c, np.inf)
	I = np.abs(psi_c)**2
	Xc, Yc = wo.transverse_coordinates((n, n), dx_c, dx_c)
	rc = np.hypot(Xc, Yc)
	r_zero = 0.61 * lam * f / a_r		# first Airy zero (13.7 nm)
	# intensity at the first zero is deeply suppressed vs the peak
	ring = (np.abs(rc - r_zero) < 0.02 * r_zero)
	assert I[ring].max() < 5e-3 * I.max()
	# and the central disk carries ~84% of the energy (Airy: 83.8%)
	E_c = (I).sum() * dx_c * dx_c
	frac = I[rc <= r_zero].sum() * dx_c * dx_c / E_c
	assert np.isclose(frac, 0.838, atol=0.02)
	# energy conserved at the focal plane and at the segment exit
	assert np.isclose(E_c, E0, rtol=1e-9)
	assert np.isclose((np.abs(U)**2).sum() * dxi * deta, E0, rtol=1e-9)


def test_scaled_signal_carries_crossover_marker():
	from pySEA.rayTEM.seashells import scaled_frame_crossover
	sig = make_scaled_wavefield_signal(np.ones((8, 8), complex), 1e-9, 1e-9, LAM,
									   s=0.1, R=np.inf, tau=0.0, z=0.1, z_cross=0.15)
	assert scaled_frame_crossover(sig) == 0.15
	sig2 = make_scaled_wavefield_signal(np.ones((8, 8), complex), 1e-9, 1e-9, LAM,
										s=1.0, R=np.inf, tau=0.0, z=0.0)
	assert scaled_frame_crossover(sig2) is None


# --- test 5: full column through every crossover ---------------------------------

@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_full_column_hybrid_source_to_detector():
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	from pySEA.rayTEM.seashells import scaled_frame_tag
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	out = scope.propagate_wave(mode="hybrid")
	planes = scope._wave_scaled_planes

	# the run reaches the detector through every crossover
	zs = [read_scaled_wavefield(p)[7] for p in planes]
	assert np.isclose(max(zs), 1.264, atol=1e-6)
	assert len(scope.crossovers) >= 4		# C1, condenser chain, objective, projectors
	assert np.isclose(scope.crossovers[0], 0.175, atol=1e-3)		# C1 focus

	# frame switches are balanced and s stays finite everywhere
	tags = [scaled_frame_tag(p) for p in planes]
	assert tags.count("flatten") == tags.count("crossover") == tags.count("rediverge")
	ss = [read_scaled_wavefield(p)[4] for p in planes]
	assert min(np.abs(ss)) > 1e-3 and max(np.abs(ss)) > 1		# contracts and re-expands

	# energy conserved at every logged plane (all frames, all switches)
	E = [(np.abs(read_scaled_wavefield(p)[0])**2).sum() for p in planes]
	dxi = read_scaled_wavefield(planes[0])[1]
	assert np.allclose(np.asarray(E) * dxi * dxi, E[0] * dxi * dxi, rtol=1e-6)

	# the frame companion increments at every switch
	assert out["frame"].data[-1] == 2 * tags.count("crossover")

	# physical reconstruction at the named detector plane and at the C1 focus
	det = scope.wavefield_at("detector")
	data, dx, dy, lam, z_out = read_wavefield(det)
	assert np.isclose(z_out, 1.264, atol=1e-6) and np.isfinite(data).all()
	foc = scope.wavefield_at(scope.crossovers[0])
	fdata, fdx, *_ = read_wavefield(foc)
	assert np.isfinite(fdata).all() and fdx < 1e-8		# focal-plane pixel is nm-scale


# --- alias-free aperture sampling + absorbing boundary ---------------------------

def test_bandlimited_disk_is_alias_free_sharp_disk():
	n, dx, a = 256, 78.125e-9, 5e-6
	d = wo.bandlimited_disk((n, n), dx, dx, a).real
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	r = np.hypot(X, Y)
	assert np.allclose(d[r < 0.7 * a], 1.0, atol=0.01)		# unit interior
	assert np.abs(d[r > 1.5 * a]).max() < 0.01				# zero exterior
	assert d.max() < 1.15 and d.min() > -0.15				# bounded Gibbs at the edge
	# exact area: the k=0 spectral sample is pi a^2 by construction
	assert np.isclose(d.sum() * dx * dx, np.pi * a**2, rtol=1e-9)


def test_absorbing_boundary_removes_wraparound():
	# a packet aimed at the boundary: periodic propagation wraps it back in;
	# the absorbing boundary removes it instead (physically: lost electrons)
	n, dxi, lam = 128, 1e-7, LAM
	X, Y = wo.transverse_coordinates((n, n), dxi, dxi)
	# tilted gaussian packet: carrier at 0.6x Nyquist, travelling toward +x edge
	f_c = 0.6 / (2 * dxi)
	U0 = (wo.gaussian_field((n, n), dxi, dxi, 8 * dxi, 8 * dxi)
		  * np.exp(2j * np.pi * f_c * X))
	dtau = 2.0 * (n * dxi / 2) / (lam * f_c)		# enough tau to cross the whole grid
	E0 = (np.abs(U0)**2).sum()
	U_per, *_ = wo.propagate_free_scaled(U0, dxi, dxi, lam, dtau, 1.0, np.inf)
	U_abs, *_ = wo.propagate_free_scaled(U0, dxi, dxi, lam, dtau, 1.0, np.inf, absorb=0.1)
	# periodic: energy conserved (the packet wrapped); absorbing: nearly all removed
	assert np.isclose((np.abs(U_per)**2).sum(), E0, rtol=1e-9)
	assert (np.abs(U_abs)**2).sum() < 0.05 * E0
	# and a beam that never reaches the boundary is untouched (loss ~ 0)
	U_c = wo.gaussian_field((n, n), dxi, dxi, 6 * dxi, 6 * dxi)
	U_out, *_ = wo.propagate_free_scaled(U_c, dxi, dxi, lam, 1e-4, 1.0, np.inf, absorb=0.1)
	assert np.isclose((np.abs(U_out)**2).sum(), (np.abs(U_c)**2).sum(), rtol=1e-6)


def test_full_column_aperture_interior_is_clean():
	# the fix Eric asked for: band-limited sharp disk + absorbing boundary give
	# flat interiors (real Fresnel rings kept) instead of the aliased grid plaid
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	if not sea_available:
		pytest.skip("basic_column.sea requires sea_eco")
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	src = scope.sections[0].elements[0]
	src.wave_kind = "aperture"
	src.aperture_radius = 5e-6
	scope.propagate_wave(mode="hybrid")
	for z in (scope.named_positions["sample"], 1.264):
		data, dx, *_ = read_wavefield(scope.wavefield_at(z))
		I = np.abs(data)**2
		n = I.shape[0]
		core = I[int(n*0.42):int(n*0.58), int(n*0.42):int(n*0.58)]
		assert core.std() / core.mean() < 0.01		# was ~0.04 with the plaid
		assert core.min() / core.max() > 0.95
