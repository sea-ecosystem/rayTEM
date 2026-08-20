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
		Source(voltage=200, field_shape=(32, 32), field_extent=2e-3, angle=(0.02, 0.02)),
		Drift(length=0.05), Lens(strength=6.0, length=0.0),
		Quadrapole(strength=1.0, length=0.0), Dipole(strength=1e-6, axis="x"),
		Drift(length=0.05)])
	out = sec.propagate_wave()
	data, dx, dy, lam, z = read_wavefield(out)

	# manual legacy chain on the same initial field
	f0, *_ = read_wavefield(sec.elements[0].field())
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
	src = Source(voltage=200, field_shape=(n, n), field_extent=n * dx)
	psi0, dx_src, dy_src, _, _ = read_wavefield(src.aperture_field(radius))
	assert np.isclose(dx_src, dx) and np.isclose(dy_src, dx)
	assert np.allclose(psi0, wo.aperture_mask(wo.plane_wave((n, n)), dx, dx, radius))
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
	return [Source(voltage=200, field_shape=(64, 64), field_extent=64 * 2.5e-7),
			Drift(length=1e-3), Lens(strength=K, length=0.0, name="OL"),
			Drift(length=20e-3)]


def test_column_integration_scaled():
	radius = 5e-6
	sec = MicroscopeSection(elements=_scaled_column())
	out = sec.propagate_wave_scaled(field0=sec.elements[0].scaled_field(radius))

	# standalone chained element calls must match the driver bit-for-bit
	f = sec.elements[0].scaled_field(radius)
	for ele in sec.elements:
		f = ele.propagate_wave_scaled(f)
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
		assert out.get_dataset_names() == ["U", "s", "R", "tau"]
		n_planes = out["U"].data.shape[0]
		assert all(out[nm].dimensions[0].size == n_planes for nm in ("U", "s", "R", "tau"))
		assert np.isclose(out["s"].data[-1], s_end) and np.isclose(out["tau"].data[-1], tau_end)

	# dispatcher routes kind="wave-scaled" (element, section, and microscope)
	sec2 = MicroscopeSection(elements=_scaled_column())
	sec2.propagate(field0=sec2.elements[0].scaled_field(radius), kind="wave-scaled")
	U2 = read_scaled_wavefield(sec2._wave_scaled_planes[-1])[0]
	assert np.allclose(U2, U_drv)

	# Microscope driver + physical-wave reconstruction at a requested plane
	mic = Microscope(sections=[MicroscopeSection(elements=_scaled_column())])
	mic.propagate_wave_scaled(field0=mic.sections[0].elements[0].scaled_field(radius))
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
		Source(voltage=200, field_shape=(64, 64), field_extent=64 * 2.5e-7),
		Lens(strength=K, length=0.0), Drift(length=44.96e-3)])
	with pytest.raises(ValueError, match="crossover"):
		sec.propagate_wave_scaled(field0=sec.elements[0].scaled_field(5e-6))
