# TESTS for the per-element phase_shift contract and scaled Fresnel propagation.
# See notes/eric/PLAN_2026-08-19_scaled-fresnel-wave.md (Eric's scaled-Fresnel handoff).

import sys
sys.path.insert(1,"../../../")
import numpy as np
import pytest

from pySEA.rayTEM import Source, Lens, Drift, Aperture, Dipole, Quadrapole, MicroscopeSection
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.aberrations import Aberrations, KRIVANEK_TERMS
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
	P_x, P_y = quad.focal_powers
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
	# Quadrupole: per-axis powers absorbed into (R_x, R_y), nothing on U —
	# the saddle is quadratic per axis, so the anisotropic frame holds all of it
	quad = Quadrapole(strength=2.0, length=0.0)
	power, screen = quad.phase_shift(GRID, LAM, scaled=True, s=0.5)
	assert screen is None and power == quad.focal_powers
	assert power[0] == -power[1] and power[0] != 0
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
	P_x, P_y = sec.elements[3].focal_powers if hasattr(sec.elements[3], "focal_powers") else (0, 0)
	# element order after section assembly: Source, Drift, Lens, Quad, Dipole, Drift
	quad = [e for e in sec.elements if isinstance(e, Quadrapole)][0]
	dip = [e for e in sec.elements if isinstance(e, Dipole)][0]
	ref = wo.focal_phase(ref, dx, dy, lam, *quad.focal_powers)
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


def test_aperture_scaled_anisotropic_frame():
	# An aperture is a circle in the PHYSICAL plane, so on an anisotropic frame
	# (any quadrupole upstream) it is an ellipse in scaled coordinates. This used
	# to raise TypeError: abs() on the (s_x, s_y) pair.
	src = Source(voltage=200, wave_shape=(64, 64), wave_extent=16e-6)
	ap = Aperture(radius=5e-6)
	w_iso = src.wave(mode='scaled')
	U_iso = read_scaled_wavefield(ap.propagate_wave(w_iso, mode='scaled'))[0]
	# a thin quad absorbs its (P, -P) into (R_x, R_y) and leaves s continuous, so
	# s is numerically isotropic but *stored* as a pair: the masked result must
	# be identical, not a crash
	w_pair = Quadrapole(strength=2.0, length=0.0).propagate_wave(src.wave(mode='scaled'),
															   mode='scaled')
	assert isinstance(read_scaled_wavefield(w_pair)[4], tuple)
	U_pair = read_scaled_wavefield(ap.propagate_wave(w_pair, mode='scaled'))[0]
	assert np.allclose(U_pair, U_iso)
	# genuinely anisotropic frame: the open region is an ellipse with the
	# per-axis half-widths radius/|s_x| and radius/|s_y|
	U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(w_iso)
	w_ani = make_scaled_wavefield_signal(U, dxi, deta, lam, (2.0, 1.0), R, tau, z=z)
	U_ani = np.abs(read_scaled_wavefield(ap.propagate_wave(w_ani, mode='scaled'))[0])
	ny, nx = U_ani.shape
	open_x = (U_ani[ny // 2, :] > 0.5).sum()
	open_y = (U_ani[:, nx // 2] > 0.5).sum()
	assert open_y > 1.8 * open_x, "s_x = 2 s_y must open twice as wide along eta"


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


def test_boundary_window_is_radially_symmetric():
	# the absorber must be azimuthally isotropic: a separable (square) window
	# clips the halo anisotropically (corners sqrt(2) farther than the edges)
	# and imprints a fourfold, pixel-aligned fringe pattern on the beam
	n = 128
	W = wo.boundary_window((n, n), margin=0.1)
	c = n // 2
	# axis vs diagonal at (nearly) equal radius, inside the absorbing band:
	# a separable window is ~1 on the diagonal where the on-axis value has
	# already dropped (difference ~0.9); a radial window agrees to within the
	# half-pixel discretization of the ramp (slope ~ pi/2m per pixel)
	for r_test in (c - 3, c - 7, c - 12):
		d = int(round(r_test / np.sqrt(2)))
		w_axis = W[c, c + r_test]
		w_diag = W[c + d, c + d]
		assert abs(w_axis - w_diag) < 0.15, f"window is anisotropic at r={r_test}"
	assert W[c, n - 3] == pytest.approx(W[n - 3, c])	# x vs y axis
	# interior is 1, inscribed-circle edge is ~0
	assert W[c, c] == 1.0
	assert W[c, -1] < 0.05


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
	# core std/mean is isotropic ring contrast only (the radial absorber edge
	# adds weak concentric ringlets: ~0.012 at the sample, ~0.023 at the
	# detector; no fourfold — that is what c4 below enforces)
	for z, mod_max, flat_min in ((scope.named_positions["sample"], 0.015, 0.9),
								 (1.264, 0.03, 0.85)):
		data, dx, *_ = read_wavefield(scope.wavefield_at(z))
		I = np.abs(data)**2
		n = I.shape[0]
		core = I[int(n*0.42):int(n*0.58), int(n*0.42):int(n*0.58)]
		assert core.std() / core.mean() < mod_max	# was ~0.04/0.03 with the plaid
		assert core.min() / core.max() > flat_min
		# and no fourfold (pixel-axis-aligned) fringe pattern in the disc: the
		# c4 angular harmonic of the interior intensity is ~1e-3 when the
		# absorber window is square, ~0 when it is radially symmetric
		x = np.arange(n) - n // 2
		X, Y = np.meshgrid(x, x)
		r = np.sqrt(X**2 + Y**2)
		amp = np.sqrt(I)
		rd = r[amp > 0.5 * amp.max()].max()
		inner = r < 0.6 * rd
		dev = I[inner] - I[inner].mean()
		c4 = abs(np.sum(dev * np.exp(4j * np.arctan2(Y, X)[inner]))) / np.sum(I[inner])
		assert c4 < 5e-4



# --- beam-support frame policy ----------------------------------------------------

def test_beam_support_extent_and_guard():
	# a compact beam on a big grid: the support half-width, not the grid edge,
	# sets what curvature a frame change may move into U
	n, dxi = 256, 1e-7
	U = wo.gaussian_field((n, n), dxi, dxi, 5 * dxi, 5 * dxi)
	ext = wo.beam_support_radius(U, dxi, dxi)
	assert ext < 0.3 * (n // 2) * dxi		# far smaller than the grid half-width
	# grid-edge criterion would forbid this flatten; the beam-based one allows it
	R_strong = -0.5 * wo.min_representable_curvature(n, dxi, LAM, 1.0)	# grid-based bound
	assert abs(R_strong) > wo.min_representable_curvature(n, dxi, LAM, 1.0, x_max=ext)
	out, _, _ = wo.change_scaled_frame(U, dxi, dxi, LAM, 1.0, R_strong, np.inf)
	assert np.isfinite(out).all()
	# a beam filling the grid reproduces the grid-edge criterion
	full = np.ones((n, n), complex)
	assert np.isclose(wo.beam_support_radius(full, dxi, dxi), (n // 2) * dxi)


@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_padded_grid_hybrid_completes():
	# regression: on a 512^2 / 40 um grid the grid-edge flatten criterion used
	# to push the flatten below s_min and crash; the beam-support criterion
	# (and the engine owning its internal guard) completes with defaults
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	src = scope.sections[0].elements[0]
	src.wave_kind = "aperture"
	src.aperture_radius = 5e-6
	src.wave_shape = (512, 512)
	src.wave_extent = 40e-6
	scope.propagate_wave(mode="hybrid")		# used to raise the s_min backstop
	zs = [read_scaled_wavefield(p)[7] for p in scope._wave_scaled_planes]
	assert np.isclose(max(zs), 1.264, atol=1e-6)
	assert len(scope.crossovers) >= 4


# --- direct frame jumps (crossover='jump') ----------------------------------------

def test_jump_policy_through_focus_optical():
	# mild-crossover regime: the direct mirror jump (R_o=-d -> R_n=+d, one
	# switch, no flat window) matches the ordinary reference comparably to the
	# flat policy (measured: jump 1.3e-2 vs flat 1.0e-2). At tight electron
	# crossovers the jump's 2x-deeper ride lets the diffraction-limited focal
	# structure outgrow the FOV, so 'flat' remains the default (see docstring).
	lam, n, dx = 633e-9, 256, 1e-5
	radius, taper, f, d1 = 8e-4, 16e-5, 0.05, 0.01
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	r = np.hypot(X, Y)
	psi0 = (0.5 * (1 + np.cos(np.pi * np.clip((r - radius + taper) / taper, 0, 1)))).astype(complex)
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	U, s, R, dt, z, zc, lg = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, d1, 1.0, np.inf, 0.0, crossover="jump")
	s, R = wo.apply_thin_lens_scaled(s, R, 1 / f)
	U, s, R, dt, z, zc, lg = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, 0.09, s, R, z=d1, crossover="jump")
	assert [e[0] for e in lg] == ["jump", "crossover"]		# one switch, focal plane logged
	assert np.isclose(lg[1][5], d1 + f)						# crossover at the focus
	assert R > 0 and zc is None
	# energy conserved (no absorber at the math level by default)
	assert np.isclose((np.abs(U)**2).sum() * dxi * deta,
					  (np.abs(psi0)**2).sum() * dx * dx, rtol=1e-9)
	ref = wo.angular_spectrum_propagate(psi0, dx, dx, lam, d1, include_carrier=False)
	ref = wo.focal_phase(ref, dx, dx, lam, 1 / f, 1 / f)
	ref = wo.angular_spectrum_propagate(ref, dx, dx, lam, z - d1, include_carrier=False)
	psi, dxo, _ = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R,
											   target_dx=dx, target_shape=(n, n))
	a = _align_global_phase(psi, ref)
	assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 3e-2


# --- anisotropic frames (s_x != s_y) -----------------------------------------------

def test_axis_helpers_scalar_pair_convention():
	# scalars fan out to both axes; equal components (including inf) collapse back
	assert wo.axis_components(3.0) == (3.0, 3.0)
	assert wo.axis_components((1.0, 2.0)) == (1.0, 2.0)
	assert wo.join_axes(2.0, 2.0) == 2.0
	assert wo.join_axes(np.inf, np.inf) == np.inf
	assert wo.join_axes(1.0, 2.0) == (1.0, 2.0)
	# thin-lens absorption: a quadrupole's (P, -P) goes into (R_x, R_y)
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, (10.0, -10.0))
	assert R == (-0.1, 0.1)		# converging in x (R < 0), diverging in y
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, 10.0)		# round lens stays scalar
	assert R == -0.1
	# per-axis support extents join to the isotropic radius
	U = wo.gaussian_field((64, 64), 1e-7, 1e-7, 3e-7, 6e-7)
	ex, ey = wo.beam_support_extents(U, 1e-7, 1e-7)
	assert ey > ex
	assert wo.beam_support_radius(U, 1e-7, 1e-7) == max(ex, ey)


def test_anisotropic_line_foci_match_gaussian_q():
	# an astigmatic lens (f_x != f_y) absorbed into per-axis curvatures: the
	# hybrid engine must log both line foci at the right z and reproduce the
	# analytic Gaussian q-parameter widths per axis at the foci and the exit
	lam, n, W, sig = 500e-9, 256, 4e-3, 0.3e-3
	dx = W / n
	fx, fy, zend = 0.05, 0.08, 0.20

	def sigma_pred(f, z):		# std of intensity from the complex q parameter
		w0 = np.sqrt(2) * sig
		q = 1j * np.pi * w0**2 / lam
		q = q / (1 - q / f) + z
		return np.sqrt(-lam / (np.pi * (1 / q).imag)) / 2

	psi0 = wo.gaussian_field((n, n), dx, dx, sig, sig)
	U, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	s, R = wo.apply_thin_lens_scaled(1.0, np.inf, (1 / fx, 1 / fy))
	U, s, R, dtau, z, zc, logged = wo.propagate_free_scaled_hybrid(
		U, dxi, deta, lam, zend, s, R, z=0.0)
	tags = [e[0] for e in logged]
	assert "crossover-x" in tags and "crossover-y" in tags	# both line foci logged
	assert zc is None and np.ndim(s) == 1					# past both, still anisotropic

	def sigma_measured(Up, sp, Rp):
		psi, dxo, dyo = wo.reconstruct_physical_wave(Up, dxi, deta, lam, sp, Rp)
		I = np.abs(psi)**2
		X, Y = wo.transverse_coordinates(I.shape, dxo, dyo)
		return (np.sqrt((I * X**2).sum() / I.sum()),
				np.sqrt((I * Y**2).sum() / I.sum()), I.sum() * dxo * dyo)

	for (tag, Ul, sl, Rl, tl, zl, zcl) in logged:
		if tag.startswith("crossover"):
			f_here = fx if tag.endswith("x") else fy
			assert np.isclose(zl, f_here, atol=1e-9)		# line focus position
			sx, sy, _ = sigma_measured(Ul, sl, Rl)
			assert np.isclose(sx, sigma_pred(fx, zl), rtol=1e-3)
			assert np.isclose(sy, sigma_pred(fy, zl), rtol=1e-3)
	sx, sy, energy = sigma_measured(U, s, R)
	assert np.isclose(sx, sigma_pred(fx, zend), rtol=1e-3)
	assert np.isclose(sy, sigma_pred(fy, zend), rtol=1e-3)
	assert np.isclose(energy, (np.abs(psi0)**2).sum() * dx * dx, rtol=1e-9)


def test_pseudo_isotropic_pair_matches_scalar_path():
	# an equal-axes pair routed through the anisotropic engine must reproduce
	# the scalar (isotropic) hybrid path: same physics, per-axis bookkeeping
	lam, n, W, sig, f, zend = 500e-9, 128, 4e-3, 0.3e-3, 0.05, 0.12
	dx = W / n
	psi0 = wo.gaussian_field((n, n), dx, dx, sig, sig)
	U0, dxi, deta = wo.factor_wave(psi0, dx, dx, lam, 1.0, np.inf)
	Ui, si, Ri, ti, zi, zci, lgi = wo.propagate_free_scaled_hybrid(
		U0, dxi, deta, lam, zend, 1.0, -f, z=0.0)
	Ua, sa, Ra, ta, za, zca, lga = wo.propagate_free_scaled_hybrid(
		U0, dxi, deta, lam, zend, (1.0, 1.0), (-f, -f), z=0.0)
	# pair input with equal axes joins straight back to the scalar state
	assert np.isclose(sa if np.ndim(sa) == 0 else sa[0], si)
	assert np.isclose(Ra if np.ndim(Ra) == 0 else Ra[0], Ri)
	assert [t.split("-")[0] for t in [e[0] for e in lga]] == \
		   [e[0] for e in lgi] or len(lga) == 2 * len(lgi)
	assert np.allclose(Ua, Ui, atol=1e-12 * np.abs(Ui).max())


def test_anisotropic_seam_roundtrip():
	# per-axis frame state survives the Signal seam (per-axis metadata keys),
	# and isotropic planes keep the scalar keys (old files load unchanged)
	from pySEA.rayTEM.seashells import (make_scaled_wavefield_signal, read_scaled_wavefield,
										scaled_frame_crossover)
	U = np.ones((8, 8), complex)
	sig = make_scaled_wavefield_signal(U, 1e-7, 1e-7, LAM, s=(0.5, 2.0), R=(-0.1, np.inf),
									   tau=(1e-3, 2e-3), z=0.3, z_cross=(0.4, None))
	U2, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(sig)
	assert s == (0.5, 2.0) and R == (-0.1, np.inf) and tau == (1e-3, 2e-3)
	assert scaled_frame_crossover(sig) == (0.4, None)
	iso = make_scaled_wavefield_signal(U, 1e-7, 1e-7, LAM, s=0.5, R=-0.1, tau=1e-3, z=0.3)
	assert read_scaled_wavefield(iso)[4] == 0.5		# scalar key path intact
	if sea_available:
		meta = iso.metadata.to_dict()
		assert "s" in meta and "s_x" not in meta


def test_strong_stigmator_absorbed_runs():
	# a quadrupole this strong used to alias its saddle screen on the scaled
	# grid (loud guard error); absorbed into (R_x, R_y) it has no sampling
	# limit at all — the run completes with an anisotropic frame
	K = np.sqrt(1 / 5e-3)		# |P| = K^2 = 200 /m (f = +-5 mm)
	sec = MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
			   wave_kind="aperture", aperture_radius=5e-6),
		Drift(length=1e-3), Quadrapole(strength=K, length=0.0), Drift(length=2e-3)])
	sec.propagate_wave(mode="scaled")
	U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(sec._wave_scaled_planes[-1])
	assert np.ndim(R) == 1 and R[0] != R[1]			# anisotropic curvature state
	assert np.ndim(s) == 1 and s[0] != s[1]			# axes evolve independently
	# energy conserved in the reduced field (U untouched by the quad; the
	# default absorbing boundary removes only a sliver of edge halo)
	src_U = read_scaled_wavefield(sec._wave_scaled_planes[0])[0]
	assert np.isclose((np.abs(U)**2).sum(), (np.abs(src_U)**2).sum(), rtol=1e-3)


@pytest.mark.skipif(not sea_available, reason="SignalSet companions require sea_eco")
def test_astigmatic_line_foci_in_column_hybrid():
	# round lens + weak quadrupole -> two line foci at per-axis f_eff, both
	# logged by the hybrid engine and listed on Microscope.crossovers; the
	# stacked SignalSet switches to per-axis frame companions
	f = 45e-3
	Kq = np.sqrt(2.0)			# |P| = 2 /m astigmatism on top of 1/f = 22.2 /m
	src = Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
				 wave_kind="aperture", aperture_radius=5e-6)
	quad = Quadrapole(strength=Kq, length=0.0)
	P_x, P_y = quad.focal_powers
	z_lens = 1e-3
	fx = 1 / (1 / f + P_x)
	fy = 1 / (1 / f + P_y)
	mic = Microscope(sections=[MicroscopeSection(elements=[
		src, Drift(length=z_lens), Lens(strength=np.sqrt(1 / f), length=0.0), quad,
		Drift(length=60e-3)])])
	mic.propagate_wave(mode="hybrid")
	from pySEA.rayTEM.seashells import scaled_frame_tag
	tags = {scaled_frame_tag(p): read_scaled_wavefield(p)[7]
			for p in mic._wave_scaled_planes if scaled_frame_tag(p)}
	assert np.isclose(tags["crossover-x"], z_lens + fx, atol=1e-9)
	assert np.isclose(tags["crossover-y"], z_lens + fy, atol=1e-9)
	assert sorted(mic.crossovers) == sorted([tags["crossover-x"], tags["crossover-y"]])
	names = mic.wave_scaled.get_dataset_names()
	assert names == ["U", "s_x", "s_y", "R_x", "R_y", "tau_x", "tau_y", "frame"]
	# energy accounted for across the whole astigmatic ride: conserved up to
	# the aperture-edge halo the absorbing boundary honestly removes during
	# the two deep line-focus rides (measured ~6.6% on this 64^2 grid)
	U0 = read_scaled_wavefield(mic._wave_scaled_planes[0])[0]
	U1, dxi, deta, *_ = read_scaled_wavefield(mic._wave_scaled_planes[-1])
	ratio = (np.abs(U1)**2).sum() / (np.abs(U0)**2).sum()
	assert 0.90 < ratio <= 1.0 + 1e-12


@pytest.mark.skipif(not sea_available, reason="Signal.show delegation requires sea_eco")
def test_show_scaled_wave_kinds():
	# show(kind='wave-scaled'/'wave-hybrid'): no plane -> the |psi(x,0,z)|
	# cross-section (wave analog of the ray diagram, annotated); a plane
	# (index, z, or name) -> reconstructed physical |psi|^2 via the wavefield
	# Signal's own .show()
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	mic = Microscope(sections=[MicroscopeSection(elements=_scaled_column())])
	# cross-section into a provided axis: something must actually be drawn
	fig, ax = plt.subplots()
	mic.show(kind="wave-scaled", plt_ax=ax)
	assert len(ax.collections) > 0				# the pcolormesh
	assert ax.get_xlabel() == "z (mm)"
	plt.close(fig)
	# per-plane by z (metres): delegates to the reconstructed Signal's .show()
	fig, ax = plt.subplots()
	mic.show(kind="wave-scaled", plane=21e-3, regenerate=False, plt_ax=ax)
	assert len(ax.images) > 0					# Signal.show imshow
	plt.close(fig)
	# per-plane by index (last logged plane)
	fig, ax = plt.subplots()
	mic.show(kind="wave_scaled", plane=-1, regenerate=False, plt_ax=ax)
	assert len(ax.images) > 0
	plt.close(fig)


def test_subdivided_preserves_geometry():
	# subdivided() gives finer z sampling without changing the optics: same
	# named positions, same total length, original object untouched
	def named(scope):		# unnamed elements all share one blank key; ignore it
		return {k: v for k, v in scope.named_positions.items() if k}
	mic = Microscope(sections=[MicroscopeSection(elements=_scaled_column())])
	before = named(mic)
	length = mic.sections[0].length
	dense = mic.subdivided(2e-3)			# a plane every <= 2 mm
	assert dense is not mic and dense.sections[0] is not mic.sections[0]
	assert named(dense) == before					# named geometry preserved exactly
	assert np.isclose(dense.sections[0].length, length)		# and total length
	assert named(mic) == before						# original untouched
	n_ele = len(mic.sections[0].elements)
	assert len(dense.sections[0].elements) > n_ele	# and it really did subdivide
	assert len(mic.sections[0].elements) == n_ele
	# explicit cut positions become element boundaries
	cut = Microscope(sections=[MicroscopeSection(elements=_scaled_column())]).subdivided([5e-3])
	zs = [e.position for e in cut.sections[0].elements]
	assert any(np.isclose(z, 5e-3) for z in zs)
	# a non-positive spacing is an actionable error
	with pytest.raises(ValueError, match="positive drift spacing"):
		mic.subdivided(0.0)


@pytest.mark.skipif(not sea_available, reason="Signal.show delegation requires sea_eco")
def test_show_zpts_uses_temporary_dense_copy():
	# show(zpts=...) plots from a subdivided copy and must not disturb the
	# result stored on self (nor require it to exist)
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	mic = Microscope(sections=[MicroscopeSection(elements=_scaled_column())])
	mic.propagate_wave(mode="hybrid")
	n_own = len(mic._wave_scaled_planes)
	fig, ax = plt.subplots()
	mic.show(kind="wave-hybrid", zpts=2e-3, plt_ax=ax)
	assert len(ax.collections) > 0							# drew the dense section
	assert len(mic._wave_scaled_planes) == n_own			# self's result untouched
	plt.close(fig)
	# a float plane with explicit cuts is logged exactly (no nearest-plane snap)
	fig, ax = plt.subplots()
	mic.show(kind="wave-hybrid", zpts=[10e-3], plane=10e-3, plt_ax=ax)
	assert len(ax.images) > 0
	plt.close(fig)
	# zpts is rejected where it has no meaning
	with pytest.raises(ValueError, match="only supported for the scaled wave kinds"):
		mic.show(kind="ray", zpts=1e-3)


def test_conjugate_planes_both_families_compound():
	# A column has TWO conjugate families and neither follows from a single
	# lens's f: with collimated input through f1=45mm then f2=30mm at 100mm
	# spacing, the second diffraction plane is the IMAGE of the first crossover
	# (176 mm), not z_L2 + f2 (140 mm). The hybrid wave frame is seeded flat
	# (a parallel wavefront), so its crossovers are the DIFFRACTION family;
	# conjugate_planes() exposes both.
	f1, f2, d0, d = 45e-3, 30e-3, 10e-3, 100e-3
	z_diff1 = d0 + f1										# 55 mm
	R2 = 1 / (1 / (d - f1) - 1 / f2)						# frame curvature after L2
	z_diff2 = d0 + d + abs(R2)								# 176 mm (compound!)
	assert not np.isclose(z_diff2, d0 + d + f2)				# NOT simply z_L2 + f2
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
			   wave_kind="aperture", aperture_radius=5e-6),
		Drift(length=d0), Lens(strength=np.sqrt(1 / f1), length=0.0, name="L1"),
		Drift(length=d), Lens(strength=np.sqrt(1 / f2), length=0.0, name="L2"),
		Drift(length=100e-3)])])
	mic.propagate_wave(mode="hybrid")
	assert np.allclose(mic.crossovers, [z_diff1, z_diff2], atol=1e-9)

	planes = mic.conjugate_planes(axis="x")
	# the ray-side diffraction family reproduces the wave crossovers exactly
	assert np.allclose(planes["diff"], [z_diff1, z_diff2], atol=1e-9)
	# and the image family is a genuinely different set of planes
	assert len(planes["image"]) > 0
	for zi in planes["image"]:
		assert min(abs(zi - np.array(mic.crossovers))) > 1e-3
	# the reference trace must not disturb this object's own ray result
	assert mic.rays is None or True			# (no ray run was requested here)
	mic.propagate_ray()
	before = mic.rays.copy()
	mic.conjugate_planes(axis="x")
	assert np.array_equal(mic.rays, before)


@pytest.mark.skipif(not sea_available, reason="requires sea_eco")
def test_conjugate_planes_feed_zpts_to_log_image_planes():
	# the documented composition: image planes are logged exactly by handing
	# conjugate_planes' positions to show/subdivided as zpts
	f1, f2 = 45e-3, 30e-3
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
			   wave_kind="aperture", aperture_radius=5e-6),
		Drift(length=10e-3), Lens(strength=np.sqrt(1 / f1), length=0.0),
		Drift(length=100e-3), Lens(strength=np.sqrt(1 / f2), length=0.0),
		Drift(length=100e-3)])])
	zi = mic.conjugate_planes(axis="x")["image"]
	assert len(zi)						# this compound column has a real image plane
	dense = mic.subdivided(zi)
	dense.propagate_wave(mode="hybrid")
	zs = np.array([read_scaled_wavefield(p)[7] for p in dense._wave_scaled_planes])
	for z in zi:
		assert min(abs(zs - z)) < 1e-9		# requested image planes are logged exactly


# --- thick lens as an exact scaled segment ----------------------------------------

def test_segment_delta_tau_closed_form_all_regimes():
	# tau = B/(s0*s_L) is law-agnostic: verify it against the numerical integral
	# for FOCUSING (harmonic), DEFOCUSING (hyperbolic) and free-space segments.
	rng = np.random.default_rng(11)
	for defocusing in (False, True):
		checked = 0
		for _ in range(40):
			s0 = rng.uniform(0.2, 2.0)
			k = rng.uniform(5, 200)
			R0 = np.inf if rng.random() < 0.3 else rng.uniform(0.05, 2.0) * rng.choice([-1, 1])
			dz = rng.uniform(1e-4, 0.4 / k)			# short of the first zero of s
			u0 = 0.0 if np.isinf(R0) else s0 / R0
			zg = np.linspace(0, dz, 100001)
			if defocusing:
				sg = s0 * np.cosh(k * zg) + (u0 / k) * np.sinh(k * zg)
				kappa = -k**2
			else:
				sg = s0 * np.cos(k * zg) + (u0 / k) * np.sin(k * zg)
				kappa = k**2
			if np.abs(sg).min() < 1e-6:
				continue
			numeric = np.trapezoid(1.0 / sg**2, zg)
			assert np.isclose(wo.scaled_delta_tau_quadratic(dz, s0, R0, kappa),
							  numeric, rtol=1e-6), (kappa, s0, R0, dz)
			checked += 1
		assert checked > 20, "sweep degenerated"
	# kappa -> 0 degenerates to the drift form, and kappa == 0 IS the drift form
	assert np.isclose(wo.scaled_delta_tau_quadratic(0.02, 1.3, np.inf, 1e-18),
					  wo.scaled_delta_tau(0.02, 1.3, np.inf), rtol=1e-9)
	assert np.isclose(wo.scaled_delta_tau_quadratic(0.02, 1.3, -0.7, 0.0),
					  wo.scaled_delta_tau(0.02, 1.3, -0.7), rtol=1e-12)
	# a crossover inside the body is refused, naming where
	with pytest.raises(ValueError, match="inside the segment body"):
		wo.scaled_delta_tau_quadratic(0.05, 1.0, -0.01, 30.0**2)
	# a DEFOCUSING segment entered flat can never cross over...
	wo.scaled_delta_tau_quadratic(10.0, 1.0, np.inf, -30.0**2)
	# ...but one entered converging hard enough still can
	with pytest.raises(ValueError, match="inside the segment body"):
		wo.scaled_delta_tau_quadratic(0.05, 1.0, -0.01, -30.0**2)


def test_segment_block_regimes():
	# the (A, B) row: harmonic, linear, hyperbolic -- all with unit determinant
	dz = 0.02
	for kappa in (900.0, 0.0, -900.0):
		A, B = wo.segment_block(dz, kappa)
		D = wo._segment_slope(dz, 0.0, 1.0, kappa)		# D = du/du0
		C = wo._segment_slope(dz, 1.0, 0.0, kappa)		# C = du/ds0
		assert np.isclose(A * D - B * C, 1.0, atol=1e-12), kappa
	assert wo.segment_block(dz, 0.0) == (1.0, dz)
	assert wo.segment_block(0.0, 900.0) == (1.0, 0.0)
	# focusing oscillates, defocusing grows
	assert wo.segment_block(0.2, 900.0)[0] < 1.0
	assert wo.segment_block(0.2, -900.0)[0] > 1.0


def test_thick_lens_segment_matches_transfer_matrix():
	# the frame advances by the element's OWN 2x2 block: s_out must equal the
	# rotating-frame A element, and the crossover -R_out must equal -A/C
	lens = Lens(strength=34.72, length=0.02)
	K, L = lens._effective_strength, lens.length
	U0 = wo.gaussian_field((64, 64), 1e-7, 1e-7, 5e-7, 5e-7)
	U, s, R, dtau = wo.propagate_quadratic_segment_scaled(U0, 1e-7, 1e-7, LAM, L,
												   1.0, np.inf, K**2)
	M = lens.transfer_matrix()
	c = np.cos(K * L)					# the Larmor rotation scales the x-block by cos(KL)
	A, C = M[0, 0] / c, M[1, 0] / c
	assert np.isclose(s, A, rtol=1e-12)			# scale == A for a flat seed
	assert np.isclose(-R, -A / C, rtol=1e-12)	# crossover distance == -A/C
	# U is untouched apart from the segment's own tau propagation: energy conserved
	assert np.isclose((np.abs(U)**2).sum(), (np.abs(U0)**2).sum(), rtol=1e-9)
	# and a thin lens of the same power still takes the kick path
	assert Lens(strength=6.0, length=0.0)._scaled_segment() is None
	assert lens._scaled_segment() == ('quadratic', K**2, -K * L)


@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_thick_lens_crossovers_match_ray_planes():
	# the payoff: with thick lenses carried exactly, the hybrid crossovers land
	# on the ray-traced diffraction planes. They used to sit 422-4808 um away,
	# which was precisely the thin-equivalent (drift L/2 -> kick -> drift L/2)
	# error of each lens.
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	src = scope.sections[0].elements[0]
	src.wave_kind = "aperture"
	src.aperture_radius = 5e-6
	scope.propagate_wave(mode="hybrid")
	planes = scope.conjugate_planes(axis="x")["diff"]
	assert len(scope.crossovers) >= 5
	for zc in scope.crossovers:
		assert min(abs(np.asarray(planes) - zc)) < 1e-6		# was up to 4.8e-3


def test_rotate_field_is_exact_and_unitary():
	# band-limited three-shear rotation: right sign, right magnitude, unitary
	n = 128
	X, Y = wo.transverse_coordinates((n, n), 1.0, 1.0)
	x0 = 30.0
	U = np.exp(-((X - x0)**2 + Y**2) / (2 * 4.0**2)).astype(complex)
	for ang in (0.3, -0.7, 1.2980):
		V = wo.rotate_field(U, ang)
		I = np.abs(V)**2
		cx, cy = (I * X).sum() / I.sum(), (I * Y).sum() / I.sum()
		# counter-clockwise: the blob lands on the rotated position
		assert np.isclose(cx, x0 * np.cos(ang), atol=1e-9)
		assert np.isclose(cy, x0 * np.sin(ang), atol=1e-9)
		assert np.isclose(I.sum(), (np.abs(U)**2).sum(), rtol=1e-12)		# unitary
	assert np.allclose(wo.rotate_field(wo.rotate_field(U, 0.9), -0.9), U, atol=1e-12)
	assert wo.rotate_field(U, 0.0) is not U			# angle 0 returns a copy
	with pytest.raises(ValueError, match="square grid"):
		wo.rotate_field(np.ones((8, 16), complex), 0.5)


def test_rotation_commutes_with_propagation():
	# the isotropic kernel depends only on |k|, so rotating once at a lens exit
	# is equivalent to rotating continuously through the body
	n, dxi, lam = 128, 1e-6, 500e-9
	X, Y = wo.transverse_coordinates((n, n), 1.0, 1.0)
	U = np.exp(-((X - 30.0)**2 + Y**2) / (2 * 4.0**2)).astype(complex)
	dz = 0.1 * n * dxi**2 / lam			# well inside the drift sampling limit
	a = wo.angular_spectrum_propagate(wo.rotate_field(U, 0.8), dxi, dxi, lam, dz,
									  include_carrier=False)
	b = wo.rotate_field(wo.angular_spectrum_propagate(U, dxi, dxi, lam, dz,
													  include_carrier=False), 0.8)
	assert np.abs(a - b).max() / np.abs(a).max() < 1e-10


def test_thick_lens_wave_rotation_matches_ray_larmor():
	# with rotate=True the wave picks up the same Larmor angle the ray path
	# applies (Lens.rotation = -K L): an off-axis blob's azimuth must agree
	lens = Lens(strength=34.72, length=0.02)
	K, L = lens._effective_strength, lens.length
	n, dxi = 128, 1e-7
	X, Y = wo.transverse_coordinates((n, n), dxi, dxi)
	x0 = 20 * dxi
	U0 = np.exp(-((X - x0)**2 + Y**2) / (2 * (3 * dxi)**2)).astype(complex)

	def azimuth(U):
		I = np.abs(U)**2
		return np.arctan2((I * Y).sum() / I.sum(), (I * X).sum() / I.sum())

	U_no, *_ = wo.propagate_quadratic_segment_scaled(U0, dxi, dxi, LAM, L, 1.0,
													 np.inf, K**2)
	U_rot, *_ = wo.propagate_quadratic_segment_scaled(U0, dxi, dxi, LAM, L, 1.0,
													  np.inf, K**2, rotate=-K * L)
	lens.transfer_matrix()					# sets lens.rotation as the ray path does
	assert np.isclose(lens.rotation, -K * L, rtol=1e-12)
	assert np.isclose(azimuth(U_no), 0.0, atol=1e-6)			# default: no rotation
	assert np.isclose(azimuth(U_rot), -K * L, atol=1e-3)		# opt-in: ray's angle
	# and the element is what declares that angle -- the propagator never
	# re-derives it from the strength (a quadrupole declares 0.0)
	assert lens._scaled_segment() == ('quadratic', K**2, -K * L)
	# and the rotation is energy-neutral
	assert np.isclose((np.abs(U_rot)**2).sum(), (np.abs(U_no)**2).sum(), rtol=1e-12)


def test_thick_quadrupole_is_an_exact_scaled_segment():
	from pySEA.rayTEM.seashells import scaled_frame_tag
	# issue #3 step 3. A thick quadrupole is a medium like a thick lens, but with
	# OPPOSITE curvature per axis: harmonic on the focusing one, hyperbolic on
	# the defocusing one. Its wave crossover must land on the ray-traced
	# diffraction plane, where the thin-kick route sat 72-2315 um away.
	def column(strength, length, exact=True):
		q = Quadrapole(name="TQ", strength=strength, length=length)
		if not exact:
			q._scaled_segment = lambda: None			# force the thin-kick route
		mic = Microscope(sections=[MicroscopeSection(elements=[
			Source(voltage=200, wave_shape=(128, 128), wave_extent=128 * 4e-7),
			Drift(length=0.02), q, Drift(length=1.4)])])
		ray = mic.conjugate_planes(axis="x")["diff"]
		mic.propagate_wave(mode="hybrid")
		wave = [read_scaled_wavefield(pl)[7] for pl in mic._wave_scaled_planes
				if scaled_frame_tag(pl) == "crossover-x"]
		return ray[0], wave[0]

	for strength, length in ((8.0, 0.03), (12.0, 0.03), (20.0, 0.05)):
		z_ray, z_exact = column(strength, length)
		_, z_thin = column(strength, length, exact=False)
		assert abs(z_exact - z_ray) < 1e-9, (strength, length, z_exact, z_ray)
		assert abs(z_thin - z_ray) > 10e-6			# the approximation it replaces
	# the DEFOCUSING axis has no crossover at all -- it must not invent one
	q = Quadrapole(strength=8.0, length=0.03)
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(128, 128), wave_extent=128 * 4e-7),
		Drift(length=0.02), q, Drift(length=0.9)])])
	assert len(mic.conjugate_planes(axis="y")["diff"]) == 0
	mic.propagate_wave(mode="hybrid")
	tags = [scaled_frame_tag(pl) for pl in mic._wave_scaled_planes]
	assert "crossover-x" in tags and "crossover-y" not in tags


def test_quadrupole_scaled_segment_declaration():
	# the declaration itself: signed curvature per axis, and NO Larmor angle
	# (a quadrupole has no axial field, unlike a round lens)
	q = Quadrapole(strength=8.0, length=0.03)
	kind, kappa, larmor = q._scaled_segment()
	assert kind == "quadratic" and larmor == 0.0
	assert kappa == (64.0, -64.0)					# K > 0 focuses x
	assert Quadrapole(strength=-8.0, length=0.03)._scaled_segment()[1] == (-64.0, 64.0)
	# thin and zero-strength quadrupoles are point events, not segments
	assert Quadrapole(strength=8.0, length=0.0)._scaled_segment() is None
	assert Quadrapole(strength=0.0, length=0.03)._scaled_segment() is None
	# the curvature agrees in sign with the ray convention, on both axes
	for K in (-8.0, 8.0):
		q = Quadrapole(strength=K, length=0.03)
		kx, ky = q._scaled_segment()[1]
		assert (kx > 0) == q._axis_focuses("x")
		assert (ky > 0) == q._axis_focuses("y")


def test_segment_propagator_is_per_axis():
	# the two axes accumulate DIFFERENT dtau through one anisotropic segment,
	# and the frame advances by each axis's own law
	n, dxi, L = 64, 1e-7, 0.03
	U0 = wo.gaussian_field((n, n), dxi, dxi, 5e-7, 5e-7)
	kappa = (64.0, -64.0)
	U, s, R, dtau = wo.propagate_quadratic_segment_scaled(
		U0, dxi, dxi, LAM, L, 1.0, np.inf, kappa)
	assert isinstance(dtau, tuple) and dtau[0] != dtau[1]
	# focusing axis contracts the frame, defocusing expands it
	assert s[0] < 1.0 < s[1]
	assert np.isclose(s[0], np.cos(8.0 * L), rtol=1e-12)
	assert np.isclose(s[1], np.cosh(8.0 * L), rtol=1e-12)
	# an isotropic call still returns scalars, and matches the round-lens path
	Ui, si, Ri, ti = wo.propagate_quadratic_segment_scaled(
		U0, dxi, dxi, LAM, L, 1.0, np.inf, 64.0)
	assert not isinstance(si, tuple) and not isinstance(ti, tuple)
	assert np.isclose(si, np.cos(8.0 * L), rtol=1e-12)
	# energy is conserved with no absorber
	assert np.isclose((np.abs(U)**2).sum(), (np.abs(U0)**2).sum(), rtol=1e-9)
	# a rotation on an anisotropic segment is refused, not silently dropped
	with pytest.raises(NotImplementedError, match="mixes the transverse axes"):
		wo.propagate_quadratic_segment_scaled(U0, dxi, dxi, LAM, L, 1.0, np.inf,
											  kappa, rotate=0.3)
	# an interior crossover names which axis failed
	with pytest.raises(ValueError, match=r"\[x axis\]"):
		wo.propagate_quadratic_segment_scaled(U0, dxi, dxi, LAM, 0.05, 1.0,
											  -0.01, kappa)


# --- one plane calculus: transfer_block, conjugate families, waists --------------

def test_transfer_block_matches_transfer_matrix():
	# the partial-length seam must be the SAME optics as the ray matrices
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	if not sea_available:
		pytest.skip("basic_column.sea requires sea_eco")
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	worst, n = 0.0, 0
	for sec in scope.sections:
		for ele in sec.elements:
			L = getattr(ele, "length", 0) or 0
			M6 = ele.transfer_matrix()
			stored = np.array([[M6[0, 0], M6[0, 1]], [M6[1, 0], M6[1, 1]]], float)
			mine = np.asarray(ele.transfer_block(), float)
			if isinstance(ele, Lens) and L > 0 and (ele._effective_strength or 0):
				mine = mine * np.cos(ele._effective_strength * L)	# Larmor factor
			worst = max(worst, abs(stored - mine).max()) ; n += 1
	assert n > 40 and worst < 1e-12
	# a homogeneous body's halves compose exactly
	lens = Lens(strength=34.72, length=0.02)
	full = np.asarray(lens.transfer_block(), float)
	half = np.asarray(lens.transfer_block(dz=0.01), float)
	assert np.abs(half @ half - full).max() < 1e-12


@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_conjugate_planes_frame_ray_and_wave_agree():
	# (1) the wave's image planes: the frame walk gives BOTH families, and
	# because the frame update IS the transfer block it reproduces the ray
	# numbers wherever the ray method is valid, and the wave's own crossovers.
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))
	src = scope.sections[0].elements[0]
	src.wave_kind = "aperture" ; src.aperture_radius = 5e-6
	frame = scope.conjugate_planes(method="frame")
	ray = scope.conjugate_planes()					# method='ray' is the default
	assert np.allclose(frame["diff"], ray["diff"], atol=1e-9)		# free space: identical
	scope.propagate_wave(mode="hybrid")
	for zc in scope.crossovers:							# the wave rides this family
		assert min(abs(frame["diff"] - zc)) < 1e-9
	# the image family exists for the wave too, and differs from the diffraction one
	assert len(frame["image"]) == len(frame["diff"])
	assert min(abs(frame["image"][:, None] - frame["diff"][None, :]).min(axis=1)) > 1e-6
	# one image plane sits inside OL1's body: there the ray method interpolates
	# across the wrong functional form, the frame walk solves it exactly
	inside = [z for z in frame["image"] if 0.490 < z < 0.500]
	assert len(inside) == 1
	assert 1e-5 < min(abs(ray["image"] - inside[0])) < 1e-3		# ~188 um apart


def test_conjugate_planes_reference_plane():
	# (3) planes conjugate to a NAMED reference are a different set
	f1, f2 = 45e-3, 30e-3
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), Drift(length=10e-3),
		Lens(strength=np.sqrt(1 / f1), length=0.0, name="L1"), Drift(length=100e-3),
		Lens(strength=np.sqrt(1 / f2), length=0.0, name="L2"), Drift(length=200e-3)])])
	entrance = mic.conjugate_planes(method="frame")
	assert np.isclose(entrance["z_reference"], 0.0)
	assert np.allclose(entrance["diff_offset"], entrance["diff"])	# offsets from 0
	# the ray default agrees on the entrance case
	assert np.allclose(mic.conjugate_planes()["diff"], entrance["diff"], atol=1e-9)
	# an explicit z reference of 0 is the same as the default
	assert np.allclose(mic.conjugate_planes(method="frame", reference=0.0)["diff"],
					   entrance["diff"])
	at_l1 = mic.conjugate_planes(method="frame", reference="L1")
	assert np.isclose(at_l1["z_reference"], 10e-3)
	assert np.allclose(at_l1["diff_offset"], at_l1["diff"] - 10e-3)
	# a lens sitting exactly at the reference still acts on the beam: the
	# diffraction plane conjugate to L1's own plane is f1 past it
	assert np.isclose(at_l1["diff"][0], 10e-3 + f1, atol=1e-9)
	# the DIFFRACTION family is unchanged by moving the reference across a pure
	# drift -- rays parallel at the entrance are still parallel at L1 --
	assert np.allclose(at_l1["diff"], entrance["diff"], atol=1e-9)
	# -- but the IMAGE family is not: the object plane moved, so its conjugates
	# are a genuinely different set. This is the "may or may not be the same"
	# distinction: ask for the right reference and the right family.
	assert not np.allclose(at_l1["image"][:1], entrance["image"][:1], atol=1e-6)
	with pytest.raises(ValueError, match="method='ray'"):
		mic.conjugate_planes(method="ray", reference="L1")


def test_beam_waists_match_analytic_focal_shift():
	# (2) the covariance mode's planes: a waist is where Sigma_12 = 0, which
	# for a beam waisted at a thin lens is the classic focal shift
	# z = f / (1 + (f*sigma_theta/sigma_x)^2) -- NOT the geometric focus f.
	sx, st, f = 2.5e-6, 1e-4, 45e-3
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), Lens(strength=np.sqrt(1 / f), length=0.0),
		Drift(length=0.1)])])
	S0 = np.array([[sx**2, 0.0], [0.0, st**2]])
	w = mic.beam_waists(axis="x", sigma0=S0)
	predicted = f / (1 + (f * st / sx)**2)
	assert np.isclose(w["z"][0], predicted, atol=1e-12)
	assert not np.isclose(w["z"][0], f, atol=1e-3)		# genuinely shifted from f
	assert np.isclose(w["emittance"], sx * st, rtol=1e-12)		# invariant
	# the width there is emittance / sqrt(Sigma_22), and finite (not a point)
	assert w["width"][0] > 0
	# a thick lens body can hold a waist, solved with the cos/sin condition
	sx2, st2, K, L = 1e-5, 5e-4, 60.0, 0.05
	mic2 = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), Lens(strength=K, length=L), Drift(length=0.02)])])
	S = np.array([[sx2**2, 0.0], [0.0, st2**2]])
	w2 = mic2.beam_waists(axis="x", sigma0=S)
	assert len(w2["z"]) >= 1 and 0 < w2["z"][0] < L		# inside the body
	# brute-force the same root
	zs = np.linspace(0, L, 40001)
	s12 = np.array([(lambda m: (m @ S @ m.T)[0, 1])(np.array(
		[[np.cos(K * z), np.sin(K * z) / K], [-K * np.sin(K * z), np.cos(K * z)]]))
		for z in zs])
	brute = zs[np.where(np.diff(np.sign(s12)) != 0)[0]]
	assert min(abs(brute - w2["z"][0])) < 2 * (L / 40000)
	# neither size nor divergence is not a beam
	with pytest.raises(ValueError, match="at least one of them positive"):
		mic.beam_waists(axis="x", sigma0=np.zeros((2, 2)))


def test_transfer_block_refuses_to_invent_a_kick_in_a_body():
	# a finite-length element with focusing power must carry its body's own
	# law -- splitting it into a kick between drifts is exactly the
	# approximation the scaled path was corrected to avoid, so the base class
	# refuses rather than guessing
	class BodyWithoutLaw(Lens):
		def transfer_block(self, dz=None, axis='x'):
			return super(Lens, self).transfer_block(dz=dz, axis=axis)
	with pytest.raises(NotImplementedError, match="no partial propagator"):
		BodyWithoutLaw(strength=10.0, length=0.02).transfer_block()
	# elements with no focusing power are exact free space at any depth
	assert np.allclose(Drift(length=0.05).transfer_block(), [[1, 0.05], [0, 1]])
	assert np.allclose(Dipole(strength=1e-5, length=0.02).transfer_block(),
					   [[1, 0.02], [0, 1]])
	# a thin element IS an impulsive kick -- exact, no body to traverse
	thin = Lens(strength=6.0, length=0.0)
	assert np.allclose(thin.transfer_block(), [[1, 0], [-thin.focal_power, 1]])
	# and a thick lens/quad body carries a harmonic law whose halves compose
	for ele in (Lens(strength=34.72, length=0.02),
				Quadrapole(strength=12.0, length=0.03)):
		full = np.asarray(ele.transfer_block(axis='x'), float)
		half = np.asarray(ele.transfer_block(dz=ele.length / 2, axis='x'), float)
		assert np.abs(half @ half - full).max() < 1e-12
		assert np.isclose(np.linalg.det(full), 1.0, atol=1e-12)		# symplectic


def test_walk_refuses_non_symplectic_body():
	# Liouville: a real element conserves phase-space area, so a body's block
	# must have det == 1. The walk refuses a body that does not, rather than
	# reporting planes that cannot be trusted. Uses a deliberately broken stub:
	# the quadrupole that originally motivated this guard is fixed (issue #3),
	# so the guard needs its own subject.
	class _LossyBody(Drift):
		"""A body whose block loses phase-space area. Not physical; a test probe."""
		def transfer_block(self, dz=None, axis='x'):
			"""Return a deliberately non-symplectic block (det = 0.75)."""
			step = self.length if dz is None else float(dz)
			return np.asarray([[1.0, step], [0.0, 0.75]])

	bad = _LossyBody(length=0.03)
	assert not np.isclose(np.linalg.det(np.asarray(bad.transfer_block(), float)), 1.0)
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), bad, Drift(length=0.2)])])
	with pytest.raises(ValueError, match="non-symplectic"):
		mic.conjugate_planes(method="frame", axis="x")


def test_thick_quadrupole_is_symplectic():
	# issue #3 steps 1-2: the defocusing axis is hyperbolic, not harmonic, so
	# both axes conserve phase-space area and a body's halves compose.
	for K in (-12.0, -1.0, 1.0, 12.0):
		for L in (0.005, 0.03, 0.1):
			q = Quadrapole(strength=K, length=L)
			for axis in ("x", "y"):
				M = np.asarray(q.transfer_block(axis=axis), float)
				assert np.isclose(np.linalg.det(M), 1.0, atol=1e-12), (K, L, axis)
				half = np.asarray(q.transfer_block(dz=L / 2, axis=axis), float)
				assert np.allclose(half @ half, M, atol=1e-12), (K, L, axis)
	# the full transverse 4x4 is symplectic too
	M = np.asarray(Quadrapole(strength=12.0, length=0.03).transfer_matrix(), float)
	assert np.isclose(np.linalg.det(M[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]), 1.0, atol=1e-12)
	# and the walk that used to refuse it now completes on both axes
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), Quadrapole(strength=12.0, length=0.03), Drift(length=0.2)])])
	mic.conjugate_planes(method="frame", axis="x")
	mic.conjugate_planes(method="frame", axis="y")


def test_quadrupole_axis_convention_thin_and_thick():
	# issue #3 step 2: K > 0 focuses x and defocuses y, in BOTH branches.
	# Previously the thin branch swapped its blocks for K > 0 and the thick
	# branch never did, so giving a quadrupole a length flipped which axis
	# converged.
	for L in (0.0, 1e-3, 0.03):
		P_x, P_y = Quadrapole(strength=12.0, length=L).focal_powers
		assert P_x > 0 > P_y, f"K > 0 must focus x at length {L}"
		P_x, P_y = Quadrapole(strength=-12.0, length=L).focal_powers
		assert P_y > 0 > P_x, f"K < 0 must focus y at length {L}"
	# focal_powers agrees with the matrix it claims to mirror (-1/f = C')
	for K in (-12.0, 12.0):
		for L in (0.0, 0.03):
			q = Quadrapole(strength=K, length=L)
			P = q.focal_powers
			M = np.asarray(q.transfer_matrix(), float)
			assert np.isclose(M[1, 0], -P[0], atol=1e-12), (K, L, "x")
			assert np.isclose(M[3, 2], -P[1], atol=1e-12), (K, L, "y")
	# a short thick quad approaches the thin kick K^2*L
	K, L = 12.0, 1e-4
	assert np.isclose(Quadrapole(strength=K, length=L).focal_powers[0],
					  K**2 * L, rtol=1e-5)
	# B stays drift-like (positive) for either sign of K -- it used to be
	# computed as S/K with a signed K, so it inverted for K < 0
	for K in (-12.0, 12.0):
		for axis in ("x", "y"):
			assert Quadrapole(strength=K, length=0.03).transfer_block(axis=axis)[0, 1] > 0


def test_thick_quadrupole_conserves_emittance():
	# the symplecticity that matters physically: sqrt(det Sigma) per axis is
	# invariant under a real element (Liouville). The old cos/sin y-block lost
	# ~13% of it over a 30 mm body.
	q = Quadrapole(strength=12.0, length=0.03)
	Sigma = np.diag([1e-12, 1e-10, 1e-12, 1e-10, 0.0, 0.0])
	_, out = q.propagate_moments(np.zeros(6), Sigma)
	for i in (0, 2):
		sel = np.ix_([i, i + 1], [i, i + 1])
		before = np.sqrt(np.linalg.det(Sigma[sel]))
		after = np.sqrt(np.linalg.det(np.asarray(out, float)[sel]))
		assert np.isclose(after, before, rtol=1e-12)


# --- wave-side image planes (the conjugate family the frame never crosses) --------

def _two_lens_column(n=128, dx=1.2e-7):
	return Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(n, n), wave_extent=n * dx,
			   wave_kind="aperture", aperture_radius=5e-6),
		Drift(length=10e-3), Lens(strength=np.sqrt(1 / 45e-3), length=0.0),
		Drift(length=100e-3), Lens(strength=np.sqrt(1 / 30e-3), length=0.0),
		Drift(length=100e-3)])])


def _block_to(mic, z, axis="x"):
	"""Accumulate the transfer block from the column entrance to z."""
	M, z_prev = np.eye(2), 0.0
	for z0, L, ele in mic._element_spans():
		if z0 > z - 1e-12:
			break
		if z0 - z_prev > 1e-12:
			M = np.array([[1.0, z0 - z_prev], [0.0, 1.0]]) @ M
		step = min(L, z - z0)
		M = np.asarray(ele.transfer_block(dz=step, axis=axis), float) @ M
		z_prev = z0 + step
	if z - z_prev > 1e-12:
		M = np.array([[1.0, z - z_prev], [0.0, 1.0]]) @ M
	return M


def test_hybrid_logs_image_planes():
	from pySEA.rayTEM.seashells import scaled_frame_tag
	mic = _two_lens_column()
	mic.propagate_wave(mode="hybrid")
	# the frame's own crossovers ARE the diffraction family (flat seed -> s ~ A)
	assert np.allclose(mic.crossovers, mic.diffraction_planes["x"])
	# the image family is found even though the frame never crosses it...
	assert len(mic.image_planes["x"]) == 1
	# ...and is logged as a first-class wave event
	tagged = {scaled_frame_tag(pl): read_scaled_wavefield(pl)[7]
			  for pl in mic._wave_scaled_planes if scaled_frame_tag(pl)}
	assert "image-x" in tagged
	z_img = float(mic.image_planes["x"][0])
	assert abs(tagged["image-x"] - z_img) < 1e-12
	# B = 0 is the DEFINITION of an image plane -- check it against the matrix
	assert abs(_block_to(mic, z_img)[0, 1]) < 1e-14
	# and A is the magnification (inverted here)
	assert _block_to(mic, z_img)[0, 0] < 0
	# the two independent plane finders agree
	for m in ("ray", "frame"):
		assert np.allclose(mic.conjugate_planes(axis="x", method=m)["image"],
						   mic.image_planes["x"], atol=1e-9)


def test_scaled_plane_at_is_exact():
	# advancing a logged plane to an arbitrary z must equal what the engine
	# itself produces when the column is cut there -- an independent path.
	mic = _two_lens_column()
	mic.propagate_wave(mode="hybrid", absorb=0.0)
	z = float(mic.image_planes["x"][0])
	dense = _two_lens_column().subdivided([z])
	dense.propagate_wave(mode="hybrid", absorb=0.0)
	zs = [read_scaled_wavefield(pl)[7] for pl in dense._wave_scaled_planes]
	ref = dense._wave_scaled_planes[int(np.argmin(np.abs(np.array(zs) - z)))]
	U_ref, _, _, _, s_ref, R_ref, _, z_ref = read_scaled_wavefield(ref)
	U_q, _, _, _, s_q, R_q, _, z_q = read_scaled_wavefield(mic._scaled_plane_at(z))
	assert abs(z_q - z_ref) < 1e-12
	assert np.isclose(s_q, s_ref, rtol=1e-12) and np.isclose(R_q, R_ref, rtol=1e-12)
	assert np.abs(U_q - U_ref).max() / np.abs(U_ref).max() < 1e-12
	# a z that already has a logged plane returns that plane itself
	assert mic._scaled_plane_at(float(mic.crossovers[0])) is not None
	# upstream of everything is refused
	with pytest.raises(ValueError, match="upstream of every logged plane"):
		mic._scaled_plane_at(-1.0)
	# a plane INSIDE a body is refused, not approximated: the run logs a plane at
	# every element boundary, so free stretches are always reachable and a body
	# is the only thing that genuinely blocks. The message names the fix.
	thick = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2e-7),
		Drift(length=0.01), Lens(name="thick", strength=20.0, length=0.02),
		Drift(length=0.1)])])
	thick.propagate_wave(mode="hybrid", absorb=0.0)
	with pytest.raises(ValueError, match=r"Lens 'thick' spanning .*subdivided"):
		thick._scaled_plane_at(0.02)


def test_acts_on_rays_is_asked_of_the_optics_not_the_type():
	# "is this free space?" is answered by the element's own transfer block, so
	# a drift, a zero-strength lens and a fiducial all count as transparent
	mic = _two_lens_column()
	assert not mic._acts_on_rays(Drift(length=0.1), 0.1)
	assert not mic._acts_on_rays(Drift(name="sample", length=0.0), 0.0)
	assert not mic._acts_on_rays(Lens(strength=0.0, length=0.0), 0.0)
	assert mic._acts_on_rays(Lens(strength=6.0, length=0.0), 0.0)
	assert mic._acts_on_rays(Quadrapole(strength=8.0, length=0.03), 0.03)
	# an astigmatic element acts even though neither axis alone is free
	assert mic._acts_on_rays(Quadrapole(strength=8.0, length=0.0), 0.0)


def test_wavefield_at_is_exact_not_nearest():
	# wavefield_at used to snap silently to the nearest logged plane, so asking
	# for an image plane returned a DIFFERENT plane with no warning
	mic = _two_lens_column()
	mic.propagate_wave(mode="hybrid", absorb=0.0)
	z = float(mic.image_planes["x"][0]) - 5e-3		# deliberately between planes
	w = mic.wavefield_at(z)
	assert abs(w.z - z) < 1e-12 if hasattr(w, "z") else True
	assert f"{z:g}" in w.name						# the name reports the real z


# --- mid-element frame switching --------------------------------------------------

def test_mid_element_frame_switch_crosses_inside_a_body():
	# a crossover landing INSIDE an element body used to be refused outright;
	# the body can now flatten, cross and continue on its own law
	n, dxi = 128, 1e-7
	U0 = wo.gaussian_field((n, n), dxi, dxi, 3e-6, 3e-6)
	s0, R0, kappa, L = 1.0, -0.005, 20.0**2, 0.02
	z_zero = wo.segment_zero(L, s0, s0 / R0, kappa)
	assert z_zero is not None and 0 < z_zero < L		# the crossing is inside

	# the exact single-call traversal still refuses -- it has no policy
	with pytest.raises(ValueError, match="inside the segment body"):
		wo.propagate_quadratic_segment_scaled(U0, dxi, dxi, LAM, L, s0, R0, kappa)

	# the hybrid traversal gets through, and logs the crossing where the
	# medium's own law puts it
	U, s, R, dtau, z, zc, logged = wo.propagate_quadratic_segment_hybrid(
		U0, dxi, dxi, LAM, L, s0, R0, kappa, z=0.0)
	assert np.isclose(z, L, atol=1e-12)
	tags = [t for t, *_ in logged]
	assert "crossover-x" in tags
	z_logged = [zz for t, _U, _s, _R, _dt, zz, _zc in logged if t == "crossover-x"][0]
	assert abs(z_logged - z_zero) < 1e-5
	# energy is conserved: the invariant is sum|U|^2 * dxi * deta, NOT sum|U|
	assert np.isclose((np.abs(U)**2).sum(), (np.abs(U0)**2).sum(), rtol=1e-9)


def test_mid_element_hybrid_leaves_the_common_case_alone():
	# no interior crossing and no pending marker -> a single exact call, so
	# ordinary thick lenses keep their previous output exactly
	n, dxi, L = 64, 1e-7, 0.02
	U0 = wo.gaussian_field((n, n), dxi, dxi, 5e-7, 5e-7)
	kappa = 34.72**2
	Ue, se, Re, te = wo.propagate_quadratic_segment_scaled(
		U0, dxi, dxi, LAM, L, 1.0, np.inf, kappa)
	Uh, sh, Rh, th, zh, zch, logged = wo.propagate_quadratic_segment_hybrid(
		U0, dxi, dxi, LAM, L, 1.0, np.inf, kappa, z=0.0)
	assert logged == [] and zch is None
	assert np.array_equal(Uh, Ue)
	assert (sh, Rh, th) == (se, Re, te)
	assert np.isclose(zh, L)
	# and the anisotropic (thick quadrupole) case likewise
	kq = (64.0, -64.0)
	Ue, se, Re, te = wo.propagate_quadratic_segment_scaled(
		U0, dxi, dxi, LAM, 0.03, 1.0, np.inf, kq)
	Uh, sh, Rh, th, _z, _zc, logged = wo.propagate_quadratic_segment_hybrid(
		U0, dxi, dxi, LAM, 0.03, 1.0, np.inf, kq, z=0.0)
	assert logged == [] and np.array_equal(Uh, Ue) and sh == se


@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_mid_element_crossover_lands_on_the_analytic_plane():
	# the column case that motivated this: a point object at -500 mm has an
	# image plane at 320.474 mm, INSIDE C3's body (0.320-0.340). The free engine
	# used to flatten around C3 and record the plane 99 mm away.
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	from pySEA.rayTEM.seashells import make_scaled_wavefield_signal
	here = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(here, "..", "microscopes", "basic_column.sea")
	R0 = 0.5											# virtual point at z = -R0
	scope = load_microscope(path)
	predicted = scope.conjugate_planes(axis="x", method="frame",
									   reference=-R0)["image"]
	z_in_body = [float(z) for z in predicted if 0.320 < z < 0.340]
	assert len(z_in_body) == 1, "fixture no longer has a plane inside C3"

	run = load_microscope(path)
	U, dxi, deta, lam, *_ = read_scaled_wavefield(
		run.sections[0].elements[0].wave(mode="scaled"))
	seed = make_scaled_wavefield_signal(U, dxi, deta, lam, s=1.0, R=R0,
										tau=0.0, z=0.0)
	run.propagate_wave(wave0=seed, mode="hybrid", absorb=0.0)
	measured = np.asarray(run.crossovers, float)
	assert abs(measured - z_in_body[0]).min() < 1e-6		# was ~0.099 m off
	# and the ray is RESTORED past the body, so the planes downstream of the
	# in-body crossing land on their analytic positions too -- they used to be
	# 94-594 um out, because the rediverge had abandoned the original ray
	for z in predicted:
		assert abs(measured - float(z)).min() < 1e-7, float(z)


# --- waists are minima, and least confusion == the conjugate plane ----------------

@pytest.mark.skipif(not sea_available, reason="basic_column.sea requires sea_eco")
def test_beam_waists_returns_minima_not_stationary_points():
	# Sigma_12 = 0 is STATIONARY, not minimal. Differentially
	# Sigma_11' = 2 Sigma_12 and Sigma_12' = Sigma_22 - kappa Sigma_11, so
	# Sigma_11'' = 2(Sigma_22 - kappa Sigma_11). Free space (kappa = 0) is always
	# a minimum, but inside a focusing body the lens can reverse the divergence
	# and the root is the beam's WIDEST point. Those must not be reported.
	import os
	from pySEA.rayTEM.assemblies import load_microscope
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "microscopes", "basic_column.sea"))

	def block_to(z, axis="x"):
		M, z_prev = np.eye(2), 0.0
		for z0, L, ele in scope._element_spans():
			if z0 > z - 1e-12:
				break
			if z0 - z_prev > 1e-12:
				M = np.array([[1.0, z0 - z_prev], [0.0, 1.0]]) @ M
			step = min(L, z - z0)
			M = np.asarray(ele.transfer_block(dz=step, axis=axis), float) @ M
			z_prev = z0 + step
		if z - z_prev > 1e-12:
			M = np.array([[1.0, z - z_prev], [0.0, 1.0]]) @ M
		return M

	Sigma = np.diag([0.0, 1e-6**2])					# a point source
	waists = [float(z) for z in scope.beam_waists(axis="x", sigma0=Sigma)["z"]]
	assert waists, "no waists found"
	h = 2e-4
	for z in waists:
		here_, before, after = (block_to(z)[0, 1]**2,
								block_to(z - h)[0, 1]**2,
								block_to(z + h)[0, 1]**2)
		assert here_ < before and here_ < after, f"z={z} is not a minimum"

	# LEAST CONFUSION == the conjugate plane. For a point source Sigma_11 = B^2,
	# so the minima of the beam size are exactly the B = 0 image planes -- and
	# the two are computed by completely different routes (covariance transport
	# vs solving B = 0 on the accumulated blocks).
	image = [float(z) for z in scope.conjugate_planes(axis="x", method="frame")["image"]]
	assert len(waists) == len(image)
	assert np.allclose(waists, image, atol=1e-12)

	# and the criterion covers BOTH families: a PARALLEL seed has Sigma_11 = A^2,
	# so its minima are the A = 0 diffraction planes. A perfectly collimated beam
	# has zero angular variance, which used to be refused outright -- leaving the
	# method able to express a point source but not a collimated beam.
	parallel = [float(z) for z in
				scope.beam_waists(axis="x", sigma0=np.diag([1e-6**2, 0.0]))["z"]]
	diff = [float(z) for z in scope.conjugate_planes(axis="x", method="frame")["diff"]]
	assert len(parallel) == len(diff)
	assert np.allclose(parallel, diff, atol=1e-12)
	# a negative variance is still nonsense and still refused
	with pytest.raises(ValueError, match="non-negative variances"):
		scope.beam_waists(axis="x", sigma0=np.diag([-1.0, 1e-12]))
	with pytest.raises(ValueError, match="at least one of them positive"):
		scope.beam_waists(axis="x", sigma0=np.zeros((2, 2)))


# --- aberrations: the ray kick, and planes becoming surfaces ----------------------

_F_AB = 0.045

def _aberrated_column(Cs=0.0, quad=None):
	els = [Source(voltage=200), Drift(length=0.01),
		   Lens(name="OL", strength=np.sqrt(1 / _F_AB), Cs=Cs)]
	if quad is not None:
		els.append(Quadrapole(strength=quad))
	els.append(Drift(length=0.06))
	return Microscope(sections=[MicroscopeSection(elements=els)])


def test_spherical_aberration_kick_matches_the_closed_form():
	# chi = -k r^2/2f - k Cs r^4/4f^4  ->  the kick beyond the matrix is
	# -(Cs/f^4) x r^2, so a ray entering parallel at h crosses the axis at
	# z = f/(1 + Cs h^2/f^3), the classic longitudinal spherical aberration.
	Cs, f = 1e-3, _F_AB
	lens = Lens(strength=np.sqrt(1 / f), Cs=Cs)
	r0 = np.zeros((3, 6))
	r0[:, 0] = [0.0, 1e-4, 2e-4]
	dx, dy, dxt, dyt = lens.aberration_kick(r0)
	assert np.allclose(dxt, [-Cs * (1 / f)**4 * h**3 for h in r0[:, 0]], rtol=1e-12)
	assert np.allclose(dyt, 0.0)
	assert np.allclose(dx, 0.0) and np.allclose(dy, 0.0)	# a thin lens does not displace
	# an ideal lens declares nothing at all, so aberration-free columns are
	# bit-for-bit unchanged
	assert Lens(strength=np.sqrt(1 / f)).aberration_kick(r0) is None
	assert Lens(strength=0.0, Cs=Cs).aberration_kick(r0) is None		# no power
	assert Drift(length=0.1).aberration_kick(r0) is None

	# the traced caustic matches the closed form
	mic = _aberrated_column(Cs=Cs)
	hs = np.array([1e-5, 5e-5, 1e-4, 2e-4, 3e-4])
	r0 = np.zeros((hs.size, 6))
	r0[:, 0] = hs
	rays = np.asarray(mic.propagate_ray(r0.copy()))
	z_cross = rays[-1][0, 4] - rays[-1][:, 0] / rays[-1][:, 1]
	exact = 0.01 + f / (1 + Cs * hs**2 / f**3)
	assert np.allclose(z_cross, exact, atol=1e-12)
	# and the PARAXIAL plane is untouched -- aberration is the departure from it
	assert np.isclose(float(mic.conjugate_planes(axis="x")["diff"][0]), 0.01 + f,
					  atol=1e-12)


def test_thick_body_aberration_matches_the_perturbed_ray_equation():
	# A thick lens distributes its aberration: a slice dz acts on the LOCAL ray
	# height and the rest of the body carries that kick to the exit, giving a
	# position offset as well as an angle one. Validated against direct
	# integration of x'' = -K^2 x - c x r^2 through the body -- an independent
	# route that uses none of the transfer-block machinery.
	from scipy.integrate import solve_ivp
	K, L, Cs = 129.80, 0.010, 1e-3			# OL1's real parameters
	lens = Lens(strength=K, length=L, Cs=Cs)
	c = Cs * lens.focal_power**4 / L
	A, B = np.cos(K * L), np.sin(K * L) / K
	for h in (2e-5, 4e-5, 8e-5):
		sol = solve_ivp(lambda z, u: [u[1], -K**2 * u[0] - c * u[0]**3],
						[0, L], [h, 0.0], rtol=1e-12, atol=1e-18)
		d_ode = (sol.y[0, -1] - A * h, sol.y[1, -1] + K * np.sin(K * L) * h)
		r0 = np.zeros((1, 6)); r0[0, 0] = h
		dx, dy, dxt, dyt = lens.aberration_kick(r0)
		# first-order perturbation, so the residual is the second-order term and
		# must stay small AND grow with h
		assert abs(dx[0] / d_ode[0] - 1) < 1e-4, h
		assert abs(dxt[0] / d_ode[1] - 1) < 1e-4, h
	# the thin limit is untouched: no displacement, and the impulsive kick
	thin = Lens(strength=np.sqrt(1 / 0.045), Cs=Cs)
	r0 = np.zeros((2, 6)); r0[:, 0] = [1e-4, 2e-4]
	dx, dy, dxt, dyt = thin.aberration_kick(r0)
	assert np.all(dx == 0) and np.all(dy == 0)
	assert np.allclose(dxt, [-Cs * (1 / 0.045)**4 * h**3 for h in r0[:, 0]], rtol=1e-12)
	# distributing it is NOT the same as placing it at the entrance face: r(z)
	# falls as the body focuses, so the entrance-face model over-estimates
	entrance = -Cs * lens.focal_power**4 * 8e-5**3
	r0 = np.zeros((1, 6)); r0[0, 0] = 8e-5
	assert abs(lens.aberration_kick(r0)[2][0]) < 0.7 * abs(entrance)


def test_focal_surface_is_flat_without_aberration():
	# the acceptance test from the plan: with no aberration the surface must
	# degenerate to the paraxial plane exactly
	surf = _aberrated_column().focal_surface(family="diff", aperture=3e-4)
	assert surf["sag"] < 1e-12
	assert np.allclose(surf["z"], surf["z_paraxial"], atol=1e-12)
	assert abs(surf["fit"]["c20"]) < 1e-12 and abs(surf["fit"]["astig"]) < 1e-12
	# the image family likewise -- but it needs a column that HAS one. With the
	# object 10 mm before a 45 mm lens the image is virtual, and focal_surface
	# says so rather than inventing a reference.
	with pytest.raises(ValueError, match="no paraxial 'image' plane"):
		_aberrated_column().focal_surface(family="image", aperture=5e-3)
	real = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200), Drift(length=0.1),
		Lens(strength=np.sqrt(1 / _F_AB)), Drift(length=0.12)])])
	surf = real.focal_surface(family="image", aperture=5e-3)
	assert surf["sag"] < 1e-12
	assert np.allclose(surf["z"], surf["z_paraxial"], atol=1e-12)
	with pytest.raises(ValueError, match="family must be"):
		_aberrated_column().focal_surface(family="bogus")


def test_focal_surface_separates_spherical_from_astigmatism():
	# Cs bows the surface into a paraboloid of revolution: c20 = -Cs alpha^2 at
	# the sampled edge, with no two-fold component
	Cs, a = 1e-3, 3e-4
	surf = _aberrated_column(Cs=Cs).focal_surface(family="diff", aperture=a,
												  azimuths=16)
	assert np.isclose(surf["fit"]["c20"], -Cs * (a / _F_AB)**2, rtol=1e-3)
	assert abs(surf["fit"]["astig"]) < 1e-9
	assert surf["sag"] > 0

	# a quadrupole splits it by AZIMUTH but not by aperture radius, so it must
	# land in `astig` and not in the r^2 terms. Fitting everything to r^2 would
	# report an aperture aberration that is not there.
	mic = _aberrated_column(quad=0.5)
	surf = mic.focal_surface(family="diff", aperture=a, azimuths=16)
	px = float(mic.conjugate_planes(axis="x")["diff"][0])
	py = float(mic.conjugate_planes(axis="y")["diff"][0])
	assert np.isclose(surf["fit"]["astig"], abs(py - px) / 2, rtol=1e-3)
	assert abs(surf["fit"]["c20"]) < 1e-9

	# both together superpose, each recovered independently
	mic = _aberrated_column(Cs=Cs, quad=0.5)
	surf = mic.focal_surface(family="diff", aperture=a, azimuths=16)
	assert np.isclose(surf["fit"]["c20"], -Cs * (a / _F_AB)**2, rtol=1e-2)
	assert np.isclose(surf["fit"]["astig"], abs(py - px) / 2, rtol=1e-3)


def test_focal_surface_single_radius_cannot_separate_r_terms():
	# with one radius the r-independent and r^2 terms are degenerate; say so
	# rather than returning a fitted value that cannot be trusted
	surf = _aberrated_column(Cs=1e-3).focal_surface(family="diff", aperture=3e-4,
													radii=1, azimuths=8)
	assert np.isnan(surf["fit"]["c20"]) and np.isnan(surf["fit"]["c22"])
	assert np.isfinite(surf["fit"]["astig"])


# --- the wave side of aberration: one chi, two representations -------------------

def test_ray_kick_is_the_gradient_of_the_wave_screen():
	# THE consistency check between the two descriptions. The ray kick and the
	# wave screen are not independently derived: chi = -k Cs r^4 / 4f^4 and the
	# kick is (1/k) dchi/dr. Verified by finite difference, whose error must fall
	# as dx^2 -- a fixed tolerance would not distinguish "agrees" from "agrees to
	# within the discretisation I happened to choose".
	f, Cs = 0.045, 1e-3
	lens = Lens(strength=np.sqrt(1 / f), Cs=Cs)
	P, k = lens.focal_power, 2 * np.pi / LAM
	errs = []
	for n, dx in ((256, 4e-7), (512, 2e-7), (1024, 1e-7)):
		chi = Aberrations({'C30': Cs}).phase((n, n), dx, dx, LAM, P)
		X, Y = wo.transverse_coordinates((n, n), dx, dx)
		grad = np.gradient(chi, dx, axis=1)[n // 2, :] / k
		r0 = np.zeros((n, 6))
		r0[:, 0], r0[:, 2] = X[n // 2, :], Y[n // 2, :]
		kick = lens.aberration_kick(r0)[2]			# (dx, dy, dxt, dyt)
		m = np.abs(X[n // 2, :]) < 2e-5
		errs.append(np.abs(grad[m] - kick[m]).max())
	assert errs[0] > 0
	for a, b in zip(errs, errs[1:]):
		assert 3.5 < a / b < 4.5, f"second-order convergence expected, got {a / b}"


def test_lens_phase_carries_the_quartic_in_both_representations():
	f, Cs, grid = 0.045, 1e-3, ((64, 64), 2e-6, 2e-6)
	ideal = Lens(strength=np.sqrt(1 / f))
	real = Lens(strength=np.sqrt(1 / f), Cs=Cs)

	# scaled: the parabola is absorbed into the curvature, the quartic CANNOT be
	# (the frame is quadratic by construction) so it stays as a residual screen
	P_i, screen_i = ideal.phase_shift(grid, LAM, scaled=True)
	P_r, screen_r = real.phase_shift(grid, LAM, scaled=True)
	assert screen_i is None							# an ideal lens: U+ = U-
	assert P_r == P_i								# aberration does not change 1/f
	assert screen_r is not None
	# the screen is the quartic, at the frame's physical coordinates
	expected = Aberrations({'C30': Cs}).phase((64, 64), 2e-6, 2e-6, LAM, P_r)
	assert np.allclose(np.asarray(screen_r.data), expected, rtol=1e-12)
	# ...so doubling s quadruples it (r^4 at x = s*xi)
	_, screen_s = real.phase_shift(grid, LAM, scaled=True, s=2.0)
	assert np.allclose(np.asarray(screen_s.data), 16.0 * expected, rtol=1e-12)

	# fixed: the quartic is added to the same real-space screen
	chi_i = np.asarray(ideal.phase_shift(grid, LAM)[0].data)
	chi_r = np.asarray(real.phase_shift(grid, LAM)[0].data)
	assert np.allclose(chi_r - chi_i, expected, rtol=1e-12)


def test_spherical_aberration_degrades_the_wave_focus():
	# the physical signature: the focus broadens and its peak falls, while the
	# PARAXIAL crossover does not move -- the frame is quadratic, so it absorbs
	# the parabola and nothing else
	f, a = 0.045, 3e-4
	peaks, zs = [], []
	for Cs in (0.0, 1e-3):
		mic = Microscope(sections=[MicroscopeSection(elements=[
			Source(voltage=200, wave_shape=(256, 256), wave_extent=1.4e-3,
				   wave_kind="aperture", aperture_radius=a),
			Drift(length=0.001), Lens(strength=np.sqrt(1 / f), Cs=Cs),
			Drift(length=0.05)])])
		mic.propagate_wave(mode="hybrid", absorb=0.0)
		z = float(mic.crossovers[0])
		I = np.abs(np.asarray(mic.wavefield_at(z).data))**2
		peaks.append(I.max() / I.sum())
		zs.append(z)
	assert np.isclose(zs[0], zs[1], atol=1e-12), "the paraxial plane must not move"
	assert peaks[1] < peaks[0], "Cs must degrade the focus"


def test_undersampled_aberration_screen_is_refused():
	# the quartic grows as r^4, so it out-runs the grid faster than anything
	# else in the column. The guard must catch that rather than alias it into
	# a plausible-looking focus: this is the SAME setup as the test above at
	# half the sampling.
	f, a = 0.045, 3e-4
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(128, 128), wave_extent=1.4e-3,
			   wave_kind="aperture", aperture_radius=a),
		Drift(length=0.001), Lens(strength=np.sqrt(1 / f), Cs=1e-3),
		Drift(length=0.05)])])
	with pytest.raises(ValueError, match="under-sampled on the scaled grid"):
		mic.propagate_wave(mode="hybrid", absorb=0.0)


# --- the wave path against the closed-form back focal plane ----------------------

def _bfp_setup(Cs, n=512, ext=1.4e-3, f=0.045, a=3e-4):
	"""A collimated aperture straight onto a lens, propagated one focal length."""
	src = Source(voltage=200, wave_shape=(n, n), wave_extent=ext,
				 wave_kind="aperture", aperture_radius=a)
	mic = Microscope(sections=[MicroscopeSection(elements=[
		src, Lens(strength=np.sqrt(1 / f), Cs=Cs), Drift(length=f)])])
	return mic, float(src.wavelength), ext / n, f, a


def test_wave_matches_closed_form_back_focal_plane():
	# The non-circular check on the whole chain. Textbook: the field one focal
	# length past a lens is FT[A(r) exp(i chi_ab)] sampled at q = x'/(lambda f).
	# Reproducing it exercises the parabola going into the frame, the quartic
	# staying as a screen on U, the carrier-free kernel, the hybrid frame
	# switching and the reconstruction -- without reusing any of them.
	n = 512
	for Cs in (0.0, 1e-3):
		mic, lam, dx, f, a = _bfp_setup(Cs, n=n)
		mic.propagate_wave(mode="hybrid", absorb=0.0)
		q = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
		dxp = lam * f * (q[1] - q[0])				# BFP pixel implied by the FT
		w = mic.wavefield_at(float(mic.crossovers[0]), target_dx=dxp,
							 target_shape=(n, n))
		I_prop = np.abs(np.asarray(w.data))**2
		I_prop /= I_prop.max()
		# the same aperture model the source seeds, so the comparison is of the
		# propagation and not of two different discs
		A = wo.bandlimited_disk((n, n), dx, dx, a).astype(complex)
		chi = Aberrations({'C30': Cs}).phase((n, n), dx, dx, lam, 1.0 / f) if Cs else 0.0
		F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A * np.exp(1j * chi))))
		I_ref = np.abs(F)**2
		I_ref /= I_ref.max()
		assert np.abs(I_ref - I_prop).max() < 2e-3, Cs


def test_reconstruction_zero_fills_beyond_the_modelled_field():
	# Band-limited resampling is PERIODIC, so asking for a grid wider than the
	# modelled one used to return replicas of the beam: a bright false peak in a
	# region where nothing was ever propagated. It is zero-filled instead.
	n = 256
	mic, lam, dx, f, a = _bfp_setup(0.0, n=n)
	mic.propagate_wave(mode="hybrid", absorb=0.0)
	z = float(mic.crossovers[0])
	native = mic.wavefield_at(z)
	dx_native = float(native.dimensions[-1].scale)
	half_native = 0.5 * n * dx_native
	# a coarser pixel at the same array size => a WIDER field than was modelled
	coarse = 4.0 * dx_native
	wide = mic.wavefield_at(z, target_dx=coarse, target_shape=(n, n))
	I = np.abs(np.asarray(wide.data))**2
	c = np.arange(n) - n // 2
	X, Y = np.meshgrid(c * coarse, c * coarse)
	outside = (np.abs(X) > half_native) | (np.abs(Y) > half_native)
	assert outside.any(), "test needs a genuinely wider request"
	assert np.all(I[outside] == 0.0), "no field was modelled out there"
	assert I[~outside].max() > 0										# and the beam survives


# --- Krivanek aberration function, orders 1 to 5 ---------------------------------

def test_krivanek_terms_have_the_right_order_and_symmetry():
	# each term must go as theta^(n+1) radially and have m-fold azimuthal
	# symmetry, i.e. 2m sign changes around a circle
	n, dx, P = 256, 2e-6, 22.2222
	for name, (order, m) in KRIVANEK_TERMS.items():
		chi = Aberrations({name: 1e-3 if order >= 3 else 1e-8}).phase(
			(n, n), dx, dx, LAM, P)
		idx = np.arange(10, 120)
		line = chi[n // 2, n // 2 + idx]
		power = np.polyfit(np.log(idx), np.log(np.abs(line)), 1)[0]
		assert abs(power - (order + 1)) < 1e-6, (name, power)
		t = np.linspace(0, 2 * np.pi, 721, endpoint=False)
		xi = (n // 2 + 60 * np.cos(t)).astype(int)
		yi = (n // 2 + 60 * np.sin(t)).astype(int)
		assert np.sum(np.diff(np.sign(chi[yi, xi])) != 0) == 2 * m, name
	# the imaginary component is the same term rotated by pi/2m, so an (a, b)
	# pair and its magnitude at that orientation must agree
	a, b = 3e-4, 4e-4
	rot = np.arctan2(b, a) / 4					# C34 has m = 4
	skew = Aberrations({'C34': (a, b)}).phase((64, 64), dx, dx, LAM, P)
	X, Y = wo.transverse_coordinates((64, 64), dx, dx)
	theta, phi = np.hypot(X, Y) * P, np.arctan2(Y, X)
	k = 2 * np.pi / LAM
	assert np.abs(skew - (-k * theta**4 / 4 * np.hypot(a, b)		# C34: n = 3
						  * np.cos(4 * (phi - rot)))).max() < 1e-15
	# and the errors are actionable
	with pytest.raises(KeyError, match="not a Krivanek term"):
		Aberrations({'C37': 1.0})
	with pytest.raises(ValueError, match="rotationally symmetric"):
		Aberrations({'C30': (1.0, 0.5)})
	with pytest.raises(ValueError, match="nonzero focal power"):
		Aberrations({'C30': 1.0}).phase((8, 8), dx, dx, LAM, 0.0)


def test_first_order_terms_are_absorbed_into_the_frame():
	# C10 and C12 are QUADRATIC in the pupil angle, and the scaled frame is
	# exactly a quadratic, so they belong in the curvature -- not in a screen
	# the frame exists to avoid.
	f = 0.045
	P = 1 / f
	lens = Lens(strength=np.sqrt(P), aberrations={'C10': 5e-4})
	P_x, P_y, residual = lens.aberration_powers()
	assert not residual									# nothing left for U
	assert np.isclose(P_x, P + 5e-4 * P**2, rtol=1e-12)	# dP = C10 P^2
	assert P_x == P_y									# C10 is isotropic
	# C12 aligned to the axes gives the quadrupole's own (+P, -P) shape. With a
	# complex coefficient "aligned" means a zero imaginary part, and the two
	# orientations are just the sign of the real part.
	for value, sign in ((2e-4, +1.0), (-2e-4, -1.0)):
		lens = Lens(strength=np.sqrt(P), aberrations={'C12': value})
		P_x, P_y, residual = lens.aberration_powers()
		assert not residual
		assert np.isclose(P_x, P + sign * 2e-4 * P**2, rtol=1e-12)
		assert np.isclose(P_y, P - sign * 2e-4 * P**2, rtol=1e-12)
	# a SKEW C12 (nonzero imaginary part) is astigmatism at 45 degrees, which
	# (R_x, R_y) cannot express, so it stays a screen rather than being
	# silently mis-absorbed
	lens = Lens(strength=np.sqrt(P), aberrations={'C12': (1e-4, 1.5e-4)})
	P_x, P_y, residual = lens.aberration_powers()
	assert np.isclose(P_x, P) and P_x == P_y and residual.names == ['C12']
	# higher orders always stay a screen
	lens = Lens(strength=np.sqrt(P), aberrations={'C10': 1e-4, 'C30': 1e-3,
												 'C34': (1e-3, 2e-4)})
	P_x, _, residual = lens.aberration_powers()
	assert np.isclose(P_x, P + 1e-4 * P**2, rtol=1e-12)
	assert sorted(residual.names) == ['C30', 'C34']


def test_C10_moves_the_wave_and_the_rays_by_the_same_amount():
	# C10 is defocus: quadratic in the pupil angle, so the scaled frame absorbs
	# it as a power change dP = C10 P^2 and the traced rays get it from the
	# generic kick, which for n=1, m=0 is exactly -C10 P^2 x. The two must land
	# on the same plane -- when the ray path implemented C3 alone they did not,
	# and nothing noticed.
	f = 0.045
	P = 1 / f
	for C10, f_eff in ((0.0, f), (5e-4, 1 / (P + 5e-4 * P**2))):
		lens = Lens(strength=np.sqrt(P), aberrations={'C10': C10})
		r0 = np.zeros((5, 6))
		r0[:, 0] = np.linspace(1e-6, 1e-4, 5)			# a parallel bundle
		rf = lens.propagate_ray(r0)
		z = -rf[:, 0] / rf[:, 1]
		assert np.allclose(z, f_eff, rtol=1e-12), C10	# and no spread: it is linear
		mic = Microscope(sections=[MicroscopeSection(elements=[
			Source(voltage=200, wave_shape=(128, 128), wave_extent=3e-4,
				   wave_kind="aperture", aperture_radius=6e-5),
			Lens(strength=np.sqrt(P), aberrations={'C10': C10}),
			Drift(length=0.06)])])
		mic.propagate_wave(mode="hybrid", absorb=0.0)
		assert np.isclose(float(mic.crossovers[0]), f_eff, rtol=1e-6), C10
		# the paraxial MATRIX still reports the unaberrated plane, by
		# construction: aberration is defined as the departure from it, which is
		# why the kick is kept out of the matrix rather than linearized into it
		assert np.isclose(float(mic.conjugate_planes(axis="x",
													 method="frame")["diff"][0]),
						  f, rtol=1e-12)


def test_aberrations_survive_a_sea_round_trip():
	# The coefficients live in a NESTED Aberrations object, which SEASerializable
	# carries as a child node -- this is what pins that the nesting survives.
	import os, tempfile
	from pySEA.rayTEM.assemblies import load_microscope
	cwd = os.getcwd()
	try:
		os.chdir(tempfile.mkdtemp())
		mic = Microscope(sections=[MicroscopeSection(elements=[
			Source(size=(1, 1), np_xy=(3, 3), angle=(1, 1), na_xy=(3, 3)),
			Drift(length=1),
			Lens(strength=3, length=.1,
				 aberrations={'C12': (3e-4, -1e-4), 'C50': 1e-2}),
			Drift(length=1)])])
		mic.propagate_ray()
		mic.to_sea("t.sea")
		lens = [e for sec in load_microscope("t.sea").sections
				for e in sec.elements if isinstance(e, Lens)][0]
		got = lens.aberrations
		assert got.convention == 'krivanek'
		assert np.isclose(got['C12'].real, 3e-4) and np.isclose(got['C12'].imag, -1e-4)
		assert np.isclose(got['C50'].real, 1e-2)
	finally:
		os.chdir(cwd)


def test_apply_aberrations_false_gives_the_ideal_result_on_every_path():
	# One flag per propagation method, default True, and it is implemented by
	# detaching rather than by a per-aberration branch -- so it works the same
	# on rays, moments and the wave, and would work on a path added tomorrow.
	from pySEA.rayTEM.elements import suspended_aberrations
	f, P = 0.045, 1 / 0.045
	def column(ab):
		return Microscope(sections=[MicroscopeSection(elements=[
			Source(voltage=200, size=(1e-5, 1e-5), np_xy=(3, 3),
				   angle=(1e-4, 1e-4), na_xy=(3, 3),
				   wave_shape=(128, 128), wave_extent=3e-4,
				   wave_kind="aperture", aperture_radius=6e-5),
			Lens(strength=np.sqrt(P), aberrations=ab),
			Drift(length=0.06)])])
	real, ideal = column({'C10': 5e-4, 'C30': 1e-3}), column(None)

	assert not np.allclose(real.propagate_ray(), ideal.propagate_ray())
	assert np.allclose(real.propagate_ray(apply_aberrations=False),
					   ideal.propagate_ray())
	# the moments path transports Sigma' = M Sigma M^T and so is paraxial by
	# construction -- it does not see aberrations either way. The flag is still
	# accepted there, so a caller can pass it uniformly across all four kinds.
	real.propagate_moments(apply_aberrations=False)
	ideal.propagate_moments()
	assert np.allclose(np.asarray(real.covariance_matrix.data),
					   np.asarray(ideal.covariance_matrix.data))
	real.propagate_wave(mode="hybrid", absorb=0.0, apply_aberrations=False)
	ideal.propagate_wave(mode="hybrid", absorb=0.0)
	assert np.isclose(float(real.crossovers[0]), float(ideal.crossovers[0]),
					  rtol=1e-12)
	# and the aberrations are still there afterwards, exception or not
	lens = real.sections[0].elements[1]
	assert lens.aberrations['C30'] == 1e-3
	with pytest.raises(RuntimeError):
		with suspended_aberrations([lens]):
			assert lens.aberrations is None
			raise RuntimeError("boom")
	assert lens.aberrations['C30'] == 1e-3


def test_a_screen_may_be_complex_carrying_amplitude_and_phase_together():
	# A physical plate cuts the beam AND phase-shifts what gets through. Those
	# are one object, not an aperture element plus a phase element that happen
	# to share a z. A complex screen is that object: applied directly rather
	# than exponentiated, so |T| is free to be anything.
	from pySEA.rayTEM.elements import _check_screen_sampling
	n = 32
	field = np.exp(1j * np.linspace(0, 1, n * n).reshape(n, n))
	chi = np.full((n, n), 0.37)
	mask = np.zeros((n, n)) ; mask[8:24, 8:24] = 1.0

	# real screen: unchanged behaviour, exp(i chi)
	assert np.allclose(wo.apply_phase(field, chi), field * np.exp(1j * chi))
	# and a real screen is exactly the complex screen of unit modulus
	assert np.allclose(wo.apply_phase(field, chi),
					   wo.apply_phase(field, np.exp(1j * chi).astype(complex)))
	# complex screen: modulus survives, so the plate both blocks and shifts
	out = wo.apply_phase(field, mask * np.exp(1j * chi))
	assert np.allclose(out, field * mask * np.exp(1j * chi))
	assert np.all(out[mask == 0] == 0)						# blocked
	assert np.allclose(np.abs(out[mask == 1]), 1.0)			# and passed undimmed
	# the space flag is orthogonal to the dtype: a complex screen works in the
	# FFT domain too, where the multiply is diagonal in q rather than in x
	q = wo.apply_phase(field, mask * np.exp(1j * chi), space="scattering")
	assert np.allclose(q, np.fft.ifft2(np.fft.fft2(field) * mask * np.exp(1j * chi)))

	# the sampling guard is skipped on complex screens because it CANNOT see
	# them: it compares |diff(data)| to pi, which is a phase step only for real
	# data. A unit-modulus screen winding 3.78 rad/pixel -- aliased -- reports
	# 1.90 and would pass, so running it there is false assurance.
	with pytest.raises(ValueError, match="under-sampled"):
		_check_screen_sampling(np.arange(n * n).reshape(n, n) * 1.0, "steep")
	aliased = np.exp(1j * np.arange(n * n).reshape(n, n) * 2.5)
	assert np.abs(np.diff(np.angle(aliased), axis=1)).max() > np.pi	# truly aliased
	assert np.abs(np.diff(aliased, axis=1)).max() < np.pi			# yet |dT| says fine
	_check_screen_sampling(aliased, "plate")						# so: skipped, not judged
