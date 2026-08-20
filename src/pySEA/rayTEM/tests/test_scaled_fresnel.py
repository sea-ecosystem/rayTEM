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
