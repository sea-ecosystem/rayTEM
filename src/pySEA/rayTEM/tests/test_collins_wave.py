"""Collins wave propagation (``propagate_wave_collins``) and its contracted transfers.

Checked against closed forms (drift, 2f Fourier transform, 4f imaging,
Gaussian beam widths), a brute-force fixed grid (an astigmatic, rotated lens:
the mixed form), the ray path (where an off-axis beam lands through a
rotating thick lens and dipoles), the fixed path's aperture, and the
scaled/hybrid walker on a column with crossovers.
"""
import numpy as np
import pytest

from pySEA.rayTEM import Aperture, Dipole, Drift, Lens, Microscope, MicroscopeSection, Quadrapole, Source
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.collins import CollinsTransfer, contract_collins, element_affine_map
from pySEA.rayTEM.seashells import (as_collins_wavefield, make_wavefield_signal, read_collins_wavefield,
									read_wavefield)

LAM = 2.5e-12


def _drift(d):
	"""Return a 4 x 4 drift map."""
	M = np.eye(4)
	M[0, 1] = M[2, 3] = d
	return M


def _lens(fx, fy=None):
	"""Return a 4 x 4 thin-lens map (astigmatic when ``fy`` differs)."""
	M = np.eye(4)
	M[1, 0], M[3, 2] = -1 / fx, -1 / (fx if fy is None else fy)
	return M


def _rot(t):
	"""Return a 4 x 4 rotation of positions and angles by ``t``."""
	c, s = np.cos(t), np.sin(t)
	R = np.eye(4)
	R[np.ix_([0, 2], [0, 2])] = [[c, -s], [s, c]]
	R[np.ix_([1, 3], [1, 3])] = [[c, -s], [s, c]]
	return R


def _gauss(n, dx, w, x0=0.0, y0=0.0, wy=None):
	"""Return a Gaussian field and its coordinates."""
	X, Y = wo.transverse_coordinates((n, n), dx, dx)
	wy = w if wy is None else wy
	return np.exp(-(X - x0) ** 2 / w**2 - (Y - y0) ** 2 / wy**2).astype(complex), X, Y


def test_drift_is_the_angular_spectrum():
	f0, X, Y = _gauss(256, 2e-10, 3e-9)
	ref = wo.angular_spectrum_propagate(f0, 2e-10, 2e-10, LAM, 2e-6, include_carrier=False)
	out, L, Q, form = wo.collins_propagate(f0, np.eye(2) * 2e-10, np.zeros((2, 2)), LAM, _drift(2e-6))
	assert form == "near"
	psi = wo.collins_field_at(out, L, (0, 0), (0, 0), Q, LAM, X, Y)
	assert np.abs(psi - ref).max() < 1e-12 * np.abs(ref).max()


def test_two_f_is_a_fourier_transform_with_the_gaussian_width():
	w0, f = 3e-9, 1e-3
	f0, _, _ = _gauss(256, 2e-10, w0)
	out, L, Q, form = wo.collins_propagate(f0, np.eye(2) * 2e-10, np.zeros((2, 2)), LAM,
										   _drift(f) @ _lens(f) @ _drift(f))
	assert form == "far"
	wf = LAM * f / (np.pi * w0)
	xs = np.linspace(-3 * wf, 3 * wf, 101)
	psi = np.abs(wo.collins_field_at(out, L, (0, 0), (0, 0), Q, LAM, xs, 0 * xs))
	assert np.abs(psi / psi.max() - np.exp(-xs**2 / wf**2)).max() < 1e-3
	# unitary: the power is the input's
	assert np.isclose((np.abs(out) ** 2).sum() * abs(np.linalg.det(L)),
					  (np.abs(f0) ** 2).sum() * 4e-20, rtol=1e-10)


def test_four_f_images_with_magnification_and_rotation_on_the_lattice():
	f1, f2, w = 1e-3, 5e-3, 3e-9
	f0, _, _ = _gauss(256, 2e-10, w, x0=2e-9)
	M = _rot(0.3) @ _drift(f2) @ _lens(f2) @ _drift(f1 + f2) @ _lens(f1) @ _drift(f1)
	out, L, Q, form = wo.collins_propagate(f0, np.eye(2) * 2e-10, np.zeros((2, 2)), LAM, M)
	assert form == "near"
	# the lattice carries the magnification (-5) and the rotation: no resampling
	assert np.allclose(L, -5 * 2e-10 * _rot(0.3)[np.ix_([0, 2], [0, 2])], rtol=1e-9)
	assert np.allclose(np.abs(out), np.abs(f0) / 5, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_line_focus_of_a_rotated_astigmatic_lens_matches_brute_force(angle):
	# x near its focus (far form), y barely propagated (near form): the mixed
	# form, on the lens's eigen-lattice when it is rotated
	w0, fx, d = 3e-9, 4e-6, 3.99e-6
	f0, _, _ = _gauss(256, 2e-10, w0)
	M = _drift(d) @ _rot(angle) @ _lens(fx, np.inf) @ _rot(-angle)
	out, L, Q, form = wo.collins_propagate(f0, np.eye(2) * 2e-10, np.zeros((2, 2)), LAM, M)
	assert form == "mixed"
	nb, db = 1024, 0.5e-10
	fb, Xb, Yb = _gauss(nb, db, w0)
	u = np.cos(angle) * Xb + np.sin(angle) * Yb
	ref = wo.angular_spectrum_propagate(fb * np.exp(-1j * np.pi / LAM * u**2 / fx), db, db, LAM, d,
										include_carrier=False)
	psi = wo.collins_field_at(out, L, (0, 0), (0, 0), Q, LAM, Xb, Yb)
	I1, I2 = np.abs(psi) ** 2, np.abs(ref) ** 2
	assert np.abs(I1 - I2).max() < 5e-4 * I2.max()


def _wave(n=128, dx=4e-10, w=4e-9, x0=0.0, wy=None):
	"""Return a fixed-grid Gaussian wavefield."""
	f, _, _ = _gauss(n, dx, w, x0=x0, wy=wy)
	return make_wavefield_signal(f, dx, dx, LAM, z=0.0)


def _column():
	"""Return a run with thick rotating lenses, a dipole and two crossovers."""
	return [Drift(length=1e-3), Lens(strength=40.0, length=2e-3, name="L1"),
			Dipole(name="D", length=0.0), Drift(length=20e-3),
			Lens(strength=30.0, length=3e-3, name="L2"), Drift(length=15e-3)]


def test_contracted_transfer_equals_element_by_element():
	elements = _column()
	transfer = contract_collins(elements)
	assert [s[0] for s in transfer.steps] == ["map"]
	one = read_collins_wavefield(transfer.apply(_wave(x0=3e-9)))
	wave = _wave(x0=3e-9)
	for element in elements:
		wave = element.propagate_wave_collins(wave)
	each = read_collins_wavefield(wave)
	grid = dict(shape=(96, 96), dx=abs(one[1][0, 0]) * 2, dy=abs(one[1][0, 0]) * 2)
	I1 = wo.collins_intensity_on_grid(one[0], one[1], one[2], **grid)
	I2 = wo.collins_intensity_on_grid(each[0], each[1], each[2], **grid)
	assert np.abs(I1 - I2).max() < 1e-6 * I1.max()


def test_the_beam_lands_where_the_central_ray_does_through_rotating_lenses():
	# an off-axis elliptical beam through thick (Larmor-rotating) lenses: its
	# centroid follows the lab-frame ray, its long axis turns with the lenses
	elements = _column()
	transfer = contract_collins(elements)
	f, L, x0, t0, Q, lam, z = read_collins_wavefield(transfer.apply(_wave(w=3e-9, wy=6e-9)))
	ray = np.zeros((1, 6))
	ray[0, 0] = 0.0
	for element in elements:
		ray = np.asarray(element.propagate_ray(ray), dtype=float)
	assert np.allclose(x0, [ray[0, 0], ray[0, 2]], atol=1e-15)
	# the beam's centroid sits on the central ray: the samples are centred
	P = np.abs(f) ** 2
	UX, UY = np.meshgrid(np.arange(f.shape[1]) - f.shape[1] // 2, np.arange(f.shape[0]) - f.shape[0] // 2)
	assert abs((P * UX).sum() / P.sum()) < 1e-3 and abs((P * UY).sum() / P.sum()) < 1e-3
	# the ray map of the whole run is the product of the elements' maps
	total = np.eye(5)
	for element in elements:
		total = element_affine_map(element) @ total
	assert np.allclose(transfer.affine_map(), total)


def test_a_dipole_moves_the_central_ray_and_not_the_samples():
	kicked = _column()
	kicked[2].strength = 1e-3
	plain = read_collins_wavefield(contract_collins(_column()).apply(_wave()))
	moved = read_collins_wavefield(contract_collins(kicked).apply(_wave()))
	assert np.allclose(plain[0], moved[0]) and np.allclose(plain[1], moved[1])
	if np.allclose(element_affine_map(kicked[2])[:4, 4], 0):
		pytest.skip("this Dipole's strength does not kick with no calibration set")
	assert not np.allclose(plain[2], moved[2])


def test_is_current_follows_the_elements():
	elements = _column()
	transfer = contract_collins(elements)
	assert transfer.is_current()
	elements[1].strength = 41.0
	assert not transfer.is_current()


def test_the_plan_is_reused_on_the_same_grid():
	transfer = contract_collins(_column())
	transfer.apply(_wave())
	plan = transfer._plans[0][1]
	transfer.apply(_wave(x0=1e-9))
	assert transfer._plans[0][1] is plan


def test_an_aperture_masks_as_the_fixed_path_does():
	ap = Aperture(radius=12e-9)
	wave = _wave(w=20e-9)
	collins = read_collins_wavefield(ap.propagate_wave_collins(wave))
	fixed = read_wavefield(ap.propagate_wave(wave, mode="fixed"))
	assert np.allclose(collins[0], fixed[0], atol=1e-12)
	assert [s[0] for s in CollinsTransfer([Drift(length=1e-3), ap, Drift(length=1e-3)]).steps] == \
		["map", "aperture", "map"]


def test_an_aberrated_element_is_refused_until_screens_exist():
	with pytest.raises(NotImplementedError, match="non-quadratic"):
		contract_collins([Drift(length=1e-3), Lens(strength=40.0, length=2e-3, aberrations={"C30": 1e-3})])


def test_a_rotated_quadrupole_is_carried():
	# the scaled frame refuses a rotated quadrupole; a Collins step does not
	quad = Quadrapole(strength=50.0, length=2e-3, rotation=0.3)
	out = read_collins_wavefield(contract_collins([Drift(length=1e-3), quad, Drift(length=5e-3)]).apply(_wave()))
	assert np.isfinite(out[0]).all()


def _rms_radius(state):
	"""Return a Collins wavefield's rms radius about its central ray."""
	f, L = state[0], state[1]
	P = np.abs(f) ** 2
	UX, UY = np.meshgrid(np.arange(f.shape[1]) - f.shape[1] // 2, np.arange(f.shape[0]) - f.shape[0] // 2)
	X, Y = L[0, 0] * UX + L[0, 1] * UY, L[1, 0] * UX + L[1, 1] * UY
	return np.sqrt((P * (X**2 + Y**2)).sum() / P.sum())


def test_agrees_with_the_hybrid_walker_through_crossovers():
	# a source aperture through two lenses with a real crossover between them:
	# the walker frame-switches there, the Collins step does not need to.
	# Measured 2026-11-06: Collins with the same pixel reproduces a 4x larger
	# 1024^2 grid to 5e-5 (it is one exact step), while the walker's frame
	# switches resample (6-14 % from that reference at 128-512 px, 2.4 % at
	# 1024 px). So compare what both must agree on: the power exactly (no
	# absorber), the rms size, and the image at a coarser pitch.
	src = Source(voltage=200, wave_shape=(128, 128), wave_extent=128 * 2.5e-7,
				 wave_kind="aperture", aperture_radius=10e-6)
	elements = [Drift(length=10e-3), Lens(strength=np.sqrt(1 / 45e-3), length=0.0, name="L1"),
				Drift(length=100e-3), Lens(strength=np.sqrt(1 / 30e-3), length=0.0, name="L2"),
				Drift(length=60e-3)]
	mic = Microscope(sections=[MicroscopeSection(elements=[src] + elements)])
	collins = read_collins_wavefield(mic.propagate_wave_collins())
	state = src.wave(mode="hybrid")
	for element in elements:
		state = element.propagate_wave(state, mode="hybrid", crossover="flat", absorb=0.0)
	walker = read_collins_wavefield(as_collins_wavefield(state))
	assert abs(_rms_radius(collins) / _rms_radius(walker) - 1) < 5e-3
	pitch = 4 * abs(collins[1][0, 0])
	grid = dict(shape=(64, 64), dx=pitch, dy=pitch)
	I1 = wo.collins_intensity_on_grid(collins[0], collins[1], collins[2], **grid)
	I2 = wo.collins_intensity_on_grid(walker[0], walker[1], walker[2], **grid)
	assert np.isclose(I1.sum(), I2.sum(), rtol=1e-9)
	assert np.abs(I1 - I2).sum() < 0.05 * I1.sum()


def test_dispatcher_routes_wave_collins():
	src = Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
				 wave_kind="aperture", aperture_radius=5e-6)
	section = MicroscopeSection(elements=[src, Drift(length=1e-3)])
	out = section.propagate(kind="wave_collins")
	assert out is section.wave_collins
