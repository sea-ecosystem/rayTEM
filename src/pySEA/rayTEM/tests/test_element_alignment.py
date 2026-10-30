"""An element's ``alignment``: where its own axis is, off the column's.

Eric (2026-10-30): default None skips the step; otherwise an affine frame change
around the element's transfer matrix accounts for any offset or tilt, on every
element, "so that we can begin to calibrate what the actual geometry is in the
microscope". Each claim is checked against what the physics says the element
must do, not against the frame change itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from pySEA.rayTEM import Aperture, Drift, Lens, Microscope, MicroscopeSection, Source
from pySEA.rayTEM.elements import Rays, columnByName, convention

X, XT, Y, YT = (columnByName(name) for name in ("x", "xt", "y", "yt"))


def _rays(count: int = 7, seed: int = 0) -> np.ndarray:
	"""Return random paraxial rays, shape ``(count, len(convention))``."""
	rays = np.zeros((count, len(convention)))
	rays[:, [X, Y]] = np.random.default_rng(seed).normal(0.0, 1e-6, (count, 2))
	rays[:, [XT, YT]] = np.random.default_rng(seed + 1).normal(0.0, 1e-3, (count, 2))
	return rays


def test_an_element_on_the_axis_is_exactly_as_before() -> None:
	lens = Lens(length=0.004, focal_length=0.01)
	before = lens.propagate_ray(_rays())
	assert lens.alignment is None and lens.alignment_offsets() is None
	lens.alignment = (0.0, 0.0, 0.0, 0.0)
	np.testing.assert_array_equal(lens.propagate_ray(_rays()), before)


def test_a_displaced_lens_steers_the_beam_toward_its_own_axis() -> None:
	"""A thin lens of focal length f, displaced by d, bends an axial ray by d/f."""
	lens = Lens(focal_length=0.01)
	lens.alignment = (2e-6, 0.0, -1e-6, 0.0)
	out = lens.propagate_ray(np.zeros((1, len(convention))))[0]
	assert out[[X, XT, Y, YT]] == pytest.approx([0.0, 2e-6 / 0.01, 0.0, -1e-6 / 0.01], abs=1e-15)


def test_a_drift_is_the_same_however_it_lies() -> None:
	"""Free space has no axis: an offset or tilted drift changes nothing."""
	drift = Drift(length=0.05)
	straight = drift.propagate_ray(_rays())
	drift.alignment = (3e-6, 2e-3, -1e-6, -4e-3)
	np.testing.assert_allclose(drift.propagate_ray(_rays()), straight, atol=1e-18)


def test_the_affine_matrix_is_the_ray_path() -> None:
	lens = Lens(length=0.004, focal_length=0.01)
	lens.alignment = (1e-6, 5e-4, -2e-6, 1e-3)
	rays = _rays()
	homogeneous = np.hstack([rays, np.ones((len(rays), 1))])
	np.testing.assert_allclose((lens.affine_matrix() @ homogeneous.T).T[:, :-1], lens.propagate_ray(rays),
							   rtol=0, atol=1e-15)
	# Only the last column moves: the linear part is the lens's own matrix.
	np.testing.assert_array_equal(lens.affine_matrix()[:-1, :-1], lens.transfer_matrix())


def test_a_tilted_lens_is_a_lens_about_a_tilted_axis() -> None:
	"""A ray along the tilted axis is the lens's axial ray: it leaves along the axis."""
	lens = Lens(length=0.004, focal_length=0.01)
	tilt = 1e-3
	lens.alignment = (0.0, tilt, 0.0, 0.0)
	along = np.zeros((1, len(convention)))
	along[0, X], along[0, XT] = -tilt * 0.002, tilt		# on the axis at the entrance face
	out = lens.propagate_ray(along)[0]
	assert out[X] == pytest.approx(tilt * 0.002, abs=1e-15)
	assert out[XT] == pytest.approx(tilt, abs=1e-15)


def test_a_displaced_aperture_clips_about_its_own_centre() -> None:
	aperture = Aperture(radius=1e-6)
	aperture.alignment = (3e-6, 0.0, 0.0, 0.0)
	rays = np.zeros((2, len(convention)))
	rays[1, X] = 3e-6
	out = aperture.propagate_ray(Rays(rays, I_per_ray=np.ones(2), I_per_plane=2.0))
	assert list(np.asarray(out.I_per_ray)) == [0.0, 1.0]
	np.testing.assert_allclose(np.asarray(out.rays)[:, :4], rays[:, :4], atol=1e-18)


def test_the_mean_moves_and_the_spread_does_not() -> None:
	lens = Lens(focal_length=0.01)
	mu = np.zeros(len(convention))
	sigma = np.diag([1e-12, 1e-6, 1e-12, 1e-6, 0.0, 0.0])
	mu_on, sigma_on = lens.propagate_moments(mu, sigma)
	lens.alignment = (2e-6, 0.0, 0.0, 0.0)
	mu_off, sigma_off = lens.propagate_moments(mu, sigma)
	np.testing.assert_allclose(sigma_off, sigma_on, rtol=0, atol=0)
	assert mu_off[XT] - mu_on[XT] == pytest.approx(2e-6 / 0.01, abs=1e-15)


def test_a_gun_off_the_axis_sends_the_beam_along_its_own() -> None:
	source = Source(size=(1e-9, 1e-9), np_xy=(3, 3), angle=(1e-4, 1e-4), na_xy=(3, 3))
	centred = np.asarray(source.rays())
	source.alignment = (1e-6, 2e-4, 0.0, 0.0)
	moved = np.asarray(source.rays()) - centred
	assert moved[:, X] == pytest.approx(np.full(len(moved), 1e-6))
	assert moved[:, XT] == pytest.approx(np.full(len(moved), 2e-4))
	assert source.moments()[0][[X, XT]] == pytest.approx([1e-6, 2e-4])


def test_the_fixed_wave_clips_and_steers_about_the_elements_axis() -> None:
	from pySEA.rayTEM.seashells import make_wavefield_signal, read_wavefield
	from pySEA.rayTEM.waveoptics import transverse_coordinates

	shape, dx, wavelength = (128, 128), 5e-8, 2.5e-12
	aperture = Aperture(radius=4e-7)
	aperture.alignment = (1e-6, 0.0, 0.0, 0.0)
	field = make_wavefield_signal(np.ones(shape, complex), dx, dx, wavelength, z=0.0)
	data = read_wavefield(aperture.propagate_wave(field))[0]
	X_m, Y_m = transverse_coordinates(shape, dx, dx)
	weight = np.abs(data) ** 2
	assert float(np.sum(weight * X_m) / np.sum(weight)) == pytest.approx(1e-6, abs=2e-8)
	assert float(np.sum(weight * Y_m) / np.sum(weight)) == pytest.approx(0.0, abs=2e-8)


def test_the_scaled_wave_says_it_does_not_carry_an_alignment_yet() -> None:
	scope = Microscope(sections=[MicroscopeSection(name="sec", elements=[
		Source(voltage=200, size=(1e-9, 1e-9)), Drift(length=0.01), Lens(name="L", focal_length=0.01),
		Drift(length=0.01)])])
	scope.sections[0]["L"].alignment = (1e-6, 0.0, 0.0, 0.0)
	with pytest.raises(NotImplementedError, match="alignment"):
		scope.propagate_wave(mode="hybrid")


def test_a_saved_alignment_comes_back_and_an_old_file_reads_as_on_axis(tmp_path) -> None:
	def column() -> Microscope:
		return Microscope(sections=[MicroscopeSection(name="sec", elements=[
			Source(voltage=200, size=(1e-9, 1e-9)), Drift(length=0.01), Lens(name="L", focal_length=0.01)])])

	scope = column()
	scope.sections[0]["L"].alignment = (1e-6, 2e-4, -3e-6, 0.0)
	scope.to_sea(str(tmp_path / "aligned.sea"))
	again = Microscope()
	again.from_sea(str(tmp_path / "aligned.sea"))
	assert list(again.sections[0]["L"].alignment) == pytest.approx([1e-6, 2e-4, -3e-6, 0.0])
	column().to_sea(str(tmp_path / "plain.sea"))
	plain = Microscope()
	plain.from_sea(str(tmp_path / "plain.sea"))
	assert plain.sections[0]["L"].alignment is None
