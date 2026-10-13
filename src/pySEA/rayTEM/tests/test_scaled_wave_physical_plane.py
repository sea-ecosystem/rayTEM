"""Tests for seashells.scaled_wave_physical_plane: ψ of one stored scaled plane."""

import numpy as np
import pytest

from pySEA.rayTEM import Drift, Lens, Microscope, MicroscopeSection, Source
from pySEA.rayTEM.seashells import (make_scaled_wave_signalset, read_wavefield, sea_available,
									scaled_wave_physical_plane)

pytestmark = pytest.mark.skipif(not sea_available, reason="needs sea_eco for SignalSets")


def _scaled_microscope():
	"""A source, drift, f = 45 mm lens and drift, propagated in scaled mode.

	Returns
	-------
	Microscope
		The column with ``wave_scaled`` stored.
	"""
	K = np.sqrt(1 / 45e-3)
	mic = Microscope(sections=[MicroscopeSection(elements=[
		Source(voltage=200, wave_shape=(64, 64), wave_extent=64 * 2.5e-7,
			   wave_kind="aperture", aperture_radius=5e-6),
		Drift(length=1e-3), Lens(strength=K, length=0.0, name="OL"), Drift(length=20e-3)])])
	mic.propagate_wave(mode="scaled")
	return mic


def test_every_stored_plane_matches_wavefield_at():
	"""The stored result alone gives what the live model's wavefield_at gives."""
	mic = _scaled_microscope()
	z = np.asarray(mic.wave_scaled["U"].dimensions[0].values, dtype=float)
	for plane in range(len(z)):
		if np.count_nonzero(z == z[plane]) > 1:
			continue  # wavefield_at names a plane by z, so a repeated z is ambiguous there
		got, dx, dy, lam, z_got = read_wavefield(scaled_wave_physical_plane(mic.wave_scaled, plane))
		want, dx_want, dy_want, lam_want, z_want = read_wavefield(mic.wavefield_at(z[plane]))
		assert np.allclose(got, want) and np.isclose(dx, dx_want) and np.isclose(dy, dy_want)
		assert np.isclose(z_got, z_want) and np.isclose(lam, lam_want)


def test_a_round_trip_through_sea_keeps_the_physical_plane(tmp_path):
	"""An exported, reloaded scaled wave reconstructs the same physical plane."""
	from pySEA.sea_eco.architecture.base_structure import SignalSet

	mic = _scaled_microscope()
	path = tmp_path / "scaled.sea"
	mic.wave_scaled.to_sea(str(path))
	loaded = SignalSet.load(str(path))
	stored = loaded if isinstance(loaded, SignalSet) else next(iter(loaded))
	last = len(mic.wave_scaled["U"].dimensions[0].values) - 1
	got, dx, *_ = read_wavefield(scaled_wave_physical_plane(stored, last))
	want, dx_want, *_ = read_wavefield(scaled_wave_physical_plane(mic.wave_scaled, last))
	assert np.allclose(got, want) and np.isclose(dx, dx_want)


def test_an_anisotropic_frame_uses_each_axis_scale():
	"""Per-axis s_x/s_y companions give per-axis pixel sizes and amplitude 1/√(s_x s_y)."""
	stack = make_scaled_wave_signalset(np.ones((2, 4, 6), complex), 1e-9, 2e-9, 2.5e-12,
									   s=[(1.0, 1.0), (0.5, 2.0)], R=[np.inf, np.inf],
									   tau=[0.0, 0.0], z=[0.0, 0.1])
	psi = scaled_wave_physical_plane(stack, 1)
	data, dx, dy, *_ = read_wavefield(psi)
	assert np.isclose(dx, 0.5e-9) and np.isclose(dy, 4e-9) and data.shape == (4, 6)
	assert np.allclose(np.abs(data), 1.0)


def test_it_refuses_what_is_not_a_stored_scaled_wave():
	"""A set without U on (plane_z, eta, xi), or a plane out of range, is refused."""
	stack = make_scaled_wave_signalset(np.ones((2, 4, 4), complex), 1e-9, 1e-9, 2.5e-12,
									   s=[1.0, 0.5], R=[np.inf, np.inf], tau=[0, 0], z=[0.0, 0.1])
	with pytest.raises(IndexError):
		scaled_wave_physical_plane(stack, 2)
	with pytest.raises(ValueError, match="U member"):
		scaled_wave_physical_plane([stack["s"]], 0)


def test_a_set_written_before_normalisation_reads_the_same():
	"""A legacy set (ξ in m, dimensionless s, no reference length) gives the same ψ."""
	stack = make_scaled_wave_signalset(np.ones((2, 4, 6), complex), 1e-9, 1e-9, 2.5e-12,
									   s=[1.0, 0.5], R=[np.inf, np.inf], tau=[0, 0], z=[0.0, 0.1])
	U, s = stack["U"], stack["s"]
	L = U.metadata["reference_half_width_m"]
	legacy = make_scaled_wave_signalset(np.ones((2, 4, 6), complex), 1e-9, 1e-9, 2.5e-12,
										s=[1.0, 0.5], R=[np.inf, np.inf], tau=[0, 0], z=[0.0, 0.1])
	legacy["s"].data = np.asarray(legacy["s"].data) / L
	delattr(legacy["U"].metadata, "reference_half_width_m")
	new, dx, *_ = read_wavefield(scaled_wave_physical_plane(stack, 1))
	old, dx_old, *_ = read_wavefield(scaled_wave_physical_plane(legacy, 1))
	assert np.allclose(new, old) and np.isclose(dx, dx_old) and np.isclose(dx, 0.5e-9)
	assert np.isclose(float(np.asarray(s.data)[1]), 0.5 * L)


def test_the_stacks_carry_the_columns_marks_bound_to_their_plane_axis():
	"""U and the fixed-grid stack carry overlays bound to their own z axis."""
	mic = _scaled_microscope()
	marks = mic.wave_scaled["U"].metadata.Display.Annotations
	assert {a.annotation_type for a in marks} >= {"Line"}
	assert all(a.role == "overlay" and a.coordinate_reference.get("x_dim") == "plane_z" for a in marks)
	assert any(a.text == "OL" for a in marks)
	mic.propagate_wave(mode="fixed")
	marks = mic.wave.metadata.Display.Annotations
	assert all(a.coordinate_reference.get("x_dim") == "z" for a in marks) and len(marks) > 0
