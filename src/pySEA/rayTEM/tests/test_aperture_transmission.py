"""An aperture's ``transmission``: current the traced rays do not sample.

A gun's anode aperture accepts a tiny solid angle of what the tip emits -- a share
no affordable ray fan resolves by clipping. ``transmission`` states it, and the ray
current, the wave's flux and a saved column all carry it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pySEA.rayTEM import Aperture, Drift, Microscope, MicroscopeSection, Source


def _column(transmission: float) -> Microscope:
	"""A source, an aperture wide enough to pass every ray, and a drift."""
	return Microscope(sections=[MicroscopeSection(name="sec", elements=[
		Source(voltage=200, beam_current=1.5e-6, size=(1e-9, 1e-9), np_xy=(3, 3), angle=(1e-4, 1e-4), na_xy=(3, 3)),
		Drift(length=0.01),
		Aperture(name="A1", radius=1e-3, transmission=transmission),
		Drift(length=0.01),
	])])


def test_the_ray_current_after_the_aperture_is_scaled_by_its_transmission():
	full, tiny = _column(1.0), _column(1 / 3000)
	full.propagate()
	tiny.propagate()
	assert full.beam_current == pytest.approx(1.5e-6)
	assert tiny.beam_current == pytest.approx(1.5e-6 / 3000)


def test_a_transmission_outside_zero_to_one_is_refused():
	for bad in (0.0, -0.1, 1.5):
		with pytest.raises(ValueError, match="transmission"):
			Aperture(radius=1e-4, transmission=bad)


def test_the_wave_screen_passes_the_same_share_of_flux():
	open_screen = Aperture(radius=1e-6).phase_shift(((32, 32), 1e-7, 1e-7), 2.5e-12)
	dim_screen = Aperture(radius=1e-6, transmission=0.25).phase_shift(((32, 32), 1e-7, 1e-7), 2.5e-12)
	flux = lambda program: float(np.sum(np.abs(np.asarray(program[0])) ** 2))
	assert flux(dim_screen) == pytest.approx(0.25 * flux(open_screen))


def test_a_saved_aperture_keeps_its_transmission_and_an_old_one_reads_as_one(tmp_path):
	scope = _column(0.01)
	scope.to_sea(str(tmp_path / "now.sea"))
	again = Microscope()
	again.from_sea(str(tmp_path / "now.sea"))
	assert again.sections[0]["A1"].transmission == pytest.approx(0.01)
	# a file written before transmission existed has no such field
	old = _column(1.0)
	del old.sections[0]["A1"].transmission
	old.to_sea(str(tmp_path / "old.sea"))
	reread = Microscope()
	reread.from_sea(str(tmp_path / "old.sea"))
	assert reread.sections[0]["A1"].transmission == 1.0
