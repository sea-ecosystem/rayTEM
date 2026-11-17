"""View levels (``schema.view_level_of``): how expert a view must be to show an object."""

import pytest

from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection
from pySEA.rayTEM.elements import Aperture, Dipole, Drift, Lens, Quadrapole, Source
from pySEA.rayTEM.schema import check_view_level, default_view_level, view_level_of
from pySEA.rayTEM.seashells import sea_available


def _column() -> Microscope:
	"""Return a small condenser column: a source, a dipole, a lens, a quad, an aperture and the sample.

	Returns
	-------
	Microscope
		One section ``C``.
	"""

	return Microscope(name="levels", sections=[MicroscopeSection(name="C", elements=[
		Source(), Dipole(name="CD1a"), Drift(length=0.01), Lens(name="CL1", strength=0.0), Drift(length=0.01),
		Quadrapole(name="CQ1"), Aperture(name="CA1", radius=1e-4), Drift(name="sample", length=0.0),
	])])


def test_defaults_follow_the_type_code():
	"""Eric (2026-11-17): the microscope 0, sections and apertures 1, round lenses 3, multipoles 7, everything 10."""
	column = _column()
	section = column.sections[0]
	assert view_level_of(column) == 0
	assert view_level_of(section) == 1
	levels = {element.name or type(element).__name__: view_level_of(element) for element in section.elements}
	assert levels["CL1"] == 3
	assert levels["CA1"] == 1
	assert levels["CD1a"] == 7
	assert levels["CQ1"] == 7
	assert levels["Drift"] == 10            # an unnamed layout drift
	assert levels["sample"] == 3            # a named plane is no drift volume
	assert default_view_level(Drift(name="PV1", length=0.1)) == 10


def test_a_set_level_overrides_the_default():
	"""``view_level`` set by hand wins, for elements and sections alike."""
	section = _column().sections[0]
	section["CD1a"].view_level = 4
	section.view_level = 2
	assert view_level_of(section["CD1a"]) == 4
	assert view_level_of(section) == 2


@pytest.mark.parametrize("value", [-1, 11, 2.5, "three", True])
def test_a_level_outside_0_to_10_is_refused(value):
	"""A level is a whole number 0..10 or None; anything else is an error, not a guess."""
	with pytest.raises(ValueError, match="0 to 10"):
		check_view_level(value)
	lens = Lens(name="CL1")
	lens.view_level = value
	with pytest.raises(ValueError, match="CL1"):
		view_level_of(lens)


def test_a_set_level_survives_a_sea_round_trip_and_an_unset_one_is_not_stored(tmp_path):
	"""The level is saved with the column; the default is not written at all."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so there is nothing to round trip")
	column = _column()
	column.sections[0].view_level = 2
	column.sections[0]["CL1"].view_level = 4
	target = str(tmp_path / "levels.sea")
	column.to_sea(target)
	back = _column()
	back.from_sea(target)
	assert back.sections[0].view_level == 2
	assert back.sections[0]["CL1"].view_level == 4
	assert "view_level" not in back.sections[0]["CD1a"].__dict__
