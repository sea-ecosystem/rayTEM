# TESTS BELOW INCLUDE:
# that a rayTEM Microscope stamps itself with a SEA ID saying what it is (a digital twin, TWN),
# that the stamp survives a .sea round trip, and that it degrades to None rather than raising
# when the pySEA data layer is not installed.
#
# NOTE: rayTEM never imports sea_eco or sea_sand directly — everything crossing into the pySEA
# data layer goes through seashells, minting included, so rayTEM installed on its own still runs.

import sys, os
import pytest

sys.path.insert(1, "../../../")

from pySEA.rayTEM.assemblies import MICROSCOPE_CLSID, Microscope, MicroscopeSection
from pySEA.rayTEM.elements import Drift, Lens
from pySEA.rayTEM.seashells import new_seaid, sea_available


def _clsid(seaid):
	"""Return the CLSID field of a hyphenated SEA ID."""
	return seaid.split("-")[1]


def test_microscope_is_stamped_as_a_digital_twin():
	"""A simulated column is a twin of an instrument, not an instrument."""
	m = Microscope(name="probe")
	if not sea_available:
		assert m.Provenance is None
		return
	assert _clsid(m.Provenance) == MICROSCOPE_CLSID
	assert MICROSCOPE_CLSID.startswith("TWN")


def test_each_microscope_gets_its_own_identifier():
	"""The CLSID says the kind; the date/time/random fields say which one."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so nothing is minted")
	first, second = Microscope(name="a"), Microscope(name="b")
	assert first.Provenance != second.Provenance
	assert _clsid(first.Provenance) == _clsid(second.Provenance)


def test_the_identifier_survives_a_sea_round_trip(tmp_path):
	"""An identifier that does not persist identifies nothing."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so there is nothing to round trip")
	column = Microscope(name="probe", sections=[
		MicroscopeSection(name="s", elements=[Lens(name="L", strength=1.0), Drift(name="D", length=0.1)])
	])
	minted = column.Provenance
	target = str(tmp_path / "column.sea")
	column.to_sea(target)

	back = Microscope()
	back.from_sea(target)
	assert back.Provenance == minted


def test_a_mistyped_clsid_is_normalized_rather_than_refused():
	"""Which is why the conformance check, not the minter, catches a typo.

	sea-sand substitutes the characters Crockford Base32 omits and pads or
	truncates to five, so a wrong CLSID still mints -- as something else. The
	stamp is only trustworthy because sea_eco's conformance check compares it
	against what the class declares.
	"""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so nothing is minted")
	assert _clsid(new_seaid("TOOLONG")) == "T0010"
	assert _clsid(new_seaid("TEMUU")) == "TEM00"
