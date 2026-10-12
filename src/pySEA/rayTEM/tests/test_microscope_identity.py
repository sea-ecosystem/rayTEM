# TESTS BELOW INCLUDE:
# that a rayTEM Microscope stamps itself with a Provenance whose Self says what it is (a digital
# twin, TWN), that the identity survives a .sea round trip and is written in sea_eco's layout
# (Provenance/Self), that a column saved with the pre-2026-10-12 string id still loads with the
# same id, and that it degrades to None rather than raising when the pySEA data layer is not
# installed.
#
# NOTE: rayTEM never imports sea_eco or sea_sand directly — everything crossing into the pySEA
# data layer goes through seashells, minting included, so rayTEM installed on its own still runs.

import sys, os
import pytest

sys.path.insert(1, "../../../")

from pySEA.rayTEM.assemblies import MICROSCOPE_CLSID, Microscope, MicroscopeSection
from pySEA.rayTEM.elements import Drift, Lens
from pySEA.rayTEM.seashells import new_provenance, sea_available


def _clsid(provenance):
	"""Return the CLSID field of a Provenance's own SEA ID."""
	return provenance.Self.seaid.split("-")[1]


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
	assert back.Provenance.Self.seaid == minted.Self.seaid


def test_the_identity_is_written_as_self_and_relations(tmp_path):
	"""The file holds sea_eco's layout, not a plain string attribute."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so nothing is written")
	import h5py
	column = Microscope(name="probe", sections=[MicroscopeSection(name="s", elements=[Drift(name="D", length=0.1)])])
	target = str(tmp_path / "column.sea")
	column.to_sea(target)
	with h5py.File(target, "r") as f:
		root = f["probe"]
		assert "Provenance" not in root.attrs
		assert root["Provenance/Self"].attrs["seaid"] == column.Provenance.Self.seaid


def test_a_column_saved_with_a_string_id_keeps_it(tmp_path):
	"""Columns saved 2026-09-22..2026-10-12 stored the id as a string; it is wrapped on load."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so there is nothing to load")
	import h5py
	column = Microscope(name="probe", sections=[MicroscopeSection(name="s", elements=[Drift(name="D", length=0.1)])])
	target = str(tmp_path / "column.sea")
	column.to_sea(target)
	old_id = column.Provenance.Self.seaid
	with h5py.File(target, "r+") as f:					# rewrite it in the old layout
		root = f["probe"]
		del root["Provenance"]
		root.attrs["Provenance"] = old_id
	back = Microscope()
	with pytest.warns(DeprecationWarning, match="2026-12-03"):
		back.from_sea(target)
	assert back.Provenance.Self.seaid == old_id
	assert _clsid(back.Provenance) == MICROSCOPE_CLSID


def test_the_json_save_writes_the_identity(tmp_path):
	"""The flat JSON layout carries the Provenance in sea_eco's JSON form."""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so nothing is minted")
	import json
	column = Microscope(name="probe", sections=[MicroscopeSection(name="s", elements=[Drift(name="D", length=0.1)])])
	column.save(str(tmp_path / "column"))
	written = json.loads((tmp_path / "column.json").read_text())["Provenance"]
	assert written["sea_type"] == "Provenance"
	assert written["Self"]["seaid"] == column.Provenance.Self.seaid


def test_a_mistyped_clsid_is_normalized_rather_than_refused():
	"""Which is why the conformance check, not the minter, catches a typo.

	sea-sand substitutes the characters Crockford Base32 omits and pads or
	truncates to five, so a wrong CLSID still mints -- as something else. The
	stamp is only trustworthy because sea_eco's conformance check compares it
	against what the class declares.
	"""
	if not sea_available:
		pytest.skip("sea_eco is not installed, so nothing is minted")
	assert _clsid(new_provenance("TOOLONG")) == "T0010"
	assert _clsid(new_provenance("TEMUU")) == "TEM00"
