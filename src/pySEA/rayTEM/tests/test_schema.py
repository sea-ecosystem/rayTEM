# TESTS BELOW INCLUDE: (please run: "pytest test_schema.py")
# parsing and building of schema-conforming element names (CL1, PD1_a, PSa, ...)
# resolution of section names and aliases to section codes (condensers -> C)
# validation of sections and whole microscopes, including the three strict modes
# the exemptions that keep existing scopes valid: unmanaged sections, Drift/Source
#   named planes, and the detector container section
# extension of the vocabulary via register_section / register_element_type
# the vocabulary as plain data (get_schema_vocabulary), live after registrations

import copy
import json
import re

import pytest

from pySEA.rayTEM import Aperture, Dipole, Drift, Lens, Quadrapole, Source
from pySEA.rayTEM import Microscope, MicroscopeSection
from pySEA.rayTEM import schema
from pySEA.rayTEM.schema import (
	SchemaError,
	canonical_element_name,
	describe_schema,
	element_type_codes_for,
	get_schema_vocabulary,
	is_schema_typed,
	parse_element_name,
	section_code,
	validate_element_name,
	validate_microscope,
	validate_section,
)


@pytest.fixture
def vocabulary():
	"""Snapshot and restore the module-level vocabulary.

	``register_section`` and ``register_element_type`` mutate module state, so
	any test touching them must leave the vocabulary as it found it.

	Yields
	------
	module
		The :mod:`pySEA.rayTEM.schema` module.
	"""

	saved = (
		copy.deepcopy(schema.NAME_REQUIRED_CLASSES),
		copy.deepcopy(schema.SECTION_CODES),
		copy.deepcopy(schema.SECTION_ALIASES),
		copy.deepcopy(schema.ELEMENT_TYPE_CODES),
		copy.deepcopy(schema.ELEMENT_TYPE_CLASSES),
		schema._TYPED_CLASSES.copy(),
		schema._ELEMENT_RE,
	)
	yield schema
	(
		required,
		codes,
		aliases,
		types,
		type_classes,
		typed,
		pattern,
	) = saved
	schema.NAME_REQUIRED_CLASSES.clear(), schema.NAME_REQUIRED_CLASSES.update(required)
	schema.SECTION_CODES.clear(), schema.SECTION_CODES.update(codes)
	schema.SECTION_ALIASES.clear(), schema.SECTION_ALIASES.update(aliases)
	schema.ELEMENT_TYPE_CODES.clear(), schema.ELEMENT_TYPE_CODES.update(types)
	schema.ELEMENT_TYPE_CLASSES.clear(), schema.ELEMENT_TYPE_CLASSES.update(type_classes)
	schema._TYPED_CLASSES = typed
	schema._ELEMENT_RE = pattern


def condenser_section():
	"""Build a schema-conforming condenser section.

	Returns
	-------
	MicroscopeSection
		Section named ``"C"`` holding CL1, CA1 and CL2.
	"""

	return MicroscopeSection(
		name="C",
		elements=[
			Lens(name="CL1", strength=3, length=0.1),
			Drift(length=0.5),
			Aperture(name="CA1", radius=50e-6),
			Drift(length=0.5),
			Lens(name="CL2", strength=3, length=0.1),
		],
	)


# names already used across the project must all parse, and the parts must come
# back in the documented order
@pytest.mark.parametrize(
	"name,expected",
	[
		("CL1", ("C", "L", 1, None)),
		("CL3", ("C", "L", 3, None)),
		("OL2", ("O", "L", 2, None)),
		("PL4", ("P", "L", 4, None)),
		("CA1", ("C", "A", 1, None)),
		("SP1", ("S", "P", 1, None)),
		("PD1_a", ("P", "D", 1, "a")),
		("PSa", ("P", "S", None, "a")),
		("PSb", ("P", "S", None, "b")),
		("OD2_x", ("O", "D", 2, "x")),
		# probe corrector section, and the drift-volume type
		("AL1", ("A", "L", 1, None)),
		("AA1", ("A", "A", 1, None)),
		("PV1", ("P", "V", 1, None)),
		("DV", ("D", "V", None, None)),
	],
)
def test_parse_element_name_accepts_project_names(name, expected):
	parsed = parse_element_name(name)
	assert (parsed.section, parsed.type, parsed.index, parsed.axis) == expected
	assert parsed.attachment is None


# names that predate the schema, or are named planes, must parse to None rather
# than to something plausible-but-wrong
@pytest.mark.parametrize(
	"name", ["C1", "MC", "Diff", "IM", "sample", "VOA", "CCD", "CLs", "condenser_aperture", "", None]
)
def test_parse_element_name_rejects_non_conforming(name):
	assert parse_element_name(name) is None


def test_canonical_element_name_round_trips():
	for section, type_, index, axis in [("C", "L", 1, None), ("P", "D", 1, "a"), ("P", "S", None, "b")]:
		name = canonical_element_name(section, type_, index, axis)
		parsed = parse_element_name(name)
		assert (parsed.section, parsed.type, parsed.index, parsed.axis) == (section, type_, index, axis)


# a deflector is named for the element it sits on: CL1_D1a is dipole 1, axis a,
# on condenser lens 1 -- mirroring the AS2 client names (CL1_1Da)
@pytest.mark.parametrize(
	"name,host,attached",
	[
		("CL1_D1a", ("C", "L", 1), ("D", 1, "a")),
		("CL3_D2b", ("C", "L", 3), ("D", 2, "b")),
		("PL4_D1a", ("P", "L", 4), ("D", 1, "a")),
		("PV1_D1b", ("P", "V", 1), ("D", 1, "b")),
		("DV_D1a", ("D", "V", None), ("D", 1, "a")),
		("CL2_S1a", ("C", "L", 2), ("S", 1, "a")),
		("AL1_D1a", ("A", "L", 1), ("D", 1, "a")),
		("CL1_Q1", ("C", "L", 1), ("Q", 1, None)),
	],
)
def test_attached_multipole_names(name, host, attached):
	parsed = parse_element_name(name)
	assert (parsed.section, parsed.type, parsed.index) == host
	assert tuple(parsed.attachment) == attached
	assert canonical_element_name(*host, attachment=attached) == name


# the lower-case axis form must not be mistaken for an attached multipole
def test_bare_axis_is_not_an_attachment():
	assert parse_element_name("PD1_a").attachment is None
	assert parse_element_name("PSa").attachment is None


# CL1_D1a IS the dipole; the host letter only records where it sits
def test_attached_multipole_class_matches_the_multipole_not_the_host():
	dipole = Dipole(name="CL1_D1a", strength=0.1, length=0)
	assert validate_element_name("CL1_D1a", section="C", element=dipole) == []
	lens = Lens(name="CL1_D1a", strength=3, length=0.1)
	issues = validate_element_name("CL1_D1a", section="C", element=lens)
	assert [i.severity for i in issues] == ["error"]
	assert "Lens" in issues[0].message


def test_deflectors_validate_inside_a_condenser_section():
	section = MicroscopeSection(
		name="condensers",
		elements=[
			Dipole(name="CL1_D1a", strength=0.1, length=0),
			Lens(name="CL1", strength=3, length=0.1),
			Dipole(name="CL1_D2a", strength=0.1, length=0),
		],
	)
	assert section.validate_schema(strict=True).ok


def test_canonical_element_name_accepts_alias_and_rejects_junk():
	assert canonical_element_name("condensers", "L", 3) == "CL3"
	with pytest.raises(ValueError):
		canonical_element_name("flux capacitor", "L", 1)
	with pytest.raises(ValueError):
		canonical_element_name("C", "Z", 1)
	with pytest.raises(ValueError):
		canonical_element_name("C", "L", 1, "q")


@pytest.mark.parametrize(
	"name,code",
	[
		("C", "C"), ("c", "C"), ("condenser", "C"), ("condensers", "C"), ("CLs", "C"),
		("gun", "G"), ("projectors", "P"), ("detectors", "D"), ("EELS", "S"),
		("corrector", "A"), ("aberrations", "A"), ("probe corrector", "A"),
		("sec1", None), ("wien filter", None), ("", None), (None, None),
	],
)
def test_section_code_resolution(name, code):
	assert section_code(name) == code


def test_element_type_codes_follow_the_class():
	assert element_type_codes_for(Lens) == ("L",)
	assert element_type_codes_for(Dipole) == ("D",)
	assert element_type_codes_for(Aperture) == ("A",)
	# a Quadrapole may be named as either a stigmator or a plain quadrupole
	assert set(element_type_codes_for(Quadrapole)) == {"Q", "S"}
	# a Drift may be named as a drift volume; a Source has no type letter at all
	assert element_type_codes_for(Drift) == ("V",)
	assert element_type_codes_for(Source) == ()
	# "may be named" is not "must be named" -- neither is schema-typed
	assert is_schema_typed(Lens)
	assert not is_schema_typed(Drift) and not is_schema_typed(Source)


def test_validate_element_name_flags_wrong_section_letter():
	issues = validate_element_name("PL1", section="C")
	assert [i.severity for i in issues] == ["error"]
	assert "CL1" in issues[0].message


def test_validate_element_name_flags_type_class_mismatch():
	# a Dipole named as if it were a round lens
	issues = validate_element_name("CL1", section="C", element=Dipole(name="CL1", strength=0.1, length=0))
	assert [i.severity for i in issues] == ["error"]
	assert "Dipole" in issues[0].message


def test_validate_element_name_accepts_conforming():
	assert validate_element_name("CL1", section="condensers", element=Lens(name="CL1", strength=3, length=0.1)) == []


def test_conforming_section_validates_clean():
	report = validate_section(condenser_section(), strict=True)
	assert report.ok and len(report) == 0


def test_non_conforming_element_is_an_error():
	section = MicroscopeSection(
		name="condensers",
		elements=[Lens(name="CL1", strength=3, length=0.1), Drift(length=0.5), Aperture(name="condenser_aperture", radius=50e-6)],
	)
	report = validate_section(section, strict=False)
	assert not report.ok
	assert [i.path for i in report.errors] == ["condensers/condenser_aperture"]


# an ad-hoc scope must stay valid: a section nobody schema-named is simply not
# managed, so its elements are never name-checked
def test_unmanaged_section_is_info_not_error():
	section = MicroscopeSection(name="sec1", elements=[Lens(name="L1", strength=3, length=0.1)])
	report = validate_section(section, strict=True)
	assert report.ok
	assert [i.severity for i in report] == ["info"]


# named planes are Drift/Source objects and must never be name-checked, even
# inside a managed section
def test_named_planes_are_exempt():
	section = MicroscopeSection(
		name="O",
		elements=[
			Source(name="OL Source", size=(2e-4, 2e-4), np_xy=(3, 3), angle=(0, 0), na_xy=(1, 1)),
			Lens(name="OL1", strength=3, length=0.1),
			Drift(name="sample", length=0.5),
			Lens(name="OL2", strength=3, length=0.1),
		],
	)
	assert validate_section(section, strict=True).ok


# D is the drift volume the beam crosses on its way to the detector, plus any
# optics on it -- detector *devices* are roles in the registry, not elements
def test_detector_volume_section():
	assert section_code("detector volume") == "D"
	assert section_code("detectors") == "D"
	section = MicroscopeSection(
		name="detector volume",
		elements=[
			Drift(name="DV", length=0.5),
			Dipole(name="DV_D1a", strength=0.1, length=0),
			Dipole(name="DV_D1b", strength=0.1, length=0),
		],
	)
	assert section.validate_schema(strict=True).ok


def test_duplicate_names_within_a_section_are_an_error():
	section = MicroscopeSection(
		name="C",
		elements=[Lens(name="CL1", strength=3, length=0.1), Drift(length=0.5), Lens(name="CL1", strength=3, length=0.1)],
	)
	report = validate_section(section, strict=False)
	assert not report.ok
	assert "duplicate" in report.errors[0].message


def test_duplicate_names_across_sections_are_an_error():
	scope = Microscope(
		sections=[
			MicroscopeSection(name="C", elements=[Drift(name="sample", length=0.5)]),
			MicroscopeSection(name="O", elements=[Drift(name="sample", length=0.5)], position=2.0),
		]
	)
	report = validate_microscope(scope, strict=False)
	assert not report.ok
	assert "used in 2 sections" in report.errors[0].message


def test_microscope_validates_and_reports_per_section():
	scope = Microscope(
		sections=[
			condenser_section(),
			MicroscopeSection(name="projectors", elements=[Lens(name="OL1", strength=3, length=0.1)], position=3.0),
		]
	)
	report = validate_microscope(scope, strict=False)
	assert not report.ok
	assert report.errors[0].path == "projectors/OL1"
	assert "PL1" in report.errors[0].message


def test_strict_modes():
	bad = MicroscopeSection(name="C", elements=[Lens(name="nope", strength=3, length=0.1)])
	# True (the default) raises
	with pytest.raises(SchemaError):
		validate_section(bad)
	# "warn" warns but returns the report
	with pytest.warns(UserWarning):
		assert not validate_section(bad, strict="warn").ok
	# False is silent
	report = validate_section(bad, strict=False)
	assert not report.ok
	# anything else is a programming error
	with pytest.raises(ValueError):
		validate_section(bad, strict="maybe")


def test_assembly_convenience_methods():
	section = condenser_section()
	assert section.validate_schema().ok
	scope = Microscope(sections=[section])
	assert scope.validate_schema(strict=True).ok
	with pytest.raises(SchemaError):
		Microscope(sections=[MicroscopeSection(name="C", elements=[Lens(name="XX9", strength=3, length=0.1)])]).validate_schema()


def test_register_section_extends_the_vocabulary(vocabulary):
	section = MicroscopeSection(name="wien filter", elements=[Quadrapole(name="WSa", strength=0.5, length=0.1)])
	# before registration the section is simply unmanaged
	assert validate_section(section, strict=True).ok
	vocabulary.register_section("W", "wien filter", aliases=("wien",))
	assert section_code("wien filter") == "W" and section_code("wien") == "W"
	parsed = parse_element_name("WSa")
	assert (parsed.section, parsed.type, parsed.axis) == ("W", "S", "a")
	assert validate_section(section, strict=True).ok


# a Drift may be named as a drift volume, but is never required to be: that is
# what keeps named planes (sample, VOA, CCD) and auto-inserted drifts valid
def test_drift_may_be_named_but_is_never_required():
	from pySEA.rayTEM.schema import NAME_REQUIRED_CLASSES
	assert "Drift" not in NAME_REQUIRED_CLASSES
	assert element_type_codes_for(Drift) == ("V",)
	assert not is_schema_typed(Drift)
	section = MicroscopeSection(
		name="P",
		elements=[Drift(name="PV1", length=0.5), Lens(name="PL1", strength=3, length=0.1), Drift(name="sample", length=0.2)],
	)
	assert section.validate_schema(strict=True).ok


# ... but a name that does parse is still held to the schema
def test_a_parseable_name_on_an_exempt_class_is_still_checked():
	section = MicroscopeSection(name="C", elements=[Drift(name="PV1", length=0.5)])
	report = section.validate_schema(strict=False)
	assert not report.ok
	assert "does not match owning section" in report.errors[0].message


def test_probe_corrector_section():
	section = MicroscopeSection(
		name="corrector",
		elements=[Lens(name="AL1", strength=3, length=0.1), Drift(length=0.2), Aperture(name="AA1", radius=50e-6)],
	)
	assert section_code("corrector") == "A"
	assert section_code("aberrations") == "A"
	assert section.validate_schema(strict=True).ok


def test_register_element_type_extends_the_vocabulary(vocabulary):
	assert parse_element_name("OZ1") is None
	vocabulary.register_element_type("Z", "octupole", ("Quadrapole",), required=False)
	parsed = parse_element_name("OZ1")
	assert (parsed.section, parsed.type, parsed.index) == ("O", "Z", 1)
	section = MicroscopeSection(name="O", elements=[Quadrapole(name="OZ1", strength=0.5, length=0.1)])
	assert validate_section(section, strict=True).ok


def test_register_rejects_collisions(vocabulary):
	with pytest.raises(ValueError):
		vocabulary.register_section("C", "something else")
	with pytest.raises(ValueError):
		vocabulary.register_section("XX", "two letters")
	with pytest.raises(ValueError):
		vocabulary.register_element_type("L", "not a lens", ("Lens",))


def test_describe_schema_lists_the_live_vocabulary():
	text = describe_schema()
	for token in ["condenser", "probe corrector", "spectrometer", "stigmator",
	              "drift volume", "CL1", "CL1_D1a", "Attached multipole"]:
		assert token in text


# the structured vocabulary: the same live data as describe_schema, as plain
# JSON-serialisable values an agent or another process can read
def test_get_schema_vocabulary_is_plain_json():
	vocabulary = get_schema_vocabulary()
	assert json.loads(json.dumps(vocabulary)) == vocabulary
	assert set(vocabulary) == {
		"sections", "element_types", "axis_suffixes", "patterns", "name_regex",
		"name_required_classes", "exempt_classes", "rules",
	}


def test_get_schema_vocabulary_lists_sections_with_aliases():
	sections = {s["code"]: s for s in get_schema_vocabulary()["sections"]}
	assert sorted(sections) == sorted(schema.SECTION_CODES)
	assert sections["C"]["name"] == "condenser"
	assert {"condensers", "cls"} <= set(sections["C"]["aliases"])
	# the canonical name is the name, not repeated as an alias
	assert "condenser" not in sections["C"]["aliases"]
	assert {"corrector", "aberrations"} <= set(sections["A"]["aliases"])


def test_get_schema_vocabulary_lists_element_types():
	types = {t["code"]: t for t in get_schema_vocabulary()["element_types"]}
	assert sorted(types) == sorted(schema.ELEMENT_TYPE_CODES)
	assert types["L"] == {"code": "L", "description": "round lens", "classes": ["Lens"], "name_required": True}
	assert types["S"]["classes"] == ["Quadrapole"]
	# V may name a Drift but never requires it
	assert types["V"]["classes"] == ["Drift"] and types["V"]["name_required"] is False


def test_get_schema_vocabulary_patterns_and_exemptions():
	vocabulary = get_schema_vocabulary()
	assert vocabulary["axis_suffixes"] == list(schema.AXIS_SUFFIXES)
	assert vocabulary["patterns"]["host"]["form"] == "<SECTION><TYPE>[INDEX]"
	assert vocabulary["patterns"]["attached_multipole"]["form"] == "<HOST>_<TYPE>[INDEX][AXIS]"
	for pattern in vocabulary["patterns"].values():
		for example in pattern["examples"]:
			assert parse_element_name(example) is not None
	assert re.match(vocabulary["name_regex"], "CL1_D1a")
	assert not re.match(vocabulary["name_regex"], "sample")
	assert vocabulary["name_required_classes"] == sorted(schema.NAME_REQUIRED_CLASSES)
	assert vocabulary["exempt_classes"] == ["Drift", "Element", "Source"]


def test_get_schema_vocabulary_shows_registered_additions(vocabulary):
	vocabulary.register_section("W", "wien filter", aliases=("wien",))
	vocabulary.register_element_type("Z", "octupole", ("Octupole",), required=False)
	data = get_schema_vocabulary()
	sections = {s["code"]: s for s in data["sections"]}
	assert sections["W"] == {"code": "W", "name": "wien filter", "aliases": ["wien"]}
	types = {t["code"]: t for t in data["element_types"]}
	assert types["Z"] == {"code": "Z", "description": "octupole", "classes": ["Octupole"], "name_required": False}
	assert "Octupole" in data["exempt_classes"]
	assert re.match(data["name_regex"], "WZ1")
