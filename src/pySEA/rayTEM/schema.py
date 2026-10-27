"""Column naming schema for rayTEM microscopes.

Defines the controlled vocabulary for microscope section and element names --
the ``<SECTION><TYPE>[INDEX][AXIS]`` convention (``CL1``, ``PL4``, ``CD1a``,
``PSa``) -- together with a parser and a validator.

The schema is a *name-pattern* convention: the ``name`` string on a
:class:`~pySEA.rayTEM.assemblies.MicroscopeSection` or
:class:`~pySEA.rayTEM.elements.Element` stays the single source of truth, so
``section["CL1"]`` lookup, ``.sea`` round-trips and every existing microscope
definition keep working unchanged. No new attribute is introduced. In
particular ``Element.kind`` is deliberately left alone: it is load-bearing for
``ignoreLensThickness`` and ``__delitem__`` in :mod:`pySEA.rayTEM.assemblies`,
and it describes the transfer matrix rather than the position in the column.

Sections
--------
A section code says where in the column the section sits:

==== ===============  =========================================
Code Section          Typical contents
==== ===============  =========================================
G    gun              source, extractor, gun lens/deflectors
M    monochromator    monochromator optics, slit
C    condenser        condenser lenses, apertures, stigmators
A    probe corrector  corrector optics and its multipoles
O    objective        objective lenses, sample, stage, coils
P    projector        projector/intermediate lenses, volumes
E    spectrometer     prism, spectrometer multipoles
D    detector volume  drift volume to the detector, and its optics
==== ===============  =========================================

``A`` and ``C`` are both section codes and element type codes. Position
resolves it: the first letter is always the section, the second the type, so
``AA1`` is the corrector's first aperture and ``CL1`` the first condenser
lens.

Element types
-------------
The second letter says what the element is, and constrains which rayTEM class
it may be:

==== ====================  ============
Code Meaning               Class
==== ====================  ============
L    round lens            Lens
D    dipole / deflector    Dipole
S    sextupole             Sextupole
Q    quadrupole/stigmator  Quadrapole
A    aperture              Aperture
P    prism / sector        Prism
V    drift volume          Drift
==== ====================  ============

``V`` is the one type that may be named but is never *required* to be -- see
:data:`NAME_REQUIRED_CLASSES`.

A quadrupole is ``Q`` whether it focuses or stigmates. ``S`` meant stigmator until
2026-10-27; it now names a sextupole (Eric: "quads are for stigmation; use S for
sextupoles"), as in a gun lens's multipole stack. rayTEM has no
``Sextupole`` element yet, so no class is typed ``S`` and a ``Quadrapole`` named with
``S`` is reported, with ``Q`` as the letter to use.

Multipoles
----------
A deflector, quadrupole or sextupole is named on its own in its section, like
any element, never relative to the lens it sits by (Eric, 2026-10-21):
``CD1`` is the condenser's first dipole and ``CD1a``/``CD1b`` its two
windings. ``CL1_D1a`` is refused, with ``CD1a`` named in the message.

Numbers count each type within its section, in column order. They are the
schema's, not a vendor's: they need not match any controller's slot numbers,
and a number may be missing (``GD1``, ``GD2``, ``GD3``, ``GD10``), but they
never run backwards.

The axis suffix follows the index directly (``CD1a``) or, with no index,
the type (``PSa``). An older underscore (``PD1_a``) still parses.

The spectrometer was ``S`` until 2026-10-21; a file that still has a section
named ``S`` is renamed to ``E`` as it is read (:func:`migrate_legacy_section`).

Notes
-----
Validation is *scoped by section*. A section whose name does not resolve to a
code is simply not schema-managed and its elements are left alone, so ad-hoc
scopes (``MicroscopeSection(name="sec1")``) and bare test fixtures stay valid.
Only a section that opts in by carrying a recognised name has its typed
elements checked.

``Drift``, ``Source`` and plain ``Element`` objects are never name-checked:
drifts are auto-inserted by
:class:`~pySEA.rayTEM.assemblies.MicroscopeSection`, and named planes such as
``"sample"``, ``"VOA"`` or ``"CCD"`` are the intended use of those classes.

The detector section ``D`` holds the drift volume the beam crosses on its way
to the detector (``DV``) and any optics on it (``DV_D1a``), the same way the
projector section holds ``PV1``/``PV2``. Detector *devices* are not column
elements at all: a detector is a role resolved by sea-pearl's
``DetectorRoleRegistry``, which is where ``"HAADF"``, ``"EELS"`` and
``"Ronchigram"`` belong.

Examples
--------
>>> parse_element_name("CL1")
ElementName(section='C', type='L', index=1, axis=None)
>>> parse_element_name("CD1a")
ElementName(section='C', type='D', index=1, axis='a')
>>> parse_element_name("CL1_D1a") is None
True
>>> canonical_element_name("C", "L", 2)
'CL2'
>>> section_code("condensers")
'C'
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable, Iterator, NamedTuple, Sequence

__all__ = [
	"SECTION_CODES",
	"SECTION_ALIASES",
	"ELEMENT_TYPE_CODES",
	"ELEMENT_TYPE_CLASSES",
	"AXIS_SUFFIXES",
	"NAME_REQUIRED_CLASSES",
	"ElementName",
	"SchemaIssue",
	"SchemaReport",
	"SchemaError",
	"section_code",
	"parse_element_name",
	"canonical_element_name",
	"element_type_codes_for",
	"is_schema_typed",
	"validate_element_name",
	"validate_section",
	"validate_microscope",
	"describe_schema",
	"migrate_legacy_section",
	"register_section",
	"register_element_type",
]


SECTION_CODES: dict[str, str] = {
	"G": "gun",
	"M": "monochromator",
	"C": "condenser",
	"A": "probe corrector",
	"O": "objective",
	"P": "projector",
	"E": "spectrometer",
	"D": "detector volume",
}
"""Single-letter section code mapped to its canonical section name."""

SECTION_ALIASES: dict[str, str] = {
	# canonical names
	"gun": "G",
	"monochromator": "M",
	"condenser": "C",
	"probe corrector": "A",
	"corrector": "A",
	"aberration corrector": "A",
	"aberrations": "A",
	"objective": "O",
	"projector": "P",
	"spectrometer": "E",
	"detector volume": "D",
	"detector": "D",
	# plurals and shorthands already in use across the project
	"condensers": "C",
	"objectives": "O",
	"projectors": "P",
	"spectrometers": "E",
	"detectors": "D",
	"gls": "G",
	"mls": "M",
	"cls": "C",
	"ols": "O",
	"pls": "P",
	"source": "G",
	"eels": "E",
	"prism": "E",
}
"""Lower-cased free-form section name mapped to a section code.

Extends the single-letter codes so that names already in the wild
(``"condensers"``, ``"CLs"``, ``"projectors"``) resolve without a rename.
"""

ELEMENT_TYPE_CODES: dict[str, str] = {
	"L": "round lens",
	"D": "dipole / deflector",
	"S": "sextupole",
	"Q": "quadrupole / stigmator",
	"A": "aperture",
	"P": "prism / sector",
	"V": "drift volume",
}
"""Single-letter element type code mapped to a human-readable description."""

ELEMENT_TYPE_CLASSES: dict[str, tuple[str, ...]] = {
	"L": ("Lens",),
	"D": ("Dipole",),
	"S": ("Sextupole",),
	"Q": ("Quadrapole",),
	"A": ("Aperture",),
	"P": ("Prism",),
	"V": ("Drift",),
}
"""Element type code mapped to the rayTEM class names it may name.

Matching is done against the whole MRO, so sea-pearl's ``LensModel`` satisfies
``"L"`` through its ``Lens`` base. A class absent from every entry here (for
example ``Drift``, ``Source`` or plain ``Element``) is not schema-typed and is
never name-checked.
"""

AXIS_SUFFIXES: tuple[str, ...] = ("a", "b", "x", "y", "u", "v")
"""Permitted axis/component suffixes, as in ``CD1a`` or ``PSa``."""

NAME_REQUIRED_CLASSES: set[str] = {"Lens", "Dipole", "Quadrapole", "Aperture", "Prism"}
"""Classes that must carry a schema name inside a managed section.

Deliberately narrower than :data:`ELEMENT_TYPE_CLASSES`. ``Drift`` can be
*named* as a drift volume (``PV1``, ``DV``) but is never *required* to be:
drifts are auto-inserted by :class:`~pySEA.rayTEM.assemblies.MicroscopeSection`
and are also how named planes (``sample``, ``VOA``, ``CCD``) are represented.
``Source`` is likewise nameable in prose but never required.
"""


def _element_pattern() -> "re.Pattern[str]":
	"""Build the element-name regex from the current vocabulary.

	Returns
	-------
	re.Pattern[str]
		Compiled pattern with ``section``, ``type``, ``index`` and ``axis``
		groups.
	"""

	sections = "".join(sorted(SECTION_CODES))
	types = "".join(sorted(ELEMENT_TYPE_CODES))
	axes = "|".join(sorted(AXIS_SUFFIXES))
	return re.compile(
		# CL1, PL4, PV1, DV, and a winding: CD1a, PSa (PD1_a still parses)
		rf"^(?P<section>[{sections}])"
		rf"(?P<type>[{types}])"
		r"(?P<index>\d+)?"
		rf"(?:_?(?P<axis>{axes}))?$"
	)


_ELEMENT_RE = _element_pattern()

# Classes that are schema-typed, i.e. whose names must follow the pattern when
# they sit inside a schema-managed section. Rebuilt by register_element_type.
_TYPED_CLASSES: set[str] = set(NAME_REQUIRED_CLASSES)


#: The retired attached form (``CL1_D1a``): recognised only to say what to write.
_ATTACHED_RE = re.compile(r"^[A-Z][A-Z]\d*_(?P<type>[A-Z])(?P<index>\d+)?(?P<axis>[a-z])?$")


class ElementName(NamedTuple):
	"""A parsed schema-conforming element name.

	Attributes
	----------
	section : str
		Section code, one of :data:`SECTION_CODES`.
	type : str
		Element type code, one of :data:`ELEMENT_TYPE_CODES`.
	index : int or None
		Ordinal within the section, or ``None`` when the name carries no index
		(as in ``"PSa"``).
	axis : str or None
		Axis/component suffix (the winding ``a`` of ``CD1a``), or ``None``.
	"""

	section: str
	type: str
	index: int | None
	axis: str | None


class SchemaIssue(NamedTuple):
	"""A single schema finding.

	Attributes
	----------
	severity : str
		``"error"`` for a violation, ``"info"`` for a note that does not fail
		validation (an un-managed section, an unnamed element).
	path : str
		Where the issue was found, e.g. ``"condensers/condenser_aperture"``.
	message : str
		Human-readable description, including the expected form where known.
	"""

	severity: str
	path: str
	message: str

	def __str__(self) -> str:
		"""Return ``severity: path -- message``."""

		return f"{self.severity}: {self.path} -- {self.message}"


class SchemaError(ValueError):
	"""Raised by a strict validation that found at least one error."""


class SchemaReport:
	"""Result of validating a section or microscope against the schema.

	Parameters
	----------
	issues : Iterable[SchemaIssue], optional
		Findings collected during validation, by default ().

	Attributes
	----------
	issues : list[SchemaIssue]
		Every finding, in traversal order.
	"""

	def __init__(self, issues: Iterable[SchemaIssue] = ()) -> None:
		"""Initialize a report from an iterable of issues."""

		self.issues: list[SchemaIssue] = list(issues)

	@property
	def errors(self) -> list[SchemaIssue]:
		"""list[SchemaIssue]: Only the error-severity findings."""

		return [i for i in self.issues if i.severity == "error"]

	@property
	def infos(self) -> list[SchemaIssue]:
		"""list[SchemaIssue]: Only the informational findings."""

		return [i for i in self.issues if i.severity == "info"]

	@property
	def ok(self) -> bool:
		"""bool: ``True`` when no error-severity finding was recorded."""

		return not self.errors

	def __bool__(self) -> bool:
		"""Return :attr:`ok`, so ``if report:`` reads as "conforms"."""

		return self.ok

	def __iter__(self) -> Iterator[SchemaIssue]:
		"""Iterate over every finding."""

		return iter(self.issues)

	def __len__(self) -> int:
		"""Return the number of findings."""

		return len(self.issues)

	def __repr__(self) -> str:
		"""Return a one-line summary."""

		return f"<SchemaReport {len(self.errors)} error(s), {len(self.infos)} info>"

	def __str__(self) -> str:
		"""Return the findings one per line, errors first."""

		if not self.issues:
			return "schema OK (no findings)"
		lines = [str(i) for i in self.errors] + [str(i) for i in self.infos]
		head = f"schema: {len(self.errors)} error(s), {len(self.infos)} info"
		return "\n".join([head, *lines])

	def raise_for_errors(self) -> "SchemaReport":
		"""Raise :class:`SchemaError` when any error-severity finding exists.

		Returns
		-------
		SchemaReport
			This report, when it is clean.

		Raises
		------
		SchemaError
			If :attr:`errors` is non-empty.
		"""

		if self.errors:
			raise SchemaError(str(self))
		return self


def _resolve(report: SchemaReport, strict: "bool | str") -> SchemaReport:
	"""Apply the ``strict`` policy to a finished report.

	Parameters
	----------
	report : SchemaReport
		Collected findings.
	strict : bool or str
		``True`` raises on errors, ``"warn"`` emits a :class:`UserWarning`,
		``False`` returns the report silently.

	Returns
	-------
	SchemaReport
		The same report.

	Raises
	------
	SchemaError
		If ``strict`` is ``True`` and errors were found.
	ValueError
		If ``strict`` is not ``True``, ``False`` or ``"warn"``.
	"""

	if strict is True:
		return report.raise_for_errors()
	if strict is False:
		return report
	if isinstance(strict, str) and strict.lower() == "warn":
		if report.errors:
			warnings.warn(str(report), UserWarning, stacklevel=3)
		return report
	raise ValueError(f"strict must be True, False or 'warn'; got {strict!r}")


def section_code(name: "str | None") -> "str | None":
	"""Resolve a section name to its schema code.

	Parameters
	----------
	name : str or None
		Section name, either a bare code (``"C"``), a canonical name
		(``"condenser"``) or a known alias (``"CLs"``, ``"condensers"``).

	Returns
	-------
	str or None
		The single-letter code, or ``None`` when the name is not recognised
		and the section is therefore not schema-managed.

	Examples
	--------
	>>> section_code("CLs"), section_code("C"), section_code("sec1")
	('C', 'C', None)
	"""

	if not name:
		return None
	text = str(name).strip()
	if text.upper() in SECTION_CODES:
		return text.upper()
	return SECTION_ALIASES.get(text.lower())


def parse_element_name(name: "str | None") -> "ElementName | None":
	"""Parse an element name into its schema parts.

	Parameters
	----------
	name : str or None
		Element name such as ``"CL1"``, ``"CD1a"`` or ``"PSa"``.

	Returns
	-------
	ElementName or None
		The parsed parts, or ``None`` when ``name`` does not follow the
		pattern.

	Examples
	--------
	>>> parse_element_name("OL2")
	ElementName(section='O', type='L', index=2, axis=None)
	>>> parse_element_name("sample") is None
	True
	"""

	if not name:
		return None
	match = _ELEMENT_RE.match(str(name).strip())
	if match is None:
		return None
	index = match.group("index")
	return ElementName(
		section=match.group("section"),
		type=match.group("type"),
		index=int(index) if index is not None else None,
		axis=match.group("axis"),
	)


def canonical_element_name(
	section: str,
	type: str,
	index: "int | None" = None,
	axis: "str | None" = None,
) -> str:
	"""Build a schema-conforming element name.

	Parameters
	----------
	section : str
		Section code or a name resolvable by :func:`section_code`.
	type : str
		Element type code, one of :data:`ELEMENT_TYPE_CODES`.
	index : int or None, optional
		Ordinal within the section, by default None.
	axis : str or None, optional
		Axis suffix (a winding), one of :data:`AXIS_SUFFIXES`, by default None.

	Returns
	-------
	str
		The assembled name, e.g. ``"CL1"``, ``"CD1a"`` or ``"PSa"``.

	Raises
	------
	ValueError
		If ``section``, ``type`` or ``axis`` is outside the vocabulary.

	Examples
	--------
	>>> canonical_element_name("condensers", "L", 3)
	'CL3'
	>>> canonical_element_name("C", "D", 1, "a")
	'CD1a'
	"""

	code = section_code(section)
	if code is None:
		raise ValueError(
			f"unknown section {section!r}; expected one of {sorted(SECTION_CODES)}"
		)
	letter = str(type).upper()
	if letter not in ELEMENT_TYPE_CODES:
		raise ValueError(
			f"unknown element type {type!r}; expected one of {sorted(ELEMENT_TYPE_CODES)}"
		)
	name = f"{code}{letter}"
	if index is not None:
		name += str(int(index))
	if axis is not None:
		suffix = str(axis).lower()
		if suffix not in AXIS_SUFFIXES:
			raise ValueError(
				f"unknown axis {axis!r}; expected one of {list(AXIS_SUFFIXES)}"
			)
		name += suffix
	return name


def _mro_names(obj: object) -> tuple[str, ...]:
	"""Return the class names in an object's MRO.

	Parameters
	----------
	obj : object
		Element instance or class.

	Returns
	-------
	tuple[str, ...]
		Names of every class in the MRO, most derived first.
	"""

	cls = obj if isinstance(obj, type) else type(obj)
	return tuple(c.__name__ for c in cls.__mro__)


def element_type_codes_for(element: object) -> tuple[str, ...]:
	"""Return the type codes an element's class may legitimately carry.

	Parameters
	----------
	element : object
		Element instance or class.

	Returns
	-------
	tuple[str, ...]
		Permitted type codes, empty when the class is not schema-typed (for
		example ``Drift``, ``Source`` or a plain ``Element``).

	Examples
	--------
	>>> from pySEA.rayTEM.elements import Lens, Source
	>>> element_type_codes_for(Lens), element_type_codes_for(Source)
	(('L',), ())
	"""

	names = set(_mro_names(element))
	return tuple(
		code
		for code, classes in sorted(ELEMENT_TYPE_CLASSES.items())
		if names.intersection(classes)
	)


def is_schema_typed(element: object) -> bool:
	"""Whether an element's class is subject to the name pattern.

	Parameters
	----------
	element : object
		Element instance or class.

	Returns
	-------
	bool
		``True`` when the class (or a base) appears in
		:data:`ELEMENT_TYPE_CLASSES`.
	"""

	return bool(set(_mro_names(element)).intersection(_TYPED_CLASSES))


def validate_element_name(
	name: "str | None",
	section: "str | None" = None,
	element: object = None,
	path: "str | None" = None,
) -> list[SchemaIssue]:
	"""Check one element name against the schema.

	Parameters
	----------
	name : str or None
		The element name to check.
	section : str or None, optional
		Owning section name or code. When given, the parsed section letter
		must match it, by default None.
	element : object, optional
		The element instance, used to check the type letter against its class,
		by default None.
	path : str or None, optional
		Display path for any issue raised, by default the name itself.

	Returns
	-------
	list[SchemaIssue]
		Findings for this name; empty when it conforms.

	Examples
	--------
	>>> [i.severity for i in validate_element_name("C1", section="C")]
	['error']
	>>> validate_element_name("CL1", section="condensers")
	[]
	"""

	where = path or (name or "<unnamed>")
	parsed = parse_element_name(name)
	if parsed is None:
		attached = _ATTACHED_RE.match(str(name or "").strip())
		code = section_code(section) if section is not None else None
		if attached is not None and code is not None:
			expected = f"{code}{attached['type']}{attached['index'] or ''}{attached['axis'] or ''}"
			return [
				SchemaIssue(
					"error",
					where,
					f"name {name!r} names a multipole by the element it sits by; "
					f"it is named on its own in its section, e.g. {expected!r}",
				)
			]
		return [
			SchemaIssue(
				"error",
				where,
				f"name {name!r} does not match <SECTION><TYPE>[INDEX][AXIS], "
				f"e.g. 'CL1' or 'CD1a'",
			)
		]

	issues: list[SchemaIssue] = []
	code = section_code(section) if section is not None else None
	if code is not None and parsed.section != code:
		expected = canonical_element_name(code, parsed.type, parsed.index, parsed.axis)
		issues.append(
			SchemaIssue(
				"error",
				where,
				f"section letter {parsed.section!r} does not match owning section "
				f"{code!r} ({SECTION_CODES[code]}); expected {expected!r}",
			)
		)

	if element is not None:
		allowed = element_type_codes_for(element)
		effective = parsed.type
		if allowed and effective not in allowed:
			cls_name = _mro_names(element)[0]
			issues.append(
				SchemaIssue(
					"error",
					where,
					f"type letter {effective!r} ({ELEMENT_TYPE_CODES[effective]}) "
					f"does not match class {cls_name}; expected one of {list(allowed)}",
				)
			)
	return issues


def _iter_named(items: Sequence[object]) -> Iterator[tuple[int, object, str]]:
	"""Yield ``(index, item, name)`` for items carrying a non-empty name.

	Parameters
	----------
	items : Sequence[object]
		Elements or sections.

	Yields
	------
	tuple[int, object, str]
		Position, object and its stripped name.
	"""

	for i, item in enumerate(items or []):
		name = getattr(item, "name", None)
		if name:
			yield i, item, str(name).strip()


def _duplicate_issues(
	items: Sequence[object], prefix: str, what: str
) -> list[SchemaIssue]:
	"""Report names used more than once among ``items``.

	Parameters
	----------
	items : Sequence[object]
		Elements or sections sharing a namespace.
	prefix : str
		Path prefix for the reported issues.
	what : str
		Noun used in the message, e.g. ``"element"``.

	Returns
	-------
	list[SchemaIssue]
		One error per duplicated name.
	"""

	seen: dict[str, int] = {}
	for _, _, name in _iter_named(items):
		seen[name] = seen.get(name, 0) + 1
	return [
		SchemaIssue(
			"error",
			f"{prefix}{name}",
			f"duplicate {what} name {name!r} ({count}x); name lookup is ambiguous",
		)
		for name, count in seen.items()
		if count > 1
	]


def _cross_section_duplicates(sections: Sequence[object]) -> list[SchemaIssue]:
	"""Report element names reused across two or more sections.

	Names duplicated *within* one section are reported by
	:func:`validate_section`; this covers only the column-wide ambiguity that
	breaks ``microscope["name"]`` lookup.

	Parameters
	----------
	sections : Sequence[object]
		Sections of one microscope.

	Returns
	-------
	list[SchemaIssue]
		One error per name appearing in more than one section.
	"""

	owners: dict[str, list[str]] = {}
	for i, section in enumerate(sections or []):
		label = str(getattr(section, "name", "") or "").strip() or f"<section {i}>"
		for _, _, name in _iter_named(getattr(section, "elements", None) or []):
			if label not in owners.setdefault(name, []):
				owners[name].append(label)
	return [
		SchemaIssue(
			"error",
			name,
			f"element name {name!r} is used in {len(labels)} sections "
			f"({', '.join(labels)}); column-wide lookup is ambiguous",
		)
		for name, labels in owners.items()
		if len(labels) > 1
	]


def validate_section(
	section: object,
	code: "str | None" = None,
	strict: "bool | str" = True,
	prefix: str = "",
) -> SchemaReport:
	"""Validate one microscope section against the schema.

	Parameters
	----------
	section : object
		A :class:`~pySEA.rayTEM.assemblies.MicroscopeSection`, or anything
		with ``name`` and ``elements``.
	code : str or None, optional
		Override the section code instead of deriving it from the name, by
		default None.
	strict : bool or str, optional
		``True`` raises :class:`SchemaError` on any error, ``"warn"`` warns,
		``False`` reports silently, by default True.
	prefix : str, optional
		Path prefix used when called from :func:`validate_microscope`, by
		default "".

	Returns
	-------
	SchemaReport
		Findings for this section.

	Raises
	------
	SchemaError
		If ``strict`` is ``True`` and the section does not conform.
	"""

	issues: list[SchemaIssue] = []
	name = str(getattr(section, "name", "") or "").strip()
	elements = list(getattr(section, "elements", None) or [])
	resolved = code if code is not None else section_code(name)
	where = prefix or (name or "<unnamed section>")

	issues.extend(_duplicate_issues(elements, f"{where}/", "element"))

	if resolved is None:
		issues.append(
			SchemaIssue(
				"info",
				where,
				f"section name {name!r} is not a schema code; elements not "
				f"name-checked (codes: {sorted(SECTION_CODES)})",
			)
		)
		return _resolve(SchemaReport(issues), strict)

	last_index: dict[str, tuple[int, str]] = {}
	for element in elements:
		element_name = getattr(element, "name", None)
		required = is_schema_typed(element)
		if not element_name:
			if required:
				issues.append(
					SchemaIssue(
						"info",
						f"{where}/<unnamed {_mro_names(element)[0]}>",
						"schema-typed element has no name; cannot be referenced by name",
					)
				)
			continue
		if not required and parse_element_name(element_name) is None:
			# a named plane (sample, VOA, CCD) on a class the schema never requires
			continue
		issues.extend(
			validate_element_name(
				element_name,
				section=resolved,
				element=element,
				path=f"{where}/{element_name}",
			)
		)
		# Numbers run with the column: a gap is fine, going back is not.
		parsed = parse_element_name(element_name)
		if parsed is not None and parsed.index is not None:
			before = last_index.get(parsed.type)
			if before is not None and parsed.index < before[0]:
				issues.append(
					SchemaIssue(
						"error",
						f"{where}/{element_name}",
						f"{element_name!r} comes after {before[1]!r} in the column; "
						f"{ELEMENT_TYPE_CODES[parsed.type]} numbers run in column order",
					)
				)
			else:
				last_index[parsed.type] = (parsed.index, element_name)
	return _resolve(SchemaReport(issues), strict)


def validate_microscope(scope: object, strict: "bool | str" = True) -> SchemaReport:
	"""Validate a whole microscope against the column naming schema.

	Walks every section, resolves its code, and name-checks the schema-typed
	elements inside the sections that opt in. Also reports duplicate section
	names and element names reused across sections, since both make name
	lookup ambiguous.

	Parameters
	----------
	scope : object
		A :class:`~pySEA.rayTEM.assemblies.Microscope`, or anything with a
		``sections`` sequence.
	strict : bool or str, optional
		``True`` raises :class:`SchemaError` on any error, ``"warn"`` emits a
		:class:`UserWarning`, ``False`` reports silently, by default True.

	Returns
	-------
	SchemaReport
		Every finding, in column order.

	Raises
	------
	SchemaError
		If ``strict`` is ``True`` and the microscope does not conform.

	Examples
	--------
	>>> report = validate_microscope(scope, strict=False)  # doctest: +SKIP
	>>> print(report)  # doctest: +SKIP
	"""

	sections = list(getattr(scope, "sections", None) or [])
	issues: list[SchemaIssue] = list(_duplicate_issues(sections, "", "section"))

	for i, section in enumerate(sections):
		name = str(getattr(section, "name", "") or "").strip()
		where = name or f"<section {i}>"
		issues.extend(validate_section(section, strict=False, prefix=where).issues)

	issues.extend(_cross_section_duplicates(sections))
	return _resolve(SchemaReport(issues), strict)


def migrate_legacy_section(section: object) -> bool:
	"""Rename a section saved under a retired code, as it is read.

	The spectrometer was ``S`` until 2026-10-21 and is ``E`` now. A section
	named ``S`` becomes ``E``, and its elements named on the old code
	(``SP1``) follow (``EP1``). Nothing else is touched.

	Parameters
	----------
	section : object
		A :class:`~pySEA.rayTEM.assemblies.MicroscopeSection`, or anything
		with ``name`` and ``elements``.

	Returns
	-------
	bool
		Whether anything was renamed.

	Examples
	--------
	>>> class _Section:
	...     name, elements = "S", []
	>>> section = _Section()
	>>> migrate_legacy_section(section), section.name
	(True, 'E')
	"""

	if str(getattr(section, "name", "") or "").strip() != "S":
		return False
	section.name = "E"
	for element in getattr(section, "elements", None) or []:
		name = str(getattr(element, "name", "") or "")
		if re.match(r"^S[A-Z]", name):
			element.name = "E" + name[1:]
	return True


def register_section(
	code: str,
	name: str,
	aliases: Iterable[str] = (),
) -> None:
	"""Add a section code to the vocabulary.

	Use for column regions this schema does not ship, such as an aberration
	corrector, so a site-specific scope can validate strictly without patching
	the module.

	Parameters
	----------
	code : str
		Single letter, not already in use.
	name : str
		Canonical section name.
	aliases : Iterable[str], optional
		Additional free-form names resolving to this code, by default ().

	Raises
	------
	ValueError
		If ``code`` is not a single letter or is already registered.
	"""

	global _ELEMENT_RE
	letter = str(code).upper()
	if len(letter) != 1 or not letter.isalpha():
		raise ValueError(f"section code must be a single letter; got {code!r}")
	if letter in SECTION_CODES:
		raise ValueError(
			f"section code {letter!r} is already {SECTION_CODES[letter]!r}"
		)
	SECTION_CODES[letter] = name
	SECTION_ALIASES[name.lower()] = letter
	for alias in aliases:
		SECTION_ALIASES[str(alias).lower()] = letter
	_ELEMENT_RE = _element_pattern()


def register_element_type(
	code: str, description: str, classes: Iterable[str], required: bool = True
) -> None:
	"""Add an element type code to the vocabulary.

	Parameters
	----------
	code : str
		Single letter, not already in use.
	description : str
		Human-readable meaning, e.g. ``"octupole"``.
	classes : Iterable[str]
		rayTEM class names this code may name. Matched against the whole MRO.
	required : bool, optional
		Whether those classes must then carry a schema name inside a managed
		section. Pass ``False`` for a code that only *permits* a name, the way
		``V`` does for ``Drift``, by default True.

	Raises
	------
	ValueError
		If ``code`` is not a single letter or is already registered.
	"""

	global _ELEMENT_RE, _TYPED_CLASSES
	letter = str(code).upper()
	if len(letter) != 1 or not letter.isalpha():
		raise ValueError(f"element type code must be a single letter; got {code!r}")
	if letter in ELEMENT_TYPE_CODES:
		raise ValueError(
			f"element type {letter!r} is already {ELEMENT_TYPE_CODES[letter]!r}"
		)
	ELEMENT_TYPE_CODES[letter] = description
	ELEMENT_TYPE_CLASSES[letter] = tuple(classes)
	if required:
		NAME_REQUIRED_CLASSES.update(ELEMENT_TYPE_CLASSES[letter])
	_TYPED_CLASSES = set(NAME_REQUIRED_CLASSES)
	_ELEMENT_RE = _element_pattern()


def describe_schema() -> str:
	"""Return a printable summary of the current vocabulary.

	Intended for builders and agents to print (or paste into a prompt) before
	constructing a microscope, so element names are chosen from the live
	vocabulary rather than from memory.

	Returns
	-------
	str
		Multi-line description of sections, element types and the name
		pattern.
	"""

	lines = ["rayTEM column naming schema", "", "Sections:"]
	for code, name in sorted(SECTION_CODES.items()):
		lines.append(f"  {code}  {name}")
	lines += ["", "Element types:"]
	for code, desc in sorted(ELEMENT_TYPE_CODES.items()):
		classes = "/".join(ELEMENT_TYPE_CLASSES.get(code, ()))
		lines.append(f"  {code}  {desc:<20} {classes}")
	lines += [
		"",
		"Element:  <SECTION><TYPE>[INDEX][AXIS]",
		"  e.g. CL1 (first condenser lens), PV1 (projector drift volume 1),",
		"       CA1 (condenser aperture 1), EP1 (spectrometer prism),",
		"       CD1a (winding a of the condenser's first dipole)",
		f"  axes: {', '.join(AXIS_SUFFIXES)}",
		"  Multipoles are named on their own in their section: CD1a, not CL1_D1a.",
		"  Numbers run in column order within a section; gaps are allowed.",
		"",
		"Section name: a code letter, a canonical name, or a registered alias.",
		"A section whose name resolves to no code is not schema-managed.",
		"Drift, Source and plain Element objects are never name-checked.",
	]
	return "\n".join(lines)
