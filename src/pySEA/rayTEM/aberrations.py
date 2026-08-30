"""Axial wave aberrations in Krivanek notation, and how to apply them.

One class, :class:`Aberrations`, holds a set of coefficients and knows how to
turn them into a phase (for wave optics) or a ray deflection (for geometric
optics). Nothing else in rayTEM needs to know *which* aberration is present:
propagation asks for the phase or its gradient and applies whatever is there.

The storage convention is **Krivanek** ``C_{n,m}``, matching what sea-eco's
swift reader already pulls out of a Nion file, so a measured instrument and a
simulated one speak the same names.
"""
import numpy as np

from .seashells import SEASerializable


def krivanek_terms(max_order: int = 5) -> dict:
	r"""The Krivanek ``C_{n,m}`` term names up to a given order.

	For an aberration of order ``n`` the wave phase goes as
	:math:`\theta^{n+1}` and the transverse ray aberration as
	:math:`\theta^{n}`. The azimuthal multiplicity ``m`` runs from ``n + 1``
	down to 0 or 1 in steps of two, so a term is either rotationally symmetric
	(``m = 0``) or has ``m``-fold symmetry.

	=====  ====================================
	order  terms
	=====  ====================================
	1      ``C10`` defocus, ``C12`` twofold
	2      ``C21`` coma, ``C23`` threefold
	3      ``C30`` spherical, ``C32``, ``C34``
	4      ``C41``, ``C43``, ``C45``
	5      ``C50``, ``C52``, ``C54``, ``C56``
	=====  ====================================

	Parameters
	----------
	max_order : int, optional
		Highest order to include, by default 5.

	Returns
	-------
	dict
		``{name: (n, m)}`` in ascending order.

	Raises
	------
	ValueError
		If ``max_order`` is less than 1.

	Related
	-------
	Aberrations : Holds coefficients keyed by these names.

	Notes
	-----
	sea-eco's swift reader currently stops at order 4; fifth order is included
	here so a simulation is not limited by what a file happens to carry.

	Examples
	--------
	>>> sorted(krivanek_terms(1))
	['C10', 'C12']
	"""
	if max_order < 1:
		raise ValueError(f"max_order must be at least 1, got {max_order}.")
	out = {}
	for n in range(1, int(max_order) + 1):
		for m in range(n + 1, -1, -2):
			out[f"C{n}{m}"] = (n, m)
	return out


#: Krivanek ``C_{n,m}`` name -> ``(order, multiplicity)``, orders 1 through 5.
KRIVANEK_TERMS = krivanek_terms(5)

#: Name of the chromatic coefficient inside an :class:`Aberrations` set. It is
#: deliberately NOT a Krivanek term: every ``C_{n,m}`` is a function of pupil
#: coordinate alone, whereas chromatic multiplies the pupil coordinate by the
#: beam's *energy* deviation. It lives in the same object so that one
#: declaration carries everything an element does beyond its matrix -- and so
#: it survives serialization, gets detached by ``suspended_aberrations``, and
#: cannot be forgotten -- but it is kept out of ``names``/``items()`` so the
#: Krivanek machinery never sees a term it cannot interpret.
CHROMATIC_TERM = 'Cc'

#: The letter convention (Nion/CEOS style) mapped onto Krivanek names. These
#: are *different notations for the same quantities*, which is why they get a
#: conversion rather than a second storage format.
LETTER_TO_KRIVANEK = {
	'C1': 'C10', 'A1': 'C12',
	'B2': 'C21', 'A2': 'C23',
	'C3': 'C30', 'S3': 'C32', 'A3': 'C34',
	'B4': 'C41', 'D4': 'C43', 'A4': 'C45',
	'C5': 'C50', 'S5': 'C52', 'R5': 'C54', 'A5': 'C56',
}


class Aberrations(SEASerializable):
	r"""A set of axial aberration coefficients, and the phase they produce.

	Coefficients are **complex**: for a term of multiplicity ``m`` the two
	components fix both size and orientation, exactly as a Nion file's
	``C12.a`` / ``C12.b`` pair does. The contribution to the phase is

	.. math::

		\frac{\theta^{n+1}}{n+1}\,\big(a\cos m\phi + b\sin m\phi\big),
		\qquad C_{n,m} = a + ib

	so ``a`` is the component aligned with the x axis and ``b`` the one at
	:math:`45^\circ/m`. Rotationally symmetric terms (``m = 0``) are real; a
	nonzero imaginary part on one is rejected rather than silently ignored.

	Attributes
	----------
	convention : str
		Which notation the stored names are in; ``'krivanek'``. Recorded so a
		bare set of numbers is never ambiguous.
	names : list of str
		Term names, e.g. ``['C10', 'C30']``.
	real, imag : list of float
		The two components of each coefficient, in metres. Stored as parallel
		lists rather than a dict or complex array because those are what a
		``.sea`` file can carry.

	Methods
	-------
	phase_at(X, Y, wavelength, power)
		The wave aberration function ``chi`` at given coordinates.
	deflection_at(X, Y, power)
		The transverse ray aberration ``(1/k) grad chi``, which is what ray
		optics needs; wavelength-free.
	phase(shape, dx, dy, wavelength, power)
		``phase_at`` on a sampled grid, for the wave path.
	gradient(shape, dx, dy, wavelength, power)
		``(d chi/dx, d chi/dy)`` on a sampled grid.
	setdefault(name, value)
		Set a coefficient only if it is not already nonzero.
	as_dict()
		The nonzero coefficients in the flat, JSON-safe ``a``/``b`` form.
	from_metadata(metadata)
		Read that same form back, as sea-eco's swift reader also writes it.
	from_letters(coefficients)
		Build from the ``C1``/``A1``/``B2`` letter notation.
	to_letters()
		Express the stored set in that notation.

	Raises
	------
	KeyError
		On an unknown term name.
	ValueError
		On an orientation given for a rotationally symmetric term.

	Related
	-------
	KRIVANEK_TERMS : The names and their ``(order, multiplicity)``.
	elements.Element.aberrations : Where one of these is attached.

	Notes
	-----
	These are **axial** aberrations: functions of pupil position alone, with no
	field dependence.

	Examples
	--------
	>>> ab = Aberrations({'C30': 1e-3})
	>>> ab['C30']
	(0.001+0j)
	"""

	def __init__(self, coefficients=None, convention: str = 'krivanek',
				 chromatic: float = 0.0):
		"""Create a coefficient set.

		Parameters
		----------
		coefficients : dict, optional
			``{name: value}`` where the value is a real number, a complex
			number, or an ``(a, b)`` pair. ``None`` (default) is an empty set,
			which is a perfect lens. The name ``'Cc'`` is accepted here and
			routed to ``chromatic``.
		convention : str, optional
			Notation of the names, by default ``'krivanek'``.
		chromatic : float, optional
			Chromatic aberration coefficient :math:`C_c` in metres, by default
			0 (achromatic).

		Raises
		------
		KeyError
			If a name is not a known Krivanek term (or ``'Cc'``).
		ValueError
			If a rotationally symmetric term is given a nonzero second
			component, or ``'Cc'`` is given a complex value.
		"""
		self.convention = str(convention)
		self.names = []
		self.real = []
		self.imag = []
		self.chromatic = float(chromatic)
		for name, value in (coefficients or {}).items():
			self[name] = value

	# ------------------------------------------------------------------ core
	@staticmethod
	def order_and_multiplicity(name: str) -> tuple:
		"""Look up ``(n, m)`` for a term name.

		Parameters
		----------
		name : str
			Krivanek term name, e.g. ``'C30'``.

		Returns
		-------
		tuple of int
			``(order, multiplicity)``.

		Raises
		------
		KeyError
			If the name is not a known term.

		Related
		-------
		KRIVANEK_TERMS : The table consulted.
		"""
		try:
			return KRIVANEK_TERMS[name]
		except KeyError:
			raise KeyError(f"{name!r} is not a Krivanek term this class knows; "
						   f"expected one of {sorted(KRIVANEK_TERMS)}.") from None

	def __setitem__(self, name: str, value) -> None:
		"""Set one coefficient.

		Parameters
		----------
		name : str
			Krivanek term name.
		value : float, complex, or Sequence[float]
			Magnitude, complex coefficient, or ``(a, b)`` pair.

		Returns
		-------
		None

		Raises
		------
		KeyError
			If the name is unknown.
		ValueError
			If a nonzero ``b`` is given for an ``m = 0`` term, or the value is
			not a scalar or a pair.
		"""
		if name == CHROMATIC_TERM:
			if np.ndim(value) != 0 or complex(value).imag:
				raise ValueError(f"{CHROMATIC_TERM!r} is a single real coefficient in "
								 f"metres; it has no orientation, got {value!r}.")
			self.chromatic = float(complex(value).real)
			return
		n, m = self.order_and_multiplicity(name)
		if np.ndim(value) == 0:
			a, b = complex(value).real, complex(value).imag
		else:
			try:
				a, b = (float(v) for v in value)
			except (TypeError, ValueError):
				raise ValueError(f"coefficient {name!r} must be a number or an "
								 f"(a, b) pair, got {value!r}.") from None
		if m == 0 and b:
			raise ValueError(f"{name!r} is rotationally symmetric (m = 0), so it has "
							 f"no orientation; its second component must be 0, got {b}.")
		if name in self.names:
			i = self.names.index(name)
			self.real[i], self.imag[i] = float(a), float(b)
		else:
			self.names.append(name)
			self.real.append(float(a))
			self.imag.append(float(b))

	def __getitem__(self, name: str) -> complex:
		"""Return one coefficient as a complex number.

		Parameters
		----------
		name : str
			Krivanek term name.

		Returns
		-------
		complex
			``a + ib``; zero when the term is not set.

		Raises
		------
		KeyError
			If the name is not a known term.
		"""
		if name == CHROMATIC_TERM:
			return complex(self.chromatic, 0.0)
		self.order_and_multiplicity(name)
		if name not in self.names:
			return 0j
		i = self.names.index(name)
		return complex(self.real[i], self.imag[i])

	def __contains__(self, name: str) -> bool:
		"""Whether a term is set to a nonzero value.

		Parameters
		----------
		name : str
			Krivanek term name.

		Returns
		-------
		bool
			True if present and nonzero.

		Raises
		------
		None
		"""
		if name == CHROMATIC_TERM:
			return bool(self.chromatic)
		return name in self.names and self[name] != 0

	def __bool__(self) -> bool:
		"""Whether any coefficient is nonzero.

		Returns
		-------
		bool
			False for a perfect lens, so ``if lens.aberrations:`` reads
			naturally.

		Raises
		------
		None
		"""
		return bool(self.chromatic) or any(a or b for a, b in zip(self.real, self.imag))

	def __repr__(self) -> str:
		"""Compact, readable summary.

		Returns
		-------
		str
			The nonzero terms and the convention.

		Raises
		------
		None
		"""
		terms = [f"{n}={self[n]:g}" for n in self.names if self[n] != 0]
		if self.chromatic:
			terms.append(f"{CHROMATIC_TERM}={self.chromatic:g}")
		body = ", ".join(terms)
		return f"Aberrations({body or 'ideal'}, convention={self.convention!r})"

	def items(self):
		"""Iterate ``(name, complex)`` over the nonzero terms.

		Returns
		-------
		list of tuple
			``[(name, coefficient), ...]`` in insertion order.

		Raises
		------
		None

		Related
		-------
		as_dict : The same content as a mapping.
		"""
		return [(n, self[n]) for n in self.names if self[n] != 0]

	def setdefault(self, name: str, value) -> complex:
		"""Set a coefficient only if it is not already nonzero.

		Used where a default must not silently overwrite a term the caller
		named — an explicitly given coefficient wins, because naming the term
		is the more specific statement.

		Parameters
		----------
		name : str
			Krivanek term name.
		value : float, complex, or Sequence[float]
			Value to use when the term is currently zero or unset.

		Returns
		-------
		complex
			The coefficient in force afterwards.

		Raises
		------
		KeyError
			If the name is unknown.
		ValueError
			If a nonzero orientation is given for an ``m = 0`` term.

		Related
		-------
		__setitem__ : The unconditional form.
		"""
		if name not in self:
			self[name] = value
		return self[name]

	def as_dict(self) -> dict:
		"""The nonzero coefficients in the flat ``name`` / ``name.a`` / ``name.b`` form.

		The storage form, and the only one this class exposes to the outside:
		rotationally symmetric terms as a single real number, oriented terms
		split into the ``a``/``b`` pair a Nion file uses, and the chromatic
		coefficient under :data:`CHROMATIC_TERM`. It is exactly what
		:meth:`from_metadata` reads, so the two round-trip, and it contains no
		complex numbers -- which is what lets a plain JSON writer carry an
		aberrated column.

		Returns
		-------
		dict
			``{name: float}`` and ``{name.a: float, name.b: float}`` entries
			for the nonzero terms, plus ``'Cc'`` when chromatic is nonzero.
			Empty for an ideal set.

		Raises
		------
		None

		Related
		-------
		from_metadata : Reads this form back.
		items : The same content as complex pairs, for doing arithmetic with.
		assemblies.Microscope.save : The JSON writer that needs this.

		Examples
		--------
		>>> Aberrations({'C30': 1e-3, 'C12': (2e-9, 3e-9)}).as_dict()
		{'C30': 0.001, 'C12.a': 2e-09, 'C12.b': 3e-09}
		"""
		out = {}
		for name, value in self.items():
			if value.imag:
				out[f"{name}.a"] = float(value.real)
				out[f"{name}.b"] = float(value.imag)
			else:
				out[name] = float(value.real)
		if self.chromatic:
			out[CHROMATIC_TERM] = float(self.chromatic)
		return out

	# --------------------------------------------------------- conversions
	@classmethod
	def from_letters(cls, coefficients: dict) -> 'Aberrations':
		"""Build from the letter notation (``C1``, ``A1``, ``B2``, ``S3``...).

		The letters are a different naming of the same quantities, common in
		corrector software. They are converted on the way in so there is only
		ever one stored convention.

		Parameters
		----------
		coefficients : dict
			``{letter_name: value}``; values as for :meth:`__setitem__`.

		Returns
		-------
		Aberrations
			The equivalent set in Krivanek names.

		Raises
		------
		KeyError
			If a letter name is not recognised.

		Related
		-------
		to_letters : The inverse.
		LETTER_TO_KRIVANEK : The mapping used.
		"""
		out = {}
		for name, value in (coefficients or {}).items():
			if name not in LETTER_TO_KRIVANEK:
				raise KeyError(f"{name!r} is not a letter-notation aberration; "
							   f"expected one of {sorted(LETTER_TO_KRIVANEK)}.")
			out[LETTER_TO_KRIVANEK[name]] = value
		return cls(out)

	def to_letters(self) -> dict:
		"""Express the stored set in the letter notation.

		Returns
		-------
		dict
			``{letter_name: complex}`` for the nonzero terms.

		Raises
		------
		None

		Related
		-------
		from_letters : The inverse.
		"""
		inverse = {v: k for k, v in LETTER_TO_KRIVANEK.items()}
		return {inverse[n]: c for n, c in self.items() if n in inverse}

	@classmethod
	def from_metadata(cls, metadata) -> 'Aberrations':
		r"""Read coefficients as sea-eco's swift reader stores them.

		A Nion file carries each term's two components under separate keys,
		``C12.a`` and ``C12.b``, with the rotationally symmetric ones (``C10``,
		``C30``) as a single value. This performs the ``a``/``b`` → complex
		step the reader itself leaves as a TODO.

		Accepts a sea-eco ``Metadata`` tree or a plain mapping, and looks for
		an ``Aberrations`` node anywhere within it, so either the whole
		metadata or just the aberration sub-tree can be handed in.

		Parameters
		----------
		metadata : Metadata or Mapping
			Metadata as loaded from an experimental file.

		Returns
		-------
		Aberrations
			The coefficients found; empty if there are none.

		Raises
		------
		None
			Unknown keys are ignored rather than raising, because instrument
			metadata legitimately carries entries this class does not model.

		Related
		-------
		sea_eco.io.swift_to_sea_metadata : Writes the structure read here.

		Examples
		--------
		>>> Aberrations.from_metadata({'C30': 1e-3, 'C12.a': 2e-9})  # doctest: +SKIP
		"""
		flat = _flatten_metadata(metadata)
		found = {}
		chromatic = 0.0
		for key, value in flat.items():
			stem, _, part = key.partition('.')
			if stem == CHROMATIC_TERM and value is not None:
				try:
					chromatic = float(value)
				except (TypeError, ValueError):
					pass
				continue
			if stem not in KRIVANEK_TERMS or value is None:
				continue
			try:
				value = float(value)
			except (TypeError, ValueError):
				continue
			a, b = found.get(stem, (0.0, 0.0))
			if part == 'b':
				found[stem] = (a, value)
			else:
				found[stem] = (value, b)
		return cls({k: v for k, v in found.items()}, chromatic=chromatic)

	# ------------------------------------------------------------- physics
	def _polar(self, X, Y, power: float) -> tuple:
		r"""Pupil angle and azimuth at given transverse coordinates.

		Shared by every evaluator so they cannot disagree about geometry.

		Parameters
		----------
		X, Y : np.ndarray or float
			Transverse coordinates in the element plane (metres). Any shape:
			a grid for a wavefield, a flat array for a ray table.
		power : float
			Focal power ``1/f``, converting ray height to pupil angle.

		Returns
		-------
		tuple
			``(theta, phi)``, same shape as the inputs.

		Raises
		------
		ValueError
			If ``power`` is zero, leaving the pupil angle undefined.

		Related
		-------
		deflection_at, phase_at : The consumers.
		"""
		if power == 0:
			raise ValueError("an aberration needs a nonzero focal power: the pupil "
							 "angle theta = r/f is undefined for a lens with no power.")
		X = np.asarray(X, dtype=float)
		Y = np.asarray(Y, dtype=float)
		return np.hypot(X, Y) * abs(power), np.arctan2(Y, X)

	def _grid(self, shape: tuple, dx: float, dy: float) -> tuple:
		"""Transverse coordinates of a sampled field.

		Parameters
		----------
		shape : tuple of int
			Field shape ``(ny, nx)``.
		dx, dy : float
			Sample spacings (metres).

		Returns
		-------
		tuple of np.ndarray
			``(X, Y)``, each shape ``(ny, nx)``.

		Raises
		------
		None

		Related
		-------
		waveoptics.transverse_coordinates : The centring convention used.
		"""
		from .waveoptics import transverse_coordinates
		return transverse_coordinates(shape, dx, dy)

	def phase_at(self, X, Y, wavelength: float, power: float) -> np.ndarray:
		r"""The wave aberration function :math:`\chi` at given coordinates.

		.. math::

			\chi = -k \sum_{n,m} \frac{\theta^{n+1}}{n+1}
			       \big(a_{nm}\cos m\phi + b_{nm}\sin m\phi\big)

		written on the element plane, where ray height is proportional to pupil
		angle (:math:`\theta = r/f`).

		The sign follows this package's convention, in which a converging lens
		carries :math:`\chi = -k r^2/2f` and the field is multiplied by
		:math:`e^{i\chi}` — the opposite sign to texts writing
		:math:`e^{-i\chi}`. With it, a positive ``C30`` focuses marginal rays
		short, as spherical aberration does.

		Parameters
		----------
		X, Y : np.ndarray or float
			Transverse coordinates (metres).
		wavelength : float
			Wavelength (metres).
		power : float
			Focal power ``1/f`` (1/metres).

		Returns
		-------
		np.ndarray
			Phase in radians, broadcast to the shape of ``X``; all zeros for an
			ideal set.

		Raises
		------
		ValueError
			If ``power`` is zero.

		Related
		-------
		phase : The same thing on a sampled grid.
		deflection_at : The ray-side counterpart.
		"""
		chi = np.zeros(np.shape(X), dtype=float)
		if not self.items():			# chromatic is not a pupil function; it never reaches here
			return chi
		theta, phi = self._polar(X, Y, power)
		k = 2 * np.pi / wavelength
		for name, c in self.items():
			n, m = KRIVANEK_TERMS[name]
			g = c.real * np.cos(m * phi) + c.imag * np.sin(m * phi) if m else c.real
			chi = chi - k * theta**(n + 1) / (n + 1) * g
		return chi

	def deflection_at(self, X, Y, power: float) -> tuple:
		r"""The transverse ray aberration at given coordinates, for every term.

		This is what geometric optics needs, and it is the *same* function the
		wave side uses: the extra angle a ray picks up is

		.. math::

			\Delta\theta = \frac{1}{k}\nabla\chi

		exact in the eikonal limit at **every** order, so one expression covers
		``C10`` through ``C56`` and no propagation code needs to know which
		aberration it is applying. Because :math:`\chi \propto k`, the ratio is
		wavelength-free — which is why this takes no wavelength and works on a
		ray table that carries none.

		With :math:`g(\phi) = a\cos m\phi + b\sin m\phi`:

		.. math::

			\Delta\theta_x = -P \theta^{n}
			   \Big(g\cos\phi - \frac{g'\sin\phi}{n+1}\Big), \quad
			\Delta\theta_y = -P \theta^{n}
			   \Big(g\sin\phi + \frac{g'\cos\phi}{n+1}\Big)

		Parameters
		----------
		X, Y : np.ndarray or float
			Transverse coordinates (metres).
		power : float
			Focal power ``1/f`` (1/metres).

		Returns
		-------
		tuple of np.ndarray
			``(delta_theta_x, delta_theta_y)`` in radians.

		Raises
		------
		ValueError
			If ``power`` is zero.

		Related
		-------
		phase_at : The function differentiated here.
		elements.Element._aberration_kick : The consumer on the ray path.

		Notes
		-----
		Analytic rather than a finite difference, so it is exact and costs no
		extra sampling. The expression is regular on axis: every term carries
		:math:`\theta^{n}` with :math:`n \ge 1`.
		"""
		zeros = np.zeros(np.shape(X), dtype=float)
		if not self.items():			# chromatic is not a pupil function; it never reaches here
			return zeros, zeros.copy()
		theta, phi = self._polar(X, Y, power)
		P = abs(power)
		gx, gy = zeros, zeros.copy()
		for name, c in self.items():
			n, m = KRIVANEK_TERMS[name]
			if m:
				g = c.real * np.cos(m * phi) + c.imag * np.sin(m * phi)
				dg = m * (-c.real * np.sin(m * phi) + c.imag * np.cos(m * phi))
			else:
				g, dg = c.real, 0.0
			common = -P * theta**n
			gx = gx + common * (g * np.cos(phi) - dg * np.sin(phi) / (n + 1))
			gy = gy + common * (g * np.sin(phi) + dg * np.cos(phi) / (n + 1))
		return gx, gy

	def phase(self, shape: tuple, dx: float, dy: float, wavelength: float,
			  power: float) -> np.ndarray:
		r"""The wave aberration function :math:`\chi` on a sampled grid.

		A thin wrapper over :meth:`phase_at` for the wave path, which works with
		a shape and a sampling rather than explicit coordinates.

		Parameters
		----------
		shape : tuple of int
			Field shape ``(ny, nx)``.
		dx, dy : float
			Sample spacings (metres, or physical spacings ``s·Δξ`` on a scaled
			grid).
		wavelength : float
			Wavelength (metres).
		power : float
			Focal power ``1/f`` (1/metres).

		Returns
		-------
		np.ndarray
			Real phase array in radians, shape ``(ny, nx)``.

		Raises
		------
		ValueError
			If ``power`` is zero.

		Related
		-------
		phase_at : The underlying evaluation.

		Examples
		--------
		>>> Aberrations({'C30': 1e-3}).phase((8, 8), 1e-6, 1e-6, 2.5e-12, 22.2)  # doctest: +SKIP
		"""
		if not self.items():			# chromatic is not a pupil function; it never reaches here
			return np.zeros(tuple(shape), dtype=float)
		X, Y = self._grid(shape, dx, dy)
		return self.phase_at(X, Y, wavelength, power)

	def gradient(self, shape: tuple, dx: float, dy: float, wavelength: float,
				 power: float) -> tuple:
		r"""``(∂χ/∂x, ∂χ/∂y)`` on a sampled grid, in radians per metre.

		The spatial gradient of :meth:`phase`, i.e. :math:`k` times
		:meth:`deflection_at`. Provided for code that wants the phase slope
		itself; ray propagation should use :meth:`deflection_at`, which needs no
		wavelength.

		Parameters
		----------
		shape : tuple of int
			Field shape ``(ny, nx)``.
		dx, dy : float
			Sample spacings (metres).
		wavelength : float
			Wavelength (metres).
		power : float
			Focal power ``1/f`` (1/metres).

		Returns
		-------
		tuple of np.ndarray
			``(dchi_dx, dchi_dy)``.

		Raises
		------
		ValueError
			If ``power`` is zero.

		Related
		-------
		deflection_at : The wavelength-free ray form.
		"""
		if not self.items():			# chromatic is not a pupil function; it never reaches here
			zeros = np.zeros(tuple(shape), dtype=float)
			return zeros, zeros.copy()
		X, Y = self._grid(shape, dx, dy)
		gx, gy = self.deflection_at(X, Y, power)
		k = 2 * np.pi / wavelength
		return k * gx, k * gy

def _flatten_metadata(metadata, prefix: str = "") -> dict:
	"""Flatten a metadata tree to ``{key: value}``, ignoring the hierarchy.

	Instrument metadata nests aberrations under different paths depending on
	the source (``Instrument/Condensers/Aberrations`` for one detector,
	``Instrument/Aberrations`` for another), so :meth:`Aberrations.from_metadata`
	searches by leaf name rather than by path.

	Parameters
	----------
	metadata : Metadata, Mapping, or object
		Tree to flatten. Anything without children contributes nothing.
	prefix : str, optional
		Internal, used by the recursion; ignored by callers.

	Returns
	-------
	dict
		Leaf name to value. Later leaves with the same name win.

	Raises
	------
	None

	Related
	-------
	Aberrations.from_metadata : The caller.
	"""
	out = {}
	if metadata is None:
		return out
	if hasattr(metadata, "items") and not isinstance(metadata, (str, bytes)):
		pairs = metadata.items()
	elif hasattr(metadata, "__dict__"):
		pairs = vars(metadata).items()
	else:
		return out
	for key, value in pairs:
		key = str(key)
		if value is None or isinstance(value, (str, bytes, int, float, bool)):
			out[key] = value
		else:
			out.update(_flatten_metadata(value, key))
			out.setdefault(key, value)
	return out
