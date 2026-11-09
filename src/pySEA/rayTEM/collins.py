"""Collins wave propagation: runs of linear elements contracted into one step each.

A run of elements with no aperture and no non-quadratic screen inside it is a
linear paraxial system: its ray map is the product of the elements'
:meth:`~pySEA.rayTEM.elements.Element.affine_matrix` (Larmor rotation, rotated
quadrupoles, dipole kicks and element alignment included), and the wave it
carries is the Collins integral of that map -- one step however many lenses,
drifts and crossovers the run contains. :class:`CollinsTransfer` contracts a
sequence of elements into such steps, with each aperture applied as a mask
between them, and caches each step's precomputed kernels for the grid it last
saw, so applying it to a new wave on the same grid costs one or two FFT
pairs.

This is rayTEM's fifth propagation mode (``propagate_wave_collins``), separate
from the scaled/hybrid walker: Eric (2026-11-06) chose it for the emulator's
detector planes, where the walker's per-crossover frame switches would leave
almost nothing to contract. The scaled wave is unchanged and still carries no
Collins lens body.

An aberration is not linear, so it cannot sit inside a contracted step, and
where along the column it originates matters. An aberrated element therefore
ends the run before it and starts the next after it, and is walked: a thin one
is its screen at its plane, a thick one ``MEDIUM_SLICES`` Collins half-slab
steps around distributed screens, as the scaled walker slices a medium
(decision D2, Eric 2026-11-09: "we only will assign aberrations in a few
places so its just a few breaks in the contractions").

Related
-------
waveoptics.CollinsPlan : One planned step.
seashells.make_collins_wavefield_signal : The wavefield it carries.
elements.Element.propagate_wave_collins : The per-element entry point.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .seashells import as_collins_wavefield, make_collins_wavefield_signal, read_collins_wavefield
from .waveoptics import CollinsPlan

#: The ray components a Collins map acts on, in ``convention`` order.
_RAY_COMPONENTS = ("x", "xt", "y", "yt")


def _ray_indices() -> list:
	"""Return the ``convention`` indices of ``x, xt, y, yt``.

	Returns
	-------
	list of int
		Four indices into a ray vector.
	"""
	from .elements import columnByName

	return [columnByName(name) for name in _RAY_COMPONENTS]


def element_affine_map(element: Any) -> np.ndarray:
	"""Return an element's paraxial map on ``(x, xt, y, yt, 1)`` as a 5 x 5 homogeneous matrix.

	Taken from :meth:`~pySEA.rayTEM.elements.Element.affine_matrix`: the
	transfer matrix (lab frame, Larmor rotation and a quadrupole's rotation
	included), a dipole's kick and the element's alignment in the last
	column. Aberrations are not linear and are not in it.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Returns
	-------
	numpy.ndarray
		Shape ``(5, 5)``.

	Raises
	------
	ValueError
		From the element's ``affine_matrix`` (an alignment it cannot place).
	"""
	full = np.asarray(element.affine_matrix(), dtype=float)
	idx = _ray_indices() + [full.shape[0] - 1]
	return full[np.ix_(idx, idx)]


def _is_aperture(element: Any) -> bool:
	"""Say whether an element masks the beam (an aperture).

	Parameters
	----------
	element : Element
		A rayTEM element.

	Returns
	-------
	bool
		True for an :class:`~pySEA.rayTEM.elements.Aperture`.
	"""
	from .elements import Aperture

	return isinstance(element, Aperture)


def _aberration_power(element: Any) -> float:
	"""Return the power an element's aberrations are written against, or 0 when it has none to apply.

	The same power the ray and wave paths use: an
	:class:`~pySEA.rayTEM.elements.AberrationScreen`'s ``pupil_power``, else
	the element's ``focal_power``.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Returns
	-------
	float
		The pupil power (1/metres); 0 for an element with no aberrations, or
		one whose power is 0 (a transparent plate).
	"""
	if not getattr(element, "aberrations", None):
		return 0.0
	power = getattr(element, "pupil_power", None)
	if power is None:
		power = getattr(element, "focal_power", 0.0)
	return float(power or 0.0)


def _refuse_screen(element: Any) -> None:
	"""Refuse an element whose wave action a Collins transfer cannot carry.

	An aberrated element is walked (:meth:`CollinsTransfer._walk`), but a
	supplied screen is a sampled array with no place on a Collins lattice,
	and an aberrated element off the column's axis would need its aberration
	evaluated about its own axis, which the walk does not do yet.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Raises
	------
	NotImplementedError
		For an element carrying a supplied screen, or an aberrated element
		with an alignment.
	"""
	name = getattr(element, "name", "") or type(element).__name__
	if hasattr(element, "_has_screen") and element._has_screen():
		raise NotImplementedError(
			f"{name!r} carries a supplied screen, which a Collins transfer cannot sample. "
			"Propagate it with mode='hybrid' or 'fixed'.")
	if _aberration_power(element) and getattr(element, "alignment", None) is not None:
		raise NotImplementedError(
			f"{name!r} is aberrated and off the column's axis; a Collins transfer evaluates "
			"aberrations about the column's axis only. Propagate it with mode='hybrid' or 'fixed'.")


def _element_signature(element: Any) -> tuple:
	"""Return what a contracted step depends on for one element.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Returns
	-------
	tuple
		The element's affine map (bytes), its aberration coefficients, and
		for an aperture its radius, transmission and alignment.
	"""
	if _is_aperture(element):
		return ("aperture", float(element.radius), float(getattr(element, "transmission", 1.0)),
				repr(getattr(element, "alignment", None)), element_affine_map(element).tobytes())
	aberrations = getattr(element, "aberrations", None)
	coefficients = repr(sorted(aberrations.items())) if aberrations else ""
	return ("map", element_affine_map(element).tobytes(), coefficients, _aberration_power(element))


def _slab(element: Any) -> Any:
	"""Return a thick element's slab: a shallow copy one ``2 * MEDIUM_SLICES``-th as long.

	The scaled walker's slicing (``Element._propagate_wave_scaled``):
	``MEDIUM_SLICES`` slices, each half a slab, a screen, half a slab.

	Parameters
	----------
	element : Element
		A thick element (``length != 0``).

	Returns
	-------
	Element
		The slab, sharing everything but ``length``.

	Raises
	------
	NotImplementedError
		When the slabs' maps do not multiply back to the element's (an element
		whose map is not a uniform medium), so slicing it would misplace it.
	"""
	from .elements import MEDIUM_SLICES

	slab = type(element).__new__(type(element))
	slab.__dict__.update(element.__dict__)
	object.__setattr__(slab, "length", float(element.length) / (2 * MEDIUM_SLICES))
	whole = element_affine_map(element)
	product = np.linalg.matrix_power(element_affine_map(slab), 2 * MEDIUM_SLICES)
	if not np.allclose(product, whole, rtol=1e-9, atol=1e-12 * max(1.0, float(np.abs(whole).max()))):
		name = getattr(element, "name", "") or type(element).__name__
		raise NotImplementedError(
			f"{name!r} is aberrated and thick, but its body is not a uniform medium that can be "
			"sliced. Propagate it with mode='hybrid' or 'fixed'.")
	return slab


class CollinsTransfer:
	"""A sequence of elements contracted into Collins steps, with apertures as masks between them.

	Contiguous linear elements (lenses, drifts, dipoles, quadrupoles and
	any other element whose wave action is its ray map) are multiplied into
	one affine map each; an aperture ends a run and is applied as a mask at
	its entrance plane. An aberrated element (an aberrated lens, an
	:class:`~pySEA.rayTEM.elements.AberrationScreen`) also ends a run and is
	**walked** (decision D2, Eric 2026-11-09): a thin one is its aberration
	screen and its map; a thick one is ``MEDIUM_SLICES`` slices of half a
	slab, a ``1/MEDIUM_SLICES`` share of its aberration at the local ray
	height, half a slab -- the scaled walker's slicing -- so where along the
	column an aberration originates is kept. An element with a supplied
	screen, or an aberrated one off the column's axis, is refused.

	Each step keeps the :class:`~pySEA.rayTEM.waveoptics.CollinsPlan` it
	last built, keyed by the input grid, so applying the transfer to many
	waves on one grid (an emulator's frames) recomputes no kernel.
	Contracting itself costs a product of small matrices: rebuild the
	transfer whenever :meth:`is_current` says an element changed.

	Parameters
	----------
	elements : Sequence[Element]
		The elements, in beam order. A :class:`~pySEA.rayTEM.elements.Source`
		is skipped (it starts a wave, it does not act on one).

	Attributes
	----------
	elements : list of Element
		The contracted elements, sources removed.
	steps : list of tuple
		``("map", affine, z_length)``, ``("aperture", element, 0.0)`` or
		``("walk", element, z_length)``, in beam order; ``affine`` is 5 x 5
		on ``(x, xt, y, yt, 1)``.
	signature : tuple
		Per-element :func:`_element_signature` at contraction.

	Methods
	-------
	is_current()
		Whether every element still has the settings it was contracted with.
	affine_map()
		The whole transfer's ray map (apertures pass rays unchanged).
	apply(signal, log=None)
		Carry a wave through the transfer.

	Raises
	------
	NotImplementedError
		From the constructor, for an element with a supplied screen, an
		aberrated element off axis, or a thick aberrated element that is not
		a uniform medium.

	See Also
	--------
	contract_collins : Build one.

	Examples
	--------
	>>> from pySEA.rayTEM.elements import Lens, Drift
	>>> transfer = CollinsTransfer([Drift(length=0.01), Lens(focal_length=0.01), Drift(length=0.01)])  # doctest: +SKIP
	>>> [step[0] for step in transfer.steps]  # doctest: +SKIP
	['map']
	"""

	def __init__(self, elements: Sequence[Any]) -> None:
		"""Contract the elements.

		Parameters
		----------
		elements : Sequence[Element]
			As the class.

		Raises
		------
		NotImplementedError
			As the class.
		"""
		from .elements import Source

		self.elements = [e for e in elements if not isinstance(e, Source)]
		self.steps: list = []
		self._slabs: dict = {}
		affine, length = np.eye(5), 0.0
		pending = False
		for element in self.elements:
			if _is_aperture(element):
				if pending:
					self.steps.append(("map", affine, length))
					affine, length, pending = np.eye(5), 0.0, False
				self.steps.append(("aperture", element, 0.0))
				# an aperture of finite length masks at its entrance, then is a drift
				if float(getattr(element, "length", 0.0) or 0.0):
					affine, length, pending = element_affine_map(element), float(element.length), True
				continue
			_refuse_screen(element)
			if _aberration_power(element):
				if pending:
					self.steps.append(("map", affine, length))
					affine, length, pending = np.eye(5), 0.0, False
				thick = float(getattr(element, "length", 0.0) or 0.0)
				self.steps.append(("walk", element, thick))
				self._slabs[id(element)] = _slab(element) if thick else None
				continue
			affine = element_affine_map(element) @ affine
			length += float(getattr(element, "length", 0.0) or 0.0)
			pending = True
		if pending:
			self.steps.append(("map", affine, length))
		self.signature = tuple(_element_signature(e) for e in self.elements)
		self._plans: dict = {}

	def is_current(self) -> bool:
		"""Say whether every element still has the settings it was contracted with.

		Returns
		-------
		bool
			False once any element's map (a lens current, a dipole, an
			alignment) or an aperture changed; rebuild the transfer then.
		"""
		try:
			return tuple(_element_signature(e) for e in self.elements) == self.signature
		except Exception:  # noqa: BLE001 - an element that cannot be read any more has changed
			return False

	def affine_map(self) -> np.ndarray:
		"""Return the whole transfer's ray map on ``(x, xt, y, yt, 1)``.

		Returns
		-------
		numpy.ndarray
			5 x 5, the product of the steps' maps (apertures pass rays; a
			walked element contributes its paraxial map, not its aberration).
		"""
		total = np.eye(5)
		for kind, item, _length in self.steps:
			if kind == "map":
				total = item @ total
			elif kind == "walk":
				total = element_affine_map(item) @ total
		return total

	def _plan(self, index: Any, shape: tuple, lattice: np.ndarray, curvature: np.ndarray,
			  wavelength: float, matrix: np.ndarray) -> CollinsPlan:
		"""Return step ``index``'s plan for this input grid, reusing the last one when it matches.

		Parameters
		----------
		index : int or tuple
			The step, or ``(step, part)`` for one part of a walked element.
		shape : tuple of int
			Input shape.
		lattice, curvature : numpy.ndarray
			Input lattice and curvature.
		wavelength : float
			Wavelength (metres).
		matrix : numpy.ndarray
			The step's 4 x 4 linear map.

		Returns
		-------
		CollinsPlan
			The planned step.
		"""
		key = (tuple(shape), np.asarray(lattice, dtype=float).tobytes(),
			   np.asarray(curvature, dtype=float).tobytes(), float(wavelength))
		cached = self._plans.get(index)
		if cached is not None and cached[0] == key:
			return cached[1]
		plan = CollinsPlan(shape, lattice, curvature, wavelength, matrix)
		self._plans[index] = (key, plan)
		return plan

	def apply(self, signal: Any, log: list | None = None) -> Any:
		"""Carry a wave through the transfer.

		Parameters
		----------
		signal : Signal or wavefield fallback
			Any rayTEM wavefield (fixed, scaled or Collins); converted by
			:func:`~pySEA.rayTEM.seashells.as_collins_wavefield` without
			resampling.
		log : list, optional
			When given, the wave after each step is appended to it.

		Returns
		-------
		Signal or seashells._CollinsWavefield
			The Collins wavefield after the last step.

		Raises
		------
		ValueError
			From :class:`~pySEA.rayTEM.waveoptics.CollinsPlan`, for a step
			the input grid cannot sample, or from
			``elements._check_screen_sampling``, for an aberration whose phase
			steps more than pi between samples inside the beam.
		"""
		wave = as_collins_wavefield(signal)
		f, L, x0, t0, Q, wavelength, z = read_collins_wavefield(wave)
		f = np.asarray(f)
		z = 0.0 if z is None else float(z)
		name = getattr(wave, "name", "collins wavefield") or "collins wavefield"
		for index, (kind, item, length) in enumerate(self.steps):
			if kind == "aperture":
				f = f * _aperture_transmission(item, f.shape, L, x0)
			elif kind == "walk":
				f, L, Q, x0, t0 = self._walk(index, item, f, L, Q, x0, t0, wavelength)
				z += length
			else:
				f, L, Q, x0, t0 = self._step(index, item, f, L, Q, x0, t0, wavelength)
				z += length
			if log is not None:
				log.append(make_collins_wavefield_signal(f, L, x0, t0, Q, wavelength, z=z, name=name))
		return make_collins_wavefield_signal(f, L, x0, t0, Q, wavelength, z=z, name=name)

	def _step(self, index: Any, affine: np.ndarray, f: np.ndarray, L: np.ndarray, Q: np.ndarray,
			  x0: np.ndarray, t0: np.ndarray, wavelength: float) -> tuple:
		"""Carry the field through one affine map: one Collins step, and the central ray with it.

		Parameters
		----------
		index : int or tuple
			The plan's cache key (see :meth:`_plan`).
		affine : numpy.ndarray
			5 x 5 map on ``(x, xt, y, yt, 1)``.
		f, L, Q : numpy.ndarray
			Field samples, lattice and curvature.
		x0, t0 : numpy.ndarray
			Central ray position and angle.
		wavelength : float
			Wavelength (metres).

		Returns
		-------
		tuple
			``(f, L, Q, x0, t0)`` after the map.

		Raises
		------
		ValueError
			From :class:`~pySEA.rayTEM.waveoptics.CollinsPlan`.
		"""
		plan = self._plan(index, f.shape, L, Q, wavelength, affine[:4, :4])
		f = plan.apply(f)
		ray = affine @ np.array([x0[0], t0[0], x0[1], t0[1], 1.0])
		return (f, plan.lattice_out, plan.curvature_out,
				np.array([ray[0], ray[2]]), np.array([ray[1], ray[3]]))

	def _walk(self, index: int, element: Any, f: np.ndarray, L: np.ndarray, Q: np.ndarray,
			  x0: np.ndarray, t0: np.ndarray, wavelength: float) -> tuple:
		"""Carry the field through one aberrated element, applying its aberration where it acts.

		A thin element: its aberration screen at its plane, then its map. A
		thick one: ``MEDIUM_SLICES`` slices of half a slab (a Collins step),
		``1/MEDIUM_SLICES`` of the aberration, half a slab -- the scaled
		walker's distributed screens (``Element._propagate_wave_scaled``), so
		each share acts at the local ray height and the rest of the body
		focuses its kick. The aberration is
		:meth:`~pySEA.rayTEM.aberrations.Aberrations.phase_at` the samples'
		physical positions (lattice times index plus the central ray), against
		the element's pupil power (:func:`_aberration_power`), C10 and C12
		included as on the scaled path's thick lens.

		Parameters
		----------
		index : int
			The step (its parts cache plans as ``(index, part)``).
		element : Element
			The aberrated element.
		f, L, Q : numpy.ndarray
			Field samples, lattice and curvature.
		x0, t0 : numpy.ndarray
			Central ray position and angle.
		wavelength : float
			Wavelength (metres).

		Returns
		-------
		tuple
			``(f, L, Q, x0, t0)`` after the element.

		Raises
		------
		ValueError
			From ``elements._check_screen_sampling`` (an aliased aberration)
			or :class:`~pySEA.rayTEM.waveoptics.CollinsPlan`.
		"""
		from .elements import MEDIUM_SLICES, _check_screen_sampling
		from .waveoptics import _index_grid

		name = getattr(element, "name", "") or type(element).__name__
		power = _aberration_power(element)

		def screen(f: np.ndarray, L: np.ndarray, x0: np.ndarray, fraction: float) -> np.ndarray:
			"""Apply ``fraction`` of the element's aberration at the samples' positions.

			Parameters
			----------
			f, L : numpy.ndarray
				Field samples and lattice.
			x0 : numpy.ndarray
				Central ray position.
			fraction : float
				The share of the aberration this plane carries.

			Returns
			-------
			numpy.ndarray
				The field with the screen applied.

			Raises
			------
			ValueError
				From ``_check_screen_sampling``.
			"""
			UX, UY = _index_grid(f.shape)
			X = L[0, 0] * UX + L[0, 1] * UY + x0[0]
			Y = L[1, 0] * UX + L[1, 1] * UY + x0[1]
			chi = fraction * element.aberrations.phase_at(X, Y, wavelength, power)
			_check_screen_sampling(chi, name, field=f)
			return f * np.exp(1j * chi)

		slab = self._slabs.get(id(element))
		if slab is None:
			f = screen(f, L, x0, 1.0)
			return self._step((index, 0), element_affine_map(element), f, L, Q, x0, t0, wavelength)
		half = element_affine_map(slab)
		for k in range(MEDIUM_SLICES):
			f, L, Q, x0, t0 = self._step((index, 2 * k), half, f, L, Q, x0, t0, wavelength)
			f = screen(f, L, x0, 1.0 / MEDIUM_SLICES)
			f, L, Q, x0, t0 = self._step((index, 2 * k + 1), half, f, L, Q, x0, t0, wavelength)
		return f, L, Q, x0, t0


def _aperture_transmission(aperture: Any, shape: tuple, lattice: np.ndarray, center: np.ndarray) -> np.ndarray:
	"""Return an aperture's amplitude transmission at a Collins wavefield's samples.

	The same edge as the fixed path (:func:`~pySEA.rayTEM.waveoptics.aperture_transmission`:
	a one-pixel area-coverage ramp), evaluated at each sample's physical
	position about the aperture's own axis.

	Parameters
	----------
	aperture : Aperture
		The aperture.
	shape : tuple of int
		Field shape ``(ny, nx)``.
	lattice : numpy.ndarray
		2 x 2 lattice.
	center : numpy.ndarray
		The central ray's position (metres).

	Returns
	-------
	numpy.ndarray
		Real transmission, ``shape``; the square root of the aperture's
		``transmission`` scales it.
	"""
	from .waveoptics import _index_grid

	UX, UY = _index_grid(shape)
	axis = np.zeros(2)
	offsets = aperture.alignment_offsets() if hasattr(aperture, "alignment_offsets") else None
	if offsets is not None:
		entrance = np.asarray(offsets[0], dtype=float)
		idx = _ray_indices()
		axis = np.array([entrance[idx[0]], entrance[idx[2]]])
	X = lattice[0, 0] * UX + lattice[0, 1] * UY + center[0] - axis[0]
	Y = lattice[1, 0] * UX + lattice[1, 1] * UY + center[1] - axis[1]
	pixel = float(np.sqrt(abs(np.linalg.det(lattice))))
	T = np.clip(0.5 + (float(aperture.radius) - np.hypot(X, Y)) / pixel, 0.0, 1.0)
	return T * float(getattr(aperture, "transmission", 1.0)) ** 0.5


def contract_collins(elements: Sequence[Any]) -> CollinsTransfer:
	"""Contract a sequence of elements into a :class:`CollinsTransfer`.

	Any contiguous run works: a whole section, several sections joined
	(G+C+A+OL1), or a slice between two named elements.

	Parameters
	----------
	elements : Sequence[Element]
		The elements, in beam order.

	Returns
	-------
	CollinsTransfer
		The contracted transfer.

	Raises
	------
	NotImplementedError
		For an element with a supplied screen, an aberrated element off axis,
		or a thick aberrated element that is not a uniform medium.

	See Also
	--------
	pySEA.rayTEM.assemblies.Microscope.collins_transfer : Between two named elements.
	"""
	return CollinsTransfer(elements)
