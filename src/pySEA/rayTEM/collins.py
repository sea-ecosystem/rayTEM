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


def _refuse_screen(element: Any) -> None:
	"""Refuse an element whose wave action is not quadratic: its screen has no place in a Collins step.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Raises
	------
	NotImplementedError
		For an aberration screen, an element carrying aberrations, or one
		carrying a supplied phase screen.
	"""
	from .elements import AberrationScreen

	name = getattr(element, "name", "") or type(element).__name__
	carries = (isinstance(element, AberrationScreen) or bool(getattr(element, "aberrations", None))
			   or (hasattr(element, "_has_screen") and element._has_screen()))
	if carries:
		raise NotImplementedError(
			f"{name!r} carries a non-quadratic phase (aberrations or a screen), which a Collins step "
			"cannot hold yet. Propagate it with mode='hybrid' or 'fixed', or suspend its aberrations "
			"(apply_aberrations=False).")


def _element_signature(element: Any) -> tuple:
	"""Return what a contracted step depends on for one element.

	Parameters
	----------
	element : Element
		A rayTEM element.

	Returns
	-------
	tuple
		The element's affine map (bytes), and for an aperture its radius,
		transmission and alignment.
	"""
	if _is_aperture(element):
		return ("aperture", float(element.radius), float(getattr(element, "transmission", 1.0)),
				repr(getattr(element, "alignment", None)), element_affine_map(element).tobytes())
	return ("map", element_affine_map(element).tobytes())


class CollinsTransfer:
	"""A sequence of elements contracted into Collins steps, with apertures as masks between them.

	Contiguous linear elements (lenses, drifts, dipoles, quadrupoles and
	any other element whose wave action is its ray map) are multiplied into
	one affine map each; an aperture ends a run and is applied as a mask at
	its entrance plane. An element carrying a non-quadratic phase (an
	aberration screen, an aberrated lens) is refused: Collins steps hold no
	screen yet (decision D2 open).

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
		``("map", affine, z_length)`` or ``("aperture", element, 0.0)``, in
		beam order; ``affine`` is 5 x 5 on ``(x, xt, y, yt, 1)``.
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
		From the constructor, for an element with a non-quadratic phase.

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
			5 x 5, the product of the steps' maps (apertures pass rays).
		"""
		total = np.eye(5)
		for kind, affine, _length in self.steps:
			if kind == "map":
				total = affine @ total
		return total

	def _plan(self, index: int, shape: tuple, lattice: np.ndarray, curvature: np.ndarray,
			  wavelength: float, matrix: np.ndarray) -> CollinsPlan:
		"""Return step ``index``'s plan for this input grid, reusing the last one when it matches.

		Parameters
		----------
		index : int
			The step.
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
			the input grid cannot sample.
		"""
		wave = as_collins_wavefield(signal)
		f, L, x0, t0, Q, wavelength, z = read_collins_wavefield(wave)
		f = np.asarray(f)
		z = 0.0 if z is None else float(z)
		name = getattr(wave, "name", "collins wavefield") or "collins wavefield"
		for index, (kind, item, length) in enumerate(self.steps):
			if kind == "aperture":
				f = f * _aperture_transmission(item, f.shape, L, x0)
			else:
				plan = self._plan(index, f.shape, L, Q, wavelength, item[:4, :4])
				f = plan.apply(f)
				L, Q = plan.lattice_out, plan.curvature_out
				ray = item @ np.array([x0[0], t0[0], x0[1], t0[1], 1.0])
				x0, t0 = np.array([ray[0], ray[2]]), np.array([ray[1], ray[3]])
				z += length
			if log is not None:
				log.append(make_collins_wavefield_signal(f, L, x0, t0, Q, wavelength, z=z, name=name))
		return make_collins_wavefield_signal(f, L, x0, t0, Q, wavelength, z=z, name=name)


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
		For an element with a non-quadratic phase.

	See Also
	--------
	pySEA.rayTEM.assemblies.Microscope.collins_transfer : Between two named elements.
	"""
	return CollinsTransfer(elements)
