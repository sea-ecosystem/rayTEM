from __future__ import annotations

from typing import Sequence, Literal
from numpy.typing import ArrayLike

import numpy as xp
flag_gpu = False
import traceback,inspect
from warnings import warn

from .seashells import SEASerializable
from .aberrations import Aberrations

from copy import deepcopy

# CONVENTION: a ray is a purely *geometric* state vector: lateral positions (x,y),
# angles (xt,yt, "t" for theta θ or tilt), position down the column (z), and energy (E).
# rays at a given position are 2D: a list of these sextuplets (grab the 'x' column to grab
# each ray's x position, for example). rays throughout the microscope are 3D: a list of the above.
# Intensity (I) and cumulative Larmor rotation (R) are NOT ray coordinates — they are tracked
# as separate parallel arrays alongside the rays (see MicroscopeSection/Microscope .I and .R)
# because they do not participate in the ray-transfer matrix and would otherwise masquerade as
# geometric coordinates. This keeps transfer matrices purely geometric.
# The columnByName function is used universally, so additional geometric columns can be added
# (or reordered) without every Element needing to be updated.

convention = ["x","xt","y","yt","z","E"]
# given a keyword, return the column associated. r0[:,columnByName('x')] should return every ray's x position
def columnByName(name):
	return convention.index(name)
# given a transfer_matrix defined by a subset of columns (e.g. a 2x2 lens for focusing in x only) "inflate" out to 7x7 based on convention set by columnByName. [ x₂ θ₂ ] = [2x2] @ [ x₁ θ₁ ] (https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis) would become [ x₂ xθ₂ y₂ yθ₂ ....] = [7x7] @ [ x₁ xθ₁ y₁ yθ₁....]
def fix_mat_dims(m,columnNames):
	new=xp.eye(len(convention))
	for i,n1 in enumerate(columnNames):
		for j,n2 in enumerate(columnNames):
			new[columnByName(n1),columnByName(n2)]=m[i,j]
	return new
# similar to fix_mat_dims, but for rays
def fix_ray_dims(rays,columnNames):
	new=xp.zeros((len(rays),len(convention)))
	for i,name in enumerate(columnNames):
		new[:,columnByName(name)]=rays[:,i]
	return new

# Canonical mapping from a propagation-mode keyword to (method name, forced kwargs).
# Used by the unified `propagate(kind=...)` dispatcher on Element/MicroscopeSection/Microscope;
# the wave kinds all route to the one propagate_wave method with its `mode` selector.
_PROPAGATE_KINDS = {
	"ray":         ("propagate_ray", {}),
	"rays":        ("propagate_ray", {}),
	"moments":     ("propagate_moments", {}),
	"envelope":    ("propagate_moments", {}),
	"covariance":  ("propagate_moments", {}),
	"wave":        ("propagate_wave", {"mode": "fixed"}),
	"wave-scaled": ("propagate_wave", {"mode": "scaled"}),
	"wave_scaled": ("propagate_wave", {"mode": "scaled"}),
	"wave-hybrid": ("propagate_wave", {"mode": "hybrid"}),
	"wave_hybrid": ("propagate_wave", {"mode": "hybrid"}),
}
def _propagate_method_name(kind:str) -> tuple:
	"""Resolve a propagation-mode keyword to its method name and forced kwargs.

	Parameters
	----------
	kind : str
		Mode keyword: ``'ray'``/``'rays'``, ``'moments'``/``'envelope'``/
		``'covariance'``, ``'wave'``, ``'wave-scaled'``/``'wave_scaled'``, or
		``'wave-hybrid'``/``'wave_hybrid'``.

	Returns
	-------
	tuple
		``(method_name, forced_kwargs)`` — the concrete method plus the
		keyword overrides the kind implies (the wave kinds force the ``mode``
		selector on ``propagate_wave``).

	Raises
	------
	ValueError
		If ``kind`` is not a recognized propagation mode.
	"""
	try:
		return _PROPAGATE_KINDS[kind]
	except KeyError:
		raise ValueError(f"Unknown propagation kind {kind!r}; expected one of {sorted(set(_PROPAGATE_KINDS))}.")

def _kernel_item(ny:int, nx:int, dy:float, dx:float, wavelength:float, dz:float,
				 name:str="free segment"):
	"""Build a reciprocal-space free-segment phase item for a phase program.

	Wraps :func:`waveoptics.kernel_phase` (carrier included — the fixed-grid
	path propagates the full wave) as a space-tagged phase Signal.

	Parameters
	----------
	ny, nx : int
		Grid shape.
	dy, dx : float
		Sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	dz : float
		Segment length (metres).
	name : str, optional
		Item name, by default ``"free segment"``.

	Returns
	-------
	Signal or seashells._Phase
		Scattering-space phase item.
	"""
	from .waveoptics import kernel_phase
	from .seashells import make_kernel_phase_signal
	chi = kernel_phase((ny, nx), dx, dy, wavelength, dz, include_carrier=True)
	fx = xp.fft.fftfreq(nx, d=dx)
	fy = xp.fft.fftfreq(ny, d=dy)
	return make_kernel_phase_signal(chi, fx, fy, name=name)


def _screen_item(chi, dx:float, dy:float, name:str):
	"""Build a real-space phase-screen item for a phase program.

	Parameters
	----------
	chi : xp.ndarray
		Real phase χ (radians), shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings of the grid χ was built on.
	name : str
		Item name.

	Returns
	-------
	Signal or seashells._Phase
		Position-space phase item.
	"""
	from .seashells import make_screen_phase_signal
	return make_screen_phase_signal(chi, dx, dy, name=name)


def _check_screen_sampling(chi, name:str):
	"""Guard against an aliased phase screen applied to the scaled field U.

	The scaled path applies non-absorbable element phases (quadrupole saddle,
	dipole tilt, aberrations) explicitly to U; a screen whose phase steps more
	than π between neighbouring samples aliases silently. This check fails
	loudly instead (handoff Eqs 47–48 sampling requirement ``|∂χ/∂ξ| < π/Δξ``).

	**Complex screens are skipped, because this check cannot see them.** What
	it measures is ``|diff(data)|`` against π, which is a phase step only when
	the data *is* a phase. On a complex transmission the same quantity mixes
	modulus and phase and is bounded by ``2|T|``, so it sits below π even when
	the phase underneath winds far faster than the grid can carry: a unit-
	modulus screen stepping 3.78 rad per pixel — aliased — reports 1.90 and
	passes. Running it on complex data would give false assurance, not
	protection.

	Checking ``arg(T)`` instead is not a fix either: it is undefined wherever
	``T = 0``, and genuinely discontinuous at a plate's edge. Telling that edge
	apart from aliasing needs to know how the screen was built, which a
	supplied array does not record. So a supplied complex screen is the
	caller's responsibility, and this says so rather than pretending to cover
	it.

	Parameters
	----------
	chi : xp.ndarray
		Screen data: real phase χ (radians), or a complex transmission (which
		is passed through unchecked), shape ``(ny, nx)``.
	name : str
		Element/screen name for the error message.

	Returns
	-------
	None
		Passes silently when adequately sampled, or when the screen is complex.

	Raises
	------
	ValueError
		If the per-pixel phase step of a *real* screen reaches π anywhere.

	Related
	-------
	waveoptics.apply_phase : The consumer, and the real/complex convention.
	"""
	if xp.iscomplexobj(chi):
		return
	step = 0.0
	if chi.shape[1] > 1:
		step = max(step, float(xp.abs(xp.diff(chi, axis=1)).max()))
	if chi.shape[0] > 1:
		step = max(step, float(xp.abs(xp.diff(chi, axis=0)).max()))
	if step >= xp.pi:
		raise ValueError(f"Phase screen {name!r} is under-sampled on the scaled grid "
						 f"(max per-pixel phase step {step:.2f} rad >= pi): reduce the element "
						 "strength, refine the wave grid, or enlarge the field of view.")



class suspended_aberrations:
	"""Context manager that temporarily detaches elements' aberrations.

	This is how ``apply_aberrations=False`` is implemented on every propagation
	method. Detaching is preferable to a flag threaded down through three levels
	of driver and four propagation kinds: an element already answers "am I
	aberrated?" by looking at :attr:`Element.aberrations`, so removing them for
	the duration of a run makes *every* path ideal at once, including any path
	added later. Nothing in the propagation code has to know the flag exists.

	Doing nothing (``suspend=False``, or no aberrated element) costs one
	attribute read per element.

	Parameters
	----------
	elements : Sequence[Element]
		Elements to suspend. Safe to include elements with no aberrations.
	suspend : bool, optional
		Whether to actually detach, by default True. ``False`` makes the
		context a no-op, so a caller can write
		``with suspended_aberrations(elems, not apply_aberrations):``
		unconditionally.

	Attributes
	----------
	elements : list of Element
		The elements being managed.
	suspend : bool
		Whether this context detaches anything.

	Methods
	-------
	__enter__()
		Detach and remember.
	__exit__(*exc)
		Restore, including when the body raised.

	Raises
	------
	None

	Related
	-------
	assemblies.Microscope.propagate_ray : One of the callers.
	Element.aberration_kick : What goes quiet while suspended.

	Notes
	-----
	Restoration happens in ``__exit__``, so an exception mid-propagation cannot
	leave a microscope silently de-aberrated.

	Examples
	--------
	>>> with suspended_aberrations(section.elements):  # doctest: +SKIP
	...     ideal = section.propagate_ray(r0)
	"""

	def __init__(self, elements:Sequence["Element"], suspend:bool=True):
		"""Remember what to suspend.

		Parameters
		----------
		elements : Sequence[Element]
			Elements to suspend.
		suspend : bool, optional
			Whether to actually detach, by default True.

		Raises
		------
		None
		"""
		self.elements = list(elements or ())
		self.suspend = bool(suspend)
		self._saved = []

	def __enter__(self) -> "suspended_aberrations":
		"""Detach each element's aberrations, remembering them.

		Returns
		-------
		suspended_aberrations
			Self, so the context can be named if a caller wants it.

		Raises
		------
		None
		"""
		self._saved = []
		if not self.suspend:
			return self
		for e in self.elements:
			self._saved.append((e, getattr(e, "aberrations", None)))
			e.aberrations = None
		return self

	def __exit__(self, *exc) -> bool:
		"""Restore every detached set.

		Parameters
		----------
		*exc : tuple
			Exception triple, unused — this never suppresses.

		Returns
		-------
		bool
			False, so any exception from the body propagates.

		Raises
		------
		None
		"""
		for e, ab in self._saved:
			e.aberrations = ab
		self._saved = []
		return False


def _as_aberrations(value) -> Aberrations:
	"""Coerce a user-supplied aberration specification to an :class:`Aberrations`.

	Element constructors accept either the object or a bare ``{name: value}``
	mapping, so a quick script does not have to import the class. ``None``
	stays ``None``, which is what "ideal, and cost nothing" looks like on the
	propagation paths.

	Parameters
	----------
	value : Aberrations, Mapping, or None
		The specification given to an element.

	Returns
	-------
	Aberrations or None
		The object to store. A supplied :class:`Aberrations` is returned
		unchanged (not copied), so a set shared between elements stays shared.

	Raises
	------
	TypeError
		If the value is neither a mapping, an :class:`Aberrations`, nor
		``None``.

	Related
	-------
	Element.aberration_kick : Consumes the result on the ray path.
	aberrations.Aberrations : The storage class.

	Examples
	--------
	>>> _as_aberrations({'C30': 1e-3})
	Aberrations(C30=0.001+0j, convention='krivanek')
	"""
	if value is None or isinstance(value, Aberrations):
		return value
	if hasattr(value, "items"):
		return Aberrations(dict(value.items()))
	raise TypeError("aberrations must be an Aberrations object, a {name: value} "
					f"mapping, or None; got {type(value).__name__}.")


class Element(SEASerializable):
	#: Stored attributes that are not constructor parameters, so
	#: :func:`seashells.safeReinstantiate` restores them explicitly on reload.
	#: ``aberrations`` is a kwarg on ``Element`` and ``Lens`` but not on every
	#: subclass; ``_screen``'s kwarg is spelled ``screen``. Both would be
	#: silently dropped otherwise -- and a supplied screen has no other copy.
	_restore_attrs = ("aberrations", "_screen")

	def __init__(self, name:str='', kind:str=None,
				 aberrations=None, screen=None ) -> SEASerializable:
		"""General microscope element class. Only the basic/required attributes (name and kind) are populated, as additional attributed can be defined at the inheriting class level. e.g. a Lens has a "strength", but a Drift section does not.
		The base class carries a working transparent default for every propagation kind (identity transfer_matrix, phase shift of nothing), so inheriting classes only override what their physics requires: transfer_matrix and/or phase_shift, and *may* define a custom propagate_ray function if the standard "[ x₂ xθ₂ y₂ yθ₂ ....] = [6x6] @ [ x₁ xθ₁ y₁ yθ₁....]" is not applicable

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		kind : str, optional
			Type of element, by default None
		aberrations : Aberrations or dict, optional
			Axial wave aberrations in Krivanek ``C_{n,m}`` notation, by default
			``None`` (ideal). Accepts an :class:`aberrations.Aberrations` or a
			bare ``{name: value}`` mapping. Applied generically by
			:meth:`aberration_kick` on the ray path and :meth:`phase_shift` on
			the wave path, so no element needs per-aberration code.
		screen : Signal or numpy.ndarray, optional
			A screen supplied by the caller rather than generated, by default
			``None``. Real means a phase χ in radians; complex means a
			transmission ``T`` carrying amplitude *and* phase, which is what a
			fabricated plate has. See :attr:`screen`.
		"""
		self.name = name
		self.kind = kind
		# Every element may carry aberrations; None means "I am exactly my
		# matrix", so ideal columns stay bit-for-bit unchanged.
		self.aberrations = _as_aberrations(aberrations)
		# A SUPPLIED screen is stored because nothing can recompute it. A screen
		# derivable from `aberrations` is not stored -- the coefficients are the
		# storage, and they are smaller than the grid they generate.
		self._screen = screen
		self.length = 0		# transparent default: zero physical extent (subclasses overwrite)

	#####################################
    # region: Dunders

	# print function: look for specific attributes on inheriting class object, and display as columns
	def __repr__(self,header=True) -> str:
		whitelist = [ "name", "kind", "position", "length", "strength", "calibration", "axis" ]
		rep = { k:getattr(self,k) for k in self.__dict__ if k in whitelist }
		#rep = {'name':self.name,
		#	   'kind':self.kind,
		#	   }
		h = [] ; s = []
		for k,v in rep.items():
			h.append(k+" "*(8-len(k)))
			if v is None:
				v="[None]"
			if isinstance(v,float):
				v = xp.round(v,7)
			v=str(v) ; v=v+" "*(8-len(v)) ; v=v[:8]
			s.append(v)
		if header:
			return " ".join(h)+"\n"+" ".join(s)
		return " ".join(s)
	
	def __str__(self):
		if self.name is None or self.name=='': name = 'Unnamed'
		else: name = self.name
		if self.kind is None: kind = 'Unkown'
		else: kind = self.kind
		return f'{name} ({kind} Element)'

	def copy(self):
		return deepcopy(self)
		dic = self.__dict__
		allowed_kwargs = inspect.signature(type(self)).parameters.keys() # infer allowed kwargs from function itself, and filter down to only those.
		dic = { k:v for k,v in dic.items() if k in allowed_kwargs } # e.g., Source doesn't accept "length" even though it
		return type(self)(**dic)

	def kget(self, key:str):
		"""Get an element attribute by name.

		A small keyed accessor used by fitting helpers (e.g.
		:func:`postprocessing.fitForCrossover`) to read a parameter such as
		``"strength"`` generically.

		Parameters
		----------
		key : str
			Attribute name to read.

		Returns
		-------
		object
			The value of ``self.<key>``.

		Raises
		------
		AttributeError
			If the element has no attribute ``key``.

		Related
		-------
		kset : Keyed setter counterpart.
		"""
		return getattr(self, key)

	def kset(self, key:str, value) -> None:
		"""Set an element attribute by name.

		Keyed setter counterpart to :meth:`kget`, used by fitting helpers to write
		a parameter such as ``"strength"`` generically.

		Parameters
		----------
		key : str
			Attribute name to set.
		value : object
			New value to assign to ``self.<key>``.

		Returns
		-------
		None

		Related
		-------
		kget : Keyed getter counterpart.
		"""
		setattr(self, key, value)

	# e.position should be read-only! user should not set position of an element within a section, they should use s.move(...)
	@property
	def position(self):
		return self._position
	#@position.setter			# commented out: "position" attribute should be read-only! this setter only exists to ensure pytest tests as expected (i.e., failing a test when "position" is writeable)
	#def position(self,val):
	#	self._position = val

	# endregion
	#####################################

	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propagation: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis

		The ray-side element contract. The base ``Element`` returns the
		**identity** — a transparent element that transports rays (and moments)
		unchanged — so every propagation kind works on any element by default.
		Subclasses with ray physics override this, typically defining the
		relevant 2×2 block(s) and inflating with :func:`fix_mat_dims`.

		Returns
		-------
		xp.ndarray
			The ``len(convention) × len(convention)`` transfer matrix (identity
			on the base class).

		Related
		-------
		phase_shift : The wave-side counterpart (transparent on the base class).
		propagate_ray, propagate_moments : Consumers of this matrix.
		"""
		return xp.eye(len(convention))

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""Wave-side element contract: the phase this element imprints on a wave.

		The wave counterpart of :meth:`transfer_matrix`: each element class states
		its wave physics explicitly as a scalar, projected-potential-like phase
		χ (radians) that the propagators apply as ``exp(i·χ)``.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid: a sea_eco ``Dimensions`` whose trailing axes are the
			calibrated y/x dimensions, or the fallback ``((ny, nx), dx, dy)``.
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			Select the representation, by default False.
			``False`` → return the ordered **phase program** for the fixed-grid
			propagator: a list of space-tagged phase Signals (Dimension
			``space='position'`` = real-space screen; ``'scattering'`` =
			reciprocal-space free-segment kernel). A finite-length element yields
			``[kernel(L/2), screen(χ), kernel(L/2)]``.
			``True`` → return the scaled-representation split ``(power,
			screen)``: ``power`` is the focusing power absorbed into the
			curvature state (``1/R⁺ = 1/R⁻ − power``, handoff Eq 45) — a
			per-axis ``(P_x, P_y)`` pair for astigmatic elements, absorbed
			into the anisotropic curvature ``(R_x, R_y)`` — and ``screen`` is
			the phase applied explicitly to U (handoff Eqs 47–48; ``None``
			when fully absorbed), evaluated at physical coordinates
			``x = s·ξ``.
		s : float or Sequence[float], optional
			Current transverse scale factor (used only when ``scaled=True``);
			an ``(s_x, s_y)`` pair on an anisotropic frame, by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: list of phase Signals in application order.
			``scaled=True``: ``(power, screen_or_None)``.

		Related
		-------
		transfer_matrix : The ray-side counterpart (identity on the base class).
		propagate_wave : Consumer of this contract in every wave mode.

		Notes
		-----
		The base ``Element`` is **transparent** — the wave analog of the
		identity transfer matrix: a phase of nothing, plus free-space transport
		over the element's ``length`` (the fixed path returns a single
		full-length kernel, or an empty program for zero length; the scaled
		path returns ``(0.0, None)`` — its drivers already run the free
		segments). Element classes with wave physics (``Lens``, ``Quadrapole``,
		``Dipole``, ``Drift``) override this with their explicit phase;
		non-phase elements (``Source``, ``Aperture``, ``Prism``) override it to
		fail loudly because their wave action lives elsewhere.
		"""
		from .seashells import grid_of
		ny, nx, dy, dx = grid_of(dimensions)
		chi = self._combine_screen(None, (ny, nx), dx, dy)	# None unless one was supplied
		if scaled:
			return 0.0, (None if chi is None else _screen_item(chi, dx, dy,
															   self.name or type(self).__name__))
		return self._phase_program(dimensions, wavelength, chi,
								   self.name or type(self).__name__)

	def transfer_block(self, dz:float=None, axis:Literal['x','y']='x') -> xp.ndarray:
		r"""Rotating-frame 2x2 transfer block for one transverse axis.

		(Renamed from ``transfer_xblock``: the ``x`` meant *transverse*, not the
		x axis, which made ``transfer_xblock(axis='y')`` read as a
		contradiction.)

		The ``(position, angle)`` sub-block of :meth:`transfer_matrix` for a
		single axis, with the Larmor rotation left out — the frame
		:func:`postprocessing.findPlanes` works in, and the object that locates
		special planes: with ``M = [[A, B], [C, D]]`` accumulated from a
		reference plane, ``A = 0`` marks a diffraction (back-focal) plane and
		``B = 0`` an image plane of that reference.

		Unlike :meth:`transfer_matrix` this accepts a **partial** length, so a
		plane falling *inside* an element body can be located exactly rather
		than interpolated between its faces. The base implementation is a thin
		kick followed by free space, which is right for every point-like
		element; :meth:`Lens.transfer_block` overrides it with the ``cos/sin``
		law of a thick body.

		Parameters
		----------
		dz : float, optional
			Distance into the element (metres); ``None`` (default) uses the
			full ``length``.
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``. Only astigmatic elements
			(quadrupoles) differ between the two.

		Returns
		-------
		xp.ndarray
			The ``2x2`` block ``[[A, B], [C, D]]`` for this element alone.

		Related
		-------
		transfer_matrix : The full 6x6 ray matrix (rotation included).
		Lens.transfer_block : The thick-body override.
		Microscope.conjugate_planes : Accumulates these to locate planes.

		Notes
		-----
		At ``dz = length`` this reproduces the corresponding sub-block of
		:meth:`transfer_matrix` up to the ``cos(K L)`` factor the Larmor
		rotation applies to a thick lens's x-block — verified element by
		element in the test suite.
		"""
		L = getattr(self, "length", 0) or 0.0
		step = L if dz is None else float(dz)
		power = 0.0
		if isinstance(self, Quadrapole):
			power = self.focal_powers[0 if axis == 'x' else 1]
		elif hasattr(self, "focal_power"):
			power = self.focal_power
		if power == 0:
			return xp.asarray([[1.0, step], [0.0, 1.0]])		# free space, exact
		if L == 0:
			return xp.asarray([[1.0, 0.0], [-power, 1.0]])		# a thin element IS a kick
		raise NotImplementedError(
			f"{type(self).__name__} {self.name or ''!r} has a finite length "
			f"({L} m) and focusing power, but no partial propagator. Splitting it "
			"into a kick between drifts is exactly the approximation the scaled "
			"path was corrected to avoid, so it is not done here: give this class "
			"a transfer_block override carrying its body's own law (see "
			"Lens.transfer_block / Quadrapole.transfer_block), or model the "
			"element as thin (length = 0).")

	@property
	def screen(self):
		"""The screen this element imprints, or ``1`` when it imprints none.

		A **screen** is what the field is multiplied by at this element's plane:
		real data means a phase χ in radians (applied as ``exp(iχ)``), complex
		data means a transmission ``T`` applied directly, carrying amplitude and
		phase together the way a fabricated plate does.

		Only a **supplied** screen lives here. A screen derivable from
		:attr:`aberrations` is deliberately absent: the coefficients are its
		storage, and they are far smaller than the grid they generate, so
		:meth:`phase_shift` recomputes it on the grid actually in use rather
		than pinning an array to one sampling.

		When nothing is supplied this returns the scalar ``1`` — the identity
		for the operation a screen takes part in, so ``field * element.screen``
		is a genuine no-op with nothing allocated.

		Returns
		-------
		Signal, numpy.ndarray, or int
			The supplied screen, or ``1``.

		Raises
		------
		None

		Related
		-------
		_has_screen : Ask whether one is present without type-testing this.
		phase_shift : Where a supplied screen is combined with a generated one.
		waveoptics.apply_phase : The real/complex convention.

		Notes
		-----
		The return type is deliberately polymorphic, so code that needs to
		*branch* rather than multiply should ask :meth:`_has_screen` instead of
		testing what came back.

		Examples
		--------
		>>> Element().screen
		1
		"""
		return 1 if self._screen is None else self._screen

	@screen.setter
	def screen(self, value) -> None:
		"""Supply, replace, or clear this element's screen.

		Parameters
		----------
		value : Signal, numpy.ndarray, or None
			The screen to store. ``None`` clears it, restoring the ``1``
			identity. ``1`` is accepted and treated as ``None``, so a value
			round-tripped through the getter clears rather than storing a
			meaningless scalar.

		Returns
		-------
		None

		Raises
		------
		None
		"""
		self._screen = None if (value is None or (xp.ndim(value) == 0 and value == 1)) else value

	def _has_screen(self) -> bool:
		"""Whether a screen was supplied to this element.

		The predicate that lets callers branch without type-testing what
		:attr:`screen` returned — the getter is polymorphic by design, and
		``isinstance`` checks scattered through the propagators would be the
		cost of that.

		Returns
		-------
		bool
			True when a screen is stored.

		Raises
		------
		None

		Related
		-------
		screen : The value this reports on.
		"""
		return self._screen is not None

	def _screen_data(self):
		"""The supplied screen's array and its own sample spacings.

		A screen supplied as a calibrated Signal knows the grid it was made on,
		which is what lets it be resampled onto the propagation grid. A bare
		array does not, so it reports ``None`` spacings and may only be used on
		a grid of its own shape.

		Parameters
		----------
		None

		Returns
		-------
		tuple
			``(data, dx, dy)`` with ``data`` ``None`` when no screen is
			supplied, and ``dx``/``dy`` ``None`` for an uncalibrated array.

		Raises
		------
		None

		Related
		-------
		_combine_screen : The consumer.
		"""
		if self._screen is None:
			return None, None, None
		data = xp.asarray(getattr(self._screen, "data", self._screen))
		dims = getattr(self._screen, "dimensions", None)
		if dims is None:
			return data, None, None
		from .seashells import grid_of
		try:
			_ny, _nx, dy, dx = grid_of(dims)
		except Exception:					# an array-like without a usable calibration
			return data, None, None
		return data, float(dx), float(dy)

	def _combine_screen(self, chi, shape:tuple, dx:float=None, dy:float=None):
		r"""Merge a generated phase with this element's supplied screen.

		Both are screens, so they compose by multiplying transmissions. Two
		real phases are added instead, which is the same thing
		(:math:`e^{i\chi_1}e^{i\chi_2} = e^{i(\chi_1+\chi_2)}`) but keeps the
		result real — worth doing, because a real screen is half the memory and
		is the only form the sampling guard can check.

		A supplied screen on a different grid is **resampled** when it carries
		its own calibration (i.e. it was supplied as a Signal), and refused
		when it does not, because there is then nothing to resample *from*.

		Parameters
		----------
		chi : xp.ndarray or None
			Generated phase (radians), or ``None`` when the element generates
			none.
		shape : tuple of int
			Shape of the grid being propagated on.
		dx, dy : float, optional
			Sample spacings of that grid (metres), by default ``None``. Needed
			only to resample a supplied screen; without them a shape match is
			required.

		Returns
		-------
		xp.ndarray or None
			The combined screen, or ``None`` when there is nothing to apply.

		Raises
		------
		ValueError
			If a supplied screen's shape differs from the propagation grid and
			it cannot be resampled — either it carries no calibration, or the
			propagation grid's spacings were not passed in.

		Related
		-------
		screen : Where the supplied half comes from.
		waveoptics.resample_screen : The bilinear resampling, and why bilinear.
		waveoptics.apply_phase : Applies the result.
		"""
		supplied, sdx, sdy = self._screen_data()
		if supplied is None:
			return chi
		if tuple(supplied.shape) != tuple(shape):
			if sdx is None or dx is None:
				raise ValueError(
					f"screen on {self.name or type(self).__name__!r} is {tuple(supplied.shape)} "
					f"but the wave grid is {tuple(shape)}, and it cannot be resampled: "
					+ ("it carries no sample calibration (supply it as a Signal, not a bare "
					   "array)." if sdx is None else
					   "the propagation grid's spacings were not supplied."))
			from .waveoptics import resample_screen
			supplied = resample_screen(supplied, sdx, sdy, tuple(shape), dx, dy)
		elif sdx is not None and dx is not None and not (
				xp.isclose(sdx, dx) and xp.isclose(sdy, dy)):
			# same shape, different pitch: still a different grid
			from .waveoptics import resample_screen
			supplied = resample_screen(supplied, sdx, sdy, tuple(shape), dx, dy)
		if chi is None:
			return supplied
		if xp.iscomplexobj(supplied) or xp.iscomplexobj(chi):
			# transmissions multiply; a real phase becomes exp(i*chi) first
			T_s = supplied if xp.iscomplexobj(supplied) else xp.exp(1j * supplied)
			T_c = chi if xp.iscomplexobj(chi) else xp.exp(1j * chi)
			return T_s * T_c
		return supplied + chi				# both real: phases add, stays real

	def aberration_kick(self, r0:xp.ndarray):
		r"""This element's **non-linear** angular kick, from its aberrations.

		:meth:`transfer_matrix` can only express optics that are linear in the
		ray vector, which is the paraxial approximation. Everything an
		aberration is lives outside it. This is the companion declaration: the
		element states the extra deflection, and the generic
		:meth:`propagate_ray` applies it — the same declare/consume split the
		matrix already uses, so no element needs its own propagation method.

		The kick is *not* written per aberration. It is the gradient of the
		element's wave aberration function,
		:math:`\Delta\theta = k^{-1}\nabla\chi`, which is exact in the eikonal
		limit at **every** order, so the same code carries ``C10`` defocus,
		``C30`` spherical and ``C56`` sixfold alike:
		:meth:`aberrations.Aberrations.deflection_at` supplies it and this
		method only decides *where along the element* it acts. An element with
		no :attr:`aberrations` returns ``None`` — "I am exactly my matrix" —
		which keeps aberration-free columns bit-for-bit unchanged.

		Parameters
		----------
		r0 : xp.ndarray
			Rays **entering** the element, shape ``(n_rays, len(convention))``.

		Returns
		-------
		tuple of xp.ndarray or None
			``(delta_x, delta_y, delta_xt, delta_yt)`` — offsets in metres and
			radians, each shape ``(n_rays,)`` — or ``None`` when the element is
			purely linear. A point-like element contributes angle only, but a
			**body** also displaces the ray: its aberration acts part-way
			through and the remaining length converts that kick into a position
			offset as well.

		Raises
		------
		None
			An element with no aberrations, or no focal power to define a pupil
			angle, returns ``None`` rather than raising.

		Related
		-------
		aberrations.Aberrations.deflection_at : The physics, for all orders.
		aberrations.Aberrations.phase_at : The same function on the wave path.
		transfer_matrix : The linear part, which this deliberately does not touch.
		propagate_ray : The generic consumer.

		Notes
		-----
		A **thin** element (``length == 0``) takes one impulsive kick at its
		plane, which is exact. A **thick** body distributes the perturbation
		along its length: a slice ``dz`` acts on the *local* ray height, and the
		remaining body then carries that kick to the exit, turning part of it
		into a position offset. To first order,

		.. math::

			\Delta x_{exit} = \frac{1}{L}\int_0^L\!\Delta\theta_x\big(r(z)\big)\,
			   B(L-z)\,dz, \quad
			\Delta\theta_{exit} = \frac{1}{L}\int_0^L\!
			   \Delta\theta_x\big(r(z)\big)\,D(L-z)\,dz

		with :math:`B, D` entries of the body's own :meth:`transfer_block` over
		the *remaining* length, and the :math:`1/L` chosen so the
		:math:`L \to 0` limit is the thin kick. The integral is Simpson over 64
		slices.

		This matters on a real objective. OL1 in ``basic_column`` is 10 mm thick
		with :math:`KL = 1.30`, and putting the whole aberration at its entrance
		face over-estimates the exit angle by **3.3x**: :math:`r(z)` falls as the
		body focuses, and the kick from each slice is then itself focused by the
		rest of the body. Both effects reduce it, and the weight
		:math:`D(L-z) = \cos K(L-z)` is what makes the factor 0.31 rather than
		the 0.51 that :math:`\int\cos^3` alone would suggest.

		The paraxial planes from
		:meth:`assemblies.Microscope.conjugate_planes` are unaffected by this by
		construction: they are properties of the matrix, and aberration is
		defined as the *departure* from them. That is why the kick is kept out
		of the matrix rather than linearized into it.
		"""
		ab = getattr(self, "aberrations", None)
		P = getattr(self, "focal_power", 0.0) or 0.0
		if not ab or P == 0:
			return None
		x = r0[:, columnByName("x")]
		y = r0[:, columnByName("y")]
		L = self.length or 0.0
		if L <= 0:							# thin: one impulsive kick, exact
			dxt, dyt = ab.deflection_at(x, y, P)
			z = xp.zeros_like(x)
			return z, z.copy(), dxt, dyt
		# Thick: the perturbation is DISTRIBUTED along the body. A slice dz acts
		# on the LOCAL ray height, and the rest of the body then turns that kick
		# into a position offset too, so integrating is not the same as placing
		# the whole thing at one face.
		xt = r0[:, columnByName("xt")]
		yt = r0[:, columnByName("yt")]
		n = 64								# Simpson over the body
		zs = xp.linspace(0.0, L, n + 1)
		w = xp.ones(n + 1) ; w[1:-1:2] = 4.0 ; w[2:-1:2] = 2.0
		w = w * (1.0 / n) / 3.0				# note: no L, so L -> 0 gives the thin kick
		dx = xp.zeros_like(x) ; dy = xp.zeros_like(y)
		dxt = xp.zeros_like(x) ; dyt = xp.zeros_like(y)
		for zi, wi in zip(zs, w):
			A_i, B_i = self.transfer_block(dz=float(zi))[0]			# to the slice
			x_i = A_i * x + B_i * xt
			y_i = A_i * y + B_i * yt
			kx, ky = ab.deflection_at(x_i, y_i, P)
			rest = self.transfer_block(dz=float(L - zi))			# and onward
			B_u, D_u = float(rest[0, 1]), float(rest[1, 1])
			dx = dx + wi * kx * B_u
			dy = dy + wi * ky * B_u
			dxt = dxt + wi * kx * D_u
			dyt = dyt + wi * ky * D_u
		return dx, dy, dxt, dyt

	def _scaled_segment(self):
		r"""Report this element as a *segment* for the scaled wave path, if it is one.

		Most elements act on the scaled representation as a point event: a
		curvature kick and/or a phase screen (:meth:`phase_shift` with
		``scaled=True``), sandwiched between two half-length free segments. A
		**thick round lens** is different — it is a quadratic-index *medium*, so
		the scaled factorization can carry it exactly as one segment whose scale
		law is sinusoidal instead of linear, with no screen and no kick. This
		hook lets such an element say so — and a thick **quadrupole** likewise,
		with opposite curvature on the two axes.

		Driver-only, hence private: it is consumed by
		:meth:`_propagate_wave_scaled` and never called by user code. The
		boolean half of the question is derivable from ``length`` alone, but the
		*payload* is not — a drift is a medium too (``K = 0``, free space) and a
		dipole is a medium that is not quadratic-index, so the driver needs the
		law and the strength, not just "do I occupy space".

		Returns
		-------
		tuple or None
			``('quadratic', kappa, larmor)`` for a constant-curvature medium, or
			``None`` (the base class) meaning "treat me as a point event inside
			free space". ``kappa`` is the signed curvature in 1/metres² of
			``u'' + kappa·u = 0`` — positive focuses (harmonic), negative
			defocuses (hyperbolic) — scalar for an isotropic medium or an
			``(x, y)`` pair for an astigmatic one. ``larmor`` is the body's
			rotation angle in radians, declared by the element because only it
			knows whether it has an axial field.

		Related
		-------
		phase_shift : The point-event contract used when this returns ``None``.
		waveoptics.propagate_quadratic_segment_scaled : Consumes the ``'quadratic'`` case.

		Notes
		-----
		Only the scaled/hybrid wave paths consult this; the fixed-grid path still
		slices a thick element into half-length kernels around a phase screen,
		because a fixed grid cannot follow the medium's scale law.
		"""
		return None

	def _phase_program(self, dimensions, wavelength:float, chi, name:str):
		"""Assemble the fixed-grid phase program ``[kernel(L/2), screen, kernel(L/2)]``.

		Shared by the concrete ``phase_shift`` implementations: wraps the
		element's real-space phase χ (if any) between two half-length free
		segments; a screen-less element yields a single full-length segment.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`phase_shift`).
		wavelength : float
			Wavelength (metres).
		chi : xp.ndarray or None
			Real-space phase screen, or ``None`` for a pure free segment.
		name : str
			Screen item name.

		Returns
		-------
		list
			Phase Signals in application order (possibly empty for a
			zero-length, screen-less element).
		"""
		from .seashells import grid_of
		ny, nx, dy, dx = grid_of(dimensions)
		L = self.length
		items = []
		if chi is None:
			if L != 0:
				items.append(_kernel_item(ny, nx, dy, dx, wavelength, L))
			return items
		if L != 0:
			items.append(_kernel_item(ny, nx, dy, dx, wavelength, L / 2))
		items.append(_screen_item(chi, dx, dy, name))
		if L != 0:
			items.append(_kernel_item(ny, nx, dy, dx, wavelength, L / 2))
		return items

	def propagate_ray(self, r0:xp.ndarray,
					  z:float=None, z0:float=0) -> xp.ndarray:
		"""propagate an array through an element.

		Parameters
		----------
		r0 : xp.ndarray
			List of rays with possible initial conditions (x, θx, y, θy, E).
		z : None | int | float | xp.ndarray, optional
			Positions in the element to propagate to by default None
		z0 : None | float, optional
			Initial position of the element, by default 0

		Returns
		-------
		xp.ndarray
			List of propagated rays with initial condition (x, θx, y, θy, z, E)
		"""
		m = self.transfer_matrix()
		rf = xp.einsum('mn,in->im', m, r0) # matrix multiplication for a "list of vectors"
		# additive terms: z_new = z_old+length (rotation is handled separately, see apply_rotation)
		rf[:,columnByName("z")] += self.length
		rf[:,columnByName("x")] += getattr(self,"shift_x",0)
		rf[:,columnByName("y")] += getattr(self,"shift_y",0)
		rf[:,columnByName("xt")] += getattr(self,"tilt_x",0)
		rf[:,columnByName("yt")] += getattr(self,"tilt_y",0)
		# aberration: the part of the element's optics that is NOT a matrix.
		# Declared by the element, applied here, exactly as transfer_matrix is.
		kick = self.aberration_kick(r0)
		if kick is not None:
			dx, dy, dxt, dyt = kick
			rf[:,columnByName("x")] += dx
			rf[:,columnByName("y")] += dy
			rf[:,columnByName("xt")] += dxt
			rf[:,columnByName("yt")] += dyt

		return rf

	def apply_intensity(self, I:xp.ndarray, r0:xp.ndarray) -> xp.ndarray:
		"""Return the beam intensity after passing through this element.

		Intensity is tracked as a parallel array rather than as a ray coordinate.
		Most elements leave it unchanged; overriding classes (e.g. ``Aperture``)
		attenuate it. Called by the section/microscope drivers *before*
		``propagate_ray`` transforms the rays, so ``r0`` is the incoming ray table.

		Parameters
		----------
		I : xp.ndarray
			Per-ray intensity entering the element, shape ``(n_rays,)``.
		r0 : xp.ndarray
			Incoming ray table (geometric coordinates), used by elements whose
			attenuation depends on ray positions (e.g. an aperture).

		Returns
		-------
		xp.ndarray
			Per-ray intensity leaving the element, shape ``(n_rays,)``.

		Related
		-------
		Aperture.apply_intensity : Attenuates intensity by the cropped-area fraction.
		"""
		return I

	def apply_rotation(self, R:xp.ndarray) -> xp.ndarray:
		"""Return the cumulative Larmor rotation after this element.

		Rotation is tracked as a parallel array rather than as a ray coordinate.
		Thick lenses accumulate rotation via ``self.rotation`` (set as a side effect
		of :meth:`transfer_matrix`), so this must be called *after* ``propagate_ray``.

		Parameters
		----------
		R : xp.ndarray
			Per-ray cumulative rotation (radians) entering the element,
			shape ``(n_rays,)``.

		Returns
		-------
		xp.ndarray
			Per-ray cumulative rotation leaving the element, shape ``(n_rays,)``.

		Related
		-------
		Lens.transfer_matrix : Sets ``self.rotation`` for finite-thickness lenses.
		"""
		return R + getattr(self, "rotation", 0)

	def propagate_moments(self, mu:xp.ndarray, Sigma:xp.ndarray) -> tuple:
		r"""Propagate the beam's first and second moments through this element.

		Describes the ensemble by a mean state ``mu`` and covariance ``Sigma`` over
		the geometric phase space and transports them analytically through the same
		ray-transfer matrix used by :meth:`propagate_ray`:

		.. math::

			\mu' = M \mu + a, \qquad \Sigma' = M \Sigma M^{\mathsf T}

		where ``M`` is :meth:`transfer_matrix` and ``a`` collects the affine terms
		(drift length, dipole tilt, ...). Covariance is invariant to the affine
		offset ``a``, so the mean is obtained by reusing :meth:`propagate_ray`.

		Parameters
		----------
		mu : xp.ndarray
			Mean state vector, shape ``(len(convention),)``.
		Sigma : xp.ndarray
			Covariance matrix, shape ``(len(convention), len(convention))``.

		Returns
		-------
		tuple of xp.ndarray
			``(mu_out, Sigma_out)`` after the element.

		Related
		-------
		propagate_ray : Ray transport sharing the same transfer matrix.
		Source.moments : Seeds the initial ``(mu, Sigma)``.

		Notes
		-----
		Valid in the paraxial/linear regime, where the transfer matrix acts as a
		linear map on phase space and a Gaussian ensemble stays Gaussian.
		"""
		M = self.transfer_matrix()
		Sigma_out = M @ Sigma @ M.T
		mu_out = self.propagate_ray(mu.reshape(1, -1))[0]
		return mu_out, Sigma_out

	def propagate_wave(self, signal, mode:Literal['fixed','scaled','hybrid']='fixed',
					   s_min:float=1e-3, log:list=None, absorb:float=0.1,
				   crossover:Literal['flat','jump']='flat', rotate:bool=False):
		r"""Propagate a wavefield through this element in the selected wave mode.

		The one wave-optics analog of :meth:`propagate_ray`, covering all three
		wave representations via ``mode``:

		- ``'fixed'`` — paraxial wave on a fixed physical grid. Consumes the
		  element's :meth:`phase_shift` program (space-tagged screens and
		  free-segment kernels).
		- ``'scaled'`` — scaled-Fresnel wave (handoff Eqs 23–48): the state is
		  the reduced field ``U(ξ, η)`` of ``ψ = (1/s)·U·exp[ik(x²+y²)/2R]``
		  plus the frame scalars ``(s, R, τ)``. A finite length is split as
		  free ``L/2`` → element action → free ``L/2``;
		  :meth:`phase_shift(scaled=True)` supplies the split into curvature
		  (``1/R⁺ = 1/R⁻ − power``, Eq 45) and a residual screen applied to U
		  under a sampling guard (Eqs 47–48). A single frame: propagation
		  raises before a beam crossover (the frame's ``s = 0`` singularity).
		- ``'hybrid'`` — the scaled representation with automatic frame
		  switching (:func:`waveoptics.propagate_free_scaled_hybrid`):
		  converging frames flatten before their crossover, the wave crosses
		  the real focus on a flat frame — the crossover (back-focal) plane is
		  logged — and re-factors onto a diverging frame past it.

		Parameters
		----------
		signal : Signal or seashells._Wavefield or seashells._ScaledWavefield
			Incoming wavefield: physical for ``'fixed'``, scaled for
			``'scaled'``/``'hybrid'`` (from :meth:`Source.wave`).
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Wave representation, by default ``'fixed'``.
		s_min : float, optional
			Backstop crossover guard for the scaled/hybrid paths (handoff
			Eq 52), by default ``1e-3``. Ignored for ``'fixed'``.
		log : list, optional
			Scaled/hybrid only: interior frame-switch and crossover planes are
			appended to this list as scaled Signals (tags ``flatten`` /
			``crossover`` / ``rediverge`` in metadata). ``None`` (default)
			discards them.
		absorb : float, optional
			Scaled/hybrid only: absorbing-boundary margin fraction (default
			0.1). Field diffracting out of the modeled field of view is
			absorbed (physically: those electrons leave the beam) instead of
			wrapping around the periodic grid and interfering with the beam
			as an axis-aligned artifact. ``0`` restores pure periodic
			propagation (exact energy conservation).

		Returns
		-------
		Signal or seashells._Wavefield or seashells._ScaledWavefield
			Wavefield at the element exit in the same representation.

		Raises
		------
		ValueError
			Unknown ``mode``; from the scaled path: the single frame reaching
			its crossover, or an under-sampled screen.

		Related
		-------
		phase_shift : The per-element wave physics consumed by every mode.
		waveoptics.propagate_free_scaled_hybrid : The hybrid crossover engine.
		Source.wave : Seeds the matching initial state per mode.

		Notes
		-----
		Larmor rotation of thick lenses is not applied to the wavefield
		(documented approximation); on the scaled paths a thick element is
		treated as thin between two half-length free segments.
		"""
		if mode == 'fixed':
			return self._propagate_wave_fixed(signal)
		if mode in ('scaled', 'hybrid'):
			return self._propagate_wave_scaled(signal, hybrid=(mode == 'hybrid'), rotate=rotate,
											   s_min=s_min, log=log, absorb=absorb,
											   crossover=crossover)
		raise ValueError(f"Unknown wave mode {mode!r}; expected 'fixed', 'scaled', or 'hybrid'.")

	def _propagate_wave_fixed(self, signal):
		"""Fixed-grid wave step: apply the element's phase program.

		Parameters
		----------
		signal : Signal or seashells._Wavefield
			Incoming physical wavefield.

		Returns
		-------
		Signal or seashells._Wavefield
			Wavefield at the element exit, on the same transverse grid.

		Related
		-------
		propagate_wave : The mode-dispatching public method.
		"""
		from .waveoptics import apply_phase
		from .seashells import make_wavefield_signal, read_wavefield, phase_space_of
		data, dx, dy, wavelength, z = read_wavefield(signal)
		dimensions = getattr(signal, "dimensions", None)
		if dimensions is None:
			dimensions = (data.shape, dx, dy)
		for phase in self.phase_shift(dimensions, wavelength):
			data = apply_phase(data, phase.data, phase_space_of(phase))
		z_out = (z if z is not None else 0.0) + self.length
		return make_wavefield_signal(data, dx, dy, wavelength, z=z_out,
									 name=getattr(signal, "name", "wavefield"))

	def _propagate_wave_scaled(self, signal, hybrid:bool=False, s_min:float=1e-3,
							   log:list=None, absorb:float=0.1,
							   crossover:Literal['flat','jump']='flat',
							   rotate:bool=False):
		"""Scaled-frame wave step: free L/2 → element action → free L/2.

		Parameters
		----------
		signal : Signal or seashells._ScaledWavefield
			Incoming scaled wavefield.
		hybrid : bool, optional
			Route the free segments through the frame-switching engine
			(:func:`waveoptics.propagate_free_scaled_hybrid`), by default False
			(single frame; raises before a crossover).
		s_min : float, optional
			Backstop crossover guard, by default ``1e-3``.
		log : list, optional
			Hybrid only: interior logged planes are appended here as scaled
			Signals.

		Returns
		-------
		Signal or seashells._ScaledWavefield
			Scaled wavefield at the element exit (same ξ/η grid; updated
			``s``/``R``/``τ``/``z`` and crossover marker).

		Related
		-------
		propagate_wave : The mode-dispatching public method.
		"""
		from .waveoptics import (propagate_free_scaled, propagate_free_scaled_hybrid,
								 apply_thin_lens_scaled, apply_phase,
								 axis_components, join_axes)
		from .seashells import (make_scaled_wavefield_signal, read_scaled_wavefield,
								scaled_frame_crossover, phase_space_of)
		U, dxi, deta, wavelength, s, R, tau, z = read_scaled_wavefield(signal)
		z = z if z is not None else 0.0
		z_cross = scaled_frame_crossover(signal)
		name = getattr(signal, "name", "scaled wavefield")

		def tau_add(t, dt):
			# scalar-or-pair addition (anisotropic frames carry per-axis tau)
			tx, ty = axis_components(t)
			dtx, dty = axis_components(dt)
			return join_axes(tx + dtx, ty + dty)

		def free(U, s, R, tau, z, z_cross, dz):
			if dz == 0:
				return U, s, R, tau, z, z_cross
			if hybrid:
				U, s, R, dt, z, z_cross, logged = propagate_free_scaled_hybrid(
					U, dxi, deta, wavelength, dz, s, R, z, z_cross, s_min=s_min,
					absorb=absorb, crossover=crossover)
				if log is not None:
					for tag, U_l, s_l, R_l, dt_l, z_l, zc_l in logged:
						log.append(make_scaled_wavefield_signal(
							U_l, dxi, deta, wavelength, s_l, R_l, tau_add(tau, dt_l),
							z=z_l, z_cross=zc_l, tag=tag, name=name))
				tau = tau_add(tau, dt)
			else:
				U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, dz, s, R,
													s_min=s_min, absorb=absorb)
				tau = tau_add(tau, dt) ; z += dz
			return U, s, R, tau, z, z_cross

		L = getattr(self, "length", 0)
		segment = self._scaled_segment()
		if segment is not None and L != 0:
			# quadratic-index medium: one exact segment, no screen and no kick
			from .waveoptics import (propagate_quadratic_segment_scaled,
									 propagate_quadratic_segment_hybrid)
			_kind, kappa, larmor = segment
			spin = larmor if rotate else 0.0
			if hybrid:
				# a crossover can fall INSIDE the body; the hybrid traversal
				# switches frames there rather than leaving the free engine to
				# flatten around the element and misplace the plane
				U, s, R, dt, z, z_cross, logged = propagate_quadratic_segment_hybrid(
					U, dxi, deta, wavelength, L, s, R, kappa, z, z_cross,
					s_min=s_min, absorb=absorb, crossover=crossover, rotate=spin)
				if log is not None:
					for tag, U_l, s_l, R_l, dt_l, z_l, zc_l in logged:
						log.append(make_scaled_wavefield_signal(
							U_l, dxi, deta, wavelength, s_l, R_l, tau_add(tau, dt_l),
							z=z_l, z_cross=zc_l, tag=tag, name=name))
			else:
				U, s, R, dt = propagate_quadratic_segment_scaled(
					U, dxi, deta, wavelength, L, s, R, kappa, s_min=s_min,
					absorb=absorb, rotate=spin)
				z += L
			tau = tau_add(tau, dt)
			return make_scaled_wavefield_signal(U, dxi, deta, wavelength, s, R, tau,
												z=z, z_cross=z_cross, name=name)
		U, s, R, tau, z, z_cross = free(U, s, R, tau, z, z_cross, L / 2 if L != 0 else 0)
		power, screen = self.phase_shift((U.shape, dxi, deta), wavelength, scaled=True, s=s)
		if xp.ndim(power) > 0 or power != 0:		# scalar or per-axis (quadrupole) power
			s, R = apply_thin_lens_scaled(s, R, power)
		if screen is not None:
			_check_screen_sampling(screen.data, self.name or type(self).__name__)
			U = apply_phase(U, screen.data, phase_space_of(screen))
		U, s, R, tau, z, z_cross = free(U, s, R, tau, z, z_cross, L / 2 if L != 0 else 0)
		return make_scaled_wavefield_signal(U, dxi, deta, wavelength, s, R, tau, z=z,
											z_cross=z_cross, name=name)

	def propagate(self, *args, kind:Literal["ray","rays","moments","envelope","covariance","wave","wave-scaled","wave_scaled","wave-hybrid","wave_hybrid"]="ray", **kwargs):
		"""Unified propagation dispatcher across the three modes.

		Routes to :meth:`propagate_ray`, :meth:`propagate_moments`, or
		:meth:`propagate_wave` according to ``kind``; all positional and keyword
		arguments are forwarded unchanged to the selected method.

		Parameters
		----------
		*args
			Positional arguments forwarded to the selected ``propagate_*`` method.
		kind : {'ray','rays','moments','envelope','covariance','wave'}, optional
			Propagation mode, by default ``'ray'``. ``'moments'``/``'envelope'``/
			``'covariance'`` select beam-envelope propagation.
		**kwargs
			Keyword arguments forwarded to the selected ``propagate_*`` method.

		Returns
		-------
		object
			Whatever the selected ``propagate_*`` method returns.

		Raises
		------
		ValueError
			If ``kind`` is not a recognized propagation mode.

		Examples
		--------
		>>> element.propagate(r0, kind="ray")            # doctest: +SKIP
		>>> element.propagate(mu, Sigma, kind="moments") # doctest: +SKIP
		>>> element.propagate(field, kind="wave")        # doctest: +SKIP
		"""
		method, forced = _propagate_method_name(kind)
		return getattr(self, method)(*args, **{**kwargs, **forced})

class Source(Element):
	"""Source element class. Source element can be put in a MicroscopeSection and then propagating "through" the section will mean the starting rays r0 are generated by the Source (instead of requiring the user pass in starting rays)

		Parameters
		----------
		name : str, optional
			Name of the Source, by default ''
		size : tuple, optional
			size in x and y: rays will be emitted from a square grid of points
		np_xy : tuple, optional
			number of grid points in x and y
		angle : tuple, optional
			maximum angle in x and y: rays will be emitted at multiple angles from each grid point
		na_xy : tuple, optional
			number of angles for emitted rays in x and y
		position : float, optional
			The position of the element along the z-axis, by default 0
		voltage : float, optional
			Accelerating voltage in kilovolts. When provided, it seeds the electron
			``wavelength`` (used by wave-optics/envelope propagation) and populates the
			per-ray ``E`` (beam energy, keV) column. When ``None`` (default), ``E``
			stays 0 and no wavelength is defined, preserving purely geometric behavior.
		wave_shape : tuple, optional
			Wave-optics grid ``(ny, nx)``, by default ``(128, 128)``.
		wave_extent : float, optional
			Wave-optics grid physical size (metres); ``None`` (default) derives
			``8 * max(size)``.
		wave_kind : {'plane', 'gaussian', 'point', 'aperture'}, optional
			Which initial wavefunction :meth:`wave` generates, by default
			``'gaussian'``. ``'aperture'`` is the flat-intensity hard-aperture
			wave Θ(a−r) and requires ``aperture_radius``.
		aperture_radius : float, optional
			Aperture radius ``a`` (metres) for ``wave_kind='aperture'``; must
			fit inside the grid half-extent.

		Attributes
		----------
		voltage : float or None
			Accelerating voltage in kilovolts, or ``None`` if unset.
		wavelength : float or None
			Relativistic electron wavelength in metres, or ``None`` if ``voltage`` is unset.
		"""

	def __init__(self, name:str=None,
			size:tuple=(2e-3,2e-3), # size in x and y (square grid)
			np_xy:tuple=(3,3),		# number of grid points in x and y. (0,0) --> point-source. (1,1) --> single ray at x,y=size
			angle:tuple=(1,1),		# angles in x,y (ranges of xt yt)
			na_xy:tuple=(3,3),		# number of angles. (0,0) --> parallel rays. (1,1) --> ray at xt,yt=angle only
			position:float=None,
			voltage:float=None,
			wave_shape:tuple=(128,128),	# wave-optics grid (ny, nx)
			wave_extent:float=None,		# wave-optics grid physical size (m); None -> derived from size
			wave_kind:Literal['plane','gaussian','point','aperture']='gaussian',
			aperture_radius:float=None) -> SEASerializable:	# radius (m) for wave_kind='aperture'
		super().__init__(name=name, kind='Source')

		self.size = size
		self.np_xy = np_xy
		self.angle = angle
		self.na_xy = na_xy
		self._position = position
		self.length = 0
		self.strength = 0
		self.calibration = None
		self.voltage = voltage
		# derived: relativistic wavelength (metres); None when voltage is unset
		from .utilities import relativistic_wavelength
		self.wavelength = relativistic_wavelength(voltage) if voltage is not None else None
		# wave-optics initial-wave parameters
		self.wave_shape = wave_shape
		self.wave_extent = wave_extent
		self.wave_kind = wave_kind
		self.aperture_radius = aperture_radius

	# Source term, initialize rays at sweep of angles and positions
	def rays(self):
		#print("SOURCE GENERATING RAYS",self.np_xy,self.size,self.na_xy,self.angle)
		xs=xp.zeros(1) ; ys=xp.zeros(1) # central ray only by default if np_xy is zero
		if self.np_xy[0]:
			xs=xp.linspace(-self.size[0],self.size[0],self.np_xy[0])
		if self.np_xy[1]:
			ys=xp.linspace(-self.size[1],self.size[1],self.np_xy[1])
		xts=xp.zeros(1) ; yts=xp.zeros(1) # zero-angle ray only by default if na_xy is zero
		if abs(self.angle[0])>0 and self.na_xy[0]:
			xts=xp.linspace(-self.angle[0],self.angle[0],self.na_xy[0])
		if abs(self.angle[1])>0 and self.na_xy[1]:
			yts=xp.linspace(-self.angle[1],self.angle[1],self.na_xy[1])
		shape=(len(xs),len(ys),len(xts),len(yts))
		array=xp.zeros((len(xs)*len(ys)*len(xts)*len(yts),4))
		array[:,0]=(xs[:,None,None,None]*xp.ones(shape)).flat
		array[:,1]=(ys[None,:,None,None]*xp.ones(shape)).flat
		array[:,2]=(xts[None,None,:,None]*xp.ones(shape)).flat
		array[:,3]=(yts[None,None,None,:]*xp.ones(shape)).flat
		array=fix_ray_dims(array,["x","y","xt","yt"])
		if self.voltage is not None:					# beam energy (keV) rides in the E column when defined
			array[:,columnByName("E")] = self.voltage
		return array

	# dummy propagation in case someone tries to propagate through since this is technically an element
	def propagate_ray(self, r0:xp.ndarray, **kwargs) -> xp.ndarray:
		return r0

	def moments(self) -> tuple:
		r"""Seed the initial mean and covariance for beam-envelope propagation.

		The analog of :meth:`rays` for :meth:`propagate_moments`. Builds a centered
		mean (``mu0 = 0``, with the ``E`` component set to ``voltage`` when defined)
		and a diagonal covariance whose entries are the squared source ``size`` (real
		space) and ``angle`` (angular spread), i.e. these are treated as RMS values.

		Returns
		-------
		tuple of xp.ndarray
			``(mu0, Sigma0)`` with shapes ``(len(convention),)`` and
			``(len(convention), len(convention))``.

		Related
		-------
		rays : Ray-mode analog that generates the initial ray bundle.
		Element.propagate_moments : Consumes the seeded moments.
		"""
		var = xp.zeros(len(convention))
		var[columnByName("x")]  = self.size[0]**2
		var[columnByName("y")]  = self.size[1]**2
		var[columnByName("xt")] = self.angle[0]**2
		var[columnByName("yt")] = self.angle[1]**2
		Sigma0 = xp.diag(var)
		mu0 = xp.zeros(len(convention))
		if self.voltage is not None:
			mu0[columnByName("E")] = self.voltage
		return mu0, Sigma0

	def propagate_moments(self, mu:xp.ndarray, Sigma:xp.ndarray) -> tuple:
		"""Pass moments through unchanged (the source only originates the beam).

		Mirrors :meth:`propagate_ray`, which returns ``r0`` untouched. The driver
		seeds ``(mu, Sigma)`` from :meth:`moments`, so the source's own step is a
		no-op.

		Parameters
		----------
		mu : xp.ndarray
			Mean state vector.
		Sigma : xp.ndarray
			Covariance matrix.

		Returns
		-------
		tuple of xp.ndarray
			The inputs ``(mu, Sigma)`` unchanged.
		"""
		return mu, Sigma

	def wave(self, mode:Literal['fixed','scaled','hybrid']='fixed'):
		"""Build the initial wavefield for wave-optics propagation.

		The wave-mode analog of :meth:`rays` and :meth:`moments` — the source's
		one wavefunction generator. Constructs a 2D scalar field on a calibrated
		grid whose physical extent is ``wave_extent`` (or ``8 * max(size)``
		when unset) sampled at ``wave_shape`` points, of the kind given by
		``wave_kind``: ``'plane'``, ``'gaussian'`` sized by ``size``,
		``'point'``, or ``'aperture'`` (a flat-intensity plane wave clipped at
		``aperture_radius``, via :meth:`_aperture_wave`). Requires a defined
		wavelength (set ``voltage``).

		Parameters
		----------
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Which representation to seed (matching
			:meth:`Element.propagate_wave`), by default ``'fixed'`` — the
			physical wavefield Signal. ``'scaled'``/``'hybrid'`` seed the
			scaled state (handoff Eqs 10–11): the initial frame is ``s = 1``,
			``R = ∞``, ``τ = 0``, so the reduced field is the physical one,
			``U₀ = ψ₀``, with ``Δξ = Δx``.

		Returns
		-------
		Signal or seashells._Wavefield or seashells._ScaledWavefield
			A calibrated wavefield at the source plane in the requested
			representation.

		Raises
		------
		ValueError
			If no wavelength is defined (``voltage`` unset), if the grid extent
			cannot be derived (zero source ``size`` and no ``wave_extent``), if
			``wave_kind`` is not recognized, or if ``wave_kind='aperture'``
			with no ``aperture_radius`` set (or one that does not fit the grid).

		Related
		-------
		_aperture_wave : The Θ(a−r) builder behind ``wave_kind='aperture'``.
		Element.propagate_wave : Transports this wave through an element.
		seashells.make_wavefield_signal : Wraps the array as a calibrated Signal.
		"""
		from .waveoptics import plane_wave, gaussian_field, point_source
		from .seashells import make_wavefield_signal
		if mode in ('scaled', 'hybrid'):
			from .seashells import make_scaled_wavefield_signal, read_wavefield
			data, dx, dy, wavelength, z = read_wavefield(self.wave(mode='fixed'))
			return make_scaled_wavefield_signal(data, dx, dy, wavelength, s=1.0,
												R=xp.inf, tau=0.0, z=z,
												name=(self.name or 'source') + ' scaled wavefield')
		if self.wavelength is None:
			raise ValueError("Source.wave requires a wavelength; construct Source(voltage=<kV>).")
		if self.wave_kind == 'aperture':
			if self.aperture_radius is None:
				raise ValueError("wave_kind='aperture' requires aperture_radius (metres).")
			return self._aperture_wave(self.aperture_radius)
		ny, nx = self.wave_shape
		extent = self.wave_extent if self.wave_extent is not None else 8 * max(self.size)
		if extent <= 0:
			raise ValueError("Cannot derive a wavefield grid extent from a zero source size; pass wave_extent (metres).")
		dx = extent / nx ; dy = extent / ny
		if self.wave_kind == 'plane':
			data = plane_wave((ny, nx))
		elif self.wave_kind == 'gaussian':
			data = gaussian_field((ny, nx), dx, dy, self.size[0], self.size[1])
		elif self.wave_kind == 'point':
			data = point_source((ny, nx))
		else:
			raise ValueError(f"Unknown wave_kind {self.wave_kind!r}; expected 'plane', 'gaussian', 'point', or 'aperture'.")
		z0 = self._position if self._position is not None else 0.0
		return make_wavefield_signal(data, dx, dy, self.wavelength, z=z0,
									 name=(self.name or 'source') + ' wavefield')

	def _aperture_wave(self, radius:float, antialias:bool=True):
		r"""Build a hard-aperture initial wavefield :math:`\psi_0 = \Theta(a - r)`.

		The handoff's reference initial wave (Eq 9): a unit-amplitude sharp
		disk of radius ``radius`` on the source's wave grid
		(``wave_shape``/``wave_extent``). By default the grid holds the
		**band-limited projection** of the sharp disk
		(:func:`waveoptics.bandlimited_disk`): every representable Fresnel
		fringe of the hard edge is preserved exactly, while the above-Nyquist
		edge content — which a point-sampled binary mask folds back and
		propagates as a spurious grid texture — is removed. Requires a defined
		wavelength (set ``voltage``).

		Parameters
		----------
		radius : float
			Aperture radius (metres); must fit inside the grid half-extent.
		antialias : bool, optional
			Use the alias-free band-limited disk, by default True. ``False``
			restores the point-sampled binary mask (comparison/regression use).

		Returns
		-------
		Signal or seashells._Wavefield
			The calibrated hard-aperture wavefield at the source plane.

		Raises
		------
		ValueError
			If no wavelength is defined, or ``radius`` does not fit on the grid.

		Related
		-------
		wave : The wave generator that dispatches here for wave_kind='aperture'.
		waveoptics.bandlimited_disk : The alias-free sharp-disk builder.
		"""
		from .waveoptics import plane_wave, aperture_mask, bandlimited_disk
		from .seashells import make_wavefield_signal
		if self.wavelength is None:
			raise ValueError("Source._aperture_wave requires a wavelength; construct Source(voltage=<kV>).")
		ny, nx = self.wave_shape
		extent = self.wave_extent if self.wave_extent is not None else 8 * max(self.size)
		if extent <= 0 or radius >= extent / 2:
			raise ValueError(f"Aperture radius {radius} m does not fit on the grid half-extent {extent/2} m; "
							 "increase wave_extent.")
		dx = extent / nx ; dy = extent / ny
		if antialias:
			data = bandlimited_disk((ny, nx), dx, dy, radius)
		else:
			data = aperture_mask(plane_wave((ny, nx)), dx, dy, radius, antialias=False)
		z0 = self._position if self._position is not None else 0.0
		return make_wavefield_signal(data, dx, dy, self.wavelength, z=z0,
									 name=(self.name or 'source') + ' aperture wavefield')

	def propagate_wave(self, signal, mode:Literal['fixed','scaled','hybrid']='fixed',
					   s_min:float=1e-3, log:list=None, absorb:float=0.1,
				   crossover:Literal['flat','jump']='flat', rotate:bool=False):
		"""Pass the wavefield through unchanged (the source only originates the beam).

		Mirrors :meth:`propagate_ray`/:meth:`propagate_moments`: the driver
		seeds the wave from :meth:`wave` (in the matching representation), so
		the source's own step is a no-op in every mode.

		Parameters
		----------
		signal : Signal or seashells._Wavefield or seashells._ScaledWavefield
			Incoming wavefield.
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Unused (accepted for driver-signature uniformity).
		s_min : float, optional
			Unused.
		log : list, optional
			Unused.

		Returns
		-------
		Signal or seashells._Wavefield or seashells._ScaledWavefield
			The input ``signal`` unchanged.
		"""
		return signal

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		"""A source originates waves; it imprints no phase (not part of this contract).

		Overrides :meth:`Element.phase_shift` to fail loudly: the source's wave
		role is generating the initial wave (:meth:`wave`), and its propagation
		step is a passthrough.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Unused.
		wavelength : float
			Unused.
		scaled : bool, optional
			Unused.
		s : float, optional
			Unused.

		Returns
		-------
		None
			Never returns.

		Raises
		------
		NotImplementedError
			Always.
		"""
		raise NotImplementedError("Source originates waves (wave/wave_scaled); "
								  "it has no phase_shift.")

class Aperture(Element):
	"""Aperture element class. An aperture serves to crop the beam, and the total beam intensity is reduced dependent on the area of the beam and the area of aperture.

		Parameters
		----------
		name : str, optional
			Name of the Aperture, by default ''
		radius : float, optional
			radius of the round aperture
		calibration : float, optional
			linear scaling factor applied to the radius. e.g. nominally 2mm aperture but fitting tells us it is actually 1.95
		position : float, optional
			The position of the element along the z-axis, by default 0
		"""

	def __init__(self, name:str='', radius:float=0., calibration:float=None, position:float=None) -> SEASerializable:
		super().__init__(name=name, kind='Aperture')
		self._position = position
		self.radius = radius
		self.calibration = calibration

	#def transfer_matrix(self) -> xp.ndarray:
	#	r"""Transfer matrix for ray propogation.
	#	"""
	#	#print("WARNING: APERTURE",self.name,"TRANSFER MATRIX CALLED")
	#	#m = xp.eye(6)#[...,None]*xp.ones_like(s)
	#	m = xp.eye(4) # drift tube updates x from xθ and y from yθ
	#	return fix_mat_dims(m,["x","xt","y","yt"])

	# TWO WAYS TO IMPLEMENT AN APERTURE:
	# 1) set the intensity of any rays "outside" the aperture to zero. this is fine for plotting and we can capture beam current by looking at how many rays are zeroed out. *BUT*, this will be problematic during fitting, as rays which "pop" into and out of view will yield an intensity vs [whatever] function with step edges.
	# def propagate_ray(self, r0:xp.ndarray,
	#				  z:float=None, z0:float=0) -> xp.ndarray:
	#	rf=xp.zeros(r0.shape)+r0
	#	radii=xp.sqrt( r0[:,columnByName("x")]**2 + r0[:,columnByName("y")]**2 )
	#	rf[radii>self.radius,columnByName("I")]=0
	#	return rf
	# 2) aperture can rescale all rays based on the outer ray's position, or the area covered by the rays. we can thus calculate reductions in beam current based on the aperture's reduction in intensity (area cropped out). we're effectively pretending the originating rays were less divergent or something, which is actually sort of what we see IRL; you can't tell the divergence of the beam from the gun because the VOA masks out a bunch of it. This will only work for one aperture in the system though (otherwise second aperture undoes the scaling of the first one? or should we only allow the aperture to scale-down, so if the first aperture scales down, second scales down further (second is smaller), or first scale down, second leaves it alone (second is larger, we'd be able to see our first aperture in the CCD for example). and how do we handle apertures of different shapes??
	def _aperture_scales(self, r0:xp.ndarray) -> tuple:
		"""Return the x and y demagnification factors imposed by the aperture.

		The aperture rescales the beam based on the outermost ray's position
		relative to the aperture radius (see the class-level discussion). The same
		factors drive both the geometric rescaling (in :meth:`propagate_ray`) and
		the intensity attenuation (in :meth:`apply_intensity`), so they are computed
		once here from the incoming rays.

		Parameters
		----------
		r0 : xp.ndarray
			Incoming ray table (geometric coordinates).

		Returns
		-------
		tuple of float
			``(scale_x, scale_y)``, each in ``(0, 1]``.
		"""
		xmax = xp.amax(r0[:,columnByName("x")])
		ymax = xp.amax(r0[:,columnByName("y")])
		scale_x = 1 if xmax<self.radius else self.radius/xmax
		scale_y = 1 if ymax<self.radius else self.radius/ymax
		return scale_x, scale_y

	def propagate_ray(self, r0:xp.ndarray,
					  z:float=None, z0:float=0) -> xp.ndarray:
		scale_x, scale_y = self._aperture_scales(r0)
		#print("Aperture",self.name,"radius",self.radius,"scale x,y",scale_x,scale_y)
		rf=xp.zeros(r0.shape)+r0
		rf[:,columnByName("x")]*=scale_x
		rf[:,columnByName("xt")]*=scale_x
		rf[:,columnByName("y")]*=scale_y
		rf[:,columnByName("yt")]*=scale_y
		return rf

	def apply_intensity(self, I:xp.ndarray, r0:xp.ndarray) -> xp.ndarray:
		"""Attenuate intensity by the fraction of beam area the aperture passes.

		Extends :meth:`Element.apply_intensity`. The transmitted fraction is
		``scale_x * scale_y`` (the cropped-area fraction), matching the geometric
		rescaling applied to the ray positions in :meth:`propagate_ray`.

		Parameters
		----------
		I : xp.ndarray
			Per-ray intensity entering the aperture, shape ``(n_rays,)``.
		r0 : xp.ndarray
			Incoming ray table, used to compute the demagnification factors.

		Returns
		-------
		xp.ndarray
			Attenuated per-ray intensity, shape ``(n_rays,)``.
		"""
		scale_x, scale_y = self._aperture_scales(r0)
		return I * scale_x * scale_y

	def propagate_moments(self, mu:xp.ndarray, Sigma:xp.ndarray) -> tuple:
		"""Pass moments through unchanged (aperture is treated as non-truncating here).

		Overrides :meth:`Element.propagate_moments`. An aperture has no ray-transfer
		matrix, and a hard circular truncation is non-linear — it would break the
		Gaussian-moment propagation and cannot be expressed as ``M Sigma Mᵀ``. In
		envelope mode the aperture therefore leaves the mean and covariance untouched
		(intensity attenuation is captured only in ray mode, via
		:meth:`apply_intensity`). This is a documented approximation.

		Parameters
		----------
		mu : xp.ndarray
			Mean state vector.
		Sigma : xp.ndarray
			Covariance matrix.

		Returns
		-------
		tuple of xp.ndarray
			The inputs ``(mu, Sigma)`` unchanged.
		"""
		return mu, Sigma

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""The aperture's screen: a real transmission carried as a complex one.

		Overrides :meth:`Element.phase_shift`. An aperture used to apply itself
		through its own :meth:`propagate_wave` override, because a screen was
		unit-modulus by construction and no real ``χ`` makes ``|exp(iχ)|``
		anything but 1. A screen may now be **complex**, so an aperture is
		simply a screen whose modulus is its transmission and whose phase is
		zero — the same mechanism a phase plate uses, differing only in whether
		``arg(T)`` is nonzero.

		On the scaled path the transmission is built at **physical**
		coordinates ``(s_x·ξ, s_y·η)``, so a circular aperture is correctly an
		**ellipse** in scaled coordinates whenever the frame is anisotropic
		(``s_x ≠ s_y``, i.e. any quadrupole upstream), and identical to masking
		at ``radius/|s|`` when the axes agree. Sign-safe past a crossover, since
		the pitches are squared.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`Element.phase_shift`).
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			Select the representation, by default False.
		s : float or Sequence[float], optional
			Current transverse scale factor, used only when ``scaled=True``,
			by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: the phase program, a single complex screen (an
			aperture has zero length, so there are no free segments).
			``scaled=True``: ``(0.0, screen)`` — an aperture absorbs no
			curvature.

		Raises
		------
		ValueError
			If a screen supplied via :attr:`screen` does not match the grid.

		Related
		-------
		waveoptics.aperture_transmission : Builds the transmission.
		waveoptics.apply_phase : The real/complex screen convention.
		apply_intensity : The **ray**-path counterpart, deliberately separate.

		Notes
		-----
		The ray path is untouched by this: per-ray attenuation stays in
		:meth:`apply_intensity`, which is a different quantity computed from
		demagnification factors rather than from a grid. Only the wave
		behaviour moved.

		Examples
		--------
		>>> Aperture(radius=2e-5).phase_shift(((64, 64), 1e-6, 1e-6), 2.5e-12)  # doctest: +SKIP
		"""
		from .waveoptics import aperture_transmission, axis_components
		from .seashells import grid_of
		ny, nx, dy, dx = grid_of(dimensions)
		px, py = (dx, dy)
		if scaled:
			s_x, s_y = axis_components(s)
			px, py = s_x * dx, s_y * dy
		T = aperture_transmission((ny, nx), px, py, self.radius).astype(complex)
		T = self._combine_screen(T, (ny, nx), px, py)
		if scaled:
			return 0.0, _screen_item(T, dx, dy, self.name or "aperture")
		return self._phase_program(dimensions, wavelength, T,
								   self.name or "aperture")

class Drift(Element):
	"""Drift element class for free-space propagation.

		Parameters
		----------
		name : str, optional
			Name of Drift segment, useful for creating named planes which do not affect beam propagation (e.g., a "sample plane")
		length : float, optional
			length for free-space propagation
		calibration : float, optional
			linear scaling applied to length. e.g. nominally we believe the next lens is at 10 mm, but maybe fitting says 9.9
		position : float, optional
			The position of the element along the z-axis, by default None

		References
		----------
		https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Free_space_example
		"""

	def __init__(self, name:str='', length:float=0., calibration:float=None, position:float=None) -> SEASerializable:

		super().__init__(name=name,kind='Drift')
		self._position = position
		self.length = length
		self.calibration = calibration

	def transfer_matrix(self) -> xp.ndarray:

		m = xp.eye(4) # drift tube updates x from xθ and y from yθ

		s = self.length

		if self.calibration is not None:
			s *= self.calibration

		if s != 0:
			m[0,1] = s
			m[2,3] = s
		elif self.length == 0:
			pass

		return fix_mat_dims(m,["x","xt","y","yt"])

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""Free-space phase: the reciprocal-space Fresnel kernel over ``length``.

		Extends :meth:`Element.phase_shift`. A drift imprints no real-space
		screen — its entire action is the paraxial propagator phase
		:math:`-\pi\lambda\,\Delta z\,(f_\xi^2 + f_\eta^2)` applied in the FFT
		domain (handoff Eq 33).

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`Element.phase_shift`).
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			See :meth:`Element.phase_shift`, by default False.
		s : float, optional
			Unused for a drift, by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: ``[kernel(length)]`` (empty for zero length).
			``scaled=True``: ``(0.0, None)`` — a drift absorbs nothing into R and
			applies nothing to U; its free-segment updates (Δτ, s, R) are handled
			by the scaled driver from ``self.length``.
		"""
		if scaled:
			return 0.0, None
		return self._phase_program(dimensions, wavelength, None, "drift")

class Quadrapole(Element):
	def __init__(self, name:str='', 
				 position:float=None, length:float=0.,
				 strength:float=0, calibration:float=None) -> SEASerializable:

		"""Quadripole.

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		position : float, optional
			The position of the element along the z-axis, by default 0
		length : int, optional
			Length of the element, by default 0
		strength : float, optional
			Defined as the field strength (related to inverse focal length,
			see equations in brown1983), by default 0
		calibration : float, optional
			Currnet calibration of the lens in units of ???/A, by default None
		label : bool, optional
			If the element should be labeled when plotted, by default False
		print_fancy : bool, optional
			If a fancy table should be used when printed, by default True
		"""
		
		if length == 0: kind = 'Thin quad'
		else:		   kind = 'Quad'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration

	@property
	def _effective_strength(self) -> float:
		"""Return the calibration-scaled quadrupole strength ``K``.

		Applies the same calibration mapping used by :meth:`transfer_matrix`
		(linear scale for numeric calibration; ``K**p · c`` for a ``(c, p)``
		tuple), so the ray and wave representations always see the same
		effective strength.

		Returns
		-------
		float
			Effective strength ``K`` after calibration.

		Related
		-------
		transfer_matrix, phase_shift
		"""
		K=self.strength
		if self.calibration is not None:
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				c,p = self.calibration
				K = K**p * c
		return K

	def _axis_focuses(self, axis:Literal['x','y']='x') -> bool:
		r"""Whether this quadrupole focuses the given transverse axis.

		The single place the sign convention lives: **``K > 0`` focuses x and
		defocuses y**, and reversing the sign of ``K`` swaps them. Every other
		quadrupole method asks this rather than re-deriving it, so the thin and
		thick branches cannot disagree about which axis converges — they did
		before, because the thin branch swapped its blocks for ``K > 0`` and the
		thick branch never did.

		Parameters
		----------
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``.

		Returns
		-------
		bool
			True if the axis converges, False if it diverges. Meaningless at
			zero strength (returns True; callers short-circuit on ``K == 0``).

		Raises
		------
		ValueError
			If ``axis`` is not ``'x'`` or ``'y'``.

		Related
		-------
		_body_block : Chooses the trig or hyperbolic law from this.
		focal_powers : Signs its powers from this.
		"""
		if axis not in ('x', 'y'):
			raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
		return (self._effective_strength > 0) == (axis == 'x')

	def _body_block(self, dz:float, axis:Literal['x','y']='x') -> xp.ndarray:
		r"""The exact 2x2 body block over ``dz``, for one transverse axis.

		The single source of truth for thick-quadrupole ray optics:
		:meth:`transfer_matrix`, :meth:`transfer_block` and
		:meth:`focal_powers` all read it, so they cannot drift apart. A
		quadrupole body is a medium of constant strength, so with
		:math:`k = |K|` the motion is harmonic on the focusing axis and
		**hyperbolic** on the defocusing one (:math:`u'' \mp k^2 u = 0`):

		.. math::

			\mathrm{focusing:}\quad
			\begin{pmatrix} \cos k\,dz & \sin(k\,dz)/k \\
			-k\sin k\,dz & \cos k\,dz \end{pmatrix}
			\qquad
			\mathrm{defocusing:}\quad
			\begin{pmatrix} \cosh k\,dz & \sinh(k\,dz)/k \\
			+k\sinh k\,dz & \cosh k\,dz \end{pmatrix}

		Both have **unit determinant** (:math:`\cos^2+\sin^2` and
		:math:`\cosh^2-\sinh^2`), so phase-space area is conserved as Liouville
		requires, and both compose (``M(dz/2)² = M(dz)``) as a homogeneous
		medium must.

		Parameters
		----------
		dz : float
			Distance into the body (metres). May be partial, so a plane inside
			the body is found exactly rather than interpolated across it.
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``.

		Returns
		-------
		xp.ndarray
			The ``2x2`` block.

		Raises
		------
		ValueError
			If ``axis`` is not ``'x'`` or ``'y'``.

		Related
		-------
		_axis_focuses : Picks the trig or hyperbolic law.
		transfer_matrix, transfer_block, focal_powers : The three consumers.

		Notes
		-----
		``k = |K|`` deliberately: the off-diagonal ``B`` term is drift-like and
		must stay positive for a forward step. The previous code wrote ``S/K``
		with a *signed* ``K``, which inverted it for ``K < 0``.
		"""
		k = abs(self._effective_strength)
		if k == 0 or dz == 0:
			return xp.asarray([[1.0, float(dz)], [0.0, 1.0]])
		kz = k * abs(float(dz))
		if self._axis_focuses(axis):
			c, sn = xp.cos(kz), xp.sin(kz)
			return xp.asarray([[c, sn / k], [-k * sn, c]])
		c, sn = xp.cosh(kz), xp.sinh(kz)
		return xp.asarray([[c, sn / k], [k * sn, c]])

	def transfer_block(self, dz:float=None, axis:Literal['x','y']='x') -> xp.ndarray:
		r"""Rotating-frame 2x2 block of a quadrupole, exact at any partial length.

		Overrides :meth:`Element.transfer_block`. A thick quadrupole body is a
		medium of constant strength, so its block is exact at any depth: harmonic
		on the focusing axis, hyperbolic on the defocusing one (see
		:meth:`_body_block`, which this and :meth:`transfer_matrix` share).
		A thin quadrupole (``length == 0``) defers to the base class, where the
		impulsive kick :meth:`focal_powers` is exact.

		Parameters
		----------
		dz : float, optional
			Distance into the quadrupole (metres); ``None`` uses ``length``.
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``. The two axes differ in the sign
			of the focusing term — that is what makes it a quadrupole.

		Returns
		-------
		xp.ndarray
			The ``2x2`` block.

		Related
		-------
		transfer_matrix : The full 6x6 matrix this mirrors.
		focal_powers : The thin-element powers.

		Notes
		-----
		Delegates to :meth:`_body_block`, the same helper :meth:`transfer_matrix`
		uses, so plane finding and ray tracing cannot disagree.
		"""
		L = self.length or 0.0
		step = L if dz is None else float(dz)
		if L <= 0 or self._effective_strength == 0:
			return super().transfer_block(dz=step, axis=axis)
		return self._body_block(step, axis)

	def _scaled_segment(self):
		r"""A thick quadrupole is a segment too — with opposite curvature per axis.

		Overrides :meth:`Element._scaled_segment`. The quadrupole body is a
		medium of constant strength that **focuses one transverse axis and
		defocuses the other**, so its signed curvature is ``(+K², −K²)`` in the
		order set by :meth:`_axis_focuses` (``K > 0`` focuses x). The scaled
		frame follows each axis exactly — harmonic on one, hyperbolic on the
		other — so a thick quadrupole costs the sampled field nothing, exactly
		as a thick round lens does.

		With ``length == 0`` there is no body to traverse and the thin route is
		used instead: :meth:`phase_shift` absorbs the per-axis powers into
		``(R_x, R_y)``.

		Returns
		-------
		tuple or None
			``('quadratic', (kappa_x, kappa_y), 0.0)`` when this quadrupole has
			a finite length and nonzero strength, else ``None``. The Larmor
			angle is **zero**: a quadrupole has no axial field, so unlike a round
			lens it does not rotate the beam.

		Related
		-------
		Lens._scaled_segment : The isotropic case.
		_axis_focuses : Sets which axis gets the positive curvature.
		phase_shift : Supplies the thin-quadrupole curvature kick.
		"""
		K = self._effective_strength
		if self.length > 0 and K != 0:
			kappa = float(K**2)
			pair = (kappa, -kappa) if self._axis_focuses('x') else (-kappa, kappa)
			return ('quadratic', pair, 0.0)
		return None

	@property
	def focal_powers(self) -> tuple:
		r"""Return the astigmatic focusing powers ``(1/f_x, 1/f_y)``.

		The quadrupole is the spatially asymmetric round lens: one transverse
		axis focuses while the other diverges, so the two powers have opposite
		sign. Signs come from :meth:`_axis_focuses` (``K > 0`` focuses x), and
		the magnitudes from the ``-1/f = C'`` entry of :meth:`_body_block`, so
		these powers always agree with the ray matrix.

		Thin (``length == 0``): the impulsive kick ``±K²``. Thick: ``+k·sin(kL)``
		on the focusing axis and ``−k·sinh(kL)`` on the defocusing one, with
		``k = |K|``. The hyperbolic magnitude grows without bound in ``kL``,
		which is correct — a long defocusing quadrupole throws rays out
		exponentially, and the old ``sin`` on both axes hid that.

		Returns
		-------
		tuple of float
			``(power_x, power_y)`` in 1/metres; ``(0, 0)`` at zero strength.

		Related
		-------
		_body_block : The matrix these mirror.
		phase_shift : Uses these powers for the saddle phase screen.
		"""
		k = abs(self._effective_strength)
		if k == 0:
			return 0.0, 0.0
		if self.length == 0:
			powers = (k**2, -(k**2))
		else:
			kL = k * abs(self.length)
			powers = (float(k * xp.sin(kL)), float(-k * xp.sinh(kL)))
		focus, defocus = powers
		return (focus, defocus) if self._axis_focuses('x') else (defocus, focus)

	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propagation through the quadrupole.

		A quadrupole focuses one transverse axis and defocuses the other. The
		homogeneous equation of motion :math:`u'' \pm k^2 u = 0` (``k = |K|``)
		is therefore **harmonic** on the focusing axis and **hyperbolic** on the
		defocusing one, and both solutions have unit determinant, as Liouville
		requires. The per-axis blocks come from :meth:`_body_block`; a thin
		quadrupole (``length == 0``) is the impulsive limit, a pure kick of
		``∓K²`` from :meth:`focal_powers`.

		**Convention: ``K > 0`` focuses x and defocuses y**, and reversing the
		sign of ``K`` swaps the two axes. This holds identically in the thin and
		thick branches — it did not before, because the thin branch swapped its
		blocks for ``K > 0`` while the thick branch never did, so giving a
		quadrupole a length changed which axis converged.

		Returns
		-------
		xp.ndarray
			The ``len(convention) × len(convention)`` transfer matrix.

		Related
		-------
		_body_block : The per-axis body law, shared with :meth:`transfer_block`.
		_axis_focuses : The sign convention.
		focal_powers : The thin-equivalent powers, signed the same way.

		Notes
		-----
		Brown 1983 p. 46. The defocusing axis previously reused ``cos``/``sin``,
		giving ``det = cos(2|KL|)`` — 0.75 over a 30 mm body, so a quarter of
		the phase-space area vanished and the block's halves did not compose.

		A **skew** (rotated) quadrupole is not supported: that couples x and y,
		so it cannot be written as two independent 2×2 blocks.

		References
		----------
		.. [1] K. L. Brown, "A First- and Second-Order Matrix Theory for the
		   Design of Beam Transport Systems and Charged Particle Spectrometers",
		   SLAC-75 (1983), p. 46.
		"""

		#m = xp.eye(6)#[...,None]*xp.ones_like(s) # TWP 2025/08/27 - adding ones_like expression so m is 6x6x1, otherwise eigsum in propagate will fail
		#m = xp.eye(4) # quadrupole updates xθ from x and yθ from y

		K = self._effective_strength

		if K==0:
			return fix_mat_dims(xp.eye(4),["x","xt","y","yt"])

		if self.length==0:
			# impulsive limit: a pure kick, -1/f on the [1,0] entry
			P_x, P_y = self.focal_powers
			X=xp.asarray([[ 1 , 0 ],
					     [ -P_x , 1 ]])
			Y=xp.asarray([[ 1 , 0 ],
						 [ -P_y , 1 ]])
		else:
			# harmonic on the focusing axis, hyperbolic on the defocusing one
			X = self._body_block(self.length, 'x')
			Y = self._body_block(self.length, 'y')

		m=xp.matmul( fix_mat_dims(X,["x","xt"]) , fix_mat_dims(Y,["y","yt"]) )
		#print("QUAD",m,self.strength,K,self.calibration,self.length)
		return m

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""Quadrupole phase: the astigmatic saddle :math:`\chi = -k(P_x x^2 + P_y y^2)/2`.

		Extends :meth:`Element.phase_shift`. The quadrupole is the spatially
		asymmetric version of the round lens — ``P_x = −P_y`` (from
		:meth:`focal_powers`), so one transverse axis focuses while the other
		diverges: :math:`\chi \propto (x^2 - y^2)`.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`Element.phase_shift`).
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			See :meth:`Element.phase_shift`, by default False.
		s : float, optional
			Transverse scale for ``scaled=True``: the screen is evaluated at
			physical coordinates ``x = s·ξ``, by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: ``[kernel(L/2), screen(χ), kernel(L/2)]``.
			``scaled=True``: ``((P_x, P_y), None)`` — the per-axis powers are
			absorbed into the anisotropic curvature state ``(R_x, R_y)``
			exactly like a round lens absorbs one power into one curvature
			(``1/R_a⁺ = 1/R_a⁻ − P_a`` per axis), so the saddle never touches
			the sampled field U and arbitrarily strong quadrupoles carry no
			sampling limit. ``(0.0, None)`` at zero strength.
		"""
		from .waveoptics import quadratic_phase
		from .seashells import grid_of
		P_x, P_y = self.focal_powers
		if scaled:
			if P_x == 0 and P_y == 0:
				return 0.0, None
			return (float(P_x), float(P_y)), None
		ny, nx, dy, dx = grid_of(dimensions)
		chi = quadratic_phase((ny, nx), dx, dy, wavelength, P_x, P_y) if (P_x or P_y) else None
		return self._phase_program(dimensions, wavelength, chi, self.name or "quadrupole")


class Dipole(Element):
	def __init__(self, name:str='',
				 position:float=None, length:float=0.,
				 strength:float=0, calibration:float=None,
				 axis: Literal['x','y'] | float | Sequence='x'
				 ) -> SEASerializable:
		"""Dipole.

		Parameters
		----------
		name : str, optional
			Name given to the dipole, by default ''.
		position : float, optional
			The position of the element along the z-axis, by default None.
		length : float, optional
			Length of the element, by default 0.
		strength : float, optional
			Angular kick applied by the dipole, by default 0.
		calibration : float or tuple, optional
			Calibration applied to ``strength``. Numeric values apply a
			linear scale; tuple values are interpreted as ``(scale, power)``,
			matching ``Quadrapole`` behavior.
		axis : {'x', 'y', float, Sequence}, optional
			Transverse axis receiving the kick. Can be 'x', 'y', a float angle in radians, or a sequence [x, y], by default 'x'.

		Raises
		------
		UserWarning
			If ``axis`` is not ``'x'``, ``'y'``, a float, or a sequence.
		"""
		if length == 0: kind = 'Thin dipole'
		else:		   kind = 'Dipole'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration
		self.axis = axis	# keep the user's axis spec so .sea round-trips preserve the rotation

		# strings must be tested before calling .lower() (floats have no .lower())
		if isinstance(axis, str) and axis.lower() == 'x':
			self.phi = 0
		elif isinstance(axis, str) and axis.lower() == 'y':
			self.phi = xp.pi/2
		elif isinstance(axis, (int, float)):
			if axis > 0 and axis <= 2*xp.pi:
				self.phi = axis
			else:
				self.phi = xp.remainder(axis + xp.pi, 2 * xp.pi) - xp.pi
		elif isinstance(axis, Sequence):
			self.phi = xp.arctan2(axis[1],axis[0])
		else:
			raise UserWarning(f'A float. sequence, "x", or "y" are valid `axis` values but a value of {axis} was provided which is a {type(axis)}.')

	def effective_tilts(self) -> tuple:
		"""Return the calibration-scaled deflection angles ``(tilt_x, tilt_y)``.

		Computes the same angular kick that :meth:`transfer_matrix` stores on
		``self.tilt_x``/``self.tilt_y`` (calibration mapping, axis projection
		via ``phi``, and the length scaling for finite-length dipoles) without
		mutating the element, so the wave path can read it side-effect-free.

		Returns
		-------
		tuple of float
			``(tilt_x, tilt_y)`` in radians.

		Related
		-------
		transfer_matrix, phase_shift
		"""
		K = self.strength
		if self.calibration is not None:
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				c,p = self.calibration
				K = K**p * c
		Kx = K * xp.cos(self.phi)
		Ky = K * xp.sin(self.phi)
		if self.length == 0:
			return float(Kx), float(Ky)
		return float(Kx * self.length), float(Ky * self.length)

	def transfer_matrix(self) -> ArrayLike:
		r"""Transfer matrix for ray propogation.

		Notes
		-----
		The ray vector is purely geometric and has no homogeneous coordinate, so
		the dipole's constant steering term cannot ride inside the transfer matrix.
		Instead this method stores the kick on ``self.tilt_x``/``self.tilt_y``, which
		:meth:`Element.propagate_ray` adds to the ray angles as an affine term.

		Returns
		-------
		xp.ndarray
			Identity transfer matrix; the steering kick is applied additively via
			``self.tilt_x``/``self.tilt_y`` in :meth:`Element.propagate_ray`.
		"""
		K = self.strength

		# Apply a calibration
		if self.calibration is not None:
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				c,p = self.calibration
				K = K**p * c
		
		# Project the strength
		Kx = K * xp.cos(self.phi)
		Ky = K * xp.sin(self.phi)

		if self.length == 0:
			self.tilt_x = Kx
			self.tilt_y = Ky
		else:
			self.tilt_x = Kx * self.length
			self.tilt_y = Ky * self.length

		return fix_mat_dims(xp.eye(4),["x","xt","y","yt"])

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""Dipole phase: the linear tilt :math:`\chi = k(\theta_x x + \theta_y y)`.

		Extends :meth:`Element.phase_shift`. The deflection angles come from
		:meth:`effective_tilts` (calibration + the ``phi`` axis projection, so
		both orientations of a 45° dipole pair are covered).

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`Element.phase_shift`).
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			See :meth:`Element.phase_shift`, by default False.
		s : float, optional
			Transverse scale for ``scaled=True``: the screen is evaluated at
			physical coordinates ``x = s·ξ``, by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: ``[kernel(L/2), screen(χ), kernel(L/2)]``.
			``scaled=True``: ``(0.0, screen)`` — a linear phase is not quadratic,
			so nothing is absorbed into R and the full tilt is applied to U
			(handoff Eqs 47–48). ``(0.0, None)`` at zero strength.
		"""
		from .waveoptics import linear_phase, axis_components
		from .seashells import grid_of
		tilt_x, tilt_y = self.effective_tilts()
		ny, nx, dy, dx = grid_of(dimensions)
		if scaled:
			if tilt_x == 0 and tilt_y == 0:
				return 0.0, None
			s_x, s_y = axis_components(s)		# per-axis physical pitch on anisotropic frames
			chi = linear_phase((ny, nx), s_x * dx, s_y * dy, wavelength, tilt_x, tilt_y)
			return 0.0, _screen_item(chi, dx, dy, self.name or "dipole")
		chi = linear_phase((ny, nx), dx, dy, wavelength, tilt_x, tilt_y) if (tilt_x or tilt_y) else None
		return self._phase_program(dimensions, wavelength, chi, self.name or "dipole")

		#m = xp.zeros((7,8))
		#xp.fill_diagonal(m, 1, wrap=False)
		#
		#if self.length == 0:
		#	m[2, 7] = Kx
		#	m[3, 7] = Ky
		#else:
		#	L = self.length
		#
		#	# Drift terms
		#	m[0, 2] = L   # x <- ux
		#	m[1, 3] = L   # y <- uy
		#	m[4, 4] = 1.0 # z stays identity (already set)
		#
		#	# z advance
		#	m[4, 7] = L
		#
		#	# Angular kicks
		#	m[2, 7] = Kx * L
		#	m[3, 7] = Ky * L
		#
		#	# Position offsets (affine)
		#	m[0, 7] = 0.5 * Kx * L**2
		#	m[1, 7] = 0.5 * Ky * L**2
		#
		#return m
	
	#def propagate_ray(self, r0:xp.ndarray,
	#				  z:float=None, z0:float=0) -> xp.ndarray:
	#	"""propagate an array through an element.
	#
	#	Parameters
	#	----------
	#	r0 : xp.ndarray
	#		List of rays with possible initial conditions (x, θx, y, θy, E).
	#	z : None | int | float | xp.ndarray, optional
	#		Positions in the element to propagate to by default None
	#	z0 : None | float, optional
	#		Initial position of the element, by default 0
	#
	#	Returns
	#	-------
	#	xp.ndarray
	#		List of propagated rays with initial condition (x, θx, y, θy, z, E)
	#	"""
	#	m = self.transfer_matrix()
	#	#print(f'm: {m.shape}')#FLAG
	#	ones = xp.ones((r0.shape[0], 1), dtype=r0.dtype)
	#	r0_aug = xp.concatenate([r0, ones], axis=1)
	#	#print(f'r0: { r0.shape} to {r0_aug.shape}')#FLAG
	#
	#	rf = xp.einsum('mn,in->im', m, r0_aug)
	#	return rf


class Lens(Element):
	"""Lens element class for round lenses / symmetric focusing. https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Thin_lens_example or Brown1983 page 105

		Parameters
		----------
		name : str, optional
			Name of the lens, by default ''
		length : float, optional
			thickness of the lens, which affects the rotation of the beam
		strength : float, optional
			lens strength, proportional to lens current I, or magnetic field strength B. strength^2 is propogational to 1/f
		calibration : list or float, optional
			if a float is provided, a linear scaling will be applied to strength
			if a list is provided, terms are used in a series: strength =A+B*nominal+C*nominal^(1/2)+D*nominal^(1/3)+...
		Cs : float, optional
			Third-order spherical aberration coefficient (metres), by default 0
			(an ideal lens). A convenience alias for the Krivanek ``C30`` term,
			folded into :attr:`aberrations` at construction.
		aberrations : Aberrations or dict, optional
			Axial wave aberrations in Krivanek ``C_{n,m}`` notation through
			fifth order, by default ``None`` (an ideal lens). A dict is
			converted; an :class:`aberrations.Aberrations` is attached as-is,
			so one measured from an instrument's metadata can be moved onto a
			simulated lens unchanged. Whatever is here acts on **both** the ray
			and the wave path, at every order — see
			:meth:`aberration_kick` and :meth:`phase_shift`.
		position : float, optional
			The position of the element along the z-axis, by default None
		rotation : bool, optional
			if set to False, lens rotation for finite-thickness lenses is overridden and turned off.
		"""
	def __init__(self, name:str='', length:float=0.,
				 strength:float=0, calibration:float=None, Cs:float=0.0,
				 aberrations:dict=None,
				 position:float=None) -> SEASerializable:
		
		if length == 0: kind = 'Thin lens'
		else:		   kind = 'QLens'

		super().__init__(name=name,kind=kind)
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration
		self.rotation = 0
		# One nested Aberrations object, not a scatter of flat scalars: it is a
		# SEASerializable itself, so .sea and JSON carry it as a child node, and
		# every order is applied by one generic expression rather than per term.
		self.aberrations = _as_aberrations(aberrations)
		if Cs:							# convenience alias for the C30 term
			if self.aberrations is None:
				self.aberrations = Aberrations()
			self.aberrations.setdefault('C30', Cs)


	@property
	def _effective_strength(self) -> float:
		"""Return the calibration-scaled lens strength ``K``.

		Applies the same calibration mapping used by :meth:`transfer_matrix`
		(linear scale for numeric calibration; the ``A + B·K^(1/1) + C·K^(1/2) +
		...`` series for sequence calibration), so the ray and wave
		representations always see the same effective strength.

		Returns
		-------
		float
			Effective strength ``K`` after calibration.

		Related
		-------
		transfer_matrix, phase_shift
		"""
		K=self.strength
		if self.calibration is not None:
			# linear scaling from mA (lens current) to lens strength?
			if isinstance(self.calibration,(int,float)):
				c = self.calibration
				K *= c
			else:
				# A + B*x^(1/1) + C*x^(1/2) + D*x^(1/3) + ....
				Kvals = [self.calibration[0]] + [ v*K**(1/(i+1)) for i,v in enumerate(self.calibration[1:]) ]
				K = sum( Kvals )
		return K

	@property
	def focal_power(self) -> float:
		r"""Return the focusing power ``1/f`` of this lens.

		Thin lens (``length == 0``): ``1/f = sign(K)·K²`` (matching the
		``sign·K²`` matrix cell). Thick lens: the Brown (1983) focusing relation
		``1/f = K·sin(K·L)``.

		Returns
		-------
		float
			Focal power ``1/f`` (1/metres); 0 for a zero-strength lens.

		Related
		-------
		phase_shift : Uses this power for the quadratic phase screen.
		"""
		K = self._effective_strength
		if K == 0:
			return 0.0
		if self.length == 0:
			return float(xp.sign(K) * K**2)
		return float(K * xp.sin(K * self.length))

	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation.
		"""

		# HANDLE CALIBRATION SCALING (shared with the wave path via _effective_strength)
		K = self._effective_strength

		# FINITE LENGTH LENS, ZERO STRENGTH = DRIFT (try inserting a zero-strength lens and seeing if the result changes)
		if K==0:
			m = xp.eye(4)
			m[0,1]=self.length
			m[2,3]=self.length
			self.rotation = 0
			return fix_mat_dims(m,["x","xt","y","yt"])

		# THIN LENS, NO ROTATION (thick lens math will have sine term going to zero)
		if self.length==0:
			sign = -1*xp.sign(K) # sign allows negative calibration to give you diverging beams???
			X=xp.asarray([[    1   , 0 ],
					     [ sign*(K**2) , 1 ]])
			Y=xp.asarray([[    1   , 0 ],
						 [ sign*(K**2) , 1 ]])
			self.rotation = 0
			return xp.matmul( fix_mat_dims(X,["x","xt"]) , fix_mat_dims(Y,["y","yt"]) )

		# THICK LENS, FINITE K (zero K will have iK going to infinite)
		kL=K*self.length ; iK=1/K
		C=xp.cos(kL) ; S=xp.sin(kL)
		#XY=xp.asarray([	[ C**2  , iK*S*C  ,   S*C  , iK*S**2 ],	# Brown1983 page 105
		#				[-K*S*C ,  C**2   ,-K*S**2 ,   S*C   ],	# similar to standard
		#				[ -S*C  ,-iK*S**2 ,   C**2 , iK*S*C  ],	# [  1   0 ] but with
		#				[ K*S**2,  -S*C   , -K*S*C ,  C**2   ]] )# [ -1/f 1 ] rotation
		XY= xp.asarray([[  C , iK*S ,  0  ,  0   ],				# Brown1983 page 106
						[-K*S,  C   ,  0  ,  0   ],				# alternate definition
						[  0 ,  0   ,  C  , iK*S ],				# M = R(-KL) @ M_alt
						[  0 ,  0   ,-K*S ,  C   ]])
		R = xp.asarray([[  C ,  0 ,  S ,  0 ],					# | c -s | is normal,
						[  0 ,  C ,  0 ,  S ],					# | s  c | applied to x,y
						[ -S ,  0 ,  C ,  0 ],					# and xt,yt independently
						[  0 , -S ,  0 ,  C ]])					# here, flip signs to -KL
		#print(xp.matmul(R,XY2)-XY)
		#XY = xp.matmul(R,XY) # *xp.asarray([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])

		# TWP 2026-05-12: new procedure: never rotate, but Element.propagate_ray will track rotation angle JK SEE BELOW
		#if not self.rotation:
		#	XY = XY2
		#else:
		#XY = xp.matmul(R,XY2)
		#	zeroer=xp.asarray([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
		#	XY*=zeroer
		#print("lens",self.name,"adds rotation",kL)
		# TWP 2026-07-23: upon discussion with Eric, we decided to always rotate. R is still tracked to allow you to return to the rotating reference frame for the purposes of quick-and-easy plane detection etc, although that stuff should be improved too (e.g., once we add aberrations, we will need to look for a beam waist. interpolate between drift endpoints, calculate Diameter(z) from all rays, d^2 diameter / dz^2 tells you where the beam is at a minimum diameter. check bundles of rays for diffraction planes?)
		XY = xp.matmul(R,XY)
		self.rotation = -kL
		M = fix_mat_dims(XY,["x","xt","y","yt"])
		return M

	# unlike below(?), here we'll *measure* focal length at the current K=I*C and L, then adjust C and L to preserve focal length and set beam rotation (K*L) to match R in radians at this current I.
	def get_C_L_from_rotation_at_I(self,I,R):
		from scipy.optimize import minimize
		print(self.name,I,R)
		def FR(C,L):
			new = Lens(strength = I, calibration = C, length = L)
			columns = [ columnByName(k) for k in ["x","xt","y","yt"] ]
			M = new.transfer_matrix()[columns,:][:,columns]
			r0 = [1,0,1,0] # parallel starting ray
			r1 = xp.matmul(M,r0)
			x = xp.sqrt(r1[0]**2+r1[2]**2) ; xt = xp.sqrt(r1[1]**2+r1[3]**2)
			f = x/xt # f = x/theta
			rot = new.rotation
			return f,rot
		f0,_ = FR(self.calibration,self.length)	# initial focal length
		print("currently focuses to",f0)
		def dz(vals):
			f,rot = FR(*vals)
			return ((f-f0)/f0)**2 + ((R-rot)/R)**2
		res = minimize(dz,x0=(self.calibration,self.length))
		print(res)
		f,rot = FR(*res['x'])
		print( "fitted focuses to",f,"and with rotation of",rot )
		return res['x']
	def transfer_block(self, dz:float=None, axis:Literal['x','y']='x') -> xp.ndarray:
		r"""Rotating-frame 2x2 block of a round lens, exact at any partial length.

		Overrides :meth:`Element.transfer_block`. A thick lens body is a medium
		of constant strength ``K``, so its block is sinusoidal and exact for any
		distance into it:

		.. math::

			\begin{pmatrix} \cos K\,dz & \sin(K\,dz)/K \\
			-K\sin K\,dz & \cos K\,dz \end{pmatrix}

		(Brown 1983; the same block :meth:`transfer_matrix` builds before
		applying the Larmor rotation). A thin lens (``length == 0``) falls back
		to the base thin-kick form.

		Parameters
		----------
		dz : float, optional
			Distance into the lens (metres); ``None`` uses the full ``length``.
		axis : {'x', 'y'}, optional
			Transverse axis, by default ``'x'``; a round lens is identical on
			both.

		Returns
		-------
		xp.ndarray
			The ``2x2`` block.

		Related
		-------
		_scaled_segment : Reports the same body to the scaled wave path.
		waveoptics.propagate_quadratic_segment_scaled : Advances the wave frame by it.
		"""
		L = self.length or 0.0
		step = L if dz is None else float(dz)
		K = self._effective_strength
		if L <= 0 or K == 0:
			return super().transfer_block(dz=step, axis=axis)
		c, s = xp.cos(K * step), xp.sin(K * step)
		return xp.asarray([[c, s / K], [-K * s, c]])

	def _scaled_segment(self):
		r"""A thick round lens is a quadratic-index segment; a thin one is not.

		Overrides :meth:`Element._scaled_segment`. With ``length > 0`` the lens
		body is a medium of constant strength ``K``, which the scaled frame
		follows exactly (sinusoidal ``s(z)``, closed-form Δτ, no phase screen).
		With ``length == 0`` there is no body to traverse, so the thin-lens
		route is used instead: the full power ``sign(K)·K²`` is absorbed into the
		curvature by :meth:`phase_shift` (``scaled=True``).

		Returns
		-------
		tuple or None
			``('quadratic', kappa, larmor)`` when this lens has a finite length
			and nonzero strength, else ``None``. A round lens is isotropic and
			focusing, so ``kappa = K**2`` on both axes; ``larmor = -K*L`` is the
			body's rotation angle, declared here rather than re-derived by the
			propagator (a quadrupole has none).

		Related
		-------
		Quadrapole._scaled_segment : The astigmatic case, ``(+kappa, -kappa)``.
		phase_shift : Supplies the thin-lens curvature kick.
		waveoptics.propagate_quadratic_segment_scaled : Propagates the segment.
		"""
		K = self._effective_strength
		if self.length > 0 and K != 0:
			return ('quadratic', float(K**2), float(-K * self.length))
		return None

	def phase_shift(self, dimensions, wavelength:float, scaled:bool=False, s:float=1.0):
		r"""Round-lens phase: :math:`\chi = -k(x^2+y^2)/(2f)` (handoff Eq 12).

		Extends :meth:`Element.phase_shift`. The focal power ``1/f`` comes from
		:meth:`focal_power` (thin: ``sign(K)·K²``; thick: ``K·sin(K·L)``, Brown
		1983 — the pure focusing relation, so a thick lens's Larmor rotation
		never contaminates the wave-path power).

		Any :attr:`aberrations` are added as the wave aberration function
		:math:`\chi`, whatever terms they happen to contain — the same
		:math:`\chi` the ray path differentiates, so the two representations
		cannot drift apart.

		Parameters
		----------
		dimensions : Dimensions or tuple
			Transverse grid (see :meth:`Element.phase_shift`).
		wavelength : float
			Wavelength (metres).
		scaled : bool, optional
			See :meth:`Element.phase_shift`, by default False.
		s : float, optional
			Frame scale, used on the scaled path to place the screen at
			physical coordinates ``x = s·xi``, by default 1.

		Returns
		-------
		list or tuple
			``scaled=False``: ``[kernel(L/2), screen(χ), kernel(L/2)]``.
			``scaled=True``: ``(power, screen)`` — the parabola is absorbed into
			the curvature state (Eq 45), together with the quadratic part of the
			aberrations (:meth:`aberration_powers`), and the rest stays as a
			residual screen on ``U`` (``None`` for an ideal lens, giving
			``U⁺ = U⁻``, Eq 15). ``power`` is a scalar, or an ``(x, y)`` pair
			when an aligned ``C12`` makes the two axes differ.

		Raises
		------
		None

		Related
		-------
		aberrations.Aberrations.phase_at : Builds the aberration function.
		aberration_kick : The ray-side gradient of the same function.
		"""
		from .waveoptics import quadratic_phase, axis_components
		from .seashells import grid_of
		P = self.focal_power
		ab = self.aberrations
		ny, nx, dy, dx = grid_of(dimensions)
		if scaled:
			if not ab or P == 0:
				s_x, s_y = axis_components(s)
				bare = self._combine_screen(None, (ny, nx), s_x * dx, s_y * dy)
				return float(P), (None if bare is None else
								  _screen_item(bare, dx, dy, self.name or "lens"))
			# first-order terms are QUADRATIC, so they belong in the frame's
			# curvature, not in a screen the frame exists to avoid
			P_x, P_y, residual = self.aberration_powers()
			powers = float(P_x) if P_x == P_y else (float(P_x), float(P_y))
			if not residual:
				return powers, None
			# the parabola is absorbed into the curvature exactly as before; the
			# rest of the aberration function CANNOT be -- the frame is
			# quadratic by construction, and every remaining term is of higher
			# order or lower symmetry -- so it stays as a residual screen on U
			# at physical coords x = s*xi. That is the aberration function, in
			# the place it belongs.
			s_x, s_y = axis_components(s)
			chi = residual.phase((ny, nx), s_x * dx, s_y * dy, wavelength, P)
			chi = self._combine_screen(chi, (ny, nx), s_x * dx, s_y * dy)
			return powers, _screen_item(chi, dx, dy, self.name or "lens")
		chi = quadratic_phase((ny, nx), dx, dy, wavelength, P, P) if P != 0 else None
		if ab and P != 0:
			extra = ab.phase((ny, nx), dx, dy, wavelength, P)
			chi = extra if chi is None else chi + extra
		chi = self._combine_screen(chi, (ny, nx), dx, dy)
		return self._phase_program(dimensions, wavelength, chi, self.name or "lens")

	def aberration_powers(self) -> tuple:
		r"""Split the aberration function into frame powers and a residual.

		The scaled frame *is* a quadratic: :math:`(s, R)` can represent any
		phase of the form :math:`x^2/2R`. So the **first-order** Krivanek terms,
		which are quadratic in the pupil angle, do not belong in the residual
		screen at all — they belong in the curvature, exactly as the lens's own
		parabola does:

		- ``C10`` (defocus, :math:`m = 0`) is isotropic and quadratic, so it is
		  a pure change of focal power, :math:`\Delta P = C_{10} P^2`.
		- ``C12`` (twofold astigmatism, :math:`m = 2`) is quadratic but
		  astigmatic, giving :math:`\pm C_{12} P^2` on the two axes — the same
		  ``(P, -P)`` shape a quadrupole absorbs into :math:`(R_x, R_y)`.

		Everything of second order and above is genuinely non-quadratic and
		stays as a screen on ``U``. Absorbing the low-order terms is not merely
		tidy: a quadratic screen is precisely what the scaled frame exists to
		avoid, and leaving ``C10`` in the screen would both waste sampling and
		put the logged crossover in the wrong place.

		Returns
		-------
		tuple
			``(power_x, power_y, residual)``: the per-axis focal powers
			including the lens's own, and an :class:`aberrations.Aberrations`
			holding the terms that must still be applied to ``U``.
			``power_x == power_y`` for a round lens with no ``C12``.

		Raises
		------
		None

		Related
		-------
		aberrations.Aberrations : The storage this splits.
		phase_shift : The consumer.

		Notes
		-----
		``C12`` is absorbed only when it is **aligned** with the grid axes,
		which for a complex coefficient means a zero imaginary part. A rotated
		quadratic is a *skew* astigmatism, which a per-axis :math:`(R_x, R_y)`
		frame cannot represent — the frame would need off-diagonal terms — so a
		skew ``C12`` is left in the residual screen instead. Same limitation as
		a skew quadrupole.
		"""
		P = float(self.focal_power)
		residual = Aberrations()
		P_x = P_y = P
		if not self.aberrations:
			return P_x, P_y, residual
		for name, c in self.aberrations.items():
			if name == 'C10':						# isotropic quadratic: pure power
				P_x += c.real * P**2
				P_y += c.real * P**2
				continue
			if name == 'C12' and c.imag == 0.0:		# aligned: +-a on the two axes
				P_x += c.real * P**2
				P_y -= c.real * P**2
				continue
			residual[name] = c
		return P_x, P_y, residual

	def calibration_from_f_and_I(self,f,I,rotationPerAmp=None):
		print("for lens",self.name,"seeking a calibration factor C, which focuses strength",I,"to focal length",f,"and rotationPerAmp",rotationPerAmp)
		# noting xt=f(x) cell from matrix is -1/f or -K*sin(K*L)*cos(K*L):
		# APPROXIMATION: noting that at small angle, sin(K*L) ≈ K*L and cos(K*L) ≈ 1.
		# if K=C*I (strength = linear scaling * electrical current)
		# 1/f ≈ K²L = (C*I)² L  --> C = √(1/f/L)/I
		# and if length L is not fixed, but rotationPerAmp is given:
		# R = K*L = C*I*L --> R/I = C*L
		# BEWARE: inaccurate for large L. we really need to "solve for" C
		if self.length == 0:
			self.calibration = xp.sign(f)*xp.sqrt(1/abs(f))/I # 1/f = K² = (I*C)²,
			return
		self.calibration = xp.sqrt(1/f/self.length)/I
		# NOT AN approximation: 1/f = K*S*C = (C*I)*sin(C*I*L)*cos(C*I*L)
		# is there an analytical solution?
		# trig identity: sin(x)*cos(y) = 1/2*(sin(x+y)+sin(x-y)) so if x=y, sin(x)*cos(x) = 1/2*sin(2*x)
		#1/f = (C*I)*½*sin(2*C*I*L) idk how to solve this lol. if all you have is a hammer (scipy minimize) everything looks like a nail (a minimization problem)
		# OPE: 1/f = K*sin, not 1/f = K*sin*cos. (note how the alternate form of M from Brown1983 eliminated the cos. the cos is there for rotation)
		from scipy.optimize import minimize
		if rotationPerAmp is None:
			def dz(C):
				#return ( C*I*xp.sin(C*I*self.length)*xp.cos(C*I*self.length)-1/f )**2 # 1/f = K*S*C = (C*I)*sin(C*I*L)*cos(C*I*L)
				return (C*I*xp.sin(C*I*self.length)-1/f)**2	# 1/f = K*S = (C*I)*sin(C*I*L)
			x0 = self.calibration
			self.calibration = minimize(dz,x0=x0)['x'][0]
		else:
			def dz(CL):
				#print(CL)
				C,L=CL
				#return ( C*I*xp.sin(C*I*L)*xp.cos(C*I*L)-1/f )**2 + ( rotationPerAmp-C*L )**2
				return ( C*I*xp.sin(C*I*L)-1/f )**2 + ( rotationPerAmp+C*L )**2 # rad/A so don't multiply by I
			x0 = ( self.calibration, self.length )
			self.calibration,self.length = minimize(dz,x0=x0)['x']
			#import matplotlib.pyplot as plt
			#Cs = xp.linspace(0,4*x0,100) ; Ys=Cs*I*xp.sin(Cs*I*self.length)#*xp.cos(Cs*I*self.length)
			#plt.plot(Cs,Ys) ; plt.plot(Cs,[1/f]*100) ; plt.show()
		print("for lens",self.name,"found calibration factor",self.calibration,"and len",self.length)

		#print("calibration_from_f_and_I found",self.calibration,"from starting guess",x0,dz(self.calibration))

class Prism(Element):
	def __init__(self, name:str='', 
				 position:float=None, length:float=0.,
				 radius:float=None, angle:float=45., w:float=1., g:float=1., k1:float=0.,
				strength:float=0, calibration:float=None) -> SEASerializable:
		"""Prism.

		Parameters
		----------
		name : str, optional
			Name given to the lens, by default ''
		position : float, optional
			The position of the element along the z-axis, by default 0
		length : int, optional
			Length of the element, by default 0
		strength : float, optional
			Defined as the field strength (related to inverse focal length,
			see equations in brown1983), by default 0
			Note this in not the focusing strength (K) and is simply f.
			A thin lens is defind as KL=-1/fas L goes to zero.
		calibration : float, optional
			Currnet calibration of the lens in units of ???/A, by default None
		label : bool, optional
			If the element should be labeled when plotted, by default False
		print_fancy : bool, optional
			If a fancy table should be used when printed, by default True

		To do
		-----
		#TODO: Include dispersion (R16). See Egerton eq 2.14
		#TODO: Include other spectrometer aberrations. See Egerton eq 2.13
		"""

		self.angle = angle
		self.angle_rad = xp.deg2rad(angle)
		if radius is not None and length is not None:
			raise ValueError('Only specify the length or radius as providing both with radius is not unique.')
		elif radius is not None:
			length = self.angle_rad * radius
		elif length is not None:
			radius = length/self.angle_rad
		else:
			raise ValueError('Either radius or length need to be specified.')

		super().__init__(name=name,kind='Prism')
		self._position = position
		self.length = length
		self.strength = strength
		self.calibration = calibration
		self.radius = radius
		self.w = w
		self.g = g
		self.K1 = k1

	def focus_matrix(self,
					 #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
					 ) -> xp.ndarray:
		r"""Transfer matrix for the entrance/exit surfaces of the spectrometer used for ray propogation.
		"""
		m = xp.eye(6)

		if self.strength!=0:
			m[1,0] = xp.tan(self.strength) / self.radius
			if self.K1 == 0:
				m[1,0] = - xp.tan(self.strength) / self.radius
			else: #include fringe fields
				psi = (self.g/self.radius) * self.K1 * (1+xp.sin(self.strength)**2)/xp.cos(self.strength)
				m[2:4,2:4] = - xp.tan(self.strength - psi) / self.radius
		else: #drif
			pass

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])
	
	def bending_matrix(self,
					   s:float,
					   #type='Hills' TODO: Add `type` in paramaters to describe the type of transfer matrix. Hill's, Twiss, etc.
					   ) -> xp.ndarray:
		r"""Transfer matrix for the bending of the spectrometer used for ray propogation.
		"""
		m = xp.eye(6)
		u = s/self.radius

		if self.strength!=0:
			m[0,0] = xp.cos(u)
			m[1,1] = xp.cos(u)
			m[0,1] = self.radius * xp.sin(u)
			m[1,0] = - xp.sin(u)/self.radius
		else:
			m = Drift.transfer_matrix(s)

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])
		
	def transfer_matrix(self) -> xp.ndarray:
		r"""Transfer matrix for ray propogation.
		"""
		
		m_focus1 = self.focus_matrix()
		m_bend   = self.bending_matrix(self.strength)
		m_focus2 = self.focus_matrix()

		m = m_focus2 @ m_bend @ m_focus1

		return fix_mat_dims(m,["x","xt","y","yt","z","E"])

	def propagate_wave(self, signal, mode:Literal['fixed','scaled','hybrid']='fixed',
					   s_min:float=1e-3, log:list=None, absorb:float=0.1,
				   crossover:Literal['flat','jump']='flat', rotate:bool=False):
		"""Wave-optics propagation through a prism/spectrometer (not implemented).

		Overrides :meth:`Element.propagate_wave` for every mode. A dispersive
		bending prism is not a simple thin phase screen plus drift, so
		wave-optics support is deferred.

		Parameters
		----------
		signal : Signal or seashells._Wavefield or seashells._ScaledWavefield
			Incoming wavefield.
		mode : {'fixed', 'scaled', 'hybrid'}, optional
			Unused.
		s_min : float, optional
			Unused.
		log : list, optional
			Unused.

		Returns
		-------
		Signal
			Never returns.

		Raises
		------
		NotImplementedError
			Always; wave-optics propagation is not implemented for ``Prism``.
		"""
		raise NotImplementedError("Wave-optics propagation is not implemented for Prism (spectrometer).")


element_list = ["Element"] + [subclass.__name__ for subclass in Element.__subclasses__()]
