"""
seashells serves as a wrapper around the sea_eco SEASerializable object, enabling easy integration with sea_eco.
to install sea_eco and rayTEM side-by-side, this module should be installed as a plugin inside the pySEA folder, as a sibling to sea_eco, also in the pySEA folder
if sea_eco IS installed, the SEASerializable object is wrapped, enabling direct access to, or wrapping of, all SEASerializable functions
to wrap a function, we simply define it, do our custom stuff, then call super().funcname to call up to SEASerializable's version, and do more custom stuff after
if sea_eco is NOT installed, we create a dummy SEASerializable object, with dummy functions (which raise warnings) for the functions we expect to use
All objects we then expect to integrate with sea_eco then inherit the SEASerializable class from here (whether it is wrapping sea_eco's SEASerializable, or using the dummy version)
"""

import sys,inspect
from warnings import warn

sea_available = False
try:
	sys.path.insert(1,"../../")
	from pySEA.sea_eco.architecture.base_structure import SEASerializable as _SEASerializable
	from pySEA.sea_eco.architecture.base_structure import Signal as _Signal, Dimension as _Dimension, SignalSet as _SignalSet
	from pySEA.sea_eco.architecture.base_structure import Metadata as _Metadata
	sea_available = True
except Exception as e:
	print(str(e)+". This is just a warning from rayTEM.seashells: rayTEM pySEA integration will not work.")
	pass

if sea_available:
	class SEASerializable(_SEASerializable):
		def to_sea(self,filename,force_datasets=None,validate_seaid=None):		# sea_eco's SEASerializable will default naming like "Drift_2" but we'll set them to None so we can easily undo the naming later
			if hasattr(self,"sections"):										# user might to_sea a Microscope object (loop section > elements)
				if self.name is None or len(self.name)==0:
					self.name = "None_Microscope"
				for s,sec in enumerate(self.sections):
					if sec.name is None or len(sec.name)==0:
						sec.name = "None_"+str(s)
					for e,ele in enumerate(sec.elements):
						if ele.name is None or len(ele.name)==0:
							ele.name = "None_"+str(e)
			elif hasattr(self,"elements"):										# ... or a MicroscopeSection object (loop elements only)
				if self.name is None or len(self.name)==0:
					self.name = "None_Section"
				for e,ele in enumerate(self.elements):
					if ele.name is None or len(ele.name)==0:
						ele.name = "None_"+str(e)
			super().to_sea(filename,
						   force_datasets=[] if force_datasets is None else list(force_datasets),
						   validate_seaid=validate_seaid)
		def from_hdf5_group(self,group):
			"""Populate this object from an HDF5 group, then reshape an old identity.

			Parameters
			----------
			group : h5py.Group
				The group sea_eco reads this object from.

			Returns
			-------
			None

			Notes
			-----
			A column saved before 2026-10-12 stores ``Provenance`` as a SEA ID
			string; :func:`_reshape_legacy_identity` wraps it. Every read goes
			through here, whether the object is a file's root or a child.
			"""
			super().from_hdf5_group(group)
			_reshape_legacy_identity(self)
		def from_sea(self,filename):											# sea_eco will reaload purely-SEASerializable objects, so reinitalize as our shared-inheritance object
			super().from_sea(filename)
			if hasattr(self,"sections"):
				if "None" in self.name:
					self.name = ""
				sections = []
				for s,sec in enumerate(self.sections):
					if "None" in sec.name:
						sec.name = ""
					elements = []
					for e,ele in enumerate(sec.elements):
						if "None" in ele.name:
							ele.name = ""
						elements.append( safeReinstantiate(ele,ele.kind) )
					sec.elements = elements
					sections.append( safeReinstantiate(sec,"Section") )
				self.sections = sections
			elif hasattr(self,"elements"):
				if "None" in self.name:
					self.name = ""
				elements = []
				for e,ele in enumerate(self.elements):
					if "None" in ele.name:
						ele.name = ""
					elements.append( safeReinstantiate(ele,ele.kind) )
				self.elements = elements
else:
	class SEASerializable():
		def __init__(self):
			pass
		def to_sea(self,filename,force_datasets=None,validate_seaid=None):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")
		def from_sea(self,filename):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")


def new_provenance(clsid: str):
	"""Mint the identity of a rayTEM object, through the sea_eco seam.

	rayTEM does not import sea_eco (or sea_sand) directly -- everything that
	crosses into the pySEA data layer goes through this module, so that a
	rayTEM installed on its own still runs. Minting is no different: with the
	ecosystem present this returns a sea_eco ``Provenance`` whose ``Self`` is a
	newly minted SEA ID, and without it returns ``None`` and the object simply
	has no identity.

	Parameters
	----------
	clsid : str
		The five-character CLSID to mint under, prefix plus serial (for
		example ``"TWN01"`` for a digital twin). Serial ``00`` is the class
		base; real instances of a non-enumerated class use ``01``.

	Returns
	-------
	pySEA.sea_eco.architecture.base_structure.Provenance | None
		The new identity (``Self`` minted, no relations beyond the creator
		sea_eco stamps), or ``None`` when the pySEA data layer is not
		installed.

	Raises
	------
	None
		A missing or unhappy minter yields ``None`` rather than raising: an ID
		is provenance, and failing to stamp one must never stop a simulation
		from running.

	Warnings
	--------
	sea-sand *normalizes* rather than rejects: it silently substitutes the
	characters Crockford Base32 omits and pads or truncates to five, so
	``"TOOLONG"`` mints as ``T0010`` and ``"TEMUU"`` as ``TEM00``. Nothing
	here can catch a typo -- that is what
	:func:`~pySEA.sea_eco.seaid_conformance.validate_seaid_conformance` is
	for, which compares the minted prefix against what the class declares.

	See Also
	--------
	pySEA.sea_eco.architecture.base_structure.Provenance
		The identity this returns: ``Self`` plus ``Relations``.
	pySEA.sea_eco.seaid_conformance.validate_seaid_conformance
		The check that a minted ID matches the class carrying it.

	Notes
	-----
	The identity is stamped once, at construction, and is written the way
	sea_eco writes every identity: a ``Provenance`` group holding ``Self``
	(the SEA ID) and, when any are stored, ``Relations``. The id string is
	``provenance.Self.seaid``; ``str(provenance)`` is a description.

	Examples
	--------
	>>> new_provenance("TWN01").Self.seaid       # doctest: +SKIP
	'0CE-TWN01-261012-084500-K3QM7A'
	"""

	if not sea_available:
		return None
	try:
		from pySEA.sea_eco.architecture.base_structure import Provenance
		return Provenance(clsid=clsid)
	except Exception:														# sea_eco/sea_sand absent, or the CLSID is not one they will mint; see Raises
		return None


def _reshape_legacy_identity(target) -> None:
	"""Wrap a SEA ID string read as ``Provenance`` into a sea_eco Provenance.

	Remove after 2026-12-03. Columns saved between 2026-09-22 and 2026-10-12
	stored their twin id as a plain SEA ID string attribute; sea_eco's
	identity model stores a ``Provenance`` group (``Self`` plus
	``Relations``). Reading the old shape wraps the same id, so the twin keeps
	its identity, and warns once per object.

	Parameters
	----------
	target : SEASerializable
		An object just populated from a file.

	Returns
	-------
	None

	Raises
	------
	None

	Warns
	-----
	DeprecationWarning
		When a string identity is wrapped.

	Notes
	-----
	Wrapping never mints and never stamps a creator: the id is the one the
	file recorded.
	"""

	identity = target.__dict__.get("Provenance")
	if not isinstance(identity, str) or not identity.strip() or identity.strip() == "None":
		return
	from pySEA.sea_eco.architecture.base_structure import Provenance
	warn("This column was saved before 2026-10-12 and stores its Provenance as a plain SEA ID; "
		 "it is read as a Provenance with that Self. Save it again to rewrite it. Reading the old "
		 "layout stops after 2026-12-03.", DeprecationWarning, stacklevel=3)
	object.__setattr__(target, "Provenance", Provenance(identity.strip()))


class _Wavefield:
	"""Lightweight fallback wavefield container used when sea_eco is unavailable.

	Mirrors the read surface of a sea_eco wavefield ``Signal`` (data plus transverse
	sampling, wavelength, and z position) so wave-optics propagation still runs and
	round-trips in-memory without the calibrated-Signal machinery. Serialization
	(``to_sea``) is not available in this mode.

	Parameters
	----------
	data : numpy.ndarray
		Complex field, shape ``(ny, nx)`` for a single plane or ``(nz, ny, nx)``
		for a stack.
	dx, dy : float
		Transverse sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	z : float or numpy.ndarray or None
		Plane position (2D) or array of plane positions (3D), metres.

	Attributes
	----------
	data : numpy.ndarray
		The complex field.
	dx, dy : float
		Transverse sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	z : float or numpy.ndarray or None
		Plane position(s) (metres).
	"""
	def __init__(self, data, dx, dy, wavelength, z):
		self.data = data
		self.dx = dx
		self.dy = dy
		self.wavelength = wavelength
		self.z = z


def make_wavefield_signal(data, dx, dy, wavelength, z=None, name="wavefield"):
	"""Wrap a complex field array as a calibrated sea_eco wavefield ``Signal``.

	Builds a ``Signal`` whose transverse axes carry the pixel-size calibration and
	whose metadata carries the wavelength (and, for a single plane, the z position);
	for a stack, ``z`` becomes an unstructured z axis with explicit coordinates.
	When sea_eco is unavailable a :class:`_Wavefield` fallback is returned instead so
	wave propagation still works in-memory (with a warning).

	Parameters
	----------
	data : numpy.ndarray
		Complex field. ``(ny, nx)`` for one plane, or ``(nz, ny, nx)`` for a stack.
	dx, dy : float
		Transverse sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	z : float or Sequence[float] or None, optional
		Plane position for a single plane, or the ``nz`` plane positions for a
		stack. Stack positions become an unstructured z axis.
	name : str, optional
		Signal name, by default ``"wavefield"``.

	Returns
	-------
	Signal or _Wavefield
		A calibrated complex ``Signal`` when sea_eco is present, otherwise a
		:class:`_Wavefield` fallback.

	Related
	-------
	read_wavefield : Inverse accessor returning ``(data, dx, dy, wavelength, z)``.
	"""
	import numpy as _np
	data = _np.asarray(data)
	if not sea_available:
		warn("sea_eco is not installed; make_wavefield_signal returns a lightweight "
			 "_Wavefield fallback (no .sea serialization).")
		return _Wavefield(data, dx, dy, wavelength, z)
	ny, nx = data.shape[-2], data.shape[-1]
	# size is set on every axis so Dimensions.ndim matches data.ndim (Signal.show relies on it)
	xdim = _Dimension(name="x", space="position", scale=dx, offset=-(nx // 2) * dx, size=nx, units="m")
	ydim = _Dimension(name="y", space="position", scale=dy, offset=-(ny // 2) * dy, size=ny, units="m")
	# calibration is also mirrored into metadata for robust, parse-free read-back
	meta = {"wavelength_m": float(wavelength), "dx_m": float(dx), "dy_m": float(dy)}
	if data.ndim == 3:
		zvals = _np.asarray(z if z is not None else range(data.shape[0]), dtype=float)
		# unstructured z: pass a representative scale/offset so sea_eco skips its
		# values-without-scale inference (which is only a plotting fallback here).
		zscale = float(zvals[1] - zvals[0]) if len(zvals) > 1 else 1.0
		zdim = _Dimension(name="z", space="position", scale=zscale, offset=float(zvals[0]),
						  size=len(zvals), values=zvals, units="m", unstructured=True)
		dimensions = [zdim, ydim, xdim]
	else:
		meta["z_m"] = float(z) if z is not None else 0.0
		dimensions = [ydim, xdim]
	return _Signal(data=data, name=name, dimensions=dimensions, metadata=meta, signal_type="Image")


def make_rays_signalset(rays, I, R, components, name="rays"):
	"""Wrap traced rays and their intensity/rotation as a sea_eco ``SignalSet``.

	Assembles a Signal-backed view of a ray-mode result: the geometric ray table
	plus the separate intensity (``I``) and rotation (``R``) arrays, sharing an
	unstructured plane-``z`` axis and an unstructured ray-index axis. The ray table
	additionally carries an unstructured component axis (the ``convention`` columns,
	listed in metadata). Returns ``None`` (with a warning) when sea_eco is absent.

	Parameters
	----------
	rays : numpy.ndarray
		Geometric rays, shape ``(n_planes, n_rays, n_components)``.
	I : numpy.ndarray
		Per-plane, per-ray intensity, shape ``(n_planes, n_rays)``.
	R : numpy.ndarray
		Per-plane, per-ray cumulative rotation, shape ``(n_planes, n_rays)``.
	components : Sequence[str]
		Names of the ray-vector components (the ``convention`` list).
	name : str, optional
		Name for the SignalSet, by default ``"rays"``.

	Returns
	-------
	SignalSet or None
		A SignalSet of ``[rays, I, R]`` when sea_eco is present, else ``None``.

	Related
	-------
	MicroscopeSection.rays_signalset, Microscope.rays_signalset
	"""
	import numpy as _np
	if not sea_available:
		warn("sea_eco is not installed; rays_signalset requires sea_eco and returns None.")
		return None
	rays = _np.asarray(rays) ; I = _np.asarray(I) ; R = _np.asarray(R)
	n_planes, n_rays, n_comp = rays.shape
	z_index = components.index("z") if "z" in components else 0
	zvals = _np.asarray(rays[:, 0, z_index], dtype=float)
	# plane-z is genuinely non-uniform (unstructured); ray/component are index axes.
	# A representative scale/offset is passed so sea_eco skips its values-without-scale
	# inference path (a plotting fallback that also emits stray debug output).
	zscale = float(zvals[1] - zvals[0]) if len(zvals) > 1 else 1.0
	zdim = _Dimension(name="plane_z", space="position", scale=zscale, offset=float(zvals[0]),
					  size=n_planes, values=zvals, units="m", unstructured=True)
	rdim = _Dimension(name="ray", scale=1, offset=0, size=n_rays, units="", unstructured=True)
	cdim = _Dimension(name="component", scale=1, offset=0, size=n_comp, units="", unstructured=True)
	ray_sig = _Signal(data=rays, name="rays", dimensions=[zdim, rdim, cdim],
					  metadata={"components": list(components)})
	I_sig = _Signal(data=I, name="I", dimensions=[zdim, rdim])
	R_sig = _Signal(data=R, name="R", dimensions=[zdim, rdim])
	return _SignalSet(signals=[ray_sig, I_sig, R_sig], main_signal=0, name=name)


class _Phase:
	"""Lightweight fallback phase container used when sea_eco is unavailable.

	Mirrors the read surface consumed by the wave propagators: the real phase
	array plus its application domain. Produced by the phase-Signal factories in
	place of a calibrated ``Signal``.

	Parameters
	----------
	data : numpy.ndarray
		Real phase χ (radians), shape ``(ny, nx)``.
	space : str
		Application domain: ``'position'`` (real-space screen) or
		``'scattering'`` (reciprocal-space propagator phase).
	name : str
		Human-readable name.

	Attributes
	----------
	data : numpy.ndarray
		The phase array.
	space : str
		The application domain.
	name : str
		The name.
	"""
	def __init__(self, data, space, name):
		self.data = data
		self.space = space
		self.name = name


def make_screen_phase_signal(data, dx, dy, name="phase screen"):
	"""Wrap a real-space screen as a space-tagged Signal.

	The returned Signal's transverse Dimensions carry the pixel calibration and
	``space='position'``, which is how the wave propagators recognize a screen
	applied by multiplying the field. Falls back to :class:`_Phase` when
	sea_eco is absent.

	The data may be **real**, meaning a phase χ in radians applied as
	``exp(iχ)``, or **complex**, meaning a transmission ``T`` applied directly
	— amplitude and phase in one object, which is what a physical plate has.
	See :func:`waveoptics.apply_phase` for the convention.

	Parameters
	----------
	data : numpy.ndarray
		Real phase χ (radians) or complex transmission ``T``. ``(ny, nx)`` for
		a screen at one plane, or ``(n, ny, nx)`` for a **volume** — a medium
		described slice by slice, which the propagator turns into a multislice
		through the element's body.
	dx, dy : float
		Sample spacings of the grid the phase was built on (metres).
	name : str, optional
		Signal name, by default ``"phase screen"``.

	Returns
	-------
	Signal or _Phase
		The domain-tagged phase.

	Related
	-------
	make_kernel_phase_signal : Reciprocal-space counterpart.
	phase_space_of : Reads the domain tag back.
	"""
	import numpy as _np
	data = _np.asarray(data)
	if not sea_available:
		return _Phase(data, "position", name)
	ny, nx = data.shape[-2:]
	xdim = _Dimension(name="x", space="position", scale=dx, offset=-(nx // 2) * dx, size=nx, units="m")
	ydim = _Dimension(name="y", space="position", scale=dy, offset=-(ny // 2) * dy, size=ny, units="m")
	dims = [ydim, xdim]
	if data.ndim == 3:
		# a VOLUME screen: slices through the element's body, evenly spaced by
		# construction (the propagator divides the length by their count), so
		# the leading axis is an index rather than a calibrated z
		dims = [_Dimension(name="slice", scale=1, offset=0, size=data.shape[0],
						   units="", unstructured=True)] + dims
	return _Signal(data=data, name=name, dimensions=dims, signal_type="Image")


def make_kernel_phase_signal(data, fx, fy, name="propagator phase"):
	"""Wrap a reciprocal-space propagator phase χ(f_x, f_y) as a space-tagged Signal.

	The returned Signal's Dimensions carry the (unshifted, fftfreq-ordered)
	spatial frequencies and ``space='scattering'``, which is how the wave
	propagators recognize a phase to apply in the FFT domain. Falls back to
	:class:`_Phase` when sea_eco is absent.

	Parameters
	----------
	data : numpy.ndarray
		Real phase χ (radians), shape ``(ny, nx)``, in fftfreq order.
	fx, fy : numpy.ndarray
		Spatial-frequency axes (1/m), fftfreq order, lengths ``nx``/``ny``.
	name : str, optional
		Signal name, by default ``"propagator phase"``.

	Returns
	-------
	Signal or _Phase
		The domain-tagged phase.

	Related
	-------
	make_screen_phase_signal : Real-space counterpart.
	phase_space_of : Reads the domain tag back.
	"""
	import numpy as _np
	data = _np.asarray(data)
	if not sea_available:
		return _Phase(data, "scattering", name)
	fxdim = _Dimension(name="f_x", space="scattering", scale=1, offset=0, size=len(fx),
					   values=_np.asarray(fx, float), units="1/m", unstructured=True)
	fydim = _Dimension(name="f_y", space="scattering", scale=1, offset=0, size=len(fy),
					   values=_np.asarray(fy, float), units="1/m", unstructured=True)
	return _Signal(data=data, name=name, dimensions=[fydim, fxdim], signal_type="Image")


def phase_space_of(phase):
	"""Return the application domain of a phase produced by the factories above.

	Parameters
	----------
	phase : Signal or _Phase
		A domain-tagged phase.

	Returns
	-------
	str
		``'position'`` or ``'scattering'`` (read from the fallback attribute or
		from the first Dimension's ``space``).
	"""
	if isinstance(phase, _Phase):
		return phase.space
	return phase.dimensions.dimensions[0].space


def grid_of(dimensions):
	"""Normalize a transverse-grid description to ``(ny, nx, dy, dx)``.

	Accepts either a sea_eco ``Dimensions`` object whose last two axes are the
	calibrated transverse y/x dimensions (the wavefield-Signal layout), or a
	plain ``(shape, dx, dy)`` tuple used on the sea_eco-absent fallback path.

	Parameters
	----------
	dimensions : Dimensions or tuple
		Grid description: a ``Dimensions`` with trailing y/x axes, or
		``((ny, nx), dx, dy)``.

	Returns
	-------
	tuple
		``(ny, nx, dy, dx)``.

	Raises
	------
	TypeError
		If ``dimensions`` is neither form.
	"""
	if isinstance(dimensions, tuple) and len(dimensions) == 3:
		(ny, nx), dx, dy = dimensions
		return ny, nx, dy, dx
	dims = dimensions.dimensions
	ydim, xdim = dims[-2], dims[-1]
	return int(ydim.size), int(xdim.size), float(ydim.scale), float(xdim.scale)


def as_ndarray(x):
	"""Return the raw ndarray behind a sea_eco ``Signal`` (passthrough for arrays).

	Discriminates on the presence of a ``dimensions`` attribute (Signals have one;
	plain ndarrays and the ``_Wavefield`` fallback do not), so it cannot be fooled
	by ``numpy.ndarray.data`` (which is a memoryview, not the array).

	Parameters
	----------
	x : Signal or numpy.ndarray or _Wavefield
		A calibrated Signal or a raw array-like.

	Returns
	-------
	numpy.ndarray
		``x.data`` when ``x`` is a Signal, otherwise ``x`` unchanged.
	"""
	return x.data if hasattr(x, "dimensions") else x


def make_covariance_signal(covariance, z, components, name="covariance"):
	"""Wrap a stack of per-plane covariance matrices as a calibrated sea_eco ``Signal``.

	Builds a ``(n_planes, n_comp, n_comp)`` Signal for the beam-envelope result: an
	unstructured plane-``z`` axis plus two component (row/col) index axes whose
	labels (the ``convention`` columns) are recorded in metadata. Returns the raw
	ndarray (with a warning) when sea_eco is unavailable.

	Parameters
	----------
	covariance : numpy.ndarray
		Per-plane covariance matrices, shape ``(n_planes, n_comp, n_comp)``.
	z : Sequence[float]
		The ``n_planes`` plane positions (metres) for the unstructured z axis.
	components : Sequence[str]
		Names of the phase-space components (the ``convention`` list).
	name : str, optional
		Signal name, by default ``"covariance"``.

	Returns
	-------
	Signal or numpy.ndarray
		A calibrated ``Signal`` when sea_eco is present, else ``covariance`` unchanged.

	Related
	-------
	make_rays_signalset, make_wavefield_signal, as_ndarray
	"""
	import numpy as _np
	covariance = _np.asarray(covariance)
	if not sea_available:
		warn("sea_eco is not installed; make_covariance_signal returns a raw ndarray.")
		return covariance
	zvals = _np.asarray(z, dtype=float)
	ncomp = covariance.shape[1]
	zscale = float(zvals[1] - zvals[0]) if len(zvals) > 1 else 1.0
	zdim = _Dimension(name="z", space="position", scale=zscale, offset=float(zvals[0]),
					  size=len(zvals), values=zvals, units="m", unstructured=True)
	# row/col are regular (structured) index axes so Signal.show renders a heatmap,
	# not an unstructured scatter.
	rowdim = _Dimension(name="row", scale=1, offset=0, size=ncomp, units="", unstructured=False)
	coldim = _Dimension(name="col", scale=1, offset=0, size=ncomp, units="", unstructured=False)
	return _Signal(data=covariance, name=name, dimensions=[zdim, rowdim, coldim],
				   metadata={"components": list(components)}, signal_type="Image")


def read_wavefield(signal):
	"""Read ``(data, dx, dy, wavelength, z)`` from a wavefield Signal or fallback.

	Inverse of :func:`make_wavefield_signal`. Reads the transverse sampling and
	wavelength from metadata (mirrored there by the factory) so it does not depend
	on parsing ``Dimension`` objects, and works for both a real sea_eco ``Signal``
	and the :class:`_Wavefield` fallback.

	Parameters
	----------
	signal : Signal or _Wavefield
		A wavefield produced by :func:`make_wavefield_signal`.

	Returns
	-------
	tuple
		``(data, dx, dy, wavelength, z)`` where ``data`` is the complex field and
		``z`` is a float (single plane) or ndarray (stack) or ``None``.
	"""
	if isinstance(signal, _Wavefield):
		return signal.data, signal.dx, signal.dy, signal.wavelength, signal.z
	meta = signal.metadata.to_dict() if signal.metadata is not None else {}
	dx = meta.get("dx_m")
	dy = meta.get("dy_m")
	wavelength = meta.get("wavelength_m")
	z = meta.get("z_m", None)
	return signal.data, dx, dy, wavelength, z


class _ScaledWavefield:
	"""Lightweight fallback scaled-wavefield container used when sea_eco is unavailable.

	Mirrors the read surface of a scaled-wavefield ``Signal`` produced by
	:func:`make_scaled_wavefield_signal`: the reduced field ``U(ξ, η)`` of the
	scaled-Fresnel factorization ``ψ = (1/s)·U·exp[ik(x²+y²)/2R]`` plus its scaled
	sampling and the frame state ``(s, R, τ)``, so scaled propagation still runs
	and round-trips in-memory without the calibrated-Signal machinery.

	Parameters
	----------
	data : numpy.ndarray
		Complex reduced field ``U``, shape ``(neta, nxi)``.
	dxi, deta : float
		Scaled transverse sample spacings Δξ, Δη (metres).
	wavelength : float
		Wavelength (metres).
	s : float
		Scale factor at this plane (physical pixel size is ``|s|·Δξ``).
	R : float
		Wavefront radius of curvature (metres); ``numpy.inf`` for a flat frame.
	tau : float
		Reduced propagation coordinate ``τ = ∫ dz/s²`` accumulated so far.
	z : float or None
		Physical plane position (metres).

	Attributes
	----------
	data : numpy.ndarray
		The complex reduced field ``U``.
	dxi, deta : float
		Scaled sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	s, R, tau : float
		Chart state at this plane.
	z : float or None
		Physical plane position (metres).
	"""
	def __init__(self, data, dxi, deta, wavelength, s, R, tau, z, z_cross=None):
		self.data = data
		self.dxi = dxi
		self.deta = deta
		self.wavelength = wavelength
		self.s = s
		self.R = R
		self.tau = tau
		self.z = z
		self.z_cross = z_cross


def make_scaled_wavefield_signal(U, dxi, deta, wavelength, s, R, tau, z=None,
								 z_cross=None, tag=None, name="scaled wavefield"):
	"""Wrap a reduced field ``U(ξ, η)`` as a single-plane scaled-wavefield ``Signal``.

	Builds the in-flight state of scaled-Fresnel propagation: the reduced field of
	the factorization ``ψ = (1/s)·U(ξ, η)·exp[ik(x²+y²)/2R]`` on a ``Signal`` whose
	ξ/η axes carry the *scaled* sampling Δξ/Δη as their calibration, with the chart
	scalars ``(s, R, τ)``, the wavelength, and the physical plane position in
	metadata. When sea_eco is unavailable a :class:`_ScaledWavefield` fallback is
	returned instead (with a warning).

	Parameters
	----------
	U : numpy.ndarray
		Complex reduced field, shape ``(neta, nxi)``.
	dxi, deta : float
		Scaled transverse sample spacings Δξ, Δη (metres). The physical pixel size
		at this plane is ``|s|·Δξ``.
	wavelength : float
		Wavelength (metres).
	s : float or Sequence[float]
		Scale factor at this plane; an ``(s_x, s_y)`` pair on an anisotropic
		frame (stored as per-axis metadata keys ``s_x``/``s_y``).
	R : float or Sequence[float]
		Wavefront radius of curvature (metres); ``numpy.inf`` for a flat frame;
		per-axis pair on an anisotropic frame (keys ``R_x_m``/``R_y_m``).
	tau : float or Sequence[float]
		Accumulated reduced propagation coordinate ``τ = ∫ dz/s²``; per-axis
		pair on an anisotropic frame (keys ``tau_x``/``tau_y``).
	z : float or None, optional
		Physical plane position (metres), stored in metadata.
	z_cross : float, Sequence, or None, optional
		Position of the crossover a flat frame is currently traversing
		(set by the hybrid frame-switching policy; ``None`` otherwise).
		Stored in metadata as ``z_cross_m`` only when set; a per-axis pair
		(entries ``None`` where inactive) stores ``z_cross_x_m``/``z_cross_y_m``.
	tag : str or None, optional
		Frame-event tag for interior planes logged by the hybrid policy
		(``'flatten'``/``'crossover'``/``'rediverge'``); stored in metadata as
		``frame_tag`` only when set.
	name : str, optional
		Signal name, by default ``"scaled wavefield"``.

	Returns
	-------
	Signal or _ScaledWavefield
		A calibrated complex ``Signal`` when sea_eco is present, otherwise a
		:class:`_ScaledWavefield` fallback.

	Related
	-------
	read_scaled_wavefield : Inverse accessor.
	make_scaled_wave_signalset : Stacked multi-plane variant.
	make_wavefield_signal : The physical (unscaled) wavefield factory.
	"""
	import numpy as _np
	U = _np.asarray(U)
	if not sea_available:
		warn("sea_eco is not installed; make_scaled_wavefield_signal returns a "
			 "lightweight _ScaledWavefield fallback (no .sea serialization).")
		out = _ScaledWavefield(U, dxi, deta, wavelength, s, R, tau, z, z_cross)
		out.tag = tag
		return out
	neta, nxi = U.shape[-2], U.shape[-1]
	xidim = _Dimension(name="xi", space="position", scale=dxi, offset=-(nxi // 2) * dxi, size=nxi, units="m")
	etadim = _Dimension(name="eta", space="position", scale=deta, offset=-(neta // 2) * deta, size=neta, units="m")
	# frame scalars mirrored into metadata for robust, parse-free read-back;
	# isotropic frames keep the scalar keys (old files load unchanged), an
	# anisotropic frame stores per-axis keys instead
	meta = {"wavelength_m": float(wavelength), "dxi_m": float(dxi), "deta_m": float(deta),
			"z_m": float(z) if z is not None else 0.0}
	if _np.ndim(s) == 0 and _np.ndim(R) == 0 and _np.ndim(tau) == 0:
		meta.update({"s": float(s), "R_m": float(R), "tau": float(tau)})
	else:
		from .waveoptics import axis_components
		s_x, s_y = axis_components(s)
		R_x, R_y = axis_components(R)
		t_x, t_y = axis_components(tau)
		meta.update({"s_x": s_x, "s_y": s_y, "R_x_m": R_x, "R_y_m": R_y,
					 "tau_x": t_x, "tau_y": t_y})
	if z_cross is not None:
		if _np.ndim(z_cross) == 0:
			meta["z_cross_m"] = float(z_cross)
		else:
			if z_cross[0] is not None:
				meta["z_cross_x_m"] = float(z_cross[0])
			if z_cross[1] is not None:
				meta["z_cross_y_m"] = float(z_cross[1])
	if tag is not None:
		meta["frame_tag"] = str(tag)
	return _Signal(data=U, name=name, dimensions=[etadim, xidim], metadata=meta, signal_type="Image")


def read_scaled_wavefield(signal):
	"""Read ``(U, dxi, deta, wavelength, s, R, tau, z)`` from a scaled wavefield.

	Inverse of :func:`make_scaled_wavefield_signal`. Reads the scaled sampling and
	the frame state from metadata (mirrored there by the factory) so it does not
	depend on parsing ``Dimension`` objects, and works for both a real sea_eco
	``Signal`` and the :class:`_ScaledWavefield` fallback.

	Parameters
	----------
	signal : Signal or _ScaledWavefield
		A scaled wavefield produced by :func:`make_scaled_wavefield_signal`.

	Returns
	-------
	tuple
		``(U, dxi, deta, wavelength, s, R, tau, z)`` where ``U`` is the complex
		reduced field and the remaining entries are floats — ``s``/``R``/``tau``
		come back as ``(x, y)`` pairs when the plane was stored anisotropic.

	Related
	-------
	make_scaled_wavefield_signal, read_wavefield
	"""
	if isinstance(signal, _ScaledWavefield):
		return (signal.data, signal.dxi, signal.deta, signal.wavelength,
				signal.s, signal.R, signal.tau, signal.z)
	meta = signal.metadata.to_dict() if signal.metadata is not None else {}
	if "s_x" in meta:		# anisotropic frame: per-axis keys
		return (signal.data, meta.get("dxi_m"), meta.get("deta_m"), meta.get("wavelength_m"),
				(meta["s_x"], meta["s_y"]), (meta["R_x_m"], meta["R_y_m"]),
				(meta["tau_x"], meta["tau_y"]), meta.get("z_m", None))
	return (signal.data, meta.get("dxi_m"), meta.get("deta_m"), meta.get("wavelength_m"),
			meta.get("s"), meta.get("R_m"), meta.get("tau"), meta.get("z_m", None))


def scaled_frame_crossover(signal):
	"""Read the crossover marker of a scaled wavefield's flat frame, if any.

	The hybrid frame-switching policy records the position of the crossover a
	flat frame is currently traversing so the marker survives element
	boundaries; this accessor reads it back from either a real sea_eco
	``Signal`` (metadata key ``z_cross_m``) or the :class:`_ScaledWavefield`
	fallback.

	Parameters
	----------
	signal : Signal or _ScaledWavefield
		A scaled wavefield produced by :func:`make_scaled_wavefield_signal`.

	Returns
	-------
	float, tuple, or None
		The crossover position ``z_cross`` (metres), ``None`` when the frame
		is not traversing a crossover, or a per-axis ``(x, y)`` pair (entries
		``None`` where inactive) on an anisotropic frame.

	Related
	-------
	make_scaled_wavefield_signal : Stores the marker.
	"""
	if isinstance(signal, _ScaledWavefield):
		return signal.z_cross
	meta = signal.metadata.to_dict() if signal.metadata is not None else {}
	if "z_cross_x_m" in meta or "z_cross_y_m" in meta:		# per-axis markers
		return (meta.get("z_cross_x_m", None), meta.get("z_cross_y_m", None))
	return meta.get("z_cross_m", None)


def scaled_frame_tag(signal):
	"""Read the frame-event tag of a scaled wavefield plane, if any.

	Interior planes logged by the hybrid frame-switching policy carry a tag
	(``'flatten'``/``'crossover'``/``'rediverge'``); ordinary element-exit
	planes carry none. Works for both a real sea_eco ``Signal`` (metadata key
	``frame_tag``) and the :class:`_ScaledWavefield` fallback.

	Parameters
	----------
	signal : Signal or _ScaledWavefield
		A scaled wavefield produced by :func:`make_scaled_wavefield_signal`.

	Returns
	-------
	str or None
		The tag, or ``None`` for an untagged plane.

	Related
	-------
	make_scaled_wavefield_signal : Stores the tag.
	"""
	if isinstance(signal, _ScaledWavefield):
		return getattr(signal, "tag", None)
	meta = signal.metadata.to_dict() if signal.metadata is not None else {}
	return meta.get("frame_tag", None)


def make_scaled_wave_signalset(U, dxi, deta, wavelength, s, R, tau, z, tags=None,
							   name="scaled wave"):
	"""Wrap a stack of reduced fields and their frame state as a sea_eco ``SignalSet``.

	Assembles the stacked result of a scaled-Fresnel run: the reduced field
	``U(z, η, ξ)`` on an unstructured plane-``z`` axis plus calibrated ξ/η axes
	(the ξ,η grid is fixed for the whole run), with companion 1-D Signals ``s(z)``,
	``R(z)``, ``tau(z)`` sharing the same plane-z axis — the same pattern as
	:func:`make_rays_signalset`. Returns ``None`` (with a warning) when sea_eco
	is absent.

	Parameters
	----------
	U : numpy.ndarray
		Reduced-field stack, shape ``(n_planes, neta, nxi)``.
	dxi, deta : float
		Scaled transverse sample spacings Δξ, Δη (metres), shared by every plane.
	wavelength : float
		Wavelength (metres), stored in metadata.
	s, R, tau : Sequence
		Chart state per plane, each of length ``n_planes``; entries are scalars
		or per-axis ``(x, y)`` pairs (anisotropic planes). Physical pixel size
		at plane ``i`` is ``|s[i]|·Δξ``; ``R`` may contain ``numpy.inf``.
	z : Sequence[float]
		The ``n_planes`` physical plane positions (metres) for the unstructured
		z axis.
	tags : Sequence[str or None], optional
		Per-plane frame-event tags from the hybrid policy (``None`` entries
		for ordinary planes). When given, a ``frame`` companion Signal is added
		(an integer frame index that increments at each ``flatten``/
		``rediverge`` switch) and the tags are stored on the U Signal's
		metadata as the comma-joined string ``plane_tags``.
	name : str, optional
		Name for the SignalSet, by default ``"scaled wave"``.

	Returns
	-------
	SignalSet or None
		A SignalSet of ``[U, s, R, tau]`` (per-axis companions ``s_x``/``s_y``,
		``R_x``/``R_y``, ``tau_x``/``tau_y`` replace a companion whose axes
		differ anywhere; plus ``frame`` when ``tags`` is given; main Signal
		``U``) when sea_eco is present, else ``None``.

	Related
	-------
	make_scaled_wavefield_signal, make_rays_signalset
	"""
	import numpy as _np
	if not sea_available:
		warn("sea_eco is not installed; make_scaled_wave_signalset requires sea_eco and returns None.")
		return None
	U = _np.asarray(U)
	n_planes, neta, nxi = U.shape
	zvals = _np.asarray(z, dtype=float)
	zscale = float(zvals[1] - zvals[0]) if len(zvals) > 1 else 1.0
	zdim = _Dimension(name="plane_z", space="position", scale=zscale, offset=float(zvals[0]),
					  size=n_planes, values=zvals, units="m", unstructured=True)
	xidim = _Dimension(name="xi", space="position", scale=dxi, offset=-(nxi // 2) * dxi, size=nxi, units="m")
	etadim = _Dimension(name="eta", space="position", scale=deta, offset=-(neta // 2) * deta, size=neta, units="m")
	meta = {"wavelength_m": float(wavelength), "dxi_m": float(dxi), "deta_m": float(deta)}
	if tags is not None:
		meta["plane_tags"] = ",".join(tag or "" for tag in tags)
	U_sig = _Signal(data=U, name="U", dimensions=[zdim, etadim, xidim], metadata=meta,
					signal_type="Image")
	signals = [U_sig]

	def _axes(seq):
		# per-plane scalar-or-pair sequence -> per-axis float arrays
		xs = _np.array([v if _np.ndim(v) == 0 else v[0] for v in seq], dtype=float)
		ys = _np.array([v if _np.ndim(v) == 0 else v[1] for v in seq], dtype=float)
		return xs, ys

	for label, seq in (("s", s), ("R", R), ("tau", tau)):
		xs, ys = _axes(seq)
		if _np.array_equal(xs, ys):		# isotropic run: one scalar companion
			signals.append(_Signal(data=xs, name=label, dimensions=[zdim]))
		else:							# anisotropic: per-axis companions
			signals.append(_Signal(data=xs, name=label + "_x", dimensions=[zdim]))
			signals.append(_Signal(data=ys, name=label + "_y", dimensions=[zdim]))
	if tags is not None:
		# integer frame index: increments at each frame switch (per-axis
		# switches like 'flatten-x' count too)
		frame = _np.cumsum([1 if (tag or "").startswith(("flatten", "rediverge", "jump"))
							else 0 for tag in tags])
		signals.append(_Signal(data=frame.astype(float), name="frame", dimensions=[zdim]))
	return _SignalSet(signals=signals, main_signal=0, name=name)


# --- DISPLAY SIGNALS ---------------------------------------------------------
# Views of a stored propagation result (ray diagram, beam envelope, scaled-wave
# cross-section), built here so the only sea_eco imports stay in this module.
# They are derived on every call and never stored on the column: the stored
# result is the storage (notes/eric/PLAN_2026-08-25_derived-signal-properties.md).

#: First part of every marker annotation's name. A marker is named
#: ``raytem.<kind>.<nnn>.<label>`` -- ``kind`` is one of the keys below (or
#: ``section``), ``nnn`` counts that kind from 000 in drawing order -- so a
#: renderer, or :func:`read_display_signal`, can tell the kinds apart.
_MARKER_PREFIX = "raytem"

#: Colour stored on each line-marker kind (PLOT-8 honours ``style.color``).
#: Mid-tones, so they read on a light line plot and on the magma cross-section.
_MARKER_COLORS = {"element": "#808080", "diff": "#00ffff", "crossover": "#00ffff",
				  "image": "#ff00ff"}

#: Colours section spans cycle through: plot2D's g/b/r section shading.
_SECTION_COLORS = ("#008000", "#0000ff", "#ff0000")


class _DisplaySignal:
	"""Lightweight fallback display Signal used when sea_eco is unavailable.

	Mirrors what :func:`read_display_signal` reads from a real display Signal
	-- the data, its axes, the markers and the descriptive metadata -- so a
	caller that draws from a display Signal (:meth:`assemblies.Microscope.show`
	for the scaled-wave cross-section) works the same without the pySEA data
	layer. Serialization (``to_sea``) is not available in this mode.

	Parameters
	----------
	data : numpy.ndarray
		The displayed values, in the axis order of ``axes``.
	name : str
		Display name, e.g. ``"ray diagram (x)"``.
	axes : Sequence[tuple]
		One ``(name, values, units)`` per data axis, in data-axis order.
	markers : Sequence[Mapping]
		Marker records as :func:`read_display_signal` returns them.
	metadata : Mapping
		Descriptive metadata (``kind`` and the view's own keys).

	Attributes
	----------
	data : numpy.ndarray
		The displayed values.
	name : str
		Display name.
	axes : list of tuple
		``(name, values, units)`` per data axis.
	markers : list of dict
		Marker records.
	metadata : dict
		Descriptive metadata.

	Methods
	-------
	None
		A plain container.

	Raises
	------
	None

	Related
	-------
	read_display_signal : Reads this and a real display Signal alike.
	_Wavefield : The same fallback pattern for wavefields.

	Notes
	-----
	The axes are deliberately not called ``dimensions``: :func:`as_ndarray`
	treats any object with a ``dimensions`` attribute as a Signal.
	"""
	def __init__(self, data, name, axes, markers, metadata):
		self.data = data
		self.name = name
		self.axes = list(axes)
		self.markers = [dict(m) for m in markers]
		self.metadata = dict(metadata)


class _AnnotationDraft:
	"""A bare ``metadata`` holder the display markers are drafted on.

	sea_eco's :func:`~pySEA.sea_eco.annotations.mark_axis_value` and
	:func:`~pySEA.sea_eco.annotations.mark_axis_span` record one edit per call
	on a tracked Signal. A display Signal carries dozens of markers, and
	recording each as a step bloated a 50 kB ray diagram to 1.7 MB of history
	and took two seconds. Marking this untracked holder instead builds the same
	``Annotation`` objects with nothing recorded; the finished collection then
	becomes part of the new Signal's original state.

	Attributes
	----------
	metadata : Metadata
		Receives ``Display.Annotations`` / ``Display.ROI`` as marks are made.

	Methods
	-------
	None
		A plain container.

	Raises
	------
	None

	Related
	-------
	_marker_annotations : The only user.
	"""
	def __init__(self):
		self.metadata = _Metadata()


def _marker_records(markers):
	"""Normalize marker requests into records carrying kind, label, position and colour.

	Parameters
	----------
	markers : Sequence[Mapping]
		Requests from the column: ``{"kind", "label", "z"}`` for a line, or
		``{"kind", "label", "z0", "z1"}`` for a span (positions in metres).

	Returns
	-------
	list of dict
		The same records, ``z``/``z0``/``z1`` as floats, each with the
		``color`` it is stored with: sections cycle through
		``_SECTION_COLORS``, lines take ``_MARKER_COLORS[kind]``.

	Raises
	------
	KeyError
		If a request lacks ``kind``, ``label`` or a position.

	Related
	-------
	_marker_annotations : Turns these records into sea_eco annotations.
	"""
	records = [] ; spans = 0
	for marker in markers:
		record = {"kind": str(marker["kind"]), "label": str(marker["label"])}
		if "z0" in marker:
			record.update(z0=float(marker["z0"]), z1=float(marker["z1"]),
						  color=_SECTION_COLORS[spans % len(_SECTION_COLORS)])
			spans += 1
		else:
			record.update(z=float(marker["z"]),
						  color=_MARKER_COLORS.get(record["kind"], "#ffae42"))
		records.append(record)
	return records


def _marker_annotations(records, along, extent):
	"""Build the sea_eco annotations for a display Signal's markers.

	Every marker goes through sea_eco's own constructors -- a span through
	``mark_axis_span``, a line through ``mark_axis_value`` -- so the stored
	shape (PLOT-8 ``Span`` / ``Line``, calibrated coordinates) is defined in
	one place, sea_eco's.

	Parameters
	----------
	records : Sequence[Mapping]
		Marker records from :func:`_marker_records`.
	along : str
		The dimension the markers sit on: the horizontal axis of the plot
		they are authored for.
	extent : Sequence[float]
		``(low, high)`` every ``Line`` spans on the other axis, in its units.

	Returns
	-------
	tuple
		``(AnnotationCollection, ROICollection)``; the ROI collection is empty.

	Raises
	------
	ValueError
		If a span's ``z0`` is not below its ``z1``.

	Related
	-------
	_AnnotationDraft : Why the marks are made on a holder, not the Signal.

	Notes
	-----
	``mark_*`` replaces every annotation whose name starts with its ``owner``,
	so each marker gets its own owner, ``raytem.<kind>.<nnn>``; the name is
	``<owner>.<label>`` and ``text`` is the label.
	"""
	from pySEA.sea_eco.annotations import (mark_axis_span, mark_axis_value,
										   get_display_annotations)
	draft = _AnnotationDraft()
	counts = {}
	for record in records:
		kind = record["kind"]
		index = counts.get(kind, 0) ; counts[kind] = index + 1
		owner = f"{_MARKER_PREFIX}.{kind}.{index:03d}"
		if "z0" in record:
			mark_axis_span(draft, record["z0"], record["z1"], along, name=record["label"],
						   owner=owner, color=record["color"], text=record["label"])
		else:
			mark_axis_value(draft, record["z"], along, name=record["label"], owner=owner,
							extent=extent, color=record["color"], text=record["label"])
	return get_display_annotations(draft)


def _finite_range(data):
	"""Low and high finite value of an array, for the extent a line marker spans.

	Parameters
	----------
	data : numpy.ndarray
		Values that may contain NaN (ghosted rays).

	Returns
	-------
	tuple of float
		``(min, max)`` over the finite values, or ``(0.0, 0.0)`` when there are
		none.

	Raises
	------
	None
	"""
	import numpy as _np
	finite = _np.asarray(data, dtype=float)
	finite = finite[_np.isfinite(finite)]
	return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)


def _plane_z_dimension(name, z):
	"""An unstructured, explicitly valued z axis in metres.

	The same construction the existing result factories use for plane z: the
	values are genuinely non-uniform and repeat at zero-length elements and
	section boundaries, and a representative scale/offset is passed so sea_eco
	skips its values-without-scale inference.

	Parameters
	----------
	name : str
		Dimension name (``"plane_z"`` or ``"z"``).
	z : numpy.ndarray
		Plane positions (metres).

	Returns
	-------
	Dimension
		The sea_eco dimension.

	Raises
	------
	None
	"""
	zscale = float(z[1] - z[0]) if len(z) > 1 else 1.0
	return _Dimension(name=name, space="position", scale=zscale, offset=float(z[0]),
					  size=len(z), values=z, units="m", unstructured=True)


def _display_signal(data, dimensions, records, along, extent, name, quantity, metadata,
					signal_type=None):
	"""Assemble a display Signal from prepared parts, markers attached.

	Parameters
	----------
	data : numpy.ndarray
		The displayed values.
	dimensions : Sequence[Dimension]
		One sea_eco dimension per data axis.
	records : Sequence[Mapping]
		Marker records from :func:`_marker_records`.
	along : str
		Dimension the markers sit on.
	extent : Sequence[float]
		``(low, high)`` the line markers span.
	name : str
		Signal name.
	quantity : Mapping
		The signal quantity, ``{"name", "units"}``.
	metadata : Mapping
		Descriptive metadata stored beside ``Display``.
	signal_type : str or None, optional
		sea_eco semantic type hint, by default ``None``.

	Returns
	-------
	Signal
		The display Signal; markers live in ``metadata.Display.Annotations``.

	Raises
	------
	ValueError
		If a span marker does not order.
	"""
	annotations, rois = _marker_annotations(records, along, extent)
	meta = _Metadata(dict(metadata))
	meta.Display = _Metadata()
	meta.Display.Annotations = annotations
	meta.Display.ROI = rois
	return _Signal(data=data, name=name, dimensions=list(dimensions),
				   signal_quantities=[dict(quantity)], metadata=meta,
				   signal_type=signal_type)


def make_ray_diagram_signal(positions, z, markers, axis="x", frame="rotating",
							name=None, source=""):
	"""Wrap a ray diagram -- every ray's transverse position per plane -- as a sea_eco Signal.

	The overlaid-rays view of a ray-mode result, as data a renderer can draw:
	shape ``[ray, plane_z]``, values the transverse position (metres) of each
	ray at each logged plane, NaN once a ray is ghosted. Section spans,
	element positions and conjugate planes travel with it as PLOT-8
	annotations along ``plane_z``. When sea_eco is unavailable a
	:class:`_DisplaySignal` fallback is returned instead (with a warning).

	Parameters
	----------
	positions : numpy.ndarray
		Transverse positions, shape ``(n_rays, n_planes)`` (metres).
	z : Sequence[float]
		The ``n_planes`` plane positions (metres), as logged -- repeats at
		zero-length elements and section boundaries are kept.
	markers : Sequence[Mapping]
		``{"kind", "label", "z"}`` lines and ``{"kind", "label", "z0", "z1"}``
		spans, positions in metres. ``kind`` is ``section``, ``element``,
		``diff`` or ``image``.
	axis : {'x', 'y'}, optional
		Which transverse coordinate ``positions`` holds, by default ``'x'``.
	frame : {'rotating', 'lab'}, optional
		The frame the positions are in, by default ``'rotating'``.
	name : str or None, optional
		Signal name, by default ``"ray diagram (<axis>)"``.
	source : str, optional
		Name of the section or column the rays came from, stored as metadata
		``source``, by default ``""``.

	Returns
	-------
	Signal or _DisplaySignal
		Dimensions ``ray`` (unitless index) and ``plane_z`` (unstructured,
		values in m); signal quantity ``<axis>`` in m; metadata ``kind =
		"ray-diagram"``, ``axis``, ``frame``, ``source``.

	Raises
	------
	ValueError
		If a span marker does not order.

	Related
	-------
	assemblies.Microscope.ray_diagram_signal : Builds the inputs.
	read_display_signal : Inverse accessor.

	Notes
	-----
	Line markers span the finite range of ``positions``: the vertical axis of
	the line plot they are authored for (PLOT-8 clips, never rescales).
	"""
	import numpy as _np
	positions = _np.asarray(positions, dtype=float)
	z = _np.asarray(z, dtype=float)
	name = name or f"ray diagram ({axis})"
	records = _marker_records(markers)
	metadata = {"kind": "ray-diagram", "axis": str(axis), "frame": str(frame), "source": str(source)}
	if not sea_available:
		warn("sea_eco is not installed; make_ray_diagram_signal returns a lightweight "
			 "_DisplaySignal fallback (no .sea serialization).")
		axes = [("ray", _np.arange(positions.shape[0]), ""), ("plane_z", z, "m")]
		return _DisplaySignal(positions, name, axes, records, metadata)
	rdim = _Dimension(name="ray", scale=1, offset=0, size=positions.shape[0], units="")
	return _display_signal(positions, [rdim, _plane_z_dimension("plane_z", z)], records,
						   "plane_z", _finite_range(positions), name,
						   {"name": str(axis), "units": "m"}, metadata)


def make_envelope_signal(widths, z, quantities, units, markers, name="beam envelope",
						 source=""):
	"""Wrap per-plane RMS beam widths as a sea_eco Signal with column markers.

	The beam-envelope view of a covariance-mode result: shape
	``[quantity, z]``, one row per RMS width. The quantities do not share a
	unit -- positions are metres, angles radians -- so the signal quantity is
	unitless and each row's unit is stored in metadata ``quantity_units``.
	When sea_eco is unavailable a :class:`_DisplaySignal` fallback is
	returned instead (with a warning).

	Parameters
	----------
	widths : numpy.ndarray
		RMS widths, shape ``(n_quantities, n_planes)``.
	z : Sequence[float]
		The ``n_planes`` plane positions (metres).
	quantities : Sequence[str]
		Row labels, the categorical values of the ``quantity`` dimension
		(e.g. ``sigma_x``).
	units : Sequence[str]
		The unit of each row, same order as ``quantities``.
	markers : Sequence[Mapping]
		As for :func:`make_ray_diagram_signal`.
	name : str, optional
		Signal name, by default ``"beam envelope"``.
	source : str, optional
		Name of the section or column, stored as metadata ``source``, by
		default ``""``.

	Returns
	-------
	Signal or _DisplaySignal
		Dimensions ``quantity`` (categorical, values ``quantities``) and ``z``
		(unstructured, values in m); signal quantity ``rms width`` (unitless);
		metadata ``kind = "beam-envelope"``, ``quantity_units``, ``source``.

	Raises
	------
	ValueError
		If ``units`` and ``quantities`` differ in length, or a span marker
		does not order.

	Related
	-------
	assemblies.Microscope.envelope_signal : Builds the inputs.
	read_display_signal : Inverse accessor.

	Notes
	-----
	Line markers span the finite range of all rows together, so they cover
	the view of whichever row is drawn.
	"""
	import numpy as _np
	widths = _np.asarray(widths, dtype=float)
	z = _np.asarray(z, dtype=float)
	quantities = [str(q) for q in quantities] ; units = [str(u) for u in units]
	if len(units) != len(quantities):
		raise ValueError(f"make_envelope_signal needs one unit per quantity; got "
						 f"{len(units)} units for {len(quantities)} quantities.")
	records = _marker_records(markers)
	metadata = {"kind": "beam-envelope", "quantity_units": units, "source": str(source)}
	if not sea_available:
		warn("sea_eco is not installed; make_envelope_signal returns a lightweight "
			 "_DisplaySignal fallback (no .sea serialization).")
		axes = [("quantity", _np.asarray(quantities), ""), ("z", z, "m")]
		return _DisplaySignal(widths, name, axes, records, metadata)
	qdim = _Signal.build_component_dimension(quantities, units="", name="quantity")
	return _display_signal(widths, [qdim, _plane_z_dimension("z", z)], records, "z",
						   _finite_range(widths), name, {"name": "rms width", "units": ""},
						   metadata)


def make_wave_cross_section_signal(profile, z, x, markers, name="wave cross-section",
								   source=""):
	"""Wrap a scaled-wave |ψ(x, 0, z)| cross-section as a sea_eco image Signal.

	The wave analog of the ray diagram: shape ``[x, z]`` -- z is the
	**last** (horizontal) axis, the orientation the figure has always been
	drawn in and the one its markers are authored for (a PLOT-8 ``Line`` or
	``Span`` marks the horizontal axis). Element, crossover and image-plane
	markers travel with it. When sea_eco is unavailable a
	:class:`_DisplaySignal` fallback is returned instead (with a warning).

	Parameters
	----------
	profile : numpy.ndarray
		Per-plane peak-normalized ``|ψ(x, 0)|``, shape ``(n_x, n_planes)``.
	z : Sequence[float]
		The ``n_planes`` plane positions (metres), ascending.
	x : Sequence[float]
		The common, uniform transverse axis (metres), symmetric about 0.
	markers : Sequence[Mapping]
		``{"kind", "label", "z"}`` lines; ``kind`` is ``element``,
		``crossover`` or ``image``.
	name : str, optional
		Signal name, by default ``"wave cross-section"``.
	source : str, optional
		Name of the column, stored as metadata ``source``, by default ``""``.

	Returns
	-------
	Signal or _DisplaySignal
		Dimensions ``x`` (structured, m) and ``z`` (unstructured, values in
		m); signal quantity ``|psi| / peak`` (unitless); signal type
		``Image``; metadata ``kind = "wave-cross-section"``,
		``normalization = "peak per plane"``, ``source``.

	Raises
	------
	None

	Related
	-------
	assemblies.Microscope.wave_cross_section_signal : Builds the inputs.
	read_display_signal : Inverse accessor.

	Notes
	-----
	Line markers span the x axis as stored, its first to last coordinate.
	``x`` is stored as ``offset = x[0]`` and a uniform scale, so ``-x[0]``
	recovers the axis half-width exactly.
	"""
	import numpy as _np
	profile = _np.asarray(profile, dtype=float)
	z = _np.asarray(z, dtype=float)
	x = _np.asarray(x, dtype=float)
	records = _marker_records(markers)
	metadata = {"kind": "wave-cross-section", "normalization": "peak per plane",
				"source": str(source)}
	if not sea_available:
		warn("sea_eco is not installed; make_wave_cross_section_signal returns a "
			 "lightweight _DisplaySignal fallback (no .sea serialization).")
		return _DisplaySignal(profile, name, [("x", x, "m"), ("z", z, "m")], records, metadata)
	xscale = float(x[1] - x[0]) if len(x) > 1 else 1.0
	xdim = _Dimension(name="x", space="position", scale=xscale, offset=float(x[0]),
					  size=len(x), units="m")
	stored = _np.asarray(xdim.values, dtype=float)		# the axis as a reader will see it
	return _display_signal(profile, [xdim, _plane_z_dimension("z", z)], records, "z",
						   (float(stored[0]), float(stored[-1])), name,
						   {"name": "|psi| / peak", "units": ""}, metadata,
						   signal_type="Image")


def read_display_signal(signal):
	"""Read ``(data, axes, markers, metadata)`` from a display Signal or its fallback.

	Inverse of the ``make_*_signal`` display factories. Works for a real
	sea_eco Signal (including one reloaded from a ``.sea`` file) and for the
	:class:`_DisplaySignal` fallback, so a caller can draw from either.

	Parameters
	----------
	signal : Signal or _DisplaySignal
		A display Signal from :func:`make_ray_diagram_signal`,
		:func:`make_envelope_signal` or :func:`make_wave_cross_section_signal`.

	Returns
	-------
	tuple
		``(data, axes, markers, metadata)``: the values; one
		``(name, values, units)`` per data axis; marker records
		``{"kind", "label", "color", "z"}`` (lines) or
		``{"kind", "label", "color", "z0", "z1"}`` (spans) in stored order;
		and the descriptive metadata (without ``Display`` and without the
		metadata container's own ``Provenance`` stamp).

	Raises
	------
	None

	Related
	-------
	read_wavefield : The same accessor pattern for wavefields.

	Notes
	-----
	Only annotations named with the ``raytem.`` prefix are read back as
	markers; any a user added to the Signal afterwards are left out.
	"""
	import numpy as _np
	if isinstance(signal, _DisplaySignal):
		return (signal.data, list(signal.axes), [dict(m) for m in signal.markers],
				dict(signal.metadata))
	from pySEA.sea_eco.annotations import get_display_annotations
	axes = [(d.name, _np.asarray(d.values), d.units) for d in signal.dimensions.dimensions]
	markers = []
	for annotation in get_display_annotations(signal)[0]:
		parts = str(annotation.name).split(".", 3)
		if len(parts) != 4 or parts[0] != _MARKER_PREFIX:
			continue
		coordinates = annotation.coordinates
		record = {"kind": parts[1], "label": annotation.text,
				  "color": annotation.style.get("color")}
		if annotation.annotation_type == "Span":
			record.update(z0=float(coordinates["x0"]), z1=float(coordinates["x1"]))
		else:
			record["z"] = float(coordinates["x0"])
		markers.append(record)
	metadata = signal.metadata.to_dict() if signal.metadata is not None else {}
	for key in ("Display", "Provenance"):		# the markers, and the Metadata's own stamp
		metadata.pop(key, None)
	return signal.data, axes, markers, metadata


#SEASerializable.from_sea will create a purely-SEASerializable object. rayTEM objects (Element, MicroscopeSection, Microscope, etc) will have inherited from SEASerializable, so we may need to reinstantiate rayTEM objects to ensure they have the rayTEM-specific functionality (e.g. "scope=Microscope(); scope.from_sea" will find scope.sections is a list of purely-SEASerializable objects without functions like "propagate_ray").
def safeReinstantiate(source,cls):
	from .elements import Drift,Lens,Source,Gun,Dipole,Quadrapole,Aperture,Prism,AberrationScreen
	from .assemblies import Microscope,MicroscopeSection
	# Every `kind` an element can report has to appear here or the object cannot
	# be reloaded at all. Aperture and Prism were missing, so any saved column
	# containing one raised KeyError on load -- which meant no column with an
	# aperture had ever round-tripped.
	known = {"Drift":Drift, "QLens":Lens, "Thin lens":Lens, "Source":Source, "Gun":Gun,
			 "Microscope":Microscope, "Section":MicroscopeSection,
			 "Dipole":Dipole, "Thin dipole":Dipole,
			 "Quad":Quadrapole, "Thin quad":Quadrapole,
			 "Aperture":Aperture, "Prism":Prism,
			 "AberrationScreen":AberrationScreen}
	if cls not in known:
		raise KeyError(
			f"cannot reinstantiate a stored object of kind {cls!r}: it is not in "
			f"safeReinstantiate's table, which knows {sorted(known)}. A new element "
			"class needs its `kind` added there, or it will save but never load.")
	cls = known[cls]
	dic = source.__dict__
	allowed_kwargs = inspect.signature(cls).parameters.keys()	# infer allowed kwargs from the class itself
	kwargs = { k:v for k,v in dic.items() if k in allowed_kwargs }	# and filter kwargs to those accepted
	obj = cls(**kwargs)
	# The constructor's only job here is to produce an object of the right
	# class (methods, defaults, validation); the RECORDED state then wins,
	# verbatim. The kwarg filter above cannot be trusted to carry everything:
	# any attribute whose __dict__ spelling differs from its constructor kwarg
	# (a property storing `_x` for kwarg `x`), or that is not a constructor
	# parameter at all, would be silently dropped -- Source.beam_current
	# reverted to its default on every reload this way. Writing the stored
	# __dict__ back restores exactly what was saved, including names the class
	# no longer declares (same leniency as Element.from_hdf5_group: an old
	# file must stay openable). Nested objects are safe because the from_sea
	# rebuild reinstantiates children before their parent, so `dic` already
	# holds real rayTEM objects.
	obj.__dict__.update(dic)
	return obj
