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
	sea_available = True
except Exception as e:
	print(str(e)+". This is just a warning from rayTEM.seashells: rayTEM pySEA integration will not work.")
	pass

if sea_available:
	class SEASerializable(_SEASerializable):
		def to_sea(self,filename):												# sea_eco's SEASerializable will default naming like "Drift_2" but we'll set them to None so we can easily undo the naming later
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
			super().to_sea(filename)
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
		def to_sea(self,filename):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")
		def from_sea(self,filename):
			print("WARNING: sea_eco does not appear to be installed, so microscope.to_sea is unavailable. Please install sea_eco, or use microscope.save instead")


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


#SEASerializable.from_sea will create a purely-SEASerializable object. rayTEM objects (Element, MicroscopeSection, Microscope, etc) will have inherited from SEASerializable, so we may need to reinstantiate rayTEM objects to ensure they have the rayTEM-specific functionality (e.g. "scope=Microscope(); scope.from_sea" will find scope.sections is a list of purely-SEASerializable objects without functions like "propagate_ray").
def safeReinstantiate(source,cls):
	from .elements import Drift,Lens,Source,Dipole,Quadrapole
	from .assemblies import Microscope,MicroscopeSection
	cls = {"Drift":Drift, "QLens":Lens, "Thin lens":Lens, "Source":Source, "Microscope":Microscope, "Section":MicroscopeSection, "Dipole":Dipole, "Thin dipole":Dipole, "Quad":Quadrapole, "Thin quad":Quadrapole }[cls]
	dic = source.__dict__
	allowed_kwargs = inspect.signature(cls).parameters.keys()	# infer allowed kwargs from the class itself
	dic = { k:v for k,v in dic.items() if k in allowed_kwargs }	# and filter kwargs to those accepted
	return cls(**dic)
