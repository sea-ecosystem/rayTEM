"""Paraxial scalar wave-optics primitives for rayTEM.

This module holds the backend-neutral numerical core of the wave-optics
propagation mode: initial-field builders, free-space (angular-spectrum)
propagation, and the pointwise phase/mask operators for lenses, quadrupoles,
dipoles, and apertures. All functions operate on plain complex ``numpy`` arrays
and metre-based sampling; the sea_eco ``Signal`` wrapping (calibration +
serialization) lives in :mod:`seashells`, and the orchestration (reading a
wavefield, applying an element, re-wrapping) lives in :mod:`elements`.

Related
-------
seashells.make_wavefield_signal : Wrap an array as a calibrated wavefield Signal.
elements.Element.propagate_wave : Applies these operators per element.

Notes
-----
Everything here is paraxial (small-angle). Free-space propagation uses the
Fresnel (paraxial) form of the angular-spectrum transfer function, which
preserves the transverse sampling grid across propagation — this is what lets a
whole z-stack of wavefields share one calibrated ``(x, y)`` grid.
"""

from __future__ import annotations

import numpy as np


def transverse_coordinates(shape: tuple, dx: float, dy: float) -> tuple:
	"""Centered real-space coordinate grids for a wavefield.

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx : float
		Sample spacing along x (metres).
	dy : float
		Sample spacing along y (metres).

	Returns
	-------
	tuple of np.ndarray
		``(X, Y)`` coordinate arrays, each of shape ``(ny, nx)``, with the
		origin at index ``(ny//2, nx//2)``.
	"""
	ny, nx = shape
	x = (np.arange(nx) - nx // 2) * dx
	y = (np.arange(ny) - ny // 2) * dy
	return np.meshgrid(x, y)


def plane_wave(shape: tuple) -> np.ndarray:
	"""Build a unit-amplitude plane wave (uniform field).

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.

	Returns
	-------
	np.ndarray
		Complex field of ones, shape ``(ny, nx)``.
	"""
	return np.ones(shape, dtype=complex)


def gaussian_field(shape: tuple, dx: float, dy: float,
				   sigma_x: float, sigma_y: float) -> np.ndarray:
	"""Build a Gaussian amplitude field centered on the grid.

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	sigma_x, sigma_y : float
		Gaussian amplitude 1/e widths (metres) along x and y.

	Returns
	-------
	np.ndarray
		Complex Gaussian field, shape ``(ny, nx)``.
	"""
	X, Y = transverse_coordinates(shape, dx, dy)
	return np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2))).astype(complex)


def point_source(shape: tuple) -> np.ndarray:
	"""Build a discrete point source (single lit pixel at the grid center).

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.

	Returns
	-------
	np.ndarray
		Complex field, zero everywhere except unit amplitude at
		``(ny//2, nx//2)``.
	"""
	ny, nx = shape
	field = np.zeros(shape, dtype=complex)
	field[ny // 2, nx // 2] = 1.0
	return field


def kernel_phase(shape: tuple, dx: float, dy: float, wavelength: float,
				 dz: float, include_carrier: bool = True) -> np.ndarray:
	r"""Reciprocal-space phase of the paraxial free-space transfer function.

	Returns the real phase :math:`\chi(k_x, k_y)` such that the angular-spectrum
	transfer function is :math:`H = e^{i\chi}`:

	.. math::

		\chi = k\,dz - (k_x^2 + k_y^2)\, dz / (2k), \qquad k = 2\pi/\lambda

	on the unshifted ``numpy.fft.fftfreq`` frequency grid. The leading carrier
	term ``k·dz`` is included only when ``include_carrier`` is True; the scaled
	representation propagates the carrier-free paraxial wave (handoff Eq 32),
	so the scaled path passes ``include_carrier=False``.

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres, or scaled-coordinate units for the scaled path).
	wavelength : float
		Wavelength (metres).
	dz : float or Sequence[float]
		Propagation distance (metres; the scaled path passes Δτ, per-axis
		``(Δτ_x, Δτ_y)`` on anisotropic frames).
	include_carrier : bool, optional
		Include the on-axis carrier term ``k·dz``, by default True (meaningful
		for isotropic/physical distances only).

	Returns
	-------
	np.ndarray
		Real phase array, shape ``(ny, nx)``, in fftfreq (unshifted) order.

	Related
	-------
	angular_spectrum_propagate : Applies ``exp(i·kernel_phase)`` in the FFT domain.
	"""
	ny, nx = shape
	k = 2 * np.pi / wavelength
	kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
	ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
	KX, KY = np.meshgrid(kx, ky)
	dzx, dzy = axis_components(dz)
	chi = -(KX**2 * dzx + KY**2 * dzy) / (2 * k)
	if include_carrier:
		chi = chi + k * dzx
	return chi


def apply_phase(field: np.ndarray, phase: np.ndarray,
				space: str = "position") -> np.ndarray:
	r"""Apply a scalar phase to a field as :math:`e^{i\chi}` in its domain.

	Real-space screens (``space='position'``) multiply the field directly;
	reciprocal-space phases (``space='scattering'``, e.g. the free-space kernel)
	are applied in the FFT domain. This is the single application primitive used
	by both the fixed-grid and scaled wave propagators.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	phase : np.ndarray
		Real phase χ (radians), same shape as ``field`` (reciprocal phases in
		unshifted fftfreq order).
	space : {'position', 'scattering'}, optional
		Domain of ``phase``, by default ``'position'``.

	Returns
	-------
	np.ndarray
		Field with ``exp(i·phase)`` applied, shape ``(ny, nx)``.

	Raises
	------
	ValueError
		If ``space`` is not ``'position'`` or ``'scattering'``.
	"""
	if space == "position":
		return field * np.exp(1j * phase)
	if space == "scattering":
		return np.fft.ifft2(np.fft.fft2(field) * np.exp(1j * phase))
	raise ValueError(f"Unknown phase space {space!r}; expected 'position' or 'scattering'.")


def angular_spectrum_propagate(field: np.ndarray, dx: float, dy: float,
							   wavelength: float, dz: float,
							   include_carrier: bool = True) -> np.ndarray:
	r"""Propagate a scalar field a distance ``dz`` in free space (paraxial).

	Uses the Fresnel (paraxial) angular-spectrum transfer function

	.. math::

		H(k_x, k_y) = e^{i k\, dz}\, e^{-i (k_x^2 + k_y^2)\, dz / (2k)}, \quad k = 2\pi/\lambda

	applied in the spatial-frequency domain. The transverse sampling grid is
	preserved (``dx``/``dy`` unchanged), so successive planes share one grid.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	dz : float
		Propagation distance (metres). ``dz == 0`` returns the field unchanged.
	include_carrier : bool, optional
		Include the on-axis carrier ``e^{ik·dz}``, by default True. The scaled
		representation (handoff Eq 32) propagates the carrier-free paraxial wave
		and passes False.

	Returns
	-------
	np.ndarray
		Propagated complex field, shape ``(ny, nx)``.

	Related
	-------
	kernel_phase : The real phase this function exponentiates.
	"""
	if np.all(np.asarray(dz) == 0):
		return field.astype(complex, copy=True)
	chi = kernel_phase(field.shape, dx, dy, wavelength, dz, include_carrier=include_carrier)
	return apply_phase(field, chi, space="scattering")


def quadratic_phase(shape: tuple, dx: float, dy: float, wavelength: float,
					power_x: float, power_y: float) -> np.ndarray:
	r"""Real-space quadratic (focusing) phase screen.

	Returns the real phase :math:`\chi = -k (P_x x^2 + P_y y^2)/2` (handoff
	Eq 12), where the focal powers ``power_x``/``power_y`` are inverse focal
	lengths ``1/f``. Equal powers describe a round lens; opposite powers the
	quadrupole saddle (one axis focusing, the other diverging).

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres, or physical spacings ``s·Δξ`` on a scaled grid).
	wavelength : float
		Wavelength (metres).
	power_x, power_y : float
		Focal powers ``1/f`` along x and y (1/metres).

	Returns
	-------
	np.ndarray
		Real phase array χ (radians), shape ``(ny, nx)``.

	Related
	-------
	focal_phase : Applies ``exp(i·quadratic_phase)`` to a field.
	"""
	X, Y = transverse_coordinates(shape, dx, dy)
	k = 2 * np.pi / wavelength
	return -k * (power_x * X**2 + power_y * Y**2) / 2


def linear_phase(shape: tuple, dx: float, dy: float, wavelength: float,
				 tilt_x: float, tilt_y: float) -> np.ndarray:
	r"""Real-space linear phase ramp (wavefront tilt / dipole steering).

	Returns the real phase :math:`\chi = k (\theta_x x + \theta_y y)`.

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres, or physical spacings ``s·Δξ`` on a scaled grid).
	wavelength : float
		Wavelength (metres).
	tilt_x, tilt_y : float
		Deflection angles (radians) along x and y.

	Returns
	-------
	np.ndarray
		Real phase array χ (radians), shape ``(ny, nx)``.

	Related
	-------
	tilt_phase : Applies ``exp(i·linear_phase)`` to a field.
	"""
	X, Y = transverse_coordinates(shape, dx, dy)
	k = 2 * np.pi / wavelength
	return k * (tilt_x * X + tilt_y * Y)


def focal_phase(field: np.ndarray, dx: float, dy: float, wavelength: float,
				power_x: float, power_y: float) -> np.ndarray:
	r"""Apply a (possibly astigmatic) thin focusing phase.

	Multiplies by :math:`\exp[-i k (P_x x^2 + P_y y^2)/2]`, where the focal powers
	``power_x``/``power_y`` are inverse focal lengths (``1/f``). Equal powers give a
	round lens; opposite powers give a quadrupole. Thin wrapper around
	:func:`quadratic_phase`.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	power_x, power_y : float
		Focal powers ``1/f`` along x and y (1/metres).

	Returns
	-------
	np.ndarray
		Field with the focusing phase applied.
	"""
	return apply_phase(field, quadratic_phase(field.shape, dx, dy, wavelength, power_x, power_y))


def tilt_phase(field: np.ndarray, dx: float, dy: float, wavelength: float,
			   tilt_x: float, tilt_y: float) -> np.ndarray:
	r"""Apply a linear phase ramp that tilts the wavefront (dipole steering).

	Multiplies by :math:`\exp[i k (\theta_x x + \theta_y y)]`.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	tilt_x, tilt_y : float
		Deflection angles (radians) along x and y.

	Returns
	-------
	np.ndarray
		Field with the tilt phase applied.
	"""
	if tilt_x == 0 and tilt_y == 0:
		return field
	return apply_phase(field, linear_phase(field.shape, dx, dy, wavelength, tilt_x, tilt_y))


def axis_components(value) -> tuple:
	r"""Split a scalar-or-pair frame quantity into its (x, y) components.

	The scaled frame is isotropic in the common case — one ``s``, ``R``, ``τ``
	for both axes — but quadrupoles make it **anisotropic**:
	:math:`\psi = (s_x s_y)^{-1/2} U(x/s_x, y/s_y)\,
	e^{ik(x^2/2R_x + y^2/2R_y)}`. Frame quantities therefore travel as scalars
	(isotropic) or 2-sequences ``(x, y)``; this helper normalizes either form.

	Parameters
	----------
	value : float or Sequence[float]
		Scalar (applied to both axes) or ``(x, y)`` pair.

	Returns
	-------
	tuple
		``(vx, vy)`` floats.

	Related
	-------
	join_axes : The inverse (collapses equal components to a scalar).
	"""
	if np.ndim(value) == 0:
		return float(value), float(value)
	return float(value[0]), float(value[1])


def join_axes(vx: float, vy: float):
	r"""Join per-axis frame components, collapsing to a scalar when equal.

	Keeps the isotropic case exactly as before — scalar in, scalar out — so
	round-lens columns carry scalar ``s``/``R``/``τ`` bit-for-bit, while
	astigmatic states travel as ``(x, y)`` tuples.

	Parameters
	----------
	vx, vy : float
		Per-axis components.

	Returns
	-------
	float or tuple
		``vx`` when the components are equal (including both infinite),
		else ``(vx, vy)``.

	Related
	-------
	axis_components : The inverse.
	"""
	return vx if vx == vy else (vx, vy)


def scaled_delta_tau(dz: float, s0: float, R0: float) -> float:
	r"""Scaled propagation increment Δτ for a linear-s free segment.

	Implements handoff Eq 29 (and its ``R₀ = ∞`` special case, Eq 31):

	.. math::

		\Delta\tau = \frac{\Delta z}{s_0^2\left[1 + \Delta z / R_0\right]}
		\qquad\left(\Delta\tau = \Delta z / s_0^2 \text{ for } R_0 = \infty\right)

	which is the closed form of :math:`\int dz/s^2(z)` for
	:math:`s(z) = s_0[1 + (z - z_0)/R_0]` — verified against the numerical
	integral in the test suite before production use (handoff requirement).

	Parameters
	----------
	dz : float
		Physical segment length (metres).
	s0 : float
		Transverse scale at the segment start (dimensionless).
	R0 : float
		Reference radius of curvature at the segment start (metres);
		``numpy.inf`` for a flat reference wavefront.

	Returns
	-------
	float
		The scaled increment Δτ (metres).

	Raises
	------
	ValueError
		If the linear scaling crosses zero inside the segment
		(``1 + dz/R0 <= 0``) — the scaled frame is singular there
		(handoff Eq 52); stop before the crossover or switch frames.
	"""
	if np.isinf(R0):
		return dz / s0**2
	growth = 1.0 + dz / R0
	if growth <= 0:
		raise ValueError(f"Scaled frame crosses s=0 inside this segment (crossover at dz = {-R0} m "
						 f"of {dz} m); stop before the crossover or switch to a new frame.")
	return dz / (s0**2 * growth)


def rotate_field(U: np.ndarray, angle: float) -> np.ndarray:
	r"""Rotate a sampled complex field about the optical axis, band-limited-exactly.

	Magnetic round lenses rotate the beam (Larmor rotation): the ray path
	accumulates it on ``MicroscopeSection.R`` and a thick lens reports it as
	``Lens.rotation = -K L``. A scalar wave rotates the same way — the rotation
	is a coordinate rotation of the transverse plane, ``psi_out(r) =
	psi_in(R^-1 r)`` — so it can be applied to the sampled field directly.

	The rotation is realized as the exact three-shear decomposition

	.. math::

		R(\theta) = S_x(-\tan\tfrac{\theta}{2})\; S_y(\sin\theta)\;
					S_x(-\tan\tfrac{\theta}{2})

	with each shear a per-row (or per-column) subpixel **shift**, applied as a
	linear phase ramp in the conjugate direction. Every step is therefore
	unitary and interpolation-free for band-limited content: no spline blur, and
	the total intensity is preserved to round-off.

	Parameters
	----------
	U : np.ndarray
		Complex field ``(n, n)`` on a square grid with equal pitches.
	angle : float
		Rotation angle (radians), counter-clockwise in the array's ``(x, y)``
		convention. ``0`` returns a copy.

	Returns
	-------
	np.ndarray
		The rotated field, same shape.

	Raises
	------
	ValueError
		If ``U`` is not square (shears assume one common pitch on both axes).

	Related
	-------
	propagate_quadratic_segment_scaled : Applies this over a quadratic body's Larmor angle.

	Notes
	-----
	Rotation commutes exactly with both the free-space kernel (which depends
	only on ``|k|``) and the isotropic reference phase, so applying the whole
	angle once at a lens exit is equivalent to rotating continuously through the
	body — verified in the test suite. For a rotationally symmetric field it is
	analytically a no-op, so applying it there only adds resampling noise; that
	is why the wave paths leave it off by default.

	References
	----------
	Larkin, K. G. et al., "Fast Fourier method for the accurate rotation of
	sampled images," *Opt. Commun.* **139**, 99 (1997).
	"""
	if angle == 0:
		return U.astype(complex, copy=True)
	ny, nx = U.shape
	if ny != nx:
		raise ValueError(f"rotate_field needs a square grid; got {U.shape}. The shear "
						 "decomposition assumes one common pitch on both axes.")
	n = nx
	coords = np.arange(n) - n // 2					# centred pixel coordinates
	fx = np.fft.fftfreq(n)							# cycles per pixel
	t = np.tan(angle / 2.0)
	sn = -np.sin(angle)

	def shear_x(field, amount):
		"""Shift every row in x by ``amount * y`` pixels (phase ramp along x)."""
		ramp = np.exp(2j * np.pi * np.outer(coords * amount, fx))
		return np.fft.ifft(np.fft.fft(field, axis=1) * ramp, axis=1)

	def shear_y(field, amount):
		"""Shift every column in y by ``amount * x`` pixels (phase ramp along y)."""
		ramp = np.exp(2j * np.pi * np.outer(fx, coords * amount))
		return np.fft.ifft(np.fft.fft(field, axis=0) * ramp, axis=0)

	out = shear_x(U.astype(complex), t)
	out = shear_y(out, sn)
	return shear_x(out, t)


def segment_block(dz: float, kappa: float) -> tuple:
	r"""The ``(A, B)`` row of a constant-curvature segment's transfer block.

	A segment of signed curvature :math:`\kappa` obeys :math:`u'' + \kappa u = 0`,
	whose solution over ``dz`` is harmonic, linear or hyperbolic according to the
	sign of :math:`\kappa`. Only the first row is returned, because that is all
	the scaled frame needs: :math:`s(dz) = A s_0 + B u_0`.

	============  ==========================  ==================================
	:math:`\kappa`  regime                      :math:`(A, B)`
	============  ==========================  ==================================
	``> 0``       focusing (harmonic)         :math:`(\cos k\,dz,\ \sin(k\,dz)/k)`
	``= 0``       free space                  :math:`(1,\ dz)`
	``< 0``       defocusing (hyperbolic)     :math:`(\cosh k\,dz,\ \sinh(k\,dz)/k)`
	============  ==========================  ==================================

	with :math:`k = \sqrt{|\kappa|}` in both non-trivial rows. Every case has
	unit determinant, so :func:`scaled_delta_tau_quadratic`'s closed form applies
	uniformly.

	Parameters
	----------
	dz : float
		Distance through the segment (metres).
	kappa : float
		Signed curvature (1/metres²). Positive focuses, negative defocuses.

	Returns
	-------
	tuple of float
		``(A, B)``.

	Related
	-------
	scaled_delta_tau_quadratic : Uses ``B`` for the closed-form Δτ.
	segment_zero : Finds where ``s`` vanishes under the same law.

	Examples
	--------
	>>> A, B = segment_block(0.0, 900.0)
	>>> (round(A, 12), round(B, 12))
	(1.0, 0.0)
	"""
	if kappa == 0:
		return 1.0, float(dz)
	k = np.sqrt(abs(kappa))
	if kappa > 0:
		return float(np.cos(k * dz)), float(np.sin(k * dz) / k)
	return float(np.cosh(k * dz)), float(np.sinh(k * dz) / k)


def segment_zero(dz: float, s0: float, u0: float, kappa: float):
	r"""First zero of the reference scale strictly inside a segment, if any.

	The scaled frame is singular where :math:`s(z) = 0` — that is a beam
	crossover, and :math:`\tau = \int dz/s^2` diverges there. The closed-form Δτ
	cannot see an interior zero (it reads only the endpoints), so it is detected
	here from the law.

	A **focusing** segment oscillates and can cross zero repeatedly; a
	**defocusing** one, :math:`s = a\cosh kz + b\sinh kz`, crosses at most once
	and only when :math:`|b| > |a|` (an entering beam converging hard enough to
	reach the axis before the defocusing takes over); free space crosses at most
	once.

	Parameters
	----------
	dz : float
		Segment length (metres).
	s0 : float
		Reference scale at entry.
	u0 : float
		Reference slope at entry (``s0/R0``, or 0 on a flat frame).
	kappa : float
		Signed curvature (1/metres²).

	Returns
	-------
	float or None
		Position of the first interior zero (metres from entry), or ``None``.

	Related
	-------
	segment_block : The same law, as a transfer block.
	scaled_delta_tau_quadratic : Raises on what this finds.
	"""
	tol = 1e-12 * max(abs(dz), 1.0)
	inside = lambda z: (z is not None) and (tol < z < dz - tol)
	if kappa == 0:
		z = None if u0 == 0 else -s0 / u0
		return z if inside(z) else None
	k = np.sqrt(abs(kappa))
	if kappa > 0:
		# s = C cos(k z - phi) vanishes at z = (phi + pi/2 + n pi)/k
		phi = np.arctan2(u0 / k, s0)
		hits = sorted(z for n in range(-(int(k * abs(dz) / np.pi) + 2),
									   int(k * abs(dz) / np.pi) + 3)
					  if inside(z := (phi + np.pi / 2 + n * np.pi) / k))
		return hits[0] if hits else None
	# s = a cosh(kz) + b sinh(kz) vanishes iff |b| > |a|, at atanh(-a/b)/k
	a, b = s0, u0 / k
	if b == 0 or abs(a) >= abs(b):
		return None
	z = np.arctanh(-a / b) / k
	return float(z) if inside(z) else None


def scaled_delta_tau_quadratic(dz: float, s0: float, R0: float, kappa: float) -> float:
	r"""Scaled increment Δτ across a constant-curvature segment.

	An element with a finite body and finite strength is not a phase screen but
	a **medium**: a thick round lens (focusing on both axes) or a thick
	quadrupole (focusing on one, defocusing on the other). Inside it the
	reference scale obeys the same equation as a ray in that medium,
	:math:`s'' + \kappa s = 0`, so :math:`\tau = \int dz/s^2` is closed form —
	and, remarkably, in a form that does not depend on which regime the segment
	is in:

	.. math::

		\Delta\tau = \frac{B}{s_0\, s(\Delta z)}, \qquad s(\Delta z) = A s_0 + B u_0

	with :math:`(A, B)` the segment's own transfer row (:func:`segment_block`)
	and :math:`u_0 = s_0/R_0`. This is a consequence of :math:`\det M = 1`: for
	any solution :math:`s_1`, the second independent solution is
	:math:`s_1\int dz/s_1^2`, and unit Wronskian fixes the constant. Free space
	(:math:`B = \Delta z`) reproduces :func:`scaled_delta_tau` exactly, so the
	harmonic, hyperbolic and linear cases are one formula rather than three.

	Parameters
	----------
	dz : float
		Distance travelled inside the medium (metres).
	s0 : float
		Transverse scale at the segment start.
	R0 : float
		Reference radius of curvature at the segment start (metres);
		``numpy.inf`` for a flat reference wavefront.
	kappa : float
		Signed curvature of the medium (1/metres²): positive focuses (harmonic),
		negative defocuses (hyperbolic), zero is free space.

	Returns
	-------
	float
		The scaled increment Δτ (metres).

	Raises
	------
	ValueError
		If the reference scale passes through zero inside the segment — the
		crossover lies *within the element body*, where this frame is singular
		(the integral diverges). Switch frames before entering, or shorten the
		step; mid-element frame switching is not implemented.

	Related
	-------
	scaled_delta_tau : The free-space form, which this generalizes.
	segment_block, segment_zero : The law, and its singularity.
	propagate_quadratic_segment_scaled : Consumes this, per axis.

	Notes
	-----
	The divergence is the ordinary crossover singularity of a converging frame,
	not a new pathology: it is the same ``s → 0`` the hybrid policy flattens
	through in free space. A purely defocusing segment entered flat or diverging
	can never reach it.

	Examples
	--------
	>>> round(scaled_delta_tau_quadratic(0.05, 1.0, np.inf, 0.0), 12)   # free
	0.05
	"""
	u0 = 0.0 if np.isinf(R0) else s0 / R0
	z_zero = segment_zero(dz, s0, u0, kappa)
	if z_zero is not None:
		raise ValueError(f"Scaled frame reaches s = 0 inside the segment body (at "
						 f"{z_zero:.6g} m of {dz:.6g} m): the crossover lies within "
						 "the element, where this frame is singular. Switch frames "
						 "before the element, or stop the step there (mid-element "
						 "frame switching is not implemented).")
	A, B = segment_block(dz, kappa)
	s_end = A * s0 + B * u0
	if s_end == 0:
		raise ValueError("Scaled frame reaches s = 0 exactly at the segment exit; "
						 "stop the step short of the crossover.")
	return float(B / (s0 * s_end))


def propagate_quadratic_segment_scaled(U: np.ndarray, dxi: float, deta: float,
									   wavelength: float, dz: float, s, R, kappa,
									   s_min: float = 1e-3, absorb: float = 0.0,
									   rotate: float = 0.0) -> tuple:
	r"""Propagate the scaled field through a constant-curvature segment.

	The honest treatment of any element that declares itself a segment
	(:meth:`elements.Element._scaled_segment`): rather than a thin kick placed
	between two half-length drifts, the element is one **medium**, and the frame
	follows its scale law exactly. The frame advances by the element's own
	transfer row (:func:`segment_block`) applied to :math:`(s, s/R)` —
	legitimate because the frame *is* a reference ray — and the reduced field
	``U`` propagates over the segment's own Δτ
	(:func:`scaled_delta_tau_quadratic`) with the same carrier-free kernel used
	for free space. **No phase screen and no curvature kick are applied**: the
	scaled factorization solves the paraxial equation in a constant-curvature
	medium exactly, so a thick body costs ``U`` nothing in sampling, exactly
	like a drift.

	``kappa`` is **per-axis**, so this covers a thick round lens (equal positive
	curvature on both axes) and a thick **quadrupole** (:math:`+\kappa` on one
	axis, :math:`-\kappa` on the other) with the same code: the two axes simply
	accumulate different Δτ, and the paraxial kernel is separable, so an
	anisotropic :math:`(\Delta\tau_x, \Delta\tau_y)` is applied in one transform
	pair at no extra cost.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(ny, nx)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	wavelength : float
		Wavelength (metres).
	dz : float
		Length of body traversed (metres). ``0`` returns the state unchanged.
	s, R : float or Sequence[float]
		Frame state at the entrance (``R = numpy.inf`` = flat), scalar or an
		``(x, y)`` pair.
	kappa : float or Sequence[float]
		Signed curvature of the medium (1/metres²), scalar or per-axis: positive
		focuses (harmonic), negative defocuses (hyperbolic), zero degenerates to
		a drift.
	s_min : float, optional
		Crossover backstop on the exit scale, by default ``1e-3``.
	absorb : float, optional
		Absorbing-boundary margin forwarded to the sub-stepped propagation (see
		:func:`propagate_free_scaled`), by default 0.
	rotate : float, optional
		**Larmor rotation angle** of the body in radians, applied to the field
		with :func:`rotate_field`; 0 (default) applies none. The element declares
		this — a round lens rotates by ``−K·L``, a quadrupole not at all — so
		this function never derives it from the strength. It is analytically a
		no-op for a rotationally symmetric field, where it would only add
		resampling noise.

	Returns
	-------
	tuple
		``(U_out, s_out, R_out, dtau)``; the last three are scalars when the
		frame and curvature are isotropic and ``(x, y)`` pairs otherwise.

	Raises
	------
	ValueError
		If the frame reaches ``s = 0`` inside the body on either axis (the
		message names which), or an exit scale violates ``|s_out| > s_min``.
	NotImplementedError
		If a rotation is requested on an anisotropic segment or frame.

	Related
	-------
	propagate_free_scaled : The ``kappa = 0`` counterpart.
	scaled_delta_tau_quadratic, segment_block : The per-axis law.
	apply_thin_lens_scaled : What a *thin* (``length == 0``) element does instead.

	Notes
	-----
	Anisotropic rotation is refused: a Larmor rotation mixes the transverse
	axes, so it is only meaningful where those axes are equivalent. No element
	declares both at once today — a quadrupole has no axial field, hence no
	Larmor rotation.

	Examples
	--------
	>>> U, s, R, dtau = propagate_quadratic_segment_scaled(
	...     U0, 1e-7, 1e-7, 2.5e-12, 0.02, 1.0, np.inf, 34.7**2)   # doctest: +SKIP
	"""
	if dz == 0:
		return U.astype(complex, copy=True), s, R, 0.0
	s_x, s_y = axis_components(s)
	R_x, R_y = axis_components(R)
	k_x, k_y = axis_components(kappa)
	if k_x == 0 and k_y == 0:
		return propagate_free_scaled(U, dxi, deta, wavelength, dz, s, R,
									 s_min=s_min, absorb=absorb)
	if rotate and (k_x != k_y or s_x != s_y or R_x != R_y):
		raise NotImplementedError(
			"Larmor rotation mixes the transverse axes, so it is only defined on "
			f"an isotropic segment and frame (got kappa = {kappa}, s = {s}, "
			f"R = {R}). Drop the rotation, or reconcile the axes first.")

	out = []
	for axis, s_a, R_a, k_a in (('x', s_x, R_x, k_x), ('y', s_y, R_y, k_y)):
		try:
			dtau_a = scaled_delta_tau_quadratic(dz, s_a, R_a, k_a)
		except ValueError as exc:				# name the axis: a quadrupole can
			raise ValueError(f"[{axis} axis] {exc}") from exc	# fail on one only
		A, B = segment_block(dz, k_a)
		u_a = 0.0 if np.isinf(R_a) else s_a / R_a
		# the frame IS a reference ray, so it advances by the segment's own block
		s_out = A * s_a + B * u_a
		u_out = _segment_slope(dz, s_a, u_a, k_a)
		if abs(s_out) <= s_min:
			raise ValueError(f"Scaled frame reaches |s| = {abs(s_out):.3e} <= s_min = "
							 f"{s_min} at the {axis}-axis segment exit; switch frames "
							 "before the element, or lower s_min knowingly.")
		out.append((s_out, np.inf if u_out == 0 else s_out / u_out, dtau_a))
	(sx_out, Rx_out, dtau_x), (sy_out, Ry_out, dtau_y) = out

	# U evolves purely by the segment's own dtau, per axis (no screen, no kick)
	dtau = join_axes(dtau_x, dtau_y)
	if absorb and absorb > 0:
		n = U.shape[0]
		band = absorb * n * abs(dxi)
		dtau_step = 2 * band * abs(dxi) / wavelength
		n_steps = max(1, int(np.ceil(max(abs(dtau_x), abs(dtau_y)) / dtau_step)))
		W = boundary_window(U.shape, margin=absorb)
		U_out = U.astype(complex, copy=True) * W
		sub = join_axes(dtau_x / n_steps, dtau_y / n_steps)
		for _ in range(n_steps):
			U_out = angular_spectrum_propagate(U_out, dxi, deta, wavelength, sub,
											   include_carrier=False)
			U_out = U_out * W
	else:
		U_out = angular_spectrum_propagate(U, dxi, deta, wavelength, dtau,
										   include_carrier=False)
	if rotate:
		# commutes with the isotropic propagation above, so applying the whole
		# angle once here is exact
		U_out = rotate_field(U_out, rotate)
	return U_out, join_axes(sx_out, sy_out), join_axes(Rx_out, Ry_out), dtau


def _segment_slope(dz: float, s0: float, u0: float, kappa: float) -> float:
	r"""Reference slope at the exit of a constant-curvature segment.

	The second row of the segment's transfer block applied to
	:math:`(s_0, u_0)` — the companion of :func:`segment_block`, split out
	because only the frame advance needs it.

	Parameters
	----------
	dz : float
		Distance through the segment (metres).
	s0 : float
		Reference scale at entry.
	u0 : float
		Reference slope at entry.
	kappa : float
		Signed curvature (1/metres²).

	Returns
	-------
	float
		Reference slope at ``dz``.

	Related
	-------
	segment_block : The first row, and the regime table.
	"""
	if kappa == 0:
		return float(u0)
	k = np.sqrt(abs(kappa))
	if kappa > 0:
		return float(-k * np.sin(k * dz) * s0 + np.cos(k * dz) * u0)
	return float(k * np.sinh(k * dz) * s0 + np.cosh(k * dz) * u0)


def beam_support_radius(U: np.ndarray, dxi: float, deta: float,
						threshold: float = 1e-6) -> float:
	r"""Per-axis half-width of the beam's support on the ξ grid.

	The largest per-axis scaled coordinate (``max(|ξ|, |η|)``) at which ``|U|``
	still exceeds ``threshold`` times its maximum. The per-pixel step of a
	quadratic reference phase along an axis is set by that axis's coordinate,
	so this half-width — not the corner radius — is what the sampling criteria
	need. Frame-change phases applied where the field is essentially zero are
	harmless, so measuring at the support instead of the (possibly much larger,
	empty) grid lets frames switch at the earliest plane representable for the
	*actual beam*; a beam filling the grid reproduces the grid-edge criterion
	exactly.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(ny, nx)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	threshold : float, optional
		Amplitude fraction defining the support, by default ``1e-6``.

	Returns
	-------
	float
		Support half-width in ξ units (at least one pixel).

	Related
	-------
	min_representable_curvature : Consumes this as its ``x_max``.
	"""
	ext_x, ext_y = beam_support_extents(U, dxi, deta, threshold=threshold)
	return max(ext_x, ext_y)


def beam_support_extents(U: np.ndarray, dxi: float, deta: float,
						 threshold: float = 1e-6) -> tuple:
	r"""Per-axis half-widths of the beam's support on the ξ/η grid.

	The largest ``|ξ|`` and largest ``|η|`` at which ``|U|`` still exceeds
	``threshold`` times its maximum, returned separately per axis. Anisotropic
	frames need the axes individually — each axis's reference phase is sampled
	against that axis's own support — while :func:`beam_support_radius` keeps
	the combined (max) form for isotropic criteria.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(ny, nx)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	threshold : float, optional
		Amplitude fraction defining the support, by default ``1e-6``.

	Returns
	-------
	tuple
		``(ext_x, ext_y)`` support half-widths in ξ/η units (each at least one
		pixel of its own axis).

	Related
	-------
	beam_support_radius : The combined (max over axes) form.
	"""
	amp = np.abs(U)
	mask = amp > threshold * amp.max()
	X, Y = transverse_coordinates(U.shape, abs(dxi), abs(deta))
	ext_x = max(float(np.abs(X[mask]).max()), abs(dxi))
	ext_y = max(float(np.abs(Y[mask]).max()), abs(deta))
	return ext_x, ext_y


def min_representable_curvature(n: int, dxi: float, wavelength: float, s: float,
								safety: float = 0.5, x_max: float = None) -> float:
	r"""Smallest reference-curvature radius a scaled frame can absorb or release.

	A frame change moves the quadratic reference phase
	:math:`k\,x^2/2 \cdot (1/R_o - 1/R_n)` into or out of the sampled field U.
	That phase is representable only while its per-pixel step stays below
	``safety * pi`` at the outermost point that matters, which bounds the
	curvature radius:

	.. math::

		|R|_{\min} = \frac{k\, x_{\max}\, \Delta x}{\text{safety}\,\pi}
		= \frac{k\, s^2\, \xi_{\max}\, \Delta\xi}{\text{safety}\,\pi}

	with physical pixel :math:`\Delta x = |s|\Delta\xi` and
	:math:`x_{\max} = |s|\,\xi_{\max}`. By default :math:`\xi_{\max}` is the
	grid half-width ``(n/2)·Δξ``; pass the beam-support radius
	(:func:`beam_support_radius`) to measure at the beam instead — phase
	applied to empty grid is harmless, and the beam-based bound lets frames
	switch earlier (larger s, larger pixels). Used by
	:func:`change_scaled_frame` as its sampling guard and by the hybrid
	crossover policy to place the flatten/re-diverge planes.

	Parameters
	----------
	n : int
		Samples per side of the (square) grid.
	dxi : float
		Scaled-coordinate sample spacing Δξ (metres).
	wavelength : float
		Wavelength (metres).
	s : float
		Transverse scale of the frame at the plane in question.
	safety : float, optional
		Fraction of the π-per-pixel Nyquist step reserved for the reference
		phase (the rest is headroom for U's own spectrum), by default 0.5.
	x_max : float, optional
		ξ radius at which to evaluate the criterion (e.g. the beam-support
		radius); ``None`` (default) uses the grid half-width ``(n//2)·Δξ``.

	Returns
	-------
	float
		The minimum representable curvature radius ``|R|_min`` (metres).

	Related
	-------
	beam_support_radius : The beam-based ``x_max``.
	change_scaled_frame : Enforces this bound.
	propagate_free_scaled_hybrid : Uses it to place frame switches.
	"""
	k = 2 * np.pi / wavelength
	xi_max = (n // 2) * abs(dxi) if x_max is None else abs(x_max)
	return k * s**2 * xi_max * abs(dxi) / (safety * np.pi)


def change_scaled_frame(U: np.ndarray, dxi: float, deta: float, wavelength: float,
						s_old, R_old, R_new,
						s_new=None, safety: float = 0.5) -> tuple:
	r"""Re-express the same physical wave in a different scaled frame (Eq 5).

	A *frame* is a choice of factorization
	:math:`\psi = (s_x s_y)^{-1/2}\,U(x/s_x, y/s_y)\,
	e^{ik(x^2/2R_x + y^2/2R_y)}` (isotropic: one ``s``, one ``R``); this
	primitive transforms :math:`(s_o, R_o, U_o) \to (s_n, R_n, U_n)` while
	keeping ψ identical:

	.. math::

		U_n = \sqrt{\frac{s_{nx} s_{ny}}{s_{ox} s_{oy}}}\, U_o \,
		\exp\!\left[\frac{ik}{2}\left(
		x^2\!\left(\frac{1}{R_{ox}} - \frac{1}{R_{nx}}\right) +
		y^2\!\left(\frac{1}{R_{oy}} - \frac{1}{R_{ny}}\right)\right)\right]

	using the **physical-grid-continuous convention**: the new pitches are
	:math:`\Delta\xi_n = \Delta\xi_o\, s_{ox}/s_{nx}` (and likewise per axis),
	so the samples sit at the same physical points and the operation is
	pointwise — no interpolation. The physical representation is the special
	frame ``(s=1, R=inf)``, making :func:`factor_wave` and
	:func:`reconstruct_physical_wave` special cases.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(n, n)`` in the old frame.
	dxi, deta : float
		Old-frame sample spacings Δξ/Δη.
	wavelength : float
		Wavelength (metres).
	s_old : float or Sequence[float]
		Old-frame transverse scale (nonzero); ``(s_x, s_y)`` pair when
		anisotropic.
	R_old : float or Sequence[float]
		Old-frame reference curvature radius (metres); ``numpy.inf`` = flat.
	R_new : float or Sequence[float]
		New-frame reference curvature radius (metres); ``numpy.inf`` = flat.
	s_new : float or Sequence[float], optional
		New-frame scale (nonzero); ``None`` (default) keeps ``s_old`` (pitch
		unchanged — the flatten/re-diverge cases).
	safety : float, optional
		Sampling-guard fraction passed to the per-pixel phase-step check, by
		default 0.5. ``None`` disables the guard (pure-converter use).

	Returns
	-------
	tuple
		``(U_new, dxi_new, deta_new)``.

	Raises
	------
	ValueError
		If the reference-phase difference is not representable on this grid —
		the error names the minimum representable ``|R|`` — or if ``s_new``
		is zero on either axis.

	Related
	-------
	min_representable_curvature : The guard's threshold.
	factor_wave, reconstruct_physical_wave : The (1, inf) special cases.
	axis_components, join_axes : The scalar-or-pair convention.

	Notes
	-----
	The singularity of a converging frame at its crossover (``s -> 0``)
	belongs to the frame, not to the wave; this operation is how propagation
	steps onto a better frame before that happens (the hybrid policy in
	:func:`propagate_free_scaled_hybrid`). On an isotropic change the
	amplitude ratio is the signed ``s_n/s_o`` of Eq 5 (a sign flip's implied
	ξ-grid inversion is exact for the even quadratic phase and is the caller's
	relabeling to track); anisotropic changes use the principal square root of
	the scale-product ratio.
	"""
	s_ox, s_oy = axis_components(s_old)
	if s_new is None:
		s_nx, s_ny = s_ox, s_oy
	else:
		s_nx, s_ny = axis_components(s_new)
	if s_nx == 0 or s_ny == 0:
		raise ValueError("change_scaled_frame requires a nonzero s_new on both axes; the "
						 "frame is singular at s = 0 (switch frames before the crossover).")
	R_ox, R_oy = axis_components(R_old)
	R_nx, R_ny = axis_components(R_new)
	c_ox = 0.0 if np.isinf(R_ox) else 1.0 / R_ox
	c_oy = 0.0 if np.isinf(R_oy) else 1.0 / R_oy
	c_nx = 0.0 if np.isinf(R_nx) else 1.0 / R_nx
	c_ny = 0.0 if np.isinf(R_ny) else 1.0 / R_ny
	if s_ox == s_oy and s_nx == s_ny:
		amp = s_nx / s_ox			# signed Eq 5 ratio (isotropic, bit-for-bit)
	else:
		amp = np.sqrt(complex((s_nx * s_ny) / (s_ox * s_oy)))
		amp = amp.real if amp.imag == 0 else amp
	U_out = amp * U.astype(complex)
	if c_ox != c_nx or c_oy != c_ny:
		# physical coordinates of the (shared) sample points, per-axis pitch
		X, Y = transverse_coordinates(U.shape, abs(s_ox) * dxi, abs(s_oy) * deta)
		k = 2 * np.pi / wavelength
		chi = k / 2 * (X**2 * (c_ox - c_nx) + Y**2 * (c_oy - c_ny))
		if safety is not None:
			# measure the per-pixel step over the beam's support only — phase
			# applied where the field is essentially zero is harmless
			ext_x, ext_y = beam_support_extents(U, dxi, deta)
			step_x = k * abs(s_ox)**2 * abs(dxi) * ext_x * abs(c_ox - c_nx)
			step_y = k * abs(s_oy)**2 * abs(deta) * ext_y * abs(c_oy - c_ny)
			step = max(step_x, step_y)
			if step > safety * np.pi:
				axis, s_g, d_g, e_g = (("x", s_ox, dxi, ext_x) if step_x >= step_y
									   else ("y", s_oy, deta, ext_y))
				R_min = min_representable_curvature(U.shape[0], d_g, wavelength, s_g,
													safety, x_max=e_g)
				raise ValueError(f"Frame change from R={R_old} m to R={R_new} m is not representable "
								 f"on this grid (per-pixel phase step {step:.2f} rad at the beam "
								 f"support > {safety:.2g}*pi on the {axis} axis): the curvature "
								 f"moved into U must satisfy |R| >= {R_min:.3e} m. Switch frames "
								 "closer to the plane where the reference curvature is weaker, or "
								 "refine the grid.")
		U_out = U_out * np.exp(1j * chi)
	return U_out, dxi * s_ox / s_nx, deta * s_oy / s_ny


def factor_wave(psi: np.ndarray, dx: float, dy: float, wavelength: float,
				s, R) -> tuple:
	r"""Factor an ordinary paraxial wave into the scaled representation (handoff Eq 55).

	Removes the reference quadratic phase and the coordinate scaling:

	.. math::

		U(\xi,\eta) = s\,\psi(s\xi, s\eta)\,
		\exp\!\left[-\frac{ik\,(x^2+y^2)}{2R}\right],\qquad x = s\xi

	evaluated on matching grids (the returned ξ grid holds the same samples with
	pitch ``Δξ = dx/s``). The factorization is exact and discards no information.

	Parameters
	----------
	psi : np.ndarray
		Ordinary paraxial wave ``(ny, nx)`` on a grid of pitch ``dx``/``dy``.
	dx, dy : float
		Physical sample spacings (metres).
	wavelength : float
		Wavelength (metres).
	s : float or Sequence[float]
		Chosen transverse scale (nonzero); ``(s_x, s_y)`` pair for an
		anisotropic frame.
	R : float or Sequence[float]
		Chosen reference radius of curvature (metres); ``numpy.inf`` for none;
		per-axis pair for an anisotropic frame.

	Returns
	-------
	tuple
		``(U, dxi, deta)`` — the scaled field and its ξ/η sample spacings.

	Related
	-------
	reconstruct_physical_wave : Exact inverse (handoff Eq 37).
	change_scaled_frame : The general frame change this is a special case of.

	Notes
	-----
	Delegates to :func:`change_scaled_frame` as the transform from the
	physical frame ``(1, inf)`` to ``(s, R)``, guard-free (a pure converter).
	"""
	s_x, s_y = axis_components(s)
	U, _, _ = change_scaled_frame(psi, dx, dy, wavelength, s_old=1.0, R_old=np.inf,
								  R_new=R, s_new=s, safety=None)
	return U, dx / s_x, dy / s_y


def fourier_resample(field: np.ndarray, d_in: float, n_out: int, d_out: float) -> np.ndarray:
	r"""Band-limited (Fourier-domain) resampling of a 2D field.

	Exact evaluation of the field's trigonometric (sinc) interpolant on the
	requested output grid, preferred by the handoff over low-order real-space
	interpolation. The interpolant is separable, so the evaluation is two small
	matrix products (a nonuniform inverse DFT per axis) — exact at *any* pitch
	ratio, not just commensurate grids. Values (not sums) are preserved; grids
	are centred per :func:`transverse_coordinates` (sample ``n//2`` at 0).

	Parameters
	----------
	field : np.ndarray
		Complex field ``(n, n)`` (square), pitch ``d_in``.
	d_in : float
		Input sample spacing.
	n_out : int
		Output samples per side.
	d_out : float
		Output sample spacing (delivered exactly).

	Returns
	-------
	np.ndarray
		Resampled complex field ``(n_out, n_out)``.
	"""
	n = field.shape[0]
	# spectrum referenced to the centre sample (index n//2 -> 0)
	F = np.fft.fft2(np.fft.ifftshift(field))
	freqs = np.fft.fftfreq(n, d_in)
	x_out = (np.arange(n_out) - n_out // 2) * d_out
	# nonuniform inverse-DFT matrix per axis: field(x) = (1/n^2) sum F e^{2*pi*i f x}
	A = np.exp(2j * np.pi * np.outer(x_out, freqs))
	return (A @ F @ A.T) / n**2


def reconstruct_physical_wave(U: np.ndarray, dxi: float, deta: float,
							  wavelength: float, s, R,
							  target_dx: float | None = None,
							  target_shape: tuple | None = None) -> tuple:
	r"""Reconstruct the ordinary paraxial wave from the scaled representation.

	Handoff Eq 37 (boxed):

	.. math::

		\psi(x,y) = \frac{1}{s}\,U\!\left(\frac{x}{s}, \frac{y}{s}\right)
		\exp\!\left[\frac{ik\,(x^2+y^2)}{2R}\right]

	With no target grid (Eq 41) the scaled samples are reinterpreted in place:
	the physical pitch is ``Δx = |s|·Δξ`` and no interpolation occurs. With a
	prescribed grid (Eq 44) U is band-limited-resampled in ξ to evaluate at
	``ξ_p = x_p/s`` before the amplitude and quadratic phase are restored. The
	longitudinal carrier ``e^{ikz}`` is NOT applied (handoff Eq 37 note).

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(ny, nx)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	wavelength : float
		Wavelength (metres).
	s : float or Sequence[float]
		Transverse scale at this plane (nonzero); ``(s_x, s_y)`` pair when
		anisotropic (native pixels ``dx = |s_x|·Δξ``, ``dy = |s_y|·Δη``).
	R : float or Sequence[float]
		Reference radius of curvature (metres); ``numpy.inf`` for none;
		per-axis pair when anisotropic.
	target_dx : float, optional
		Prescribed physical pixel size; ``None`` (default) reconstructs on the
		native grid ``Δx = |s|·Δξ``.
	target_shape : tuple, optional
		Prescribed output shape ``(ny, nx)``; required with ``target_dx``.

	Returns
	-------
	tuple
		``(psi, dx, dy)`` — the physical wave and its pixel sizes.

	Raises
	------
	NotImplementedError
		If a ``target_dx`` grid is requested on an anisotropic frame — the
		band-limited resampler is square-grid only; reconstruct on the native
		(rectangular-pixel) grid instead.

	Related
	-------
	factor_wave : Exact inverse (handoff Eq 55).
	fourier_resample : The band-limited resampler used for target grids.
	"""
	s_x, s_y = axis_components(s)
	R_x, R_y = axis_components(R)
	if target_dx is not None:
		if s_x != s_y:
			raise NotImplementedError("target_dx resampling is square-grid only; an anisotropic "
									  "frame (s_x != s_y) reconstructs on its native rectangular "
									  "pixels (dx = |s_x| dxi, dy = |s_y| deta). Omit target_dx.")
		n_out = target_shape[0]
		U = fourier_resample(U, dxi, n_out, target_dx / abs(s_x))
		dxi = deta = target_dx / abs(s_x)
	dx = abs(s_x) * dxi
	dy = abs(s_y) * deta
	if s_x == s_y:
		psi = U.astype(complex) / s_x		# signed Eq 37 amplitude (isotropic)
	else:
		amp = np.sqrt(complex(s_x * s_y))
		psi = U.astype(complex) / (amp.real if amp.imag == 0 else amp)
	if not (np.isinf(R_x) and np.isinf(R_y)):
		X, Y = transverse_coordinates(psi.shape, dx, dy)
		k = 2 * np.pi / wavelength
		c_x = 0.0 if np.isinf(R_x) else 1.0 / R_x
		c_y = 0.0 if np.isinf(R_y) else 1.0 / R_y
		psi = psi * np.exp(1j * k * (X**2 * c_x + Y**2 * c_y) / 2)
	return psi, dx, dy


def apply_thin_lens_scaled(s, R, power) -> tuple:
	r"""Absorb a thin-lens focusing power into the scaled curvature state.

	Handoff Eqs 45–46: :math:`1/R^+ = 1/R^- - 1/f` with ``s`` continuous
	through the lens and ``s' = s/R`` re-derived from the new curvature. The
	scaled field U is untouched (Eq 15) — only the reference state changes.
	All arguments accept per-axis ``(x, y)`` pairs: a quadrupole absorbs its
	``(P, -P)`` powers into ``(R_x, R_y)`` exactly like a round lens absorbs
	one power into one curvature, making the frame anisotropic.

	Parameters
	----------
	s : float or Sequence[float]
		Transverse scale at the lens plane (unchanged, returned for symmetry).
	R : float or Sequence[float]
		Incoming reference radius of curvature (metres); ``numpy.inf`` for flat.
	power : float or Sequence[float]
		Focusing power ``1/f`` to absorb (1/metres); per-axis pair for
		astigmatic elements.

	Returns
	-------
	tuple
		``(s, R_out)`` — the (unchanged) scale and the updated curvature
		(``numpy.inf`` when the outgoing wavefront is flat). Scalars when the
		axes agree, ``(x, y)`` pairs otherwise.

	Related
	-------
	axis_components, join_axes : The scalar-or-pair convention.
	"""
	def one_axis(R_a, P_a):
		curvature = (0.0 if np.isinf(R_a) else 1.0 / R_a) - P_a
		return np.inf if curvature == 0 else 1.0 / curvature
	R_x, R_y = axis_components(R)
	P_x, P_y = axis_components(power)
	return s, join_axes(one_axis(R_x, P_x), one_axis(R_y, P_y))


def boundary_window(shape: tuple, margin: float = 0.1) -> np.ndarray:
	r"""Absorbing-boundary window: 1 in the interior, radial cosine → 0 at the edge.

	The FFT propagator is periodic: field that diffracts out of the modeled
	field of view re-enters coherently from the opposite side and interferes
	with the beam. Physically those electrons leave the beam and never
	return, so the boundary should absorb them — and it must do so
	**azimuthally isotropically**: a separable (square) window puts its
	corners :math:`\sqrt{2}` farther out than its edges, so it clips the
	beam's diffraction halo anisotropically at every step and the surviving
	halo interferes back into the beam as a fourfold, pixel-axis-aligned
	fringe pattern. This window is therefore radially symmetric: 1 inside
	the inscribed circle minus the band, raised-cosine to 0 at the
	inscribed-circle edge (corners beyond it are fully absorbed).

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	margin : float, optional
		Fraction of the shorter axis occupied by the absorbing band,
		by default 0.1.

	Returns
	-------
	np.ndarray
		Real window, shape ``(ny, nx)``, values in [0, 1]; a function of
		radius only.

	Related
	-------
	propagate_free_scaled : Applies it between τ sub-steps when ``absorb > 0``.
	"""
	ny, nx = shape
	x = np.arange(nx) - nx // 2
	y = np.arange(ny) - ny // 2
	X, Y = np.meshgrid(x, y)
	r = np.sqrt(X**2 + Y**2)
	edge = min(nx, ny) // 2						# inscribed-circle radius
	m = max(1, int(round(margin * min(nx, ny))))
	t = np.clip((r - (edge - m)) / m, 0.0, 1.0)	# 0 interior -> 1 at the edge
	return 0.5 * (1 + np.cos(np.pi * t))


def propagate_free_scaled(U: np.ndarray, dxi: float, deta: float, wavelength: float,
						  dz: float, s, R, s_min: float = 1e-3,
						  absorb: float = 0.0) -> tuple:
	r"""Propagate the scaled field U through one free segment of length ``dz``.

	Handoff Eqs 23–33: the scale evolves linearly, ``s(z) = s₀[1 + Δz/R₀]``
	(constant for ``R₀ = ∞``), the curvature as ``R(z) = R₀ + Δz``, and U
	Fresnel-propagates over the scaled increment Δτ (:func:`scaled_delta_tau`)
	with the carrier-free angular-spectrum kernel (Eq 32).

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(ny, nx)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	wavelength : float
		Wavelength (metres).
	dz : float
		Physical segment length (metres). ``dz == 0`` returns the state unchanged.
	s : float or Sequence[float]
		Transverse scale at the segment start; ``(s_x, s_y)`` pair when
		anisotropic.
	R : float or Sequence[float]
		Reference radius of curvature at the segment start (``numpy.inf`` =
		flat); per-axis pair when anisotropic.
	s_min : float, optional
		Crossover guard: the segment must keep ``|s| > s_min`` on both axes
		(handoff Eq 52), by default ``1e-3``.
	absorb : float, optional
		Absorbing-boundary margin fraction (:func:`boundary_window`), by
		default 0 (pure periodic propagation). When > 0 the segment is
		sub-stepped in τ so that no spectral component can traverse the
		absorbing band within one FFT step, and the window is applied between
		steps — field diffracting out of the modeled field of view is removed
		(physically: those electrons leave the beam) instead of wrapping
		around and interfering.

	Returns
	-------
	tuple
		``(U_out, s_out, R_out, dtau)``. Frame outputs (and ``dtau``) are
		scalars when the axes agree, ``(x, y)`` pairs otherwise. With
		``absorb > 0`` the total ``|U|²`` decreases by the power lost through
		the boundary.

	Raises
	------
	ValueError
		If the frame crosses ``s = 0`` inside the segment, or the exit scale
		violates ``|s_out| > s_min`` on either axis — with the crossover
		position named, since the singularity belongs to the frame, not the
		physical wave.
	"""
	if dz == 0:
		return U.astype(complex, copy=True), s, R, 0.0
	s_x, s_y = axis_components(s)
	R_x, R_y = axis_components(R)
	# also guards the in-segment zero crossing (per axis)
	dtau_x = scaled_delta_tau(dz, s_x, R_x)
	dtau_y = scaled_delta_tau(dz, s_y, R_y)
	s_out_x = s_x if np.isinf(R_x) else s_x * (1.0 + dz / R_x)
	s_out_y = s_y if np.isinf(R_y) else s_y * (1.0 + dz / R_y)
	if abs(s_out_x) <= s_min or abs(s_out_y) <= s_min:
		axis, s_bad, R_bad = (("x", s_out_x, R_x) if abs(s_out_x) <= abs(s_out_y)
							  else ("y", s_out_y, R_y))
		z_cross = -R_bad if not np.isinf(R_bad) else np.inf
		raise ValueError(f"Scaled frame reaches |s| = {abs(s_bad):.3e} <= s_min = {s_min} on the "
						 f"{axis} axis at the segment end (frame crossover at dz = {z_cross} m); "
						 "stop before the crossover, switch frames (hybrid mode), or lower "
						 "s_min knowingly.")
	R_out_x = R_x + dz if not np.isinf(R_x) else np.inf
	R_out_y = R_y + dz if not np.isinf(R_y) else np.inf
	s_out = join_axes(s_out_x, s_out_y)
	R_out = join_axes(R_out_x, R_out_y)
	dtau = join_axes(dtau_x, dtau_y)
	if absorb and absorb > 0:
		# absorbing boundary: sub-step in tau so no spectral component can
		# traverse the absorbing band unattenuated within one FFT step
		# (max transverse travel per step = lambda * f_Nyquist * dtau_step)
		n = U.shape[0]
		band = absorb * n * abs(dxi)					# absorber width in xi
		dtau_step = 2 * band * abs(dxi) / wavelength	# travel at Nyquist = band
		n_steps = max(1, int(np.ceil(max(abs(dtau_x), abs(dtau_y)) / dtau_step)))
		W = boundary_window(U.shape, margin=absorb)
		U_out = U.astype(complex, copy=True) * W
		sub = join_axes(dtau_x / n_steps, dtau_y / n_steps)
		for _ in range(n_steps):
			U_out = angular_spectrum_propagate(U_out, dxi, deta, wavelength,
											   sub, include_carrier=False)
			U_out = U_out * W
		return U_out, s_out, R_out, dtau
	U_out = angular_spectrum_propagate(U, dxi, deta, wavelength, dtau, include_carrier=False)
	return U_out, s_out, R_out, dtau


def propagate_free_scaled_hybrid(U: np.ndarray, dxi: float, deta: float,
								 wavelength: float, dz: float, s, R,
								 z: float, z_cross=None,
								 safety: float = 0.5, s_min: float = 1e-3,
								 absorb: float = 0.0,
								 crossover: str = 'flat') -> tuple:
	r"""Propagate one free segment with automatic frame switching at crossovers.

	The hybrid crossover policy: far from a focus the wave rides its scaled
	frame; when a converging frame's reference curvature becomes representable
	on the (shrinking) grid, the frame is **flattened**
	(:func:`change_scaled_frame` to ``R = inf`` with s kept — a pointwise
	operation) and the wave crosses the real focus by ordinary carrier-free
	Fresnel propagation, which has no difficulty there; once safely past, the
	wave is re-factored onto a fresh **diverging** frame. All split points have
	closed forms because s and R are linear in z within a frame:

	- flatten where :math:`|R| = R^2/(A s^2)` (an invariant of the frame),
	  with :math:`A = k\,\xi_{supp}\,\Delta\xi/(\text{safety}\,\pi)`
	  evaluated at the **beam-support radius**
	  (:func:`beam_support_radius`, with a 1.2× spreading margin) — i.e. where
	  :func:`min_representable_curvature` is first satisfied for the actual
	  beam, not the (possibly much larger, empty) grid;
	- the **crossover plane** ``z_cross = z + |R|`` (recorded at the flatten)
	  is split out and logged — it is the focal / back-focal plane;
	- re-diverge at :math:`d = z - z_{cross} \ge A s^2` with ``R_new = +d``.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(n, n)``.
	dxi, deta : float
		Scaled-coordinate sample spacings (constant across all switches — the
		physical pixel is always ``|s|·Δξ``).
	wavelength : float
		Wavelength (metres).
	dz : float
		Physical segment length (metres).
	s, R : float or Sequence[float]
		Frame state at the segment start (``R = numpy.inf`` = flat); per-axis
		``(x, y)`` pairs on an anisotropic frame, whose axes then flatten and
		re-diverge independently at their own **line foci**.
	z : float
		Physical position at the segment start (metres).
	z_cross : float or Sequence, optional
		Position of the crossover a flat frame is currently traversing
		(recorded by an earlier flatten; carried in the scaled Signal's
		metadata between elements). ``None`` when not in a flat window;
		per-axis ``(x, y)`` pair (entries ``None`` where inactive) on an
		anisotropic frame.
	safety : float, optional
		Sampling-guard fraction for the frame changes, by default 0.5.
	s_min : float, optional
		Retained for signature compatibility with the single-frame path; the
		engine's closed-form splits are its own guard (a converging frame
		always flattens strictly before its crossover), so internal legs run
		unguarded and a legitimately deep flatten cannot trip the backstop,
		by default ``1e-3``.
	absorb : float, optional
		Absorbing-boundary margin fraction forwarded to
		:func:`propagate_free_scaled` (see there), by default 0.
	crossover : {'flat', 'jump'}, optional
		Crossover-traversal policy, by default ``'flat'`` (flatten → ordinary
		Fresnel through the focus → re-diverge). ``'jump'`` switches the
		converging frame **directly** onto its mirror-image diverging frame
		(``R_o = -d → R_n = +d``): the moved reference phase is twice the
		flatten phase, so the jump plane sits at half the flatten threshold
		(``|R_jump| = R²/(2 A s²)``, still closed-form); U then diffracts
		through its own focus inside the expanding frame, the crossover plane
		is still split out and logged, and there is no flat window at all
		(one switch instead of two). **Measured guidance**: the double phase
		budget makes the jump ride the converging frame twice as deep, and at
		tight crossovers the diffraction-limited focal structure in ξ (which
		grows as Airy/s) can outrun the field of view before the jump plane is
		reached — the electron-scale ``basic_column`` loses most of its beam
		this way, while the optical-regime through-focus test matches the flat
		policy (1.3e-2 vs 1.0e-2). ``'flat'`` is therefore the robust default;
		use ``'jump'`` only for mild crossovers (focal spot ≪ field of view at
		the jump depth).

	Returns
	-------
	tuple
		``(U, s, R, dtau, z, z_cross, logged)`` — the exit state, the total
		scaled increment Δτ over the segment, the exit position, the updated
		crossover marker (``None`` once re-diverged), and ``logged``: an
		ordered list of interior states ``(tag, U, s, R, dtau_cum, z, z_cross)``
		with ``tag`` in ``{'flatten', 'crossover', 'rediverge', 'jump'}`` and
		``dtau_cum`` the Δτ accumulated since the segment entry. On an
		anisotropic frame the per-axis events carry axis-suffixed tags
		(``'flatten-x'``, ``'crossover-y'``, ...) and the frame outputs are
		``(x, y)`` pairs; isotropic frames keep scalar outputs and unsuffixed
		tags bit-for-bit.

	Raises
	------
	ValueError
		Only from the backstop guards; the policy switches frames before the
		singularity is approached.

	Related
	-------
	change_scaled_frame : The frame-change primitive applied at each switch.
	min_representable_curvature : Places the flatten/re-diverge planes.

	Notes
	-----
	The frame — not the physical wave — is singular at a crossover: the linear
	scaling follows a geometric reference wavefront that collapses to a point,
	while the diffracted wave stays finite. Flattening simply steps onto a
	coordinate basis that remains useful through the focus.
	"""
	k = 2 * np.pi / wavelength
	remaining = float(dz)
	logged = []
	U = U.astype(complex, copy=True)
	tol = 1e-9 * (abs(dz) + abs(z) + 1.0)		# float tolerance on split positions
	isotropic = (np.ndim(s) == 0 and np.ndim(R) == 0
				 and (z_cross is None or np.ndim(z_cross) == 0))
	if not isotropic:
		return _hybrid_anisotropic(U, dxi, deta, wavelength, remaining, s, R, z,
								   z_cross, safety, absorb, crossover, k, tol, logged)
	dtau_total = 0.0
	while remaining > tol:
		# switch criterion at the beam's support (1.2x margin for spreading
		# within the leg; the frame-change guard re-checks the exact support)
		A = k * 1.2 * beam_support_radius(U, dxi, deta) * abs(dxi) / (safety * np.pi)
		if not np.isinf(R) and R < 0:
			# converging frame: switch where the moved reference phase first
			# becomes representable (frame-invariant thresholds)
			R_flat = R**2 / (A * s**2)
			R_switch = R_flat if crossover == 'flat' else R_flat / 2
			if abs(R) <= R_switch + tol:
				if crossover == 'flat':
					U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength, s, R, np.inf,
													   safety=safety)
					z_cross = z + abs(R)
					R = np.inf
					logged.append(("flatten", U, s, R, dtau_total, z, z_cross))
				else:
					# direct jump onto the mirror-image diverging frame
					d = abs(R)
					U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength, s, R, d,
													   safety=safety)
					z_cross = z + d
					R = d
					logged.append(("jump", U, s, R, dtau_total, z, z_cross))
				continue
			step = min(remaining, abs(R) - R_switch)
			U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, step, s, R, s_min=0.0, absorb=absorb)
			dtau_total += dt ; z += step ; remaining -= step
			continue
		if np.isinf(R) and z_cross is not None:
			if z < z_cross - tol:
				# flat frame heading into the focus: split at the crossover
				step = min(remaining, z_cross - z)
				U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, step, s, R, s_min=0.0, absorb=absorb)
				dtau_total += dt ; z += step ; remaining -= step
				if z >= z_cross - tol:
					logged.append(("crossover", U, s, R, dtau_total, z, z_cross))
				continue
			d = z - z_cross
			d_min = A * s**2
			if d >= d_min - tol:
				# safely past the focus: re-factor onto a diverging frame
				U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength, s, np.inf, d,
												   safety=safety)
				R = d
				z_cross = None
				logged.append(("rediverge", U, s, R, dtau_total, z, z_cross))
				continue
			step = min(remaining, d_min - d)
			U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, step, s, R, s_min=0.0, absorb=absorb)
			dtau_total += dt ; z += step ; remaining -= step
			continue
		if z_cross is not None and not np.isinf(R) and R > 0 and z < z_cross - tol:
			# jump policy: diverging frame carrying the beam through its focus —
			# split at the crossover so the focal plane is logged
			step = min(remaining, z_cross - z)
			U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, step, s, R, s_min=0.0, absorb=absorb)
			dtau_total += dt ; z += step ; remaining -= step
			if z >= z_cross - tol:
				logged.append(("crossover", U, s, R, dtau_total, z, z_cross))
				z_cross = None
			continue
		# flat with no crossover ahead, or diverging: plain scaled propagation
		U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, remaining, s, R, s_min=0.0, absorb=absorb)
		dtau_total += dt ; z += remaining ; remaining = 0.0
	return U, s, R, dtau_total, z, z_cross, logged


def _hybrid_anisotropic(U: np.ndarray, dxi: float, deta: float, wavelength: float,
						remaining: float, s, R, z: float, z_cross,
						safety: float, absorb: float, crossover: str,
						k: float, tol: float, logged: list) -> tuple:
	r"""Anisotropic (per-axis) branch of :func:`propagate_free_scaled_hybrid`.

	Runs the hybrid crossover policy with the two frame axes handled
	independently: each axis carries its own ``(s, R, τ, z_cross)`` and
	flattens / crosses / re-diverges at its own **line focus**, at planes set
	by that axis's own beam-support extent and pitch. Events carry
	axis-suffixed tags (``'flatten-x'``, ``'crossover-y'``, ...). The
	isotropic fast path in the public function never enters here, so
	round-lens columns are untouched.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(n, n)``.
	dxi, deta : float
		Scaled-coordinate sample spacings.
	wavelength : float
		Wavelength (metres).
	remaining : float
		Segment length still to propagate (metres).
	s, R : float or Sequence[float]
		Frame state at entry (scalar-or-pair).
	z : float
		Physical position at entry (metres).
	z_cross : float or Sequence or None
		Crossover marker(s) at entry; per-axis entries may be ``None``.
	safety : float
		Sampling-guard fraction for the frame changes.
	absorb : float
		Absorbing-boundary margin forwarded to :func:`propagate_free_scaled`.
	crossover : {'flat', 'jump'}
		Crossover-traversal policy (see the public function).
	k : float
		Wavenumber ``2π/λ``.
	tol : float
		Float tolerance on split positions.
	logged : list
		List to append interior states to (owned by the caller).

	Returns
	-------
	tuple
		``(U, s, R, dtau, z, z_cross, logged)`` with per-axis quantities
		joined by :func:`join_axes` (``z_cross`` collapses to ``None`` when
		both axes are clear, a scalar when equal, else an ``(x, y)`` pair).

	Raises
	------
	ValueError
		Only from the frame-change sampling guard; the policy switches frames
		before either axis's singularity is approached.

	Related
	-------
	propagate_free_scaled_hybrid : The public entry point and policy spec.
	"""
	ss = list(axis_components(s))
	RR = list(axis_components(R))
	if z_cross is None:
		zc = [None, None]
	elif np.ndim(z_cross) == 0:
		zc = [float(z_cross), float(z_cross)]
	else:
		zc = [None if z_cross[0] is None else float(z_cross[0]),
			  None if z_cross[1] is None else float(z_cross[1])]
	tau = [0.0, 0.0]
	names = ("x", "y")
	pitches = (abs(dxi), abs(deta))

	def snapshot():
		zj = None if (zc[0] is None and zc[1] is None) else \
			(zc[0] if zc[0] == zc[1] else (zc[0], zc[1]))
		return (join_axes(ss[0], ss[1]), join_axes(RR[0], RR[1]),
				join_axes(tau[0], tau[1]), zj)

	while remaining > tol:
		exts = beam_support_extents(U, dxi, deta)
		fired = False
		dists = [np.inf, np.inf]
		for a in range(2):
			A = k * 1.2 * exts[a] * pitches[a] / (safety * np.pi)
			R_a, s_a = RR[a], ss[a]
			if not np.isinf(R_a) and R_a < 0:
				R_flat = R_a**2 / (A * s_a**2)
				R_switch = R_flat if crossover == 'flat' else R_flat / 2
				if abs(R_a) <= R_switch + tol:
					R_new = list(RR)
					R_new[a] = np.inf if crossover == 'flat' else abs(R_a)
					tag = ("flatten-" if crossover == 'flat' else "jump-") + names[a]
					U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength,
													   (ss[0], ss[1]), (RR[0], RR[1]),
													   (R_new[0], R_new[1]), safety=safety)
					zc[a] = z + abs(R_a)
					RR[a] = R_new[a]
					s_j, R_j, t_j, z_j = snapshot()
					logged.append((tag, U, s_j, R_j, t_j, z, z_j))
					fired = True
					break
				dists[a] = abs(R_a) - R_switch
			elif np.isinf(R_a) and zc[a] is not None:
				if z < zc[a] - tol:
					dists[a] = zc[a] - z
				else:
					d = z - zc[a]
					d_min = A * s_a**2
					if d >= d_min - tol:
						R_new = list(RR)
						R_new[a] = d
						U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength,
														   (ss[0], ss[1]), (RR[0], RR[1]),
														   (R_new[0], R_new[1]), safety=safety)
						RR[a] = d
						zc[a] = None
						s_j, R_j, t_j, z_j = snapshot()
						logged.append(("rediverge-" + names[a], U, s_j, R_j, t_j, z, z_j))
						fired = True
						break
					dists[a] = d_min - d
			elif zc[a] is not None and not np.isinf(R_a) and R_a > 0 and z < zc[a] - tol:
				dists[a] = zc[a] - z	# jump policy: split to log the line focus
		if fired:
			continue
		step = min(remaining, dists[0], dists[1])
		heading = [zc[a] is not None and z < zc[a] - tol for a in range(2)]
		U, s_j, R_j, dt = propagate_free_scaled(U, dxi, deta, wavelength, step,
												(ss[0], ss[1]), (RR[0], RR[1]),
												s_min=0.0, absorb=absorb)
		ss = list(axis_components(s_j))
		RR = list(axis_components(R_j))
		dt_x, dt_y = axis_components(dt)
		tau[0] += dt_x
		tau[1] += dt_y
		z += step
		remaining -= step
		for a in range(2):
			if heading[a] and z >= zc[a] - tol:
				s_j, R_j, t_j, z_j = snapshot()
				logged.append(("crossover-" + names[a], U, s_j, R_j, t_j, z, z_j))
				if not np.isinf(RR[a]) and RR[a] > 0:
					zc[a] = None		# jump path: marker consumed at the focus
	s_j, R_j, t_j, z_j = snapshot()
	return U, s_j, R_j, t_j, z, z_j, logged


def aperture_mask(field: np.ndarray, dx: float, dy: float, radius: float,
				  antialias: bool = True) -> np.ndarray:
	r"""Apply a hard circular aperture to a field.

	The physical model is the sharp mask :math:`\Theta(a - r)`; by default it
	is applied **anti-aliased**: edge pixels carry their area-coverage fraction
	(a linear ramp over one pixel — the projection of the sharp mask onto the
	grid), so the mask's above-Nyquist edge content does not fold back and
	propagate as a spurious axis-aligned interference pattern. Every
	representable Fresnel fringe of the sharp edge is unaffected.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	radius : float
		Aperture radius (metres).
	antialias : bool, optional
		Apply the edge-coverage (alias-suppressed) mask, by default True.
		``False`` restores the point-sampled binary mask.

	Returns
	-------
	np.ndarray
		Masked complex field, shape ``(ny, nx)``.

	Related
	-------
	bandlimited_disk : The exactly alias-free initial disk (Source path).
	"""
	X, Y = transverse_coordinates(field.shape, dx, dy)
	r = np.sqrt(X**2 + Y**2)
	if not antialias:
		return field * (r <= radius)
	px = max(abs(dx), abs(dy))
	return field * np.clip(0.5 + (radius - r) / px, 0.0, 1.0)


def bandlimited_disk(shape: tuple, dx: float, dy: float, radius: float) -> np.ndarray:
	r"""Exactly alias-free sampling of the sharp disk :math:`\Theta(a - r)`.

	A point-sampled binary disk carries the edge's above-Nyquist frequencies
	folded onto wrong low frequencies; propagated coherently they interfere as
	an axis-aligned grid texture inside the beam. This builder instead
	synthesizes the **band-limited projection** of the same sharp disk from its
	analytic spectrum,

	.. math::

		\tilde\Theta(k_r) = 2\pi a^2 \, \frac{J_1(a k_r)}{a k_r}
		\qquad (\pi a^2 \text{ at } k_r = 0),

	sampled on the discrete frequency grid up to Nyquist and inverse-FFT'd:
	every representable frequency is exact and nothing folds. The physical
	Fresnel edge diffraction is fully preserved; only the numerical aliasing
	is removed. The band limit itself shows as Gibbs ripple at the edge — the
	honest sampled representation of a discontinuity.

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	radius : float
		Disk radius ``a`` (metres).

	Returns
	-------
	np.ndarray
		Complex disk field, shape ``(ny, nx)``, centred per
		:func:`transverse_coordinates` (unit interior, zero exterior, Gibbs
		ripple at the edge).

	Related
	-------
	aperture_mask : The anti-aliased *mask* form for fields mid-column.

	Notes
	-----
	Uses ``scipy.special.j1`` when available; otherwise falls back to an
	8×-supersampled area-coverage mask (aliasing attenuated rather than
	exactly zero).
	"""
	ny, nx = shape
	try:
		from scipy.special import j1
	except ImportError:
		# fallback: 8x-supersampled coverage, block-averaged (sinc^2-attenuated folding)
		ss = 8
		Xf, Yf = transverse_coordinates((ny * ss, nx * ss), dx / ss, dy / ss)
		fine = (np.sqrt(Xf**2 + Yf**2) <= radius).astype(float)
		return fine.reshape(ny, ss, nx, ss).mean(axis=(1, 3)).astype(complex)
	kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
	ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
	KX, KY = np.meshgrid(kx, ky)
	KR = np.sqrt(KX**2 + KY**2)
	with np.errstate(invalid="ignore", divide="ignore"):
		F = np.where(KR == 0, np.pi * radius**2,
					 2 * np.pi * radius**2 * j1(radius * KR) / (radius * KR))
	# continuous inverse FT -> DFT samples on the centred grid
	disk = np.fft.fftshift(np.fft.ifft2(F)) / (abs(dx) * abs(dy))
	return disk.real.astype(complex)
