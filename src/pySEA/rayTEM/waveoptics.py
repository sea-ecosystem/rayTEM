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
	dz : float
		Propagation distance (metres; the scaled path passes Δτ).
	include_carrier : bool, optional
		Include the on-axis carrier term ``k·dz``, by default True.

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
	chi = -(KX**2 + KY**2) * dz / (2 * k)
	if include_carrier:
		chi = chi + k * dz
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
	if dz == 0:
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
	amp = np.abs(U)
	mask = amp > threshold * amp.max()
	X, Y = transverse_coordinates(U.shape, abs(dxi), abs(deta))
	ext = max(float(np.abs(X[mask]).max()), float(np.abs(Y[mask]).max()))
	return max(ext, abs(dxi))


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
						s_old: float, R_old: float, R_new: float,
						s_new: float = None, safety: float = 0.5) -> tuple:
	r"""Re-express the same physical wave in a different scaled frame (Eq 5).

	A *frame* is a choice of factorization
	:math:`\psi = (1/s)\,U(x/s, y/s)\,e^{ik(x^2+y^2)/2R}`; this primitive
	transforms :math:`(s_o, R_o, U_o) \to (s_n, R_n, U_n)` while keeping ψ
	identical:

	.. math::

		U_n = \frac{s_n}{s_o}\, U_o \,
		\exp\!\left[\frac{ik\,(x^2+y^2)}{2}
		\left(\frac{1}{R_o} - \frac{1}{R_n}\right)\right]

	using the **physical-grid-continuous convention**: the new pitch is
	:math:`\Delta\xi_n = \Delta\xi_o\, s_o/s_n`, so the samples sit at the same
	physical points and the operation is pointwise — no interpolation. The
	physical representation is the special frame ``(s=1, R=inf)``, making
	:func:`factor_wave` and :func:`reconstruct_physical_wave` special cases.

	Parameters
	----------
	U : np.ndarray
		Scaled field ``(n, n)`` in the old frame.
	dxi, deta : float
		Old-frame sample spacings Δξ/Δη.
	wavelength : float
		Wavelength (metres).
	s_old : float
		Old-frame transverse scale (nonzero).
	R_old : float
		Old-frame reference curvature radius (metres); ``numpy.inf`` = flat.
	R_new : float
		New-frame reference curvature radius (metres); ``numpy.inf`` = flat.
	s_new : float, optional
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
		is zero.

	Related
	-------
	min_representable_curvature : The guard's threshold.
	factor_wave, reconstruct_physical_wave : The (1, inf) special cases.

	Notes
	-----
	The singularity of a converging frame at its crossover (``s -> 0``)
	belongs to the frame, not to the wave; this operation is how propagation
	steps onto a better frame before that happens (the hybrid policy in
	:func:`propagate_free_scaled_hybrid`). A sign flip between ``s_old`` and
	``s_new`` applies the signed amplitude ratio of Eq 5; the implied ξ-grid
	inversion is exact for the (even) quadratic reference phase and is the
	caller's relabeling to track.
	"""
	if s_new is None:
		s_new = s_old
	if s_new == 0:
		raise ValueError("change_scaled_frame requires a nonzero s_new; the frame is "
						 "singular at s = 0 (switch frames before the crossover).")
	c_old = 0.0 if np.isinf(R_old) else 1.0 / R_old
	c_new = 0.0 if np.isinf(R_new) else 1.0 / R_new
	U_out = (s_new / s_old) * U.astype(complex)
	if c_old != c_new:
		# physical coordinates of the (shared) sample points
		X, Y = transverse_coordinates(U.shape, abs(s_old) * dxi, abs(s_old) * deta)
		k = 2 * np.pi / wavelength
		chi = k * (X**2 + Y**2) / 2 * (c_old - c_new)
		if safety is not None:
			# measure the per-pixel step over the beam's support only — phase
			# applied where the field is essentially zero is harmless
			r_supp = beam_support_radius(U, dxi, deta)
			step = k * abs(s_old)**2 * max(abs(dxi), abs(deta)) * r_supp * abs(c_old - c_new)
			if step > safety * np.pi:
				R_min = min_representable_curvature(U.shape[0], dxi, wavelength, s_old,
													safety, x_max=r_supp)
				raise ValueError(f"Frame change from R={R_old} m to R={R_new} m is not representable "
								 f"on this grid (per-pixel phase step {step:.2f} rad at the beam "
								 f"support > {safety:.2g}*pi): the curvature moved into U must "
								 f"satisfy |R| >= {R_min:.3e} m. Switch frames closer to the plane "
								 "where the reference curvature is weaker, or refine the grid.")
		U_out = U_out * np.exp(1j * chi)
	return U_out, dxi * s_old / s_new, deta * s_old / s_new


def factor_wave(psi: np.ndarray, dx: float, dy: float, wavelength: float,
				s: float, R: float) -> tuple:
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
	s : float
		Chosen transverse scale (nonzero).
	R : float
		Chosen reference radius of curvature (metres); ``numpy.inf`` for none.

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
	U, _, _ = change_scaled_frame(psi, dx, dy, wavelength, s_old=1.0, R_old=np.inf,
								  R_new=R, s_new=s, safety=None)
	return U, dx / s, dy / s


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
							  wavelength: float, s: float, R: float,
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
	s : float
		Transverse scale at this plane (nonzero).
	R : float
		Reference radius of curvature (metres); ``numpy.inf`` for none.
	target_dx : float, optional
		Prescribed physical pixel size; ``None`` (default) reconstructs on the
		native grid ``Δx = |s|·Δξ``.
	target_shape : tuple, optional
		Prescribed output shape ``(ny, nx)``; required with ``target_dx``.

	Returns
	-------
	tuple
		``(psi, dx, dy)`` — the physical wave and its pixel sizes.

	Related
	-------
	factor_wave : Exact inverse (handoff Eq 55).
	fourier_resample : The band-limited resampler used for target grids.
	"""
	if target_dx is not None:
		n_out = target_shape[0]
		U = fourier_resample(U, dxi, n_out, target_dx / abs(s))
		dxi = deta = target_dx / abs(s)
	dx = abs(s) * dxi
	dy = abs(s) * deta
	psi = U.astype(complex) / s
	if not np.isinf(R):
		X, Y = transverse_coordinates(psi.shape, dx, dy)
		k = 2 * np.pi / wavelength
		psi = psi * np.exp(1j * k * (X**2 + Y**2) / (2 * R))
	return psi, dx, dy


def apply_thin_lens_scaled(s: float, R: float, power: float) -> tuple:
	r"""Absorb a thin-lens focusing power into the scaled curvature state.

	Handoff Eqs 45–46: :math:`1/R^+ = 1/R^- - 1/f` with ``s`` continuous
	through the lens and ``s' = s/R`` re-derived from the new curvature. The
	scaled field U is untouched (Eq 15) — only the reference state changes.

	Parameters
	----------
	s : float
		Transverse scale at the lens plane (unchanged, returned for symmetry).
	R : float
		Incoming reference radius of curvature (metres); ``numpy.inf`` for flat.
	power : float
		Focusing power ``1/f`` to absorb (1/metres).

	Returns
	-------
	tuple
		``(s, R_out)`` — the (unchanged) scale and the updated curvature
		(``numpy.inf`` when the outgoing wavefront is flat).
	"""
	curvature = (0.0 if np.isinf(R) else 1.0 / R) - power
	R_out = np.inf if curvature == 0 else 1.0 / curvature
	return s, R_out


def boundary_window(shape: tuple, margin: float = 0.1) -> np.ndarray:
	r"""Absorbing-boundary window: 1 in the interior, cosine → 0 at the edge.

	The FFT propagator is periodic: field that diffracts out of the modeled
	field of view re-enters coherently from the opposite side and interferes
	with the beam (an axis-aligned artifact, since the wrap distance is
	shortest along the square grid's axes). Physically those electrons leave
	the beam and never return, so the boundary should absorb them: this
	separable window ramps from 1 to 0 over the outer ``margin`` fraction of
	each axis (raised cosine).

	Parameters
	----------
	shape : tuple of int
		Field shape ``(ny, nx)``.
	margin : float, optional
		Fraction of each axis occupied by the absorbing band on each side,
		by default 0.1.

	Returns
	-------
	np.ndarray
		Real window, shape ``(ny, nx)``, values in [0, 1].

	Related
	-------
	propagate_free_scaled : Applies it between τ sub-steps when ``absorb > 0``.
	"""
	def ramp(n):
		m = max(1, int(round(margin * n)))
		w = np.ones(n)
		t = (np.arange(m) + 1) / m					# 0 -> 1 toward the edge
		w[n - m:] = 0.5 * (1 + np.cos(np.pi * t))
		w[:m] = w[n - m:][::-1]
		return w
	ny, nx = shape
	return np.outer(ramp(ny), ramp(nx))


def propagate_free_scaled(U: np.ndarray, dxi: float, deta: float, wavelength: float,
						  dz: float, s: float, R: float, s_min: float = 1e-3,
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
	s : float
		Transverse scale at the segment start.
	R : float
		Reference radius of curvature at the segment start (``numpy.inf`` = flat).
	s_min : float, optional
		Crossover guard: the segment must keep ``|s| > s_min`` (handoff Eq 52),
		by default ``1e-3``.
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
		``(U_out, s_out, R_out, dtau)``. With ``absorb > 0`` the total
		``|U|²`` decreases by the power lost through the boundary.

	Raises
	------
	ValueError
		If the frame crosses ``s = 0`` inside the segment, or the exit scale
		violates ``|s_out| > s_min`` — with the crossover position named, since
		the singularity belongs to the frame, not the physical wave.
	"""
	if dz == 0:
		return U.astype(complex, copy=True), s, R, 0.0
	dtau = scaled_delta_tau(dz, s, R)		# also guards the in-segment zero crossing
	s_out = s if np.isinf(R) else s * (1.0 + dz / R)
	if abs(s_out) <= s_min:
		z_cross = -R if not np.isinf(R) else np.inf
		raise ValueError(f"Scaled frame reaches |s| = {abs(s_out):.3e} <= s_min = {s_min} at the segment "
						 f"end (frame crossover at dz = {z_cross} m); stop before the crossover, "
						 "switch frames (hybrid mode), or lower s_min knowingly.")
	R_out = R + dz if not np.isinf(R) else np.inf
	if absorb and absorb > 0:
		# absorbing boundary: sub-step in tau so no spectral component can
		# traverse the absorbing band unattenuated within one FFT step
		# (max transverse travel per step = lambda * f_Nyquist * dtau_step)
		n = U.shape[0]
		band = absorb * n * abs(dxi)					# absorber width in xi
		dtau_step = 2 * band * abs(dxi) / wavelength	# travel at Nyquist = band
		n_steps = max(1, int(np.ceil(abs(dtau) / dtau_step)))
		W = boundary_window(U.shape, margin=absorb)
		U_out = U.astype(complex, copy=True) * W
		for _ in range(n_steps):
			U_out = angular_spectrum_propagate(U_out, dxi, deta, wavelength,
											   dtau / n_steps, include_carrier=False)
			U_out = U_out * W
		return U_out, s_out, R_out, dtau
	U_out = angular_spectrum_propagate(U, dxi, deta, wavelength, dtau, include_carrier=False)
	return U_out, s_out, R_out, dtau


def propagate_free_scaled_hybrid(U: np.ndarray, dxi: float, deta: float,
								 wavelength: float, dz: float, s: float, R: float,
								 z: float, z_cross: float = None,
								 safety: float = 0.5, s_min: float = 1e-3,
								 absorb: float = 0.0) -> tuple:
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
	s, R : float
		Frame state at the segment start (``R = numpy.inf`` = flat).
	z : float
		Physical position at the segment start (metres).
	z_cross : float, optional
		Position of the crossover a flat frame is currently traversing
		(recorded by an earlier flatten; carried in the scaled Signal's
		metadata between elements). ``None`` when not in a flat window.
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

	Returns
	-------
	tuple
		``(U, s, R, dtau, z, z_cross, logged)`` — the exit state, the total
		scaled increment Δτ over the segment, the exit position, the updated
		crossover marker (``None`` once re-diverged), and ``logged``: an
		ordered list of interior states ``(tag, U, s, R, dtau_cum, z, z_cross)``
		with ``tag`` in ``{'flatten', 'crossover', 'rediverge'}`` and
		``dtau_cum`` the Δτ accumulated since the segment entry.

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
	dtau_total = 0.0
	logged = []
	U = U.astype(complex, copy=True)
	tol = 1e-9 * (abs(dz) + abs(z) + 1.0)		# float tolerance on split positions
	while remaining > tol:
		# switch criterion at the beam's support (1.2x margin for spreading
		# within the leg; the frame-change guard re-checks the exact support)
		A = k * 1.2 * beam_support_radius(U, dxi, deta) * abs(dxi) / (safety * np.pi)
		if not np.isinf(R) and R < 0:
			# converging frame: flatten where the residual curvature first
			# becomes representable (|R_flat| = R^2/(A s^2), frame-invariant)
			R_flat = R**2 / (A * s**2)
			if abs(R) <= R_flat + tol:
				U, dxi, deta = change_scaled_frame(U, dxi, deta, wavelength, s, R, np.inf,
												   safety=safety)
				z_cross = z + abs(R)
				R = np.inf
				logged.append(("flatten", U, s, R, dtau_total, z, z_cross))
				continue
			step = min(remaining, abs(R) - R_flat)
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
		# flat with no crossover ahead, or diverging: plain scaled propagation
		U, s, R, dt = propagate_free_scaled(U, dxi, deta, wavelength, remaining, s, R, s_min=0.0, absorb=absorb)
		dtau_total += dt ; z += remaining ; remaining = 0.0
	return U, s, R, dtau_total, z, z_cross, logged


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
