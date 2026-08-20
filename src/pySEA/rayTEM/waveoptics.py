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
		(``1 + dz/R0 <= 0``) — the coordinate chart is singular there
		(handoff Eq 52); stop before the crossover or reset the chart.
	"""
	if np.isinf(R0):
		return dz / s0**2
	growth = 1.0 + dz / R0
	if growth <= 0:
		raise ValueError(f"Scaled chart crosses s=0 inside this segment (crossover at dz = {-R0} m "
						 f"of {dz} m); stop before the crossover or reset the scaling chart.")
	return dz / (s0**2 * growth)


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
	"""
	U = s * psi.astype(complex)
	if not np.isinf(R):
		X, Y = transverse_coordinates(psi.shape, dx, dy)
		k = 2 * np.pi / wavelength
		U = U * np.exp(-1j * k * (X**2 + Y**2) / (2 * R))
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


def propagate_free_scaled(U: np.ndarray, dxi: float, deta: float, wavelength: float,
						  dz: float, s: float, R: float, s_min: float = 1e-3) -> tuple:
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

	Returns
	-------
	tuple
		``(U_out, s_out, R_out, dtau)``.

	Raises
	------
	ValueError
		If the chart crosses ``s = 0`` inside the segment, or the exit scale
		violates ``|s_out| > s_min`` — with the crossover position named, since
		the singularity belongs to the chart, not the physical wave.
	"""
	if dz == 0:
		return U.astype(complex, copy=True), s, R, 0.0
	dtau = scaled_delta_tau(dz, s, R)		# also guards the in-segment zero crossing
	s_out = s if np.isinf(R) else s * (1.0 + dz / R)
	if abs(s_out) <= s_min:
		z_cross = -R if not np.isinf(R) else np.inf
		raise ValueError(f"Scaled chart reaches |s| = {abs(s_out):.3e} <= s_min = {s_min} at the segment "
						 f"end (chart crossover at dz = {z_cross} m); stop before the crossover, "
						 "reset the scaling chart, or lower s_min knowingly.")
	R_out = R + dz if not np.isinf(R) else np.inf
	U_out = angular_spectrum_propagate(U, dxi, deta, wavelength, dtau, include_carrier=False)
	return U_out, s_out, R_out, dtau


def aperture_mask(field: np.ndarray, dx: float, dy: float, radius: float) -> np.ndarray:
	"""Apply a hard circular aperture, zeroing the field outside ``radius``.

	Parameters
	----------
	field : np.ndarray
		Complex field ``(ny, nx)``.
	dx, dy : float
		Sample spacings (metres).
	radius : float
		Aperture radius (metres).

	Returns
	-------
	np.ndarray
		Masked complex field, shape ``(ny, nx)``.
	"""
	X, Y = transverse_coordinates(field.shape, dx, dy)
	return field * ((X**2 + Y**2) <= radius**2)
