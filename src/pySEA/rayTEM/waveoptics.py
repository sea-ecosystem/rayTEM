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


def angular_spectrum_propagate(field: np.ndarray, dx: float, dy: float,
							   wavelength: float, dz: float) -> np.ndarray:
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

	Returns
	-------
	np.ndarray
		Propagated complex field, shape ``(ny, nx)``.
	"""
	if dz == 0:
		return field.astype(complex, copy=True)
	ny, nx = field.shape
	k = 2 * np.pi / wavelength
	kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
	ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
	KX, KY = np.meshgrid(kx, ky)
	H = np.exp(1j * k * dz) * np.exp(-1j * (KX**2 + KY**2) * dz / (2 * k))
	return np.fft.ifft2(np.fft.fft2(field) * H)


def focal_phase(field: np.ndarray, dx: float, dy: float, wavelength: float,
				power_x: float, power_y: float) -> np.ndarray:
	r"""Apply a (possibly astigmatic) thin focusing phase.

	Multiplies by :math:`\exp[-i k (P_x x^2 + P_y y^2)/2]`, where the focal powers
	``power_x``/``power_y`` are inverse focal lengths (``1/f``). Equal powers give a
	round lens; opposite powers give a quadrupole.

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
	X, Y = transverse_coordinates(field.shape, dx, dy)
	k = 2 * np.pi / wavelength
	return field * np.exp(-1j * k * (power_x * X**2 + power_y * Y**2) / 2)


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
	X, Y = transverse_coordinates(field.shape, dx, dy)
	k = 2 * np.pi / wavelength
	return field * np.exp(1j * k * (tilt_x * X + tilt_y * Y))


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
