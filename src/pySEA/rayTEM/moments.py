"""Beam-envelope state and moment-closure models for rayTEM.

The covariance ("moments") propagation mode describes the beam by its first
two moments only — a mean state vector and a covariance over the geometric
columns of :data:`elements.convention`. Linear optics transports those two
exactly, for *any* underlying distribution. A nonlinear element (an
aberration) does not: its output covariance depends on moments the beam state
does not carry, so something has to supply them.

That "something" is a :class:`MomentClosure`, and this module keeps it
**explicit and swappable** rather than buried inside the propagation code.
:class:`GaussianMomentClosure` is the one closure shipped today; it evaluates
any central moment by Wick pairing (Isserlis' theorem), which is what makes a
Gaussian ensemble's fourth and sixth moments expressible in the covariance
alone. Using it does not make the beam Gaussian — it records an assumption
made at one nonlinear step, which is why it is a separate object a caller can
replace.

:class:`CovarianceBeam` is the beam state itself, plus the resolution
quantities a covariance is actually read for: rms widths, position-angle
correlations, emittances, and the principal axes of the real-space, angular,
and transverse-momentum blocks.

Related
-------
elements.Element.propagate_moments : The transport step these serve.
elements.Element._aberration_monomials : The nonlinear kick, as a polynomial.
assemblies.Microscope.covariance_beam : Builds a beam view over a propagated column.

Notes
-----
All moments here are **central** — taken about the mean. A kick expressed in
absolute coordinates is converted first by :func:`_center_monomials`.

References
----------
.. [1] L. Isserlis, "On a formula for the product-moment coefficient of any
   order of a normal frequency distribution in any number of variables",
   Biometrika 12, 134 (1918).
"""

from __future__ import annotations

from typing import Sequence, Literal

import numpy as xp


class MomentClosure:
	"""How a nonlinear map obtains moments the beam state does not store.

	A covariance beam carries only a mean and a covariance. Pushing it through
	a map with a nonlinear term of degree ``n`` requires central moments up to
	degree ``2n``, which are simply not present in the state. A closure model
	supplies them from what *is* present, under a stated assumption.

	This base class defines the one method the propagation code calls. It is
	deliberately a single, general entry point — a closure is asked for an
	arbitrary central moment by column index, not for a menu of named special
	cases — so that swapping the closure genuinely changes the physics rather
	than merely relabelling it.

	Attributes
	----------
	name : str
		Short identifier for the closure, recorded in results and printed by
		examples so the assumption in force is visible.

	Methods
	-------
	moment(Sigma, indices)
		The central moment for the given column indices.

	Related
	-------
	GaussianMomentClosure : The closure shipped today.
	CovarianceBeam : Carries the chosen closure alongside the state.

	Notes
	-----
	Subclasses need only implement :meth:`moment`. A closure that stores
	explicit third- or fourth-moment tensors would override it to read those
	tensors for the degrees it holds and fall back for the rest.

	Examples
	--------
	>>> issubclass(GaussianMomentClosure, MomentClosure)
	True
	"""

	name: str = 'abstract'

	def moment(self, Sigma:xp.ndarray, indices:Sequence[int]) -> float:
		r"""The central moment :math:`\langle \prod_a s_{i_a} \rangle`.

		Here :math:`s = r - \mu` is the deviation from the mean, and
		``indices`` lists the columns of the product with repetition, so
		``(ix, ix)`` requests :math:`\langle s_x^2\rangle` and
		``(ix, ix, ix, ix)`` requests :math:`\langle s_x^4\rangle`.

		Parameters
		----------
		Sigma : xp.ndarray
			Covariance matrix, shape ``(len(convention),)*2``.
		indices : Sequence of int
			Column indices of the product, with repetition. The empty
			sequence is the empty product and evaluates to 1.

		Returns
		-------
		float
			The central moment.

		Raises
		------
		NotImplementedError
			Always — subclasses supply the closure.

		Related
		-------
		GaussianMomentClosure.moment : The Wick-pairing implementation.
		"""
		raise NotImplementedError("MomentClosure is abstract; use GaussianMomentClosure "
								  "or supply a closure implementing moment(Sigma, indices).")


class GaussianMomentClosure(MomentClosure):
	r"""Close moments by Isserlis' theorem, as for a centered Gaussian.

	For a centered Gaussian every odd central moment vanishes and every even
	one is the sum over perfect pairings of the indices,

	.. math::

		\langle s_i s_j s_k s_l \rangle =
		\Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{il}\Sigma_{jk},

	and likewise for higher even degrees. Evaluating that recursively — pair
	the first index with each of the others, recurse on the remainder — gives
	every moment a nonlinear kick can ask for, at any degree, with no
	per-aberration algebra.

	The assumption this encodes is local: it is applied at each nonlinear
	element to obtain the moments that element needs. It does **not** assert
	that the beam is Gaussian, and after an aberrated element the beam
	demonstrably is not. Where that matters — a second aberrated element
	acting on an already-distorted distribution — the approximation is real
	and should be stated by the caller.

	Attributes
	----------
	name : str
		``'gaussian'``.

	Methods
	-------
	moment(Sigma, indices)
		The central moment by Wick pairing.

	Related
	-------
	MomentClosure : The interface.
	elements.Element._aberration_moment_pieces : The consumer.

	Notes
	-----
	The recursion visits :math:`(n-1)!!` pairings for degree ``n``, which is
	15 at degree six — the highest any shipped aberration requires — so the
	cost is negligible next to the transport itself.

	Examples
	--------
	>>> import numpy as np
	>>> S = np.zeros((6, 6)); S[0, 0] = 4.0
	>>> GaussianMomentClosure().moment(S, (0, 0, 0, 0))    # <x^4> = 3 sigma^4
	48.0
	>>> GaussianMomentClosure().moment(S, (0, 0, 0))       # odd moments vanish
	0.0

	References
	----------
	.. [1] L. Isserlis, Biometrika 12, 134 (1918).
	"""

	name: str = 'gaussian'

	def moment(self, Sigma:xp.ndarray, indices:Sequence[int]) -> float:
		"""The central moment for ``indices`` by Wick pairing.

		Parameters
		----------
		Sigma : xp.ndarray
			Covariance matrix, shape ``(len(convention),)*2``.
		indices : Sequence of int
			Column indices of the product, with repetition.

		Returns
		-------
		float
			The central moment; 1.0 for the empty product, 0.0 for any odd
			degree.

		Raises
		------
		None

		Related
		-------
		MomentClosure.moment : The contract this satisfies.
		"""
		idx = tuple(int(i) for i in indices)
		n = len(idx)
		if n == 0:
			return 1.0
		if n % 2:
			return 0.0
		head, rest = idx[0], idx[1:]
		total = 0.0
		for k in range(len(rest)):
			total += float(Sigma[head, rest[k]]) * self.moment(Sigma, rest[:k] + rest[k + 1:])
		return total


def _center_monomials(monomials:Sequence[tuple], mu:xp.ndarray) -> list:
	r"""Rewrite an absolute-coordinate polynomial in deviations from the mean.

	A nonlinear kick is naturally written in absolute coordinates — spherical
	aberration deflects by :math:`g(x^2+y^2)x` in the pupil, not in the
	deviation from wherever the beam centroid happens to sit. Moments,
	however, are central. This expands each monomial
	:math:`\prod_a r_{i_a} = \prod_a (\mu_{i_a} + s_{i_a})` by the binomial
	theorem, returning the same polynomial in :math:`s`.

	For a centered beam (:math:`\mu_\perp = 0`) the result is the input, so
	the conversion costs nothing in the common case; it matters when an
	element sits on a beam that has already been displaced.

	Parameters
	----------
	monomials : Sequence of tuple
		``(coefficient, indices)`` pairs, where ``indices`` is a tuple of
		column indices with repetition; the empty tuple is a constant term.
	mu : xp.ndarray
		Mean state vector, shape ``(len(convention),)``.

	Returns
	-------
	list of tuple
		The equivalent polynomial in central coordinates, in the same
		``(coefficient, indices)`` form. Terms are not combined.

	Raises
	------
	None

	Related
	-------
	MomentClosure.moment : Consumes the centered form.
	elements.Element._aberration_monomials : Produces the absolute form.

	Notes
	-----
	Degree ``n`` expands to :math:`2^n` terms, which is 8 for the cubic
	spherical kick — small enough that no term pruning is worthwhile beyond
	dropping exactly-zero coefficients.

	Examples
	--------
	>>> import numpy as np
	>>> mu = np.zeros(6)
	>>> _center_monomials([(2.0, (0, 0))], mu)          # centered beam: unchanged
	[(2.0, (0, 0))]
	"""
	out = []
	for coef, idx in monomials:
		c = float(coef)
		if c == 0.0:
			continue
		idx = tuple(int(i) for i in idx)
		n = len(idx)
		for mask in range(1 << n):
			kept, factor = [], c
			for bit in range(n):
				if mask & (1 << bit):
					kept.append(idx[bit])
				else:
					factor *= float(mu[idx[bit]])
			if factor != 0.0:
				out.append((factor, tuple(kept)))
	return out


def _kick_moments(monomials_by_column:dict, Sigma:xp.ndarray,
				 closure:MomentClosure) -> tuple:
	r"""Mean, cross-covariance and self-covariance of a polynomial kick.

	Given a nonlinear kick :math:`\delta` whose component in each output
	column is a polynomial in the *central* coordinates, this returns the
	three pieces the covariance update needs:

	.. math::

		\langle\delta_c\rangle, \qquad
		\mathrm{Cov}(s_a,\delta_c), \qquad
		\mathrm{Cov}(\delta_c,\delta_d)

	Every moment is obtained from ``closure``, so the physics of the closure
	assumption enters here and nowhere else. Because the covariances are
	formed as :math:`\langle\cdot\rangle - \langle\cdot\rangle\langle\cdot\rangle`,
	a kick with a nonzero mean (any even-order aberration) is handled
	correctly: the mean shift is reported rather than silently folded into the
	width.

	Parameters
	----------
	monomials_by_column : dict
		Maps an output column index to a sequence of ``(coefficient,
		indices)`` monomials in central coordinates.
	Sigma : xp.ndarray
		Entrance covariance, shape ``(len(convention),)*2``.
	closure : MomentClosure
		Supplies the higher central moments.

	Returns
	-------
	tuple of xp.ndarray
		``(delta_mean, C, D)`` with shapes ``(n,)``, ``(n, n)`` and
		``(n, n)`` for ``n = Sigma.shape[0]``: the ensemble-mean kick, the
		state-kick cross-covariance ``C[a, c] = Cov(s_a, delta_c)``, and the
		kick self-covariance ``D[c, d] = Cov(delta_c, delta_d)``.

	Raises
	------
	None

	Related
	-------
	elements.Element._aberration_moment_pieces : Assembles these into the update.
	_center_monomials : Puts the kick in the central coordinates assumed here.

	Notes
	-----
	No aberration-specific algebra appears here: adding a new nonlinear term
	means emitting its monomials, not deriving its moments.
	"""
	n = int(xp.shape(Sigma)[0])
	delta_mean = xp.zeros(n)
	C = xp.zeros((n, n))
	D = xp.zeros((n, n))
	cols = sorted(monomials_by_column)
	for c in cols:
		for coef, idx in monomials_by_column[c]:
			delta_mean[c] += coef * closure.moment(Sigma, idx)
	for c in cols:
		for a in range(n):
			acc = 0.0
			for coef, idx in monomials_by_column[c]:
				acc += coef * closure.moment(Sigma, (a,) + idx)
			C[a, c] = acc									# <s_a> is zero by construction
	for c in cols:
		for d in cols:
			acc = 0.0
			for coef_c, idx_c in monomials_by_column[c]:
				for coef_d, idx_d in monomials_by_column[d]:
					acc += coef_c * coef_d * closure.moment(Sigma, idx_c + idx_d)
			D[c, d] = acc - delta_mean[c] * delta_mean[d]
	return delta_mean, C, D


class CovarianceBeam:
	r"""A beam described by its first two moments, and the resolution they imply.

	The moments mode's answer to :class:`elements.Rays`: where a ray table
	carries a bundle of trajectories, this carries the ensemble's mean state
	and covariance — and, deliberately, the :class:`MomentClosure` under which
	they were transported, so a result never loses track of the assumption
	that produced it.

	It also supplies what a covariance is actually read for. A single probe
	diameter throws away most of what Σ knows: the position-angle
	correlations that say whether a plane is a waist or a crossover, the
	emittance that says how much of the width is irreducible, and the
	principal axes that say whether "the resolution" is even isotropic. Those
	are the quantities here.

	Parameters
	----------
	mean : xp.ndarray
		Mean state, shape ``(len(convention),)`` for one plane or
		``(n_planes, len(convention))`` for a propagated column.
	covariance : xp.ndarray
		Covariance, shape ``(len(convention),)*2`` or
		``(n_planes,) + (len(convention),)*2``, matching ``mean``.
	moment_closure : MomentClosure, optional
		The closure the transport used, by default
		:class:`GaussianMomentClosure`. Recorded, not applied — the transport
		has already happened.
	wavelength : float, optional
		Electron wavelength (metres), by default None. Required only by
		:meth:`momentum_covariance`.

	Attributes
	----------
	mean : xp.ndarray
		As passed, at least 2-D (a single plane is stored as one row).
	covariance : xp.ndarray
		As passed, at least 3-D.
	moment_closure : MomentClosure
		The closure in force.
	wavelength : float or None
		Electron wavelength, for momentum quantities.

	Methods
	-------
	z()
		Plane positions.
	sigma(name)
		RMS width of one column.
	correlation(a, b)
		One covariance entry, per plane.
	emittance(axis)
		Transverse rms emittance.
	real_space_ellipse(index), angular_ellipse(index)
		Principal widths and axes of the position and angle blocks.
	momentum_covariance(index)
		The angular block scaled to transverse momentum.
	is_positive_semidefinite(tol)
		Whether every plane's covariance is physical.
	at(index)
		A single-plane beam.

	Raises
	------
	ValueError
		From :meth:`momentum_covariance` when no wavelength is known, and
		from :meth:`emittance` on an unknown axis.

	Related
	-------
	MomentClosure : What the beam records about how it was transported.
	assemblies.Microscope.covariance_beam : Builds one over a propagated column.
	assemblies.Microscope.beam_waists : The waist search this reports around.

	Notes
	-----
	This is a **view**, not a second store: the drivers keep writing ``.mu``
	and ``.covariance_matrix`` exactly as before, and a beam is built over
	them on request. Nothing here mutates the column.

	Examples
	--------
	>>> beam = scope.covariance_beam                       # doctest: +SKIP
	>>> beam.sigma('x')[-1], beam.emittance('x')[-1]       # doctest: +SKIP
	"""

	def __init__(self, mean:xp.ndarray, covariance:xp.ndarray,
				 moment_closure:MomentClosure=None, wavelength:float=None):
		"""Store the state, the closure it was produced under, and the wavelength.

		Parameters
		----------
		mean : xp.ndarray
			Mean state, one plane or many.
		covariance : xp.ndarray
			Covariance, matching ``mean``.
		moment_closure : MomentClosure, optional
			The closure in force, by default :class:`GaussianMomentClosure`.
		wavelength : float, optional
			Electron wavelength (metres), by default None.

		Returns
		-------
		None

		Raises
		------
		ValueError
			If ``mean`` and ``covariance`` disagree about the number of planes.
		"""
		mean = xp.atleast_2d(xp.asarray(mean, dtype=float))
		covariance = xp.asarray(covariance, dtype=float)
		if covariance.ndim == 2:
			covariance = covariance[None, ...]
		if len(mean) != len(covariance):
			raise ValueError(f"mean has {len(mean)} planes but covariance has "
							 f"{len(covariance)}; they must describe the same planes.")
		self.mean = mean
		self.covariance = covariance
		self.moment_closure = moment_closure if moment_closure is not None else GaussianMomentClosure()
		self.wavelength = wavelength

	def __len__(self) -> int:
		"""The number of planes carried.

		Returns
		-------
		int
			Plane count.

		Raises
		------
		None
		"""
		return len(self.mean)

	def __repr__(self) -> str:
		"""Plane count, closure, and the final rms widths.

		Returns
		-------
		str
			A one-line summary.

		Raises
		------
		None
		"""
		return (f"CovarianceBeam({len(self)} planes, closure={self.moment_closure.name}, "
				f"final sigma_x={self.sigma('x')[-1]:.3e} m, "
				f"sigma_xt={self.sigma('xt')[-1]:.3e} rad)")

	def _index(self, name:str) -> int:
		"""Column index for a convention name.

		Parameters
		----------
		name : str
			A name in :data:`elements.convention`.

		Returns
		-------
		int
			Its column index.

		Raises
		------
		ValueError
			If the name is not in the convention.

		Related
		-------
		elements.columnByName : The lookup this defers to.
		"""
		from .elements import columnByName
		return columnByName(name)

	def z(self) -> xp.ndarray:
		"""Plane positions along the column.

		Returns
		-------
		xp.ndarray
			The ``z`` column of the mean, shape ``(n_planes,)``.

		Raises
		------
		None

		Related
		-------
		sigma : Quantities to plot against this.
		"""
		return self.mean[:, self._index('z')]

	def sigma(self, name:Literal['x','xt','y','yt','E']) -> xp.ndarray:
		r"""RMS width of one column, per plane.

		Parameters
		----------
		name : {'x', 'xt', 'y', 'yt', 'E'}
			Which column, by :data:`elements.convention` name.

		Returns
		-------
		xp.ndarray
			:math:`\sqrt{\Sigma_{nn}}`, shape ``(n_planes,)``.

		Raises
		------
		ValueError
			If ``name`` is not in the convention.

		Related
		-------
		correlation : The off-diagonal entries.
		emittance : The width that transport cannot reduce.
		postprocessing.beam_widths : The array form, for ``x`` and ``y`` only;
			this also covers the angle and energy columns.

		Notes
		-----
		Clipped at zero before the square root, so a covariance driven very
		slightly negative by rounding reports 0 rather than a NaN.
		"""
		i = self._index(name)
		return xp.sqrt(xp.clip(self.covariance[:, i, i], 0.0, None))

	def correlation(self, a:str, b:str) -> xp.ndarray:
		r"""One covariance entry, per plane.

		Parameters
		----------
		a, b : str
			Column names, e.g. ``'x'`` and ``'xt'`` for
			:math:`\sigma_{x,xt}`.

		Returns
		-------
		xp.ndarray
			``Sigma[a, b]``, shape ``(n_planes,)``.

		Raises
		------
		ValueError
			If either name is not in the convention.

		Related
		-------
		sigma : The diagonal.

		Notes
		-----
		This is a covariance, not a squared quantity: writing it
		:math:`\sigma_{x,xt}^2` would be wrong, and it is signed — the sign
		says whether the beam is converging or diverging.
		"""
		return self.covariance[:, self._index(a), self._index(b)]

	def emittance(self, axis:Literal['x','y']='x') -> xp.ndarray:
		r"""Transverse rms emittance, per plane.

		.. math::

			\epsilon_x = \sqrt{\sigma_x^2\sigma_{xt}^2 - \sigma_{x,xt}^2}

		The phase-space area the beam occupies, and the width no linear optics
		can focus away. It is invariant under ideal transport — which makes it
		the sharpest available check that a column is behaving, and the
		sharpest available signal that an aberration has done something
		irreversible.

		Parameters
		----------
		axis : {'x', 'y'}, optional
			Which transverse plane, by default ``'x'``.

		Returns
		-------
		xp.ndarray
			Emittance in m·rad, shape ``(n_planes,)``.

		Raises
		------
		ValueError
			If ``axis`` is not ``'x'`` or ``'y'``.

		Related
		-------
		postprocessing.emittance : The array form this delegates to, so the
			formula has one implementation.
		assemblies.Microscope.beam_waists : Reports the same invariant once.

		Notes
		-----
		A nonlinear element raises this, and the rise is real: projected rms
		emittance growth is how aberration damage shows up in a moments-only
		description.

		Examples
		--------
		>>> beam.emittance('x')[0]                          # doctest: +SKIP
		2.5e-10
		"""
		if axis not in ('x', 'y'):
			raise ValueError(f"axis must be 'x' or 'y', not {axis!r}.")
		from .postprocessing import emittance as _emittance		# lazy: it imports elements
		return _emittance(self.covariance)[:, 0 if axis == 'x' else 1]

	def _block_ellipse(self, names:Sequence[str], index) -> tuple:
		"""Principal widths and axes of a 2x2 covariance block.

		Parameters
		----------
		names : Sequence of str
			The two column names spanning the block.
		index : int or None
			Which plane; ``None`` returns every plane.

		Returns
		-------
		tuple of xp.ndarray
			``(widths, axes)``. For one plane, ``widths`` has shape ``(2,)``
			(rms, largest last) and ``axes`` shape ``(2, 2)`` with the
			eigenvectors as columns. For every plane, both gain a leading
			plane axis.

		Raises
		------
		ValueError
			If a name is not in the convention.

		Related
		-------
		real_space_ellipse, angular_ellipse : The two callers.

		Notes
		-----
		Uses ``eigh``, so the block's symmetry is assumed rather than
		enforced; eigenvalues come back ascending, and are clipped at zero
		before the root.
		"""
		i, j = self._index(names[0]), self._index(names[1])
		sub = self.covariance[:, [i, j], :][:, :, [i, j]]
		if index is not None:
			sub = sub[index][None, ...]
		values, vectors = xp.linalg.eigh(sub)
		widths = xp.sqrt(xp.clip(values, 0.0, None))
		if index is not None:
			return widths[0], vectors[0]
		return widths, vectors

	def real_space_ellipse(self, index=None) -> tuple:
		r"""Principal rms widths and axes of the real-space block.

		The block

		.. math::

			\Sigma_{rr} = \begin{pmatrix}\sigma_x^2 & \sigma_{xy}\\
			                             \sigma_{xy} & \sigma_y^2\end{pmatrix}

		reduced to its eigenbasis. Reporting this rather than
		:math:`\sigma_x` and :math:`\sigma_y` separately is the difference
		between describing the resolution and describing the coordinate
		system: a beam blurred along a diagonal has no larger
		:math:`\sigma_x` than one blurred isotropically.

		Parameters
		----------
		index : int, optional
			Which plane, by default None (every plane).

		Returns
		-------
		tuple of xp.ndarray
			``(widths, axes)`` — rms widths in metres, ascending, and the
			corresponding principal axes as columns.

		Raises
		------
		None

		Related
		-------
		angular_ellipse : The same for angles.
		momentum_covariance : The angular block in momentum units.

		Examples
		--------
		>>> widths, axes = beam.real_space_ellipse(index=-1)   # doctest: +SKIP
		"""
		return self._block_ellipse(('x', 'y'), index)

	def angular_ellipse(self, index=None) -> tuple:
		r"""Principal rms widths and axes of the angular block.

		The angular analog of :meth:`real_space_ellipse`, over

		.. math::

			\Sigma_{uu} = \begin{pmatrix}\sigma_{xt}^2 & \sigma_{xt,yt}\\
			                             \sigma_{xt,yt} & \sigma_{yt}^2\end{pmatrix}

		Parameters
		----------
		index : int, optional
			Which plane, by default None (every plane).

		Returns
		-------
		tuple of xp.ndarray
			``(widths, axes)`` — rms widths in radians, ascending, and the
			principal axes as columns.

		Raises
		------
		None

		Related
		-------
		momentum_covariance : The same block scaled by :math:`k_0`.
		"""
		return self._block_ellipse(('xt', 'yt'), index)

	def momentum_covariance(self, index=None) -> xp.ndarray:
		r"""The angular block scaled to transverse momentum.

		.. math::

			\Sigma_{kk,\perp} = k_0^2\,\Sigma_{uu}, \qquad k_0 = 2\pi/\lambda

		so the angular principal widths map directly into momentum-resolution
		principal widths — the quantity a diffraction or spectrometer plane is
		actually read in.

		Parameters
		----------
		index : int, optional
			Which plane, by default None (every plane).

		Returns
		-------
		xp.ndarray
			Shape ``(2, 2)`` for one plane, ``(n_planes, 2, 2)`` otherwise, in
			units of 1/m².

		Raises
		------
		ValueError
			If the beam carries no wavelength, since :math:`k_0` is then
			undefined. Build the beam from a column whose ``Source`` has a
			``voltage``.

		Related
		-------
		angular_ellipse : The unscaled block.

		Notes
		-----
		Purely a change of units: the eigenvectors are those of
		:meth:`angular_ellipse` and the eigenvalues scale by :math:`k_0^2`.
		"""
		if not self.wavelength:
			raise ValueError("momentum_covariance needs a wavelength; propagate a column "
							 "whose Source has a voltage, or pass wavelength= when "
							 "constructing the CovarianceBeam.")
		ixt, iyt = self._index('xt'), self._index('yt')
		k0 = 2.0 * xp.pi / float(self.wavelength)
		block = self.covariance[:, [ixt, iyt], :][:, :, [ixt, iyt]] * k0**2
		return block[index] if index is not None else block

	def is_positive_semidefinite(self, tol:float=1e-12) -> bool:
		"""Whether every plane's covariance is a physically possible one.

		A covariance matrix must be symmetric positive semidefinite; a
		nonlinear update evaluated through a closure is not guaranteed to
		preserve that, so this is the check that says whether a closure has
		been pushed past where it is meaningful.

		Parameters
		----------
		tol : float, optional
			Relative tolerance on the most negative eigenvalue, scaled by the
			largest, by default 1e-12.

		Returns
		-------
		bool
			True when every plane passes.

		Raises
		------
		None

		Related
		-------
		MomentClosure : What can break this.

		Notes
		-----
		The tolerance is relative because the transverse and energy columns
		differ by many orders of magnitude, so an absolute floor would be
		meaningless for one of them.
		"""
		for S in self.covariance:
			values = xp.linalg.eigvalsh(0.5 * (S + S.T))
			scale = float(xp.max(xp.abs(values))) or 1.0
			if float(xp.min(values)) < -tol * scale:
				return False
		return True

	def at(self, index:int) -> 'CovarianceBeam':
		"""A single-plane beam, carrying the same closure and wavelength.

		Parameters
		----------
		index : int
			Plane index; negative indices count from the end.

		Returns
		-------
		CovarianceBeam
			A one-plane beam.

		Raises
		------
		IndexError
			If the index is out of range.

		Related
		-------
		__len__ : The number of planes available.
		"""
		return CovarianceBeam(self.mean[index], self.covariance[index],
							  self.moment_closure, self.wavelength)
