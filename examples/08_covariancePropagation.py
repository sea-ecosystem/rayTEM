"""Covariance propagation through the standard column, and where the resolution goes.

This example never traces a ray for its answer. It starts from a beam described
only by its first two moments -- a mean state and a covariance over
``convention = ["x","xt","y","yt","z","E"]`` -- transports those moments through
``basic_column``, and asks a question a single probe diameter cannot answer:
**how much of the final resolution is set by the source, how much by the
pre-specimen objective, how much by the post-specimen objective, how much is
chromatic, and how much appears only when both objectives are aberrated?**

The four aberration configurations share one source boundary condition and one
column state, and differ only in which objective carries aberrations:

1. ``ideal``  -- both objectives perfect. The baseline, and the exactness check.
2. ``OL1``    -- pre-specimen aberrations only.
3. ``OL2``    -- post-specimen aberrations only. Its input has never passed
   through a nonlinear element, so it is the clean test of the covariance
   machinery.
4. ``both``   -- the critical case. After OL1 the beam is provably non-Gaussian,
   but only its mean and covariance are carried forward, so OL2's update needs
   a **closure assumption**. This example states plainly that it uses
   :func:`elements._gaussian_moment` there, prints how much
   non-Gaussianity that assumption discards, and measures how far from additive
   the combined result actually is.

Each runs twice, achromatic and chromatic, so the chromatic share separates
from the geometric share.

**The operating point.** The column is loaded in its stored
``high-convergent-image`` state, whose nominal (aberration-free)
semi-convergence angle at the specimen is exactly 30 mrad -- the same alpha
``examples/06`` and ``examples/07`` work at. That angle is only sensible for an
**aberration-corrected** objective, and that is the point: at 30 mrad an
uncorrected millimetre-scale ``Cs`` would swamp everything and the four cases
would be a formality. With ``Cs`` corrected to micrometres, spherical and
chromatic land within a factor of two of each other and of the source, which is
the regime where the question is worth asking -- and it is the regime a real
corrected instrument is in, where ``Cc`` is uncorrected and becomes the limit.

**The source is a cold field emitter**, stated once in :data:`SOURCE_SIZE`,
:data:`SOURCE_ANGLE` and :data:`ENERGY_SPREAD`: a few-nm virtual source and a
few tenths of an eV. The emission half-angle is not guessed -- it is solved so
the specimen-plane convergence is exactly :data:`ALPHA_TARGET`, because alpha
here is set by the column's demagnification of the source fan, and a source
size chosen without re-solving would silently change the operating point.

**The measure that matters is emittance, not width.** A width says nothing on
its own -- a beam is wide at the detector because it was focused at the
specimen. The rms emittance ``eps = sqrt(sigma_x^2 sigma_xt^2 - sigma_x,xt^2)``
is the phase-space area, invariant under any ideal linear transport and
*raised* by a nonlinear one. It is therefore both the sharpest check that the
ideal column is behaving and the sharpest signal that an aberration has done
something no downstream lens can undo. Because each nonlinear element adds its
own area, ``eps^2`` growth is very nearly a budget that sums -- and the amount
by which it fails to sum is exactly the OL1-OL2 coupling.

What is printed: a per-configuration block giving, at the specimen and at the
detector, the rms widths, the position-angle correlation, the emittance and its
growth over ideal, and the principal axes of the real-space and angular
ellipses; then the closure-validity numbers; then an emittance budget
attributing ``eps^2`` growth to OL1 spherical, OL2 spherical, chromatic, and a
non-additive remainder.

What is plotted (panels A-E):

A. ``sigma_x(z)`` through the column, all configurations.
B. ``sigma_xt(z)``.
C. ``eps_x(z)`` -- flat for the ideal column, stepping up at each aberrated
   element and then flat again through the linear transport between them. That
   staircase is the diagnostic: a step means an element added phase-space area,
   and a slope between steps would mean a bug.
C2. Emittance growth at the detector, by configuration.
D. Real-space covariance ellipses at the specimen and the detector.
E. Angular ellipses at the same planes.

OL1 and OL2 are marked on A-C so any emittance step can be read off against the
element that produced it.

Run: ``MPLBACKEND=Agg python examples/08_covariancePropagation.py``
(writes ``figs/``; add ``--no-figure`` to print the tables only).

Related
-------
postprocessing.emittance, postprocessing.resolution_ellipses : Read the result.
elements._gaussian_moment : The closure assumption Case 4 tests.
elements.Element.propagate_moments : The transport step.
microscopes.basic_column : Builds the column this drives.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(1, "../")
from pySEA.rayTEM.aberrations import Aberrations
from pySEA.rayTEM.assemblies import Microscope
from pySEA.rayTEM.microscopes.basic_column import build_basic_column
from pySEA.rayTEM.elements import columnByName, convention, suspended_aberrations
from pySEA.rayTEM.postprocessing import beam_widths, emittance, resolution_ellipses
from pySEA.rayTEM.seashells import as_ndarray

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The stored column state this study runs in: a convergent probe at
#: :data:`ALPHA_TARGET`, solved by ``examples/07`` and saved through the
#: column's own settings mechanism.
COLUMN_STATE = "basic_column - high-convergent-image"
#: Nominal (aberration-free) semi-convergence half-angle at the specimen (rad).
ALPHA_TARGET = 30e-3

#: Cold-field-emitter virtual source rms size (m).
SOURCE_SIZE = 3e-9
#: Source rms energy spread in **kilovolts**, matching the E column. 3e-4 kV is
#: 0.3 eV -- a cold FEG. A Schottky emitter is two to three times this, a
#: thermionic gun ten.
ENERGY_SPREAD = 3e-4

#: Pre-specimen objective, on a 3 mm focal length. ``C30`` is **corrected** to
#: micrometres, which is what makes a 30 mrad aperture sensible; ``Cc`` is not,
#: because a spherical corrector does not touch it -- which is exactly why
#: chromatic is a live term in a corrected instrument.
OL1_CS, OL1_CC = 4.5e-6, 1.2e-3
#: Post-specimen objective, on a 10 mm focal length.
OL2_CS, OL2_CC = 1.5e-5, 1.2e-2

#: The four aberration configurations, in the order they are reported.
CASES = ("ideal", "OL1", "OL2", "both")
#: Planes the tables and ellipses are read at.
LANDMARKS = ("sample", "detector")


def source_angle_for(alpha:float = ALPHA_TARGET, size:float = SOURCE_SIZE) -> float:
	"""Emission half-angle giving a specimen-plane convergence of ``alpha``.

	The convergence angle at the specimen is the column's demagnification of
	the source fan, so it depends on the source the fan is emitted from. The
	stored state was solved for the column's own 2.5 um gun; swapping in a
	few-nm cold field emitter shrinks the fan and with it alpha. Rather than
	quietly accept a different operating point -- or hardcode a number that
	rots the moment a lens moves -- this solves for the emission angle that
	restores it.

	Parameters
	----------
	alpha : float, optional
		Target semi-convergence half-angle at the specimen (rad), by default
		:data:`ALPHA_TARGET`.
	size : float, optional
		Virtual source rms size (m), by default :data:`SOURCE_SIZE`.

	Returns
	-------
	float
		The emission half-angle in radians.

	Raises
	------
	ValueError
		From ``brentq`` if the target is not bracketed by the search range,
		which means the column cannot reach that convergence at all.

	Related
	-------
	build_case : Uses this to configure the gun.
	assemblies.Microscope.convergence_angle_at : What is being solved against.

	Notes
	-----
	Alpha is linear in the emission angle here, because the condenser aperture
	is not the limiting stop for a source this small -- so the solve converges
	immediately and is really just a division. It is written as a solve anyway
	so that it stays correct if the aperture ever does bite.

	Examples
	--------
	>>> round(source_angle_for() * 1e6)                     # doctest: +SKIP
	259
	"""
	def measured(emission):
		scope = _state_column(size, emission)
		return scope.convergence_angle_at(scope.get_element_position("sample"))
	return brentq(lambda a: measured(a) - alpha, 1e-6, 1e-2, xtol=1e-12)


def _state_column(size:float, emission:float, energy_spread:float = ENERGY_SPREAD) -> Microscope:
	"""The stored column state with the cold-FEG gun applied.

	Parameters
	----------
	size : float
		Virtual source rms size (m).
	emission : float
		Emission half-angle (rad).
	energy_spread : float, optional
		Source rms energy spread (kV), by default :data:`ENERGY_SPREAD`.

	Returns
	-------
	assemblies.Microscope
		An unpropagated, ideal column in the stored state.

	Raises
	------
	FileNotFoundError
		If the stored state is missing from ``examples/settings/``.

	Related
	-------
	build_case : Adds the aberrations on top of this.

	Notes
	-----
	:meth:`assemblies.Microscope.load_setting` resolves ``settings/<name>.json``
	against the working directory, so this changes into the script's own
	directory for the load and changes back -- otherwise a caller running from
	anywhere else, the test suite included, would not find the stored state.
	"""
	scope = build_basic_column()
	previous = os.getcwd()
	try:									# load_setting resolves settings/ against the cwd
		os.chdir(_HERE)
		scope.load_setting(COLUMN_STATE)
	finally:
		os.chdir(previous)
	gun = scope.sections[0].elements[0]
	gun.size = (size, size)
	gun.angle = (emission, emission)
	gun.energy_spread = energy_spread
	return scope


#: Emission half-angle (rad) that puts the specimen-plane convergence at
#: :data:`ALPHA_TARGET` for a :data:`SOURCE_SIZE` virtual source. Solved once.
SOURCE_ANGLE = source_angle_for()


def build_case(case: str, chromatic: bool = False) -> Microscope:
	"""A standard column in the study state with one aberration configuration.

	Every configuration starts from the same stored column state and the same
	source boundary condition; only the objectives' aberration content
	differs. The sets are constructed fresh per lens on purpose -- an
	``Aberrations`` object is attached by reference, so handing one instance to
	both objectives would make them share a single mutable set.

	The chromatic coefficient rides *inside* the aberration set, under the name
	``'Cc'``, so one declaration carries everything each objective does beyond
	its transfer matrix.

	Parameters
	----------
	case : {'ideal', 'OL1', 'OL2', 'both'}
		Which objectives carry aberrations.
	chromatic : bool, optional
		Whether those objectives also carry their chromatic coefficients, by
		default False. The source energy spread is seeded either way, so an
		achromatic run and a chromatic one differ only in ``Cc``.

	Returns
	-------
	assemblies.Microscope
		A fresh column, not yet propagated.

	Raises
	------
	ValueError
		If ``case`` is not one of :data:`CASES`.

	Related
	-------
	CASES : The configurations.
	aberrations.CHROMATIC_TERM : The ``'Cc'`` entry this sets.

	Examples
	--------
	>>> scope = build_case('OL1')                          # doctest: +SKIP
	"""
	if case not in CASES:
		raise ValueError(f"unknown case {case!r}; expected one of {CASES}.")
	scope = _state_column(SOURCE_SIZE, SOURCE_ANGLE)
	for name, spherical, chrom in (("OL1", OL1_CS, OL1_CC), ("OL2", OL2_CS, OL2_CC)):
		terms = {}
		if case in (name, "both"):
			terms['C30'] = spherical
		if chromatic:
			terms['Cc'] = chrom
		if terms:
			scope[name].aberrations = Aberrations(terms)
	return scope


def propagate_case(case: str, chromatic: bool = False, closure=None) -> tuple:
	"""Propagate one configuration and return its column and landmark planes.

	Parameters
	----------
	case : {'ideal', 'OL1', 'OL2', 'both'}
		Which objectives carry aberrations.
	chromatic : bool, optional
		Whether the objectives are chromatic, by default False.
	closure : callable, optional
		``closure(Sigma, indices) -> float`` for the moments a nonlinear
		element needs above the second, by default the Gaussian one. Passing
		an alternative here is how the Case 4 assumption would be varied.

	Returns
	-------
	tuple
		``(scope, planes)`` -- the propagated column, whose covariance is on
		``scope.covariance_matrix`` as a calibrated ``Signal`` and whose means
		are on ``scope.mu``, and a ``{name: index}`` map for :data:`LANDMARKS`.

	Raises
	------
	ValueError
		If ``case`` is not one of :data:`CASES`.

	Related
	-------
	build_case : Builds the column.
	postprocessing.emittance : Reads the result.

	Notes
	-----
	Landmark indices are found by matching each element's z against the logged
	plane positions rather than by hardcoding an index, so inserting an element
	upstream cannot silently move what is being reported.
	"""
	scope = build_case(case, chromatic)
	scope.propagate_moments(closure=closure)
	z = scope.mu[:, columnByName('z')]
	planes = {name: int(np.argmin(np.abs(z - scope.get_element_position(name))))
			  for name in LANDMARKS}
	return scope, planes


def resolution_report(scope: Microscope, planes: dict,
					  reference: Microscope = None) -> str:
	"""The resolution quantities at the landmark planes, as a printable block.

	Reads the propagated covariance through the three envelope-mode helpers in
	:mod:`postprocessing` -- :func:`postprocessing.beam_widths`,
	:func:`postprocessing.emittance` and
	:func:`postprocessing.resolution_ellipses` -- and the position-angle
	correlations straight off the matrix. Not a single probe diameter: the
	correlations say whether a plane is a waist or a crossover, the emittance
	says how much of the width is irreducible, and the principal axes say
	whether "the resolution" is even isotropic.

	Parameters
	----------
	scope : assemblies.Microscope
		A propagated column.
	planes : dict
		``{name: index}`` from :func:`propagate_case`.
	reference : assemblies.Microscope, optional
		The ideal column to quote emittance growth against, by default None
		(no growth column).

	Returns
	-------
	str
		A multi-line block, no trailing newline.

	Raises
	------
	None

	Related
	-------
	postprocessing.emittance : The invariant the growth column reports.
	emittance_budget : The attribution this feeds.

	Notes
	-----
	Every width here is an **rms** value. The textbook spherical and chromatic
	disc diameters are larger by a factor of two and, as a full width at half
	maximum, by a further 2.355; nothing here silently mixes the two.
	"""
	cov = as_ndarray(scope.covariance_matrix)
	z = scope.mu[:, columnByName('z')]
	widths, eps = beam_widths(cov), emittance(cov)
	real_w, real_v = resolution_ellipses(cov, 'real')
	ang_w, _ = resolution_ellipses(cov, 'angular')
	ix, ixt = columnByName('x'), columnByName('xt')
	iy, iyt = columnByName('y'), columnByName('yt')
	wavelength = scope.sections[0].elements[0].wavelength
	lines = []
	for name, i in planes.items():
		lines.append(f"  {name} (z = {z[i] * 1e3:.4f} mm)")
		lines.append(f"    sigma_x  {widths[i, 0]:.6e} m      "
					 f"sigma_y  {widths[i, 1]:.6e} m")
		lines.append(f"    sigma_xt {np.sqrt(cov[i, ixt, ixt]):.6e} rad    "
					 f"sigma_yt {np.sqrt(cov[i, iyt, iyt]):.6e} rad")
		lines.append(f"    sigma_x,xt {cov[i, ix, ixt]:+.6e} m.rad  "
					 f"sigma_y,yt {cov[i, iy, iyt]:+.6e} m.rad")
		growth = ""
		if reference is not None:
			ideal = emittance(as_ndarray(reference.covariance_matrix))
			growth = f"   ({eps[i, 0] / ideal[i, 0]:.4f}x ideal)"
		lines.append(f"    eps_x    {eps[i, 0]:.6e} m.rad  "
					 f"eps_y    {eps[i, 1]:.6e} m.rad{growth}")
		angle = np.degrees(np.arctan2(real_v[i][1, 1], real_v[i][0, 1]))
		lines.append(f"    real-space principal rms  {real_w[i, 0]:.6e}, "
					 f"{real_w[i, 1]:.6e} m (major axis {angle:+.2f} deg)")
		lines.append(f"    angular principal rms     {ang_w[i, 0]:.6e}, "
					 f"{ang_w[i, 1]:.6e} rad")
		if wavelength:
			k0 = 2.0 * np.pi / wavelength
			lines.append(f"    momentum principal rms    {k0 * ang_w[i, 0]:.6e}, "
						 f"{k0 * ang_w[i, 1]:.6e} 1/m")
	return "\n".join(lines)


def emittance_budget(chromatic: bool = False) -> dict:
	"""Attribute squared-emittance growth at the detector to its sources.

	Each nonlinear element adds phase-space area, so ``eps^2`` growth over the
	ideal column behaves very nearly as a budget that sums. This runs the
	configurations needed to separate the terms and reports the amount by which
	the sum fails — which is precisely the OL1-OL2 coupling the plan warns
	against assuming away.

	Parameters
	----------
	chromatic : bool, optional
		Whether to include the chromatic contribution, by default False.

	Returns
	-------
	dict
		``{'ideal', 'OL1', 'OL2', 'both', 'sum', 'coupling', 'chromatic'}`` —
		squared-emittance growths in (m.rad)^2, with ``coupling`` the combined
		growth minus the sum of the single-objective growths, and
		``chromatic`` the growth of the ideal column once the objectives are
		made chromatic (absent when ``chromatic`` is False).

	Raises
	------
	None

	Related
	-------
	postprocessing.emittance : The quantity being differenced.
	resolution_report : The per-plane detail behind these totals.

	Notes
	-----
	The differences are taken at the detector, where every element has acted.
	Taking them at the specimen would miss OL2 entirely, which is the point of
	separating the two objectives in the first place.
	"""
	def eps2(case, chrom):
		scope, planes = propagate_case(case, chrom)
		return float(emittance(as_ndarray(scope.covariance_matrix))[planes['detector'], 0])**2

	base = eps2('ideal', False)
	one = eps2('OL1', False) - base
	two = eps2('OL2', False) - base
	both = eps2('both', False) - base
	out = {'ideal': base, 'OL1': one, 'OL2': two, 'both': both,
		   'sum': one + two, 'coupling': both - (one + two)}
	if chromatic:
		out['chromatic'] = eps2('ideal', True) - base
	return out


def closure_validity(case: str, chromatic: bool = False) -> dict:
	r"""How much non-Gaussianity the closure is being asked to ignore.

	Gaussian closure is exact for one aberrated element acting on a Gaussian
	beam. It becomes an approximation only when a *second* nonlinear element
	acts on what the first one distorted. The size of that approximation is
	governed by one number per element: the share of the angular variance the
	aberration contributes,

	.. math::

		f = \frac{\Sigma'_{\theta\theta} - \Sigma^{ideal}_{\theta\theta}}
		         {\Sigma'_{\theta\theta}}

	because a cubic kick on a centered Gaussian leaves excess kurtosis
	:math:`\gamma_2 = 27 f^2` in the angular coordinate. That identity is what
	makes the closure auditable rather than a matter of faith: there is no
	regime in which the aberration matters to :math:`\Sigma` but the
	non-Gaussianity it induces does not, since both are set by the same ``f``.
	Small ``f`` is the licence to close; the example prints it rather than
	assuming it.

	Parameters
	----------
	case : {'ideal', 'OL1', 'OL2', 'both'}
		Which configuration to audit.
	chromatic : bool, optional
		Whether the objectives are chromatic, by default False.

	Returns
	-------
	dict
		``{element_name: {'f': float, 'excess_kurtosis': float}}`` for each
		aberrated objective; empty for the ideal case.

	Raises
	------
	None

	Related
	-------
	elements._gaussian_moment : The assumption being audited.
	emittance_budget : The coupling term whose accuracy this bounds.

	Notes
	-----
	The entrance state is taken from the last logged plane at or before the
	element, which for both objectives is the drift that ends on its entrance
	face, so it is the exact entrance rather than an interpolation.

	``f`` is signed: an aberration whose kick anticorrelates with the existing
	angular spread *reduces* the angular variance, and OL2 does exactly that
	here. Only its magnitude enters the kurtosis.
	"""
	names = {'ideal': (), 'OL1': ('OL1',), 'OL2': ('OL2',), 'both': ('OL1', 'OL2')}[case]
	scope = build_case(case, chromatic)
	scope.propagate_moments()
	z = scope.mu[:, columnByName('z')]
	cov = as_ndarray(scope.covariance_matrix)
	ixt = columnByName('xt')
	out = {}
	for name in names:
		ze = scope.get_element_position(name)
		j = int(np.max(np.nonzero(z <= ze + 1e-12)[0]))
		element = scope[name]
		_, aberrated = element.propagate_moments(scope.mu[j], cov[j])
		with suspended_aberrations([element]):
			_, ideal = element.propagate_moments(scope.mu[j], cov[j])
		f = float((aberrated[ixt, ixt] - ideal[ixt, ixt]) / aberrated[ixt, ixt])
		out[name] = {'f': f, 'excess_kurtosis': 27.0 * f * f}
	return out


def figure(results: dict, filename: str) -> None:
	"""Draw panels A-E and write them to ``filename``.

	Parameters
	----------
	results : dict
		``{case: (scope, planes)}`` for every entry of :data:`CASES`.
	filename : str
		Where to write the PNG. Its directory is created if absent.

	Returns
	-------
	None

	Raises
	------
	None

	Related
	-------
	propagate_case : Produces the entries of ``results``.

	Notes
	-----
	matplotlib is imported here rather than at module scope so importing this
	example for its numbers -- which the tests do -- needs no display and no
	backend.

	The ellipse panels carry an annotation giving each case's fractional
	deviation from ideal, because the angular outlines differ by parts in
	:math:`10^6` and would otherwise sit on top of one another and read as a
	plotting failure rather than as the result they are. The real-space
	outlines at this aperture differ by a factor of several and need no such
	help -- which is itself the point: what an aberration does to the *width*
	depends entirely on where you look, while what it does to the emittance
	does not.
	"""
	import matplotlib.pyplot as plt

	os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
	fig, axes = plt.subplots(2, 4, figsize=(21, 9))
	styles = {'ideal': ("0.25", "-", 3.2), 'OL1': ("tab:blue", "--", 2.2),
			  'OL2': ("tab:orange", "-.", 1.8), 'both': ("tab:red", ":", 1.4)}
	scope = build_case('ideal')
	marks = {name: scope.get_element_position(name) * 1e3 for name in ("OL1", "OL2")}

	ixt = columnByName('xt')
	panels = [("A: real-space width", "sigma_x (nm)",
			   lambda c: beam_widths(c)[:, 0] * 1e9),
			  ("B: angular width", "sigma_xt (mrad)",
			   lambda c: np.sqrt(c[:, ixt, ixt]) * 1e3),
			  ("C: emittance", "eps_x (m.rad)", lambda c: emittance(c)[:, 0])]
	for ax, (title, ylab, get) in zip(axes[0], panels):
		for case in CASES:
			color, dash, width = styles[case]
			scope, _ = results[case]
			z = scope.mu[:, columnByName('z')] * 1e3
			ax.plot(z, get(as_ndarray(scope.covariance_matrix)),
					color=color, ls=dash, lw=width, label=case)
		for name, zm in marks.items():
			ax.axvline(zm, color="0.75", lw=0.9, zorder=0)
			ax.annotate(name, (zm, 0.02), xycoords=("data", "axes fraction"),
						fontsize=8, color="0.4", rotation=90)
		ax.set_xlabel("z (mm)")
		ax.set_ylabel(ylab)
		ax.set_title(title)
		ax.set_yscale("log")
		ax.legend(fontsize=8)

	ax = axes[0, 3]
	def detector_eps(case):
		scope, planes = results[case]
		return emittance(as_ndarray(scope.covariance_matrix))[planes['detector'], 0]
	growth = [detector_eps(c) / detector_eps('ideal') for c in CASES]
	ax.bar(range(len(CASES)), growth,
		   color=[styles[c][0] for c in CASES])
	ax.set_xticks(range(len(CASES)))
	ax.set_xticklabels(CASES)
	ax.set_ylabel("eps_x / eps_x(ideal) at the detector")
	ax.set_title("C2: emittance growth")
	ax.set_yscale("log")
	for i, g in enumerate(growth):
		ax.annotate(f"{g:.3g}x", (i, g), ha="center", va="bottom", fontsize=8)

	def draw(ax, plane, kind, scale, unit):
		"""Overlay every case's covariance ellipse at one plane."""
		def ellipse(case):
			scope, planes = results[case]
			w, v = resolution_ellipses(as_ndarray(scope.covariance_matrix), kind)
			return w[planes[plane]], v[planes[plane]]
		ideal_w, _ = ellipse('ideal')
		for case in CASES:
			color, dash, width = styles[case]
			w, v = ellipse(case)
			t = np.linspace(0, 2 * np.pi, 361)
			pts = v @ np.vstack([w[0] * np.cos(t), w[1] * np.sin(t)])
			ax.plot(pts[0] * scale, pts[1] * scale, color=color, ls=dash,
					lw=width, label=case)
		deltas = [f"{case} {(ellipse(case)[0][1] / ideal_w[1] - 1.0):+.2e}"
				  for case in CASES[1:]]
		ax.annotate("vs ideal (major axis):\n" + "\n".join(deltas),
					(0.02, 0.02), xycoords="axes fraction", fontsize=7, va="bottom")
		ax.set_aspect("equal")
		ax.set_xlabel(f"{'x' if kind == 'real' else 'xt'} ({unit})")
		ax.set_ylabel(f"{'y' if kind == 'real' else 'yt'} ({unit})")
		ax.legend(fontsize=7, loc="upper right")

	for col, plane in enumerate(LANDMARKS):
		draw(axes[1, col], plane, 'real', 1e9, "nm")
		axes[1, col].set_title(f"D{col + 1}: real-space ellipse at {plane}")
		draw(axes[1, col + 2], plane, 'angular', 1e3, "mrad")
		axes[1, col + 2].set_title(f"E{col + 1}: angular ellipse at {plane}")

	fig.suptitle(f"covariance propagation through basic_column at "
				 f"{ALPHA_TARGET * 1e3:.0f} mrad: where the resolution goes")
	fig.tight_layout()
	fig.savefig(filename, dpi=140, bbox_inches="tight")
	plt.close(fig)


def run(make_figure: bool = True, figdir: str = "figs") -> dict:
	"""Run every configuration, print the tables, and optionally draw the figure.

	Parameters
	----------
	make_figure : bool, optional
		Whether to draw panels A-E, by default True.
	figdir : str, optional
		Directory for the figure, by default ``'figs'``.

	Returns
	-------
	dict
		``{'results': {case: (scope, planes)}, 'budget': ..., 'budget_chromatic': ...}``
		so a caller -- or a test -- can assert on the same numbers that were
		printed.

	Raises
	------
	None

	Related
	-------
	emittance_budget : The attribution printed at the end.
	resolution_report : The per-configuration blocks.
	"""
	results = {case: propagate_case(case) for case in CASES}
	reference = results['ideal'][0]

	alpha = build_case('ideal')
	alpha = alpha.convergence_angle_at(alpha.get_element_position('sample'))
	print(f"column state: {COLUMN_STATE!r}, nominal alpha at the specimen "
		  f"{alpha * 1e3:.3f} mrad")
	print(f"source: cold FEG, {SOURCE_SIZE * 1e9:.1f} nm rms virtual size, "
		  f"{SOURCE_ANGLE * 1e6:.1f} urad emission, "
		  f"{ENERGY_SPREAD * 1e3:.2f} eV rms energy spread")
	print(f"OL1 (f = 3 mm):  Cs = {OL1_CS * 1e6:.2f} um (corrected), "
		  f"Cc = {OL1_CC * 1e3:.2f} mm (not)")
	print(f"OL2 (f = 10 mm): Cs = {OL2_CS * 1e6:.2f} um (corrected), "
		  f"Cc = {OL2_CC * 1e3:.2f} mm (not)")
	print(f"closed forms at this alpha:  Cs*a^3 = {OL1_CS * alpha**3 * 1e9:.3f} nm, "
		  f"Cc*a*dE/E = {OL1_CC * alpha * (ENERGY_SPREAD / 200.0) * 1e9:.3f} nm")
	print("closure in force at every nonlinear element: Gaussian (Isserlis)")
	for case in CASES:
		scope, planes = results[case]
		print(f"\n=== {case} ===")
		print(resolution_report(scope, planes, reference if case != 'ideal' else None))

	print("\n=== closure validity ===")
	for case in ('OL1', 'OL2', 'both'):
		for name, v in closure_validity(case).items():
			print(f"  {case:5s} {name}: aberration share of the angular variance "
				  f"f = {v['f']:+.4e}, so the beam leaving it carries excess "
				  f"kurtosis 27 f^2 = {v['excess_kurtosis']:.3e}")
	print("  The closure asserts that excess kurtosis is zero. It is exact for one")
	print("  aberrated element on a Gaussian beam; the numbers above bound how wrong")
	print("  it can be at the second one.")

	budget = emittance_budget(chromatic=False)
	chrom = emittance_budget(chromatic=True)
	print("\n=== emittance budget at the detector (growth in eps_x^2, (m.rad)^2) ===")
	print(f"  ideal eps^2            {budget['ideal']:.6e}")
	print(f"  OL1 spherical          {budget['OL1']:.6e}")
	print(f"  OL2 spherical          {budget['OL2']:.6e}")
	print(f"  chromatic (both)       {chrom['chromatic']:.6e}")
	print(f"  sum of the two spheres {budget['sum']:.6e}")
	print(f"  OL1 and OL2 together   {budget['both']:.6e}")
	print(f"  non-additive remainder {budget['coupling']:+.6e}   "
		  f"({budget['coupling'] / budget['sum'] * 100:+.3f} % of the sum)")
	share = abs(budget['coupling']) / max(abs(budget['sum']), 1e-300) * 100
	print("\n  The remainder is the OL1-OL2 coupling: it is what the Gaussian")
	print("  closure at OL2 is asked to supply, since OL1 has already made the")
	print(f"  distribution non-Gaussian. At {share:.2f} % of the summed growth the")
	print("  combined degradation is very nearly additive here -- a measured")
	print("  result, not an assumption, and one that Eq. 27 of the plan says may")
	print("  not be assumed in general.")
	if budget['OL2'] < 0:
		print("\n  Note OL2's contribution is NEGATIVE. That is physical, not a sign")
		print("  error: the projected emittance is a determinant, so a kick that")
		print("  anticorrelates with the position-angle correlation already present")
		print("  at that plane REDUCES the phase-space area the beam projects onto")
		print("  this axis. It does not undo OL1 -- the distribution is still more")
		print("  distorted -- which is precisely why 'more aberration' and 'worse")
		print("  emittance' are not the same statement.")
	def detector_eps(case):
		scope, planes = results[case]
		return emittance(as_ndarray(scope.covariance_matrix))[planes['detector'], 0]
	ideal_eps = np.sqrt(budget['ideal'])
	worst_eps = max(detector_eps(c) for c in CASES)
	f_ol1 = closure_validity('OL1')['OL1']['f']
	print(f"\n  Note the two measures disagree about magnitude, and both are right:")
	print(f"  OL1's aberration is a {f_ol1:.1e} share of the angular VARIANCE, yet the")
	print(f"  worst case multiplies the detector EMITTANCE by {worst_eps / ideal_eps:.2f}.")
	print("  Emittance is a determinant, so it responds to the part of the kick that")
	print("  is uncorrelated with position -- precisely the part no downstream lens")
	print("  can focus away. A width alone would have hidden this.")

	if make_figure:
		out = os.path.join(figdir, "08_covariance_propagation.png")
		figure(results, out)
		print(f"\n  figure: {out}")
	return {'results': results, 'budget': budget, 'budget_chromatic': chrom}


if __name__ == "__main__":
	run(make_figure="--no-figure" not in sys.argv)
