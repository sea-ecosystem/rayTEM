"""The eight canonical operating states of the standard column.

``basic_column.sea`` — the ecosystem's stock 200 kV column, with its 1 nA gun
and the condenser aperture CA after C1 — driven into all 2 x 2 x 2 = 8
operating states. Nothing about the geometry is touched: a state is purely a
set of lens strengths, each *solved* from a transfer-matrix condition the way
an operator chains crossovers, never hand-tuned:

1. **current** — C1 either images the gun crossover onto CA (``'high'``: the
   focused spot passes the hole whole) or is run weak (``'low'``: a broad
   beam hits CA and most of the current is cut).
2. **probe** — C2/C3/OL1 form either a **convergent** probe at the sample
   (target 30 mrad semi-angle, via a solved chain of intermediate crossovers)
   or a **nearly parallel** patch of illumination (``D = 0``: every ray from
   a single source point arrives parallel).
3. **detector** — PL1/PL2 (PL3/PL4 kept at their stored strengths) put either
   an **image** of the sample plane on the detector (``B = 0``) or a
   **diffraction** pattern (``A = 0``: arrival position reads arrival *angle*
   at the sample, scaled by the camera length ``B``).

Each solved state is saved through the column's own settings mechanism
(``Microscope.save_as_setting``) as ``settings/basic_column - <state>.json``,
so ``scope.load_setting("basic_column - high-convergent-image")`` restores it
onto a freshly loaded column.

The same solved column is then propagated with all three methods and they are
made to answer for each other:

- **rays** are drawn *on top of* the scaled-wave ``|psi(x, z)|``
  cross-section (one figure per state, in ``figs/``). The overlay traces the
  rays the *wave* cares about — the flat-phase family the scaled frame itself
  follows (zero-angle rays at fractions of the wave envelope), plus the rays
  that graze the CA edge — rather than the source's full incoherent fan,
  which is real but wider than the single coherent mode the wave carries and
  used to make the two look mismatched.
- **moments** print the transverse covariance at the gun exit, CA, sample,
  and detector;
- the conjugate planes (image and back-focal family of every lens) are
  measured four independent ways — traced rays, accumulated transfer matrix,
  the wave run's own crossovers, and the covariance waists — and tabulated
  with their deltas.

Physics worth noticing in the output: this repository's ray-path aperture is
a beam **rescale**, not a per-ray mask — ``Aperture.propagate_ray`` shrinks
every ray (positions and angles) by ``radius / xmax`` and attenuates the
intensity by the area ratio, while the **wave** path masks genuinely. So in
the low-current state the traced current drops to ``scale_x * scale_y`` of
the stated 1 nA, yet the condensers can still pump the (rescaled) angle back
up to the 30 mrad target — and at CA the figure shows the two models differ
by design: rays compress through the hole, the coherent wave is clipped by
it.

Run ``python 07_eightConfigurations.py`` from this directory; add ``--fast``
to skip the wave runs (the solves, settings files, and tables still print).

Related
-------
microscopes.basic_column : Builds the column this drives.
assemblies.Microscope.save_as_setting : How each state is stored.
assemblies.Microscope.conjugate_planes : The plane machinery used throughout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(1, "../")
from pySEA.rayTEM.assemblies import Microscope, load_microscope, _scaled_wave_cross_section
from pySEA.rayTEM.postprocessing import convert_to_rotating_reference_frame
from pySEA.rayTEM.elements import columnByName, convention

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SEA = os.path.join(_HERE, "..", "src", "pySEA", "rayTEM", "microscopes",
						"basic_column.sea")

#: The lenses a state may retune. PL3/PL4 stay at their stored strengths.
SOLVED_LENSES = ("C1", "C2", "C3", "OL1", "PL1", "PL2")
ALPHA_TARGET = 30e-3		# convergent-probe semi-angle at the sample (rad)
KC1_LOW = 8.0				# the deliberately weak C1 of the low-current state


def load_base() -> Microscope:
	"""Load a fresh copy of the standard column.

	Returns
	-------
	Microscope
		``basic_column.sea``, exactly as stored.

	Raises
	------
	OSError
		If the ``.sea`` file is missing (run ``microscopes/basic_column.py``
		to regenerate it).
	"""
	return load_microscope(BASE_SEA)


def apply_strengths(scope: Microscope, strengths: dict) -> Microscope:
	"""Set named lens strengths on a column, in place.

	Parameters
	----------
	scope : Microscope
		The column to configure.
	strengths : dict
		``{lens_name: strength}``; lenses not named keep what they have.

	Returns
	-------
	Microscope
		``scope``, for chaining.

	Raises
	------
	KeyError
		If a name is not an element of this column.
	"""
	scope.update_with_settings({n: {"strength": v} for n, v in strengths.items()})
	return scope


def _lens_map(scope: Microscope) -> dict:
	"""The column's solvable lenses by name.

	Parameters
	----------
	scope : Microscope
		The column.

	Returns
	-------
	dict
		``{name: Lens}`` for :data:`SOLVED_LENSES`.

	Raises
	------
	None
	"""
	return {e.name: e for s in scope.sections for e in s.elements
			if e.name in SOLVED_LENSES}


def sample_plane(scope: Microscope) -> float:
	"""The z of the sample plane: the middle of the element named ``sample``.

	The middle rather than the face, for the same reason
	:attr:`Microscope.convergence_angle` measures there — the state *entering*
	a plane is what is reported, so an element's start returns the beam before
	the gap.

	Parameters
	----------
	scope : Microscope
		The column.

	Returns
	-------
	float
		Absolute z in metres.

	Raises
	------
	KeyError
		If the column has no element named ``sample``.
	"""
	ele = [e for s in scope.sections for e in s.elements if e.name == "sample"][0]
	return scope.named_positions["sample"] + (getattr(ele, "length", 0) or 0) / 2


def block_between(scope: Microscope, z_from: float, z_to: float,
				  axis: str = 'x') -> np.ndarray:
	"""Accumulated 2x2 rotating-frame transfer block between two planes.

	Composes :meth:`Microscope._accumulate_blocks` from ``z_from`` up to
	``z_to``, entering an element body partway when ``z_to`` falls inside one.
	This is the matrix the solves below place conditions on: ``B = 0`` is
	point-to-point imaging, ``A = 0`` a diffraction (angle-to-point) plane,
	``D = 0`` point-to-parallel.

	Parameters
	----------
	scope : Microscope
		The column to walk.
	z_from, z_to : float
		Start and end planes (m, absolute, ``z_from <= z_to``).
	axis : {'x', 'y'}, optional
		Transverse axis, by default ``'x'`` (round optics: both identical).

	Returns
	-------
	numpy.ndarray
		The ``[[A, B], [C, D]]`` block from ``z_from`` to ``z_to``.

	Raises
	------
	None
	"""
	Mx, zx = np.eye(2), z_from
	for z0, ele, L, M, blk in scope._accumulate_blocks(axis=axis, reference=z_from):
		if z_to <= z0 + 1e-12:					# target in free space before ele
			return (np.array([[1.0, z_to - zx], [0.0, 1.0]]) @ Mx) if z_to > zx + 1e-15 else Mx
		if L > 0 and z_to < z0 + L - 1e-12:		# target inside this body
			return np.asarray(blk(z_to - z0), float) @ M
		Mx = np.asarray(ele.transfer_block(axis=axis), float) @ M
		zx = z0 + L
	return np.array([[1.0, z_to - zx], [0.0, 1.0]]) @ Mx


def _source_grid(scope: Microscope) -> np.ndarray:
	"""The column source's exact ray grid, as (x, xt, y, yt) rows.

	Mirrors what ``Source.rays()`` emits, so a per-ray *prediction* through
	the 2x2 blocks matches the traced rays to machine precision on an ideal
	column.

	Parameters
	----------
	scope : Microscope
		The column whose source defines the fan.

	Returns
	-------
	numpy.ndarray
		``(n_rays, 4)`` phase-space start points.

	Raises
	------
	None
	"""
	src = scope.sections[0].elements[0]
	xs = np.linspace(-src.size[0], src.size[0], int(src.np_xy[0]))
	ts = np.linspace(-src.angle[0], src.angle[0], int(src.na_xy[0]))
	return np.array([(x, t, y, ty) for x in xs for y in xs for t in ts for ty in ts])


def predict_probe(scope: Microscope) -> dict:
	"""Predict the probe the configured column forms, per-ray, from its blocks.

	Sends the source grid through the source->CA and source->sample blocks
	and applies CA the way the **ray path** applies it: this repository's
	``Aperture.propagate_ray`` does not mask individual rays — it *rescales*
	the whole beam (positions and angles, per axis, by ``radius / xmax``) and
	attenuates every intensity by the area ratio ``scale_x * scale_y``. The
	rescale commutes with the linear optics downstream, so the sample-plane
	quantities are simply the unmasked ones times the scales. (The wave path
	is different by design: there CA is a genuine mask.) Exact for the ideal
	column, and smooth in the lens strengths — good to solve against.

	Parameters
	----------
	scope : Microscope
		The configured column.

	Returns
	-------
	dict
		``current_fraction`` (the transmitted intensity fraction,
		``scale_x * scale_y``), ``alpha`` (max total angle at the sample,
		rad), ``size`` (max radius at the sample, m).

	Raises
	------
	None
	"""
	Z = scope.named_positions
	ca_r = scope["CA"].radius
	Ms = block_between(scope, 0.0, sample_plane(scope))
	Mca = block_between(scope, 0.0, Z["CA"])
	x0, t0, y0, ty0 = _source_grid(scope).T
	x_ca = Mca[0, 0] * x0 + Mca[0, 1] * t0
	y_ca = Mca[0, 0] * y0 + Mca[0, 1] * ty0
	sx = min(1.0, ca_r / float(np.max(x_ca)))	# amax of the SIGNED positions,
	sy = min(1.0, ca_r / float(np.max(y_ca)))	# mirroring Aperture._aperture_scales
	alpha = np.hypot(sx * (Ms[1, 0] * x0 + Ms[1, 1] * t0),
					 sy * (Ms[1, 0] * y0 + Ms[1, 1] * ty0))
	size = np.hypot(sx * (Ms[0, 0] * x0 + Ms[0, 1] * t0),
					sy * (Ms[0, 0] * y0 + Ms[0, 1] * ty0))
	return dict(current_fraction=float(sx * sy),
				alpha=float(alpha.max()), size=float(size.max()))


def _brackets(f, ks) -> list:
	"""Sign-change intervals of ``f`` sampled at ``ks``.

	Parameters
	----------
	f : callable
		Scalar function of one variable.
	ks : Sequence[float]
		Sample points, ascending.

	Returns
	-------
	list of tuple
		``(k_i, k_{i+1})`` pairs where ``f`` changes sign.

	Raises
	------
	None
	"""
	vv = [f(k) for k in ks]
	return [(ks[i], ks[i + 1]) for i in range(len(ks) - 1)
			if np.isfinite(vv[i]) and np.isfinite(vv[i + 1]) and vv[i] * vv[i + 1] < 0]


def solve_column(current: str, probe: str, detector: str) -> dict:
	"""Solve the lens strengths that put the standard column in one state.

	Works the way an operator does — as a chain of crossovers, each lens a 1D
	imaging condition, so every solve is a bracketed root find rather than a
	multidimensional shot in the dark:

	- **C1**: ``B(source→CA) = 0`` for ``'high'``; fixed weak
	  (:data:`KC1_LOW`) for ``'low'``.
	- **convergent**: C2 images the source to a crossover z2, C3 images z2 to
	  z3, OL1 images z3 onto the sample; (z2, z3) are then swept so the
	  *predicted per-ray* semi-angle hits 30 mrad — pulling z3 toward C3
	  raises the angle. When no (z2, z3) reaches the target, the
	  angle-maximizing pair is used and ``alpha_limited`` is set.
	- **parallel**: C2 images the source to z2 = 0.28 m, C3 zeroes
	  ``D(source→sample)`` with OL1 left as stored.
	- **image**: PL1 images the sample to an intermediate plane, PL2 images
	  that to the detector. **diffraction**: PL2 zeroes
	  ``A(sample→detector)``; among the PL1 values that admit a root, the one
	  with the longest camera length ``|B|`` is kept.

	Parameters
	----------
	current : {'high', 'low'}
		C1 state: gun crossover imaged onto CA, or a broad beam cut by it.
	probe : {'convergent', 'parallel'}
		What the condensers and OL1 deliver at the sample.
	detector : {'image', 'diffraction'}
		What PL1/PL2 deliver at the detector.

	Returns
	-------
	dict
		``strengths`` (solved ``{lens: K}`` — only the lenses this state
		retunes), ``predicted`` (from :func:`predict_probe`),
		``alpha_limited`` (bool), and the detector block ``M_det`` from the
		sample.

	Raises
	------
	ValueError
		If the state names are unknown, or a required imaging solve finds no
		solution (a geometry change broke the chain — the message names the
		failing stage).
	"""
	if current not in ("high", "low") or probe not in ("convergent", "parallel") \
			or detector not in ("image", "diffraction"):
		raise ValueError(f"unknown state ({current!r}, {probe!r}, {detector!r}); "
						 "expected ('high'|'low', 'convergent'|'parallel', 'image'|'diffraction').")
	scope = load_base()
	Z = scope.named_positions
	z_samp, z_det = sample_plane(scope), Z["detector"]
	lens = _lens_map(scope)
	stored = {n: l.strength for n, l in lens.items()}
	kmax = {n: np.pi / 2 / l.length - 1e-6 for n, l in lens.items()}
	solved = {}

	def sset(**kw):
		"""Reset every solvable lens to stored, then apply overrides."""
		for n, l in lens.items():
			l.strength = stored[n]
		for n, v in {**solved, **kw}.items():
			lens[n].strength = v

	def solve1(name, z_from, z_to, entry=(0, 1)):
		"""All strengths of ``name`` zeroing one block entry between planes."""
		def f(k):
			lens[name].strength = k
			return block_between(scope, z_from, z_to)[entry]
		roots = [brentq(f, *b) for b in _brackets(f, np.linspace(0.5, kmax[name], 50))]
		lens[name].strength = solved.get(name, stored[name])
		return roots

	# --- C1: the current state
	if current == "high":
		roots = solve1("C1", 0.0, Z["CA"])
		if not roots:
			raise ValueError("C1 cannot image the source onto CA -- the gun/CA "
							 "geometry changed.")
		solved["C1"] = roots[0]
	else:
		solved["C1"] = KC1_LOW
	sset()

	# --- the probe
	alpha_limited = False
	if probe == "convergent":
		def best_at(z2, z3):
			best = None
			for kc2 in solve1("C2", 0.0, z2):
				sset(C2=kc2)
				for kc3 in solve1("C3", z2, z3):
					sset(C2=kc2, C3=kc3)
					for kol in solve1("OL1", z3, z_samp):
						sset(C2=kc2, C3=kc3, OL1=kol)
						p = predict_probe(scope)
						if best is None or p["alpha"] > best[0]["alpha"]:
							best = (p, kc2, kc3, kol)
					sset(C2=kc2, C3=kc3)
				sset(C2=kc2)
			sset()
			return best
		# z3 toward C3 raises the reachable angle; z2 fine-tunes onto target
		z2s = np.linspace(0.235, 0.30, 9)
		hit = None
		for z3 in (0.36, 0.385, 0.41, 0.44):
			g = lambda z2: ((best_at(z2, z3) or ({"alpha": np.nan},))[0]["alpha"]) - ALPHA_TARGET
			br = _brackets(g, z2s)
			if br:
				z2 = brentq(g, *br[0], xtol=1e-7)
				hit = best_at(z2, z3)
				break
		if hit is None:						# CA has cut the phase space: take the max
			cands = [(z2, z3, best_at(z2, z3)) for z3 in (0.36, 0.41) for z2 in z2s[::2]]
			cands = [c for c in cands if c[2]]
			if not cands:
				raise ValueError("no convergent-probe chain solves at all -- the "
								 "condenser/objective geometry changed.")
			z2, z3, hit = max(cands, key=lambda c: c[2][0]["alpha"])
			alpha_limited = True
		p, solved["C2"], solved["C3"], solved["OL1"] = hit
	else:								# parallel: D(source->sample) = 0 via C3
		z2 = 0.28
		best = None
		for kc2 in solve1("C2", 0.0, z2):
			sset(C2=kc2)
			for kc3 in solve1("C3", 0.0, z_samp, entry=(1, 1)):
				sset(C2=kc2, C3=kc3)
				p = predict_probe(scope)
				if best is None or p["alpha"] < best[0]["alpha"]:
					best = (p, kc2, kc3)
			sset(C2=kc2)
		if best is None:
			raise ValueError("no parallel-probe solution: C3 cannot zero "
							 "D(source->sample) with C2 imaging the source to "
							 f"z2={z2} m.")
		p, solved["C2"], solved["C3"] = best
	sset()

	# --- the projectors (independent of the condensers: everything is
	#     downstream of the sample; PL3/PL4 keep their stored strengths)
	if detector == "image":
		done = False
		for zp in (0.74, 0.76, 0.72):			# intermediate image after PL1
			for kp1 in solve1("PL1", z_samp, zp):
				sset(PL1=kp1)
				for kp2 in solve1("PL2", zp, z_det):
					solved["PL1"], solved["PL2"] = kp1, kp2
					done = True
					break
				if done: break
			if done: break
		if not done:
			raise ValueError("no imaging projector solution -- move the "
							 "intermediate plane or widen the strength scan.")
	else:								# diffraction: A(sample->detector) = 0
		found = None
		for kp1 in np.linspace(20, kmax["PL1"] * 0.9, 5):
			sset(PL1=kp1)
			for kp2 in solve1("PL2", z_samp, z_det, entry=(0, 0)):
				sset(PL1=kp1, PL2=kp2)
				M = block_between(scope, z_samp, z_det)
				if found is None or abs(M[0, 1]) > abs(found[2][0, 1]):
					found = (kp1, kp2, M)	# keep the longest camera length
			sset()
		if found is None:
			raise ValueError("no diffraction projector solution: PL2 cannot zero "
							 "A(sample->detector) anywhere on the first branch.")
		solved["PL1"], solved["PL2"], _ = found
	sset()

	M_det = block_between(scope, z_samp, z_det)
	return dict(strengths=solved, predicted=p, alpha_limited=alpha_limited,
				M_det=M_det)


# ------------------------------------------------------------- verification

def covariance_report(scope: Microscope, planes: dict) -> str:
	"""Print-ready transverse covariance blocks at named planes.

	Runs :meth:`Microscope.propagate_moments` and formats the 4x4
	``(x, xt, y, yt)`` covariance nearest each requested plane.

	Parameters
	----------
	scope : Microscope
		The column (propagated here if needed).
	planes : dict
		``{label: z}`` planes to report at (m).

	Returns
	-------
	str
		The formatted report.

	Raises
	------
	None
	"""
	scope.propagate_moments()
	cov = np.asarray(scope.covariance_matrix.data)
	zc = np.asarray(scope.mu[:, columnByName('z')], float)
	idx = [columnByName(c) for c in ("x", "xt", "y", "yt")]
	lines = []
	for label, z in planes.items():
		i = int(np.argmin(np.abs(zc - z)))
		C = cov[i][np.ix_(idx, idx)]
		lines.append(f"  Sigma(x, x', y, y') at {label} (z = {zc[i]:.4f} m), units m/rad:")
		for row in C:
			lines.append("    [" + "  ".join(f"{v: .3e}" for v in row) + "]")
	return "\n".join(lines)


def plane_table(scope: Microscope, wave_crossovers=None) -> str:
	"""Tabulate every lens's conjugate planes across the methods, with deltas.

	The image family (``B = 0``) and back-focal family (``A = 0``) come from
	:meth:`Microscope.conjugate_planes` twice — once from traced rays, once
	from the accumulated transfer matrix — and each image plane is joined by
	the covariance mode's nearest beam waist. Wave crossovers, when a wave was
	propagated, are matched against the diffraction family (the family a
	flat-phase seed belongs to). Every plane is labeled by the nearest lens
	upstream: that lens's field is what folded the beam there.

	Parameters
	----------
	scope : Microscope
		The solved, ray-propagated column.
	wave_crossovers : Sequence[float], optional
		``scope.crossovers`` from a hybrid wave run, by default None (column
		omitted).

	Returns
	-------
	str
		The formatted table: one row per plane, values and deltas vs the
		matrix answer (the exact one).

	Raises
	------
	None
	"""
	frame = scope.conjugate_planes(method='frame')
	ray = scope.conjugate_planes(method='ray')
	waists = scope.beam_waists()
	Z = scope.named_positions
	lens_names = ("C1", "C2", "C3", "OL1", "OL2", "PL1", "PL2", "PL3", "PL4")
	lenses = sorted([(z, n) for n, z in Z.items() if n in lens_names])

	def upstream_lens(z):
		names = [n for zl, n in lenses if zl < z]
		return names[-1] if names else "-"

	def nearest(z, cands):
		cands = np.asarray(cands, float)
		if cands.size == 0:
			return None
		return float(cands[np.argmin(np.abs(cands - z))])

	W = 12
	rows = [f"  {'family':<8}{'lens':<6}{'matrix z':>{W}}{'ray z':>{W}}{'d(ray)':>{W}}"
			f"{'waist z':>{W}}{'d(waist)':>{W}}{'wave z':>{W}}{'d(wave)':>{W}}"]
	for fam in ("image", "diff"):
		for zf in frame[fam]:
			zr = nearest(zf, ray[fam])
			zw = nearest(zf, waists['z']) if fam == "image" else None
			# the wave logs the family its seed belongs to; a flat-phase seed
			# (this Source's gaussian) crosses the DIFFRACTION planes
			zv = nearest(zf, wave_crossovers) \
				if wave_crossovers is not None and fam == "diff" else None
			def cell(v):
				return f"{v:>{W}.6f}" if v is not None else f"{'-':>{W}}"
			def dcell(v):
				return f"{v - zf:>{W}.2e}" if v is not None else f"{'-':>{W}}"
			rows.append(f"  {fam:<8}{upstream_lens(zf):<6}{zf:>{W}.6f}"
						f"{cell(zr)}{dcell(zr)}{cell(zw)}{dcell(zw)}{cell(zv)}{dcell(zv)}")
	rows.append("  (matrix is the closed-form reference; waists carry the emittance "
				"focal shift, so their delta is physics, not error)")
	return "\n".join(rows)


def wave_matched_rays(scope: Microscope) -> np.ndarray:
	"""The ray bundle that traces what the wave actually does.

	The scaled wave carries one coherent mode seeded with a **flat phase**, so
	the trajectories it follows are the flat-phase family: zero-angle rays
	whose height sets everything downstream. This bundle is that family —
	rays at fractions of the wave envelope's half-width (2 sigma of the seed
	gaussian), plus the pair that exactly grazes the CA edge and one pair just
	inside it, so the aperture region reads clearly. The source's full
	incoherent fan is wider than the coherent mode and drawing it over the
	wave made the two look mismatched.

	Parameters
	----------
	scope : Microscope
		The configured column (used for the seed size and the source->CA
		block that locates the CA-grazing heights).

	Returns
	-------
	numpy.ndarray
		``(n_rays, len(convention))`` start vectors for ``propagate_ray(r0=)``.

	Raises
	------
	None
	"""
	src = scope.sections[0].elements[0]
	w_env = 2.0 * src.size[0]					# the wave envelope half-width
	heights = list(np.linspace(-1.0, 1.0, 7) * w_env)
	A_ca = block_between(scope, 0.0, scope.named_positions["CA"])[0, 0]
	if abs(A_ca) > 1e-12:
		h_edge = scope["CA"].radius / abs(A_ca)
		heights += [h_edge, -h_edge, 0.85 * h_edge, -0.85 * h_edge]
	r0 = np.zeros((len(heights), len(convention)))
	r0[:, columnByName('x')] = heights			# flat family: zero angle
	return r0


def ray_over_wave_figure(scope: Microscope, title: str, filename: str) -> list:
	"""Draw the wave-matched rays on top of the wave |psi(x, z)| cross-section.

	The geometric skeleton over the coherent flesh: the same column, the same
	planes, one picture. The column is subdivided for smooth z sampling; the
	rays are the flat-phase family from :func:`wave_matched_rays`, converted
	to the rotating frame the wave propagates in, so they ride the wave
	envelope through every lens and cross exactly at its crossovers. The
	CA-grazing pair is drawn brighter.

	Parameters
	----------
	scope : Microscope
		The solved column (propagated fresh on a dense copy here).
	title : str
		Figure title.
	filename : str
		Output path (PNG).

	Returns
	-------
	list of float
		The dense copy's wave crossovers (so the caller need not re-propagate).

	Raises
	------
	None
	"""
	import matplotlib.pyplot as plt
	dense = scope.subdivided(4e-3)
	dense.propagate(kind="wave-hybrid")
	r0 = wave_matched_rays(scope)
	rays = dense.propagate_ray(r0=r0)
	rot = convert_to_rotating_reference_frame(rays, dense.R)
	fig, ax = plt.subplots(figsize=(13, 5))
	_scaled_wave_cross_section(dense._wave_scaled_planes, ax,
							   named_positions=dense.named_positions,
							   crossovers=dense.crossovers, title=title)
	zs = rays[:, 0, columnByName('z')] * 1e3
	ylim = ax.get_ylim()
	xcol = rot[:, :, columnByName('x')] * 1e6
	live = dense.I[-1] > 0						# rays the aperture let through
	n_env = 7									# the envelope fan; the rest graze CA
	for j in range(xcol.shape[1]):
		grazing = j >= n_env
		ax.plot(zs, xcol[:, j],
				lw=0.9 if grazing else 0.5,
				alpha=(0.9 if live[j] else 0.35) if grazing else 0.6,
				color=("cyan" if live[j] else "tomato") if grazing else "deepskyblue")
	ax.set_ylim(ylim)
	fig.tight_layout()
	fig.savefig(filename, dpi=140)
	plt.close(fig)
	return dense.crossovers


def run_configuration(current: str, probe: str, detector: str,
					  wave: bool = True, figdir: str = "figs") -> dict:
	"""Solve, save, propagate, and report one of the eight states.

	Parameters
	----------
	current, probe, detector : str
		The three choices — see :func:`solve_column`.
	wave : bool, optional
		Whether to run the scaled-wave propagation and figure, by default
		True. The solves, settings files, ray/moments checks, and tables run
		regardless.
	figdir : str, optional
		Where figures land, by default ``'figs'``.

	Returns
	-------
	dict
		The solve result plus measured ``beam_current`` (A), ``alpha_meas``
		(rad, live rays only), and the configured ``scope``.

	Raises
	------
	None
	"""
	tag = f"{current}-{probe}-{detector}"
	sol = solve_column(current, probe, detector)
	scope = apply_strengths(load_base(), sol["strengths"])
	Z = scope.named_positions
	scope.propagate_ray()
	stated = scope.sections[0].elements[0].beam_current

	# store the state through the column's own settings mechanism:
	# settings/basic_column - <state>.json, reloadable via load_setting
	scope.save_as_setting(f"basic_column - {tag}",
						  {n: "strength" for n in sol["strengths"]})

	print(f"\n=== {tag} ===")
	print("  strengths:", {k: round(v, 4) for k, v in sol['strengths'].items()})
	print(f"  setting:  settings/basic_column - {tag}.json")
	print(f"  current: stated {stated*1e9:.2f} nA at the gun -> "
		  f"{scope.beam_current*1e9:.3f} nA at the detector "
		  f"({sol['predicted']['current_fraction']*100:.0f}% of the current passes CA)")
	alpha_meas = scope.convergence_angle
	note = "  [aperture-limited: CA cut the phase space this angle needed]" \
		if sol["alpha_limited"] else ""
	print(f"  probe:   predicted alpha {sol['predicted']['alpha']*1e3:.3f} mrad, "
		  f"measured {alpha_meas*1e3:.3f} mrad, size {sol['predicted']['size']*1e9:.0f} nm{note}")
	A, B = sol["M_det"][0]
	if detector == "image":
		print(f"  detector: image of the sample, magnification {abs(A):.1f}x "
			  f"(B = {B:.1e} m -> 0)")
	else:
		print(f"  detector: diffraction pattern, camera length {abs(B)*1e3:.1f} mm "
			  f"(A = {A:.1e} -> 0)")

	print(covariance_report(scope, {"gun exit": 0.0, "CA": Z["CA"],
									"sample": sample_plane(scope),
									"detector": Z["detector"]}))
	crossovers = None
	if wave:
		os.makedirs(figdir, exist_ok=True)
		fig = os.path.join(figdir, f"07_{tag}.png")
		crossovers = ray_over_wave_figure(scope, f"rays over |psi(x, z)| — {tag}", fig)
		print(f"  figure: {fig}")
	print(plane_table(scope, wave_crossovers=crossovers))
	return dict(sol, scope=scope, alpha_meas=alpha_meas,
				beam_current=scope.beam_current)


if __name__ == "__main__":
	wave = "--fast" not in sys.argv
	for current in ("high", "low"):
		for probe in ("convergent", "parallel"):
			for detector in ("image", "diffraction"):
				run_configuration(current, probe, detector, wave=wave)
