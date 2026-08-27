"""The eight canonical operating states of a probe-forming column.

One column, three binary choices — 2 x 2 x 2 = 8 configurations, every lens
strength *solved* from a transfer-matrix condition rather than hand-tuned:

1. **current** — C1 either images the gun crossover onto the condenser
   aperture CA (``'high'``: the focused spot passes the hole whole) or is run
   weak (``'low'``: a broad beam hits CA and most of the current is cut).
2. **probe** — C2/C3 form either a **convergent** probe at the sample
   (target 30 mrad semi-angle, via a solved chain of intermediate crossovers)
   or a **nearly parallel** patch of illumination (``D = 0``: every ray from
   a single source point arrives parallel).
3. **detector** — PL1/PL2 put either an **image** of the sample plane on the
   detector (``B = 0``) or a **diffraction** pattern (``A = 0``: arrival
   position reads arrival *angle* at the sample, scaled by the camera length
   ``B``).

The same solved column is then propagated with all three methods and they are
made to answer for each other:

- **rays** (``propagate_ray``) are drawn *on top of* the scaled-wave
  ``|psi(x, z)|`` cross-section — the geometric skeleton over the coherent
  flesh (one figure per configuration, in ``figs/``);
- **moments** (``propagate_moments``) print the transverse covariance at the
  gun exit, CA, sample, and detector;
- the conjugate planes (image and back-focal family of every lens) are
  measured four independent ways — traced rays, accumulated transfer matrix,
  the wave run's own crossovers, and the covariance waists — and tabulated
  with their deltas.

Physics worth noticing in the output: in the **low**-current state the
30 mrad probe is *unreachable* — CA cuts angle along with current, because it
removes phase space, not just electrons. The script prints the
aperture-limited angle it settles for instead of pretending.

Run ``python 07_eightConfigurations.py`` from this directory; add ``--fast``
to skip the wave runs (the solves and tables still print).

Related
-------
06_aberratedObjective : Same objective, aberrated, wave-vs-ray comparison.
assemblies.Microscope.conjugate_planes : The plane machinery used throughout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(1, "../")
from pySEA.rayTEM.elements import Source, Drift, Lens, Aperture
from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection, _scaled_wave_cross_section
from pySEA.rayTEM.postprocessing import convert_to_rotating_reference_frame
from pySEA.rayTEM.elements import columnByName
from pySEA.rayTEM.microscopes.basic_column import round_lens

# ----------------------------------------------------------------- geometry
CA_RADIUS = 10e-6			# condenser aperture radius (m)
SRC_SIZE = 2.5e-6			# source half-size (m)
SRC_ANGLE = 1.5e-3			# source half-divergence (rad) -- the phase space
							# budget: 30 mrad at the sample must be demagnified
							# out of this, so it cannot be tiny
GUN_CURRENT = 1e-6			# stated emission current (A)
ALPHA_TARGET = 30e-3		# convergent-probe semi-angle at the sample (rad)
LENS_L_C = 0.02				# condenser bore length (m)
LENS_L_P = 0.015			# projector bore length (m)
KMAX_C = np.pi / 2 / LENS_L_C - 1e-6	# first-branch strength limits
KMAX_P = np.pi / 2 / LENS_L_P - 1e-6

# defaults: only used before the solves overwrite them
NOMINAL = dict(kc1=30.0, kc2=30.0, kc3=20.0, kp1=30.0, kp2=25.0)


def build_column(kc1: float = NOMINAL["kc1"], kc2: float = NOMINAL["kc2"],
				 kc3: float = NOMINAL["kc3"], kp1: float = NOMINAL["kp1"],
				 kp2: float = NOMINAL["kp2"], wave_shape: tuple = (128, 128),
				 wave_extent: float = 40e-6) -> Microscope:
	"""Assemble the eight-configuration column with the given lens strengths.

	Source, C1, condenser aperture CA, C2, C3, the OL1/sample/OL2 objective
	group, projectors PL1/PL2, and a named detector plane. Everything except
	the five strength arguments is fixed, so a configuration *is* its five
	numbers.

	Parameters
	----------
	kc1, kc2, kc3 : float, optional
		Condenser strengths (C1 sets the current state; C2/C3 the probe).
	kp1, kp2 : float, optional
		Projector strengths (image vs diffraction at the detector).
	wave_shape : tuple, optional
		Wave grid ``(ny, nx)``, by default ``(128, 128)``.
	wave_extent : float, optional
		Wave grid full width (m), by default 40 µm — generous around the
		2.5 µm source.

	Returns
	-------
	Microscope
		The assembled single-section column, ~0.98 m long.

	Raises
	------
	None
	"""
	els = [Source(name="G", voltage=200, size=(SRC_SIZE,) * 2, np_xy=(5, 5),
				  angle=(SRC_ANGLE,) * 2, na_xy=(3, 3), beam_current=GUN_CURRENT,
				  wave_shape=wave_shape, wave_extent=wave_extent,
				  wave_kind="gaussian"),
		   Drift(length=0.10),
		   Lens(name="C1", strength=kc1, length=LENS_L_C), Drift(length=0.07),
		   Aperture(name="CA", radius=CA_RADIUS), Drift(length=0.05),
		   Lens(name="C2", strength=kc2, length=LENS_L_C), Drift(length=0.09),
		   Lens(name="C3", strength=kc3, length=LENS_L_C), Drift(length=0.10),
		   round_lens("OL1", f=0.008, length=0.01),
		   # the sample sits ~9 mm past OL1's center: just OUTSIDE its 8 mm
		   # focal length, so a real image can land on it. 5 mm past it would
		   # be INSIDE f and no condenser setting could focus there.
		   Drift(length=0.004), Drift(name="sample", length=0.0), Drift(length=0.005),
		   round_lens("OL2", f=0.010, length=0.01), Drift(length=0.08),
		   Lens(name="PL1", strength=kp1, length=LENS_L_P), Drift(length=0.07),
		   Lens(name="PL2", strength=kp2, length=LENS_L_P), Drift(length=0.30),
		   # a short tail past the detector: a conjugate plane landing EXACTLY
		   # on the column's last z has no interval around it, so the plane
		   # search would silently drop the detector row from the table
		   Drift(name="detector", length=0.0), Drift(length=0.02)]
	return Microscope(name="eight configurations",
					  sections=[MicroscopeSection(name="col", elements=els)])


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


def _source_grid() -> np.ndarray:
	"""The source's exact ray grid, as (x, xt, y, yt) rows.

	Mirrors what ``Source.rays()`` emits (5x5 positions x 3x3 angles), so a
	per-ray *prediction* through the 2x2 blocks matches the traced rays to
	machine precision on an ideal column.

	Returns
	-------
	numpy.ndarray
		``(225, 4)`` phase-space start points.

	Raises
	------
	None
	"""
	xs = np.linspace(-SRC_SIZE, SRC_SIZE, 5)
	ts = np.linspace(-SRC_ANGLE, SRC_ANGLE, 3)
	return np.array([(x, t, y, ty) for x in xs for y in xs for t in ts for ty in ts])


def predict_probe(scope: Microscope) -> dict:
	"""Predict the probe the column forms, per-ray, from its transfer blocks.

	Sends the source grid through the source->CA and source->sample blocks,
	masks at CA, and reads off what survives. Exact for the ideal column, and
	smooth enough in the lens strengths to solve against — unlike a traced
	measurement, whose outermost-ray identity jumps.

	Parameters
	----------
	scope : Microscope
		The column to predict.

	Returns
	-------
	dict
		``current_fraction`` (of rays surviving CA), ``alpha`` (max total
		angle at the sample among survivors, rad), ``size`` (max radius, m).

	Raises
	------
	None
	"""
	Z = scope.named_positions
	Ms, Mca = block_between(scope, 0.0, Z["sample"]), block_between(scope, 0.0, Z["CA"])
	x0, t0, y0, ty0 = _source_grid().T
	live = np.hypot(Mca[0, 0] * x0 + Mca[0, 1] * t0,
					Mca[0, 0] * y0 + Mca[0, 1] * ty0) <= CA_RADIUS
	alpha = np.hypot(Ms[1, 0] * x0 + Ms[1, 1] * t0, Ms[1, 0] * y0 + Ms[1, 1] * ty0)
	size = np.hypot(Ms[0, 0] * x0 + Ms[0, 1] * t0, Ms[0, 0] * y0 + Ms[0, 1] * ty0)
	return dict(current_fraction=float(live.mean()),
				alpha=float(alpha[live].max()) if live.any() else 0.0,
				size=float(size[live].max()) if live.any() else 0.0)


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


def _solve_imaging(fixed: dict, kname: str, z_from: float, z_to: float,
				   kmax: float, entry: tuple = (0, 1)) -> list:
	"""All strengths of one lens that zero a block entry between two planes.

	The workhorse of every configuration: ``entry=(0, 1)`` solves ``B = 0``
	(imaging), ``(1, 1)`` solves ``D = 0`` (collimation), ``(0, 0)`` solves
	``A = 0`` (diffraction).

	Parameters
	----------
	fixed : dict
		Strengths already decided, passed to :func:`build_column`.
	kname : str
		The strength being solved (``'kc2'``, ``'kp1'``, ...).
	z_from, z_to : float
		The two planes the condition connects (m).
	kmax : float
		Upper strength limit (first branch, ``K L < π/2``).
	entry : tuple, optional
		Block entry to zero, by default ``(0, 1)`` (``B``).

	Returns
	-------
	list of float
		Every solution found on a 50-point bracket scan (possibly empty).

	Raises
	------
	None
	"""
	def f(k):
		scope = build_column(**{**fixed, kname: k})
		return block_between(scope, z_from, z_to)[entry]
	return [brentq(f, *b) for b in _brackets(f, np.linspace(1.0, kmax, 50))]


def solve_column(current: str, probe: str, detector: str) -> dict:
	"""Solve the five lens strengths for one of the eight configurations.

	Works the way an operator does — as a chain of crossovers, each lens a
	1D imaging condition, so every solve is a bracketed root find rather than
	a multidimensional shot in the dark:

	- **C1**: ``B(source→CA) = 0`` for ``'high'``; fixed weak (K = 5) for
	  ``'low'``.
	- **convergent**: C2 images the source to an intermediate crossover z2,
	  C3 images z2 to OL1's object plane z3 (itself solved from OL1 alone),
	  which lands the final crossover on the sample; z2 is then the single
	  knob swept so the *predicted per-ray* semi-angle hits 30 mrad. When no
	  z2 reaches it (the low-current state: CA has cut the phase space), the
	  angle-maximizing z2 is used and ``alpha_limited`` is set.
	- **parallel**: C2 images the source to z2 = 0.30 m, C3 zeroes
	  ``D(source→sample)``.
	- **image**: PL1 images the sample to an intermediate plane, PL2 images
	  that to the detector. **diffraction**: PL2 zeroes ``A(sample→detector)``
	  for a mid-range PL1.

	Parameters
	----------
	current : {'high', 'low'}
		C1 state: gun crossover imaged onto CA, or a broad beam cut by it.
	probe : {'convergent', 'parallel'}
		What C2/C3 deliver at the sample.
	detector : {'image', 'diffraction'}
		What PL1/PL2 deliver at the detector.

	Returns
	-------
	dict
		``strengths`` (the five solved K), ``predicted`` (from
		:func:`predict_probe`), ``alpha_limited`` (bool), and the detector
		block ``M_det`` from the sample.

	Raises
	------
	ValueError
		If a required imaging solve finds no solution (a geometry change
		broke the chain — the message names the failing stage).
	"""
	if current not in ("high", "low") or probe not in ("convergent", "parallel") \
			or detector not in ("image", "diffraction"):
		raise ValueError(f"unknown configuration ({current!r}, {probe!r}, {detector!r}); "
						 "expected ('high'|'low', 'convergent'|'parallel', 'image'|'diffraction').")
	Z = build_column().named_positions

	# --- C1: the current state
	if current == "high":
		kc1 = brentq(lambda k: block_between(build_column(kc1=k), 0.0, Z["CA"])[0, 1], 20, 60)
	else:
		kc1 = 5.0

	# --- OL1's object plane: where a crossover must land to reappear on the sample
	scope0 = build_column()
	zb = _brackets(lambda z: block_between(scope0, z, Z["sample"])[0, 1],
				   np.linspace(0.385, 0.473, 30))
	if not zb:
		raise ValueError("no OL1 object plane found between C3 and OL1 -- the objective "
						 "geometry no longer supports a real image at the sample.")
	z3 = brentq(lambda z: block_between(scope0, z, Z["sample"])[0, 1], *zb[0])

	alpha_limited = False
	if probe == "convergent":
		def best_at(z2):
			best = None
			for kc2 in _solve_imaging({"kc1": kc1}, "kc2", 0.0, z2, KMAX_C):
				for kc3 in _solve_imaging({"kc1": kc1, "kc2": kc2}, "kc3", z2, z3, KMAX_C):
					p = predict_probe(build_column(kc1=kc1, kc2=kc2, kc3=kc3))
					if best is None or p["alpha"] > best[0]["alpha"]:
						best = (p, kc2, kc3)
			return best
		z2s = np.linspace(0.275, 0.345, 12)
		g = lambda z2: ((best_at(z2) or ({"alpha": np.nan},))[0]["alpha"]) - ALPHA_TARGET
		br = _brackets(g, z2s)
		if br:
			z2 = brentq(g, *br[0], xtol=1e-7)
		else:								# CA has cut the phase space: take the max
			z2 = max(z2s, key=lambda z: (best_at(z) or ({"alpha": -1},))[0]["alpha"])
			alpha_limited = True
		p, kc2, kc3 = best_at(z2)
	else:									# parallel: D(source->sample) = 0
		z2 = 0.30
		sols = [(predict_probe(build_column(kc1=kc1, kc2=kc2, kc3=kc3)), kc2, kc3)
				for kc2 in _solve_imaging({"kc1": kc1}, "kc2", 0.0, z2, KMAX_C)
				for kc3 in _solve_imaging({"kc1": kc1, "kc2": kc2}, "kc3",
										  0.0, Z["sample"], KMAX_C, entry=(1, 1))]
		if not sols:
			raise ValueError("no parallel-probe solution: C3 cannot zero D(source->sample) "
							 "with C2 imaging the source to z2=0.30 m.")
		p, kc2, kc3 = min(sols, key=lambda s: s[0]["alpha"])

	# --- projectors (independent of the condensers: everything is downstream
	#     of the sample)
	if detector == "image":
		zp = 0.63							# intermediate image between PL1 and PL2
		sols = [(kp1, kp2)
				for kp1 in _solve_imaging({}, "kp1", Z["sample"], zp, KMAX_P)
				for kp2 in _solve_imaging({"kp1": kp1}, "kp2", zp, Z["detector"], KMAX_P)]
		if not sols:
			raise ValueError("no imaging projector solution for the intermediate plane "
							 f"zp={zp} m; move it or widen the strength scan.")
		kp1, kp2 = sols[0]
	else:									# diffraction: A(sample->detector) = 0
		found = None
		for kp1 in np.linspace(10, KMAX_P * 0.95, 6):
			def f(kp2):
				return block_between(build_column(kp1=kp1, kp2=kp2),
									 Z["sample"], Z["detector"])[0, 0]
			for b in _brackets(f, np.linspace(1.0, KMAX_P, 40)):
				kp2 = brentq(f, *b)
				M = block_between(build_column(kp1=kp1, kp2=kp2), Z["sample"], Z["detector"])
				if found is None or abs(M[0, 1]) > abs(found[2][0, 1]):
					found = (kp1, kp2, M)	# keep the longest camera length
		if found is None:
			raise ValueError("no diffraction projector solution: PL2 cannot zero "
							 "A(sample->detector) anywhere on the first branch.")
		kp1, kp2, _ = found

	strengths = dict(kc1=kc1, kc2=kc2, kc3=kc3, kp1=kp1, kp2=kp2)
	M_det = block_between(build_column(**strengths), Z["sample"], Z["detector"])
	return dict(strengths=strengths, predicted=p, alpha_limited=alpha_limited,
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
	propagated, are matched in the same way. Every plane is labeled by the
	nearest lens upstream: that lens's field is what folded the beam there.

	Parameters
	----------
	scope : Microscope
		The solved, ray-propagated column.
	wave_crossovers : Sequence[float], optional
		``scope.crossovers`` from a hybrid wave run, by default None (column
		omitted). The wave logs the conjugate family its seed belongs to.

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
	lenses = sorted([(z, n) for n, z in Z.items()
					 if n in ("C1", "C2", "C3", "OL1", "OL2", "PL1", "PL2")])

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


def ray_over_wave_figure(scope: Microscope, title: str, filename: str) -> None:
	"""Draw the traced rays on top of the wave |psi(x, z)| cross-section.

	The geometric skeleton over the coherent flesh: the same column, the same
	planes, one picture. The column is subdivided for smooth z sampling; the
	rays are converted to the rotating frame the wave propagates in.

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
	rays = dense.propagate_ray()
	rot = convert_to_rotating_reference_frame(rays, dense.R)
	fig, ax = plt.subplots(figsize=(13, 5))
	_scaled_wave_cross_section(dense._wave_scaled_planes, ax,
							   named_positions=dense.named_positions,
							   crossovers=dense.crossovers, title=title)
	zs = rays[:, 0, columnByName('z')] * 1e3
	ylim = ax.get_ylim()
	xcol = rot[:, :, columnByName('x')] * 1e6
	live = dense.I[-1] > 0					# rays the apertures let through
	for j in range(xcol.shape[1]):
		ax.plot(zs, xcol[:, j], lw=0.4, alpha=0.7 if live[j] else 0.25,
				color="deepskyblue" if live[j] else "tomato")
	ax.set_ylim(ylim)						# rays cut at CA may exceed the wave frame
	fig.tight_layout()
	fig.savefig(filename, dpi=140)
	plt.close(fig)
	return dense.crossovers


def run_configuration(current: str, probe: str, detector: str,
					  wave: bool = True, figdir: str = "figs") -> dict:
	"""Solve, propagate, and report one of the eight configurations.

	Parameters
	----------
	current, probe, detector : str
		The three choices — see :func:`solve_column`.
	wave : bool, optional
		Whether to run the scaled-wave propagation and figure, by default
		True. The solves, ray/moments checks, and tables run regardless.
	figdir : str, optional
		Where figures land, by default ``'figs'``.

	Returns
	-------
	dict
		The solve result plus measured ``beam_current`` (A), ``alpha_meas``
		(rad, live rays only), and the solved ``scope``.

	Raises
	------
	None
	"""
	tag = f"{current}-{probe}-{detector}"
	sol = solve_column(current, probe, detector)
	scope = build_column(**sol["strengths"])
	Z = scope.named_positions
	scope.propagate_ray()

	print(f"\n=== {tag} ===")
	print("  strengths:", {k: round(v, 4) for k, v in sol['strengths'].items()})
	print(f"  current: stated {GUN_CURRENT*1e9:.1f} nA at the gun -> "
		  f"{scope.beam_current*1e9:.2f} nA at the detector "
		  f"({sol['predicted']['current_fraction']*100:.0f}% of rays pass CA)")
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
									"sample": Z["sample"], "detector": Z["detector"]}))
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
