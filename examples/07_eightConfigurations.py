"""The eight canonical operating states of the standard column.

``basic_column.sea`` — the ecosystem's stock 200 kV column, with its 1 nA gun
and the condenser aperture CA after C1 — driven into all 2 x 2 x 2 = 8
operating states. Nothing about the geometry is touched: a state is purely a
set of lens strengths, each *solved* from a transfer-matrix condition the way
an operator chains crossovers, never hand-tuned:

1. **current** — C1 either images the gun crossover onto CA (``'high'``: the
   focused spot passes the hole whole) or is run weak (``'low'``: a broad
   beam hits CA and most of the current is cut).
2. **probe** — C2/C3 alone (the objective lenses are **never retuned**)
   form either a **convergent** 30 mrad probe at the sample — small in both
   current states because OL1's 3 mm focal length matches the mid-gap sample
   position, so the condensers only choose how much beam to land on it (the
   script still reports the reachable maximum honestly if a geometry change
   ever puts the target out of reach) — or a **nearly parallel** patch of
   illumination (``D = 0``: every ray from a single source point arrives
   parallel).
3. **detector** — the projector chain PL1–PL4, solved lens by lens as a
   relay of intermediate images, puts either an **image** of the sample
   plane on the detector (``B = 0``) or a **diffraction** pattern
   (``A = 0``: arrival position reads arrival *angle* at the sample, scaled
   by the camera length ``B``; PL1 first puts the angular spectrum at its
   back focal plane, and the relay carries that instead).

Each solved state is saved through the column's own settings mechanism
(``Microscope.save_as_setting``) as ``settings/basic_column - <state>.json``,
so ``scope.load_setting("basic_column - high-convergent-image")`` restores it
onto a freshly loaded column.

The same solved column is then propagated with all three methods and they are
made to answer for each other:

- **rays** are drawn *on top of* the scaled-wave ``|psi(x, z)|``
  cross-section (one figure per state, in ``figs/``). The overlay traces the
  rays the *wave* cares about — the flat-phase family the scaled frame itself
  follows (zero-angle rays at fractions of the wave envelope, plus the
  CA-grazing pair where it falls inside that envelope) — rather than the
  source's full incoherent fan, which is real but wider than the single
  coherent mode the wave carries and used to make the two look mismatched.
  Every drawn ray stays within the wave pattern.
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
from pySEA.rayTEM.microscopes.basic_column import strength_for_focal_length

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SEA = os.path.join(_HERE, "..", "src", "pySEA", "rayTEM", "microscopes",
						"basic_column.sea")

#: The lenses a state may retune. The objective pair NEVER changes -- probe
#: focusing is entirely the condensers' job and projection entirely the
#: projectors' (all four of them: with the 50 mm projector spacings, PL1/PL2
#: alone cannot land a conjugate on the detector past a frozen PL3/PL4).
SOLVED_LENSES = ("C1", "C2", "C3", "PL1", "PL2", "PL3", "PL4")
ALPHA_TARGET = 30e-3		# convergent-probe semi-angle at the sample (rad)
#: The low-current state OVERFOCUSES C1: its crossover lands this fraction of
#: the way from C1's exit to CA, so the beam diverges hard into the aperture
#: and most of the current is cut. (A merely weak C1 no longer works: with
#: 50 mm drifts the unfocused beam only slightly overfills the 10 µm hole.)
LOW_CROSSOVER_FRACTION = 0.2


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
	x_rot = Mca[0, 0] * x0 + Mca[0, 1] * t0
	y_rot = Mca[0, 0] * y0 + Mca[0, 1] * ty0
	# the aperture reads per-axis maxima in the LAB frame, and the fan is a
	# square grid: rotated by the Larmor angle accumulated upstream, its
	# corner grows the per-axis maximum by up to cos+sin (~8% at C1's 5 deg
	# in the overfocused low state) -- taking the rotating-frame max instead
	# made the predicted cut, and with it the solved angle, ~9% optimistic
	phi = sum(l.strength * l.length for n, l in _lens_map(scope).items()
			  if Z[n] < Z["CA"])
	x_ca = np.cos(phi) * x_rot - np.sin(phi) * y_rot
	y_ca = np.sin(phi) * x_rot + np.cos(phi) * y_rot
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
	- **convergent**: for each C2, C3 is solved so the **total**
	  ``B(source→sample) = 0`` through the frozen objective (a direct
	  condition — no intermediate crossover is prescribed, so virtual
	  objects are allowed); C2 is then swept so the *predicted per-ray*
	  semi-angle hits 30 mrad. When no C2 reaches the target — the frozen
	  objective caps what the condensers can deliver — the angle-maximizing
	  setting is used and ``alpha_limited`` is set.
	- **parallel**: C2 images the source to z2 = 0.28 m, C3 zeroes
	  ``D(source→sample)`` with OL1 left as stored.
	- **image**: the projector relay — PL1 images the sample to a plane
	  between PL1 and PL2, PL2 relays it to between PL2 and PL3, PL3 to
	  between PL3 and PL4, and PL4 lands it on the detector; four bracketed
	  1D imaging solves. **diffraction**: identical relay, except PL1's
	  first condition is ``A = 0`` (its back focal plane), so the angular
	  spectrum is what gets relayed to the detector.

	Parameters
	----------
	current : {'high', 'low'}
		C1 state: gun crossover imaged onto CA, or a broad beam cut by it.
	probe : {'convergent', 'parallel'}
		What the condensers deliver at the sample (the objective is frozen).
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
	# scan each lens's strength over a PHYSICAL focal range (2 m down to
	# 1 mm) rather than up to the first-branch cap: with thin bores the cap
	# sits orders of magnitude above any strength a column would run, and a
	# scan stretched to it would step right over the working region
	krange = {n: (strength_for_focal_length(2.0, l.length),
				  strength_for_focal_length(0.001, l.length))
			  for n, l in lens.items()}
	solved = {}

	def sset(**kw):
		"""Reset every solvable lens to stored, then apply overrides."""
		for n, l in lens.items():
			l.strength = stored[n]
		for n, v in {**solved, **kw}.items():
			lens[n].strength = v

	def solve1(name, z_from, z_to, entry=(0, 1), n_scan=50):
		"""All strengths of ``name`` zeroing one block entry between planes.

		``n_scan`` sets the bracket-scan density; a pair of roots closer
		together than one scan step is invisible, so a solve whose physical
		branch lives in a narrow window needs a denser scan.
		"""
		def f(k):
			lens[name].strength = k
			return block_between(scope, z_from, z_to)[entry]
		roots = [brentq(f, *b) for b in _brackets(f, np.linspace(*krange[name], n_scan))]
		lens[name].strength = solved.get(name, stored[name])
		return roots

	# --- C1: the current state. 'high' images the gun crossover ONTO the
	# aperture; 'low' overfocuses so the crossover lands well before it and
	# the diverging beam overfills the hole.
	z_c1 = Z["C1"]
	z_target = Z["CA"] if current == "high" else \
		z_c1 + LOW_CROSSOVER_FRACTION * (Z["CA"] - z_c1)
	roots = solve1("C1", 0.0, z_target)
	if not roots:
		raise ValueError("C1 cannot place the gun crossover at "
						 f"z = {z_target:.4f} m -- the gun/CA geometry changed.")
	solved["C1"] = roots[0]
	sset()

	# --- the probe (condensers ONLY: the objective pair is never retuned)
	alpha_limited = False
	if probe == "convergent":
		def best_at(kc2):
			# for this C2, every C3 that lands the crossover on the sample --
			# B(source->sample) = 0 THROUGH the frozen objective, so no
			# intermediate crossover is prescribed and virtual objects count.
			# Among the roots, a genuinely SMALL probe wins: B = 0 alone is
			# also satisfied by magnifying branches (a huge image of the
			# source, each point converging at a large angle), which are not
			# probes. The scan is dense because the probe branch can live in
			# a C3 window narrower than a coarse scan's step.
			best, best_any = None, None
			sset(C2=kc2)
			for kc3 in solve1("C3", 0.0, z_samp, n_scan=220):
				sset(C2=kc2, C3=kc3)
				p = predict_probe(scope)
				if p["size"] <= 1e-6 and (best is None or p["alpha"] > best[0]["alpha"]):
					best = (p, kc2, kc3)
				if best_any is None or p["alpha"] > best_any[0]["alpha"]:
					best_any = (p, kc2, kc3)
			sset()
			return best or best_any
		def alpha_of(kc2):
			return ((best_at(kc2) or ({"alpha": -1.0},))[0]["alpha"])
		# the reachable angle is a NARROW resonance in C2 (the setting that
		# lands the most beam on the frozen objective), so a two-stage search:
		# a coarse scan to find the peak, local refinement, then bisection on
		# the rising edge for the exact target crossing
		kc2s = np.linspace(*krange["C2"], 120)
		alphas = np.array([alpha_of(k) for k in kc2s])
		if not (alphas > 0).any():
			raise ValueError("no convergent-probe solution at all: C3 cannot "
							 "zero B(source->sample) for any C2 -- the "
							 "condenser/objective geometry changed.")
		if alphas.max() < ALPHA_TARGET:		# the frozen objective caps the reach
			i = int(np.argmax(alphas))
			fine = np.linspace(kc2s[max(i - 1, 0)], kc2s[min(i + 1, len(kc2s) - 1)], 40)
			hit = best_at(fine[int(np.argmax([alpha_of(k) for k in fine]))])
			alpha_limited = True
		else:
			# bisect each coarse interval whose endpoints straddle the target
			# (the crossing can sit far from the peak, on the slope), and
			# VERIFY the result: alpha can jump discontinuously where the C3
			# branch structure changes, and a bisection converging onto a
			# jump lands far from the target -- then the next interval is tried
			cross = [i for i in range(len(kc2s) - 1)
					 if (alphas[i] - ALPHA_TARGET) * (alphas[i + 1] - ALPHA_TARGET) < 0]
			hit = None
			for i in cross:
				a, b = kc2s[i], kc2s[i + 1]
				if alphas[i] > ALPHA_TARGET:		# orient: alpha(a) below target
					a, b = b, a
				for _ in range(60):
					mid = 0.5 * (a + b)
					if alpha_of(mid) < ALPHA_TARGET:
						a = mid
					else:
						b = mid
				cand = best_at(b)
				if cand and abs(cand[0]["alpha"] - ALPHA_TARGET) < 1e-4 * ALPHA_TARGET:
					hit = cand
					break
			if hit is None:				# every straddle was a branch jump
				i = int(np.argmin(np.abs(alphas - ALPHA_TARGET)))
				hit = best_at(kc2s[i])
				alpha_limited = True
		p, kc2, kc3 = hit
		solved["C2"], solved["C3"] = float(kc2), float(kc3)
	else:								# parallel: D(source->sample) = 0 via C3
		best = None
		for frac in (0.3, 0.5, 0.7):	# intermediate crossover between C2 and C3
			z2 = Z["C2"] + frac * (Z["C3"] - Z["C2"])
			for kc2 in solve1("C2", 0.0, z2):
				sset(C2=kc2)
				for kc3 in solve1("C3", 0.0, z_samp, entry=(1, 1)):
					sset(C2=kc2, C3=kc3)
					p = predict_probe(scope)
					if best is None or p["alpha"] < best[0]["alpha"]:
						best = (p, kc2, kc3)
				sset(C2=kc2)
			sset()
		if best is None:
			raise ValueError("no parallel-probe solution: C3 cannot zero "
							 "D(source->sample) for any C2 crossover between "
							 "C2 and C3.")
		p, solved["C2"], solved["C3"] = best
	sset()

	# --- the projectors (independent of the condensers: everything is
	#     downstream of the sample). A relay: each lens hands an intermediate
	#     conjugate to the next, so every solve is a bracketed 1D root.
	#     'image' relays the SAMPLE PLANE; 'diffraction' relays the angular
	#     spectrum (PL1's first condition is A = 0, its back focal plane).
	zp = [0.5 * (Z["PL1"] + Z["PL2"]), 0.5 * (Z["PL2"] + Z["PL3"]),
		  0.5 * (Z["PL3"] + Z["PL4"])]
	stages = [("PL1", z_samp, zp[0], (0, 1) if detector == "image" else (0, 0)),
			  ("PL2", zp[0], zp[1], (0, 1)),
			  ("PL3", zp[1], zp[2], (0, 1)),
			  ("PL4", zp[2], z_det, (0, 1))]
	for name, z_from, z_to, entry in stages:
		roots = solve1(name, z_from, z_to, entry=entry)
		if not roots:
			raise ValueError(f"projector relay broke at {name}: no strength "
							 f"images {z_from:.4f} m onto {z_to:.4f} m -- the "
							 "projector geometry changed.")
		solved[name] = roots[0]
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
	gaussian), plus the pair that grazes the CA edge **when that pair lies
	inside the envelope**. A grazing pair taller than the envelope belongs to
	beam the wave does not carry — drawing it put rays outside the |psi|
	pattern for no gain — and when the aperture is that much wider than the
	coherent mode it does not shape the wave anyway, so the pair is simply
	omitted. Every ray drawn stays within the wave.

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
		if h_edge < w_env:					# only where the aperture bites the wave
			heights += [h_edge, -h_edge]
	r0 = np.zeros((len(heights), len(convention)))
	r0[:, columnByName('x')] = heights			# flat family: zero angle
	return r0


def ray_over_wave_figure(scope: Microscope, title: str, filename: str) -> list:
	"""Draw the wave-matched rays on top of the wave |psi(x, z)| cross-section.

	The geometric skeleton over the coherent flesh: the same column, the same
	planes, one picture. The column is subdivided for smooth z sampling; the
	rays are the flat-phase family from :func:`wave_matched_rays`, converted
	to the rotating frame the wave propagates in, so they ride the wave
	envelope through every lens and cross exactly at its crossovers.

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
	rot = convert_to_rotating_reference_frame(rays)
	fig, ax = plt.subplots(figsize=(13, 5))
	_scaled_wave_cross_section(dense._wave_scaled_planes, ax,
							   named_positions=dense.named_positions,
							   crossovers=dense.crossovers, title=title)
	zs = rays[:, 0, columnByName('z')] * 1e3
	ylim = ax.get_ylim()
	xcol = rot[:, :, columnByName('x')] * 1e6
	for j in range(xcol.shape[1]):
		ax.plot(zs, xcol[:, j], lw=0.5, alpha=0.6, color="deepskyblue")
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
	note = "  [limited: with the objective frozen, this is the most the condensers can deliver]" \
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
