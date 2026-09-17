import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
	import marimo as mo
	mo.md(
		r"""
		# Special planes: three independent methods, cross-checked

		Validation for the proposal in
		`ai_wiki/raytem/notes/eric/PLAN_2026-08-21_matrix-conjugate-planes.md`.
		Before replacing anything, confirm that the **analytic transfer-matrix**
		criterion agrees with the two methods already in the repo.

		| method | how it finds a plane |
		|---|---|
		| **analytic matrix** (proposed) | accumulate the rotating-frame 2×2 x-block and solve `A = 0` (diffraction / back-focal) and `B = 0` (image) in closed form — including *inside* thick lens bodies, where the evolution is `cos/sin(K·dz)` rather than linear. No rays, no propagation. |
		| **ray trace** (existing) | `Microscope.conjugate_planes` → four reference rays through `postprocessing.findPlanes`, which interpolates where each pair's *difference* crosses zero between logged planes. |
		| **wave frame** (existing) | the hybrid scaled-Fresnel run's `Microscope.crossovers`: the frame collapses (`s → 0`) at `dz = −R`. Seeded flat (a parallel wavefront), so this is the *diffraction* family only. |

		The column is `basic_column` trimmed just past PL4 — the last crossover is
		upstream of PL4, and dropping the long PL4 → detector drift keeps the
		vertical scale readable (the beam fans to ~0.8 mm at the detector).

		Run with `marimo edit examples/05_planeComparison.py` (marimo comes from
		the `sea-eco[marimo]` extra; it is not currently a rayTEM dependency), or
		render headlessly with `marimo export html`.
		"""
	)
	return (mo,)


@app.cell
def _():
	import os
	import numpy as np
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	from pySEA.rayTEM import Drift, Lens, Quadrapole, MicroscopeSection, Microscope
	from pySEA.rayTEM import waveoptics as wo
	from pySEA.rayTEM.assemblies import load_microscope
	from pySEA.rayTEM.elements import columnByName, convention
	from pySEA.rayTEM.seashells import read_scaled_wavefield

	APERTURE_RADIUS = 5e-6		# m — sets the parallel reference rays too
	TRIM_AFTER = "PL4"			# keep the column up to this element
	TAIL = 0.02					# m of drift kept past it
	DZ_DENSE = 1e-3				# m, plane spacing for the continuous cross-section
	AIM_AT = "C1"				# image rays leave (0,0) and reach +-aperture here
	return (AIM_AT, APERTURE_RADIUS, DZ_DENSE, Drift, Lens, MicroscopeSection,
			Microscope, Quadrapole, TAIL, TRIM_AFTER, columnByName,
			convention, load_microscope, np,
			os, plt, read_scaled_wavefield, wo)


@app.cell
def _(APERTURE_RADIUS, Drift, MicroscopeSection, Microscope, TAIL, TRIM_AFTER,
	  load_microscope, os):
	def load_trimmed():
		"""Load basic_column with an aperture source, trimmed just past ``TRIM_AFTER``.

		Returns
		-------
		Microscope
			The trimmed column: elements after the cut are dropped and replaced
			by a short ``TAIL`` drift, so the vertical plot scale stays readable.
		"""
		# resolve from the installed package, so the notebook runs from any cwd
		import pySEA.rayTEM as _rt
		path = os.path.join(os.path.dirname(_rt.__file__), "microscopes",
							"basic_column.sea")
		scope = load_microscope(path)
		src = scope.sections[0].elements[0]
		src.wave_kind = "aperture"
		src.aperture_radius = APERTURE_RADIUS
		sections, done = [], False
		for sec in scope.sections:
			if done:
				break
			keep = []
			for ele in sec.elements:
				ele._position = None			# restack sequentially in the copy
				keep.append(ele)
				if ele.name == TRIM_AFTER:
					keep.append(Drift(length=TAIL))
					done = True
					break
			sections.append(MicroscopeSection(name=sec.name, elements=keep,
											  position=sec.position))
		return Microscope(name=scope.name, sections=sections)

	def flat_elements(scope):
		"""Flatten a column into ``(z_start, element, length)``, ordered along z.

		Returns
		-------
		list of tuple
			One entry per element, with the absolute z of its entrance.
		"""
		out = []
		for sec in scope.sections:
			for ele in sec.elements:
				out.append((sec.position + (ele.position or 0.0), ele,
							getattr(ele, "length", 0) or 0.0))
		return sorted(out, key=lambda e: e[0])

	scope = load_trimmed()
	z_max = max(z0 + L for z0, _, L in flat_elements(scope))
	print(f"trimmed column: {sum(len(s.elements) for s in scope.sections)} elements, "
		  f"z_max = {z_max*1e3:.1f} mm")
	return flat_elements, load_trimmed, scope, z_max


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## Method 1 — analytic, from the transfer matrices

		Accumulate the rotating-frame 2×2 x-block `M = [[A, B], [C, D]]` from the
		entrance, so `x_out = A·x_in + B·xt_in`. Then

		- `A = 0` → **diffraction** plane (output position independent of input position),
		- `B = 0` → **image** plane (output position independent of input angle).

		Inside each element, with `(m00(dz), m01(dz))` the element's own partial
		propagator, solve `m00·A₀ + m01·C₀ = 0` (and the same with `(B₀, D₀)`):

		- free space: `m00 = 1, m01 = dz` → `dz = −A₀/C₀` (linear, one division);
		- round lens of constant `K`: `m00 = cos(K·dz), m01 = sin(K·dz)/K`
		  → `tan(K·dz) = −K·A₀/C₀`.
		"""
	)
	return


@app.cell
def _(Lens, Quadrapole, np):
	def partial_xblock(ele, dz):
		"""Rotating-frame 2×2 x-block of ``ele`` over a partial length ``dz``.

		Returns the ``(x, xt)`` sub-space propagator with the Larmor rotation left
		out — the frame ``findPlanes`` works in. A round lens of constant strength
		``K`` is ``[[cos, sin/K], [-K sin, cos]]``, exact at any ``dz``; anything
		else is a thin kick ``-P`` at entry followed by free space.

		Parameters
		----------
		ele : Element
			The element.
		dz : float
			Distance into the element (metres).

		Returns
		-------
		tuple of float
			``(m00, m01, m10, m11)``.
		"""
		L = getattr(ele, "length", 0) or 0.0
		if isinstance(ele, Lens) and L > 0:
			K = ele.calibrated_strength
			if K != 0:
				c, s = np.cos(K * dz), np.sin(K * dz)
				return c, s / K, -K * s, c
		P = 0.0
		if isinstance(ele, Lens) and L == 0:
			P = ele.focal_power
		elif isinstance(ele, Quadrapole):
			P = ele.focal_powers[0]				# x axis
		return 1 - dz * P, dz, -P, 1.0

	def _roots(ele, L, P0, Q0):
		"""Solve ``m00(dz)*P0 + m01(dz)*Q0 = 0`` for ``0 < dz <= L``.

		Parameters
		----------
		ele : Element
			Element being traversed.
		L : float
			Its length (metres).
		P0, Q0 : float
			The accumulated pair: ``(A0, C0)`` for the diffraction family,
			``(B0, D0)`` for the image family.

		Returns
		-------
		list of float
			Roots inside the element, metres from its entrance.
		"""
		if L <= 0:
			return []
		K = ele.calibrated_strength if isinstance(ele, Lens) else 0
		if isinstance(ele, Lens) and (K or 0) != 0:
			if P0 == 0 and Q0 == 0:
				return []
			theta = np.arctan2(-K * P0, Q0)
			out = [(theta + n * np.pi) / K for n in range(-2, 4)]
			return sorted(min(d, L) for d in out if 1e-15 < d <= L + 1e-15)
		if Q0 == 0:
			return []
		dz = -P0 / Q0
		return [dz] if 1e-15 < dz <= L + 1e-15 else []

	def analytic_planes(scope, flat_elements):
		"""Locate both plane families analytically: roots of ``A(z)`` and ``B(z)``.

		Parameters
		----------
		scope : Microscope
			The column.
		flat_elements : callable
			Flattener returning ``(z0, element, length)`` in z order.

		Returns
		-------
		dict
			``{'diff': ndarray, 'image': ndarray}`` — plane z positions (metres).
		"""
		M = np.eye(2)
		found = {"diff": [], "image": []}
		z_prev = None
		for z0, ele, L in flat_elements(scope):
			if z_prev is not None and z0 - z_prev > 1e-12:
				M = np.array([[1.0, z0 - z_prev], [0.0, 1.0]]) @ M
			A0, B0, C0, D0 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
			for family, (P0, Q0) in (("diff", (A0, C0)), ("image", (B0, D0))):
				found[family] += [z0 + dz for dz in _roots(ele, L, P0, Q0)]
			m00, m01, m10, m11 = partial_xblock(ele, L)
			M = np.array([[m00, m01], [m10, m11]]) @ M
			z_prev = z0 + L
		return {k: np.asarray(sorted(v)) for k, v in found.items()}

	return analytic_planes, partial_xblock


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		### Consistency check: does `partial_xblock` reproduce the element's own matrix?

		The analytic method must not invent its own optics. At `dz = L` the partial
		propagator has to equal the x-block of the element's own
		`transfer_matrix()`. The stored matrix has the Larmor rotation applied, and
		for a thick round lens that scales its x-block by exactly `cos(K·L)`
		(`R @ XY` mixes x into y), so the check allows that known factor.
		"""
	)
	return


@app.cell
def _(Lens, flat_elements, np, partial_xblock, scope):
	def check_against_transfer_matrix(scope, flat_elements, partial_xblock):
		"""Compare ``partial_xblock(ele, L)`` with the element's own matrix.

		Parameters
		----------
		scope : Microscope
			The column.
		flat_elements, partial_xblock : callable
			The flattener and the partial propagator under test.

		Returns
		-------
		list of tuple
			``(name, kind, max_abs_residual)`` per element.
		"""
		rows = []
		for _z0, ele, L in flat_elements(scope):
			M6 = ele.transfer_matrix()
			stored = np.array([[M6[0, 0], M6[0, 1]], [M6[1, 0], M6[1, 1]]],
							  dtype=float)
			m00, m01, m10, m11 = partial_xblock(ele, L)
			mine = np.array([[m00, m01], [m10, m11]])
			if isinstance(ele, Lens) and L > 0 and (ele.calibrated_strength or 0):
				mine = mine * np.cos(ele.calibrated_strength * L)	# rotation factor
			rows.append((ele.name or ele.kind, ele.kind,
						 float(np.abs(stored - mine).max())))
		return rows

	_rows = check_against_transfer_matrix(scope, flat_elements, partial_xblock)
	_worst = max(r[2] for r in _rows)
	for _n, _k, _r in _rows:
		if _r > 1e-12:
			print(f"  MISMATCH {_n:10s} {_k:12s} residual {_r:.3e}")
	print(f"elements checked: {len(_rows)}   worst residual: {_worst:.3e}")
	print("=> partial_xblock is the same optics the ray path uses"
		  if _worst < 1e-9 else "=> MISMATCH, do not trust the analytic planes")
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## Methods 2 and 3 — the existing ray trace and the wave frame
		"""
	)
	return


@app.cell
def _(AIM_AT, APERTURE_RADIUS, DZ_DENSE, columnByName, convention, np,
	  read_scaled_wavefield, scope, wo, z_max):
	def reference_rays(scope):
		"""Trace the four reference rays at their **true** physical scale.

		Both pairs are given the column's own geometry, so neither needs
		rescaling to be visible:

		- the **parallel** pair (probing ``A``) is launched at
		  ``+-APERTURE_RADIUS`` with zero angle, so it is exactly the geometric
		  edge of the illuminated aperture and overlays the wave envelope;
		- the **on-axis** pair (probing ``B``) leaves ``(z, x) = (0, 0)`` at the
		  angle that brings it to ``+-APERTURE_RADIUS`` at ``AIM_AT``, i.e.
		  ``theta = +-a / z_aim`` — the bundle from an on-axis source point that
		  just fills that element.

		Parameters
		----------
		scope : Microscope
			The column.

		Returns
		-------
		tuple
			``(z, rays_x, theta_aim)`` — ``(n_planes,)``, ``(n_planes, 4)`` in
			metres, and the aimed angle (radians).
		"""
		z_aim = scope.named_positions[AIM_AT]
		theta_aim = APERTURE_RADIUS / z_aim
		dense = scope.subdivided(DZ_DENSE)
		r0 = np.zeros((4, len(convention)))
		xi, ti = columnByName("x"), columnByName("xt")
		r0[0, xi] = APERTURE_RADIUS ; r0[1, xi] = -APERTURE_RADIUS
		r0[2, ti] = theta_aim ; r0[3, ti] = -theta_aim
		dense.propagate_ray(r0=r0)
		rot = dense.rays.convert_to_rotating_reference_frame()
		return dense.rays[:, 0, columnByName("z")], rot[:, :, xi], theta_aim

	def wave_cross_section(scope, z_max):
		"""Densely sample the physical ``|psi(x, 0, z)|`` cross-section.

		Parameters
		----------
		scope : Microscope
			The column (propagated here on a subdivided copy).
		z_max : float
			Ignore planes beyond this z, keeping the common x axis tight.

		Returns
		-------
		tuple
			``(z, x, profile, crossovers)`` — plane positions (m), common x axis
			(m), ``(n_planes, n_x)`` peak-normalized ``|psi|``, and the run's
			crossover positions (m).
		"""
		dense = scope.subdivided(DZ_DENSE)
		dense.propagate_wave(mode="hybrid")
		recon = []
		for p in dense._wave_scaled_planes:
			U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(p)
			if z is None or z > z_max + 1e-12:
				continue
			psi, dx, dy = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
			recon.append((z, psi, dx))
		recon.sort(key=lambda r: r[0])
		n = recon[0][1].shape[1]
		half = max(abs(r[2]) * n / 2 for r in recon)
		x = np.linspace(-half, half, 700)
		prof = np.zeros((len(recon), x.size))
		for i, (z, psi, dx) in enumerate(recon):
			xs = (np.arange(n) - n // 2) * dx
			row = np.abs(psi[psi.shape[0] // 2, :])
			prof[i] = np.interp(x, xs, row / row.max(), left=0, right=0)
		return (np.array([r[0] for r in recon]), x, prof,
				np.asarray(dense.crossovers if dense.crossovers is not None else []))

	zw, xw, prof, crossovers = wave_cross_section(scope, z_max)
	ray = scope.conjugate_planes(axis="x")
	zr, rays_x, theta_aim = reference_rays(scope)
	print(f"wave planes: {len(zw)}   crossovers: {len(crossovers)}   "
		  f"ray diff: {len(ray['diff'])}   ray image: {len(ray['image'])}")
	print(f"image rays aimed from (0,0) to ({AIM_AT}, ±{APERTURE_RADIUS*1e6:g} µm) "
		  f"=> θ = ±{theta_aim*1e6:.1f} µrad")
	return crossovers, prof, ray, rays_x, theta_aim, xw, zr, zw


@app.cell(hide_code=True)
def _(mo):
	mo.md(r"""## The comparison table""")
	return


@app.cell
def _(analytic_planes, crossovers, flat_elements, np, ray, scope):
	analytic = analytic_planes(scope, flat_elements)

	def _nearest(arr, z):
		arr = np.asarray(arr)
		return (np.nan, np.nan) if not arr.size else \
			(arr[np.argmin(abs(arr - z))], float(min(abs(arr - z))))

	print(f"{'family':>6} {'analytic (mm)':>14} {'ray (mm)':>12} {'d_ray (um)':>11} "
		  f"{'wave (mm)':>11} {'d_wave (um)':>12}")
	print("-" * 72)
	for _fam in ("diff", "image"):
		for _z in analytic[_fam]:
			_zr, _dr = _nearest(ray[_fam], _z)
			if _fam == "diff":
				_zw, _dw = _nearest(crossovers, _z)
				print(f"{_fam:>6} {_z*1e3:14.5f} {_zr*1e3:12.5f} {_dr*1e6:11.1f} "
					  f"{_zw*1e3:11.5f} {_dw*1e6:12.1f}")
			else:
				# the image family is no longer '--': see the point-seeded cell
				print(f"{_fam:>6} {_z*1e3:14.5f} {_zr*1e3:12.5f} {_dr*1e6:11.1f} "
					  f"{'(pt seed)':>11} {'see below':>12}")

	_thick = [(z0, e, L) for z0, e, L in flat_elements(scope)
			  if L > 0 and e.kind != "Drift"]
	_inside = [(f, z, e.name) for f in ("diff", "image") for z in analytic[f]
			   for z0, e, L in _thick if z0 < z < z0 + L]
	print("\nplanes falling INSIDE a thick lens body — where the ray method's linear")
	print("interpolation is the wrong functional form:")
	for _f, _z, _n in _inside:
		print(f"   {_f:>5} plane at {_z*1e3:.5f} mm, inside {_n}")
	return analytic,


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## Image planes from the wave — the column that used to read `--`

		A scaled frame **is** a reference ray, so a run finds only the conjugate
		family its **seed** belongs to:

		| seed | `s(z) ∝` | its `s = 0` is |
		|---|---|---|
		| flat (parallel) | `A(z)` | a **diffraction** / back-focal plane |
		| point | `B(z)` | an **image** plane |

		The default source is flat, which is exactly why the image column was
		blank: the wave never crossed those planes. Seeding a **point** instead
		makes the run measure the image family by propagation — independently of
		the matrix, not derived from it.

		A point at `z = -R₀` produces, at the entrance, the frame `s = R₀·u₀`,
		`u = u₀`, i.e. `R = R₀`. So each measured plane is compared against the
		analytic plane conjugate to *that* object (`reference=-R₀`), not to the
		entrance.
		"""
	)
	return


@app.cell
def _(load_trimmed, np, read_scaled_wavefield):
	def image_planes_from_wave(R0):
		"""Crossovers of a point-seeded run = image planes of a point at z=-R0."""
		_m = load_trimmed()
		_w0 = _m.sections[0].elements[0].wave(mode="scaled")
		_U, _dxi, _deta, _lam, *_ = read_scaled_wavefield(_w0)
		from pySEA.rayTEM.seashells import make_scaled_wavefield_signal
		_seed = make_scaled_wavefield_signal(_U, _dxi, _deta, _lam,
											 s=1.0, R=R0, tau=0.0, z=0.0)
		_m.propagate_wave(wave0=_seed, mode="hybrid", absorb=0.0)
		return [float(_z) for _z in _m.crossovers]

	for _R0 in (0.05, 0.5):
		_meas = image_planes_from_wave(_R0)
		_pred = load_trimmed().conjugate_planes(axis="x", method="frame",
											   reference=-_R0)["image"]
		print(f"\nvirtual object at z = {-_R0*1e3:+.1f} mm")
		print(f"  {'#':>2} {'analytic (mm)':>15} {'wave measured (mm)':>20} {'delta (nm)':>12}")
		print("  " + "-" * 53)
		for _i, _z in enumerate(_pred, 1):
			_a = np.asarray(_meas)
			_zw = _a[np.argmin(abs(_a - _z))] if _a.size else np.nan
			print(f"  {_i:>2} {_z*1e3:15.6f} {_zw*1e3:20.6f} {abs(_zw-_z)*1e9:12.3f}")
	return image_planes_from_wave,


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		### The second object exercises mid-element frame switching

		Both objects now reproduce all five image planes. The one at −500 mm is
		the interesting case: its second predicted plane, 320.474 mm, falls
		**inside C3's body** (0.320–0.340 m), where the scaled frame is singular.

		That used to fail in two compounding ways. The engine could only switch
		frames in free segments, so it flattened *around* C3 and recorded the
		crossover as `z + |R|` — a straight-line extrapolation through the
		element — putting the plane at 419.803 mm, 99 mm out. And because the
		frame that crossed was then abandoned rather than restored, every plane
		*downstream* was wrong too, by 94–594 µm.

		A body now runs the flatten → cross → rediverge policy with **its own
		law**: the crossing is located by the medium, and the rediverge rebuilds
		the original ray's curvature as `B(d)/D(d)` — which is the familiar `d`
		only in free space. Restoring the ray is what fixes the downstream
		planes:

		| plane | before | after |
		|---|---|---|
		| 2 (inside C3) | 99 mm | 20 nm |
		| 3 | 594 µm | 0.058 nm |
		| 4 | 99 µm | 0.011 nm |
		| 5 | 234 µm | 0.026 nm |

		The residual 20 nm is a different thing: the *free* engine flattening at
		0.314916, just before C3, and extrapolating `z + |R|` through it. That
		needs the engine to see downstream optics, which it cannot from inside a
		free segment.
		"""
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## What the thick-lens segment fixed (historical)

		The `d_wave` column above is now **0.0 µm everywhere**. It was not always:
		the scaled path used to treat a **thick** element as *drift L/2 → thin
		kick `P` → drift L/2*, while the ray path used the exact
		`[[cos(KL), sin(KL)/K], [−K sin(KL), cos(KL)]]`. That put every crossover
		on this column 422–4808 µm off.

		The table below is why, and it is still worth reading as a measure of how
		wrong the thin-kick approximation is per lens. For a collimated ray,
		measuring from the lens **exit**:

		- exact: `d = cos(KL) / (K sin(KL))`
		- thin-equivalent: `d = 1/(K sin(KL)) − L/2`

		C1 alone predicts 422.3 µm, which is exactly what the first crossover used
		to be off by; later crossovers drifted further because the position *and*
		angle error after each lens is magnified downstream.

		A thick element is now carried as an exact **segment** — a
		constant-curvature medium the frame follows rather than a screen — so
		none of this error remains. The same treatment covers a thick quadrupole,
		with opposite curvature on the two axes.
		"""
	)
	return


@app.cell
def _(Lens, flat_elements, np, scope):
	print(f"{'lens':6s} {'L (mm)':>8} {'K':>9} {'KL (rad)':>9} {'d_exact (mm)':>13} "
		  f"{'d_thin (mm)':>12} {'diff (um)':>10}")
	print("-" * 72)
	for _z0, _ele, _L in flat_elements(scope):
		if isinstance(_ele, Lens) and _L > 0:
			_K = _ele.calibrated_strength
			_kL = _K * _L
			_P = _K * np.sin(_kL)						# == ele.focal_power
			_d_exact = np.cos(_kL) / (_K * np.sin(_kL))	# == ele.back_focal_distance
			_d_thin = 1.0 / _P - _L / 2
			print(f"{_ele.name:6s} {_L*1e3:8.1f} {_K:9.2f} {_kL:9.4f} "
				  f"{_d_exact*1e3:13.3f} {_d_thin*1e3:12.3f} "
				  f"{(_d_thin-_d_exact)*1e6:10.1f}")
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## The figure

		One panel, everything at **true physical scale** — no rays are rescaled.
		Both pairs are sized by the column's own geometry: the parallel pair is
		launched at the aperture radius (so it *is* the geometric edge of the
		illuminated beam and overlays the wave envelope), and the on-axis pair
		leaves `(0, 0)` at the angle that brings it to ±aperture radius at C1 —
		the bundle from an on-axis source point that just fills C1.
		"""
	)
	return


@app.cell
def _(AIM_AT, APERTURE_RADIUS, analytic, crossovers, np, plt, prof, ray,
	  rays_x, scope, theta_aim, xw, z_max, zr, zw):
	fig, ax = plt.subplots(figsize=(13, 7))

	_ze = np.concatenate([[zw[0] - 1e-4], (zw[:-1] + zw[1:]) / 2,
						  [zw[-1] + 1e-4]]) * 1e3
	_xe = np.linspace(xw[0], xw[-1], xw.size + 1) * 1e6
	ax.pcolormesh(_ze, _xe, prof.T, cmap="magma", shading="flat")

	_m = zr <= z_max + 1e-12
	for _j in (0, 1):
		ax.plot(zr[_m] * 1e3, rays_x[_m, _j] * 1e6, "-", color="#66ff99", lw=1.0,
				alpha=0.9,
				label=f"ray: parallel in at ±{APERTURE_RADIUS*1e6:g} µm "
					  "(probes A → diffraction)" if _j == 0 else None)
	for _j in (2, 3):
		ax.plot(zr[_m] * 1e3, rays_x[_m, _j] * 1e6, "--", color="#66ff99", lw=1.0,
				alpha=0.95,
				label=f"ray: (0,0) → ({AIM_AT}, ±{APERTURE_RADIUS*1e6:g} µm), "
					  f"θ=±{theta_aim*1e6:.0f} µrad (probes B → image)"
					  if _j == 2 else None)

	_yl, _yh = _xe[0], _xe[-1]
	for _zs, _c, _lbl in [(analytic["diff"], "cyan", "analytic A=0 (diffraction)"),
						  (analytic["image"], "magenta", "analytic B=0 (image)")]:
		for _i, _z in enumerate(np.asarray(_zs)):
			if _z <= z_max + 1e-12:
				ax.axvline(_z * 1e3, color=_c, ls="-", lw=1.0, alpha=0.8,
						   label=_lbl if _i == 0 else None)
	for _zs, _c, _lbl in [(ray["diff"], "cyan", "ray findPlanes (diffraction)"),
						  (ray["image"], "magenta", "ray findPlanes (image)")]:
		for _i, _z in enumerate(np.asarray(_zs)):
			if _z <= z_max + 1e-12:
				ax.plot([_z * 1e3] * 2, [_yl, _yl + 0.12 * (_yh - _yl)], color=_c,
						lw=3.0, alpha=0.9, label=_lbl if _i == 0 else None)
				ax.plot([_z * 1e3] * 2, [_yh - 0.12 * (_yh - _yl), _yh], color=_c,
						lw=3.0, alpha=0.9)
	for _i, _z in enumerate(np.asarray(crossovers)):
		if _z <= z_max + 1e-12:
			ax.axvline(_z * 1e3, color="yellow", ls="-.", lw=1.0, alpha=0.85,
					   label="wave frame (s=0)" if _i == 0 else None)

	for _name, _z in scope.named_positions.items():
		if _name and "_D" not in _name and _z <= z_max + 1e-12:
			ax.axvline(_z * 1e3, color="w", lw=0.5, ls="--", alpha=0.3)
			ax.text(_z * 1e3, _yh * 0.97, _name, color="w", rotation=90,
					ha="right", va="top", fontsize=7)

	ax.set_xlabel("z (mm)")
	ax.set_ylabel("x (µm) — physical, unscaled")
	ax.set_title("Special planes: analytic matrix vs ray trace vs wave frame\n"
				 "basic_column trimmed past PL4, |ψ(x, 0, z)| (each plane peak-normalized)")
	ax.legend(loc="lower left", fontsize=7, framealpha=0.45, labelcolor="w",
			  facecolor="black", ncol=2)
	fig.savefig("plane_comparison.png", dpi=150, bbox_inches="tight")
	fig
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
		## Conclusion

		```
		family  analytic (mm)     ray (mm)  d_ray (um)   wave (mm)  d_wave (um)
		  diff      174.57772    174.57772         0.0   175.00000        422.3
		  diff      305.19199    305.19199         0.0   310.00000       4808.0
		  diff      502.48759    502.48759         0.0   503.35027        862.7
		  diff      729.28633    729.28633         0.0   729.96285        676.5
		  diff      917.14737    917.14737         0.0   919.55285       2405.5
		 image      198.28347    198.28347         0.0          --           --
		 image      493.60771    493.79577       188.1          --           --
		 image      530.89392    530.89392         0.0          --           --
		 image      730.32816    730.32816         0.0          --           --
		 image      919.62720    919.62720         0.0          --           --
		```

		**The analytic criterion is validated:** 0.0 µm against `findPlanes` on 9 of
		10 planes, both families. That's expected — linear interpolation is exact in
		a drift, so the two *must* agree there, and they do to the last digit.

		**The one disagreement is the predicted failure mode, not noise.** The image
		plane at 493.608 mm falls **inside OL1's body** (490–500 mm), where `x(z)`
		goes as `cos/sin(K·z)` and interpolating between entrance and exit is simply
		the wrong functional form. The analytic root uses `tan(K·dz) = −K·A₀/C₀` and
		gets it right. The notebook detects and reports such planes automatically, so
		this check is reusable on any column.

		**The wave frame's mm-scale offsets are a separate, fully explained issue:**
		the scaled path treats a thick element as thin-equivalent between half-length
		drifts. The per-lens table above predicts C1's 422.3 µm exactly, matching the
		first crossover's measured offset to the last digit. Fixing that means
		advancing the frame through a thick element with the element's own
		`cos/sin` matrix — the frame *is* a ray `(h, u) = (s, s/R)` — instead of
		splitting it into two drifts around a thin kick.
		"""
	)
	return


if __name__ == "__main__":
	app.run()
