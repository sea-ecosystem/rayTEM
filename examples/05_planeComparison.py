"""Cross-check three independent ways of locating a column's special planes.

Validation script for the proposal in
``ai_wiki/raytem/notes/eric/PLAN_2026-08-21_matrix-conjugate-planes.md``: before
replacing anything, confirm that the analytic transfer-matrix criterion agrees
with the two methods already in the repo.

The three methods
-----------------
1. **analytic matrix** (proposed) — accumulate the rotating-frame 2x2 x-block
   and solve ``A = 0`` (diffraction / back-focal planes) and ``B = 0`` (image
   planes) in closed form, including *inside* thick lens bodies where the
   evolution is ``cos/sin(K dz)`` rather than linear. No rays, no propagation.
2. **ray trace** (existing) — ``Microscope.conjugate_planes``, i.e. four
   reference rays through ``postprocessing.findPlanes``, which interpolates
   where each pair's *difference* crosses zero between logged planes.
3. **wave frame** (existing) — the hybrid scaled-Fresnel run's own
   ``Microscope.crossovers``: the frame collapses (``s -> 0``) at ``dz = -R``.
   Seeded flat (a parallel wavefront), so this is the *diffraction* family only.

Outputs
-------
- a printed table comparing every plane found by every method, with offsets;
- ``plane_comparison.png``: the continuous (densely sampled) wave cross-section
  in **physical, unscaled** x, with the four reference rays overlaid and the
  planes from all three methods marked.

The column is ``basic_column`` trimmed just past PL4 — the last crossover is
upstream of PL4, and dropping the long PL4 -> detector drift keeps the vertical
scale readable (the beam fans to ~0.8 mm at the detector).
"""

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

APERTURE_RADIUS = 5e-6		# m
TRIM_AFTER = "PL4"			# keep the column up to this element
TAIL = 0.02					# m of drift kept past it
DZ_DENSE = 1e-3				# m, plane spacing for the continuous cross-section
X0, THETA0 = 5e-6, 1e-5		# reference-ray height / angle


def load_trimmed():
	"""Load basic_column with an aperture source, trimmed just past ``TRIM_AFTER``.

	Returns
	-------
	Microscope
		The trimmed column (elements after the cut are dropped and replaced by
		a short ``TAIL`` drift, so the vertical plot scale stays readable).
	"""
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "src", "pySEA", "rayTEM",
										 "microscopes", "basic_column.sea"))
	src = scope.sections[0].elements[0]
	src.wave_kind = "aperture"
	src.aperture_radius = APERTURE_RADIUS
	sections = []
	done = False
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
		One entry per element, with absolute z of its entrance.
	"""
	out = []
	for sec in scope.sections:
		for ele in sec.elements:
			z0 = sec.position + (ele.position or 0.0)
			out.append((z0, ele, getattr(ele, "length", 0) or 0.0))
	return sorted(out, key=lambda e: e[0])


def partial_xblock(ele, dz):
	"""Rotating-frame 2x2 x-block of ``ele`` over a partial length ``dz``.

	Returns ``(m00, m01, m10, m11)`` for the ``(x, xt)`` sub-space, with the
	Larmor rotation left out (the frame ``findPlanes`` works in). A round lens
	of constant strength ``K`` is ``[[cos, sin/K], [-K sin, cos]]``, which is
	exact at any ``dz``; everything else is a drift plus (for a thin element) a
	kick ``-P`` applied at ``dz = 0``.

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
		K = ele._effective_strength()
		if K != 0:
			c, s = np.cos(K * dz), np.sin(K * dz)
			return c, s / K, -K * s, c
	# thin focusing kick (applied once, at entry), then free space
	P = 0.0
	if isinstance(ele, Lens) and L == 0:
		P = ele.focal_power()
	elif isinstance(ele, Quadrapole):
		P = ele.focal_powers()[0]				# x axis
	# [[1, dz], [0, 1]] @ [[1, 0], [-P, 1]]
	return 1 - dz * P, dz, -P, 1.0


def analytic_planes(scope):
	"""Locate both plane families analytically: roots of ``A(z)`` and ``B(z)``.

	Accumulates the rotating-frame 2x2 x-block element by element and solves
	``m00(dz)*A0 + m01(dz)*C0 = 0`` (diffraction) and the same with
	``(B0, D0)`` (image) inside each element, in closed form.

	Parameters
	----------
	scope : Microscope
		The column.

	Returns
	-------
	dict
		``{'diff': ndarray, 'image': ndarray}`` — plane z positions (metres).
	"""
	M = np.eye(2)					# entrance -> current position
	found = {"diff": [], "image": []}
	z_prev = None
	for z0, ele, L in flat_elements(scope):
		if z_prev is not None and z0 - z_prev > 1e-12:
			# implicit gap: treat as free space
			M = np.array([[1.0, z0 - z_prev], [0.0, 1.0]]) @ M
		A0, B0, C0, D0 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
		for family, (P0, Q0) in (("diff", (A0, C0)), ("image", (B0, D0))):
			for dz in _roots(ele, L, P0, Q0):
				found[family].append(z0 + dz)
		m00, m01, m10, m11 = partial_xblock(ele, L)
		M = np.array([[m00, m01], [m10, m11]]) @ M
		z_prev = z0 + L
	return {k: np.asarray(sorted(v)) for k, v in found.items()}


def _roots(ele, L, P0, Q0):
	"""Solve ``m00(dz)*P0 + m01(dz)*Q0 = 0`` for ``0 <= dz <= L``.

	Parameters
	----------
	ele : Element
		Element being traversed.
	L : float
		Its length (metres).
	P0, Q0 : float
		The accumulated pair — ``(A0, C0)`` for the diffraction family,
		``(B0, D0)`` for the image family.

	Returns
	-------
	list of float
		Roots inside the element, in metres from its entrance.
	"""
	if L <= 0:
		return []
	if isinstance(ele, Lens) and (ele._effective_strength() or 0) != 0:
		# cos(K dz) P0 + sin(K dz) Q0/K = 0  ->  tan(K dz) = -K P0/Q0
		K = ele._effective_strength()
		if Q0 == 0 and P0 == 0:
			return []
		theta = np.arctan2(-K * P0, Q0)
		out = []
		for n in range(-2, 4):
			dz = (theta + n * np.pi) / K
			if 1e-15 < dz <= L + 1e-15:
				out.append(min(dz, L))
		return sorted(out)
	# free space (or thin kick + free space): linear in dz
	slope = Q0								# d/d(dz) of (P0 + dz Q0)
	if slope == 0:
		return []
	dz = -P0 / slope
	return [dz] if 1e-15 < dz <= L + 1e-15 else []


def reference_rays(scope):
	"""Trace the four reference rays and return ``(z, x_of_each_ray)``.

	Two rays enter parallel (zero angle, ``+-X0``) — their crossings are the
	diffraction planes; two leave the axis (zero position, ``+-THETA0``) —
	their crossings are the image planes. Traced on a densely subdivided copy
	so the polyline is smooth.

	Parameters
	----------
	scope : Microscope
		The column.

	Returns
	-------
	tuple
		``(z, rays_x)`` with ``z`` shape ``(n_planes,)`` and ``rays_x`` shape
		``(n_planes, 4)``, in metres.
	"""
	dense = scope.subdivided(DZ_DENSE)
	r0 = np.zeros((4, len(convention)))
	xi, ti = columnByName("x"), columnByName("xt")
	r0[0, xi] = X0 ; r0[1, xi] = -X0
	r0[2, ti] = THETA0 ; r0[3, ti] = -THETA0
	dense.propagate_ray(r0=r0)
	from pySEA.rayTEM.postprocessing import convert_to_rotating_reference_frame
	rays = convert_to_rotating_reference_frame(dense.rays, dense.R)
	return dense.rays[:, 0, columnByName("z")], rays[:, :, xi]


def scale_for_display(rays_x, target):
	"""Scale each reference-ray pair so both are visible on one axis.

	A paraxial trace is linear in its initial conditions, so scaling a pair
	changes nothing about where it crosses the axis — only its drawn size. The
	parallel pair (which follows the beam envelope) and the on-axis pair (which
	starts at 1e-5 rad and stays ~1 um wide) otherwise differ by orders of
	magnitude on one plot.

	Parameters
	----------
	rays_x : ndarray
		``(n_planes, 4)`` ray positions: two parallel rays then two on-axis.
	target : float
		Desired peak excursion (metres).

	Returns
	-------
	tuple
		``(scaled, factors)`` — the scaled array and the per-pair factors.
	"""
	out = rays_x.copy()
	factors = []
	for pair in ((0, 1), (2, 3)):
		peak = np.abs(rays_x[:, pair]).max()
		f = target / peak if peak > 0 else 1.0
		out[:, pair] *= f
		factors.append(f)
	return out, factors


def wave_cross_section(scope, z_max):
	"""Densely sample the physical |psi(x, 0, z)| cross-section.

	Parameters
	----------
	scope : Microscope
		The column (propagated here on a subdivided copy).
	z_max : float
		Ignore planes beyond this z (keeps the common x axis tight).

	Returns
	-------
	tuple
		``(z, x, profile, crossovers)`` — plane positions (m), the common x
		axis (m), the ``(n_planes, n_x)`` peak-normalized |psi| profile, and
		the run's crossover positions (m).
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
		xi = (np.arange(n) - n // 2) * dx
		row = np.abs(psi[psi.shape[0] // 2, :])
		prof[i] = np.interp(x, xi, row / row.max(), left=0, right=0)
	return (np.array([r[0] for r in recon]), x, prof,
			np.asarray(dense.crossovers if dense.crossovers is not None else []))


def compare(analytic, ray, wave):
	"""Print a comparison table of the three methods.

	Parameters
	----------
	analytic, ray : dict
		``{'diff': z[], 'image': z[]}`` from the analytic and ray methods.
	wave : ndarray
		Crossover positions from the hybrid wave run (diffraction family).

	Returns
	-------
	None
	"""
	def nearest(arr, z):
		arr = np.asarray(arr)
		return (np.nan, np.nan) if not arr.size else \
			(arr[np.argmin(abs(arr - z))], min(abs(arr - z)))
	print(f"\n{'family':>6} {'analytic (mm)':>14} {'ray (mm)':>12} {'d_ray (um)':>11} "
		  f"{'wave (mm)':>11} {'d_wave (um)':>12}")
	print("-" * 72)
	for family in ("diff", "image"):
		for z in analytic[family]:
			zr, dr = nearest(ray[family], z)
			if family == "diff":
				zw, dw = nearest(wave, z)
				print(f"{family:>6} {z*1e3:14.5f} {zr*1e3:12.5f} {dr*1e6:11.1f} "
					  f"{zw*1e3:11.5f} {dw*1e6:12.1f}")
			else:
				print(f"{family:>6} {z*1e3:14.5f} {zr*1e3:12.5f} {dr*1e6:11.1f} "
					  f"{'--':>11} {'--':>12}")
	for family in ("diff", "image"):
		extra = [z for z in ray[family]
				 if not len(analytic[family])
				 or min(abs(np.asarray(analytic[family]) - z)) > 5e-3]
		for z in extra:
			print(f"{family:>6} {'--':>14} {z*1e3:12.5f} {'(ray only)':>11}")


def plot(scope, out):
	"""Draw the cross-section with rays and all three methods' planes.

	Parameters
	----------
	scope : Microscope
		The column.
	out : str
		Output PNG path.

	Returns
	-------
	dict
		The three methods' results, for the caller to tabulate.
	"""
	z_max = max(z0 + L for z0, _, L in flat_elements(scope))
	zw, x, prof, crossovers = wave_cross_section(scope, z_max)
	analytic = analytic_planes(scope)
	ray = scope.conjugate_planes(axis="x")
	zr, rays_x = reference_rays(scope)

	fig, ax = plt.subplots(figsize=(13, 7))
	z_edges = np.concatenate([[zw[0] - 1e-4], (zw[:-1] + zw[1:]) / 2,
							  [zw[-1] + 1e-4]]) * 1e3
	x_edges = np.linspace(x[0], x[-1], x.size + 1) * 1e6
	ax.pcolormesh(z_edges, x_edges, prof.T, cmap="magma", shading="flat")

	m = zr <= z_max + 1e-12
	rays_disp, factors = scale_for_display(rays_x, 0.55 * abs(x[-1]))
	labels = [f"ray: parallel in (diffraction), x{factors[0]:.2g}", None,
			  f"ray: on-axis point (image), x{factors[1]:.2g}", None]
	for j, lbl in enumerate(labels):
		ax.plot(zr[m] * 1e3, rays_disp[m, j] * 1e6, "-" if j < 2 else "--",
				color="#66ff99", lw=1.0, alpha=0.9, label=lbl)

	# analytic planes as full-height lines; ray planes as ticks at the frame
	# edges, so a perfect overlap is still visible as both marks
	for zs, c, lbl in [(analytic["diff"], "cyan", "analytic A=0 (diffraction)"),
					   (analytic["image"], "magenta", "analytic B=0 (image)")]:
		for i, z in enumerate(np.asarray(zs)):
			if z <= z_max + 1e-12:
				ax.axvline(z * 1e3, color=c, ls="-", lw=1.0, alpha=0.8,
						   label=lbl if i == 0 else None)
	lo, hi = x_edges[0], x_edges[-1]
	for zs, c, lbl in [(ray["diff"], "cyan", "ray findPlanes (diffraction)"),
					   (ray["image"], "magenta", "ray findPlanes (image)")]:
		for i, z in enumerate(np.asarray(zs)):
			if z <= z_max + 1e-12:
				ax.plot([z * 1e3] * 2, [lo, lo + 0.13 * (hi - lo)], color=c,
						ls="-", lw=3.0, alpha=0.9, label=lbl if i == 0 else None)
				ax.plot([z * 1e3] * 2, [hi - 0.13 * (hi - lo), hi], color=c,
						ls="-", lw=3.0, alpha=0.9)
	for i, z in enumerate(np.asarray(crossovers)):
		if z <= z_max + 1e-12:
			ax.axvline(z * 1e3, color="yellow", ls="-.", lw=1.0, alpha=0.85,
					   label="wave frame (s=0)" if i == 0 else None)
	for name, z in scope.named_positions.items():
		# skip the dipole pre/post markers: too dense to label usefully
		if name and "_D" not in name and z <= z_max + 1e-12:
			ax.axvline(z * 1e3, color="w", lw=0.5, ls="--", alpha=0.3)
			ax.text(z * 1e3, hi * 0.97, name, color="w", rotation=90,
					ha="right", va="top", fontsize=7)
	ax.set_xlabel("z (mm)") ; ax.set_ylabel("x (µm), physical (unscaled)")
	ax.set_title("Special planes: analytic matrix vs ray trace vs wave frame\n"
				 f"basic_column trimmed past {TRIM_AFTER}, |ψ(x, 0, z)| "
				 f"(each plane peak-normalized)")
	ax.legend(loc="lower left", fontsize=7, framealpha=0.45, labelcolor="w",
			  facecolor="black", ncol=2)
	fig.tight_layout() ; fig.savefig(out, dpi=150) ; plt.close(fig)
	return analytic, ray, crossovers


def main():
	"""Run the three methods on the trimmed column, tabulate, and plot.

	Returns
	-------
	None
	"""
	scope = load_trimmed()
	analytic, ray, crossovers = plot(scope, "plane_comparison.png")
	compare(analytic, ray, crossovers)
	inside = [z for z in analytic["image"]
			  if any(z0 < z < z0 + L for z0, e, L in flat_elements(scope)
					 if L > 0 and e.kind != "Drift")]
	if inside:
		print("\nplanes falling INSIDE a thick lens body (where the ray method's "
			  "linear\ninterpolation is the wrong functional form): "
			  + ", ".join(f"{z*1e3:.5f} mm" for z in inside))
	print("\nwrote plane_comparison.png")


if __name__ == "__main__":
	main()
