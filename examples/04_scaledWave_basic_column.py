"""Full-column scaled-Fresnel wave propagation through the basic_column template.

Demonstrates ``propagate_wave(mode='hybrid')`` on ``microscopes/basic_column.sea``:
a flat-intensity hard-aperture wavefunction Θ(a−r) (200 kV; the source's
``wave_kind`` is set to ``'aperture'``) is carried from the source to the
detector at z = 1.264 m. Lenses are absorbed into the frame curvature R (their
centimetre-scale focal phases never touch the sampled array), and every beam
crossover is traversed by the hybrid frame-switching policy: the converging
frame flattens where its reference curvature becomes representable, the wave
crosses the real focus by ordinary carrier-free Fresnel propagation — the
crossover (back-focal / diffraction) plane is logged — and re-factors onto a
diverging frame past it. One ξ/η grid calibration serves the whole run while
the physical pixel |s|·Δξ spans nanometres (at the foci) to micrometres (at
the detector).

Outputs (written to the current working directory):

- ``basic_column_scaled_wave.sea`` — the stacked hybrid result: U(z, η, ξ)
  plus companion s(z)/R(z)/tau(z)/frame(z) Signals on the shared plane-z axis
  (crossover planes tagged in metadata).
- ``basic_column_scaled_wave_cross_section.png`` — |ψ| vs z in physical
  coordinates and in the scaled coordinate ξ (the wave analog of the
  geometric ``plot2D`` ray diagram), lenses and crossovers annotated.
- ``basic_column_scaled_wave_xy_slices.png`` — |ψ(x, y)|² at the key planes:
  source, C1 back-focal plane, sample, objective focus, projector focus,
  detector — each on its native physical grid.

Run from anywhere with the rayTEM environment active:

	python examples/04_scaledWave_basic_column.py
"""

import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection, load_microscope
from pySEA.rayTEM.elements import Drift
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.seashells import read_scaled_wavefield, read_wavefield, scaled_frame_tag

APERTURE_RADIUS = 5e-6	# flat-intensity initial wave: theta(a - r), a = 5 um (grid extent 20 um)
DZ_STEP = 10e-3			# drift subdivision for the dense cross-section (m)


def load_column(subdivide:float=None):
	"""Load basic_column.sea with the aperture source, optionally with split drifts.

	Parameters
	----------
	subdivide : float, optional
		Maximum drift length (metres); longer drifts are split into equal
		sub-drifts so the cross-section is logged densely. ``None`` (default)
		keeps the stored elements (planes at element exits and frame events
		only — the form saved to ``.sea``).

	Returns
	-------
	Microscope
		The column, with the source's ``wave_kind`` set to ``'aperture'``.
	"""
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "src", "pySEA", "rayTEM",
										 "microscopes", "basic_column.sea"))
	source = scope.sections[0].elements[0]
	source.wave_kind = "aperture"
	source.aperture_radius = APERTURE_RADIUS
	if subdivide is None:
		return scope
	sections = []
	for sec in scope.sections:
		elements = []
		for ele in sec.elements:
			if isinstance(ele, Drift) and ele.length > subdivide and not ele.name:
				n = int(np.ceil(ele.length / subdivide))
				elements += [Drift(length=ele.length / n) for _ in range(n)]
			else:
				ele._position = None		# let the section restack sequentially
				elements.append(ele)
		sections.append(MicroscopeSection(name=sec.name, elements=elements))
	return Microscope(name=scope.name, sections=sections)


def reconstruct_planes(planes):
	"""Reconstruct each logged scaled plane back to the physical wave.

	Parameters
	----------
	planes : list
		Scaled-plane Signals from a hybrid run.

	Returns
	-------
	list of tuple
		``(z, s, psi, dx, tag)`` per plane, with ``psi`` on its native grid
		``Δx = |s|·Δξ`` (handoff Eq 41 — no interpolation).
	"""
	out = []
	for p in planes:
		U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(p)
		psi, dx, dy = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		out.append((z, s, psi, dx, scaled_frame_tag(p)))
	return out


def plot_cross_section(recon, markers, crossovers, filename):
	"""Render |ψ| vs z in physical x and in the scaled coordinate ξ.

	Two panels share the z axis: the top shows |ψ(x, y=0, z)| in physical
	micrometres (the wave analog of the geometric ray diagram — the beam
	envelope spans nm at the foci to mm at the detector); the bottom shows the
	same planes in the scaled coordinate ξ = x/s, where the zooming grid keeps
	the internal structure visible at every z. Planes are individually
	normalized to their peak.

	Parameters
	----------
	recon : list of tuple
		``(z, s, psi, dx, tag)`` per plane from :func:`reconstruct_planes`.
	markers : dict
		``{label: z}`` element annotations.
	crossovers : Sequence[float]
		Crossover (focal-plane) z positions from the run.
	filename : str
		Output PNG path.

	Returns
	-------
	None
		Writes ``filename``.
	"""
	zs = np.array([r[0] for r in recon])
	z_edges = np.concatenate([[zs[0] - 1e-4], (zs[:-1] + zs[1:]) / 2, [zs[-1] + 1e-4]]) * 1e3
	n = recon[0][2].shape[1]

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	# top: physical coordinates (row extents differ per plane -> common grid)
	half = max(abs(r[3]) * n / 2 for r in recon)
	x_common = np.linspace(-half, half, 600)
	prof = np.zeros((len(recon), x_common.size))
	for i, (z, s, psi, dx, tag) in enumerate(recon):
		x = (np.arange(n) - n // 2) * dx
		row = np.abs(psi[psi.shape[0] // 2, :])
		prof[i] = np.interp(x_common, x, row / row.max(), left=0, right=0)
	x_edges = np.linspace(-half, half, x_common.size + 1) * 1e6
	ax1.pcolormesh(z_edges, x_edges, prof.T, cmap="magma", shading="flat")
	ax1.set_ylabel("x (µm)")
	ax1.set_title("basic_column hybrid scaled-Fresnel |ψ(x, y=0, z)| — physical coordinates")
	# bottom: scaled coordinate xi = x/s (one shared grid for the whole run)
	prof_xi = np.zeros((len(recon), n))
	for i, (z, s, psi, dx, tag) in enumerate(recon):
		row = np.abs(psi[psi.shape[0] // 2, :])
		prof_xi[i] = row / row.max()
	dxi = recon[0][3] / abs(recon[0][1])
	xi_edges = (np.arange(n + 1) - n / 2) * dxi * 1e6
	ax2.pcolormesh(z_edges, xi_edges, prof_xi.T, cmap="magma", shading="flat")
	ax2.set_ylabel("ξ = x/s (µm)")
	ax2.set_xlabel("z (mm)")
	ax2.set_title("same planes in the scaled coordinate (the grid the wave actually rides)")
	for ax in (ax1, ax2):
		for label, z in markers.items():
			ax.axvline(z * 1e3, color="w", lw=0.6, ls="--", alpha=0.6)
		for zc in crossovers:
			ax.axvline(zc * 1e3, color="cyan", lw=0.8, ls=":", alpha=0.9)
	for label, z in markers.items():
		ax1.text(z * 1e3, half * 1e6 * 0.95, label, color="w", rotation=90,
				 ha="right", va="top", fontsize=7)
	for zc in crossovers:
		ax1.text(zc * 1e3, -half * 1e6 * 0.95, "crossover", color="cyan", rotation=90,
				 ha="right", va="bottom", fontsize=7)
	fig.tight_layout()
	fig.savefig(filename, dpi=160)
	plt.close(fig)


def plot_xy_slices(scope, filename):
	"""Render |ψ(x, y)|² at the column's key planes on their native grids.

	Uses :meth:`Microscope.wavefield_at` to reconstruct the physical wave at
	the source exit, the C1 back-focal (crossover) plane, the ``sample``
	plane, the objective and projector crossovers, and the ``detector``.

	Parameters
	----------
	scope : Microscope
		A column already propagated with ``mode='hybrid'``.
	filename : str
		Output PNG path.

	Returns
	-------
	None
		Writes ``filename``.
	"""
	cross = scope.crossovers
	planes = [("source", 0.0), ("C1 back-focal", cross[0]),
			  ("sample", scope.named_positions["sample"]),
			  ("objective focus", cross[2] if len(cross) > 2 else cross[-1]),
			  ("projector focus", cross[-1]),
			  ("detector", scope.named_positions["detector"])]
	fig, axes = plt.subplots(2, 3, figsize=(11, 7))
	for ax, (label, z) in zip(axes.flat, planes):
		sig = scope.wavefield_at(z)
		data, dx, dy, lam, z_out = read_wavefield(sig)
		nn = data.shape[0]
		ext = np.array([-1, 1, -1, 1]) * (nn // 2) * dx * 1e6
		ax.imshow(np.abs(data) ** 2, extent=ext, origin="lower", cmap="magma")
		ax.set_title(f"{label}\nz = {z_out*1e3:.2f} mm   Δx = {dx*1e9:.3g} nm", fontsize=9)
		ax.set_xlabel("x (µm)", fontsize=8)
		ax.set_ylabel("y (µm)", fontsize=8)
		ax.tick_params(labelsize=7)
	fig.suptitle("basic_column hybrid scaled-Fresnel |ψ(x, y)|² — note the per-plane physical grids", y=1.0)
	fig.tight_layout()
	fig.savefig(filename, dpi=160)
	plt.close(fig)


def main():
	"""Run the full-column demo end-to-end and write the .sea result and figures.

	Returns
	-------
	None
		Writes the outputs listed in the module docstring and prints the
		crossover positions and a per-plane summary (z, s, Δx, energy).
	"""
	# 1) plain column: the result saved to .sea (element-exit + frame-event planes)
	scope = load_column()
	sset = scope.propagate_wave(mode="hybrid")
	if sset is not None:
		sset.to_sea("basic_column_scaled_wave.sea")
		n_planes = sset["U"].data.shape[0]
		print(f"saved basic_column_scaled_wave.sea ({n_planes} planes)")
	print("crossovers (m):", [round(z, 5) for z in scope.crossovers])

	recon_sparse = reconstruct_planes(scope._wave_scaled_planes)
	E0 = None
	print(f"{'z (mm)':>9} {'s':>10} {'dx (nm)':>10} {'energy/E0':>10}  tag")
	for (z, s, psi, dx, tag) in recon_sparse:
		e = (np.abs(psi) ** 2).sum() * dx * dx
		E0 = E0 or e
		print(f"{z*1e3:9.2f} {s:10.4g} {dx*1e9:10.3g} {e/E0:10.6f}  {tag or ''}")

	# 2) dense column for the cross-section figure
	dense = load_column(subdivide=DZ_STEP)
	dense.propagate_wave(mode="hybrid")
	recon = reconstruct_planes(dense._wave_scaled_planes)
	markers = {name: z for name, z in dense.named_positions.items()
			   if name and not name[-1] in "ab" and name != "G"}
	plot_cross_section(recon, markers, dense.crossovers,
					   "basic_column_scaled_wave_cross_section.png")
	plot_xy_slices(scope, "basic_column_scaled_wave_xy_slices.png")
	print("wrote basic_column_scaled_wave_cross_section.png, "
		  "basic_column_scaled_wave_xy_slices.png")


if __name__ == "__main__":
	main()
