"""Scaled-Fresnel wave propagation through the basic_column template.

Demonstrates ``propagate_wave_scaled`` on ``microscopes/basic_column.sea``:
a flat-intensity hard-aperture wavefunction Θ(a−r) (200 kV; the source's
``wave_kind`` is set to ``'aperture'``) is carried through the
accelerator drift and the C1 condenser lens on the zooming (ξ, η) grid — the lens is
absorbed into the curvature state R, so its centimetre-scale focal phase never
touches the sampled array (the fixed-grid mode cannot sample it; see
``docs/wave-optics-sampling.md``). Long drifts are subdivided so the beam's
geometric contraction ``s(z)`` is logged densely, propagation stops just
before the C1 beam crossover (the chart's ``s = 0`` singularity — automatic
chart switching through crossovers is tracked as GitHub issue #2), and the
guard's actionable error is demonstrated.

Outputs (written to the current working directory):

- ``basic_column_scaled_wave.sea`` — the stacked scaled result: U(z, η, ξ)
  plus companion s(z)/R(z)/tau(z) Signals on the shared plane-z axis.
- ``basic_column_scaled_wave_cross_section.png`` — |ψ(x, y=0, z)| in physical
  coordinates (the wave analog of the geometric ``plot2D`` ray diagram).
- ``basic_column_scaled_wave_xy_slices.png`` — |ψ(x, y)|² at several planes,
  each on its own physical grid (Δx = |s|·Δξ).

Run from anywhere with the rayTEM environment active:

	python examples/04_scaledWave_basic_column.py
"""

import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from pySEA.rayTEM.assemblies import load_microscope
from pySEA.rayTEM.elements import Drift, Source
from pySEA.rayTEM import waveoptics as wo
from pySEA.rayTEM.seashells import (read_scaled_wavefield, make_scaled_wave_signalset,
									sea_available)

S_STOP = 0.02			# log the last plane at |s| = S_STOP, just before the crossover guard
APERTURE_RADIUS = 5e-6	# flat-intensity initial wave: theta(a - r), a = 5 um (grid extent 20 um)
DZ_STEP = 5e-3		# drift subdivision for dense s(z) logging (m)


def propagate_until_crossover(scope):
	"""Thread the scaled wave through the column, stopping before the crossover.

	Walks every element of every section with
	:meth:`Element.propagate_wave_scaled`, subdividing drifts into ``DZ_STEP``
	segments so the contraction ``s(z)`` is densely logged. When the next
	segment would push ``|s|`` below ``S_STOP`` the remaining distance to
	``|s| = S_STOP`` is propagated instead, the plane is logged, and the loop
	stops (the chart is singular at the crossover itself — issue #2).

	Parameters
	----------
	scope : Microscope
		The loaded basic column.

	Returns
	-------
	tuple
		``(planes, markers, guard_message)`` — the logged scaled-plane Signals,
		``{label: z}`` annotations for plotting, and the crossover guard's
		error text (demonstrated on the remaining distance).

	Raises
	------
	ValueError
		Only from programming errors; the crossover guard is caught and
		returned as ``guard_message``.
	"""
	source = scope.sections[0].elements[0]
	# select the flat-intensity aperture wavefunction on the source's one
	# field generator (the stored template defaults to wave_kind='gaussian')
	source.wave_kind = "aperture"
	source.aperture_radius = APERTURE_RADIUS
	state = source.wave_scaled()
	planes = [state]
	markers = {}
	guard_message = None
	for section in scope.sections:
		for ele in section.elements:
			if isinstance(ele, Source):
				continue
			if isinstance(ele, Drift) and ele.length > 0:
				remaining = ele.length
				while remaining > 1e-12:
					_, _, _, _, s, R, _, z = read_scaled_wavefield(state)
					converging = R is not None and not np.isinf(R) and R < 0 and s > 0
					# finer steps as the chart converges, so the cone is logged densely
					step = min(DZ_STEP, -R / 15) if converging else DZ_STEP
					dz = min(step, remaining)
					if converging:
						# s falls to S_STOP after dz_stop = |R|*(1 - S_STOP/s)
						dz_stop = -R * (1 - S_STOP / s)
						if dz >= dz_stop:
							state = Drift(length=dz_stop).propagate_wave_scaled(state)
							planes.append(state)
							markers["stop (s=%.2g)" % S_STOP] = z + dz_stop
							markers["crossover"] = z - R
							try:	# demonstrate the actionable guard on the rest
								Drift(length=remaining - dz_stop).propagate_wave_scaled(state)
							except ValueError as err:
								guard_message = str(err)
							return planes, markers, guard_message
					state = Drift(length=dz).propagate_wave_scaled(state)
					planes.append(state)
					remaining -= dz
				continue
			state = ele.propagate_wave_scaled(state)
			if getattr(ele, "length", 0) != 0 or ele.kind == "Aperture":
				planes.append(state)
			if getattr(ele, "length", 0) != 0:
				_, _, _, _, _, _, _, z = read_scaled_wavefield(state)
				markers[ele.name or ele.kind] = z - ele.length / 2
	return planes, markers, guard_message


def reconstruct_planes(planes):
	"""Reconstruct each logged scaled plane back to the physical wave.

	Parameters
	----------
	planes : list
		Scaled-plane Signals from :func:`propagate_until_crossover`.

	Returns
	-------
	list of tuple
		``(z, s, psi, dx)`` per plane, with ``psi`` on its native grid
		``Δx = |s|·Δξ`` (handoff Eq 41 — no interpolation).
	"""
	out = []
	for p in planes:
		U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(p)
		psi, dx, dy = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		out.append((z, s, psi, dx))
	return out


def plot_cross_section(recon, markers, filename):
	"""Render |ψ(x, y=0)| vs z in physical coordinates (wave analog of plot2D).

	Each plane's central row is interpolated onto a common physical x axis;
	every plane is normalized to its own peak so the contracting beam stays
	visible across the ~1/s amplitude growth toward focus.

	Parameters
	----------
	recon : list of tuple
		``(z, s, psi, dx)`` per plane from :func:`reconstruct_planes`.
	markers : dict
		``{label: z}`` annotations (elements, crossover).
	filename : str
		Output PNG path.

	Returns
	-------
	None
		Writes ``filename``.
	"""
	zs = np.array([r[0] for r in recon])
	half = max(abs(r[3]) * r[2].shape[1] / 2 for r in recon)
	x_common = np.linspace(-half, half, 512)
	profile = np.zeros((len(recon), x_common.size))
	for i, (z, s, psi, dx) in enumerate(recon):
		n = psi.shape[1]
		x = (np.arange(n) - n // 2) * dx
		row = np.abs(psi[psi.shape[0] // 2, :])
		profile[i] = np.interp(x_common, x, row / row.max(), left=0, right=0)
	fig, ax = plt.subplots(figsize=(11, 4.5))
	# non-uniform plane spacing: pcolormesh with explicit z edges
	z_edges = np.concatenate([[zs[0] - 1e-4], (zs[:-1] + zs[1:]) / 2, [zs[-1] + 1e-4]])
	x_edges = np.linspace(-half, half, x_common.size + 1)
	m = ax.pcolormesh(z_edges * 1e3, x_edges * 1e6, profile.T, cmap="magma", shading="flat")
	for label, z in sorted(markers.items(), key=lambda kv: kv[1]):
		color = "cyan" if "crossover" in label else "w"
		ax.axvline(z * 1e3, color=color, lw=0.8, ls="--", alpha=0.8)
		# crossover label at the bottom so it never collides with the stop label
		ytext = -0.95 if "crossover" in label else 0.95
		ax.text(z * 1e3, half * 1e6 * ytext, label, color=color, rotation=90,
				ha="right", va="top" if ytext > 0 else "bottom", fontsize=8)
	ax.set_xlabel("z (mm)")
	ax.set_ylabel("x (µm)")
	ax.set_title("basic_column scaled-Fresnel |ψ(x, y=0, z)| (per-plane normalized)")
	fig.colorbar(m, ax=ax, label="|ψ| / max per plane")
	fig.tight_layout()
	fig.savefig(filename, dpi=160)
	plt.close(fig)


def plot_xy_slices(recon, filename, n_panels=6):
	"""Render |ψ(x, y)|² at several planes, each on its native physical grid.

	Panels are chosen evenly across the logged planes (always including the
	first and last), with each panel's axis extent set by its own pixel size
	``Δx = |s|·Δξ`` — the zooming grid is the point of the scaled mode.

	Parameters
	----------
	recon : list of tuple
		``(z, s, psi, dx)`` per plane from :func:`reconstruct_planes`.
	filename : str
		Output PNG path.
	n_panels : int, optional
		Number of planes to show, by default 6.

	Returns
	-------
	None
		Writes ``filename``.
	"""
	idx = np.unique(np.linspace(0, len(recon) - 1, n_panels).round().astype(int))
	fig, axes = plt.subplots(2, (len(idx) + 1) // 2, figsize=(3.4 * ((len(idx) + 1) // 2), 6.6))
	for ax, i in zip(axes.flat, idx):
		z, s, psi, dx = recon[i]
		n = psi.shape[0]
		ext = np.array([-1, 1, -1, 1]) * (n // 2) * dx * 1e6
		ax.imshow(np.abs(psi) ** 2, extent=ext, origin="lower", cmap="magma")
		ax.set_title(f"z = {z*1e3:.1f} mm   s = {s:.3f}\nΔx = {dx*1e9:.2f} nm", fontsize=9)
		ax.set_xlabel("x (µm)", fontsize=8)
		ax.set_ylabel("y (µm)", fontsize=8)
		ax.tick_params(labelsize=7)
	for ax in axes.flat[len(idx):]:
		ax.axis("off")
	fig.suptitle("basic_column scaled-Fresnel |ψ(x, y)|² — note the zooming physical grid", y=1.0)
	fig.tight_layout()
	fig.savefig(filename, dpi=160)
	plt.close(fig)


def main():
	"""Run the demo end-to-end and write the .sea result and both figures.

	Returns
	-------
	None
		Writes the outputs listed in the module docstring and prints a
		per-plane summary (z, s, R, Δx, energy) plus the crossover guard's
		error text.
	"""
	here = os.path.dirname(os.path.abspath(__file__))
	scope = load_microscope(os.path.join(here, "..", "src", "pySEA", "rayTEM",
										 "microscopes", "basic_column.sea"))
	planes, markers, guard_message = propagate_until_crossover(scope)

	# stack + save the scaled result (U + s/R/tau companions on the plane-z axis)
	reads = [read_scaled_wavefield(p) for p in planes]
	U = np.stack([r[0] for r in reads])
	dxi, deta, lam = reads[0][1], reads[0][2], reads[0][3]
	sset = make_scaled_wave_signalset(U, dxi, deta, lam,
									  s=[r[4] for r in reads], R=[r[5] for r in reads],
									  tau=[r[6] for r in reads], z=[r[7] for r in reads],
									  name="basic column scaled wave")
	if sea_available and sset is not None:
		sset.to_sea("basic_column_scaled_wave.sea")
		print("saved basic_column_scaled_wave.sea "
			  f"({len(planes)} planes, U {U.shape}, dxi = {dxi*1e9:.2f} nm)")

	recon = reconstruct_planes(planes)
	E = [(np.abs(psi) ** 2).sum() * dx * dx for _, _, psi, dx in recon]
	print(f"{'z (mm)':>8} {'s':>8} {'dx (nm)':>9} {'energy/E0':>10}")
	for (z, s, psi, dx), e in zip(recon, E):
		print(f"{z*1e3:8.2f} {s:8.4f} {dx*1e9:9.2f} {e/E[0]:10.6f}")
	if guard_message:
		print("\ncrossover guard (expected, issue #2):\n  " + guard_message)

	plot_cross_section(recon, markers, "basic_column_scaled_wave_cross_section.png")
	plot_xy_slices(recon, "basic_column_scaled_wave_xy_slices.png")
	print("wrote basic_column_scaled_wave_cross_section.png, "
		  "basic_column_scaled_wave_xy_slices.png")


if __name__ == "__main__":
	main()
