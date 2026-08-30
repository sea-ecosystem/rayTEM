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

# figures land here, resolved from the script so any cwd works (figs/ is gitignored)
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)


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


def plot_cross_section(scope, filename):
	r"""Render |ψ| vs z in physical x and in the scaled coordinate ξ.

	Both panels are :meth:`Microscope.show` with ``kind='wave-hybrid'`` --
	the wave analog of the geometric ray diagram, with the element and
	crossover annotations it already draws. The only difference between them
	is ``coordinates``: the top panel is physical micrometres, where the beam
	spans nanometres at a focus and millimetres at the detector; the bottom is
	the reduced coordinate :math:`\xi = x/s` the field actually rides on,
	where one grid keeps the internal structure resolved at every z. Moving
	between the two is the whole point of the scaled representation.

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

	Raises
	------
	None

	Related
	-------
	assemblies.Microscope.show : What both panels are.
	assemblies._scaled_wave_cross_section : The renderer behind it.
	"""
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	scope.show(kind="wave-hybrid", plt_ax=ax1, regenerate=False,
			   title="basic_column hybrid scaled-Fresnel |ψ(x, y=0, z)| — physical coordinates")
	scope.show(kind="wave-hybrid", plt_ax=ax2, regenerate=False, coordinates="scaled",
			   title="the same planes in the scaled coordinate the wave actually rides")
	fig.tight_layout()
	fig.savefig(filename, dpi=160)
	plt.close(fig)


def plot_xy_slices(scope, filename):
	"""Render |ψ(x, y)|² at the column's key planes on their native grids.

	Uses :meth:`Microscope.show` with a ``plane``, which reconstructs the
	physical wave there and lets the wavefield ``Signal`` draw itself on its
	own calibrated axes. The planes shown are the source exit, the C1 back-focal (crossover) plane, the ``sample``
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
		# show() reconstructs the plane and lets the wavefield Signal draw
		# itself on its own calibrated axes -- no extent arithmetic here
		scope.show(kind="wave-hybrid", plane=float(z), plt_ax=ax, regenerate=False)
		dx = scope.wavefield_at(z).dimensions['x'].scale
		ax.set_title(f"{label}\nz = {z*1e3:.2f} mm   Δx = {dx*1e9:.3g} nm", fontsize=9)
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
	plot_cross_section(dense, os.path.join(FIGS, "04_scaledWave_cross_section.png"))
	plot_xy_slices(scope, os.path.join(FIGS, "04_scaledWave_xy_slices.png"))
	print(f"wrote {FIGS}/04_scaledWave_cross_section.png, "
		  f"{FIGS}/04_scaledWave_xy_slices.png")


if __name__ == "__main__":
	main()
