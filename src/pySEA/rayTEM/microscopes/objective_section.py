"""Builder for the standalone objective section (``objective_section.sea``).

The objective of :mod:`basic_column` on its own, with a source and a drift in
front of it so a beam can be delivered to OL1 under known conditions. It exists
because the full column is the wrong instrument for studying an objective: a
lens aberration is a nanometre-scale effect, and over a metre of column it is
~1e-4 of the beam width — invisible in any plot of the whole thing, and
expensive to propagate a wave through.

Layout:

1. **G** — a 200 kV :class:`Source` and a drift to the objective. The source's
   own aperture/illumination is set by the caller; the defaults here give a
   collimated 30 mrad pencil at OL1.
2. **O** — *the objective section of* ``basic_column``, taken from it rather
   than re-declared, so OL1 (f = 3 mm) and OL2 (f = 10 mm) cannot drift apart
   from the real column's. It already carries 0.18 m of drift after the lenses;
   this adds the leading drift the column's own gun/condenser used to provide.

Running this module rebuilds and saves ``objective_section.sea`` next to this
file.

Related
-------
basic_column : The full column this section is lifted from.
assemblies.load_microscope : Reload the saved scope.

Examples
--------
>>> from pySEA.rayTEM.assemblies import load_microscope     # doctest: +SKIP
>>> scope = load_microscope("objective_section.sea")        # doctest: +SKIP
"""

from __future__ import annotations

import os

from ..assemblies import Microscope, MicroscopeSection
from ..elements import Drift, Source
from .basic_column import build_basic_column

#: Drift from the source to the objective's first element (metres).
ENTRANCE_DRIFT = 0.05


def build_objective_section(alpha: float = 30e-3, voltage: float = 200,
							n_rays: int = 15, wave_shape: tuple = (256, 256),
							wave_oversample: float = 1.25) -> Microscope:
	"""Assemble the objective section behind a source.

	Parameters
	----------
	alpha : float, optional
		**Convergence semi-angle at the sample** (radians), by default 0.03 —
		the quantity every axial aberration is measured against. The source
		emits a collimated fan of height ``alpha·f_OL1``, which OL1 turns into
		exactly that convergence.

		Note the angle is the ray's **total** deflection, not its x component:
		OL1 is thick, so it also rotates the ray by its Larmor angle
		(``KL = 1.30`` rad here). A collimated ray at 240 µm leaves the body at
		30.000 mrad total, of which only 8.08 mrad is in x — the rest has been
		rotated into y. Reading ``xt`` alone under-reports the convergence by
		``cos(KL)``.
	voltage : float, optional
		Accelerating voltage in kilovolts, by default 200.
	n_rays : int, optional
		Number of rays across the fan, by default 15.
	wave_shape : tuple, optional
		Wave-optics grid ``(ny, nx)``, by default ``(256, 256)``.
	wave_oversample : float, optional
		Grid half-extent as a multiple of the aperture radius, by default 1.25.
		Deliberately tight: the aberration screen is evaluated over the whole
		grid, not just inside the aperture, and its gradient goes as ``r³`` — so
		every extra bit of empty grid makes the screen harder to sample, not
		easier. Too small wraps the beam.

	Returns
	-------
	Microscope
		Two sections, ``G`` and ``O``.

	Raises
	------
	None

	Related
	-------
	basic_column.build_basic_column : Supplies the ``O`` section verbatim.

	Notes
	-----
	The grid is sized by the **aberration**, not by the beam. A screen
	``χ = -k C₃₀ r⁴/4f⁴`` has gradient ``k C₃₀ r³/f⁴``, so the samples needed
	go as ``C₃₀ α⁴`` — and it is the grid **corner** that binds, at ``√2``
	times the half-extent, i.e. 4× the requirement of the edge. That is also
	why ``wave_oversample`` is tight: empty grid beyond the aperture costs
	sampling rather than buying safety.

	256 is ample here because a thick lens now distributes its screen over
	``elements.MEDIUM_SLICES`` slices, so each applies ``1/16`` of the phase.
	At 30 mrad it carries ``C₃₀`` up to ~0.5 mm; 1 mm still aliases, and
	``_check_screen_sampling`` says so rather than quietly returning nonsense.

	Examples
	--------
	>>> scope = build_objective_section(alpha=15e-3)         # doctest: +SKIP
	"""
	column = build_basic_column()
	# Rewrap the elements in a fresh section: the lifted one still reports the
	# z it occupied inside the full column (0.49 m), and Microscope inserts a
	# drift to honour that, which would put half a metre of nothing in front of
	# the objective. Same element objects, so OL1/OL2 stay identical.
	objective = MicroscopeSection(name="O", elements=column["O"].elements,
								  position=ENTRANCE_DRIFT)
	f_ol1 = 0.003									# OL1's focal length in basic_column
	aperture = alpha * f_ol1
	source = Source(name="G", voltage=voltage,
					size=(aperture, 0), np_xy=(n_rays, 1),
					angle=(0, 0), na_xy=(1, 1),
					wave_shape=wave_shape,
					wave_extent=2 * wave_oversample * aperture,
					wave_kind="aperture", aperture_radius=aperture)
	gun = MicroscopeSection(name="G", elements=[source, Drift(length=ENTRANCE_DRIFT)])
	return Microscope(name="objective section", sections=[gun, objective])


if __name__ == "__main__":
	scope = build_objective_section()
	here = os.path.dirname(os.path.abspath(__file__))
	target = os.path.join(here, "objective_section.sea")
	scope.to_sea(target)
	print("saved", target)
	print("total length (m):", round(sum(s.length for s in scope.sections), 4))
