"""Builder for the default generic TEM column (``basic_column.sea``).

Constructs a complete, instrument-agnostic microscope template:

1. **G** — gun section: electron ``Source`` (200 kV).
2. **C** — condenser section: three round lenses (C1–C3), each bracketed by a
   dipole pair (one pair before, one pair after), with a quadrupole pair at the
   section end.
3. **O** — objective section: upper/lower lenses OL1 & OL2, a dipole pair before
   OL1 and one after OL2, and a quadrupole at the very end.
4. **P** — projector section: four lenses (PL1–PL4), each with a pre and post
   dipole pair, ending in a zero-length named ``detector`` plane.

A *dipole pair* is two thin dipoles at the same plane whose kick axes are
rotated 45° with respect to each other (``axis=0`` and ``axis=π/4``). A
*quadrupole pair* is two thin quadrupoles of opposite-sign strength (a
quadrupole rotated 90° is exactly a sign flip; a true 45°/skew quadrupole is not
yet representable because ``Quadrapole`` has no axis parameter).

All deflector/stigmator strengths default to zero (an aligned column); the round
lenses carry placeholder focusing strengths meant to be calibrated per
instrument. Running this module rebuilds and saves ``basic_column.sea`` next to
this file.

Related
-------
assemblies.load_microscope : Reload the saved column from ``basic_column.sea``.
"""

from __future__ import annotations

import os
import numpy as xp

from pySEA.rayTEM.elements import Source, Drift, Lens, Dipole, Quadrapole
from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection


def dipole_pair(name: str, strength: float = 0.0) -> list:
	"""Build a dipole pair: two thin dipoles rotated 45° in the same plane.

	The first dipole kicks along ``axis=0`` (x) and the second along
	``axis=π/4``; both are zero-length so they act at the same z plane.

	Parameters
	----------
	name : str
		Base name; the two dipoles are named ``<name>a`` and ``<name>b``.
	strength : float, optional
		Angular kick of each dipole in radians, by default 0 (aligned column).

	Returns
	-------
	list of Dipole
		``[<name>a (0°), <name>b (45°)]``.
	"""
	return [Dipole(name=name + "a", strength=strength, axis=0.0),
			Dipole(name=name + "b", strength=strength, axis=float(xp.pi / 4))]


def quadrupole_pair(name: str, strength: float = 0.0) -> list:
	"""Build a quadrupole pair: two thin quadrupoles of opposite-sign strength.

	A quadrupole rotated by 90° is equivalent to a sign flip of its strength, so
	the representable orthogonal pair is ``(+K, −K)``. A true 45° (skew)
	quadrupole is not yet representable — ``Quadrapole`` has no axis parameter.

	Parameters
	----------
	name : str
		Base name; the two quadrupoles are named ``<name>a`` and ``<name>b``.
	strength : float, optional
		Strength magnitude ``K``, by default 0 (stigmators off).

	Returns
	-------
	list of Quadrapole
		``[<name>a (+K), <name>b (−K)]``.
	"""
	return [Quadrapole(name=name + "a", strength=strength),
			Quadrapole(name=name + "b", strength=-strength)]


def build_basic_column(voltage: float = 200.0) -> Microscope:
	"""Assemble the default generic TEM column.

	Parameters
	----------
	voltage : float, optional
		Accelerating voltage in kilovolts for the source, by default 200.

	Returns
	-------
	Microscope
		The assembled ``basic column`` microscope (G, C, O, P sections with a
		``detector`` plane at the very end).
	"""
	# 1) G — gun: the source, plus a drift into the condenser
	gun = MicroscopeSection(name="G", elements=[
		Source(name="G", voltage=voltage, size=(1e-3, 1e-3), np_xy=(3, 3),
			   angle=(5e-3, 5e-3), na_xy=(3, 3)),
		Drift(length=0.10),
	])

	# 2) C — condenser: three lenses, each with a dipole pair before and after,
	#    plus a quadrupole pair at the end
	c_elements = []
	for i, strength in enumerate([3.0, 3.5, 3.0], start=1):
		c_elements += dipole_pair(f"C{i}_Dpre")
		c_elements += [Lens(name=f"C{i}", strength=strength, length=0.02)]
		c_elements += dipole_pair(f"C{i}_Dpost")
		c_elements += [Drift(length=0.15)]
	c_elements += quadrupole_pair("CQ")
	c_elements += [Drift(length=0.10)]
	condenser = MicroscopeSection(name="C", elements=c_elements)

	# 3) O — objective: dipole pair, OL1, OL2, dipole pair, quadrupole at the end
	o_elements = []
	o_elements += dipole_pair("O_Dpre")
	o_elements += [Lens(name="OL1", strength=4.0, length=0.02), Drift(length=0.08),
				   Lens(name="OL2", strength=4.0, length=0.02)]
	o_elements += dipole_pair("O_Dpost")
	o_elements += [Drift(length=0.10), Quadrapole(name="OQ", strength=0.0), Drift(length=0.10)]
	objective = MicroscopeSection(name="O", elements=o_elements)

	# 4) P — projector: four lenses, each with pre and post dipole pairs
	p_elements = []
	for i, strength in enumerate([2.5, 3.0, 3.0, 2.5], start=1):
		p_elements += dipole_pair(f"PL{i}_Dpre")
		p_elements += [Lens(name=f"PL{i}", strength=strength, length=0.02)]
		p_elements += dipole_pair(f"PL{i}_Dpost")
		p_elements += [Drift(length=0.12)]
	# 5) detector plane at the very end (zero-length named marker)
	p_elements += [Drift(name="detector", length=0.0)]
	projector = MicroscopeSection(name="P", elements=p_elements)

	return Microscope(name="basic column", sections=[gun, condenser, objective, projector])


if __name__ == "__main__":
	scope = build_basic_column()
	here = os.path.dirname(os.path.abspath(__file__))
	target = os.path.join(here, "basic_column.sea")
	scope.to_sea(target)
	print("saved", target)
	print(repr(scope))
