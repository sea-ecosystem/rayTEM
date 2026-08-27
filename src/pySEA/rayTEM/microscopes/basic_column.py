"""Builder for the default generic TEM column (``basic_column.sea``).

Constructs a complete, instrument-agnostic 200 kV microscope template at
realistic scales — the source initializes all three propagation
representations consistently (geometric rays, beam-envelope covariance, and a
coherent wavefunction), and every element sits at a plausible physical z:

1. **G** — gun section: 200 kV ``Source`` (2.5 µm RMS beam, 0.1 mrad
   divergence), 12 cm accelerator drift into the condenser.
2. **C** — condenser section: three round lenses C1 (f = 45 mm), C2 (30 mm),
   C3 (90 mm), each bracketed by a dipole pair (one before, one after), a
   quadrupole pair at the section end.
3. **O** — objective section: OL1 (f = 2 mm, a short high-excitation
   probe-forming lens) with the sample plane at its back focal plane
   (~0.9 mm past the exit), OL2 (f = 10 mm) 4 mm later, a dipole pair before
   OL1 and one after OL2, and a quadrupole at the very end.
4. **P** — projector section: PL1–PL4 (f = 25/40/60/80 mm), each with a pre
   and post dipole pair, a 30 cm camera drift, and a zero-length named
   ``detector`` plane at the very end (total column ≈ 1.3 m).

Lens strengths are derived from focal lengths through the thick-lens relation
``1/f = K·sin(K·L)`` (the form used by ``Lens.transfer_matrix``), so each lens
carries a physical bore length and the resulting Larmor rotation.

A *dipole pair* is two thin dipoles at the same plane whose kick axes are
rotated 45° with respect to each other (``axis=0`` and ``axis=π/4``). A
*quadrupole pair* is two thin quadrupoles of opposite-sign strength (a
quadrupole rotated 90° is exactly a sign flip; a true 45°/skew quadrupole is
not yet representable because ``Quadrapole`` has no axis parameter).

All deflector/stigmator strengths default to zero (an aligned column).
Running this module rebuilds and saves ``basic_column.sea`` next to this file.

Notes
-----
The wave representation initializes a *coherent* 200 kV Gaussian wavefunction
matching the source size on a 20 µm / 256² grid — exact at the source plane
(and in its Fourier/qx-qy plane). Propagating it through cm-focal-length
lenses on this fixed grid is, however, outside the sampling limit of the
angular-spectrum method: the lens phase gradient ``k·x/f`` exceeds the grid
Nyquist ``π/dx`` by 10–100×, because a µm-scale 200 kV beam spans ~10⁵–10⁶
wavelengths of phase space. Use the ray and covariance modes for full-column
transport; use the wave mode near planes of interest.

Related
-------
assemblies.load_microscope : Reload the saved column from ``basic_column.sea``.
"""

from __future__ import annotations

import os
import numpy as xp

from pySEA.rayTEM.elements import Source, Drift, Lens, Dipole, Quadrapole, Aperture
from pySEA.rayTEM.assemblies import Microscope, MicroscopeSection


def strength_for_focal_length(f: float, length: float) -> float:
	r"""Solve the thick-lens strength ``K`` that yields focal length ``f``.

	Inverts the thick-lens focusing relation used by ``Lens.transfer_matrix``
	(Brown 1983 alternate form),

	.. math:: 1/f = K \sin(K L),

	on the first branch ``K L < π/2``, so a physical bore length can be kept
	while specifying optics by focal length.

	Parameters
	----------
	f : float
		Target focal length in metres.
	length : float
		Lens bore length ``L`` in metres (must be > 0).

	Returns
	-------
	float
		Lens strength ``K`` such that ``K·sin(K·length) = 1/f``.

	Raises
	------
	ValueError
		If ``f`` is shorter than the first-branch minimum ``sin(π/2)·π/(2L)``
		(i.e. no solution with ``K L < π/2``).

	Notes
	-----
	The first-branch bound means ``f`` must satisfy ``f ≥ 2·length/π``.
	"""
	from scipy.optimize import brentq
	K_max = (xp.pi / 2 - 1e-9) / length
	if 1.0 / f > K_max * xp.sin(K_max * length):
		raise ValueError(f"f={f} m is unreachable with L={length} m on the first branch; need f >= {2*length/xp.pi:.4f} m.")
	return brentq(lambda K: K * xp.sin(K * length) - 1.0 / f, 1e-9, K_max)


def round_lens(name: str, f: float, length: float) -> Lens:
	"""Build a thick round lens specified by focal length.

	Parameters
	----------
	name : str
		Lens name.
	f : float
		Focal length in metres.
	length : float
		Bore length in metres.

	Returns
	-------
	Lens
		Lens whose strength satisfies ``K·sin(K·length) = 1/f``.
	"""
	return Lens(name=name, strength=strength_for_focal_length(f, length), length=length)


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
	"""Assemble the default generic TEM column at realistic scales.

	The source seeds all three representations consistently: a 5×5 grid of rays
	over ±2.5 µm with ±0.1 mrad fans, a diagonal covariance with the same RMS
	size/divergence, and a coherent 200 kV Gaussian wavefunction of the same
	transverse size on a 20 µm / 256² grid.

	Parameters
	----------
	voltage : float, optional
		Accelerating voltage in kilovolts, by default 200.

	Returns
	-------
	Microscope
		The assembled ``basic column`` (G, C, O, P sections, ``detector`` plane
		at ≈1.3 m).
	"""
	beam_size = 2.5e-6		# RMS transverse size entering the condenser (m)
	beam_angle = 1e-4		# RMS divergence (rad)

	# 1) G — gun: the source, plus the accelerator drift into the condenser.
	#    The stated emission current: 1 nA. Everything downstream derives from it.
	gun = MicroscopeSection(name="G", elements=[
		Source(name="G", voltage=voltage, beam_current=1e-9,
			   size=(beam_size, beam_size), np_xy=(5, 5),
			   angle=(beam_angle, beam_angle), na_xy=(3, 3),
			   wave_shape=(256, 256), wave_extent=8 * beam_size,
			   wave_kind="gaussian"),
		Drift(length=0.12),
	])

	# OL1 and its working distance are computed FIRST because the condenser
	# section's trailing drift compensates for them (see below).
	# OL1: a short, strong probe-forming objective (f = 2 mm, 2 mm bore --
	# f >= 2L/pi bounds the bore), with the SAMPLE AT ITS BACK FOCAL PLANE.
	# That placement is what makes a fixed-excitation objective work the way
	# a real STEM's does: the condensers deliver a wide, nearly parallel
	# beam, and OL1 alone converts radius into angle (alpha = r/f) while
	# demagnifying the source hard -- measured, this geometry reaches
	# ~56 mrad at a ~12 nm probe, smoothly, so a 30 mrad probe is solved by
	# the condensers with the objective untouched. The working distance is
	# computed from the lens's own thick block (a parallel ray (h, 0) exits
	# the body as (A*h, C*h) and crosses the axis wd = -A/C later), so it
	# tracks any change to f or the bore instead of silently rotting.
	ol1 = round_lens("OL1", f=0.002, length=0.002)
	body = ol1.transfer_block(axis="x")
	wd = -float(body[0][0]) / float(body[1][0])

	# 2) C — condenser: C1/C2/C3, each with a dipole pair before and after,
	#    plus a quadrupole pair at the end
	c_elements = []
	for name, f, gap in [("C1", 0.045, 0.07), ("C2", 0.030, 0.09), ("C3", 0.090, 0.10)]:
		c_elements += dipole_pair(f"{name}_Dpre")
		c_elements += [round_lens(name, f=f, length=0.02)]
		c_elements += dipole_pair(f"{name}_Dpost")
		if name == "C1":
			# the condenser aperture: the beam-defining hole after C1. With the
			# gun crossover imaged onto it the whole current passes; defocused,
			# it cuts current (and with it the phase space downstream states
			# can spend). 10 µm: smaller than the ~21 µm unfocused beam here.
			c_elements += [Drift(length=0.04), Aperture(name="CA", radius=10e-6),
						   Drift(length=gap - 0.04)]
		else:
			c_elements += [Drift(length=gap)]
	c_elements += quadrupole_pair("CQ")
	# not the historical 0.05: OL1's bore shrank and a working distance was
	# inserted before the sample (below); this drift absorbs the difference so
	# the sample, OL2, and detector keep their exact z positions
	c_elements += [Drift(length=0.06 - ol1.length - wd)]
	condenser = MicroscopeSection(name="C", elements=c_elements)

	# 3) O — objective: OL1/OL2 twin around a 4 mm sample gap, dipole pair
	#    before OL1 and after OL2, quadrupole at the very end
	o_elements = []
	o_elements += dipole_pair("O_Dpre")
	# the gap after the sample marker is NAMED because repair() merges a
	# named drift with an unnamed drift that follows it -- an anonymous gap
	# here would be absorbed into the zero-length "sample" marker, silently
	# moving the measured sample plane 2 mm off the back focal plane
	o_elements += [ol1,		# built above, sample at its back focal plane
				   Drift(length=wd), Drift(name="sample", length=0.0),
				   Drift(name="sample_gap", length=0.004),
				   round_lens("OL2", f=0.010, length=0.01)]
	o_elements += dipole_pair("O_Dpost")
	o_elements += [Drift(length=0.06), Quadrapole(name="OQ", strength=0.0), Drift(length=0.12)]
	objective = MicroscopeSection(name="O", elements=o_elements)

	# 4) P — projector: PL1–PL4, each with pre and post dipole pairs, then the
	#    camera drift and the detector plane
	p_elements = []
	for name, f, gap in [("PL1", 0.025, 0.05), ("PL2", 0.040, 0.07),
						 ("PL3", 0.060, 0.09), ("PL4", 0.080, 0.0)]:
		p_elements += dipole_pair(f"{name}_Dpre")
		p_elements += [round_lens(name, f=f, length=0.015)]
		p_elements += dipole_pair(f"{name}_Dpost")
		if gap:
			p_elements += [Drift(length=gap)]
	p_elements += [Drift(length=0.30)]			# camera length to the detector
	# 5) detector plane (zero-length named marker), plus a short tail: a
	#    conjugate plane landing EXACTLY on the column's last z has no
	#    interval around it, so plane searches would silently skip the
	#    detector. 1 cm of nothing keeps it an interior plane.
	p_elements += [Drift(name="detector", length=0.0), Drift(length=0.01)]
	projector = MicroscopeSection(name="P", elements=p_elements)

	return Microscope(name="basic column", sections=[gun, condenser, objective, projector])


if __name__ == "__main__":
	scope = build_basic_column()
	here = os.path.dirname(os.path.abspath(__file__))
	target = os.path.join(here, "basic_column.sea")
	scope.to_sea(target)
	print("saved", target)
	print("total length (m):", round(sum(s.length for s in scope.sections), 4))
