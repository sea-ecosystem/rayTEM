"""Spherical aberration on the objective, in rays and in the wave.

Uses ``microscopes/objective_section.sea`` — basic_column's own objective
(OL1, f = 3 mm EFL, 0.08 mm bore) behind a source — rather than the whole
column, because an aberration is a nanometre-scale effect and over a metre of
column it is ~1e-4 of the beam width: invisible in any plot of the whole
thing, and expensive to propagate a wave through.

Six panels, three rows of two. Left column ideal, right column aberrated, so
each row is one comparison:

A, B  the RAY caustic through focus.
C, D  the WAVE, |psi(x, z)| over the same window — the same caustic, built by
      a completely different calculation.
E     the focal surface: the ISOLATED thin lens follows the closed form
      -C30 alpha^2 exactly, which validates the implementation. OL1's bore is
      0.08 mm (KL = 0.16, nearly thin), so its distributed aberration lands
      within a few percent of the same closed form — the panel measures the
      delivered fraction rather than assuming it.
F     the focus itself, as a Strehl loss.

ALPHA is the convergence semi-angle AT THE SAMPLE, and it is the ray's total
deflection: OL1's body rotates the ray by its (small) Larmor angle, so the
x-component alone under-reports.

Why C30 = 4.5 um. Everything scales as C30*alpha^4, so at 30 mrad the choice
is narrow. A corrected instrument (C30 < 300 nm) is well under a radian of
peak phase — diffraction-limited, which is the POINT of correction and shows
nothing. At the other end, tens of um simply destroy the focus (10 um is
already Strehl 0.10), which shows nothing either. 4.5 um lands near Strehl
0.6: a focus that is visibly degraded and still recognisable, which is the
regime where the number means something.

The delivered aberration is measured in panel E as the ratio of OL1's fitted
focal-surface curvature to the closed form; with the near-thin bore it is
~0.97, so the Rayleigh quarter-wave limit at 30 mrad sits essentially at the
thin-lens value.

Run: python examples/06_aberratedObjective.py   (writes figures/)
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pySEA.rayTEM import Source, Drift, Lens, MicroscopeSection, Microscope
from pySEA.rayTEM.aberrations import Aberrations
from pySEA.rayTEM.microscopes.objective_section import build_objective_section
from pySEA.rayTEM.seashells import read_wavefield

ALPHA, C30, F_OL = 30e-3, 4.5e-6, 3e-3	# F_OL follows basic_column's OL1 (f = 3 mm EFL)
N_WAVE, N_PLANES = 256, 80
LAM = 2.5078e-12


def scope(c30, **kw):
	"""Build the objective section, optionally aberrated.

	Parameters
	----------
	c30 : float
		Spherical aberration coefficient (metres); 0 for an ideal objective.
	**kw
		Forwarded to :func:`build_objective_section`.

	Returns
	-------
	Microscope
		The objective section, not yet propagated.
	"""
	m = build_objective_section(alpha=ALPHA, wave_shape=(N_WAVE, N_WAVE), **kw)
	if c30:
		m["OL1"].aberrations = Aberrations({'C30': c30})
	return m


def focal_window(c30, z_lo, z_hi):
	"""Propagate the objective with the focal window densely sampled.

	Parameters
	----------
	c30 : float
		Spherical aberration coefficient (metres); 0 for an ideal objective.
	z_lo, z_hi : float
		Window bounds (metres); ``N_PLANES`` planes are logged across them.

	Returns
	-------
	Microscope
		The propagated column, ready for :meth:`Microscope.show` (the
		cross-section) and :meth:`Microscope.wavefield_at` (the focal
		line-out on its native, picometre grid).

	Notes
	-----
	:meth:`Microscope.subdivided` cuts the drifts at the requested absolute z,
	which is what puts a logged plane on each sample of the window. It cuts
	only UNNAMED drifts -- the focus here sits inside the 3 mm drift ahead of
	the specimen, so the ``sample`` marker survives, the lengths stay positive
	and z stays monotonic.
	"""
	m = scope(c30).subdivided(list(np.linspace(z_lo, z_hi, N_PLANES)))
	m.propagate_wave(mode="hybrid", absorb=0.0)
	return m


# ---- the focal window, from the ideal ray crossover ----------------------
ideal, aberrated = scope(0.0), scope(C30)
for m in (ideal, aberrated):
	m.propagate_ray()
Z_PAR = float(ideal.conjugate_planes(axis="x")["diff"][0])
# Near a focus the meaningful scales are DIFFRACTION ones, not the aberration's:
# the depth of focus lambda/alpha^2 and the Airy radius 0.61*lambda/alpha. Framing
# on the aberration instead gives a window one depth of focus wide, in which the
# wave is a featureless blob. Sized this way the aberration is seen in
# proportion -- which is the honest picture at Strehl 0.6: a fraction of the
# depth of focus, not a catastrophe.
#
# The fraction of the nominal C30 that OL1 actually delivers is MEASURED in
# panel E (fitted focal-surface curvature / closed form) and used in panel F's
# annotations; with the 0.08 mm bore it comes out ~0.97 — nearly thin.
DOF = LAM / ALPHA ** 2						# depth of focus
AIRY = 0.61 * LAM / ALPHA					# Airy radius
DZ = 2.5 * DOF
X_HALF = 2.5 * AIRY
Z_LO, Z_HI = Z_PAR - DZ, Z_PAR + DZ

fig = plt.figure(figsize=(13.5, 14))
gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.26)

# ---- A, B: the ray caustic ----------------------------------------------
for k, (m, lbl) in enumerate(((ideal, "A   rays, ideal objective"),
								(aberrated, f"B   rays, C30 = {C30*1e6:g} " + r"$\mu$m"))):
	ax = fig.add_subplot(gs[0, k])
	m.show(kind="ray", plt_ax=ax, regenerate=False, conjugates=False, title=lbl)
	for t in ax.texts:
		t.set_visible(False)
	ax.set_xlim(Z_LO, Z_HI)
	ax.set_ylim(-X_HALF, X_HALF)
	ax.axvline(Z_PAR, color="0.3", lw=1.0, ls=":")
	ax.set_xlabel(f"z  —  {(Z_HI-Z_LO)*1e9:.0f} nm across the panel")
	ax.set_ylabel(f"x  —  {2*X_HALF*1e12:.0f} pm across the panel")

# ---- C, D: the same window, in the wave ---------------------------------
# The shared cross-section renderer already does what these panels need: each
# plane peak-normalized (the focus is orders of magnitude brighter than the
# converging beam), and |psi| rather than |psi|^2 -- which IS the power-1/2
# stretch, so no `norm` is called for. `zlims` windows it: planes outside the
# focal window are dropped before the common transverse grid is built, so the
# panel is sized by the focus and not by the far end of the column.
columns = [focal_window(c30, Z_LO, Z_HI) for c30 in (0.0, C30)]
for k, (m, lbl) in enumerate(zip(columns,
		("C   wave, ideal objective", f"D   wave, C30 = {C30*1e6:g} " + r"$\mu$m"))):
	ax = fig.add_subplot(gs[1, k])
	m.show(kind="wave-hybrid", plt_ax=ax, regenerate=False, conjugates=False,
		   zlims=(Z_LO, Z_HI), ylims=(-X_HALF, X_HALF), title=lbl)
	ax.axvline(Z_PAR, color="w", lw=1.0, ls=":", alpha=0.7)
	ax.set_xlabel(f"z  —  {(Z_HI-Z_LO)*1e9:.0f} nm across the panel")

# ---- E: the focal surface -----------------------------------------------
axE = fig.add_subplot(gs[2, 0])
iso = Microscope(sections=[MicroscopeSection(elements=[
	Source(voltage=200), Lens(strength=np.sqrt(1 / F_OL), aberrations={'C30': C30}),
	Drift(length=0.02)])])
si = iso.focal_surface(family="diff", aperture=ALPHA * F_OL, radii=12, azimuths=4)
ai = si["radius"] / F_OL
oi = np.argsort(ai)
axE.plot(ai[oi] * 1e3, (si["z"][oi] - si["z_paraxial"]) * 1e9, "o", ms=6,
			color="tab:blue", label="isolated thin lens, traced")
aa = np.linspace(0, ALPHA, 100)
axE.plot(aa * 1e3, -C30 * aa ** 2 * 1e9, "-", lw=1.6, color="k",
			label=r"closed form $-C_{30}\alpha^2$")
sc = aberrated.focal_surface(family="diff", aperture=ALPHA * F_OL, radii=12,
								azimuths=4, near=Z_PAR)
r0 = np.zeros((sc["radius"].size, 6)); r0[:, 0] = sc["radius"]
rr = np.asarray(aberrated.propagate_ray(r0.copy()))
# ray heights at OL1's entrance: pick the logged plane at OL1's z, rather than
# hardcoding a plane index that silently rots if the column changes
z_ol = aberrated.get_element_position("OL1")
i_ol = int(np.argmin(np.abs(rr[:, 0, 4] - z_ol)))
h_f = np.abs(rr[i_ol, :, 0]) / F_OL
oc = np.argsort(h_f)
axE.plot(h_f[oc] * 1e3, (sc["z"][oc] - sc["z_paraxial"]) * 1e9, "s", ms=6,
			color="tab:red", label="OL1 (0.08 mm bore), traced")
axE.set_xlabel(r"$h/f$ at OL1 (mrad) — what enters the kick")
axE.set_ylabel(r"$z-z_{\rm paraxial}$ (nm)")
axE.set_title("E   the plane becomes a surface")
# both go in the empty lower-left triangle: the squares run along the top and
# the closed form sweeps the diagonal
axE.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.20),
			framealpha=0.92)
axE.grid(alpha=0.3)
# the DELIVERED fraction: OL1's fitted curvature over the closed form. The
# bore is 0.08 mm (KL = 0.16, nearly thin), so this should be close to 1 --
# measured here rather than assumed, and reused by panel F's annotations.
DELIVERED = float(sc["fit"]["c20"] / (-C30 * ALPHA**2))
axE.text(0.03, 0.04,
	"OL1's bore is 0.08 mm (KL = 0.16, nearly thin), so its\n"
	"distributed aberration lands close to the closed form:\n"
	f"delivered fraction = {DELIVERED:.3f}",
	transform=axE.transAxes, fontsize=7.6, color="tab:red")

# ---- F: the focus, as a Strehl loss -------------------------------------
axF = fig.add_subplot(gs[2, 1])
ref = None
for m, c30, c in zip(columns, (0.0, C30), ("tab:blue", "tab:red")):
	# the focal plane on its NATIVE grid -- at the focus the physical pixel is
	# picometres, and the cross-section's common grid would misreport the peak
	# the Strehl is measured from
	_psi, _dx, _dy, _lam, _zn = read_wavefield(m.wavefield_at(Z_PAR))
	In = np.abs(_psi[_psi.shape[0] // 2, :]) ** 2
	xn = (np.arange(In.size) - In.size // 2) * _dx
	if ref is None:
		ref = In.max()							# normalise BOTH to the ideal peak,
	line = In / ref								# or the Strehl loss is hidden
	axF.plot(xn * 1e12, line, "-", color=c, lw=1.6,
				label=f"C30 = {c30*1e6:g} um   (peak {line.max():.3f})")
	if c30:
		strehl = line.max()
axF.set_xlim(-X_HALF * 1e12, X_HALF * 1e12)
axF.set_xlabel("x at the paraxial focus (pm)")
axF.set_ylabel(r"$|\psi|^2$, both scaled to the IDEAL peak")
axF.set_title(f"F   the focus, {ALPHA*1e3:.0f} mrad")
axF.legend(fontsize=8)
axF.grid(alpha=0.3)
axF.text(0.03, 0.62,
	f"Strehl = {strehl:.3f}\n"
	r"peak quartic phase $kC_{30}\alpha^4/4$ = "
	f"{2*np.pi/LAM*C30*ALPHA**4/4:.2f} rad nominal\n"
	f"{2*np.pi/LAM*DELIVERED*C30*ALPHA**4/4:.2f} rad delivered "
	f"(x{DELIVERED:.3f}, measured in panel E)\n"
	r"Rayleigh $\pi/2$ limit at this $\alpha$: "
	f"{1.5708*4*LAM/(2*np.pi*ALPHA**4)/DELIVERED*1e6:.1f} " + r"$\mu$m",
	transform=axF.transAxes, fontsize=7.6)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
					"OL1_spherical_aberration.png")
fig.suptitle(f"Spherical aberration on objective_section's OL1   "
				f"(C30 = {C30*1e6:g} " + r"$\mu$m" + f", {ALPHA*1e3:.0f} mrad)", fontsize=14, y=0.995)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"paraxial focus z = {Z_PAR*1e3:.6f} mm")
print(f"isolated: c20={si['fit']['c20']*1e9:8.3f} nm  vs closed form "
		f"{-C30*ALPHA**2*1e9:.3f} nm")
print(f"OL1     : c20={sc['fit']['c20']*1e9:8.3f} nm  sag={sc['sag']*1e9:.2f} nm")
print(f"Strehl  : {strehl:.3f}")
