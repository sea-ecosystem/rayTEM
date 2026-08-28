"""Spherical aberration on the objective, in rays and in the wave.

Uses ``microscopes/objective_section.sea`` — basic_column's own objective (OL1,
f = 8 mm, 10 mm thick) behind a source — rather than the whole column, because
an aberration is a nanometre-scale effect and over a metre of column it is
~1e-4 of the beam width: invisible in any plot of the whole thing, and
expensive to propagate a wave through.

Six panels, three rows of two. Left column ideal, right column aberrated, so
each row is one comparison:

A, B  the RAY caustic through focus.
C, D  the WAVE, |psi(x, z)| over the same window — the same caustic, built by
      a completely different calculation.
E     the focal surface: the ISOLATED thin lens follows the closed form
      -C30 alpha^2 exactly, which validates the implementation; OL1 does not,
      and should not — it is 10 mm thick, so its aberration is distributed
      along the body rather than applied at one plane.
F     the focus itself, as a Strehl loss.

ALPHA is the convergence semi-angle AT THE SAMPLE, and it is the ray's total
deflection: OL1 is thick, so it rotates the ray by its Larmor angle too, and
only 8.08 of the 30 mrad is in x.

Why C30 = 10 um. Everything scales as C30*alpha^4, so at 30 mrad the choice is
narrow. A corrected instrument (C30 < 300 nm) is 0.15 rad of peak phase --
diffraction-limited, which is the POINT of correction and shows nothing. At the
other end, 100 um is 50.7 rad and simply destroys the focus, which shows
nothing either. 10 um lands at Strehl 0.62: a focus that is visibly degraded
and still recognisable, which is the regime where the number means something.

Note the aberration OL1 DELIVERS is ~0.12x the nominal C30, because it is 10 mm
thick and its aberration is distributed along the body. The Rayleigh quarter-
wave limit at 30 mrad therefore lands at C30 ~ 25 um for this lens, not the
3.1 um a thin one would need.

Run: python examples/06_aberratedObjective.py   (writes figures/)
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pySEA.rayTEM import Source, Drift, Lens, MicroscopeSection, Microscope
from pySEA.rayTEM.elements import Drift as _Drift
from pySEA.rayTEM.aberrations import Aberrations
from pySEA.rayTEM.microscopes.objective_section import build_objective_section
from pySEA.rayTEM.seashells import read_scaled_wavefield
from pySEA.rayTEM import waveoptics as wo

ALPHA, C30, F_OL = 30e-3, 1e-5, 2e-3	# F_OL follows basic_column's OL1
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


def subdivide_focus(m, z_lo, z_hi, n_steps):
	"""Split the sample gap so wave planes are logged densely across the focus.

	The stored column logs a plane per element, which is far too coarse to
	resolve a sub-micron caustic. This replaces the ``sample`` drift with a
	lead-in, ``n_steps`` equal steps across the window, and a tail.

	Parameters
	----------
	m : Microscope
		Scope to modify in place.
	z_lo, z_hi : float
		Window bounds (metres, absolute z).
	n_steps : int
		Number of planes across the window.

	Returns
	-------
	Microscope
		The same object, for chaining.
	"""
	sec = m["O"]
	els = list(sec.elements)
	i = [j for j, e in enumerate(els) if e.name == "sample"][0]
	z0 = sum(e.length for e in els[:i]) + m["O"].position
	total = els[i].length
	lead, span = z_lo - z0, z_hi - z_lo
	els[i:i + 1] = ([_Drift(length=lead)] + [_Drift(length=span / n_steps)] * n_steps
					+ [_Drift(length=total - lead - span)])
	sec.elements = els
	return m


def wave_cross_section(c30, z_lo, z_hi, x_half, z_at, n_x=400):
	"""|psi(x, y=0, z)| across the focal window, on a common x grid.

	Parameters
	----------
	c30 : float
		Spherical aberration coefficient (metres).
	z_lo, z_hi : float
		Window bounds (metres).
	x_half : float
		Half-width of the common transverse grid (metres).
	z_at : float
		The plane whose native-grid line-out is returned alongside (metres).
	n_x : int, optional
		Samples across it, by default 400.

	Returns
	-------
	tuple
		``(z, x, intensity, native)`` with ``intensity`` shaped
		``(len(z), n_x)`` and ``native`` the ``(x, I, z)`` line-out at ``z_at``
		on its own grid. Intensities are in
		absolute units — the caller decides how to normalize, because the two
		panels need different things: a cross-section wants each plane scaled
		to its own peak (or the focus swamps everything), while the focal
		line-out must keep the ideal peak as its reference or the Strehl loss
		is hidden.
	"""
	m = subdivide_focus(scope(c30), z_lo, z_hi, N_PLANES)
	m.propagate_wave(mode="hybrid", absorb=0.0)
	x = np.linspace(-x_half, x_half, n_x)
	zs, rows, native = [], [], None
	for plane in m._wave_scaled_planes:
		U, dxi, deta, lam, s, R, tau, z = read_scaled_wavefield(plane)
		if not (z_lo - 1e-12 <= z <= z_hi + 1e-12):
			continue
		psi, dx, dy = wo.reconstruct_physical_wave(U, dxi, deta, lam, s, R)
		row = np.abs(psi[psi.shape[0] // 2, :])
		xs = (np.arange(row.size) - row.size // 2) * dx
		if native is None or abs(z - z_at) < abs(native[2] - z_at):
			# keep the focal plane on its NATIVE grid: at the focus the physical
			# pixel is picometres, and resampling it onto the cross-section's
			# common grid would misreport the peak the Strehl is measured from
			native = (xs, row ** 2, z)
		zs.append(z)
		rows.append(np.interp(x, xs, row, left=0.0, right=0.0))
	I = np.asarray(rows) ** 2
	return np.asarray(zs), x, I, native


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
# For reference, what OL1 actually delivers is ~0.12x the thin-lens closed form,
# because it is 10 mm thick and its aberration is distributed along the body
# (panel E measures this).
THICK = 0.122
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
panels = []
for c30 in (0.0, C30):
	panels.append(wave_cross_section(c30, Z_LO, Z_HI, X_HALF, Z_PAR))
for k, ((zs, xs, I, _nat), lbl) in enumerate(zip(panels,
		("C   wave, ideal objective", f"D   wave, C30 = {C30*1e6:g} " + r"$\mu$m"))):
	ax = fig.add_subplot(gs[1, k])
	z_edges = np.concatenate([[zs[0]], (zs[:-1] + zs[1:]) / 2, [zs[-1]]])
	x_edges = np.linspace(xs[0], xs[-1], xs.size + 1)
	# each plane to its own peak: the focus is orders of magnitude brighter than
	# the converging beam, so a shared scale would show a dot and nothing else
	norm = I / I.max(axis=1, keepdims=True)
	# a power stretch, not linear: the focus is orders of magnitude brighter
	# than the converging beam, and the structure worth seeing is in the wings
	ax.pcolormesh(z_edges, x_edges * 1e12, norm.T, cmap="magma", shading="flat",
					norm=PowerNorm(0.5, vmin=0, vmax=1))
	ax.axvline(Z_PAR, color="w", lw=1.0, ls=":", alpha=0.7)
	ax.set_xlim(Z_LO, Z_HI)
	ax.set_title(lbl)
	ax.set_xlabel(f"z  —  {(Z_HI-Z_LO)*1e9:.0f} nm across the panel")
	ax.set_ylabel("x (pm)")

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
			color="tab:red", label="OL1 (10 mm thick), traced")
axE.set_xlabel(r"$h/f$ at OL1 (mrad) — what enters the kick")
axE.set_ylabel(r"$z-z_{\rm paraxial}$ (nm)")
axE.set_title("E   the plane becomes a surface")
# both go in the empty lower-left triangle: the squares run along the top and
# the closed form sweeps the diagonal
axE.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.20),
			framealpha=0.92)
axE.grid(alpha=0.3)
axE.text(0.03, 0.04,
	"the red curve is NOT expected to follow the closed form:\n"
	"OL1 is 10 mm THICK, so its aberration is distributed\n"
	"along the body, not applied at one plane",
	transform=axE.transAxes, fontsize=7.6, color="tab:red")

# ---- F: the focus, as a Strehl loss -------------------------------------
axF = fig.add_subplot(gs[2, 1])
ref = None
for (_zs, _xs, _I, (xn, In, zn)), c30, c in zip(panels, (0.0, C30),
												("tab:blue", "tab:red")):
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
	f"{2*np.pi/LAM*THICK*C30*ALPHA**4/4:.2f} rad delivered "
	f"(x{THICK:.3f}, OL1 is thick)\n"
	r"Rayleigh $\pi/2$ limit at this $\alpha$: "
	f"{1.5708*4*LAM/(2*np.pi*ALPHA**4)/THICK*1e6:.0f} " + r"$\mu$m",
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
