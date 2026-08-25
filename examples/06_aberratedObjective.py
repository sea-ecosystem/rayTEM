"""Put spherical aberration on basic_column's objective (OL1) and look at it.

Five panels:

A  the column, with OL1 and the plane examined marked.
B  the OL diffraction plane with an ideal OL1 -- every ray crosses at one point.
C  the same plane with C30 = 1 mm -- the rays cross over a 40 nm spread: a caustic.
D  the focal surface. The ISOLATED thin lens follows the closed form
   -C30 alpha^2 exactly, which validates the implementation; OL1 inside the
   column does not, and should not: the beam arrives converging, and OL1 is
   10 mm thick so its aberration is distributed along the body.
E  the same aberration in the WAVE path, as a Strehl loss at the focus.

Run: python examples/06_aberratedObjective.py   (writes figures/)
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pySEA.rayTEM
from pySEA.rayTEM import Source, Drift, Lens, MicroscopeSection, Microscope
from pySEA.rayTEM.assemblies import load_microscope
from pySEA.rayTEM.elements import Lens as _L
from pySEA.rayTEM.aberrations import Aberrations

P = os.path.join(os.path.dirname(pySEA.rayTEM.__file__), "microscopes", "basic_column.sea")
CS, H_MAX, F_OL = 1e-3, 1.04e-4, 8e-3

def column(Cs):
    # Aberrations are attached as a set of Krivanek C_{n,m} coefficients, not as
    # a per-aberration attribute: spherical is C30, and the same object would
    # carry astigmatism or coma with no change to anything downstream.
    m = load_microscope(P)
    for z, L, e in m._element_spans():
        if isinstance(e, _L) and e.name == "OL1":
            e.aberrations = Aberrations({'C30': Cs}) if Cs else None
    return m

def fan(m, n=15, h=H_MAX):
    r0 = np.zeros((n, 6)); r0[:, 0] = np.linspace(-h, h, n)
    m.propagate_ray(r0.copy()); return m

fig = plt.figure(figsize=(14, 12.5))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.25, height_ratios=[1.15, 1, 1])
z_par = float([z for z in column(0.0).conjugate_planes(axis="x")["diff"] if z > 0.5][0])

# ---- A: context, full width ----------------------------------------------
axA = fig.add_subplot(gs[0, :])
fan(column(0.0)).show(kind="ray", plt_ax=axA, regenerate=False, conjugates=False,
                      title="A   basic_column, 10 mrad illumination — OL1 is the lens being aberrated")
for t in axA.texts:                                   # the element labels crowd at this scale
    t.set_visible(False)
axA.axvline(0.490, color="crimson", lw=1.6)
axA.axvline(z_par, color="seagreen", lw=1.4, ls="--")
axA.text(0.490, axA.get_ylim()[1]*0.80, "  OL1  (f = 8 mm, 10 mm thick)",
         color="crimson", fontsize=9, fontweight="bold")
axA.text(z_par, axA.get_ylim()[0]*0.80, "  plane examined: 502.4876 mm",
         color="seagreen", fontsize=9, fontweight="bold")
axA.set_xlabel("z (m)"); axA.set_ylabel("x (m)")

# ---- B, C: the focus, ideal vs aberrated, identical axes -----------------
for k, (Cs, lbl) in enumerate(((0.0, "B   OL1 ideal  (C30 = 0):  one crossing"),
                               (CS, f"C   OL1 aberrated  (C30 = {CS*1e3:.0f} mm):  a caustic"))):
    ax = fig.add_subplot(gs[1, k])
    fan(column(Cs)).show(kind="ray", plt_ax=ax, regenerate=False,
                         conjugates=False, title=lbl)
    for t in ax.texts:
        t.set_visible(False)
    ax.set_xlim(z_par - 90e-9, z_par + 30e-9)
    ax.set_ylim(-1.2e-9, 1.2e-9)
    ax.axvline(z_par, color="0.3", lw=1.2, ls=":")
    ax.text(z_par, 1.08e-9, " paraxial plane", color="0.3", fontsize=8, va="top")
    ax.set_xlabel("z  —  120 nm across the panel")
    ax.set_ylabel("x  —  2.4 nm across the panel")

# ---- D: the surface ------------------------------------------------------
axD = fig.add_subplot(gs[2, 0])
iso = Microscope(sections=[MicroscopeSection(elements=[
    Source(voltage=200), Lens(strength=np.sqrt(1 / F_OL), aberrations={'C30': CS}),
    Drift(length=0.02)])])
si = iso.focal_surface(family="diff", aperture=10e-3 * F_OL, radii=12, azimuths=4)
ai = si["radius"] / F_OL ; oi = np.argsort(ai)
axD.plot(ai[oi]*1e3, (si["z"][oi]-si["z_paraxial"])*1e9, "o", ms=6, color="tab:blue",
         label="isolated thin lens, traced")
aa = np.linspace(0, 10e-3, 100)
axD.plot(aa*1e3, -CS*aa**2*1e9, "-", lw=1.6, color="k", label=r"closed form $-C_{30}\alpha^2$")
m = column(CS)
sc = m.focal_surface(family="diff", aperture=H_MAX, radii=12, azimuths=4, near=z_par)
r0 = np.zeros((sc["radius"].size, 6)); r0[:, 0] = sc["radius"]
rr = np.asarray(m.propagate_ray(r0.copy()))
iz = int(np.argmin(np.abs(rr[:, 0, 4] - 0.490)))
h_f = np.abs(rr[iz, :, 0]) / F_OL ; oc = np.argsort(h_f)
axD.plot(h_f[oc]*1e3, (sc["z"][oc]-sc["z_paraxial"])*1e9, "s", ms=6, color="tab:red",
         label="OL1 in basic_column, traced")
axD.set_xlabel(r"$h/f$ at OL1 (mrad) — what enters the kick")
axD.set_ylabel(r"$z-z_{\rm paraxial}$ (nm)")
axD.set_title("D   the plane becomes a surface")
axD.legend(fontsize=8, loc="upper right")
axD.grid(alpha=0.3)
axD.set_ylim(-115, 22)
axD.text(0.03, 0.05,
	"the red curve is NOT expected to follow the closed form:\n"
	"  • the condenser delivers a CONVERGING beam to OL1\n"
	"  • OL1 is 10 mm THICK, so its aberration is distributed\n"
	"    along the body, not applied at one plane",
	transform=axD.transAxes, fontsize=7.6, color="tab:red")

# ---- E: the wave, same lens, same aberration ----------------------------
axE = fig.add_subplot(gs[2, 1])
f_w, a_w, n_w = F_OL, 4e-5, 512
ref = None
for Cs, c in ((0.0, "tab:blue"), (CS, "tab:red")):
    src = Source(voltage=200, wave_shape=(n_w, n_w), wave_extent=3.2e-4,
                 wave_kind="aperture", aperture_radius=a_w)
    mw = Microscope(sections=[MicroscopeSection(elements=[
        src, Lens(strength=np.sqrt(1/f_w), aberrations={'C30': Cs} if Cs else None),
        Drift(length=f_w)])])
    mw.propagate_wave(mode="hybrid", absorb=0.0)
    w = mw.wavefield_at(float(mw.crossovers[0]))
    I = np.abs(np.asarray(w.data))**2
    if ref is None:
        ref = I.max()                       # normalise BOTH to the ideal peak,
    I = I / ref                             # or the Strehl loss is hidden
    dxr = float(w.dimensions[-1].scale)
    r = (np.arange(n_w) - n_w//2) * dxr
    axE.plot(r*1e9, I[n_w//2, :], "-", color=c, lw=1.6,
             label=f"C30 = {Cs*1e3:.0f} mm   (peak {I.max():.4f})")
    if Cs:
        strehl = I.max()
axE.set_xlim(-1.0, 1.0)
axE.set_xlabel("x at the focus (nm)")
axE.set_ylabel(r"$|\psi|^2$, both scaled to the IDEAL peak")
axE.set_title(f"E   the WAVE focus, same lens, {a_w/f_w*1e3:.0f} mrad")
axE.legend(fontsize=8); axE.grid(alpha=0.3)
axE.text(0.03, 0.55,
	f"Strehl = {strehl:.4f}\n"
	r"peak quartic phase $kC_{30}\alpha^4/4$ = "
	f"{2*np.pi/2.5078e-12*CS*(a_w/f_w)**4/4:.2f} rad\n"
	"small because 5 mrad is all this grid can\nsample: the screen goes as "
	r"$r^4$",
	transform=axE.transAxes, fontsize=7.6)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "OL1_spherical_aberration.png")
fig.suptitle("Spherical aberration applied to basic_column's objective OL1   (C30 = 1 mm)",
             fontsize=14, y=0.995)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"isolated: c20={si['fit']['c20']*1e9:8.3f} nm  vs closed form {-CS*(10e-3)**2*1e9:.3f} nm")
print(f"column  : c20={sc['fit']['c20']*1e9:8.3f} nm  sag={sc['sag']*1e9:.2f} nm")
