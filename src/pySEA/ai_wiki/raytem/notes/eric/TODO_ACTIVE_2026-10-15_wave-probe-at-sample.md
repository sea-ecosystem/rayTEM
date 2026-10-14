# TODO — The wave probe at the sample plane (findings from the emulator's B0)

**Branch/worktree:** `column-first-emulator` (rayTEM), cut from `dev` when the work
starts; nothing is changed in rayTEM yet -- this note records what the sea-pearl
emulator's de-risk found, so the engine's owner can decide.
**Plan:** none (the reproduction is below). Context: sea-pearl
`notes/eric/PLAN_2026-10-15_column-first-emulator.md`, stage B0.

**Goal:** `propagate_wave(mode='hybrid')` on a probe-forming column (the emulator's
`emulator.sea`, rayTEM's basic column with a 10 um CA) should deliver a
diffraction-limited probe at the sample: FWHM within 10 % of 0.51 λ/α (43 pm at
200 kV, 29.5 mrad).

**Why:** the emulator's `wave` probe mode is meant to take the probe from the column
(Eric, 2026-10-15); today it cannot, for the three reasons below, so it takes the
column's pupil at the objective's exit instead and forms the probe itself.

## What was found (2026-10-15)

1. **A flat seed never fills the aperture.** `Source.wave(mode='scaled')` seeds
   `s=1, R=inf`. A 2.5 um Gaussian stays 2.5 um wide (divergence λ/πw ~ 0.3 urad),
   passes a 10 um CA untouched and lands on the sample as the demagnified source
   (1.2 nm FWHM) -- a source-size blob, not a probe. A diverging seed
   (`R = size/angle`, the point the ray cone comes from) or `wave_kind='point'` is
   physically right but expands the frame to 300-500 um before the aperture, and the
   objective's screen is then under-sampled on the scaled grid.
2. **Seeded at the aperture, the engine does its job.** A band-limited disk of the
   CA's radius with the ray bundle's curvature there (`R = Σ x·x' / Σ x'²` over the
   on-axis point's rays, -396.7 mm) chained through
   `element.propagate_wave(mode='hybrid')` from the CA to the sample runs in 0.5 s
   (256²) / 1.7 s (512²): the lens powers go into `R` (CL2 -0.0275 m, CL3 +0.0301 m,
   OL1 -0.002995 m), only residual screens are sampled, and the frame crossovers
   coincide with the ray trace's image crossovers to the digit. There is no
   `propagate_wave(start=<element>)` to do this from the API; a caller has to walk
   `_propagation_elements()` itself and set `_wave_scaled_planes`.
3. **At the crossover the probe comes out wrong.** With the column focused on the
   source-image crossover, the hybrid engine logs frames at the sample within
   40 nm (`s` 2.5e-5 → 1e-4, native pixels 5 → 20 pm, `crossover='jump'` or
   `'flat'`), the through-focus peak sits exactly at the sample, but the probe is
   83 pm FWHM against 43 pm, and its angular spectrum is a non-uniform ~12 mrad disk
   instead of the 29.5 mrad pupil: the frame switch at the crossover loses the pupil.
   `wavefield_at('sample')` also returns the coarse post-crossover frame (s 0.01,
   1.9 nm pixels, extent 0.5 um) because `_scaled_plane_at` takes the first plane
   within `tol` of z, which is that one.
4. **An uncorrected objective's screen needs a fine grid.** `OL1.aberrations =
   {C30: 1 mm}` at 29.5 mrad is a 384 rad screen; the per-pixel step is 20 rad at
   256², π needs ~2000² (107 s per run), even when a plate upstream carries -1 mm
   and the sum at the probe is zero. A corrector's cancellation has to be applied
   where the objective's screen is, or the net aberration stated once.
5. **Which crossover is the probe.** `focus_error`/`findPlanes` use the *diffraction*
   family (rays parallel at the source); the probe is the *image* family (rays from
   one source point). On this column they sit 4.75 um apart, and a 5 x 5 / 7 x 7
   source behind a 10 um CA leaves only the zero-angle rays alive -- the diffraction
   family. The emulator now samples one source point with 21 x 21 angles and
   measures the image crossover itself (`sea_pearl.emulator.optics.image_crossover_z`).

## Reproduction (sea-pearl + rayTEM in one environment)

```python
import numpy as np
from pySEA.sea_pearl.emulator.column import default_column_twin
from pySEA.rayTEM.elements import _as_aberrations
from pySEA.rayTEM.seashells import make_scaled_wavefield_signal, read_scaled_wavefield
from pySEA.rayTEM.waveoptics import bandlimited_disk, reconstruct_physical_wave

twin = default_column_twin()                      # focused on the source-image crossover
twin["OL1"].aberrations = _as_aberrations(None)   # finding 4: keep the screen sampled
twin["plate"].aberrations = _as_aberrations(None)
twin.propagate_ray()
rays, I = np.asarray(twin.rays), np.asarray(twin.I)
p_ca = int(np.argmax((I > 0).sum(axis=1) < I.shape[1]))         # the plane after the CA
x, xt = rays[p_ca][:, 0], rays[p_ca][:, 1]; m = np.abs(xt) > 1e-12
R_ca = float(np.sum(x[m] * xt[m]) / np.sum(xt[m] ** 2))         # -0.3967 m
r_ca = float(twin["CA"].radius); n = 256; dx = 5 * r_ca / n
U0 = bandlimited_disk((n, n), dx, dx, r_ca).astype(complex)
wave = make_scaled_wavefield_signal(U0, dx, dx, twin["gun"].wavelength, s=1.0, R=R_ca,
                                    tau=0.0, z=float(twin.get_element_position("CA")))
elements = [e for s in twin.sections for e in s._propagation_elements()]
names = [e.name for e in elements]
fi = [wave]
for ele in elements[names.index("CA") + 1: names.index("sample")]:
    interior = []
    f = ele.propagate_wave(fi[-1], mode="hybrid", log=interior, crossover="jump")
    fi.extend(interior)
    if getattr(ele, "length", 0) or ele.kind == "Aperture": fi.append(f)
    else: fi[-1] = f
z_s = float(twin.get_element_position("sample"))
plane = min(fi, key=lambda p: abs(read_scaled_wavefield(p)[7] - z_s))
U, dxi, deta, wl, s, R, tau, zp = read_scaled_wavefield(plane)
psi, dxr, _ = reconstruct_physical_wave(U, dxi, deta, wl, s, R, target_dx=4e-12, target_shape=(256, 256))
Ii = np.abs(psi) ** 2; Ii /= Ii.sum()
print("FWHM ~", 2 * np.sqrt((Ii >= 0.5 * Ii.max()).sum() / np.pi) * dxr * 1e12, "pm  (Airy 43.3)")
```

## Steps (when this is picked up)

- [ ] decide the probe seam: `propagate_wave(start=<aperture>)` (seed a disk with the
      traced curvature there), or re-frame automatically after an `Aperture`
- [ ] the crossover frame: keep the pupil through `'jump'`/`'flat'`; `_scaled_plane_at`
      prefers the finest frame among coincident planes
- [ ] a corrector plate's coefficients cancel the objective's screen where it is
      applied, or `Microscope` exposes the net aberration once
- [ ] `focus_error(family='image')`, and `findPlanes` tolerant of a point source
- [ ] test: probe FWHM within 10 % of 0.51 λ/α on the emulator column
