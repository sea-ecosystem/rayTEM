# rayTEM — Scaled-frame switching through crossovers (full-column scaled propagation)

## Context

`propagate_wave_scaled` currently stops at the first beam crossover (C1's focus,
z ≈ 174.9 mm in `basic_column`): the scaled factorization
ψ = (1/s)·U(x/s, y/s)·exp[ikr²/2R] follows a geometric reference wavefront, and
when that reference collapses (s → 0) the *coordinate frame* — not the physical
wave — becomes singular. Eric needs the full column, and explicitly the
**focal (back-focal / reciprocal) and image planes** — which sit exactly AT the
crossovers the current implementation refuses to approach.

Eric's handoff (this message) prescribes the fix and the policy:

1. **A general frame-change primitive** (his Eq 5): transform
   (s_o, R_o, U_o) → (s_n, R_n, U_n) directly, keeping ψ identical —
   rescale U, multiply by the reference-phase difference
   exp[ik x²/2·(1/R_o − 1/R_n)]. `factor_wave`/`reconstruct_physical_wave`
   become special cases (physical representation = the frame s=1, R=∞).
2. **A hybrid crossover policy** (first robust implementation): scaled frame
   far from focus → switch to a *flat* frame (constant s, R=∞) while s is
   still comfortably finite → ordinary carrier-free Fresnel through the real
   focus (which has no difficulty there) → re-factor onto a fresh diverging
   frame once safely past. Do **not** ride the converging frame down to
   s_min ~ 1e-3 (that re-enters the tiny-pixel regime the method exists to
   avoid).
3. **Predictive switching, not error-driven**: the singularity is at
   Δz = −R (Eq 7); split segments before it, using a sampling-based
   criterion (Eq 10), not just |s| < s_min.
4. **Terminology**: in code avoid "chart" — use **frame** ("scaled frame").

## The frame-change operation (waveoptics)

New `change_scaled_frame(U, dxi, deta, wavelength, s_old, R_old, R_new,
s_new=None)` implementing Eq 5 with the **physical-grid-continuous
convention**: the new pitch is Δξ_n = Δξ_o·s_o/s_n, so the samples sit at the
same physical points and the operation is **pointwise** (no interpolation):

```
U_n = (s_n/s_o) · U_o · exp[ i k x² / 2 · (1/R_o − 1/R_n) ],   x = s_o·ξ_o
```

- `s_new=None` keeps s (pitch unchanged) — the flatten/re-diverge cases.
- An explicit `s_new ≠ s_old` additionally resamples U via the existing
  `fourier_resample` (band-limited) to a caller-chosen pitch — the general case.
- **Sampling guard**: the added quadratic phase must step < safety·π per pixel
  (same |Δχ| criterion as `elements._check_screen_sampling`); raise an
  actionable error naming the minimum representable |R| otherwise.
- Refactor `factor_wave` and `reconstruct_physical_wave` (its no-target-grid
  path) to delegate to this primitive — they are the (1,∞)→(s,R) and
  (s,R)→(1,∞) special cases. Existing tests must stay green bit-for-bit
  (identity round-trip is already tested at 1e-13).
- Helper `min_representable_curvature(n, dxi, wavelength, s, safety)` =
  k·(s²·(n/2)·Δξ²·... i.e. |R|_min = k·x_max·Δx/(safety·π) with
  x_max = s·(n/2)·Δξ, Δx = s·Δξ — used by both the guard and the policy.

## The crossover policy (free-segment engine)

New pure-math engine `propagate_free_scaled_hybrid(U, dxi, deta, wavelength,
dz, s, R, z, z_cross, safety=0.5, s_min=1e-3)` in `waveoptics.py`, returning
the final state plus an ordered list of interior logged states, each tagged
(`"flatten"`, `"crossover"`, `"rediverge"`). All split points have closed
forms because s and R are linear in z within a frame:

- **Converging frame (R < 0)**: the flatten plane is where the residual
  curvature becomes representable on the (shrinking) grid:
  |R_flat| = R²/(A·s²) with A = k·X_ξ·Δξ/(safety·π), X_ξ = (n/2)Δξ —
  invariant along the frame, so `dz_to_flatten = |R| − |R_flat|`.
  If the segment reaches it: propagate scaled to that plane, `change_scaled_frame`
  → (s, ∞) (log `"flatten"`), record `z_cross = z + |R|`, continue flat.
  (For the C1 case this flattens at |R| ≈ 3.2 mm, s ≈ 0.07 — pixel 5.6 nm,
  FOV 1.4 µm; the Airy first zero there is 13.7 nm and the beam's spatial
  bandwidth a/(fλ) is under half Nyquist, so the flat window is well sampled.)
- **Flat frame (R = ∞) with a recorded z_cross ahead**: propagate ordinary
  (carrier-free, Δτ = Δz/s²) through the focus; when passing z_cross, split
  there and **log the plane** (tag `"crossover"` — this is the back-focal /
  image plane Eric wants). Past it, re-diverge when d = z − z_cross ≥ A·s²:
  `change_scaled_frame` → (s, R = +d) (log `"rediverge"`), continue scaled.
- **Flat frame with no z_cross** (e.g. the collimated launch): plain flat
  propagation, unchanged behavior.
- `s_min` stays as a pure backstop (should never trigger under the policy).

State addition: **z_cross rides the scaled Signal metadata** (optional
`z_cross_m` key; `make_scaled_wavefield_signal`/`read_scaled_wavefield` gain
the optional field, absent → None — backward compatible with existing files).

## API consolidation (per Eric): one wave method with a `mode` sub-kind

The three wave paths become **one method** at every level —
`propagate_wave(..., mode: Literal['fixed','scaled','hybrid'] = 'fixed')`
(`mode`, not `kind`, so it never collides with `propagate(kind=...)` when
arguments are forwarded). `propagate_wave_scaled` is **removed** as a separate
public method (recent API, nothing external depends on it):

- `Element.propagate_wave(signal, mode='fixed', s_min=1e-3, log=None)` —
  dispatches internally to the fixed-grid or scaled implementation (private
  `_propagate_wave_fixed` / `_propagate_wave_scaled`); `mode='hybrid'` is the
  scaled path with the frame-switching engine on its free segments, interior
  logged states appended to the optional `log` list. Source (passthrough),
  Aperture (mask — physical radius on 'fixed', radius/|s| otherwise), and
  Prism (raise) each consolidate to a single mode-aware override.
- `Source.wave(mode='fixed'|'scaled')` — one seeding generator:
  'fixed' returns the wavefield Signal (today's `wave()`), 'scaled' the
  scaled state (today's `wave_scaled()`, which is removed); 'hybrid' seeds
  identically to 'scaled'.
- `MicroscopeSection.propagate_wave(wave0=None, mode='fixed', ...)` /
  `Microscope.propagate_wave(...)` — same consolidation; results stored by
  representation: `mode='fixed'` → `.wave`, `'scaled'`/`'hybrid'` →
  `.wave_scaled` (identical structure; hybrid is the crossover policy, not a
  different representation). Flatten/crossover/rediverge planes land in
  `.wave_scaled` in z order alongside element-exit planes.
- Top-level dispatcher: `_PROPAGATE_KINDS` values become
  (method name, forced kwargs) so `propagate(kind='wave')` →
  `propagate_wave(mode='fixed')`, `kind='wave-scaled'` (alias `wave_scaled`)
  → `mode='scaled'`, `kind='wave-hybrid'` (alias `wave_hybrid`) →
  `mode='hybrid'`; the `kind` Literal on Element/Section/Microscope is one
  flat list.
- `Microscope.propagate_wave(mode='hybrid')` additionally stores
  `self.crossovers` — the crossover z positions from the run — so the focal
  planes are directly discoverable; `wavefield_at(z_cross)` hands back the
  focal-plane (diffraction) wavefield.
- Terminology sweep: "chart" → "frame" in the scaled-path error messages
  (`scaled_delta_tau`, `propagate_free_scaled`), docstrings, and docs.

## Dimensions of `.wave_scaled` (Eric's question 2)

Unchanged. Every policy switch uses the physical-grid-continuous convention
with s kept at the switch plane (flatten and re-diverge are pointwise), so
**Δξ/Δη remain one shared calibration for the entire run** — the single-stack
SignalSet contract holds exactly as today. `s(z)` stays a continuous
piecewise-linear companion (dipping toward each focus and recovering);
`R(z)` jumps at lenses and switches as it already does. Additions only:
a per-plane integer `frame` companion Signal (increments at each switch) plus
plane tags (`flatten`/`crossover`/`rediverge`) in metadata, and the in-flight
`z_cross_m` metadata key.

## Files to modify

- `src/pySEA/rayTEM/waveoptics.py` — `change_scaled_frame`,
  `min_representable_curvature`, `propagate_free_scaled_hybrid`;
  `factor_wave`/`reconstruct_physical_wave` delegate; frame terminology.
- `src/pySEA/rayTEM/seashells.py` — optional `z_cross` on
  `make_scaled_wavefield_signal`/`read_scaled_wavefield` (and the
  `_ScaledWavefield` fallback).
- `src/pySEA/rayTEM/elements.py` — consolidate to
  `propagate_wave(signal, mode=..., s_min=..., log=None)` (removing
  `propagate_wave_scaled`); hybrid engine on the free segments;
  `Source.wave(mode=...)`; dispatcher mapping with forced kwargs.
- `src/pySEA/rayTEM/assemblies.py` — consolidated drivers
  (`propagate_wave(wave0=None, mode=...)`), `log` threading,
  `Microscope.crossovers`; docstrings.
- `src/pySEA/rayTEM/tests/test_scaled_fresnel.py` — new tests (below).
- `examples/04_scaledWave_basic_column.py` — full-column demo: cross-section
  source→detector; x–y slices at the C1 back-focal (crossover) plane, the
  `sample` plane, and the `detector`; the manual stop-before-crossover logic
  and the guard demonstration are removed (the engine handles it).
- `docs/wave-optics-sampling.md`, wiki (`waveoptics.md`, `elements.md`,
  `assemblies.md`) — crossover limit replaced by the frame-switching section.
- GitHub issue #2 — close with a comment describing the implementation
  (attribution footer included).
- Collaboration protocol: TODO_ACTIVE + LOG [Under Construction] in
  `notes/eric/` first; small commits per step; ACTIVE→DONE + [Done]+Outcome
  at the end.

## Validation tests

1. **Frame-change identity** — physical wave invariant under
   (s_o,R_o)→(s_n,R_n) for random frames: pointwise path at machine
   precision; resampled path (explicit s_new) < 1e-3; factor/reconstruct
   regression already covers the (1,∞) special cases.
2. **Guard** — flattening a frame whose |R| is below the representable
   minimum raises, naming the threshold.
3. **Through-focus equivalence (optical regime)** — aperture→lens system
   propagated through the focus by the hybrid engine vs the ordinary
   fixed-grid reference past the focus (tapered aperture < 5e-3 pointwise;
   hard aperture via total-intensity + cumulative radial-energy metrics, as
   in the existing system tests).
4. **Electron-scale focal plane** — 200 kV, a = 5 µm, f = 45 mm: the logged
   crossover plane's intensity is the Airy pattern; first-zero radius
   1.22·λf/(2a) within a few percent; energy conserved across every switch;
   physical pixel |s|·Δξ continuous at each switch plane.
5. **Full column** — `basic_column` source→detector: run completes, s stays
   finite, energy conserved at all planes, `Microscope.crossovers` non-empty
   (C1/C2/C3/objective/projector chain), `wavefield_at('detector')` and
   `wavefield_at(<C1 crossover z>)` return finite calibrated Signals; all
   existing tests green.

## Development sequence

1. `change_scaled_frame` + `min_representable_curvature` + delegation +
   tests 1–2.
2. Hybrid engine + z_cross metadata + tests 3–4.
3. API consolidation (`propagate_wave(mode=...)` everywhere, remove
   `propagate_wave_scaled`, `Source.wave(mode=...)`, dispatcher with forced
   kwargs) — update all existing tests/examples to the new spelling.
4. Driver wiring (`log` threading, frame companion, `Microscope.crossovers`)
   + test 5; full suite.
5. Demo rerun (full column, focal-plane slices), docs + wiki + terminology
   sweep, close issue #2, protocol finish.

## Out of scope

Optimizing away the flat windows (direct converging→diverging frame jumps);
anisotropic frames (s_x ≠ s_y) for strong quadrupoles; post-specimen adapters.
Both remain follow-ups per the handoff ("you can later optimize away the
short ordinary-Fresnel windows if there is a performance reason").
