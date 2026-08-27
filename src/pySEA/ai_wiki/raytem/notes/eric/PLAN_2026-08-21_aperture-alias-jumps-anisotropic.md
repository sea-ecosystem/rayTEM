# rayTEM — aperture edge taper, beam-support frame policy, direct frame jumps, anisotropic frames

## Context

Three work items following the full-column hybrid landing:

1. **The "grid" texture** Eric spotted in the sample/detector images. Diagnosed
   (controls run read-only during planning): it is in the data — a mix of real
   Fresnel edge diffraction (present at ~1% rms by z = 120 mm, before any lens)
   and **aliasing of the binary aperture edge**: Θ(a−r) needs frequencies above
   the grid Nyquist, which fold back at the initial sampling and propagate
   coherently, interfering as an axis-aligned plaid (interior min/max ≈ 0.5 by
   the detector). Grid padding does NOT help (not FFT wraparound). **Eric's
   direction: the Fresnel diffraction is physical and often observed — keep
   it; remove only the gridding; stay physically rooted.** So the fix is NOT
   edge smoothing: it is *alias-free sampling of the exact hard aperture* —
   the band-limited projection of the same Θ(a−r), which keeps every
   representable Fresnel fringe and removes only the folded above-Nyquist
   content.
2. **Engine weaknesses exposed by the padded-grid control**: on a 512²/40 µm
   grid the hybrid run crashed — the flatten point fell below `s_min` —
   because `min_representable_curvature` measures the reference phase at the
   *grid* edge, where the field is empty. A **beam-support-based** criterion
   (Eric's original "based on the actual spectral support of U") fixes the
   crash and flattens much earlier (larger s → larger pixels — his stated
   preference).
3. **The two queued follow-ups from the frame-switching handoff**: direct
   converging→diverging frame jumps (optimize away the flat windows), and
   anisotropic frames (s_x ≠ s_y) so strong quadrupoles absorb into R like
   round lenses do.

## Item 1 — alias-free sampling of the exact hard aperture

The model stays Θ(a−r) (handoff Eq 9, hard edge, real Fresnel fringes kept);
only the *sampling* changes: the grid holds the band-limited projection of
the sharp disk instead of a point-sampled binary mask.

- `waveoptics.bandlimited_disk(shape, dx, dy, radius)`: synthesize the disk
  from its analytic spectrum — FT of the sharp disk is
  `2π a²·J1(a·k_r)/(a·k_r)` — sampled on the discrete `fftfreq` grid up to
  Nyquist and inverse-FFT'd (real part). Exactly alias-free: every
  representable frequency exact, nothing folded. Use `scipy.special.j1` if
  importable; otherwise fall back to 8× supersampled area-coverage of the
  binary mask block-averaged down (sinc²-attenuated foldover) — decide by
  probing the venv at implementation time.
- `Source._aperture_wave` uses it (`wave_kind='aperture'` becomes alias-free
  by default). Optional `antialias=False` escape hatch restores the
  point-sampled binary mask for regression/comparison.
- `Aperture` element mid-column: masking an existing field needs a real-space
  mask — use the anti-aliased edge-coverage mask (smooth over ~1 px, the
  projection of the sharp mask), same `antialias` escape hatch; applied in
  both the fixed mask and the scaled `radius/|s|` mask
  (`aperture_mask(field, dx, dy, radius, antialias=True)`).
- Update the one test asserting the aperture wave is exactly binary
  (`test_wave_kind_aperture_...`: unique values [0,1] → checked with
  `antialias=False`; add an alias-suppression assertion for the default).
- Measure and report: interior modulation at the sample/detector with the
  alias-free disk — the remaining ripple is genuine (band-limited Fresnel +
  Gibbs) and is the honest physical answer; verify the plaid (axis-aligned
  spectral excess) is gone while radial Fresnel ring contrast is preserved.
- Airy focal-plane test must still pass unchanged (first zero set by the
  radius, not the sampling).

## Item 2 — beam-support frame policy (fixes the s_min conflict)

- `waveoptics.change_scaled_frame`: the sampling guard measures the per-pixel
  phase step **only over the beam support** — the bounding radius where
  `|U| > 1e-6·|U|.max()` — since phase applied where the field is ~0 is
  harmless. Keep the current grid-edge behavior when the support fills the
  grid.
- `min_representable_curvature(..., x_max=None)`: optional support-radius
  override (default keeps the grid half-width).
- `propagate_free_scaled_hybrid`: compute the beam-support radius (in ξ) from
  U at each converging-segment entry (amplitude-threshold bounding radius,
  one cheap reduction) and use it in `A`, so flatten/re-diverge happen at the
  earliest representable plane for the *actual beam*. Internal
  `propagate_free_scaled` calls use an engine-owned backstop
  (`min(s_min, s_flat/2)`) so a legitimate flatten below the user's `s_min`
  no longer crashes; the user-facing `s_min` keeps its meaning for
  `mode='scaled'`.
- Tests: the 512²/40 µm padded basic_column run completes with defaults;
  flatten s is larger (earlier) with a beam smaller than the grid; the
  existing through-focus/Airy/full-column tests stay green.

## Item 3 — direct converging→diverging frame jumps (drop flat windows)

- New branch in `propagate_free_scaled_hybrid` (kwarg
  `crossover: Literal['flat','jump'] = ...`): at the switch plane, jump
  directly from R_o = −d to R_n = +d (mirror image around the focus) via
  `change_scaled_frame` — the moved phase is twice the flatten phase, so the
  jump plane is at half the flatten threshold (still closed-form; the frame
  then expands, s never crosses 0, and U diffracts through its own focus in
  τ). Split at z_cross = z + d to log the crossover plane exactly as today
  (tag `"crossover"`; new tag `"jump"` replaces the flatten/rediverge pair).
  No z_cross marker needs to survive between elements on this path (the jump
  is instantaneous), but keep the flat path fully intact.
- Exposure: `propagate_wave(..., crossover='jump'|'flat')` threaded
  Element → Section → Microscope; default decided by measurement — run the
  through-focus equivalence and full-column tests under both policies and
  default to `'jump'` if its error is comparable or better (expected: fewer
  switches, no flat-window halo accumulation), else keep `'flat'`.
- Tests: parametrize the through-focus equivalence, electron Airy, and
  full-column tests over both policies; assert both traverse all crossovers
  with energy conserved; record the measured errors in the test comments.

## Item 4 — anisotropic frames (s_x ≠ s_y): strong quadrupoles absorb into R

Generalize the frame to per-axis state — factorization
ψ = (s_x s_y)^{-1/2} · U(x/s_x, y/s_y) · exp[ik(x²/2R_x + y²/2R_y)] — so a
quadrupole absorbs (P, −P) into (R_x, R_y) exactly like a round lens, instead
of writing its saddle phase onto U under a sampling guard.

- `waveoptics`: per-axis generalizations — `kernel_phase` over (Δτ_x, Δτ_y)
  (already separable), `scaled_delta_tau` per axis, `apply_thin_lens_scaled`
  → per-axis powers, `change_scaled_frame` with per-axis (s, R),
  `propagate_free_scaled(_hybrid)` carrying 2-vectors; crossovers become
  per-axis **line foci** (tags `"crossover-x"`/`"crossover-y"`; a stigmatic
  focus logs both at one z). Internally represent state as pairs; scalars
  accepted and returned where the axes agree (the round-lens column keeps its
  current behavior and outputs bit-for-bit).
- `seashells`: metadata gains `s_x/s_y`, `R_x/R_y`, `tau_x/tau_y` (falling
  back to the scalar keys when the axes agree — old files load); SignalSet
  companions likewise (scalar companions when isotropic, per-axis when not).
- `elements`: `Quadrapole.phase_shift(scaled=True)` returns per-axis powers
  (absorbed) instead of the U screen; `Lens` returns equal pairs; `Dipole`
  keeps its U screen (linear phase has no quadratic part).
  `reconstruct_physical_wave`/`wavefield_at` handle dx = |s_x|Δξ ≠
  dy = |s_y|Δη.
- Tests: weak-quad equivalence — anisotropic absorption vs the current
  U-screen path vs the fixed-grid reference; strong stigmator (previously
  guard-blocked) now runs; astigmatic line foci logged at f_x ≠ f_y with the
  correct positions; isotropic column regression bit-for-bit; energy
  conserved.

## Files

`src/pySEA/rayTEM/waveoptics.py`, `elements.py`, `assemblies.py`,
`seashells.py`, `tests/test_scaled_fresnel.py`,
`examples/04_scaledWave_basic_column.py` (tapered default → clean interiors;
rerun figures), `microscopes/basic_column.py` + regenerated `.sea`,
`docs/wave-optics-sampling.md`, wiki (`waveoptics.md`, `elements.md`,
`assemblies.md`). Collaboration protocol: TODO_ACTIVE + LOG entry first,
small commits per item, ACTIVE→DONE + [Done]+Outcome at the end.

## Development sequence

1. Item 1 (alias-free aperture sampling) + tests + demo rerun → commit.
2. Item 2 (beam-support policy) + padded-grid regression test → commit.
3. Item 3 (direct jumps) + parametrized policy tests + measured default →
   commit.
4. Item 4 (anisotropic frames) — waveoptics core first with unit tests, then
   seam, then elements/drivers, then the astigmatic column tests → commits
   per layer.
5. Docs + wiki sync + notes finish.

## Verification

- Full suite green after each item (currently 54 tests).
- Demo: sample/detector gridding gone (axis-aligned spectral excess ≈ 1) with
  Fresnel ring contrast preserved; figures regenerated and compared by eye.
- Padded 512²/40 µm hybrid run completes with default s_min.
- Jump-vs-flat error comparison printed by the tests; default set from data.
- Astigmatic test: quad pair with strong K produces two line foci at the
  predicted z, logged as crossover-x / crossover-y.
