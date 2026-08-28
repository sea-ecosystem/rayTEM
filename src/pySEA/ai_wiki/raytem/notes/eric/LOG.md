# Eric's Work Log

Newest entries at top.

## 2026-08-28 — [Done] Merge main into Signal_and_propagation_additions_new (PR #9)
**Goal:** Land Thomas's post-#7 fixes from main here, resolve the conflicts,
keep our new functionality and docs, and merge PR #9.
**Why:** Thomas fixed his #7 merge directly on main (Rays class, measured
focal_length, calibrated_strength rename, JSON compatibility, MACSTEM
history), which made PR #9 un-mergeable.
**Outcome:** 6 conflict hunks resolved (mostly docstring overlaps; kept
Thomas's do-not-remove quad sanity comment AND our skew block; kept our
None-safe first-element repair fix, same semantics as his). Kept our full
`test_eight_configurations.py` over main's empty `dont_test_` rename — it
passes here. Post-merge fixes: made Thomas's `test_old_json_compatibility`
cwd-independent; re-scaled the thick-body aberration test's C30 (1e-3 →
5e-6) because the measured `focal_length` changes a thick lens's
`focal_power` from K·sin(KL) to K·tan(KL) (125 → 464/m on that fixture) and
the perturbation scales as P⁴ — this is the "actual failure" Thomas noted;
repaired RST that the docstring de-backticking broke (Lens.phase_shift
indentation, four bare `|K|`-style tokens). Suite 154/154; docs rebuild
clean (same 11 known duplicates); example 06 runs (Strehl now 0.094 — the
focal-power semantics shift moved delivered aberration; header narrative
still stale, flagged earlier). NOTE: the K·tan(KL) thick-lens power now
feeds every aberration pupil scale — physics choice worth a look.

## 2026-08-28 — [Done] Terminology page + audit items 1–4
**Goal:** Answer issue #8 with a docs terminology page, and clear the four
queued items: section-level Aberrations, skew quadrupole, example 06
cleanups, `Microscope.index` raising instead of print-and-None.
**Why:** Thomas is confused by the lens naming (issue #8), and the four
items have been flagged repeatedly across sessions without being started.
- [x] terminology page + docs rebuild
- [x] section-level `Aberrations`
- [x] skew quadrupole
- [x] example 06 cleanups
- [x] `Microscope.index` raises
**Outcome:** `docs/terminology.md` answers issue #8 (strength vs focal length
vs focal power, aberration powers, transfer blocks, moments, wave_kind,
body/screen, with Brown/Krivanek references). New `AberrationScreen` element
(zero-thickness plate with explicit pupil_power) backs the new
`MicroscopeSection.aberrations` — applied as a transient screen at the
section exit with the section's composite `focal_power` as pupil scale;
suspension and .sea round trips covered. `Quadrapole(skew=...)` rolls the
principal frame into the lab 4×4 (45° stigmator works; per-axis views raise).
`Microscope.index` raises KeyError with the available names instead of
print-and-None. Example 06: tab-only continuations, OL1 plane derived from z
instead of magic index 2 (output verified identical). Suite 152/152; docs
site rebuilt (11 known duplicate warnings, was 9 — the two new are
AberrationScreen re-exports). NOTE for Eric: example 06's header still
claims Strehl 0.62 / delivered 0.122x, but it has printed Strehl 0.103
since the column rebuild — the narrative needs re-deriving against the new
OL1; left untouched deliberately.

## 2026-08-27 — [Done] Sphinx docs paralleling sea-eco
**Goal:** Eric: the docs structure paralleling sea-eco was missing — build it,
keeping the existing `wave-optics-sampling.md` untouched (he may simplify it
on a private branch).
- [x] conf.py / index.rst / requirements mirroring sea-eco's five sections
- [x] Guides: getting_started, propagation_modes, operating_the_column
- [x] Example Scripts page; AI Tools page (wiki slice + recorded omissions)
- [x] "Into the SEA-weeds" landing page (mental model, invariants, seams,
  Schema omission on record, Provenance and verification, Building the docs)
- [x] `wave-optics-sampling.md` slotted in AS-IS
- [x] build verified; 4 real docstring defects fixed at the source

**Outcome:** `sphinx-build docs docs/_build/html` exits clean with nine known
duplicate-object warnings (cross-module re-exports) and nothing else. The
big win was one conf line — `napoleon_custom_sections = [("Related", "see
also")]` — our house "Related" section was tripping ~650 field-list warnings.
Real defects found by the build and fixed in code: two literal tables
narrower than their widest cells (`focus_error`, `segment_block`), a
`:math:\theta` in a non-raw docstring (rendered a TAB), bare `|psi|` pipes
parsing as RST substitution references, and an unreferenced footnote.

Note for whoever simplifies `wave-optics-sampling.md`: only `index.rst`
references it (one toctree line), so renaming/splitting it touches nothing
else.

## 2026-08-27 — [Done] OL1 f = 3 mm; suite fully green; Gun; two old bugs down
**Goal:** Eric's five calls: OL1 f matches the mid-gap sample (3 mm); keep
pushing to the branch (no PR); combine_drifts default False; diagnose the
insertion failure; add the small stuff.
- [x] OL1 f = 3 mm -> 30.000 mrad at 43 nm (high) / 2 nm (low), condensers only
- [x] `combine_drifts` defaults to **False** (three confirmed casualties of the
  merging default; tidying is opt-in with True)
- [x] **the long-standing `test_section_insertion_microscope` failure is FIXED**
- [x] `Gun(Source)` class (kind='Gun', reloadable, round-trip test)
- [x] `convergence_angle` = max total angle among LIVE rays
- [x] predict_probe reads the aperture in the LAB frame
- [x] wiki refreshed (MACSTEM stays, per Eric: "leave it")

**Outcome:** **148/148 — the suite is fully green for the first time.**

**The insertion bug**, as Eric suspected, was reconstruction robustness:
`repair()` skips i == 0, so a FIRST element placed past the section start
never got a leading drift — the offset was silently dropped and everything
downstream compressed by exactly the missing distance. The old gap handling
that covered this was commented out of `__init__` when `repair()` was
introduced; the test's committed reference rays predate that and were right
all along. repair() now inserts the leading drift.

**A second subtle one:** predicted and traced probe angles disagreed ~9% in
the low state because the aperture reads per-axis maxima in the LAB frame
while the prediction took them in the rotating frame — the square ray fan
arrives rotated by C1's Larmor angle (~5°), and a rotated square's per-axis
max grows by cos+sin. Prediction now rotates first; predicted == measured ==
30.000 mrad in both states and the predicted transmitted fraction matches
the traced current.

Still open: MACSTEM stays by Eric's decision (wiki now indexes it);
aperture rescale-vs-mask model deferred ("not now").

## 2026-08-27 — [Done] 6 mm objective gap, sample at its middle
**Goal:** Eric: OL1-OL2 gap 6 mm, sample halfway (3 mm past OL1's exit),
replacing the back-focal-plane placement.
- [x] builder + .sea + objective_section regenerated (total 776.7 mm)
- [x] all 8 states re-solved; settings + figures regenerated
- [x] suite 147 (146 pass + pre-existing insertion failure)

**Consequences (reported, not hidden):** at v ≈ 3 mm from the f = 2 mm
objective the high-angle branches magnify — the 30.000 mrad high-current
probe is ~31 µm across (surveyed: nothing under ~24 µm exists above
20 mrad), and the low state caps at 13.4 mrad (with a 6 nm probe). If a
small AND 30 mrad probe is wanted at mid-gap, the knobs are OL1's f (down
toward ~1.4 mm puts the BFP near mid-gap) or the gap. One wave-test bound
moved 1.5% → 2% ring texture at the new sample plane.

## 2026-08-27 — [Done] Compact thin-lens column, and a real hybrid-engine bug
**Goal:** rebuild `basic_column` to Eric's spec — every lens a 0.08 mm bore;
gun–50–C1–50–CA–10–50–C2–50–C3–250–OL1–20 (sample at BFP inside)–OL2–50–PL1–
50–PL2–50–PL3–50–PL4–100–detector; dipoles/stigmators kept.
- [x] new layout (790.7 mm total; focal lengths kept — 0.08 mm is the only
  bore all of them can legally share, f >= 2L/pi)
- [x] solver reworked: physical focal-range scans, overfocused-C1 low state,
  verified bisection, four-lens projector relay
- [x] hybrid-engine bug found and fixed (below)
- [x] all examples green; settings + figures regenerated

**Outcome:** 147 tests (146 pass + pre-existing insertion failure). All 8
states: 30.000 mrad probes in both current states (1.000 / 0.079 nA), image
4.5x or 25 mm-camera diffraction at the detector, objective untouched.

**The bug (waveoptics):** when a lens body ends between a wave crossover and
its rediverge, the pending marker was handed to the free-space engine —
whose converging-frame flatten knew nothing about pending markers and
OVERWROTE it with the frame's own crossing (z+|R|), a different ray family.
Every crossover logged downstream was then mislabeled (55 mm off in the
fixture; the wave itself stays exact — the labels lied). Fixed twice over:
the free engine now guards its flatten and rediverges onto the original ray,
and the body walker hands off the marker projected to the exit ray's
straight-line zero (z_end − B/D, slope-independent), which closes the last
23 µm to exact. Found only because the mid-element-crossover test had to be
rebuilt around its own thick fixture once the column went thin — the old
column had been hiding the handoff case.

**Solver changes worth knowing:** with 50 mm drifts a weak C1 no longer
overfills CA, so the low state OVERFOCUSES C1 (crossover 20% of the way to
CA — 7.9% transmitted); and PL1/PL2 alone cannot land a conjugate on the
detector past frozen PL3/PL4, so the projector is now a four-lens relay —
all four are projection lenses and Eric's rule assigns projection to them
collectively. OL1/OL2 remain the only frozen lenses (invariant tested).

## 2026-08-27 — [Done] 30 mrad for real: OL1 becomes a probe-forming objective
**Goal:** Eric: "Shorten OL1's focal length so we can actually reach 30 mrad"
— with the objective still frozen and the condensers doing the focusing.
- [x] OL1: f = 2 mm, 2 mm bore, **sample at its back focal plane**
- [x] geometry compensated (sample z = 0.5, detector 1.264 m unchanged)
- [x] solver: coarse-curve bracketing + small-probe branch preference
- [x] objective_section / example 06 / four tests follow the new lens
- [x] all six examples + ex07 full run green

**Outcome:** 147 tests (146 pass + pre-existing insertion failure). On the
standard column, objective untouched: **30.000 mrad at a 21 nm probe** (high
current) and a 5 nm probe (low, 0.2 nA); reach is smooth up to ~56 mrad.

**Shortening f alone was NOT enough** — with the sample ~2 mm from OL1's
center, any reachable f left it inside the focal length, and the condensers
capped at ~10 mrad however strong. The fix is the placement a real STEM
uses: the sample sits at OL1's back focal plane, so the condensers deliver a
wide nearly-parallel beam and OL1 alone converts radius into angle
(alpha = r/f) while demagnifying hard. The working distance is computed in
the builder from the lens's own thick block (wd = -A/C of the body), so it
tracks f/bore changes.

**Trap worth remembering:** `repair()`'s drift merging absorbed the unnamed
4 mm gap into the zero-length "sample" marker on build, silently moving the
measured sample plane 2 mm off the focal plane (predictions and traces then
disagreed hard). The gap is now the NAMED drift `sample_gap`, which the
merge leaves alone.

**Solver traps fixed:** B(source→sample)=0 is also satisfied by MAGNIFYING
branches (a 40–120 µm "probe" converging steeply — not a probe); branch
selection now prefers small size. And the 30 mrad crossing can sit far down
the slope from the alpha peak, so the target search brackets on the coarse
curve, not beside the maximum.

Fallout tracked: objective_section.py (f_ol1) and example 06 (F_OL)
followed; four tests updated (BFP-driven s contraction, Larmor factor now
computed from the lens, a plane that used to sit inside the old fat bore now
in free space, hardcoded sample z).

## 2026-08-27 — [Done] The objective is never retuned
**Goal:** enforce Eric's division of labor: probe focusing is entirely the
condensers'; projection entirely the projectors'; objective currents fixed.
- [x] OL1 removed from the solve (OL2/PL3/PL4 were already untouched)
- [x] condenser-only convergent solve (direct `B(source->sample)=0` via C3)
- [x] invariant test: no solved state carries an OL1/OL2/PL3/PL4 strength

**Outcome:** 147 tests (146 pass + the pre-existing insertion failure).

The convergent solve had quietly been retuning OL1 because the stored
objective strength has no *real* object plane imaging onto the sample. The
fix also simplified it: no intermediate-crossover chain, just C3 solved so
the total source->sample block has `B = 0` through the frozen objective
(virtual objects allowed), C2 swept toward the target.

**Consequence to decide on:** with the stored OL1 frozen, the condensers cap
at **~10.2 mrad** at the sample (C2 pinned at its first-branch strength
limit) — the 30 mrad target is out of reach on this column and the script
says so. Ways to actually reach 30 mrad, all template decisions for
Eric/Ondrej: shorten the stored OL1 focal length (or move the sample plane
relative to it), enlarge CA, or accept a lower target for the demo.

## 2026-08-27 — [Done] Eight states on the standard column
**Goal:** point example 07 at `basic_column.sea` per Eric: add CA, 1 nA gun,
solve C+OL1 -> sample and sample -> detector, save states, fix the ray/wave
overlay.
- [x] basic_column: CA (10 µm) after C1, `beam_current=1e-9`, 1 cm detector tail
- [x] solver rebuilt on the standard column (OL1 joins the probe chain)
- [x] states saved via Thomas's `save_as_setting` -> `settings/basic_column - <state>.json`
- [x] wave-matched ray overlay
- [x] tests updated (146 total; 145 pass + the pre-existing insertion failure)

**Outcome:** all 8 states solve on the stock column: 30.000 mrad probe
(25 nm) in the high state, image (6.5x, B=0) or diffraction (13.8 mm camera,
A=0) at the detector, 1 nA -> 1 nA high / 0.2 nA low.

**The ray/wave mismatch Eric saw, root-caused twice over.** (1) The overlay
had drawn the source's full incoherent fan, which is real but far wider than
the single coherent mode the wave carries. It now draws the rays the wave
actually follows — the flat-phase family (zero-angle rays at fractions of the
wave envelope; the scaled frame IS that family's reference ray) plus the pair
grazing the CA edge. Rays now ride the |psi| envelope through every lens and
cross exactly at its crossovers. (2) My `predict_probe` had modeled CA as a
per-ray mask — that is the WAVE path's model. The ray path's `Aperture` is a
beam **rescale**: `propagate_ray` shrinks every ray by `radius/xmax` and
`apply_intensity` attenuates uniformly by the area ratio. Prediction now
models exactly that and matches the trace; a consequence worth knowing is
that the low-current state still reaches 30 mrad, because the condensers can
pump the rescaled angle back up (the earlier "CA removes the phase space the
target needed" claim was an artifact of the wrong aperture model).

**No `Gun` class exists** — the emitter everywhere is `Source` (the standard
column's is named "G"). Flagging rather than adding one; a `Gun(Source)`
subclass is trivial if wanted.

**basic_column changes are shared-template changes** (CA, stated current,
detector tail; optics untouched, total length now 1.274 m with the detector
still at 1.264). Ondrej/Thomas: two hybrid tests had the column end
hardcoded and were updated.

## 2026-08-27 — [Done] Verbatim reload, examples green, the eight configurations
**Goal:** answer Eric's three challenges: reload without per-class helpers, all
examples running, and the eight-configuration demonstration wired into the
test suite.
- [x] `safeReinstantiate` restores the recorded `__dict__` verbatim; `_restore_attrs` deleted
- [x] all six examples run headless (01 needed `figs/`; 05 still called `_effective_strength()`/`focal_powers()` as methods)
- [x] `measureAtZ(live_only=)` — a beam measured after an aperture used to look uncut
- [x] `examples/07_eightConfigurations.py` + `test_eight_configurations.py`

**Outcome:** 145 tests (144 pass + the pre-existing insertion failure).

**Reload.** Eric was right that `_restore_attrs` should never have existed:
the plain SEASerializable round trip restores everything; the loss was
rayTEM's own `safeReinstantiate`, whose constructor-kwarg filter dropped any
`__dict__` name spelled differently from its kwarg. Now the constructor only
supplies the right class and validation; `obj.__dict__.update(dic)` makes the
recorded state win verbatim. (On storing the derived per-element current at
all: that call was Eric's — "if we round trip or if a user looks at the file,
that information should exist" — but moving it into serialized state deserved
an explicit prompt, noted.)

**Eight configurations** (`examples/07_eightConfigurations.py`): one column;
C1 focused onto CA or weak (high/low current), C2/C3 solved for a 30 mrad
probe or a parallel patch at the sample, PL1/PL2 solved for image or
diffraction at the detector. Every strength is a bracketed 1D root on a
transfer-block entry (B=0 imaging, D=0 collimation, A=0 diffraction), chained
crossover-by-crossover the way an operator drives a column. Verified: rays vs
matrix conjugate planes to 1e-15; wave crossovers on the diffraction family to
1e-14; covariance waists beside each image plane by the emittance focal shift
(7 mm at the 63x detector image — physics, in the table's footnote). In the
low state the 30 mrad probe is unreachable (CA removes phase space, not just
electrons); the solve reports the aperture-limited 1.24 mrad instead of
pretending. Figures: rays drawn over |psi(x,z)| for all 8 states.

**MACSTEM (open, important).** `src/pySEA/rayTEM/microscopes/MACSTEM/` is on
the PUBLIC repo's main — re-uploaded there by Thomas (tpchuckles) on 08-21/26,
commit messages say "for historical purposes". After the earlier
REMOVED_PRIVATE_INSTRUMENT_TREE scrub this needs an explicit decision between
Eric/Ondrej/Thomas; removing it from a branch fixes nothing while main and
history carry it. **The wiki refresh is deliberately NOT committed** — it
generates MACSTEM stubs and indexes its symbols, which would spread the
content further into the repo. Re-run `pysea-refresh-wiki` once the decision
is made.

## 2026-08-27 — [Done] Beam current is recorded state
**Goal:** make both the stated and the derived beam current survive a `.sea`
round trip.
**Why:** Eric: "This is an absolutely critical piece of information that should
always make it to the recorded state." He was right and I had it backwards.
- [x] the stated Source current survives reload
- [x] the derived per-element currents survive reload
- [x] regression test

**Outcome:** 136 tests, both currents round-trip to the bit.

**Two separate losses.** The *stated* source current never round-tripped at
all, and that predates the derived-current work: it is spelled `_beam_current`
in `__dict__` but `beam_current` in the constructor, so `safeReinstantiate`'s
kwarg filter dropped it and a reloaded Source silently reverted to the 1 nA
default. The value was in the HDF5 file the whole time — the loss was purely
on the read side. Fixed by naming it in `Source._restore_attrs`, the same
mechanism `_screen` already uses.

The *derived* current I had deliberately kept off the instances so it would not
be serialized, on the reasoning that a per-run result has no business in a
`.sea` file. That reasoning is contradicted by the package itself: `.I` and
`.rays` are stored on every section. It now lives on `Element._arriving_current`,
declared in `__init__` and listed in `_restore_attrs`. Stale like every other
stored result — change the source and re-propagate before trusting it.

**`subdivided` re-verified** after the `combine_drifts` fix, on
`basic_column.sea`: length exactly 1.264 m at every spacing, element count
58 -> 73 -> 159 -> 271 for dz = 50/10/5 mm, ray planes logged 25 -> 238, and
every named position preserved (only the blank `''` key moves, which is the
bucket all unnamed elements share and carries no meaning).

## 2026-08-27 — [Done] Audit the merge, land Element.beam_current
**Goal:** verify our work survived the merge into
`Signal_and_propagation_additions_new`, then continue on that branch.
**Why:** the branch was merged in an unusual way and Thomas resolved conflicts
by hand — silent losses were plausible.
- [x] audit every commit, marker and file against what we wrote
- [x] isolate and fix the one regression the merge introduced
- [x] `Element.beam_current` as a derived property

**Outcome:** 135 pass, 1 pre-existing failure.

**Audit.** Everything landed. The branch is `main` (including Thomas's PR #4)
with our commits rebased on top; all of them are present by message, all
markers and files are there, and the 6-column `convention` was correctly kept
(the merge point still carries the old 8-column form).

**The one regression.** `repair()` merges adjacent unnamed Drifts. That made
`Microscope.subdivided` a no-op — it cuts elements into slices and `repair()`
glues them back — and flattened the 60-slice envelope test. Fixed with a
`combine_drifts` flag: default `True`, so Thomas's tidying stays the default
for everyone else, and `subdivided` plus the envelope test opt out.
**Open question for Ondrej/Thomas:** should the default be `False` instead, i.e.
keep the element list exactly as a caller wrote it? Merging is convenient but
it silently discards z sampling.

**Pre-existing failure.** `test_section_insertion_microscope` fails on our
branch, at Thomas's merge point, and at the commit *before* it — so it predates
all of this. Worth knowing: `origin/main` is now at PR #6 and 11 of the 14
tests in that file fail there, versus 1 on our branch.

**`Element.beam_current`.** The Source states amps; everything else derives.
Each element reports the current *arriving* at it, so an aperture reads the
beam it receives and the next element reads what was passed. Measured on a
two-aperture column from 2 nA: 2000.0 / 2000.0 / 2000.0 / 45.0 / 45.0 / 8.889 pA.

Held in a module-level `WeakKeyDictionary`, not on the instance — everything in
`__dict__` is serialized, and a per-run result does not belong in a `.sea`
file. My first attempt put it in `__dict__` and cost 12 failures: it was stored
as `beam_current` and restored with `setattr` onto a getter-only property. A
*raising* property broke a second set of things, so elements outside a
propagated column return `None`, matching the `rays = None` convention.

## 2026-08-26 — [Done] The wave path carries current too
**Goal:** close the gap the beam-current work left open.
**Why:** Eric's model is that anything masking the beam changes the current —
"apertures that mask rays **or alter a wave's amplitude (like a mask)**". The
ray path did that; the wave path did not.
- [x] `Microscope.wave_current` / `wave_current_at(plane)`
- [x] tests for conservation, an amplitude mask, a phase screen, an aperture

**Outcome:** 133 -> 135 tests.

No new bookkeeping was needed, which is the nice part: the wave already carries
the information in its own amplitude. Anything multiplying psi by a modulus
below 1 reduces the integral of |psi|^2 by exactly that much.

The quantity is `sum |U|^2 dxi deta` on the scaled path. The factorization
`psi = U(x/s)/s` makes that equal `integral |psi|^2 dA`, so no reconstruction is
needed and it is conserved by free propagation whatever the frame does —
measured at 1.0000 through a two-drift column.

Measured, source 2 nA:
- clear column: 2000.000 pA (exactly conserved)
- 50% **amplitude** mask: 500.000 pA — |T|^2, not |T|
- pure phase screen: unchanged, since |exp(i chi)| = 1
- 20 um aperture on a 40 um beam: 490.8 pA, just under the (20/40)^2 = 25%
  a ray model gives, because the wave is genuinely masked and diffracts

Deliberately a **ratio** against the source plane rather than a renormalized
wavefunction: normalizing psi to carry amps would change every amplitude the
existing wave tests compare, for no gain. The two paths are not required to
agree exactly — an Aperture scales rays by a ratio of extents while the wave is
masked and diffracts — and the docstring says so rather than implying a
correspondence that does not hold.

**Process note for myself:** I proposed doing the amps work Eric had already
asked for and I had already done, four hours earlier in this same session
(`b328ed6`). Check `git log` before offering to build something.

## 2026-08-26 — [Done] Beam current in amps, and a reload table that lost two elements
**Goal:** Eric's C1 — the Source states a current in amps; everything
downstream derives it.
**Why:** `beam_current` reported a dimensionless fraction because nothing in
the column carried amps.
- [x] `Source.beam_current` (float, amps, default 1 nA)
- [x] `I` seeded in amps so `I.sum()` is the current at every plane
- [x] `Microscope.beam_current` / `current_at(plane)` report amps
- [x] fixed `safeReinstantiate`, which could not reload an Aperture or a Prism

**Outcome:** 133 -> 134 tests. A 2 nA source behind a 30 µm aperture now reads
45 pA at the exit, and `current_at` shows the loss happening AT the aperture.

**Design:** the Source is the only place a current is *stated*. `propagate_ray`
shares it over the rays, so `I` is amps-per-ray rather than a relative weight,
and every attenuating element reduces the total just by scaling — no separate
bookkeeping. Sections with no Source of their own inherit `I0` from the
previous section, so chaining is unchanged.

**Found on the way — a genuine bug, not mine.** `safeReinstantiate`'s kind→class
table had no entry for `Aperture` or `Prism`, so any saved column containing one
raised `KeyError` on load. **No column with an aperture had ever round-tripped**,
which matters precisely because apertures are what set the beam current. Both
added, and an unknown kind now says what is missing instead of raising a bare
`KeyError` from a dict lookup inside the loader. A new element class that
forgets this table will save fine and never load, so the message names it.

**Not done, flagged:** the wave path does not carry current. A mask that changes
`|psi|` changes it the same way, and the natural definition there is the
integral of `|psi|^2` scaled to the source's current — nothing computes it
today. There is also no `Gun` class yet; `Source` carries the attribute.

## 2026-08-26 — [Done] focus_error was dead code
**Goal:** fix the long-queued `focus_error` bug.
**Why:** it raised on any column, so it had not run in a long time.
- [x] stale `findPlanes` call
- [x] hard-coded instrument name
- [x] two unhelpful failure modes

**Outcome:** 130 -> 131 tests. `focus_error()` on basic_column now returns
0.50249 m, the first diffraction plane past C3 (there is one at 0.3052 before
it, so "first after" and "nearest" genuinely differ here).

**Three faults, one method:**
1. `findPlanes(self.rays, "x")` — the second parameter is the **rotation
   array**; the axis goes third. The string was handed to
   `convert_to_rotating_reference_frame`, which indexed it as an array:
   `TypeError: string indices must be integers`. Exactly the staleness
   CLAUDE.md warns about for the `R`/`I` split; this caller was never updated.
2. It looked up `"CL3"`, which no generic column has — basic_column's last
   condenser is `C3`. Now a parameter, `after="C3"`, because the element that
   means "after the condenser" is instrument-specific.
3. `get_element_position` unpacked `index()`'s `None` on a miss, so a wrong
   name surfaced as `TypeError: cannot unpack non-iterable NoneType`. It now
   raises `KeyError` listing the known names.

**Follow-up (same day):** `beam_current` and `convergence_angle` are now
properties, and `convergence_angle_at(z)` covers arbitrary planes. Both
docstrings state **semi**-angle explicitly. Two things that were wrong:
- `convergence_angle` read `xt` alone, which under-reports by `cos(KL)` on a
  thick objective — a factor of 3.7 on OL1. It reports `hypot(xt, yt)` now.
- it measured at the sample's entrance FACE, and `measureAtZ` returns the state
  *entering* a plane. That worked only because `0.05 + 0.01` lands an epsilon
  past `0.06` (`0.060000000000000005`); on a column landing short it would have
  silently returned the unconverged beam. It measures at the element's midpoint
  now. Round-trips against `build_objective_section(alpha=...)` exactly.

**What focus_error actually measures** (Eric asked whether it is just C10): it
is a traced crossover position, so every aberration the ray path applies is in
it — but at a condenser's pupil angle only the quadratic terms move one. On
basic_column the rays reach 9.6 µm at C3, a 0.1 mrad pupil angle, so C10's kick
is 1.2e-6 rad against C30's 1.3e-14. Measured: C10 = 1 mm shifts it 0.22 µm,
aligned C12 shifts it the other way by the same, C30 does not move it at all —
not even at 0.1 m. So it is a defocus measurement in practice, but it is not
`C10`: that is one lens's property, this is where the column puts a crossover.
And with spherical present there is no single crossover — `findPlanes` reports
where its chosen ray pair crosses, which is height-dependent.

**Mentioned, not fixed** (surgical scope):
- `Microscope.index` still *prints* `ERROR: name ... not found` and returns
  `None`. With `get_element_position` now raising, that print is pure noise on
  the way to a real exception; other callers may rely on the `None`, so I left
  it.
- `convergence_angle` hard-codes `"OL1"` the same way `focus_error` hard-coded
  `"CL3"`. It works on basic_column so nothing is broken today, but it carries
  the same instrument assumption and should take an `after=`-style parameter.
- `beam_current`/`convergence_angle`/`focus_error` are all commented
  `#@property` — someone meant them to be properties. `focus_error` now takes
  parameters, so it should stay a method.

## 2026-08-26 — [Done] Astigmatism/coma coverage, and an attribute guard
**Goal:** prove the aberration generality on the terms people actually tune,
and stop the silent-attribute failure mode.
**Why:** only C30 had ever run end-to-end, and three separate bugs this session
traced to assigning an attribute that silently did nothing.
- [x] C12 aligned/skew and C21 coma exercised end-to-end, rays and wave
- [x] `SealedAttributes` on Element, MicroscopeSection and Microscope
- [x] corrected an unverified claim about `MEDIUM_SLICES`

**Outcome:** 125 -> 130 tests.

**Aberration coverage.** Each new test pins a property a magnitude-only check
would miss:
- aligned `C12` is quadratic, so it is a per-axis POWER change: two crossovers
  at `1/(P ± C12·P²)`, each a *true* point focus (zero spread across the fan),
  both exact to 1e-12. The wave agrees at the same z and gives a line focus,
  8x longer in y than x against 1.00 for an ideal lens.
- skew `C12` leaves the axes at the ideal focus and splits the DIAGONALS by the
  same amounts — the same aberration turned by π/2m.
- `C21` vs `C30` pins PARITY, the easiest thing to get silently wrong: the ray
  aberration goes as θⁿ, so coma (n=2) deflects opposite pupil edges the SAME
  way — that is what makes a comet tail — while spherical (n=3) deflects them
  oppositely and stays centred.

**The attribute guard needed three carve-outs, all real:**
1. underscore names and anything the class declares pass, so correct code is
   untouched;
2. **deserialization is exempt** — a file may carry names the class no longer
   declares, and the writer stores a private `_x` under the public key `x`, so
   refusing them would make an old file unopenable;
3. `copy()` re-seals, because `deepcopy` restores `__dict__` without running
   `__init__`.

The seal lives in a module-level `WeakSet`, **not** on the instance: everything
in `__dict__` is serialized, so a flag there would be written into every `.sea`
file and then handed back to `setattr` on load, under a name the guard itself
would refuse. That one cost a debugging round.

Also declared `shift_x`/`shift_y`/`tilt_x`/`tilt_y`/`rotation` as class
attributes. They were always part of the ray contract (`propagate_ray` reads
them via `getattr(self, ..., 0)`) but no class declared them, so they existed
only once someone assigned one — which the guard then refused.

**Correction to my own note above:** I had written that 16 `MEDIUM_SLICES`
"matches the ray integral to ~1%". I never measured it and it is wrong twice
over — the measurement cannot resolve 1%, and the two numbers are not meant to
be equal. Measured: converged from 2-4 slices on. The wave best focus (-2.1 nm)
and the ray c20 fit (-1.1 nm) differ by ~2x because one is the brightest plane
and the other a paraxial fit coefficient. Do not tune MEDIUM_SLICES to close it.

## 2026-08-26 — [Done] Objective section, six-panel example, three wave-path bugs
**Goal:** show spherical aberration in rays AND in the wave, on the real OL1.
**Why:** Eric asked for a wave propagation with and without C30; the full column
is the wrong instrument for it (an aberration is ~1e-4 of the beam width over a
metre) and the request surfaced three bugs on the way.
- [x] `microscopes/objective_section.py` + `.sea`: basic_column's O section
      behind a source, lifted from `build_basic_column()` so OL1/OL2 cannot drift
- [x] example 06 rebuilt as six panels: A/B rays, C/D wave |psi(x,z)|, E/F
      focal surface + Strehl
- [x] three bugs found and fixed, each with a regression test

**Outcome:** 115 -> 125 tests. The figure now shows the same caustic in two
independent representations.

**Configuration, and why:** 30 mrad as asked, C30 = 0.1 mm, grid **256²**.

I first sized the grid at 2048² and told Eric 1 mm would need ~8192. That was
measured **before** the screen was distributed over `MEDIUM_SLICES`, and I did
not re-measure after. Each slice applies 1/16 of the phase, so the sampling
requirement fell 16x: 256² now carries C30 up to ~0.5 mm at 30 mrad, and the
whole example runs in 5.6 s instead of minutes, with identical numbers
(Strehl 0.033 either way). Re-measure after a change that alters the quantity
being measured.

Still true and worth keeping: the samples needed go as `C30 * alpha^4`; the
screen is evaluated over the WHOLE grid, not just inside the aperture; and it
is the grid **corner** that binds, at sqrt(2) times the half-extent, i.e. 4x
the edge requirement. So empty grid beyond the aperture costs sampling rather
than buying safety — `wave_oversample` is deliberately tight.

**ALPHA means the convergence semi-angle at the sample, and it is the ray's
TOTAL deflection.** I briefly mis-read it as 8 mrad by looking at `xt` alone:
OL1 is thick, so it also rotates the ray by its Larmor angle (KL = 1.30 rad),
leaving only `cos(KL)` of the 30.000 mrad in x and the rest in y. `focal_power`
(125, f = 8 mm) and the ray matrix's `-C` (33.68) differ by exactly that
`cos(KL)` — not a bug, but a trap for anyone reading one axis.

**The three bugs, all silent:**

1. **A thick medium dropped its screen entirely.** A thick lens reports a
   `_scaled_segment`, so the scaled path carries it as a quadratic-index medium
   and never calls `phase_shift` — discarding its aberrations and any supplied
   screen. Ideal and aberrated runs came back bit-for-bit identical.
2. **Then it over-applied it.** The first fix put the whole screen at the body
   centre. An aberration is a property of the medium: the ray side integrates
   it along the body and gets **0.122x** the thin-lens value for OL1, so a
   mid-body screen had the wave seeing ~8x more than the rays. Now distributed
   over `MEDIUM_SLICES`; the local ray height comes free because the screen is
   evaluated at `s*dxi` and s shrinks as the body focuses. Wave best focus
   -15 nm vs the ray's -11 nm c20 — same sign and order, and not expected to
   match exactly (brightest plane vs paraxial fit, 5 nm plane spacing).
3. **Sub-nanometre drifts did not propagate at all.** Both scaled engines used
   `tol = 1e-9 * (abs(dz) + abs(z) + 1.0)`, whose `+ 1.0` makes an absolute
   ~1 nm floor. `while remaining > tol` was false on the first pass, so the
   field was returned untouched and z never advanced — no error. 80 sub-nm
   planes all reported the same z, which showed up as a blank band in the
   figure. Tolerance is now relative with no floor.

**Also for Ondrej — two API sharp edges that cost me time:**
- `m["G"]` resolves to the **section** named G, not the source also named G.
  Use `m["G"]["G"]` or `m[0, 0]`. Elements and sections share a namespace.
- Assigning an unknown attribute to an Element or MicroscopeSection silently
  succeeds and does nothing (`e.Cs = ...`, `section.np_xy = ...`). Three
  separate silent no-ops in this session traced back to it. A `__setattr__`
  guard would turn them into immediate errors; not done, flagged.

## 2026-08-25 — [Done] Aberrations class, generic application, and screens
**Goal:** aberrations in one class applied generically, and a screen that can
carry amplitude as well as phase.
**Why:** aberrations were 25 flat scalars on `Lens` plus a letter-named table in
`waveoptics`, so the ray path implemented `C3` alone while the wave path
implemented everything; and a screen was unit-modulus by construction, so no
element could both block the beam and phase-shift it.
- [x] `Aberrations(SEASerializable)` in true Krivanek `C_{n,m}` through 5th order
- [x] generic `phase_at` / `deflection_at`; `Element.aberration_kick` no longer
      knows which aberration it is applying
- [x] `apply_aberrations=True` on all six drivers, via a suspend context
- [x] screens may be complex; `Aperture` folded in; supplied screens; volumes
- [x] sea-eco reader extended to 5th order (separate repo, own LOG entry)
- [x] wiki: new `aberrations.md`, updated `elements.md` / `waveoptics.md`

**Outcome:** 84 → 123 tests. Verified against the code replaced rather than
against itself: C30 phase to 5e-20, aligned C12 to 7e-18, `deflection_at` vs a
central difference to 2e-11 across seven terms of five orders, the C30 kick vs
the old closed form to 5e-26, and `Aperture`'s masking bit-for-bit including
the anisotropic ellipse.

**Decisions worth remembering:**
- **C10 must NOT fold into `focal_power`.** The generic kick already delivers
  it — for n=1, m=0 the deflection is exactly `-C10·P²·x`, so rays cross at
  `1/(P + C10 P²)`, matching the frame to 1e-12. Folding it in would apply it
  twice, and `focal_power` is also the pupil-angle scale every other term is
  measured against.
- **Screen dtype carries meaning**: real = phase, complex = transmission. It
  cannot be ambiguous, which is why it is not a flag.
- `Lens.Cs` **removed**, not kept as an alias. Once the flat attributes went,
  `lens.Cs = x` after construction silently did nothing — example 06 had been
  running an unaberrated column because of it.

**Bugs found along the way** (all fixed, all with regression tests):
1. `safeReinstantiate` keeps only constructor kwargs, so `_screen` was dropped
   on reload — and so were `aberrations` on any element that does not take them
   as a kwarg. A `Quadrapole`'s coefficients were silently lost.
2. Merging a complex screen into a real volume: first the imaginary part was
   discarded by the assignment, then `astype()` "fixed" it while changing what
   the untouched slices MEAN — real 0 is transparent, complex 0 is opaque, so
   casting blacks out the volume. Converted with `exp(iχ)` instead.
3. My own justification for exempting complex screens from the sampling guard
   was wrong. It does not reject hard edges (a binary plate steps |ΔT| = 1.0,
   under π). The real problem is that the guard cannot see complex data at all:
   a screen winding 3.78 rad/px — aliased — reports 1.90 and passes.

**For Ondrej:** if you attach aberrations to an element type that does not take
them as a constructor kwarg, they now survive reload via
`Element._restore_attrs`; add to that tuple rather than to a subclass if you
add another non-kwarg stored attribute.
Status markers: `[Under Construction]` while in progress · `[Done]` when complete.

---

<!-- Add entries here as work is completed. See notes/ondrej/LOG.md for format reference. -->

## 2026-08-23 — [Done] Aberration on the REAL objective, and distributed through its body
**Goal:** show the aberration on a lens in `basic_column`, with pictures, and make the thick-lens treatment quantitative.
**Why:** Eric: "I don't know what you are talking about because there are no visuals. Also, what lens are you aberrating?" Both fair — see below.
- [x] `examples/06_aberratedObjective.py` + figure, on OL1
- [x] `focal_surface(near=, window=)`
- [x] distributed thick-body aberration
- [x] validated against the perturbed ray ODE

**Outcome, and the honest answer to "what lens":** **none of ours, until now.**
Every Cs number I had reported came from a standalone toy lens built for the
purpose; every lens in `basic_column` still had `Cs = 0`. `examples/06` now puts
Cs = 1 mm on the actual objective **OL1** (f = 8 mm, 10 mm thick, z = 490 mm) and
plots it — ray panels through `Microscope.show(plt_ax=)`, per the standing
directive.

**Putting it on a real column immediately broke `focal_surface`.** It took the
*global* closest approach, and an objective-aperture bundle crosses the axis at
the condenser foci long before the objective, so it reported a **130 mm "sag"
belonging to a different lens**. New `near=` (which paraxial plane) and
`window=` (how far to search, defaulting to half the gap to the neighbouring
plane of the family). This was invisible on toy columns with one focus — a
reminder that single-lens tests do not exercise plane selection at all.

**Then the thick-body approximation had to go.** The whole aberration sat at the
entrance face, exact only for a thin lens. A body's perturbation is
*distributed*: a slice `dz` acts on the **local** `r(z)`, and the remaining body
carries that kick to the exit, so it produces a position offset as well as an
angle one. `aberration_kick`'s contract widened from `(dxt, dyt)` to
`(dx, dy, dxt, dyt)` accordingly.

Validated against **direct integration of `x'' = -K²x - c x r²`** through the
body — an independent route using none of the transfer-block machinery.
Agreement 1.1e-6 to 5.7e-6 relative over h = 20–80 um, with the residual growing
as `h²`: exactly the second-order term a first-order perturbation omits, which
is the signature you want rather than a flat tolerance.

**Size of the correction, and a number I got wrong first:** on OL1 (`KL = 1.30`)
the entrance-face model over-estimates the exit angle by **3.3x**. I initially
estimated "roughly twofold" from `integral cos^3 / L = 0.51`, forgetting that
each slice's kick is *itself focused by the rest of the body* — the `D(L-z) =
cos K(L-z)` weight — which brings it to 0.31. Docstring corrected. OL1's fitted
`c20` moves from **-40.3 to -63.7 nm**, sag 40.1 -> 63.2 nm.

**A change I made and reverted:** `_check_screen_sampling` measures the phase
step over the whole grid, and a quartic screen is 1235 rad at the grid edge of an
f = 8 mm lens, so I switched it to measure at the beam support. Then I measured
the support: the band-limited disc's tails exceed the 1e-6 threshold across the
*entire* grid, so the beam genuinely does reach where the screen is steep and the
original guard was right. Reverted; my earlier "the guard earns its keep" claim
stands. The real consequence is a physical limit, now stated on the figure: at
10 mrad on an 8 mm lens the screen cannot be sampled, so the wave panel runs at
5 mrad where the Strehl loss is only 1.4%.

Also: peak-normalising each wave curve **hid** that 1.4% entirely. Both are now
scaled to the ideal peak. 111 tests green (was 110).

## 2026-08-23 — [Done] Wave path verified against the closed-form BFP (step 3b)
**Goal:** check the aberrated wave path against something it does not itself compute.
**Why:** comparing the applied screen against the screen proves nothing; the plan asked for a real cross-check.
- [x] closed-form back-focal-plane comparison
- [x] fixed the resampling replica it exposed

**Outcome:** One focal length past a lens the field is
`FT[A(r) exp(i chi_ab)]` at `q = x'/(lambda f)`. Reproducing it exercises the
*whole* chain — parabola into the frame, quartic left as a screen on U,
carrier-free kernel, hybrid frame switching, reconstruction — while reusing none
of it. **Agreement: 4.1e-4 (Cs = 0) and 5.1e-4 (Cs = 1 mm)** on peak-normalised
intensities.

Two things had to be right before the comparison meant anything, both worth
remembering for future cross-checks: **the same aperture model on both sides**
(the source seeds a *band-limited* disc, so a binary reference differs by ~1e-2
for that reason alone and it would have been easy to blame the propagation), and
**the same grid**, so no interpolation sits in between.

**Bug found by the check — a resampling replica.**
`reconstruct_physical_wave`'s target-grid path is band-limited and therefore
*periodic*, so requesting a grid spanning **more** than the modelled field of
view came back filled with **replicas of the beam**. Here the native
reconstruction spans 17.2 nm while the requested grid spans 41.3 nm, and a
replica of the central peak sat 213 px out at **0.95 of full intensity, where
the true field is exactly zero** — a bright feature in a region where nothing
was ever propagated. Now zero-filled outside the modelled field, which is what
"not modelled" means, with the docstring pointing at `target_shape` rather than
a coarser `target_dx` for covering more area.

Worth stressing how it surfaced: the check read **0.95** instead of 5e-4, and
the discrepancy was present with `Cs = 0` too — which is what said "this is not
about aberration". A test that only looked at the central region, or only at the
aberrated case, would have missed it entirely.

**Remaining in step 3:** a `chi(q)` *fitter* that reports coefficients from a
logged plane. The propagation is now verified, so a fitter can be trusted once
written; the difficulty is physical rather than mechanical — at these parameters
the aberration disc (`Cs alpha^3` ~ 0.3 nm) and the diffraction disc
(`lambda/alpha` ~ 0.37 nm) are the same size, so a fit must separate them rather
than assume one dominates. 110 tests green (was 108).

## 2026-08-23 — [Done] Wave side of spherical aberration (focal-surfaces step 3a)
**Goal:** get the aberration into the wave path as the residual screen it physically is, from the *same* chi the ray side uses.
**Why:** a wave screen and a ray kick derived separately could disagree and nothing would notice.
- [x] `waveoptics.spherical_phase`
- [x] `Lens.phase_shift` carries it, fixed and scaled
- [x] ray/wave consistency test

**Outcome:** One `chi = -k Cs r^4 / 4f^4` on the lens plane; the ray kick is its
gradient `(1/k) dchi/dr`, not a separately derived expression. Substituting the
ray relation `r = -f theta` gives the familiar angular form `-k Cs theta^4/4`,
i.e. the `Cs lambda^3 q^4 / 4` term of the usual aberration function.

`Lens.phase_shift` carries it both ways: added to the real-space screen on the
fixed path, and on the scaled path the parabola is absorbed into the curvature
as before while the **quartic stays as a residual screen on U** at `x = s*xi` —
a quadratic frame cannot absorb a quartic. That is exactly why the paraxial
crossover does not move when Cs is switched on, which is the behaviour the plan
predicted and it now demonstrably does.

**Verification that distinguishes "is the gradient" from "is close":** the FD
error between kick and screen-gradient falls as `dx^2`, ratios 4.00 and 4.00
over two refinements. A fixed tolerance would not have told those apart.
Doubling `s` scales the screen by 16, as `r^4` must.

**Two things that looked like null results and were not:**
- At a 4 um aperture the quartic is ~4e-8 rad — aberration is genuinely
  invisible there, so my first "no effect" test was measuring nothing.
- A **fixed grid cannot show this at a realistic aperture either**: at
  f = 45 mm the *parabola* alone is ~1e7 rad on such a grid. The scaled path is
  the only one that can, which is what it exists for. On a 6.7 mrad aperture
  with Cs = 1 mm the focus broadens and its peak falls monotonically
  (0.02512 -> 0.02434 -> 0.02200 for Cs = 0, 0.5, 1 mm) while the crossover
  stays pinned at 46.000000 mm.

**The sampling guard earns its keep here.** The quartic grows as `r^4`, so it
out-runs the grid faster than anything else in a column: at half the sampling
of the passing test it is refused (4.51 rad/pixel) rather than aliased into a
plausible-looking focus. Both the passing and the refused case are tested.

**Not done — step 3's second half:** a reader that fits `chi(q)` from the logged
plane and reports coefficients comparable with the ray-side fit. Worth flagging
why it needs care rather than being a quick addition: at realistic parameters
the aberration disc (`Cs alpha^3` ~ 0.3 nm here) and the diffraction disc
(`lambda/alpha` ~ 0.37 nm) are the same size, so a fit has to separate them
rather than assume one dominates. The honest independent check is against the
closed-form back-focal-plane field, `FT[A(r) exp(i chi)]`, which tests the whole
propagation chain without reusing the screen. 108 tests green (was 104).

## 2026-08-23 — [Done] Spherical aberration + focal_surface (aberrated-focal-surfaces step 2)
**Goal:** a real aberration source, and the surface a plane becomes once one exists.
**Why:** step 1 pinned the criterion; without an aberration source `focal_surface` would return a constant.
- [x] `Element.aberration_kick` declaration + generic consumption in `propagate_ray`
- [x] `Lens(Cs=...)`
- [x] `Microscope.focal_surface` with sag + fit
- [x] tests

**Outcome:** `Element.aberration_kick` is the companion declaration to
`transfer_matrix`: the matrix can only express optics *linear* in the ray
vector — that **is** the paraxial approximation — and an aberration is by
definition what lies outside it. The element declares the extra angular
deflection, the generic `propagate_ray` applies it. Base returns `None`, so
aberration-free columns are bit-for-bit unchanged and the paraxial planes are
untouched *by construction*, which is exactly what lets aberration be defined as
the departure from them.

`Lens(Cs=...)`: from `chi = -k r²/2f - k Cs r⁴/4f⁴` the kick is
`-(Cs/f⁴)·x·r²`, radial as a round lens must be. Traced rays reproduce
`z = f/(1 + Cs h²/f³)` to 1e-12 — the classic longitudinal spherical aberration
`-Cs alpha²` — with the paraxial plane still at `f` exactly.

`Microscope.focal_surface` traces a sampled bundle and takes each ray's closest
approach to the axis, closed-form rather than searched (between elements a ray
is straight, so `r²(z)` is a quadratic with a vertex). Reports samples, sag, and
a fit. **Acceptance met:** with no aberration the surface is flat and equals the
paraxial plane to 1e-12.

**The fit basis needed correcting, and the reason generalizes.** I first used
the plan's `z = z0 + c20 r² + c22 r² cos2(phi-phi22)` — every term scaling as
`r²`. But **paraxial astigmatism does not scale with aperture radius**: a
quadrupole splits the focus by azimuth only, because both transverse components
scale together, so a ray at azimuth 0 meets the axis at the x focus whatever its
height. Aperture aberrations *do* grow as `r²`. Fitting everything to `r²` put a
quadrupole's splitting into `c22` and reported an aperture aberration that was
not there — **729 um where the real half-split is 506 um**. The basis now
carries an `r`-independent two-fold term separately; with one radius the two are
degenerate and returned as `nan` rather than guessed.

| column | astig | c20 | expected |
|---|---|---|---|
| ideal lens | 0 | 0 | — |
| Cs = 1 mm | 0 | -0.0444 um | `-Cs alpha²` = -0.0444 um |
| quadrupole | 506.250 um | 0 | half-split 506.314 um |
| both | 506.250 um | -0.0444 um | superposes, no cross-talk |

**Two things found while testing, both kept as behaviour:** a single-lens column
with the object 10 mm before a 45 mm lens has **no real image plane** (the image
is virtual), and `focal_surface` says so rather than inventing a reference. And
an `image` bundle leaves an on-axis point, so every ray is *on* the axis at
launch — the closest-approach search has to be bounded downstream of the optics
or it returns the launch plane. Both are now tested.

**Plan Q2 (coefficients on the Element or the Microscope) answered by default,
not by decision:** per-element, because that is how they are measured and it
composes without a transport rule. Say if you want system-level totals instead.
104 tests green (was 100).

## 2026-08-23 — [Done] Least-confusion criterion locked down (aberrated-focal-surfaces step 1)
**Goal:** pin the criterion `focal_surface` will use, in the aberration-free case where it must degenerate to the paraxial roots.
**Why:** [PLAN_2026-08-22_aberrated-focal-surfaces.md](PLAN_2026-08-22_aberrated-focal-surfaces.md) step 1 — "no new physics, pure scaffolding, and it locks the criterion down".
- [x] criterion verified against `conjugate_planes` for both families
- [x] two bugs found and fixed on the way

**Outcome:** The criterion is settled, and it needed **no new scaffolding** —
the existing covariance machinery already expresses it, once two real bugs were
out of the way. Least confusion is now verified to reproduce *both* conjugate
families exactly, by a route wholly independent of the transfer-block root
solve:

| seed | `Sigma_11` | its minima are |
|---|---|---|
| point (`diag(0, theta^2)`) | `B^2 theta^2` | the **B = 0 image** planes |
| parallel (`diag(x^2, 0)`) | `A^2 x^2` | the **A = 0 diffraction** planes |

Both match `conjugate_planes` on basic_column to 1e-12.

**Bug 1 — `beam_waists` reported maxima as waists.** `_waist_roots` solved
`Sigma_12 = 0`, which is *stationary*, not minimal. Differentially
`Sigma_11' = 2 Sigma_12` and `Sigma_12' = Sigma_22 - kappa Sigma_11`, so
`Sigma_11'' = 2(Sigma_22 - kappa Sigma_11)`. In free space `kappa = 0` and every
stationary point is a minimum — which is why this hid — but **inside a focusing
body the lens reverses the divergence and the root is the beam's widest point**.
On basic_column with a point seed, five of ten reported waists were maxima, one
inside each of C1, C3, OL2, PL1, PL3. Now classified by the sign of
`Sigma_22 - kappa Sigma_11`. This was mine, introduced with `_waist_roots`
earlier in this branch.

**Bug 2 — a collimated beam was refused.** The guard demanded a strictly
positive *angular* variance, so a perfectly parallel seed raised. It is a
legitimate beam: it never focuses in free space (which the per-segment solve
already reports by finding no root) and gains divergence at the first lens. The
effect was that `beam_waists` could express a point source but not a collimated
one — the image family but not the diffraction family. Relaxed to non-negative
variances with at least one positive, so the degenerate all-zero seed is still
refused.

**Deviation from the plan, deliberate:** step 1 called for a
`Microscope.focal_surface(...)` API. I have *not* added it. In the linear regime
it would return a constant surface for every field point and azimuth, because
the paraxial transfer really has no field dependence — so it would be an API
with nothing to say until an aberration source exists. The criterion it needed
is now pinned by tests, which was the actual point of step 1; `focal_surface`
should land together with step 2 (Cs on a round lens) so its first output is a
real surface with a non-zero sag. Flagging rather than silently reordering.

**Also settles plan open question 1 partly:** the ray side's criterion is the
minimum of the second moment, which is basis-free — the Seidel-vs-Zernike choice
only arises at the *fit* stage, not the measurement. 100 tests green (was 99).

## 2026-08-23 — [Done] Mid-element frame switching
**Goal:** let a crossover that falls inside an element body be crossed, and the plane reported where it actually is.
**Why:** Eric: "fix the mid-element frame switching issue first... just do it." It was worse than a refusal — see below.
- [x] `propagate_quadratic_segment_hybrid`: flatten/cross/rediverge using the segment's own law
- [x] restore the original ray past the crossing, so downstream planes stay right
- [x] tests, docs, notebook

**Outcome:** Done. The starting symptom was that a crossover inside a body was
refused. The real behaviour was worse: the free engine flattened **around** the
body and recorded the crossover as `z + |R|`, a straight-line extrapolation
through the element as if it were empty — so the plane was reported in the wrong
place *silently*, not reported as unavailable.

Fixing it needed three things a free segment never needs, all because a focusing
medium keeps bending the frame:

1. **A flat frame does not stay flat.** The medium re-converges it at once, so
   the flatten criterion re-fires every step and the axis never crosses.
   Suppressed while that axis has a crossing pending.
2. **The crossing is not at `z + |R|`.** Located with the body's law inside it
   and straight-line past it (`_crossing_from`).
3. **The rediverge cannot require a flat frame, nor use `R = d`.** By the time
   the frame reaches the crossing the medium has re-curved it (measured
   R = -3.66, not inf), so the branch never fired. And a ray that crossed at
   `z_c` has curvature `B(d)/D(d)` of the *medium* — which is the familiar `d`
   only when `kappa = 0`.

**Restoring the ray mattered as much as logging the plane.** My first cut logged
the in-body crossover correctly (99 mm -> 20 nm) but left every *downstream*
plane wrong by 94-594 um, because the frame that crossed was abandoned. A fourth
bug hid behind that: with the marker pending, the step ran to the body exit in
one go, so the rediverge point was never examined — the step now stops where the
rediverge first becomes representable.

basic_column, point object at -500 mm, image planes (nm from analytic):

| plane | before | after |
|---|---|---|
| 1 | 0.000 | 0.000 |
| 2 (inside C3) | **99 mm** | 20.4 |
| 3 | 594 um | 0.058 |
| 4 | 99 um | 0.011 |
| 5 | 234 um | 0.026 |

The common case is untouched **by construction**: with no interior zero and no
pending marker the traversal makes a single exact call and returns its output
unchanged (asserted bit-identical for both a thick lens and a thick quadrupole),
so flat-seeded runs still put all five diffraction planes at 0.000 um.

**Residual, different in kind:** the 20 nm on plane 2 is the *free* engine
flattening at 0.314916, just before C3, and extrapolating `z + |R|` through it —
the same straight-through assumption, now on the other side of the boundary.
Closing it needs the engine to see what optics lie downstream of a flatten,
which it cannot from inside a free segment. 6 parts in 1e8; documented, not
chased.

**Correction to my previous LOG entry:** I wrote there that "a frame switch
re-seeds the frame as a different reference ray". That was wrong as a general
claim — the free engine records `z_cross` from the *pre*-switch frame and the
rediverge restores the ray with `R = z - z_cross`, which is why flat-seeded runs
were always exact. The real defect was narrower: that restore is a free-space
identity, and it silently fails when an element sits between the flatten and the
rediverge. 99 tests green (was 96).

## 2026-08-23 — [Done] Wave image planes (D5) + `wavefield_at` exactness
**Goal:** make a hybrid run find and log image planes, not just back-focal planes.
**Why:** Eric: "for the wave optics plane finding we need to find the image planes first. I think we only found the back focal planes." Correct — and the reason is structural.
- [x] both conjugate families logged, tagged `image-x`/`image-y`
- [x] `_scaled_plane_at` — exact field at any z
- [x] fix `wavefield_at`'s silent snap to nearest
- [x] tests + dev doc

**Outcome:** The cause: a scaled frame **is** a reference ray, so a run finds
only the family its *seed* belongs to. A flat (parallel) seed makes `s ∝ A`, so
`s = 0` is a diffraction plane; a point seed makes `s ∝ B` and finds image
planes. The usual source is flat, hence back-focal planes only.

I had proposed a "shadow frame" — a second (s, R) tracked alongside the real
one. **Didn't build it.** The missing family is just the other column of the
same transfer matrix, which `_accumulate_blocks` already accumulates for
`conjugate_planes`; a shadow frame would be a second implementation of that
walk, free to drift from the first exactly as the quadrupole's three methods
had. So the run asks `conjugate_planes` for both families and logs a wavefield
at each plane it did not already produce. `Microscope.image_planes` /
`.diffraction_planes` carry both, per axis.

**A real bug underneath it:** `wavefield_at(z)` silently **snapped to the
nearest logged plane**. Asking for the field at an image plane returned a
different plane with no warning — which is precisely the query D5 makes, so it
would have quietly produced wrong pictures. New `_scaled_plane_at(z)` returns
the field at exactly z: the logged plane when one is there, else the nearest
upstream plane advanced through the intervening free space (closed form). It
refuses rather than approximating when an element intervenes, naming the
element and pointing at `subdivided()`.

"Is this stretch free space?" is asked of the **optics, not the type**:
`_acts_on_rays` compares the element's own block against `[[1, L], [0, 1]]` on
both axes, so drifts, fiducials and zero-strength lenses are transparent
without this layer knowing any element class, and an astigmatic element counts
as acting even if one axis is free. My first cut treated every element as
blocking, which made a plain drift an obstacle — caught because the image plane
then failed to log.

Numbers: `diffraction_planes` == the frame's own crossovers exactly; ray and
frame finders agree on the image plane; **`B = -2.1e-17`** at the logged image
plane (`B = 0` *is* the definition); `_scaled_plane_at` reproduces an
independent engine path — the same column cut at that z — to **8.7e-16** in the
field and 1e-16 in the frame state. 96 tests green (was 92).

**Worth knowing generally, found while cross-checking:** with `absorb > 0` the
two paths differ by 2.8e-3, because the absorber is **path dependent** — one
long sub-stepped segment windows the escaping halo differently than two short
ones. So where you cut a column changes the answer at the 1e-3 level whenever
the absorber is on. Not a bug (the absorber models real loss), but it means
exactness checks must run at `absorb=0`. Recorded in the dev doc.

**Two verification attempts that did NOT work,** noted so they are not retried:
measuring beam width in *pixels* (the reconstructed pixel size scales with `s`,
so the px width is constant by construction), and edge-sharpness or shape
correlation through focus on this column (a uniform disk under parallel
illumination has enormous depth of field — the test object cannot resolve
focus). `B = 0` against the matrix is the check that actually bites.

## 2026-08-23 — [Done] Thick quadrupole as an exact scaled segment (issue #3 step 3)
**Goal:** carry a thick quadrupole in the scaled wave path exactly, as the thick round lens already is.
**Why:** a thick quad was approximated as a thin kick between two half-length drifts, putting its wave crossover 72-2315 um off the ray-traced diffraction plane.
- [x] per-axis signed curvature in the segment propagator; drop the anisotropic refusal
- [x] `Quadrapole._scaled_segment()`
- [x] tests + wiki

**Outcome:** Done, and the generalization paid for itself. **Δτ has a
law-agnostic closed form.** For any segment whose block has unit determinant,

    Δτ = B / (s0 * s(dz)),   s(dz) = A*s0 + B*u0

with `(A, B)` the segment's own transfer row. It is a Wronskian consequence of
`det M = 1` (for any solution s1, the second independent solution is
`s1*∫dz/s1²`, and unit Wronskian fixes the constant). So free space, harmonic
and hyperbolic collapse to **one formula instead of three** — the previous
implementation carried a phase angle φ and a `tan(kΔz − φ) + tan φ` form that
only worked for the focusing case, and deriving the hyperbolic counterpart
separately would have meant three branches plus their degenerate cases.
Verified against numerical `∫dz/s²` to ~1e-16 in all three regimes. It also
ties Δτ directly to the block the element already declares for the ray side,
which is the same declare/consume shape as `transfer_matrix` -> `propagate_ray`.

Interface changes: `scaled_delta_tau_quadratic` now takes a **signed curvature**
`kappa` (1/m², of `u'' + kappa*u = 0`) instead of a focusing-only strength `K`;
new `segment_block` (the (A, B) row), `_segment_slope` (the second row) and
`segment_zero` (interior crossover, per law — a focusing segment can cross
repeatedly, a defocusing one *at most once* and only when entered converging
hard enough, free space at most once). `propagate_quadratic_segment_scaled`
takes per-axis kappa and no longer refuses anisotropic frames: the two axes
accumulate different Δτ and the paraxial kernel is separable, so one transform
pair still does it; errors name the failing axis.

`rotate` changed from a bool to the **Larmor angle in radians**. The propagator
was deriving `-K*dz` itself, which is only right for a round lens; now the
element declares it (`_scaled_segment()` returns `('quadratic', kappa, larmor)`
— Lens `(K**2, -K*L)`, Quadrapole `((+K², -K²), 0.0)`, since a quadrupole has no
axial field). A rotation on an anisotropic segment is refused rather than
silently applied, because Larmor rotation mixes the axes.

**Payoff:** a thick-quad column's wave crossover now lands on the ray-traced
diffraction plane to **0.0000 um** at every strength/length tested (8.0/30mm,
12.0/30mm, 8.0/80mm, 20.0/50mm), where the thin-kick route sat 72, 164, 1424
and 2315 um away. The defocusing axis correctly logs **no** crossover — it never
had one to find. 92 tests green (was 89).

**Issue #3 is now closed except step 4 (skew).** Skew stays a feature, not a
sign fix: a rotated quadrupole couples x and y, so it cannot be written as two
independent 2x2 blocks, and the class has no angle parameter.

## 2026-08-23 — [Done] Thick quadrupole: symplecticity + axis convention (issue #3 steps 1-2)
**Goal:** make the thick quadrupole block symplectic and give thin and thick one shared axis convention (K > 0 focuses x).
**Why:** the thick y-block `[[C, S/K], [+K S, C]]` has det = cos(2|KL|) = 0.75 over a 30 mm body — 25%% of phase-space area lost, violating Liouville; and thin/thick currently disagree on which axis focuses, so a quadrupole changes behavior when you give it a length.
- [x] symplectic blocks (trig on the focusing axis, cosh/sinh on the defocusing one)
- [x] k = |K| in the B term (currently signed, so it inverts for K < 0)
- [x] one convention in both branches; drop the thin X,Y swap
- [x] single body-block helper shared by transfer_matrix / transfer_block / focal_powers
- [x] tests: det == 1, halves compose, emittance invariant, thin/thick agree on sign

**Outcome:** Issue #3 steps 1-2 done. A quadrupole focuses one axis and
defocuses the other, so `u'' -+ k^2 u = 0` is harmonic on one and **hyperbolic**
on the other; the thick branch used cos/sin on both, so the defocusing block had
`det = cos(2|KL|)` = **0.7518** for strength 12 over 30 mm. A quarter of the
phase-space area vanished, emittance was not conserved, and the halves did not
compose (6.4e-2). The defocusing axis is now `[[cosh, sinh/k], [+k sinh, cosh]]`.

Two things the plan had not caught, both found while writing the fix:
(a) the `B` term was `S/K` with a **signed** `K`, so it went *negative* for
`K < 0` — a drift-like term must not, and this was wrong independently of the
cosh defect; (b) in the thin branch, focusing did not actually depend on
`sign(K)` at all except through the `X,Y` swap, since `-K**2` focuses x for
either sign — so the convention fix had to touch the magnitudes, not just
delete the swap.

Convention (Eric's call): **K > 0 focuses x, defocuses y**, now identical in
both branches. All of it lives in one private `_body_block(dz, axis)` reading
`_axis_focuses(axis)`, consumed by `transfer_matrix`, `transfer_block` and
`focal_powers` — which is what makes `transfer_block` match the matrix
sub-block to **0.0 exactly** across a K/L sweep, rather than by careful
duplication as before.

Numbers: `|det - 1| < 2.4e-14` and `M(L/2)^2 = M(L)` to 3.6e-15 over
K in +-{0.3, 1, 12, 30} x L in {5, 30, 100} mm x both axes; transverse 4x4 det
= 1.000000000000; `propagate_moments` conserves `sqrt(det Sigma)` on both axes
to 1e-12 (ratio 1.000000000000); a short thick quad approaches the thin kick
`K^2 L` (ratio 0.99999976 at L = 0.1 mm). The `_accumulate_blocks`
symplecticity guard stopped firing on quadrupoles **by itself**, as designed —
its test now uses a deliberately non-symplectic stub, since its original
subject is fixed. `basic_column` is unaffected: all three of its quadrupoles
sit at zero strength. 88 tests green (was 85).

Note for the defocusing axis: `|1/f| = k*sinh(kL)` grows without bound in `kL`,
where the old `sin` folded over. That asymmetry is real — at strength 12 over
30 mm the two axes differ (4.2273 vs 4.4139) — and it now shows up in
`focal_powers`, hence in the wave path's saddle screen too.

**Not done:** step 3 (the per-axis scaled segment, `Quadrapole._scaled_segment()`)
and step 4, skew. Skew needs an angle parameter and x-y coupling, so it is a
feature rather than a sign fix — a rotated quadrupole cannot be written as two
independent 2x2 blocks at all.

## 2026-08-23 — [Done] Revert the P1 wave-seam generalization
**Goal:** back out yesterday's `phase_shift(kind=)` / `amplitude_mask` rework and restore the three `propagate_wave` overrides.
**Why:** Eric's architectural correction. Three separate things were wrong with it.
- [x] `phase_shift(kind=...)` -> `phase_shift(..., scaled: bool = False)`; drop `_phase_shift_fixed`/`_phase_shift_scaled`
- [x] delete `amplitude_mask` (base + `Aperture`)
- [x] restore `Source.propagate_wave`, `Aperture.propagate_wave`, `Prism.propagate_wave`
- [x] keep the `waveoptics` rename; fix the anisotropic-aperture bug found while reverting

**Outcome:** Reverted. The three reasons, because they generalize:

1. **`kind` was a fake three-valued enum.** `'hybrid'` *is* the scaled
   representation — the dispatcher literally mapped it to `'scaled'`. A
   parameter whose third value is an alias for its second is a boolean wearing
   a costume. Back to `scaled: bool`. (`propagate_wave`'s `mode` keeps all
   three, correctly: there the three *are* distinct — fixed grid, single scaled
   frame, scaled with frame switching.)

2. **`amplitude_mask` was a single-use abstraction.** The physics behind it is
   real — `phase_shift` declares chi and the propagator applies exp(i*chi),
   which is unimodular, so no real chi can ever produce zero transmission and
   an aperture is genuinely not expressible as a phase. But the population of
   elements needing it is *one*. The repo's own coding guidelines say not to
   build single-use abstractions; a universal seam on every Element to serve
   one class is exactly that. An aperture is a different **operator**, so it
   overrides the propagator and says so.

3. **The "no element owns a propagation method" invariant was overreach.** The
   ray-side analogy holds for *phase* elements — lenses and multipoles declare
   `phase_shift` and the generic propagator consumes it, which is right and
   stays. It does not hold for elements that are not phase operators at all:
   `Source` (seeds the field, it does not act on one), `Aperture` (amplitude),
   `Prism` (unimplemented). Those override, optionally via `super()`.

**Kept:** the `waveoptics` rename (`propagate_thick_lens_scaled` ->
`propagate_quadratic_segment_scaled`, `scaled_delta_tau_lens` ->
`scaled_delta_tau_quadratic`). Note for the record, since it caused confusion:
these are `waveoptics` **primitives**, not element methods — no element owns
them. The element seam is `Lens._scaled_segment()` returning `('quadratic', K)`,
consumed by the generic `Element._propagate_wave_scaled`, which is the same
declare/consume shape as `transfer_matrix` -> `propagate_ray`.

**Bug found and fixed while reverting:** `Aperture.propagate_wave` raised
`TypeError: bad operand type for abs(): 'tuple'` on the scaled path whenever
the frame was anisotropic — i.e. after *any* quadrupole, since a thin quad
leaves `s` stored as a pair even when the axes are numerically equal. It now
masks at physical coordinates `(s_x*xi, s_y*eta)` against the physical radius,
which is identical to `radius/|s|` when the axes agree and correctly an
**ellipse** in scaled coordinates when they do not (an aperture is a circle in
the physical plane, not the scaled one). Sign-safe past a crossover too, since
the pitches are squared. Regression test added. 85 tests green (was 84).

**Open design question Eric raised, not implemented:** whether an element
should *store* its phase rather than compute it per call, so aberrations can be
composed onto it (a round lens's radial term + a stig term, Krivanek/Seidel/
Zernike). My answer to "do we need a separate stored phase for scaled and
fixed": no — there is one physical chi(x, y); the two paths differ only in
*where* it is evaluated (x = xi vs x = s*xi) and *how much* of it is absorbed
into the frame curvature. The natural single declaration is a list of terms
(coefficient + basis), with one rule: quadratic terms -> frame powers
(P_x, P_y), everything else -> residual screen at x = s*xi. That rule
generalizes exactly what `Lens` and `Quadrapole` hardcode today, and it means
aberrations need no new seam at all. Belongs with the D4 aberration work.

## 2026-08-22 — [Done] Wave seam cleanup: no element owns a propagation method
**Goal:** `phase_shift(kind=...)` dispatching to `_phase_shift_*`, an amplitude/mask declaration, removal of the `Source`/`Aperture`/`Prism` `propagate_wave` overrides, and an element-agnostic per-axis segment propagator.
**Why:** Eric's architectural direction — the ray side is the model (an element declares `transfer_matrix`, the generic `propagate_ray` consumes it), and the wave side should match; also `propagate_thick_lens_scaled` is lens-named and lens-scoped, which is what blocks a quadrupole from using it.
- [x] phase_shift(kind=) + _phase_shift_* split
- [x] amplitude_mask seam + remove the three overrides
- [x] segment propagator rename (per-axis deliberately deferred to issue #3 step 3)

**Outcome:** The wave side now matches the ray side: an element **declares**
(`phase_shift`, `amplitude_mask`, `_scaled_segment`) and the generic propagators
on `Element` consume it. `Element.phase_shift(dimensions, wavelength,
kind='fixed'|'scaled'|'hybrid', s=1)` replaces the `scaled=bool` flag and
dispatches to overridable `_phase_shift_fixed`/`_phase_shift_scaled` halves
(Drift/Quadrapole/Dipole/Lens/Prism split accordingly; `'hybrid'` *is* the
scaled representation so it maps there; an unknown kind raises). New
`Element.amplitude_mask(dimensions, kind, s)` is the multiplicative companion
for elements whose wave action is a transmission rather than a phase —
`Aperture` implements it and maps its own physical radius to `xi <=
radius/|s_x|` on scaled frames, so the driver never owns that coordinate
mapping. With those two declarations in place, `Source.propagate_wave`,
`Aperture.propagate_wave` and `Prism.propagate_wave` are **deleted**: both
drivers now apply `amplitude_mask` and pass `kind=` through, reproducing the
deleted behavior generically (`Source` and `Aperture` are simply transparent on
the phase seam — a seed and a mask are not phases; `Prism` raises, its wave
physics being genuinely unimplemented rather than architecturally special).
`waveoptics.propagate_thick_lens_scaled` -> `propagate_quadratic_segment_scaled`
and `scaled_delta_tau_lens` -> `scaled_delta_tau_quadratic`: these describe a
constant-K quadratic-index *segment*, and the lens-scoped name was what blocked
a quadrupole from reaching them. **Per-axis capability deliberately deferred**
to issue #3 step 3 rather than built here: an anisotropic segment needs a
per-axis K *and* a separable (dtau_x, dtau_y) kernel, and its only consumer is
the thick quadrupole whose transfer block is not yet symplectic — the rename is
what actually unblocks #3, and the isotropic-frame refusal already names the
limit actionably. 86 tests green (was 84); the two pre-existing regression tests
that pin this seam (`test_fixed_path_refactor_regression`,
`test_wave_kind_aperture_matches__aperture_wave`) pass unchanged, which is the
pure-refactor proof. Scope note: this cleans the **wave** seam only —
`Source`/`Aperture` still own `propagate_ray`/`propagate_moments`, since a beam
seed and a hard geometric block are not transfer matrices; the guard test says
so explicitly rather than silently allowing it.

## 2026-08-22 — [Note] Thick quadrupole defects filed as issue #3
**Goal:** Get the thick-quadrupole matrix defects in front of the other contributor rather than fixing quadrupole ray physics unilaterally, and record the wave-side work they block.
**Why:** Found while validating the analytic plane calculation: the thick `Quadrapole.transfer_matrix` y-block is non-symplectic (`det = cos(2|KL|)` = 0.7518 over a 30 mm body, ~25% of phase-space area lost, violating Liouville — the defocusing axis needs cosh/sinh), and the thin branch swaps X/Y for K > 0 while the thick branch never swaps, so thin and thick quadrupoles disagree about which axis focuses.
**Issue:** [sea-ecosystem/rayTEM#3](https://github.com/sea-ecosystem/rayTEM/issues/3) — reproducer included, plan inlined so it is self-contained.
**Plan:** [PLAN_2026-08-22_thick-quadrupole-symplecticity.md](PLAN_2026-08-22_thick-quadrupole-symplecticity.md) · **TODO:** [TODO_ACTIVE_thick-quadrupole-symplecticity.md](TODO_ACTIVE_thick-quadrupole-symplecticity.md)
**Status:** mitigation only — no quadrupole physics changed. `Element.transfer_block` mirrors `transfer_matrix` exactly (defect included) so plane finding never silently diverges from ray tracing, and the walk guards `det == 1` on any body, refusing that axis with an actionable error. `basic_column` is unaffected (its quads are thin). Blocks the thick-quad wave-optics fix: the segment propagator is lens-scoped and refuses anisotropic frames, and there is no correct per-axis body law to mirror until the matrix is fixed — so a thick quad in the wave path still falls back to drift L/2 -> kick -> drift L/2. The plan also records Eric's architectural point: `Source`/`Aperture`/`Prism` still override `propagate_wave`, which they should not need to once there is an amplitude/mask declaration alongside `phase_shift`.

## 2026-08-22 — [Done] One plane calculus (wave image planes, covariance waists, reference planes)
**Goal:** Promote the analytic plane calculation into the framework as one per-element walk of the accumulated 2x2, and use it for all three modes: wave image planes (conjugate seed), covariance waists (`Sigma_12 = 0`), and planes conjugate to a *named* reference element rather than the column entrance.
**Why:** Eric's follow-ups. The wave currently only reports the diffraction family; the covariance mode reports no planes at all; and "the planes" are always measured from the entrance, when what you usually want is "conjugate to the sample" or "conjugate to the condenser aperture" — genuinely different sets.
- [x] transfer_block seam + the walk
- [x] wave image planes, reference=, covariance waists
- [x] tests, aberrated-surface plan, docs

**Outcome:** All four of Eric's asks. Items 1-3 turned out to share one mechanism — a single per-element walk of the accumulated 2x2 (`Microscope._accumulate_blocks`), which is simultaneously the ray matrices and the scaled wave frame's own arithmetic, since the frame IS a reference ray `(h, u) = (s, s/R)`. New seam `Element.transfer_block(dz, axis)` gives the rotating-frame block at a *partial* length (base = thin kick + free space; `Lens` overrides with cos/sin), verified against `transfer_matrix` on all 58 basic_column elements to 1.7e-18 with halves composing exactly. **(1) Wave image planes:** `conjugate_planes(method='frame')` is now the default and returns both families; the wave's crossovers match its `diff` family to 5.6e-10 um, and the previously invisible `image` family is now available without a second wave run (`wavefield_at` reconstructs the field there). Frame and ray agree exactly in free space; for the one image plane inside OL1's body the frame is right and ray interpolation is 188 um off. **(3) Reference planes:** `conjugate_planes(reference='sample'|'C1'|z)` accumulates from that plane, returning absolute z and offsets. Physically instructive result encoded in a test: moving the reference across a *pure drift* leaves the **diffraction** family unchanged (parallel stays parallel) but moves the **image** family — so "may or may not be the same" is answered by picking the right reference AND the right family. **(2) Covariance:** `beam_waists(axis, sigma0)` finds `Sigma_12 = 0`, the finite-width minimum, matching the analytic focal shift `f/(1 + (f sigma_theta/sigma_x)^2)` to 1e-12 and reporting the invariant emittance. Note `Sigma_12` is *quadratic* in the element matrix (unlike the linear plane condition), so thick bodies need their own closed form `tan(2K dz) = -2K Sigma_12/(Sigma_22 - K^2 Sigma_11)` — I initially reused the plane solver here, which was wrong; the corrected version finds real waists inside the C1, OL1 and PL3 bodies that the linear form missed. Two walk bugs found by the toy column and fixed: zero-length elements exactly at the reference were skipped (so a thin lens at the reference never entered the accumulation), and a reference falling inside a body advanced by a plain drift instead of that body's own partial block. **Follow-up (Eric's review):** `method='ray'` is now the **default** (`Literal['ray','frame']`); `reference=` still needs `method='frame'` explicitly, since the ray trace only launches from the column entrance. Eric also objected to the kick in `Element.transfer_block`'s base implementation — correctly: it was applying kick+drift inside *any* finite body without an override, which is exactly the approximation the scaled path was just corrected to drop, and it silently affected **thick quadrupoles**. The base now handles only the two cases needing no body law (zero power -> exact free space; zero length -> a thin element IS a kick) and **raises** for a finite body with power and no override; `Quadrapole` gained its own harmonic `transfer_block` mirroring `transfer_matrix`. **Finding for the other contributor (not fixed here):** the thick `Quadrapole.transfer_matrix` y-block is **non-symplectic** — `[[C, S/K], [+K S, C]]` gives `det = C^2 - S^2 = cos(2|KL|)`, e.g. 0.7518 over a 30 mm body, so it loses ~25% of phase-space area in violation of Liouville; the defocusing axis should be `cosh/sinh`. The x (focusing) axis is fine (`det = 1`). Related: the *thin* quad swaps X/Y for `K > 0` while the *thick* quad never swaps, so thin and thick quadrupoles disagree about which axis focuses. Rather than silently changing quad physics I added a symplecticity guard to the walk, which refuses that axis with an actionable error naming the det — so no untrustworthy planes are ever reported. 84 tests green.
**(4)** Plan written for aberrated focal surfaces: [PLAN_2026-08-22_aberrated-focal-surfaces.md](PLAN_2026-08-22_aberrated-focal-surfaces.md) — least-confusion criterion degenerating to the paraxial root, a `z(r, phi)` fit with a sag report, and the key asymmetry that the wave frame should *stay* paraxial because the residual non-quadratic phase on U at the logged diffraction plane already IS the aberration function `chi(q)`. 82 tests green (was 78).
## 2026-08-21 — [Done] Thick lens as a scaled segment (+ wave rotation)
**Goal:** Treat a thick lens as the quadratic-index medium it is — a segment with a sinusoidal s(z) law and its own closed-form dtau — keeping the thin-lens (L == 0) path exactly as it is, and add the Larmor rotation to the wave.
**Why:** The drift L/2 -> thin kick -> drift L/2 split misplaces every crossover by the per-lens amount measured in `examples/05_planeComparison.py` (C1: 422.3 um, matching the prediction to the last digit; up to 4.8 mm downstream). The scaled factorization solves a quadratic-index medium exactly, so no approximation is needed here.
- [x] scaled_delta_tau_lens + propagate_thick_lens_scaled
- [x] element seam (L == 0 thin / L > 0 segment) + tests
- [x] wave rotation + docs

**Outcome:** A thick round lens is now carried **exactly** by the scaled path. It is a quadratic-index medium, not a screen, so it is one segment whose scale law is sinusoidal: the frame advances by the element's own `[[cos, sin/K], [-K sin, cos]]` block applied to `(s, s/R)` (legitimate because the frame IS a reference ray) and U propagates over the segment's own `Delta-tau = [tan(KL - phi) + tan(phi)]/(K C^2)` with `C^2 = s0^2 + (u0/K)^2`, `tan phi = u0/(K s0)` — no phase screen, no curvature kick, no sampling cost. New: `waveoptics.scaled_delta_tau_lens` (verified vs the numerical integral of dz/s^2 to 7.8e-13; reduces to `scaled_delta_tau` as K -> 0; raises actionably when `s -> 0` inside the body) and `waveoptics.propagate_thick_lens_scaled` (reproduces `Lens.transfer_matrix`'s rotating-frame x-block exactly, energy to 1e-9, refuses anisotropic frames rather than silently downgrading). The seam is a new `Element._scaled_segment()` hook returning `None` on the base class and `('quadratic', K)` from `Lens` only when `length > 0` and `K != 0`, so **`length == 0` keeps the thin-lens kick path bit-for-bit and `length > 0` takes the exact segment** — both kept, as Eric asked. **Payoff:** on basic_column the hybrid crossovers now match the ray-traced and analytic diffraction planes to **0.0 um at all five planes**, where they previously sat 422-4808 um away; `examples/05_planeComparison.py` shows this and its `d_wave` column is now all zeros. Also added `waveoptics.rotate_field` — the exact unitary three-shear Fourier rotation `R(t) = Sx(tan t/2) Sy(-sin t) Sx(tan t/2)` (centroid error 1e-14 px, energy 1.000000000000, round trip 5.6e-16), which **commutes** with the isotropic kernel and reference phase (1e-10 inside the drift sampling limit), so a lens body's whole Larmor angle can be applied once at its exit exactly. Exposed as `propagate_wave(..., rotate=False)` threaded Element -> Section -> Microscope; off by default because the ray path's `-K*L` rotation is analytically a no-op for rotationally symmetric fields and would only add resampling noise there, but a test confirms the wave picks up exactly the ray's azimuth when enabled. Remaining known gap (documented, not implemented): a crossover landing *inside* a lens body raises, because the hybrid engine only switches frames in free segments — the image plane at 493.60771 mm inside OL1 is a live example, harmless today only because the frame follows the diffraction family. 78 tests green (was 72).

## 2026-08-21 — [Note] Note handed over: analytic crossover planes (A = 0, B = 0)
**Goal:** A self-contained note for the ray-side owner on locating both plane families analytically from the accumulated transfer matrix — `A = 0` for diffraction (back-focal) planes, `B = 0` for image planes — instead of searching a traced ray bundle.
**Why:** Confirmed `findPlanes` finds planes by interpolating where two reference rays' *difference* is zero between logged planes. That is algebraically the same criterion (the +- pair's difference IS `2a*A` / `2b*B`), but sampled — and linear interpolation is the wrong functional form inside a thick lens body, which is where our wave-vs-ray offsets show up on basic_column (0.4-4.8 mm, vs 0 nm on a thin-lens column).
**Plan:** [PLAN_2026-08-21_matrix-conjugate-planes.md](PLAN_2026-08-21_matrix-conjugate-planes.md) — notation, the criterion, free-space roots (`-A0/C0`, `-B0/D0`), thick-lens roots (`tan`/`tanh(K dz) = -K A0/C0`), free magnification, the `1/f_sys = -C` / `BFD = -A/C` sanity identities tying it to the textbook doublet formula, verification cases, and two incidental bug findings (`focus_error`'s `findPlanes(self.rays,"x")` positional-argument slip; `whereCrossesZero` used only by `findPlanes4`). Deliberately scoped to the ray side only — the wave side has its own equivalent (`dz = -R`, see below) and does not depend on this.
**Validation:** `examples/05_planeComparison.py` (a marimo notebook) runs all three methods (analytic matrix, ray `findPlanes`, wave frame) on basic_column trimmed past PL4, prints a plane-by-plane offset table, and plots the dense physical cross-section with the four reference rays overlaid and every method's planes marked. Analytic vs `findPlanes`: **0.0 um on 9 of 10 planes**; the single 188.1 um disagreement (image plane, 493.60771 mm) falls **inside OL1's body** (490-500 mm) — exactly the predicted thick-lens interpolation failure, and the script flags such planes automatically. Table and figure are in the note.
**Wave-side offsets explained (follow-up for us, not for the note):** the hybrid crossovers sit 0.4-4.8 mm from the ray/analytic diffraction planes purely because the scaled path treats a thick element as *drift L/2 -> thin kick P -> drift L/2*. The correct treatment is not a better phase screen: a thick lens is a quadratic-index **medium**, i.e. a *segment* whose `s(z)` law is sinusoidal (`s = s0 cos(K dz) + (u0/K) sin(K dz)`) rather than linear, and the scaled factorization solves that exactly -- U needs no screen and no kick, it just propagates over the segment's own Delta-tau. Writing `s(z) = C cos(K dz - phi)` with `C^2 = s0^2 + (u0/K)^2` and `tan phi = u0/(K s0)`, the closed form is `Delta-tau = [tan(KL - phi) + tan(phi)]/(K C^2)`. That integral diverges exactly where `s -> 0` inside the body — a crossover inside the lens — which is the same singularity we already flatten through in free space, except the hybrid engine currently only switches frames in free segments and would need to switch mid-element. Not hypothetical: the image plane at 493.60771 mm sits inside OL1; it does not bite us today only because the frame follows the diffraction family and all five of those planes are in free space on this column. For a collimated ray the exact crossover from the lens exit is `cos(KL)/(K sin KL)`, the thin-equivalent gives `1/(K sin KL) - L/2`, and the difference reproduces the measurement exactly (C1: predicted 422.3 um, measured 422.3 um; later crossovers drift further as the position and angle error is magnified downstream). The fix is to advance the frame through a thick element with the element's own `cos/sin` matrix -- legitimate because the frame *is* a ray `(h, u) = (s, s/R)` -- instead of splitting it around a thin kick; the tau integral through the body has a closed form too. Not implemented yet.
**Note on the wave side (for cross-reference):** the scaled frame's `(s, R)` is the same object in disguise — `s` is a reference ray's height and `s/R` its angle, so `s(z) = A(z) s0 + B(z) s0/R0` and the frame's collapse `s = 0` sits at `dz = -R`. A flat seed makes `s ∝ A` (diffraction family, what `crossovers` holds); a point seed makes `s ∝ B` (image family). `R` composes as a Mobius transform `R' = (A R + B)/(C R + D)`, which is why the recursion handles compound systems with no combined-focal-length formula.

## 2026-08-21 — [Done] Conjugate planes: image AND diffraction families
**Goal:** `Microscope.conjugate_planes(axis)` returning both the image and diffraction (back-focal) plane positions in metres, reusing the ray side's `findPlanes`, and annotating both in the scaled cross-section.
**Why:** Eric asked which family the hybrid engine finds; measured answer is only the diffraction family (the frame is seeded flat = a parallel wavefront = findPlanes' diffraction ray), so image planes are currently invisible even though they are reconstructable.
- [x] conjugate_planes via findPlanes on a copy
- [x] cross-section annotation + docs + tests

**Outcome:** Answered Eric's two questions with measurements, then closed the gap. (1) **Compound systems are handled correctly** — the engine never uses a lens's f; the frame's curvature R accumulates the whole upstream system (`R -> R + dz` through drifts, `1/R -> 1/R - P` at lenses, which is exactly the geometric distance-to-axis-crossing of the reference ray), so `z_cross = z + |R|` is the *beam's* focus. Verified on a thin two-lens column (collimated in, f1=45 mm, 100 mm gap, f2=30 mm): wave crossovers = [55.0000, 176.0000] mm = the analytic compound values to **0 nm**, and NOT `z_L2 + f2` = 140 mm — the second crossover is the image of the first. (2) **We were logging only ONE family.** The frame is seeded `s=1, R=inf` (a parallel wavefront), which is precisely findPlanes' *diffraction ray*, so `crossovers` are the diffraction / back-focal planes; the image plane in that test column (150.8621 mm) was invisible. New `Microscope.conjugate_planes(axis='x')` returns BOTH families in metres by tracing the four reference rays of `postprocessing.findPlanes` on a `deepcopy` (self.rays never clobbered) and converting its fractional plane indices with `zFromFractional` — reusing the repo-wide diff/image convention rather than reimplementing conjugate arithmetic. `show(kind='wave-*')` now annotates image planes (magenta dash-dot) beside the crossovers (cyan dotted) via `conjugates=True`, and image planes can be logged *exactly* by composing with the sampling knob: `zpts=scope.conjugate_planes()['image']`. On basic_column: diff [0.17458, 0.30519, 0.50249, 0.72929, 0.91715] vs image [0.19828, 0.49380, 0.53089, 0.73033, 0.91963] m. NOTE for Ondrej: the wave crossovers sit 0.4–4.8 mm off the ray diff planes on basic_column because every lens there is a **thick** QLens (10–20 mm) and the scaled path treats thick elements as thin-equivalent between half-length drifts (a documented approximation) while the ray path uses the full thick-lens matrix — the thin-lens test agrees to 0 nm, so this offset is that approximation, not a bug in the frame arithmetic. Worth revisiting if thick-lens wave accuracy matters. 72 tests green.

## 2026-08-21 — [Done] Dense z sampling for show (subdivided + zpts)
**Goal:** `Microscope.subdivided(zpts)` (copy with unnamed drifts split by max-spacing dz or at explicit z) and `show(..., zpts=)` that plots the scaled cross-section from a temporary subdivided copy, leaving the original's state untouched.
**Why:** The cross-section's z resolution follows the column's logged planes (predictable but chunky); Eric wants seamless propagation and arbitrary-z plotting through show without hand-editing the column.
- [x] subdivided(zpts) helper
- [x] show zpts wiring + tests + wiki

**Outcome:** `Microscope.subdivided(zpts)` returns a NEW column whose unnamed drifts are cut — `float` = max drift length (metres), `Sequence` = absolute z positions to cut at — preserving element order, lengths, section positions and every named position exactly (cut drifts sum to the original length; the copy restacks sequentially via `MicroscopeSection.__init__`, which is why `_position=None` is set on carried-over elements). The original object and any result on it are untouched, so this is a pure "denser sampling" knob rather than a mutate-and-restore. Named drifts are left whole (a name marks a plane someone asked for). `show(kind='wave-scaled'/'wave-hybrid', zpts=...)` propagates such a copy on the spot and plots from it; a float `plane=` joins the cut set so that plane is logged exactly instead of snapping to the nearest existing one; `zpts` on any other kind raises with a pointer to `subdivided`. Verified on basic_column: `show(kind='wave-hybrid', zpts=5e-3)` gives a smooth envelope (~15 s, vs the chunky element-exit sampling) with all five crossovers still logged exactly. Also documented (Eric asked how crossovers are found — they are NOT a numerical search): in a converging frame `R(z)=R₀+Δz`, so the focus is the closed-form `z_cross = z + |R|`; the hybrid engine records it at the flatten, splits propagation exactly there, tags the plane `crossover`, and `propagate_wave` collects them into `self.crossovers` — hence `show(plane=scope.crossovers[i])` always lands exactly on the i-th focal plane regardless of drift subdivision. Small cosmetic fix: the cross-section no longer draws a line for `named_positions`' blank key (all unnamed elements collapse into it). 70 tests green.

## 2026-08-21 — [Done] Microscope.show for scaled/hybrid wave results
**Goal:** Wire `show(kind="wave-scaled"/"wave-hybrid")`: cross-section |ψ(x,0,z)| with element/crossover annotations by default, per-plane |ψ|² via the reconstructed Signal's own `.show()` when a plane is named.
**Why:** The kinds were accepted by the signature but raised; the demo's figures lived only in examples/04 — Eric wants the scaled result to plot through the standard `show`.
- [x] cross-section + per-plane show paths
- [x] headless test + wiki

**Outcome:** `Microscope.show(kind="wave-scaled"/"wave-hybrid")` works. No `plane` → the |ψ(x, y=0, z)| cross-section (new module helper `_scaled_wave_cross_section`): each logged plane reconstructed on its native `Δx=|s|Δξ` grid, peak-normalized centre rows resampled onto one x axis, element positions (white dashed + labels) and crossovers (cyan dotted) overlaid — the wave analog of the ray diagram. `plane=` (int index into `_wave_scaled_planes`, float z in metres via `wavefield_at`'s nearest-plane pick, or a named position string like "sample") → the reconstructed physical |ψ|² imaged by delegating to the wavefield Signal's own `.show()` — same composition pattern as `kind="wave"`/`"moments"` (Microscope does not inherit from Signal; it builds a calibrated Signal and calls its show). `plane` default changed −1 → None (None still means last plane for wave/moments, so old calls behave identically). Notes: cross-section z resolution follows the logged planes — a column with subdivided drifts plots smoother (examples/04 does this); planes are peak-normalized (shape, not absolute intensity). 68 tests green (headless show test: pcolormesh drawn for the cross-section, Signal imshow for plane by z and by index).

## 2026-08-21 — [Done] Radial absorbing boundary (fourfold fringe fix)
**Goal:** Replace the square separable absorbing-boundary window with a radially symmetric one so the absorber stops imprinting a fourfold, pixel-aligned fringe pattern on the beam.
**Why:** Eric spotted a fourfold pattern inside the disc at every downstream plane (band limit on or off); discrimination experiments (c4 fourfold harmonic, suspects toggled in-memory) pinned it to the square window's azimuthally anisotropic clipping of the aperture halo — radial window collapses c4 from ~1e-3 to 0.0000 at sample and detector, while the band-limit and beam-support-policy toggles change nothing.
- [x] radial boundary_window
- [x] tests (radial symmetry + c4 regression) + suite green
- [x] figures to Eric + docs/wiki + protocol finish

**Outcome:** `waveoptics.boundary_window` is now a radially symmetric raised cosine (1 inside the inscribed circle minus the band, 0 at the inscribed-circle edge; corners fully absorbed). The old separable `outer(ramp, ramp)` window was a soft SQUARE frame applied at every tau sub-step: its corners sit sqrt(2) farther than its edges, so the aperture's Fresnel halo was clipped azimuthally anisotropically and the fourfold-modulated survivor interfered back into the disc — the pixel-aligned pattern Eric flagged ("fringes overlapping" was exactly right, but overlapping through the absorber frame, not at a crossover). Discrimination table (c4 fourfold harmonic of interior intensity at sample/detector): square window 0.00095/0.00145; binary-vs-bandlimited aperture toggle no change; grid-edge-vs-beam-support flatten toggle no change; radial window 0.00004/0.00005; padded 512^2/40um halves c4 (frame farther from beam — Eric's FOV intuition). Residual interior contrast is isotropic concentric Fresnel rings only (sample std/mean 0.012, detector 0.023 — thresholds re-measured; the circular truncation starts slightly earlier than the square window's corners, so ring contrast is marginally higher but has no angular structure). bandlimited_disk kept per Eric (exact alias-free sampling; cleans the z=120mm near-field, no downstream effect). Tests 67 green: window radial-symmetry unit test + c4 < 5e-4 regression at both planes.

## 2026-08-21 — [Done] Alias-free aperture + frame-policy refinements + anisotropic frames
**Goal:** Remove the numerical "gridding" from hard-aperture runs while keeping the real Fresnel fringes (band-limited sampling of the exact Theta(a-r)); make the frame policy beam-support based; add direct frame jumps and anisotropic (s_x != s_y) frames.
**Why:** Eric spotted an axis-aligned grid texture at the sample/detector (diagnosed: aliased above-Nyquist edge content, not wraparound; physics must stay); the padded-grid control exposed the grid-edge flatten criterion crashing into s_min; the two handoff follow-ups (jumps, anisotropy) were queued next.
- [x] alias-free aperture sampling (bandlimited_disk + antialiased masks)
- [x] beam-support guard + flatten thresholds (padded-grid regression)
- [x] crossover='jump' policy + measured default
- [x] anisotropic frames (quads absorb into R; line-focus crossovers)
- [x] docs/wiki + protocol finish

**Outcome:** (1) The "gridding" had two mechanisms, both fixed physically with the model unchanged (Theta(a-r) stays sharp): the initial sampling now holds the band-limited projection of the exact disk (`waveoptics.bandlimited_disk`, analytic J1 spectrum — every representable Fresnel fringe exact, nothing folds; mid-column `aperture_mask` gets a 1-px area-coverage edge, `antialias=False` restores binary), and free segments carry an absorbing boundary (`boundary_window` + `absorb=0.1` tau sub-stepping — periodic-FFT wraparound of the edge halo removed; those electrons physically leave the beam). Sample-plane interior modulation fell 0.0397 -> 0.0017 with ~1.3%% energy honestly absorbed. (2) Frame policy is beam-support based: guard and flatten/re-diverge thresholds measure the reference phase at the beam's per-axis support half-width (`beam_support_radius`/`beam_support_extents`), not the empty grid edge; the hybrid engine owns its internal guard (closed-form splits), so the padded 512^2/40 um column completes with defaults (regression test). (3) `crossover='jump'` implemented (mirror R=-d -> +d at half the flatten threshold, one switch, no flat window) and measured: optical through-focus matches flat (1.3e-2 vs 1.0e-2), but the doubled phase budget rides 2x deeper and at tight electron crossovers the focal structure (Airy/s in xi) outruns the FOV — basic_column loses ~95%% of the beam — so 'flat' stays the default and the guidance is documented. (4) Anisotropic frames: psi = (s_x s_y)^(-1/2) U(x/s_x, y/s_y) e^(ik(x^2/2R_x + y^2/2R_y)); frame quantities travel scalar-or-pair (`axis_components`/`join_axes`), `Quadrapole.phase_shift(scaled=True)` returns ((P_x, P_y), None) absorbed into (R_x, R_y) like a round lens (no saddle screen, no sampling limit — strong stigmators run), the hybrid engine runs a per-axis event loop with line-focus tags (flatten-x/crossover-y/...) behind an isotropic fast path (round-lens columns bit-for-bit), the seam stores per-axis metadata (s_x/s_y, R_x_m/R_y_m, tau_x/tau_y, z_cross_x_m/z_cross_y_m) and per-axis SignalSet companions only when the axes differ (old files load unchanged), and reconstruction uses rectangular native pixels. Astigmatic Gaussian matches the analytic q-parameter widths at both line foci and the exit to 1e-3; Microscope.crossovers lists per-axis line foci at the predicted per-axis focal lengths. 66 tests green (was 54).

## 2026-08-20 — [Done] Scaled-frame switching through crossovers
**Goal:** Full-column scaled propagation: a general frame-change primitive (Eric's Eq 5) plus a hybrid crossover policy (scaled -> flatten near focus -> ordinary Fresnel through it -> re-diverge), consolidated as propagate_wave(mode='fixed'|'scaled'|'hybrid').
**Why:** The scaled run stops at C1's crossover (the frame's s->0 singularity), and the focal/back-focal/image planes — which sit AT the crossovers — are the most important planes.
- [x] change_scaled_frame + min_representable_curvature + delegation (tests)
- [x] hybrid engine + z_cross metadata (through-focus + Airy tests)
- [x] API consolidation propagate_wave(mode=...) / Source.wave(mode=...)
- [x] driver wiring (log, frame companion, Microscope.crossovers) + column test
- [x] demo + docs/wiki + close issue #2

**Outcome:** Full-column scaled propagation works: basic_column runs source -> detector (z = 1.264 m) through five crossovers with energy conserved at all planes and the physical pixel spanning 0.1 nm (foci) to 6.3 um (detector) on one xi-grid calibration. Built per Eric's handoff: waveoptics.change_scaled_frame (Eq 5, pointwise under the physical-grid-continuous convention, guarded by min_representable_curvature; factor_wave delegates as the (1, inf) special case) and propagate_free_scaled_hybrid (converging frame flattens at |R_flat| = R^2/(A s^2), a frame invariant with closed-form splits; ordinary carrier-free Fresnel through the real focus with the crossover/back-focal plane split out and logged; re-diverge at d = A s^2; predictive switching, s_min only a backstop). API consolidated per Eric: one propagate_wave(..., mode='fixed'|'scaled'|'hybrid') on Element/Section/Microscope (propagate_wave_scaled removed), Source.wave(mode=...), dispatcher kinds wave/wave-scaled/wave-hybrid via (method, forced-kwargs) mapping. The .wave_scaled SignalSet keeps its single shared xi/eta calibration (all switches are pointwise, s continuous) and gains a 'frame' companion + per-plane tags; z_cross rides the in-flight Signal metadata; Microscope.crossovers lists the focal planes and wavefield_at reconstructs them (electron-scale focal plane = Airy: first zero < 5e-3 of peak, 83.8% encircled energy; through-focus equivalence vs the ordinary propagator 6.3e-3 at safety=0.5 — error grows with flat-window length, confirming the default). 54 tests green; demo shows physical + scaled-coordinate cross-sections and x-y slices at source/C1-back-focal/sample/objective-focus/projector-focus/detector; issue #2 closed. Note: "chart" wording replaced by "frame" throughout per Eric.

## 2026-08-20 — [Done] Unify wave-mode naming on "wave"
**Goal:** The wave signal is the object everywhere; rename the Source "field" API and driver kwargs to wave terminology (wave/wave_scaled/wave_shape/wave_kind/wave0).
**Why:** Eric's review: "field" was an undefined synonym for the wave signal, inconsistent with rays()/moments() and the .wave/.wave_scaled containers.
- [x] Source API rename (field->wave family)
- [x] driver field0->wave0 + regenerate basic_column.sea
- [x] tests/examples/wiki sync

**Outcome:** The seeding API now matches the mode names one-to-one: Source.rays()/moments()/wave()/wave_scaled(); the aperture builder became the private _aperture_wave() behind wave_kind='aperture'; ctor/attrs are wave_shape/wave_extent/wave_kind (+aperture_radius); the Section/Microscope driver kwarg is wave0. basic_column.sea regenerated (the stored Source attribute names changed — NOTE: Sources in .sea files saved before this rename fall back to default wave grid/kind on reload since safeReinstantiate filters unknown kwargs; regenerate old files if their wave parameters matter). 48 tests green, demo unchanged.

## 2026-08-20 — [Done] Transparent Element defaults + aperture field_kind
**Goal:** Every propagation kind works as identity on the root Element class, and the aperture initial wave becomes a `field_kind` instead of a separate seeding path.
**Why:** Eric's review: propagation defaults must be shared root-class behavior (identity matrix / phase of nothing), not subclass-only; and the source should hold one wavefunction generator with kinds, not parallel seeding functions.
- [x] identity transfer_matrix + transparent phase_shift on Element
- [x] field_kind='aperture' + aperture_radius; scaled_field simplification
- [x] tests + example + wiki sync

**Outcome:** The root Element now carries a working default for every propagation kind: transfer_matrix returns the identity (no longer @abstractmethod) and phase_shift returns the transparent program (full-length free kernel on the fixed path, (0.0, None) on the scaled path), with length=0 seeded in Element.__init__ — a bare Element propagates as identity through ray/moments/wave/wave-scaled (new test). Source now has one wavefunction generator: field_kind gained 'aperture' (flat-intensity Theta(a-r) at the new aperture_radius attribute, delegating to aperture_field); scaled_field() lost its separate aperture_radius kwarg and always seeds from field(). Note: the scaled run still stops at the C1 beam crossover — that is the chart singularity s->0 of the (s, R, tau) reference frame (issue #2), not missing methods; every element in the column has all four propagation functions. 48 tests green; demo output unchanged.

## 2026-08-20 — [Done] Stale ray-doc fixes + scaled-wave basic_column demo
**Goal:** Correct the wiki's pre-refactor 8-column ray descriptions and demonstrate `propagate_wave_scaled` end-to-end on the basic_column template (saved result + cross-section and x-y slice plots).
**Why:** elements.md/assemblies.md still document I/R as ray columns and 8x8 matrices (misleading after the 6-col refactor), and the new scaled mode has no worked column example.
- [x] wiki ray-doc fixes (+ ecosystem index if env allows)
- [x] scaled-wave demo: propagate basic_column, save, cross-section + x-y slices

**Outcome:** elements.md/assemblies.md now describe the 6-col geometric convention (separate .I/.R, 6x6 matrices, apply_intensity/apply_rotation flow); AGENTS.md re-synced from CLAUDE.md (it still carried the old 8-col invariants) and the invariants list all four propagation modes. Root cause of the ecosystem-index refresh error was a discover_wiki bug (MultiplexedPath join lands on the first pySEA namespace root) — fixed in sea-ecosystem (branch claude/raytem-beam-propagation-3mi4up); pysea-refresh-wiki now regenerates the ecosystem index from sibling envs. New examples/04_scaledWave_basic_column.py: loads basic_column.sea, threads Source.scaled_field() element-by-element via propagate_wave_scaled (drifts subdivided adaptively, finer as the chart converges), saves the U + s/R/tau SignalSet to basic_column_scaled_wave.sea, and renders the |psi(x,0,z)| cross-section (wave analog of plot2D) plus x-y |psi|^2 slices on the zooming physical grid (dx: 78 nm -> 1.6 nm at s=0.02). Propagation stops just before the C1 crossover at z~174.9 mm with the guard's actionable error demonstrated — continuing through crossovers is issue #2. Energy conserved to 1e-6 at every logged plane.

**Addendum (aperture seed):** per Eric, the demo now seeds the flat-intensity hard-aperture wavefunction Theta(a-r) via the new Source.scaled_field(aperture_radius=5e-6) instead of the Gaussian field_kind default. Verified first that basic_column.sea loads the full column (G source, C1/C2/C3, OL1/OL2+sample, PL1-PL4, detector at 1.264 m) — the earlier truncated plots were the C1-crossover stop (issue #2), not a wrong scope. The aperture run shows Fresnel edge fringes developing along the drift and pre-focal diffraction rings approaching the crossover; energy still conserved to 1e-6.

## 2026-08-19 — [Done] Scaled Fresnel propagation (propagate_wave_scaled)
**Goal:** Implement Eric's scaled-Fresnel handoff — factor ψ = (1/s)·U(ξ,η,τ)·exp[ikr²/2R] so the grid rides the beam (Δx = |s|Δξ), with a per-element `phase_shift` contract shared by the fixed and scaled wave paths and reconstruction back to physical x,y at any plane.
**Why:** The fixed-grid wave mode cannot cross a real column (lens phases 10–100× over grid Nyquist; ~10⁶ transverse-scale range — see docs/wave-optics-sampling.md); this removes the reference curvature and scale from the sampled array analytically.
- [x] phase_shift contract + fixed-path refactor (regression-safe)
- [x] scaled Signal seam + factor/reconstruct identity
- [x] scaled free propagation (constant then linear s; Δτ closed form verified)
- [x] scaled element consumption (lens→R; quad/dipole→phase on U, guarded)
- [x] target-grid Fourier reconstruction; entrance-plane equivalence
- [x] drivers + .wave_scaled SignalSet + wavefield_at + dispatcher kind
- [x] docs/wiki; crossover chart-switching follow-up issue
Note: PLAN_2026-08-19_scaled-fresnel-wave.md was rewritten — the earlier pilot-Gaussian/ABCD draft it held was rejected in favor of the handoff's pure phase-factorization formulation.

**Outcome:** `propagate_wave_scaled` on Element/Section/Microscope (dispatcher `kind="wave-scaled"`), built on a per-element `phase_shift(dimensions, wavelength, scaled=False, s=1)` contract that both wave paths share: `scaled=False` returns the fixed path's space-tagged phase program (Lens −k r²/2f, Quadrapole saddle, Dipole tilt, Drift reciprocal kernel; the fixed `propagate_wave` was refactored onto it, regression-proven at atol=1e-12); `scaled=True` returns the (power-absorbed-into-R, screen-applied-to-U) split (Eqs 45–48) with a per-pixel |Δχ|<π guard on U screens. Scaled math lives in `waveoptics` (`scaled_delta_tau` — Eq 29 verified vs the numeric integral, `factor_wave`/`reconstruct_physical_wave` — exact round-trip, `fourier_resample` — exact separable trig interpolation at any pitch ratio, `apply_thin_lens_scaled`, `propagate_free_scaled` with the s_min crossover guard). State rides sea_eco: Δξ/Δη on the ξ/η Dimensions, s/R/τ in metadata in flight and as companion Signals in the `.wave_scaled` SignalSet (`seashells.make_scaled_wavefield_signal`/`read_scaled_wavefield`/`make_scaled_wave_signalset`); `Source.aperture_field(radius)`/`scaled_field()` seed it and `Microscope.wavefield_at(z_or_name, target_dx=, target_shape=)` reconstructs the physical wave at any logged plane (native Eq 41 grid or Eq 44 target grid) for external consumers. Validation: 20 new tests (46 total green) including free-prop equivalence vs the ordinary propagator (flat + curved charts), thin-lens R-absorption vs `focal_phase`, an aperture→free→lens→free system, Eq 54 normalization, Δx=|s|Δξ grid scaling, the electron-scale 200 kV / 20 µm / f=45 mm case the fixed grid cannot sample, and the actionable crossover error. docs/wave-optics-sampling.md updated (scaled Fresnel → implemented); crossover chart-switching filed as issue #2 (refs #1). Known limits (documented): crossovers need a chart switch (issue #2); thick elements are thin-equivalent between half-length free segments; quadrupoles valid at stigmator-scale strengths under the guard; Larmor rotation not applied to the wavefield.

## 2026-08-08 — [Done] Signal-backed results and new propagation modes
**Goal:** Add `propagate_moments` (beam-envelope covariance) and `propagate_wave` (paraxial wave optics, sea_eco Signal-backed), each with its own result container, on a cleaned geometric ray vector.
**Why:** rayTEM only had geometric ray transfer; the intended `[x,θx,y,θy,z,E]` reorder (I/R pulled out of the ray columns) was left half-finished (7/13 tests red on a fresh clone), and there was no wavelength, phase, or ensemble-statistics machinery.
- [x] Step 1: finish ray-representation refactor (6 geometric cols; I/R as separate arrays); tests green (13 passed). NOTE: `microscopes/` instrument scripts that read `columnByName("I")`/`("R")` must move to the separate `.I`/`.R` arrays — not updated here (instrument scripts, not framework).
- [x] Step 2: `relativistic_wavelength` + `Source.voltage`/`E`/`wavelength`; per-mode container attrs (covariance_matrix, mu, wave); rays wrapped as a `SignalSet` view (`rays_signalset()`).
- [x] Step 3: beam-envelope `propagate_moments` (Σ'=MΣMᵀ); beam_widths/emittance — verified vs Monte-Carlo and ray-optics focus.
- [x] Step 4: wave optics `propagate_wave` + `waveoptics.py` + `seashells.make_wavefield_signal`/`read_wavefield` (angular-spectrum, focal/tilt phase, aperture mask); single `(Nz,Ny,Nx)` wave Signal; verified focus, Fresnel Gaussian, `.sea` complex round-trip.
- [x] Step 5: wiki refresh + CLAUDE.md/index/layer-map updated for the new convention and three modes.
- [x] Step 6 (follow-up request): unified `propagate(*args, kind=..., **kwargs)` dispatcher on Element/Section/Microscope (kinds: ray/rays, moments/envelope/covariance, wave) + test.
- [x] Step 7 (follow-up request): geometric-ray API migration of the non-framework scripts. The instrument (`microscopes/`) tree was removed upstream to prevent leaking proprietary info, so that migration is moot on the clean history. The surviving generic `examples/` were migrated instead: `plot2D`/`plot3D` now take `R`, imports moved to `pySEA.rayTEM.*`. `examples/01_basicRays.ipynb` executes end-to-end headlessly with zero errors (verified via nbconvert). Two pre-existing, non-proprietary bugs that had made the fitting examples non-functional were also fixed: `Element.kget`/`kset` (getattr/setattr by name — `fitForCrossover` called them but they were never defined) and a stale fit-target index in `02_basicFitting.py`. Framework suite 24 passed.

**Addendum (refinements):** `.covariance_matrix` is now a calibrated sea_eco `Signal` (`(n_planes, 6, 6)`, unstructured `z` axis + `row`/`col` component axes, `convention` labels in metadata) — matching the original plan; `beam_widths`/`emittance` and the drivers accept either the Signal or a raw ndarray (via `seashells.as_ndarray`, which discriminates on `.dimensions` since `ndarray.data` is a memoryview). Added `Microscope.show(kind="ray"|"moments"|"wave")`: `ray` keeps the annotated ray diagram; `moments` plots RMS beam-envelope widths vs z from the covariance Signal; `wave` images `|E|²` of a wavefield plane — the last two plot the result `Signal.data` without element/plane overlays (that annotation is future work). 26 tests pass.

**Addendum (default column):** added `microscopes/basic_column.py` + `basic_column.sea` — a generic, instrument-agnostic column template: G (200 kV source) → C (C1–C3, each bracketed by dipole pairs, quad pair at end) → O (OL1/OL2 with pre/post dipole pairs, quad at end) → P (PL1–PL4, each with pre/post dipole pairs) → `detector` plane. A dipole pair is two thin dipoles at one plane with axes 0° and 45°; a quad pair is `(+K, −K)` since a 90° quad rotation is a sign flip (a true 45° skew quad isn't representable — `Quadrapole` has no axis param). Fixed three pre-existing `Dipole.__init__` bugs to enable this: `.lower()` called before the float check (crash on angle input), `np` vs `xp`, and `axis` not stored (so `.sea` round-trips silently reset rotated dipoles to `'x'`). Verified: reload from `.sea` preserves all sixteen 45° dipoles, all three propagation modes run on the reloaded scope, ray diagram renders.

**Outcome:** rayTEM now has three interchangeable propagation modes on one 6-col geometric ray vector — `propagate_ray`, `propagate_moments` (`.mu`/`.covariance_matrix`), and `propagate_wave` (`.wave`, a calibrated sea_eco Signal) — reachable individually or via a unified `propagate(kind=...)`. 24 tests pass and the generic examples (incl. `01_basicRays.ipynb`) run on the clean history. Branch `Signal_and_propagation_additions` is based on the cleaned, IP-scrubbed remote (the instrument tree and old PR history were removed upstream to protect proprietary info; this work was force-aligned onto that clean remote and only the non-proprietary framework + examples changes were re-applied). Follow-ups: the sea-eco `Dimension` debug-print cleanup and `pysea-discover-wiki` ecosystem-index regeneration.
