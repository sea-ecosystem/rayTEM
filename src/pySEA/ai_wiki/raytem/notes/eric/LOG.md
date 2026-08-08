# Eric's Work Log

Newest entries at top.
Status markers: `[Under Construction]` while in progress · `[Done]` when complete.

---

<!-- Add entries here as work is completed. See notes/ondrej/LOG.md for format reference. -->

## 2026-08-08 — [Under Construction] Signal-backed results and new propagation modes
**Goal:** Add `propagate_moments` (beam-envelope covariance) and `propagate_wave` (paraxial wave optics, sea_eco Signal-backed), each with its own result container, on a cleaned geometric ray vector.
**Why:** rayTEM only had geometric ray transfer; the intended `[x,θx,y,θy,z,E]` reorder (I/R pulled out of the ray columns) was left half-finished (7/13 tests red on a fresh clone), and there was no wavelength, phase, or ensemble-statistics machinery.
- [x] Step 1: finish ray-representation refactor (6 geometric cols; I/R as separate arrays); tests green (13 passed). NOTE: `microscopes/` instrument scripts that read `columnByName("I")`/`("R")` must move to the separate `.I`/`.R` arrays — not updated here (instrument scripts, not framework).
- [x] Step 2: `relativistic_wavelength` + `Source.voltage`/`E`/`wavelength`; per-mode container attrs (covariance_matrix, mu, wave). rays-as-SignalSet deferred into step 4.
- [x] Step 3: beam-envelope `propagate_moments` (Σ'=MΣMᵀ); beam_widths/emittance — verified vs Monte-Carlo and ray-optics focus.
- [ ] Step 4: wave optics `propagate_wave` + `waveoptics.py` + `seashells.make_wavefield_signal`
- [ ] Step 5: wiki refresh + docs + CLAUDE.md core-invariant update
