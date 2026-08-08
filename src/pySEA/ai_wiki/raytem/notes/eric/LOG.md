# Eric's Work Log

Newest entries at top.
Status markers: `[Under Construction]` while in progress · `[Done]` when complete.

---

<!-- Add entries here as work is completed. See notes/ondrej/LOG.md for format reference. -->

## 2026-08-08 — [Done] Signal-backed results and new propagation modes
**Goal:** Add `propagate_moments` (beam-envelope covariance) and `propagate_wave` (paraxial wave optics, sea_eco Signal-backed), each with its own result container, on a cleaned geometric ray vector.
**Why:** rayTEM only had geometric ray transfer; the intended `[x,θx,y,θy,z,E]` reorder (I/R pulled out of the ray columns) was left half-finished (7/13 tests red on a fresh clone), and there was no wavelength, phase, or ensemble-statistics machinery.
- [x] Step 1: finish ray-representation refactor (6 geometric cols; I/R as separate arrays); tests green (13 passed). NOTE: `microscopes/` instrument scripts that read `columnByName("I")`/`("R")` must move to the separate `.I`/`.R` arrays — not updated here (instrument scripts, not framework).
- [x] Step 2: `relativistic_wavelength` + `Source.voltage`/`E`/`wavelength`; per-mode container attrs (covariance_matrix, mu, wave); rays wrapped as a `SignalSet` view (`rays_signalset()`).
- [x] Step 3: beam-envelope `propagate_moments` (Σ'=MΣMᵀ); beam_widths/emittance — verified vs Monte-Carlo and ray-optics focus.
- [x] Step 4: wave optics `propagate_wave` + `waveoptics.py` + `seashells.make_wavefield_signal`/`read_wavefield` (angular-spectrum, focal/tilt phase, aperture mask); single `(Nz,Ny,Nx)` wave Signal; verified focus, Fresnel Gaussian, `.sea` complex round-trip.
- [x] Step 5: wiki refresh + CLAUDE.md/index/layer-map updated for the new convention and three modes.

**Outcome:** rayTEM now has three interchangeable propagation modes on one 6-col geometric ray vector — `propagate_ray`, `propagate_moments` (`.mu`/`.covariance_matrix`), and `propagate_wave` (`.wave`, a calibrated sea_eco Signal). 23 tests pass. Branch `Signal_and_propagation_additions`, based on the cleaned `origin/main` (the old branch's history — which still carried the purged microscope-IP files — was deliberately not reused). Follow-ups recorded in the TODO_DONE note: sea-eco `Dimension` debug-print cleanup, `pysea-discover-wiki` ecosystem-index regeneration, and migrating `microscopes/` scripts off `columnByName("I")`/`("R")`.
