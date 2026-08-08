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
- [x] Step 6 (follow-up request): unified `propagate(*args, kind=..., **kwargs)` dispatcher on Element/Section/Microscope (kinds: ray/rays, moments/envelope/covariance, wave) + test.
- [x] Step 7 (follow-up request): migrated every `microscopes/` instrument script + example off `columnByName("I")`/`("R")` to `.I`/`.R`, added the `R` arg to `findPlanes`/`plot2D`/`plot3D`, and fixed `measureAtZ` calls (`I=`/`R=`/`section=`, 6-tuple unpack). All touched scripts `py_compile` clean; framework suite 24 passed.

**Outcome:** rayTEM now has three interchangeable propagation modes on one 6-col geometric ray vector — `propagate_ray`, `propagate_moments` (`.mu`/`.covariance_matrix`), and `propagate_wave` (`.wave`, a calibrated sea_eco Signal) — reachable individually or via a unified `propagate(kind=...)`. All instrument scripts were migrated to the new geometric-ray API. 24 tests pass. Branch `Signal_and_propagation_additions`, based on the cleaned `origin/main` (the old branch's history — which still carried the purged microscope-IP files — was deliberately not reused). The plan + an unmarked copy of the checklist were also placed on `rayTEM/main`. Follow-ups recorded in the TODO_DONE note: the pre-existing broken `04_PRIVATE_INSTRUMENT.py` scratch fragment, the sea-eco `Dimension` debug-print cleanup, and `pysea-discover-wiki` ecosystem-index regeneration.
