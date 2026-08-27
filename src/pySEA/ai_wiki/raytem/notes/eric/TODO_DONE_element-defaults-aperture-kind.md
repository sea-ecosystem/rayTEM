# TODO — transparent Element defaults + aperture field_kind

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (Eric's review of the scaled-wave demo)

- [x] Step 1 — root `Element` carries a working default for every propagation
      kind: `transfer_matrix` returns the identity (no longer abstract) and
      `phase_shift` returns the transparent program (free segment over its
      length; scaled → `(0.0, None)`), so any element without its own wave/ray
      physics propagates as identity instead of raising
- [x] Step 2 — fold the aperture initial wave into the Source field machinery:
      `field_kind` gains `'aperture'` + new `aperture_radius` attribute;
      `field()` dispatches to it; `scaled_field()` loses its separate
      `aperture_radius` kwarg and always seeds from `field()`
- [x] Step 3 — tests (base-element transparency, aperture kind), example
      update, wiki/docs sync, full suite green
