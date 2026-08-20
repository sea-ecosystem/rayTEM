# TODO — unify wave-mode naming on "wave" (drop "field")

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (Eric's review: the object is the wave signal everywhere;
"field" was an unexplained synonym)

- [ ] Step 1 — Source seeding API renamed to match the mode names
      (`rays()`/`moments()`/`wave()`/`wave_scaled()`): `field()` → `wave()`,
      `scaled_field()` → `wave_scaled()`, `aperture_field()` →
      `_aperture_wave()` (private builder behind `wave_kind='aperture'`);
      ctor/attrs `field_shape`/`field_extent`/`field_kind` →
      `wave_shape`/`wave_extent`/`wave_kind`
- [ ] Step 2 — driver kwarg `field0` → `wave0` on Section/Microscope
      `propagate_wave`/`propagate_wave_scaled`; regenerate
      `microscopes/basic_column.sea` (stored attribute names changed)
- [ ] Step 3 — tests/examples/wiki/docs sync; full suite green
