# TODO — Aberrations class + generic application

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-23_aberration-class-and-generic-application.md](PLAN_2026-08-23_aberration-class-and-generic-application.md)

Eric's correction: aberrations belong in a class in **Krivanek `C_{n,m}`**
notation, the phase screen belongs in the dataset as a Signal, and propagation
must apply whatever function is attached rather than knowing about spherical.

**Unblocked.** My "nested SEASerializable breaks .sea" finding was a broken
probe (wrong `super().__init__` signature); nesting works, so the class attaches
directly and lives in rayTEM. Eric confirmed C10 folds into `focal_power()` and
the reader extends to fifth order.

- [x] cause of the serialization failure found: my probe, not the machinery
- [x] JSON `save()` now goes through `SEASerializable.to_json` (two sea-eco bugs fixed)
- [x] `Aberrations` class: Krivanek `C_{n,m}` complex storage, `convention` attr
- [x] `from_metadata`, including the reader's outstanding a/b -> complex step
- [x] convention conversions (letters now; Seidel/Zernike later)
- [x] generic `phase_at()`/`deflection_at()` (+ grid wrappers `phase`/`gradient`)
- [ ] phase screen as a Signal on the element dataset, manual or generated
- [ ] dot-access properties for the well-known Signals
- [ ] `apply_aberrations=True` flag on every propagation method
- [x] ray kick from `(1/k) grad chi` generically, all orders, on `Element`
- [x] flat `C1/A1/...` attributes retired; `Lens.Cs` kept as a `C30` alias
- [ ] retire `waveoptics.spherical_phase`/`aberration_phase`/`KRIVANEK_TERMS`
- [ ] re-point `examples/06` at the new API
