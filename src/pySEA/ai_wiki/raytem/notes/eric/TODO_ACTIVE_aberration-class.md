# TODO — Aberrations class + generic application

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** [PLAN_2026-08-23_aberration-class-and-generic-application.md](PLAN_2026-08-23_aberration-class-and-generic-application.md)

Eric's correction: aberrations belong in a class in **Krivanek `C_{n,m}`**
notation, the phase screen belongs in the dataset as a Signal, and propagation
must apply whatever function is attached rather than knowing about spherical.

**Blocked on plan §6 Q1/Q2** — a class cannot be attached to an element today
(nested SEASerializable breaks `.sea`), and where the class lives depends on
whether Metadata carries it.

- [ ] decide §6 Q1 (storage) and Q2 (which package)
- [ ] `Aberrations` class: Krivanek `C_{n,m}` complex storage, `convention` attr
- [ ] `from_metadata`, including the reader's outstanding a/b -> complex step
- [ ] convention conversions (letters now; Seidel/Zernike later)
- [ ] generic `phase()` and `gradient()`
- [ ] phase screen as a Signal on the element dataset, manual or generated
- [ ] dot-access properties for the well-known Signals
- [ ] `apply_aberrations=True` flag on every propagation method
- [ ] ray kick from `(1/k) grad chi` generically, all orders
- [ ] retire `spherical_phase`, `Lens.Cs`, the flat `C1/A1/...` attributes
- [ ] re-point `examples/06` at the new API
