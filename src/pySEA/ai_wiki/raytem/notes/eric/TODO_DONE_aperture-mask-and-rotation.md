# TODO — aperture masking + rotation rename

**Branch/worktree:** `finishing_propagation_additions` (rayTEM), merged into
`dev` on every commit.
**Plan:** none (Eric's direction: masks win for crossover fitting, plotting,
and resolution; "skew" reads as shear — the roll about z is `rotation`).

- [x] Rename: `Quadrapole.skew` → `rotation` (declared on `Element`, ignored
      by rotationally symmetric elements); the Larmor bookkeeping attribute
      `rotation` → `larmor_rotation` everywhere.
- [x] `Aperture` ray path becomes a true MASK (Thomas's original option 1 in
      the class comments): geometry passes through, blocked rays get I = 0.
      Fixes his documented one-aperture-only limitation of the rescale.
      Current stays sum(I) — sampled, quantized by I_total/n_rays.
- [x] plot2D truncates dead rays (NaN beyond the plane where I hits 0).
- [x] ex07 predict/overlay logic reworked for masking; eight-config tests
      re-verified; test_every_element fixture checked.
- [x] Docs (wiki Aperture/Lens notes, propagation_modes/operating notes if
      they mention rescale), site rebuild, suite green, examples run.
