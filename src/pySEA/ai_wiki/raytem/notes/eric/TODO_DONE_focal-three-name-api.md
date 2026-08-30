# TODO — three-name focal API + interactive docs figure

**Branch/worktree:** `finishing_propagation_additions` (rayTEM), merged into
`dev` on every commit.
**Plan:** adopted from Eric's planning session (issue #11 discussion): the
measured exit-face quantity moves off `focal_length` onto a new signed
`back_focal_distance = -A/C`; `focal_length` becomes the textbook EFL
(reciprocal of `focal_power`); `focal_power = -C` unchanged.

- [x] elements.py: `focal_length` = EFL (thin path unchanged),
      new signed `back_focal_distance`, docstrings for all three.
- [x] tests: rework `test_thick_lens_efl_vs_bfd_split` into the physics
      test list (matrix defs, A·f relation, parallel-ray angle, real BFP by
      appended drift, thin limit, KL>π/2 virtual BFP vs in-body crossover);
      stale `_thick_strength` docstring.
- [x] docs: interactive figure `docs/_static/thick_lens_focal_geometry.html`
      (scrubbed of discussion framing), terminology.md updated to the
      three-name scheme, wiki Lens note, site rebuilt.
- [x] issue #11 comment proposing the adopted scheme with commit pointer.
- [x] suite green, examples 05/06 unchanged numbers, branches in sync.
