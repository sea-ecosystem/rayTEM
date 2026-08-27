# TODO — Stale ray-doc fixes + scaled-wave basic_column demo

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (small follow-up to
[TODO_DONE_scaled-fresnel-wave.md](TODO_DONE_scaled-fresnel-wave.md))

- [x] Step 1 — fix stale wiki ray-mode sections (elements.md/assemblies.md
      still describe the pre-refactor 8-column convention with I/R as ray
      columns and 8×8 matrices); align to `convention = [x,xt,y,yt,z,E]`,
      6×6 matrices, separate `.I`/`.R`; regenerate ecosystem index.md if the
      env allows
- [x] Step 2 — scaled-wave demo on `microscopes/basic_column.sea`:
      `propagate_wave_scaled` (crossovers handled by stopping at the s_min
      guard per section), save the scaled result to `.sea`, render a
      |ψ|(x, z) cross-section (wave analog of the ray plot2D) and x–y
      intensity slices at several column planes (`examples/`)
