# TODO — Microscope.show for scaled/hybrid wave results

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (small feature; spec in LOG entry)

Eric: the scaled-wave results should plot with `show()`. Today
`show(kind="wave-scaled"/"wave-hybrid")` is in the signature but raises;
the demo figures are custom matplotlib in examples/04.

- [ ] `Microscope.show(kind="wave-scaled"|"wave-hybrid")`: default (no plane)
      draws the |ψ(x, 0, z)| cross-section with element/crossover annotations
      (wave analog of the ray diagram); `plane=` (index, z in metres, or a
      named position like "sample") images that plane's |ψ|² by delegating to
      the reconstructed wavefield Signal's own `.show()` (same composition
      pattern as kind="wave"/"moments")
- [ ] test: headless show for both forms on the small scaled column
- [ ] wiki (assemblies.md) + protocol finish
