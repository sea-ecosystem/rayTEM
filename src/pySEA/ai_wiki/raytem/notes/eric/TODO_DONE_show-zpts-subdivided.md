# TODO — dense z sampling for show: Microscope.subdivided + show(zpts=)

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (spec in LOG entry)

Eric: the chunky cross-section is predictable (planes = whatever z the column
defines) but wants (a) a helper for seamless/dense propagation like the demo's
subdivided column, and (b) plotting at arbitrary z through show — "feed in a
different set of z coordinates or redefine the z dimensions temporarily and
then feed them back to their original state."

- [x] `Microscope.subdivided(zpts)` -> NEW Microscope (original untouched):
      float dz = split unnamed drifts into <= dz chunks; sequence = cut the
      unnamed drifts at those absolute z positions
- [x] `show(kind="wave-scaled"/"wave-hybrid", zpts=...)`: propagate a
      temporary subdivided copy and plot from its planes; self's stored
      result is never modified (cleaner than mutate-and-restore)
- [x] tests: geometry preserved (named_positions), explicit z planes logged,
      show(zpts=) leaves self untouched
- [x] wiki + protocol finish
