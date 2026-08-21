# TODO — conjugate planes: image AND diffraction families

**Branch/worktree:** `Signal_and_propagation_additions` (rayTEM)
**Plan:** none (findings + spec in LOG entry)

Eric asked whether the hybrid engine finds image planes, back-focal planes,
or both. Measured answer: only ONE family. The frame is seeded flat
(s=1, R=inf) = a parallel wavefront = findPlanes' *diffraction ray*, so
`crossovers` are the diffraction / back-focal planes. Verified exactly on a
thin 2-lens compound system: wave crossovers [55.0000, 176.0000] mm ==
analytic compound values == ray diff family, to 0 nm (and NOT z_L2 + f2 =
140 mm, so compound systems are handled right — R carries all upstream
history). The image plane at 150.8621 mm is not logged.

- [x] `Microscope.conjugate_planes(axis)` -> {"diff": z[], "image": z[]} in
      METRES, by tracing the 4 reference rays on a copy and reusing
      `postprocessing.findPlanes` + `zFromFractional` (no new geometry code,
      no clobbering of self.rays)
- [x] annotate both families in the scaled cross-section (crossovers from the
      wave run + image planes from the ray reference)
- [x] document precisely what `crossovers` is (and that zpts= can log the
      image planes exactly: `zpts=scope.conjugate_planes()["image"]`)
- [x] tests (thin 2-lens: both families exact) + wiki + protocol finish
