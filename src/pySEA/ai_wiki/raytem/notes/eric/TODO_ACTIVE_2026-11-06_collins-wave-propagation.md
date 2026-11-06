# TODO — Collins wave propagation, and contracted section transfers

**Branch/worktree:** `dev` (rayTEM; sea-pearl `dev` uses it for the emulator's detector planes)
**Plan:** in sea-pearl, `src/pySEA/ai_wiki/sea_pearl/notes/eric/PLAN_2026-11-05_contracted-section-transfers.md` (and its TODO there)

Eric (2026-11-06), settling D1: "I relent… do the Collins transform as a new propagation
method. Name it similar to the other propagation methods… Then implement the contraction."
The scaled/hybrid walker is unchanged: Collins is a fifth, separate mode, not a lens body
inside the scaled wave (the 2026-08-19 decision still holds for that method).

- [ ] Collins core (`waveoptics`): one linear canonical step on a field carried on a lattice
      with a curvature matrix; near (Fresnel + imaging), far (chirp · FFT · chirp) and mixed
      per-axis forms chosen by sampling; coupled 4×4 maps (Larmor, rotated quadrupoles) native
- [ ] Collins wavefield through `seashells` (lattice, central ray, curvature; fallback class)
- [ ] contraction (`collins.py`): runs of linear elements → one affine map (`affine_matrix`
      products); apertures as masks; refuse non-quadratic screens (D2 open); kernel cache
- [ ] `propagate_wave_collins` on Element, MicroscopeSection, Microscope; `kind="wave_collins"`
- [ ] tests: analytic (drift, 2f Fourier, 2f-2f imaging, Gaussian q), rotated quadrupole and
      Larmor vs rays, against the hybrid walker
- [ ] docs: wave-optics-sampling.md decision amended; propagation_modes.md; CLAUDE.md invariant; wiki
