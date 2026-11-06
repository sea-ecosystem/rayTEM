# TODO — Collins wave propagation, and contracted section transfers

**Branch/worktree:** `dev` (rayTEM; sea-pearl `dev` uses it for the emulator's detector planes)
**Plan:** in sea-pearl, `src/pySEA/ai_wiki/sea_pearl/notes/eric/PLAN_2026-11-05_contracted-section-transfers.md` (and its TODO there)

Eric (2026-11-06), settling D1: "I relent… do the Collins transform as a new propagation
method. Name it similar to the other propagation methods… Then implement the contraction."
The scaled/hybrid walker is unchanged: Collins is a fifth, separate mode, not a lens body
inside the scaled wave (the 2026-08-19 decision still holds for that method).

- [x] Collins core (`waveoptics`): one linear canonical step on a field carried on a lattice
      with a curvature matrix; near (Fresnel + imaging), far (chirp · FFT · chirp) and mixed
      per-axis forms chosen by sampling; coupled 4×4 maps (Larmor, rotated quadrupoles) native
- [x] Collins wavefield through `seashells` (lattice, central ray, curvature; fallback class)
- [x] contraction (`collins.py`): runs of linear elements → one affine map (`affine_matrix`
      products); apertures as masks; refuse non-quadratic screens (D2 open); kernel cache
- [x] `propagate_wave_collins` on Element, MicroscopeSection, Microscope; `kind="wave_collins"`
- [x] tests: analytic (drift, 2f Fourier, 2f-2f imaging, Gaussian q), rotated quadrupole and
      Larmor vs rays, against the hybrid walker
- [ ] docs: wave-optics-sampling.md decision amended; propagation_modes.md; CLAUDE.md invariant; wiki

Verification (2026-11-06, `tests/test_collins_wave.py`, 15 pass): drift = angular spectrum to 1e-12;
2f = Gaussian far-field width to 1e-3, power conserved to 1e-10; 4f imaging carries ×(−5) and a
0.3 rad rotation in the lattice (no resampling); a rotated astigmatic lens's line focus (mixed
form, eigen-lattice) matches a 4×-finer brute-force grid to 5e-4 in intensity (the only difference
is the dropped global Gouy phase); contracted = element by element to 3e-15 pointwise; the central
ray is the lab-frame ray through rotating thick lenses; aperture = fixed path's mask. Against the
hybrid walker (two lenses, one crossover): power equal, rms size to 8e-4, coarse image L1 3.5 %.
Convergence (same column): Collins at a given pixel reproduces a 4×-larger 1024² grid to 5e-5;
the walker is 6–14 % from it at 128–512 px and 2.4 % at 1024 px (its frame switches resample).
Full suite: the 41 failures/errors are identical with and without this change (this sandbox has no
sea-sand, so sea_eco Signals are unavailable); the sea_eco Signal path of the Collins wavefield is
therefore unexercised here.
