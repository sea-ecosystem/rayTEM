# TODO — a rayTEM column carries a sea-eco Provenance

**Branch/worktree:** rayTEM `microscope-identity` (from `main`, fast-forwarded into `dev`).
**Plan:** sea-pearl `src/pySEA/ai_wiki/sea_pearl/notes/eric/PLAN_2026-10-12_microscope-identity.md`
(the change spans rayTEM, sea-pearl, rayTEM_resources and Poseidon).

- [ ] `seashells.new_seaid` becomes `new_provenance`: a sea-eco `Provenance` (`Self` `TWN01`), `None` without sea-eco
- [ ] `Microscope.save` writes the Provenance payload to JSON
- [ ] a string `Provenance` read from a file saved before 2026-10-12 is wrapped (isolated, remove after 2026-12-03)
- [ ] tests; wiki refresh
