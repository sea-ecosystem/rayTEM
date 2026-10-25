# TODO — schema: E for the spectrometer, multipoles named on their own

**Branch/worktree:** `dev` (main checkout)
**Plan:** none (decided with Eric in the U200 scope session, 2026-10-21); the
sea-pearl side is `notes/eric/TODO_ACTIVE_2026-10-21_scope-map-schema.md` in sea-pearl.

- [x] spectrometer section code `E` (was `S`); no `S` alias
- [x] a file with a section named `S` is renamed `E` as it is read (`migrate_legacy_section`, called from `seashells.SEASerializable.from_hdf5_group`)
- [x] multipoles named on their own in their section (`CD1a`); the attached form `CL1_D1a` is refused, the message names the form to write
- [x] numbers per type within a section run in column order; gaps allowed; not vendor slot numbers
- [x] canonical axis form `CD1a` (`PD1_a` still parses)
- [x] tests (`test_schema.py`), module docs, wiki `schema.md`
- [x] U200, emulator, emulator_column, emulator_usim and usim scopes validate with no errors
