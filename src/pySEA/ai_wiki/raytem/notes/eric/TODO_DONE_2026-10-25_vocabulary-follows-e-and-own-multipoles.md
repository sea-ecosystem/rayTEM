# TODO — The vocabulary accessor follows the E / own-multipole schema

**Branch/worktree:** `dev` (main checkout), as the merge of `origin/dev` into the local `dev` that held `get_schema_vocabulary`; sea-pearl's MCP side the same way on its `dev`
**Plan:** none (Eric, 2026-10-25: the remote schema takes precedence; adapt the functions that produce the vocabulary)

**Goal:** `get_schema_vocabulary()` (local, unpushed, 2026-10-16) describes the schema as it is after `d03b599` (E spectrometer, multipoles named on their own, numbers in column order).

**Why:** the accessor was written against the old schema and still reported section `S` and an `attached_multipole` pattern (`CL1_D1a`) that the validator now refuses.

## Steps

- [x] merge `origin/dev`; `__all__` keeps both `get_schema_vocabulary` and `migrate_legacy_section`
- [x] `patterns` is one `element` pattern (`<SECTION><TYPE>[INDEX][AXIS]`, `CD1a`, `EP1`)
- [x] new `retired` key: `CL1_D1a` -> `CD1a`, `SP1` -> `EP1`, each with what to write instead
- [x] rules: own-section multipoles, column-order numbers; shared letters are A, D, P (the old rule said A, C)
- [x] tests (`test_schema.py`, 85) and doctests pass; wiki `schema.md`, refresh
