# TODO — The column naming schema's vocabulary as data

**Branch/worktree:** `dev` directly in rayTEM and in sea-pearl (the MCP consumer), per Eric; no feature branches
**Plan:** none

**Goal:** `schema.get_schema_vocabulary()` returns the live vocabulary (section codes with names and aliases, element type codes with classes and whether a name is required, axis suffixes, the host and attached-multipole patterns, the exempt classes) as plain JSON-serialisable data.

**Why:** agents could read the schema only as `describe_schema()` prose, so sea-pearl's builder skill carried a hand-copied vocabulary table. A structured accessor lets sea-pearl serve the schema over MCP (`sea_pearl_get_naming_schema`) without restating its rules. Built from the module constants, so `register_section` / `register_element_type` additions show up.

## Steps

- [x] `get_schema_vocabulary()` in `schema.py`; `describe_schema()` unchanged
- [x] tests in `tests/test_schema.py` (shape, JSON round-trip, registered additions appear)
- [x] `wiki/rayTEM/schema.md`, refresh
- [x] on `dev` (local, not pushed)
