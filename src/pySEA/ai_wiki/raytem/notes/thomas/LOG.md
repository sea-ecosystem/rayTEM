# Thomas's Work Log

Newest entries at top.

## 2026-10-28 — [Under Construction] Add flatten_2D signal processing
**Goal:** Add a public function that removes linear image tilt estimated from all four edges.
**Why:** `lintilt` estimates slopes from opposite-edge means and does not fit the full edge intensity distribution.
- [ ] Add the Signal-preserving flatten function and public export.
- [ ] Update the normalization wiki and complete the task note.
