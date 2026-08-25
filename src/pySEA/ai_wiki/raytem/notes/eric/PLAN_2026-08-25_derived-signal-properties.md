# Derived-signal properties: store, recompute, or identity

**TODO:** [TODO_ACTIVE_aberration-class.md](TODO_ACTIVE_aberration-class.md)
**Branch:** `Signal_and_propagation_additions` (rayTEM)

Eric's rule for the special Signals rayTEM hangs off elements and assemblies
(wave, rays, covariance, screen, ...). It applies to every one of them, not
just the screen, so it is written down once here rather than re-argued per
property.

## The rule

A property getter answers in one of three ways, in this order:

| State | Getter returns | Stored? |
|---|---|---|
| the Signal was **supplied** | the stored Signal | yes — it cannot be recomputed |
| not supplied, but **derivable** | freshly computed, discarded after | no — the inputs are the storage |
| neither | the **identity** for the operation | nothing to store |

"Identity for the operation" means the value that makes the operation a no-op,
in the cheapest representation available:

- multiplicative screen -> scalar `1` (not `ones((ny, nx))`: same result under
  broadcasting, no allocation)
- transfer matrix -> identity matrix

Applied to the screen specifically:

- no screen supplied, no aberrations -> `1`; `field * 1` is a no-op
- no screen supplied, aberrations present -> compute chi from the
  coefficients, return it, do **not** store it
- screen supplied -> return it, and serialize it with the element

## Why store-only-if-unrecomputable

Coefficients are ~14 floats; the chi they generate is a full grid, and depends
on `(ny, nx, dx, dy, wavelength, P)` and on `s` for the scaled path. Storing
the array in place of its inputs is strictly worse: bigger, and pinned to one
grid. A supplied screen is the opposite case — a measured wavefront, a
fabricated plate, a Zernike fit have no coefficients behind them, so the array
*is* the definition and must be kept.

(An earlier objection of mine — that a stored screen goes stale when `s`
changes — was weak, and Eric is right: a Signal can carry an `s` or `z`
dimension and stay valid. The size argument is the one that stands.)

## The cost to watch

A getter that returns either a Signal or the scalar `1` is polymorphic, so
callers doing `.data` or `.dimensions` break on the identity case. Two ways
out, and this needs deciding before the pattern spreads:

1. callers only ever *multiply*, and never introspect — keep `1`
2. the getter takes the grid it should match, so it can return a real Signal
   when asked — costs the allocation the rule was avoiding

Current lean: (1), because the wave path only multiplies, plus an explicit
`has_screen`-style predicate for code that genuinely needs to branch.
