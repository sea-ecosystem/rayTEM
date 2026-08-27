# Plan — an Aberrations class, and applying it generically

**For:** Eric + the other contributor
**Status:** proposal. Written after Eric corrected the design in three ways;
nothing below is implemented yet.
**Supersedes parts of:** `PLAN_2026-08-22_aberrated-focal-surfaces.md` (its wave
section §4 stands; its naming and its per-term approach do not)
**TODO:** [TODO_ACTIVE_aberration-class.md](TODO_ACTIVE_aberration-class.md)

## 0. What I got wrong, so it is not repeated

1. **Naming.** I used `C1, A1, B2, S3, A3, ...` and called it Krivanek. Those
   letters are a *different* convention. Krivanek is `C_{n,m}` — and sea-eco's
   swift reader already speaks it: `C10`, `C12.a`, `C12.b`, `C21.a`, `C21.b`,
   `C23.a/b`, `C30`, `C32.a/b`, `C34.a/b`, `C41.a/b`, `C43.a/b`, `C45.a/b`
   (`io.py`, `map_C_abb`). `.a`/`.b` are the two Cartesian components of the
   complex coefficient — the reader's own TODO says "need to convert these from
   ab to complex".
2. **Per-aberration methods.** `spherical_phase`, `Lens.Cs`, and a bespoke
   `aberration_kick` for third-order spherical are all wrong shapes. Nothing in
   propagation should know *which* aberration it is applying.
3. **"Scalars only" was wrong.** I claimed `.sea` stores only scalars. Measured:
   `float`, `int`, `bool`, `str`, `list`, `tuple` all round-trip. `dict` and
   `numpy.ndarray` do not. **A nested `SEASerializable` attribute also does
   not** (12/13 of the serialization tests fail) — see §5, this is the one real
   obstacle in the plan.

## 1. The `Aberrations` class

A real class, because there is behaviour to hang on it, not just numbers.

- **Storage:** Krivanek `C_{n,m}` as **complex** coefficients, keyed `(n, m)`.
  `.a`/`.b` map to real/imaginary. `m = 0` terms are real.
- **`convention` attribute** naming the convention the values are in
  (`'krivanek'` initially), so a set of numbers is never ambiguous.
- **`from_metadata(metadata)`** — read a sea-eco `Metadata` tree (or a plain
  dict) as written by the swift reader, including the `a`/`b` → complex
  conversion the reader leaves undone.
- **Convention conversions** — `to_letters()` / `from_letters()` for the
  `C1/A1/B2/S3` format, and room for Seidel and Zernike later. Each conversion
  states its own reference.
- **`phase(shape, dx, dy, wavelength, power)`** — the aberration function
  `chi`, generically over whatever terms are set.
- **`gradient(...)`** — `∂chi/∂x, ∂chi/∂y`, which is what ray optics needs
  (§3).

Attachable to a `Lens`, a `MicroscopeSection`, or a `Microscope`. Attribute
defaults to `None`.

## 2. Phase shift as a Signal on the element's dataset

Array data does not belong in an attribute. The phase screen should live where
`wave`, `rays` and the scaled variants already live — as a `Signal` on the
element's dataset — and be either:

- **supplied manually**, so a measured or externally computed screen can be
  used directly, or
- **generated** from the attached `Aberrations`.

Eric's suggestion, worth taking: give each of the well-known Signals a property
for dot access (`element.phase_shift_signal`, `.wave`, `.rays`, `.covariance`),
rather than callers reaching into a dataset by string key.

## 3. Applying it generically — the core of the change

Propagation must not branch on which aberration is present.

- **Wave:** the field is multiplied by `exp(i*chi)`. `chi` comes from the
  aberration function; nothing about that step is specific to spherical, or to
  any order.
- **Ray:** the transverse ray aberration is the **gradient of the aberration
  function**, `Δθ = (1/k)·∇chi`, evaluated at the pupil. This is exact in the
  eikonal limit **at every order**, not just first — so the same one expression
  covers `C10` through `C56`. It is already how the existing `C3` kick was
  derived; the mistake was hard-coding that one case instead of taking the
  gradient of whatever function is there.
- **One flag, not one method per aberration:** every propagation method takes
  `apply_aberrations: bool = True`. If an aberration function is attached it is
  applied; if not, the flag does nothing. No `Cs=` parameter, no
  `spherical_phase`, no per-term branch.

Retired by this: `waveoptics.spherical_phase`, `Lens.Cs`, `Lens.aberration_*`
in their current per-term forms, and the flat `C1/A1/...` attributes I added.

## 4. Thick bodies

The distributed-through-the-body treatment (integrating the perturbation along
the medium, validated against the perturbed ray ODE) is independent of *which*
term is being distributed, so it survives §3 unchanged — it just takes the
gradient of the general function instead of a hard-coded cubic.

## 5. Serialization — resolved, and my earlier claim was wrong

I reported that a nested `SEASerializable` breaks `.sea`, 12/13 tests failing,
and called it a "direct tension" with attaching a class to a lens. **That was
my probe, not the machinery.** The probe called
`super().__init__(name=..., kind=...)`, which is `Element`'s constructor
convention; `SEASerializable.__init__` is `(self, *args, **kwargs)` and passes
through to `object.__init__`, so it raised
`TypeError: object.__init__() takes exactly one argument`. Every downstream
failure was that.

With a correct probe, **`.sea` nests exactly as designed**: a nested object
attached to a `Lens` round-tripped with its data intact (`names: ['C30']`,
`values: [0.001]`). `SEASerializable` does its job. So an `Aberrations` class
can simply subclass `seashells.SEASerializable` — which *is* sea_eco's when
sea_eco is present — and be attached to a `Lens`, a section, or a microscope
with no serialization work at all. It can live in **rayTEM** without importing
sea_eco directly, because seashells already mirrors it. Plan §6 Q2 answered.

**The one real limitation is elsewhere and is narrow.** `Microscope.save()`
(the JSON path, tpchuckles') does `json.dump(e.__dict__)`, so any non-primitive
attribute raises `TypeError: not JSON serializable`. It already carries an
explicit exclusion list — `rays`, `I`, `R`, `mu`, `covariance_matrix`, `wave` —
for exactly this reason. Aberration coefficients are small numbers that
*should* appear in a human-readable save, so the fix is to give `Aberrations` a
`to_dict()`/`from_dict()` and let the JSON writer use it, either directly or via
a `default=` hook on `json.dump`. That touches Thomas's function, so it is worth
flagging to him rather than doing silently.

## 6. Open questions

**Answered by Eric, 2026-08-23:**

1. ~~Storage~~ — resolved in §5: nesting works, the class attaches directly.
   Only the JSON writer needs a `to_dict` hook.
2. ~~Which package~~ — **rayTEM**, subclassing `seashells.SEASerializable`.
   Works either way because seashells mirrors sea_eco's architecture.
3. `C10` **folds into `focal_power()`**, so the ray matrix sees defocus too.
   Checked the provenance first: `focal_power` is mine (it arrived with the
   per-element `phase_shift` contract), not Thomas's, so changing it steps on
   nobody.
4. **Extend the reader to fifth order** — add `C50`, `C52.a/b`, `C54.a/b`,
   `C56.a/b` to `abb_C` in sea-eco's `io.py`.

Still open:

5. The JSON `to_dict` hook touches `Microscope.save()`, which is Thomas's.
   Add the hook there, or keep aberrations out of the JSON save?
