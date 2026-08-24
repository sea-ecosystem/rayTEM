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

## 5. The one obstacle: attaching a class to a serializable element

`Aberrations` is a container, and containers are exactly what `.sea` does not
take. Measured on the real serialization tests:

| stored on a `Lens` | `.sea` round trip |
|---|---|
| float / int / bool / str / list / tuple | works |
| dict | fails |
| numpy array | fails |
| **nested `SEASerializable`** | **fails** (12/13 tests) |

So `lens.aberrations = Aberrations(...)` cannot work today. Three ways out,
and this needs a decision before anything is written:

1. **Teach `seashells` to serialize nested objects.** `safeReinstantiate` maps
   `kind` strings to classes and filters to constructor parameters; an
   `Aberrations` would need registering there and a nested write/read path.
   Most correct, touches the sea_eco seam.
2. **Store as lists, expose as a class.** Keep `names`/`values` lists on the
   element (both serialize) and make `aberrations` a **property** returning an
   `Aberrations` view. Serialization-safe today, and dot access still works;
   the object is rebuilt rather than stored.
3. **Put the coefficients in `Metadata`**, which is where Eric notes dict-like
   instrument information belongs, and have the element hold only a reference.
   Closest to how sea-eco already carries this from real files.

**Recommendation: 3, falling back to 2.** Metadata is where the swift reader
already puts these, so a rayTEM lens and a sea-eco dataset would then speak the
same structure, and `from_metadata` stops being a conversion and becomes the
normal path.

## 6. Open questions

1. §5 — which of the three, and if (1), is changing `seashells`/sea_eco in
   scope?
2. Should the class live in **rayTEM** (it is attached to elements) or in
   **sea-eco** (it is read from Metadata, and sea-eco owns Metadata)? rayTEM
   must not import sea_eco except through `seashells`.
3. `C10` is defocus, which is a *paraxial* quantity: should it fold into
   `focal_power()` so the ray matrix sees it too, or stay a pure wave term?
   Currently measured to move the wave focus to `1/(P + C10 P^2)` while the ray
   matrix does not move at all.
4. Fifth order: sea-eco's reader stops at order 4 (`C45`). Do we extend the
   reader's list to `C50/C52/C54/C56`, or does rayTEM support terms the reader
   cannot yet supply?
