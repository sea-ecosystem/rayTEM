# rayTEM Agent Notes

This repository is a ray-tracing electron optics simulator for TEM instruments.
Read this file before adding elements, changing the ray representation, or touching fitting or calibration logic.

<!-- AGENT-COMMON:main — do not edit this block in CLAUDE.md/AGENTS.md; edit src/pySEA/ai_wiki/ecosystem/agent-common.md in sea-ecosystem and run: pysea-refresh-wiki --sync-agent-instructions -->

## Collaboration protocol

This repo is developed by two contributors (Ondrej and Eric), each working with
Claude. Changes happen fast. Follow this protocol on every task to avoid
stepping on each other's work.

### Before writing any code

1. `git pull`

2. Read the other contributor's log — last few entries tell you what they are
   working on and why:

   ```bash
   # If you are Ondrej's Claude:
   cat src/pySEA/ai_wiki/raytem/notes/eric/LOG.md | head -60
   # If you are Eric's Claude:
   cat src/pySEA/ai_wiki/raytem/notes/ondrej/LOG.md | head -60
   ```

3. Glance at their most recent note file for finer-grained context:

   ```bash
   ls -t src/pySEA/ai_wiki/raytem/notes/eric/   # or ondrej/
   ```

   Then check this repo's message board for open posts to you or to `all`
   (see "Message board" below), and tell your person about any you find:

   ```bash
   # Replace <me> with your notes-folder name (eric, ondrej, ...):
   ls src/pySEA/ai_wiki/raytem/notes/board/ 2>/dev/null | grep -E '^BOARD_OPEN_.*-to-(<me>|all)_'
   ```

4. Write a TODO note in **your own** notes folder with the task broken into
   checkboxes. State in a short header at the top:
   - **Branch/worktree:** the branch or worktree the work will be performed in
     (per repo when the task spans several). Names must be meaningful: a short
     kebab-case slug describing the task (e.g. `fix-pipeline-editor-undo`,
     `wiki-refresh-utf8`), ideally matching the TODO note's slug. Never use
     auto-generated or random names (`claude/abc123`, `worktree-1`); if a tool
     creates one automatically, rename it before starting work.
   - **Plan:** relative links to any plan/handoff files (see "Plans and
     handoffs" below), or `none`.

5. Append an **[Under Construction]** entry to **your own** `LOG.md` stating
   what you are about to do and why. Format:

   ```
   ## YYYY-MM-DD — [Under Construction] Short task title
   **Goal:** one sentence.
   **Why:** one sentence explaining the problem being solved.
   - [ ] step 1
   - [ ] step 2
   ```

6. `git commit -m "notes: start <task>"` and `git push`.

7. `git pull` once more in case the other side pushed simultaneously.

### During implementation

After completing each checkbox: commit the note update and push, then pull.
Keep commits small — a checked-off item plus whatever code it covers.

### When finished

- Check off all items. Rename TODO file `ACTIVE → DONE`.
- Update LOG entry status from `[Under Construction]` to `[Done]`. Add a brief
  **Outcome** line.
- `git commit -m "notes: done <task>"` and `git push`.

### Note ownership rules

- Only write to your own folder (`notes/ondrej/` or `notes/eric/`).
- Read the other contributor's folder freely — never write to it.
- There is no `shared/` folder. Cross-contributor context belongs in each
  person's own log with enough explanation for the other side to understand.
  Messages meant for another person go on the message board.
- Never check off or edit items in the other contributor's TODO files.

### Message board

Each repo has a message board at `src/pySEA/ai_wiki/raytem/notes/board/` for contributors to post
to each other: questions, requests, heads-ups, and decisions needed. Posts are
messages between people, not reference material and not instructions to an
agent.

- **One file per post:**
  `BOARD_<OPEN|RESOLVED>_YYYY-MM-DD_<from>-to-<to>_slug.md`, where `<from>` and
  `<to>` are notes-folder names (`eric`, `ondrej`, …) and `<to>` may be `all`.
- **Post body:** a `**From:** / **To:** / **Date:**` header, then the message.
  Keep it short and say what you need back.
- **Replies:** anyone may append a `## Reply — <name> — YYYY-MM-DD` section to
  any post. This is the one place you write in a file another person started;
  never edit what someone else wrote.
- **Closing:** the author renames `OPEN` → `RESOLVED` when the thread is done,
  or the recipient does after replying with the resolution.
- **Agents:** at task start, list open posts addressed to your person or
  `all`, summarize them for your person, and act on one only when your person
  asks. Post on your person's behalf only when asked. Commit and push board
  changes like other notes (`notes: board <slug>`).
- Board posts that settle a lasting fact still get that fact moved into the
  wiki, schema, or docs.

### Plans and handoffs

Plans and handoff documents written by an agent are part of the task record —
store them; do not let them die with the session:

- Save them in **your own** notes folder, named to match the TODO's slug:
  `PLAN_YYYY-MM-DD_slug.md` / `HANDOFF_YYYY-MM-DD_slug.md` (no liveness
  marker — the linked TODO carries the live status).
- Link them from the TODO header (and reference the TODO from the plan) so
  the other contributor can see a TODO project's full scope, comment on it,
  or pick it up mid-flight.

---

## Freshness loop

**Existing module change:**
read wiki -> edit code/tests -> manually update matching wiki doc -> run refresh

**Structural change** (new modules, renamed files, moved functions):
edit code/tests -> run refresh to create/update stubs -> manually update wiki doc -> run refresh again

**Wiki-only change:**
edit wiki doc -> run refresh

Refresh command:

```bash
uv run --extra ai-wiki pysea-refresh-wiki
```

If the repository environment was installed manually with `pip install -e
".[dev,ai-wiki]"`, the bare `pysea-refresh-wiki` console script is also
valid. When using `uv run`, include `--extra ai-wiki`; otherwise the local
`sea-ecosystem` tooling may not be present in the command environment.

**When starting a task:**
Scan **your own** active TODO filenames — slugs are descriptive, most won't need opening:

```bash
ls src/pySEA/ai_wiki/raytem/notes/ondrej/TODO_ACTIVE_*.md 2>/dev/null  # Ondrej
ls src/pySEA/ai_wiki/raytem/notes/eric/TODO_ACTIVE_*.md 2>/dev/null    # Eric
```

Open any of **your own** TODOs whose slug matches your task. Check items off
as you complete them; rename `ACTIVE` → `DONE` when all items are done.
Never check off or edit items in the other contributor's TODO files.

## Contributor notes

Write all working notes to your own folder (`notes/ondrej/` or `notes/eric/`).
Run `git config user.name` if unsure which folder is yours. Never write to the
other contributor's folder. There is no `shared/` folder; post messages for
other people on `src/pySEA/ai_wiki/raytem/notes/board/`.

## Schemas (prescriptive contracts)

Some shared surfaces carry a **schema**: a prescriptive contract (intents +
golden fixtures, plus an artifact format where one crosses implementations)
owned by one package and satisfied by every implementation — any frontend or
backend, in any technology. Distinct from the wiki *manifests*, which only
register what exists. Discover them via `schema-index.json` in a repo's wiki
slice (or the `schemas` field of its `meta.json`); current schemas include
sea-eco's `pipeline-editor` and `nd-plotting`.

- **Before creating or editing classes or functions related to a schema'd
  surface** (GUI elements, plotting/array backends, pipeline components),
  read that surface's `schema/intents.md`.
- **Behavior changes start in the schema**: update intents + fixtures in the
  owning package first, then the implementations and their conformance files
  (`docs/conformance/<schema-id>.md` in each implementing repo).
- Never re-derive or hand-copy schema rules into an implementation without
  pointing back at the schema; fixtures — not prose — keep implementations
  aligned.

## Documentation structure and provenance

Most ecosystem repositories expose these Sphinx sections in order:

1. **Guides** — how a user applies the package to their data: succinct
   scenario workflows and minimal working examples; no architecture.
2. **Example Notebooks** — executable end-to-end demonstrations.
3. **AI Tools** — applicable Install, MCP, Skills, and Subagents subsections,
   tied to canonical versioned artifacts.
4. **Into the sea-weeds<br>(For developers)** — for human developers: how the
   package is built and why, with rationale, boundaries, safe extension, and
   diagrams.
5. **Schema** — plain-language pages for schemas the repo owns or implements.
6. **API Reference** — generated from accurate NumPy-style docstrings,
   organized by submodule.

The developer landing page and navigation display **"Into the sea-weeds"** and
**"(For developers)"** on separate lines; verify the hard line break in built
HTML. Sphinx pages are for people. Detailed contracts, invariants, failure
modes, test mappings, and **Provenance** (source entry points, contracts,
tests, and the matching docs surfaces) live in the AI-wiki slice
`ai_wiki/<pkg>/`, which is not rendered into Sphinx. Agents look symbols up in
its `*-index.json` thin lookup first, then the wiki page it points to. Keep
pages succinct by cutting, never by splitting one concept across pages, and
prefer figures and diagrams generated from code over prose. Document justified
omissions instead of creating empty sections.

Every package includes **Notes on the `pySEA` namespace** in its overview,
mission/architecture page, or first general Guide. Record the real distribution
and import mapping, explain the shared lightweight and modular intent, show
representative imports for namespace members, and link to a current generated
catalog rather than copying a package list.

Follow the sea-ledger documentation pattern: Basic Usage or Getting Started
puts Install and verification before the smallest useful workflow; substantive
AI Tools content uses separate Install, MCP, Skills, and Subagents pages; and
Into the sea-weeds opens with a concise Developer Overview linking to flat,
deeper architecture, identity/storage, and integrations/extensions pages.
Building the docs follows the overview and gives verified repository-specific
dependency, API regeneration, clean strict HTML/PDF, example/notebook, output,
and generated-artifact commands and policy.

Before completing a documentation task in a repository with an API Reference,
run its canonical API generation command and `pysea-refresh-wiki` even when no
generated diff is expected. Inspect and record the results; failed or
unavailable regeneration leaves the task incomplete.

Repos that own or implement schemas render them in the **Schema** section and
name each schema in one line in Into the sea-weeds; intents, fixtures, and
conformance records stay in the AI-wiki. `pysea-refresh-wiki` warns when the
developer docs or schema references are missing. Wiki pages are updated in
place (no versioned snapshots), and `notes/` is a working record, not
reference.

## Coding behavior

- **Think before coding** — state assumptions explicitly before implementing. If multiple interpretations exist, present them. If something is unclear and would materially change the implementation, stop and ask.
- **Prefer simplicity** — write the minimum code that solves the problem. No unrequested features, single-use abstractions, or unrequested configurability. If a more complex solution is warranted, flag it and offer to implement, add to TODO, or skip.
- **Make surgical changes** — touch only what the task requires. Do not refactor unrelated code. Mention unrelated issues rather than fixing them silently. Match existing project style.
- **Work from verifiable goals** — convert tasks into concrete success criteria before implementing. For multi-step tasks, name each step and its verification.

> Full guidelines with examples: `sea-ecosystem/docs/dev/agent-coding-guidelines.md`

## Documentation behavior

- **Start from the user task** — Guides teach what a user needs to know to
  succeed in every supported scenario, including meaningful choices, expected
  outputs, and common recovery paths.
- **Preserve design intent** — Into the sea-weeds tells developers what was
  built, why each part exists, the rules that protect it, and how to extend it
  safely; the exhaustive detail goes in the AI-wiki.
- **Show before telling** — use a figure, diagram, or table when it
  communicates faster than prose.
- **Ground every claim** — trace documentation to current source, contracts,
  tests, executable examples, or verified behavior.
- **Orient every package** — keep its Notes on the `pySEA` namespace section
  truthful and link to the generated catalog instead of a static package list.
- **Keep private code understandable** — private, hidden, and dunder callables
  require docstrings even when excluded from public API pages.

> Documentation protocol and examples: `sea-ecosystem/docs/dev/agent-documentation-protocol.md`
> Reusable workflow: `sea-ecosystem/skills/maintain-documentation/SKILL.md`

## Style rules

- NumPy docstring style for **all** Python callables — public, private (leading underscore), dunder methods, properties, class methods, static methods, and classes. Documentation quality is part of code correctness.
  Required sections: short summary · extended summary · Parameters · Attributes (classes) · Methods (classes) · Returns · Raises · See Also · Notes · Examples · References. Include all sections that apply; Parameters, Returns, and Raises are required when they exist.
- Prefer explicit type annotations in signatures. Prefer `Literal` over `str` for fixed value sets. Prefer `Sequence` over `list` or `tuple` in signatures.
- Do not use dataclasses.
- Keep public method names descriptive and action-oriented.
- Make errors actionable and concise.

<!-- END AGENT-COMMON:main -->

`microscopes/` subdirectories contain instrument-specific scripts, not general framework code. Do not import from them in framework modules.

## Read first (always)

1. `src/pySEA/ai_wiki/raytem/index.md` — orientation and mental model
2. `src/pySEA/ai_wiki/raytem/layer-map.md` — module responsibilities and invariants
3. `src/pySEA/ai_wiki/raytem/wiki/<relevant-module>.md` — method-level detail for the file you are about to edit

For ecosystem context (what other repos exist and how they connect), see
`src/pySEA/ai_wiki/ecosystem/CLAUDE.md` (installed via the `sea-ecosystem` wiki dependency — `pip install -e ".[dev,wiki]"`).

## Navigation

Two-step lookup for any class or method:

1. Search `src/pySEA/ai_wiki/raytem/method-index.json` by `class` + `method` name
   → get `wiki_path` and `wiki_lineno`
2. Read `wiki_path` at `offset=wiki_lineno` — lands directly in the right section

If `wiki_method_lineno` is non-null, use it instead for a tighter read.

<!-- REPO-MAP-START — regenerated by pysea-refresh-wiki; do not edit this block -->
## Repo map

- `src/pySEA/rayTEM/__init__.py`
- `src/pySEA/rayTEM/aberrations.py` — Axial wave aberrations in Krivanek notation, and how to apply them.
- `src/pySEA/rayTEM/AS2.py`
- `src/pySEA/rayTEM/assemblies.py`
- `src/pySEA/rayTEM/collins.py` — Collins wave propagation: runs of linear elements contracted into one step each.
- `src/pySEA/rayTEM/elements.py`
- `src/pySEA/rayTEM/microscopes/basic_column.py` — Builder for the default generic TEM column (``basic_column.sea``).
- `src/pySEA/rayTEM/microscopes/MACSTEM/builder.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/CLs.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/diffraction.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/DQCM.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/OLs.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/PLs.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/rederive.py`
- `src/pySEA/rayTEM/microscopes/MACSTEM/viewAndCalculate.py`
- `src/pySEA/rayTEM/microscopes/objective_section.py` — Builder for the standalone objective section (``objective_section.sea``).
- `src/pySEA/rayTEM/moments.py` — Beam-envelope state and moment-closure models for rayTEM.
- `src/pySEA/rayTEM/postprocessing.py`
- `src/pySEA/rayTEM/references/andys_functions.py`
- `src/pySEA/rayTEM/schema.py` — Column naming schema for rayTEM microscopes.
- `src/pySEA/rayTEM/seashells.py` — seashells serves as a wrapper around the sea_eco SEASerializable object, enabling easy integration with sea_eco.
- `src/pySEA/rayTEM/tests/test_aperture_transmission.py` — An aperture's ``transmission``: current the traced rays do not sample.
- `src/pySEA/rayTEM/tests/test_collins_wave.py` — Collins wave propagation (``propagate_wave_collins``) and its contracted transfers.
- `src/pySEA/rayTEM/tests/test_covariance_propagation.py` — Tests for covariance propagation, moment closure, and examples/08.
- `src/pySEA/rayTEM/tests/test_display_signals.py` — Tests for the derived display Signals: ray diagram, beam envelope, wave cross-section.
- `src/pySEA/rayTEM/tests/test_eight_configurations.py` — Tests driving examples/07_eightConfigurations.py.
- `src/pySEA/rayTEM/tests/test_element_alignment.py` — An element's ``alignment``: where its own axis is, off the column's.
- `src/pySEA/rayTEM/tests/test_elements_sections_microscopes.py`
- `src/pySEA/rayTEM/tests/test_microscope_identity.py`
- `src/pySEA/rayTEM/tests/test_rotating_frame.py` — ``Rays.convert_to_rotating_reference_frame`` is a vectorized form of its old loop.
- `src/pySEA/rayTEM/tests/test_scaled_fresnel.py`
- `src/pySEA/rayTEM/tests/test_scaled_wave_physical_plane.py` — Tests for seashells.scaled_wave_physical_plane: ψ of one stored scaled plane.
- `src/pySEA/rayTEM/tests/test_schema.py`
- `src/pySEA/rayTEM/tests/test_source_voltage.py`
- `src/pySEA/rayTEM/tests/test_wave_and_envelope.py`
- `src/pySEA/rayTEM/utilities.py`
- `src/pySEA/rayTEM/waveoptics.py` — Paraxial scalar wave-optics primitives for rayTEM.
- `src/pySEA/rayTEM/xmlNion.py`

<!-- REPO-MAP-END -->
## Core invariants

- **Ray columns are geometric**: `convention = ["x","xt","y","yt","z","E"]` — index 0–5 as returned by `columnByName()`. The ray vector is *purely geometric*; do not reorder without updating every Element and all `columnByName` callers.
- **Intensity (I) and rotation (R) are NOT ray columns**: they travel as separate parallel arrays on `MicroscopeSection`/`Microscope` (`.I` and `.R`, shape `(n_planes, n_rays)`). Elements update them via `apply_intensity`/`apply_rotation` (R is cumulative Larmor rotation accumulated by thick lenses). Postprocessing helpers that need them (`convert_to_rotating_reference_frame`, `findPlanes`, `measureAtZ`, `plot2D`) take `R`/`I` explicitly. Any code reading `columnByName("I")`/`("R")` is stale (some `microscopes/` instrument scripts still do — update them to `.I`/`.R`).
- **Transfer matrices are 6×6**: Produced by `fix_mat_dims()` (sized off `len(convention)`). Never pass a raw 2×2 matrix directly to `propagate_ray`.
- **Five propagation modes, one geometry**: `propagate_ray` (geometric rays), `propagate_moments` (beam-envelope covariance, `Σ'=MΣMᵀ`, stored on `.mu`/`.covariance_matrix`), `propagate_wave` (paraxial 2D complex wavefield on a fixed grid, stored on `.wave` as a calibrated sea_eco `Signal`), `propagate_wave_scaled` (scaled-Fresnel wave, `ψ=(1/s)·U·e^{ikr²/2R}`, stored on `.wave_scaled` as a `SignalSet` of U + s/R/τ companions; reconstruct physical planes via `Microscope.wavefield_at`), and `propagate_wave_collins` (one Collins step per run of linear elements between apertures and aberrated elements, which are walked where they act, `collins.CollinsTransfer`; samples on a lattice with a 2×2 curvature and the central ray, stored on `.wave_collins`). Ray/moments share the transfer matrices; both wave paths share the per-element `phase_shift` contract. All run through the same bottom-up hierarchy (dispatcher `propagate(kind=...)`).
- **The scaled wave is one method**: thick bodies are quadratic-index segments crossed by the hybrid flatten → cross → rediverge policy (`propagate_quadratic_segment_hybrid`). Do not add a one-step Collins/ABCD transform for a lens body *inside the scaled wave* — rejected in `notes/eric/PLAN_2026-08-19_scaled-fresnel-wave.md`; Collins is its own mode (`propagate_wave_collins`, Eric 2026-11-06) and never replaces the walker; a failure at a crossover is a bug in the frame switch, fixed there with a test. Every stepping tolerance is **relative** to the lengths it compares: a 1 pm absolute floor in `segment_zero` once hid a focus on a half lens's exit face and hung the run (`docs/wave-optics-sampling.md`, "A focus on a body's exit face").
- **Element → MicroscopeSection → Microscope hierarchy**: Rays propagate strictly bottom-up. Microscope never reaches inside an Element; MicroscopeSection never reaches inside a Microscope.
- **seashells is the sea_eco seam**: All serialization *and* wavefield-Signal construction (`make_wavefield_signal`/`read_wavefield`) go through `seashells`, which gracefully degrades if sea_eco is absent. Do not import from sea_eco directly in framework code.
- **Wavelength**: `Source(voltage=<kV>)` sets `Source.wavelength` (via `utilities.relativistic_wavelength`) and fills the per-ray `E` column. Default `voltage=None` keeps `E=0` and no wavelength, so ray-only behavior is unchanged.

## High-risk areas

- `elements.py` — `columnByName()` and `fix_mat_dims()` are referenced everywhere; changes break all Elements
- `seashells.py` — conditional import logic; changing import path or attribute names breaks sea_eco round-trips and wavefield-Signal construction
- `waveoptics.py` — paraxial FFT math; grid centering (`transverse_coordinates`) and the angular-spectrum transfer function must stay consistent or focus/aperture positions drift
- `AS2.py` — talks to a live instrument; errors here can send bad values to real hardware

## AI workflow

- Prefer small, targeted changes.
- Update or add tests when framework behavior changes.
- Keep wiki docs synchronized with module behavior.
- If a change modifies the ray representation or transfer matrix conventions, explain the change explicitly in the wiki doc.

## Accessing wiki content programmatically

From any Python environment with `raytem` installed:

```python
from importlib.resources import files
index = (files("pySEA") / "ai_wiki" / "raytem" / "index.md").read_text()
method_index = (files("pySEA") / "ai_wiki" / "raytem" / "method-index.json").read_text()
```
