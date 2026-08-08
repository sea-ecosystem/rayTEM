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
- Never check off or edit items in the other contributor's TODO files.

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
other contributor's folder. There is no `shared/` folder.

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

## Developer docs ("Into the SEA-weeds")

Every ecosystem package must carry developer docs titled **"Into the
SEA-weeds"** under `docs/`; repos that own or implement schemas render them
there in a **Schema** section. `pysea-refresh-wiki` warns when the docs or
the schema references are missing.

## Coding behavior

- **Think before coding** — state assumptions explicitly before implementing. If multiple interpretations exist, present them. If something is unclear and would materially change the implementation, stop and ask.
- **Prefer simplicity** — write the minimum code that solves the problem. No unrequested features, single-use abstractions, or unrequested configurability. If a more complex solution is warranted, flag it and offer to implement, add to TODO, or skip.
- **Make surgical changes** — touch only what the task requires. Do not refactor unrelated code. Mention unrelated issues rather than fixing them silently. Match existing project style.
- **Work from verifiable goals** — convert tasks into concrete success criteria before implementing. For multi-step tasks, name each step and its verification.

> Full guidelines with examples: `sea-ecosystem/docs/dev/agent-coding-guidelines.md`

## Documentation behavior

- **Start simple** - for new user-facing features, write a short guide that
  states the basic premise, the main workflow, and one minimal example.
- **Split deeper detail** - put API contracts, storage details, extension
  points, error behavior, and host-integration notes in a second page under an
  "In the weeds", "Developers", or "Internals" style section.
- **Keep the first page usable** - include only critical details needed to avoid
  common mistakes; do not make the introductory guide carry maintainer-level
  detail.

> Documentation protocol and examples: `sea-ecosystem/docs/dev/agent-documentation-protocol.md`

## Style rules

- NumPy docstring style for **all** Python callables — public, private (leading underscore), dunder methods, properties, class methods, static methods, and classes. Documentation quality is part of code correctness.
  Required sections: short summary · extended summary · Parameters · Attributes (classes) · Methods (classes) · Returns · Raises · Related · Notes · Examples · References. Include all sections that apply; Parameters, Returns, and Raises are required when they exist.
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
- `src/pySEA/rayTEM/AS2.py`
- `src/pySEA/rayTEM/assemblies.py`
- `src/pySEA/rayTEM/elements.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/01_basicRays.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/02_basicFitting.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/03_lensRotation.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/04_PRIVATE_INSTRUMENT.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/json2sea.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/PLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/builder.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/CLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/diffraction.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/DQCM.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/OLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/PLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/PLs_upstream.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/rederive.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/viewAndCalculate.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/planes.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/shiftvstilt.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/calibrate.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/private_instrument_uncalibrated.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/20260424/ellipsefitting.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/20260424/processsweeplog.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/builder.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/old/Icrit_from_2D.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/old/minima_from_linear.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/old/PLs_2Dsweeps.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/old/PLs_analytical.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/old/PLs_linearsweeps.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/PLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/AS2 XML Parsing/parseXMLstrengths.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/AS2 XML Parsing/TWP-settingsAnalyzer.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/AS2 XML Parsing/xmlutils.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/course_PLs.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/demo.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/fine_PLs/ellipseFitting.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/fine_PLs/findDataFilesWithCaptions.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/liveAS2-v0.01.py`
- `REMOVED_PRIVATE_INSTRUMENT_TREE/PRIVATE_INSTRUMENT/liveAS2.py`
- `src/pySEA/rayTEM/postprocessing.py`
- `src/pySEA/rayTEM/references/andys_functions.py`
- `src/pySEA/rayTEM/seashells.py` — seashells serves as a wrapper around the sea_eco SEASerializable object, enabling easy integration with sea_eco.
- `src/pySEA/rayTEM/tests/test_elements_sections_microscopes.py`
- `src/pySEA/rayTEM/tests/test_wave_and_envelope.py`
- `src/pySEA/rayTEM/utilities.py`
- `src/pySEA/rayTEM/waveoptics.py` — Paraxial scalar wave-optics primitives for rayTEM.
- `src/pySEA/rayTEM/xmlNion.py`

<!-- REPO-MAP-END -->
## Core invariants

- **Ray columns are fixed**: [x, xθ, y, yθ, z, I, E, R] — index 0–7 as returned by `columnByName()`. R is cumulative Larmor rotation accumulated by thick lenses. Do not reorder without updating every Element and all `columnByName` callers.
- **Transfer matrices are 8×8**: Produced by `fix_mat_dims()`. Never pass a raw 2×2 matrix directly to `propagate_ray`.
- **Element → MicroscopeSection → Microscope hierarchy**: Rays propagate strictly bottom-up. Microscope never reaches inside an Element; MicroscopeSection never reaches inside a Microscope.
- **seashells is the sea_eco seam**: All serialization goes through `seashells.SEASerializable`, which gracefully degrades if sea_eco is absent. Do not import from sea_eco directly in framework code.

## High-risk areas

- `elements.py` — `columnByName()` and `fix_mat_dims()` are referenced everywhere; changes break all Elements
- `seashells.py` — conditional import logic; changing import path or attribute names breaks sea_eco round-trips
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
