# AI Tools

## The AI wiki slice

rayTEM ships a machine-readable wiki slice for agents: an orientation index,
a layer map with module responsibilities and invariants, per-module method
documentation, and a method index for direct lookup.

From any environment with `raytem` installed:

```python
from importlib.resources import files
index = (files("pySEA") / "ai_wiki" / "raytem" / "index.md").read_text()
method_index = (files("pySEA") / "ai_wiki" / "raytem" / "method-index.json").read_text()
```

Working in the repository, agents start from `CLAUDE.md` / `AGENTS.md` at the
root, which carry the collaboration protocol, the freshness loop
(`pysea-refresh-wiki`), and the core invariants.

## Justified omissions

rayTEM currently provides no MCP server, skills, or subagents of its own —
sea-eco's MCP server covers the data-model side, and this page will grow the
matching subsections if that changes. This note exists so the omission is a
decision on record rather than a gap.
