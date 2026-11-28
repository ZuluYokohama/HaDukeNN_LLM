---
name: verify-all-gates
description: Runs the full HaDukeNN_LLM repository verification gates (canonical priority anchor, control-graph gate coverage, docs budget, mutation score, and pure-stdlib checks). Do not use when running external GPU training sweeps.
---

# Verify All Gates (Plugin Skill)

Run the bundled pure-stdlib gate verification script from the repository workspace root:

```bash
python3 .agents/plugins/hadukenn-sheaf/skills/verify-all-gates/scripts/verify_all.py
```

## Safety

- Strictly read-only against `fixtures/passing/` and `fixtures/test_seal.key`.
- Never write binary weights or bypass failed gates.
