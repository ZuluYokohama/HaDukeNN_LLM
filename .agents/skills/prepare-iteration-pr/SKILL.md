---
name: prepare-iteration-pr
description: Audits export hygiene, docs budget, canonical priority anchor, and pull-request template completeness before creating a Git branch or pull request for ZuluYokohama/HaDukeNN_LLM. Do not use when pushing directly to main or upload external credentials.
---

# Prepare Iteration Pull Request (`/prepare-iteration-pr`)

Use this skill whenever preparing an iteration branch (`iter/<name>`) to open a pull request on `ZuluYokohama/HaDukeNN_LLM`.

## Workflow

1. Run the automated PR hygiene & export scanner:
   ```bash
   python3 .agents/skills/prepare-iteration-pr/scripts/check_pr_hygiene.py
   ```
2. Run `/verify-all-gates` to ensure all unit tests, coverage, control-graph traces, and mutant checks pass.
3. Populate every required section of `.github/PULL_REQUEST_TEMPLATE.md`:
   - `Gates touched`
   - `Evidence files`
   - `Ledger head before` (`307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4`)
   - `Ledger head after` (`307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4` when code-only)
   - Explicit confirmation on `amendment` status and zero binary `weights`.

## Safety

- Never run `git push --force` or push directly to `main`.
- Always confirm with the user before pushing branches or opening pull requests on external remotes.
- Scripts in `scripts/` inspect local files read-only and never execute network uploads.
