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
   - `Ledger head before` (`e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567`)
   - `Ledger head after` (`e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567` when code-only)
   - Explicit confirmation on `amendment` status and zero binary `weights`.

## Safety

- Never run `git push --force` or push directly to `main`.
- Always confirm with the user before pushing branches or opening pull requests on external remotes.
- Scripts in `scripts/` inspect local files read-only and never execute network uploads.
