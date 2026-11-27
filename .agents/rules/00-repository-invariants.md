---
trigger: always_on
description: Core repository safety, canonical anchor, licensing, and export-hygiene invariants for HaDukeNN_LLM.
---

# Repository Safety & Export-Hygiene Invariants

- **`fixtures/test_seal.key` is a public test vector and must not change.** Annotated tag `v0.1.0-priority` freezes ledger head `sheaf-verify.anchor.v1 307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4` for that tag's tree. The in-tree `fixtures/passing` may be regenerated only when the ledger record format itself changes; the current head is `sheaf-verify.anchor.v1 e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567`.
- **Preserve Dual-Licensing**: `LICENSE` must remain `PolyForm Noncommercial License 1.0.0` verbatim, `NOTICE` must retain the required `@ZuluYokohama` commercial notice, and `COMMERCIAL_LICENSING.md` documents commercial licensing. Never insert permissive MIT license grant wording into any file.
- **Export Hygiene (`tests/test_docs.py`)**:
  - Never commit binary model checkpoints or arrays (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.bin`).
  - Never write Windows drive-letter paths, private workstation hostnames, cross-project oilfield terms, or external trainer script names inside repository files.
  - Keep `docs/` under `1,000,000` bytes total (`< 300` KiB per `.html` page) with zero external `<script src="...">` tags, and preserve all recorded `G1` benchmark strings tested by `tests/test_docs.py`.
- **Git Workflow**: Never force-push or commit directly to `main`; create an `iter/<name>` branch and open a pull request matching `.github/PULL_REQUEST_TEMPLATE.md`.
