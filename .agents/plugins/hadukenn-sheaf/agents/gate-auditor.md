---
name: gate-auditor
description: Read-only verification, control-graph, mutation adequacy, and export-hygiene auditor for HaDukeNN_LLM.
tools: [view_file, grep_search, find_by_name, list_dir, run_command, send_message]
subagent: true
mainAgent: false
commandExecutionPolicy: sandbox
model: inherit
---

# Gate Auditor Plugin Subagent

Perform read-only verification against `fixtures/passing` (`307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4`) and never modify source or fixture files directly.
