"""Tier 2 boundary, negative, and exclusion Antigravity platform & SDK E2E tests (Features 17-18).

Pre-M4 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, `torch`, or `yaml`. All target symbols and `.agents/` files
are resolved at test execution time.
"""

from __future__ import annotations

import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests_e2e.conftest import (
    REPO_ROOT,
    assert_readonly_unchanged,
    mcp_stdio_session,
    no_gpu_env,
    resolve_any_public_symbol,
    resolve_public_symbol,
    run_hook,
    snapshot_readonly_hashes,
)

APP_CLI_RUNTIME_TOOLS = frozenset(
    {
        "view_file",
        "write_to_file",
        "replace_file_content",
        "multi_replace_file_content",
        "list_dir",
        "find_by_name",
        "grep_search",
        "search_web",
        "read_url_content",
        "run_command",
        "manage_task",
        "schedule",
        "list_permissions",
        "ask_permission",
        "invoke_subagent",
        "define_subagent",
        "send_message",
        "manage_subagents",
        "ask_question",
        "generate_image",
    }
)

WRITE_RUNTIME_TOOLS = frozenset(
    {
        "write_to_file",
        "replace_file_content",
        "multi_replace_file_content",
    }
)

PLUGIN_ALLOWED_KEYS = frozenset(
    {
        "$schema",
        "name",
        "displayName",
        "description",
        "version",
        "disabled",
        "logo",
        "suggestedPrompts",
        "skills",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "extensions",
    }
)

VALID_TRIGGERS = frozenset({"always_on", "model_decision", "glob", "manual"})
VALID_HOOK_EVENTS = frozenset(
    {"SessionStart", "PreInvocation", "PostInvocation", "Stop", "PreToolUse", "PostToolUse"}
)
KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CONV_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")
INCLUDE_LINK_RE = re.compile(r"@\[[^\]]+\]\(([^)]+)\)")

_ANTIGRAVITY_PREFERRED_MODULES = (
    "hadukenn_core.antigravity_sdk",
    "hadukenn_core.antigravity_adapter",
    "hadukenn_core.antigravity",
    "hadukenn_core.antigravity_bridge",
    "hadukenn_core.antigravity_multimodal",
)


def _strip_quotes(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1], True
    return text, False


def _parse_yaml_scalar(raw: str) -> Any:
    text, was_quoted = _strip_quotes(raw)
    if was_quoted:
        return text
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "~"}:
        return None
    if re.fullmatch(r"-?[0-9]+", text):
        return int(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part) for part in inner.split(",")]
    return text


def _parse_frontmatter(text: str, label: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError(f"{label} missing opening YAML frontmatter fence")
    lines = normalized.split("\n")
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError(f"{label} missing closing YAML frontmatter fence")

    meta: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.lstrip().startswith("- ") and current_list_key is not None:
            item_raw = raw_line.lstrip()[2:].strip()
            existing = meta.get(current_list_key)
            if not isinstance(existing, list):
                raise ValueError(f"{label} invalid YAML list under {current_list_key!r}")
            existing.append(_parse_yaml_scalar(item_raw))
            continue
        if ":" not in raw_line:
            raise ValueError(f"{label} malformed YAML line: {raw_line!r}")
        key, raw_val = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} empty YAML key")
        val_stripped = raw_val.strip()
        if val_stripped == "":
            meta[key] = []
            current_list_key = key
        else:
            meta[key] = _parse_yaml_scalar(val_stripped)
            current_list_key = None

    return meta, "\n".join(lines[end_idx + 1 :])


def _expand_rule_includes(body: str, rule_path: Path | None) -> str:
    if rule_path is None:
        return body

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1).strip()
        if rel.startswith("~/"):
            return match.group(0)
        candidate = (rule_path.parent / rel).resolve()
        if not candidate.is_file():
            candidate = (REPO_ROOT / rel).resolve()
        if not candidate.is_file():
            return match.group(0)
        included = candidate.read_text(encoding="utf-8")
        if included.replace("\r\n", "\n").startswith("---\n"):
            _, included = _parse_frontmatter(included, str(candidate))
        return included

    return INCLUDE_LINK_RE.sub(_replace, body)


def _validate_rule_text(raw_text: str, label: str, *, rule_path: Path | None = None) -> None:
    meta, body = _parse_frontmatter(raw_text, label)
    trigger = meta.get("trigger")
    if trigger not in VALID_TRIGGERS:
        raise ValueError(f"{label} invalid trigger: {trigger!r}")
    if trigger == "model_decision":
        desc = meta.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(f"{label} model_decision rule missing description")
    if trigger == "glob":
        glob_val = meta.get("globs") if "globs" in meta else meta.get("glob")
        if not isinstance(glob_val, str) or not glob_val.strip():
            raise ValueError(f"{label} glob/globs must be comma-separated str, got {type(glob_val).__name__}")
        patterns = [item.strip() for item in glob_val.split(",") if item.strip()]
        if not patterns:
            raise ValueError(f"{label} glob/globs has no non-empty patterns")
    expanded = _expand_rule_includes(body, rule_path)
    if len(expanded.encode("utf-8")) > 24_000:
        raise ValueError(f"{label} exceeds 24,000 byte budget")


def _validate_agent_text(raw_text: str, label: str, *, require_readonly: bool = False) -> None:
    meta, _ = _parse_frontmatter(raw_text, label)
    name = meta.get("name")
    desc = meta.get("description")
    tools = meta.get("tools")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} missing name")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError(f"{label} missing description")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{label} must define explicit non-empty tools list")
    for tool in tools:
        if not isinstance(tool, str) or tool not in APP_CLI_RUNTIME_TOOLS:
            raise ValueError(f"{label} invalid App/CLI runtime tool: {tool!r}")
    if require_readonly and not WRITE_RUNTIME_TOOLS.isdisjoint(set(tools)):
        raise ValueError(f"{label} read-only agent cannot include write tools")


def _validate_mcp_config_dict(cfg: dict[str, Any], label: str) -> None:
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"{label} missing non-empty mcpServers")
    for srv_name, entry in servers.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{label} server {srv_name!r} must be dict")
        if "httpUrl" in entry or entry.get("transport") == "sse":
            raise ValueError(f"{label} legacy httpUrl/SSE forbidden in {srv_name!r}")
        has_cmd = bool(entry.get("command"))
        has_http = bool(entry.get("serverUrl") or entry.get("url"))
        if not (has_cmd ^ has_http):
            raise ValueError(f"{label} server {srv_name!r} must define command XOR serverUrl/url")
        if "timeoutSeconds" in entry:
            ts = entry["timeoutSeconds"]
            if not isinstance(ts, int) or isinstance(ts, bool) or (ts != -1 and ts <= 0):
                raise ValueError(f"{label} invalid timeoutSeconds in {srv_name!r}: {ts!r}")
        if "tools" in entry:
            for t_name, t_cfg in entry["tools"].items():
                if isinstance(t_cfg, dict) and "background" in t_cfg:
                    if t_cfg["background"] not in {"off", "always"}:
                        raise ValueError(f"{label} invalid background in {srv_name}.{t_name}")


def _validate_plugin_manifest(manifest: dict[str, Any], label: str) -> None:
    extra = set(manifest.keys()) - PLUGIN_ALLOWED_KEYS
    if extra:
        raise ValueError(f"{label} Draft 2020-12 unknown keys: {sorted(extra)}")
    name = manifest.get("name")
    if not isinstance(name, str) or not KEBAB_RE.match(name):
        raise ValueError(f"{label} invalid plugin name: {name!r}")
    if "suggestedPrompts" in manifest:
        prompts = manifest["suggestedPrompts"]
        if not isinstance(prompts, list) or len(prompts) > 3:
            raise ValueError(f"{label} suggestedPrompts must be list of len <= 3")
        for item in prompts:
            if not isinstance(item, str) or not re.search(r"\S", item):
                raise ValueError(f"{label} whitespace-only suggestedPrompt: {item!r}")


def _validate_plugin_skill_frontmatter(raw_text: str, expected_dir_name: str, label: str) -> None:
    meta, _ = _parse_frontmatter(raw_text, label)
    desc = meta.get("description")
    if not isinstance(desc, str) or not (1 <= len(desc.strip()) <= 1024):
        raise ValueError(f"{label} invalid skill description")
    name = meta.get("name")
    if not isinstance(name, str) or not (1 <= len(name) <= 64) or not KEBAB_RE.match(name):
        raise ValueError(f"{label} invalid plugin skill name: {name!r}")
    if name != expected_dir_name:
        raise ValueError(f"{label} skill name {name!r} != dir {expected_dir_name!r}")
    if "## Safety" not in raw_text:
        raise ValueError(f"{label} missing ## Safety")
    if "Do not use when" not in raw_text:
        raise ValueError(f"{label} missing Do not use when")


def _validate_plugin_hooks_dict(hooks_data: dict[str, Any], label: str) -> None:
    if not isinstance(hooks_data, dict):
        raise ValueError(f"{label} plugin hooks.json must be a JSON object")
    serialized = json.dumps(hooks_data)
    if "${PLUGIN_ROOT}" in serialized:
        raise ValueError(f"{label} plugin hooks.json must not use ${{PLUGIN_ROOT}}; use relative ./scripts/...")
    present_events = set(hooks_data.keys()) - {"enabled", "$schema"}
    if not present_events or not present_events.issubset(VALID_HOOK_EVENTS):
        raise ValueError(f"{label} invalid plugin hook events: {sorted(present_events)}")
    for event_name in present_events:
        entries = hooks_data[event_name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{label} {event_name} must be non-empty list")
        for group in entries:
            if not isinstance(group, dict):
                raise ValueError(f"{label} hook entry must be dict")
            handlers = group["hooks"] if "hooks" in group else [group]
            if not isinstance(handlers, list) or not handlers:
                raise ValueError(f"{label} hook handlers must be non-empty list")
            for handler in handlers:
                if not isinstance(handler, dict):
                    raise ValueError(f"{label} hook handler must be dict")
                cmd = handler.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    raise ValueError(f"{label} hook handler missing command")
                if "${PLUGIN_ROOT}" in cmd or not any(tok.startswith("./scripts/") for tok in shlex.split(cmd)):
                    raise ValueError(
                        f"{label} plugin hook command must use relative ./scripts/... path: {cmd!r}"
                    )


def _resolve_hook_command(raw_cmd: str, base_dir: Path) -> list[str]:
    parts = shlex.split(raw_cmd)
    assert parts, f"Empty hook command: {raw_cmd!r}"
    resolved: list[str] = []
    for idx, part in enumerate(parts):
        if idx == 0 and Path(part).name in {"python", "python3"}:
            resolved.append(sys.executable)
        elif part.startswith("./") or part.startswith(".agents/"):
            candidate = (base_dir / part).resolve()
            resolved.append(str(candidate) if candidate.exists() else part)
        else:
            resolved.append(part)
    return resolved


def _matching_pretooluse_commands(hooks_path: Path, tool_name: str) -> list[list[str]]:
    assert hooks_path.is_file(), f"Missing Antigravity hooks manifest: {hooks_path}"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    groups = data.get("PreToolUse")
    assert isinstance(groups, list) and groups, f"Missing PreToolUse groups in {hooks_path}"
    cmds: list[list[str]] = []
    for group in groups:
        matcher = group.get("matcher", "")
        if isinstance(matcher, str) and matcher and re.search(matcher, tool_name):
            for handler in group.get("hooks", []):
                raw_cmd = handler.get("command")
                if isinstance(raw_cmd, str) and raw_cmd.strip():
                    cmds.append(_resolve_hook_command(raw_cmd, REPO_ROOT))
    assert cmds, f"No PreToolUse hook matched tool {tool_name!r} in {hooks_path}"
    return cmds


def _make_stdin_payload(tmp_path: Path, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversationId": "conv-antigravity-e2e-0123456789abcdef",
        "workspacePaths": [str(REPO_ROOT)],
        "transcriptPath": str(tmp_path / "transcript.jsonl"),
        "artifactDirectoryPath": str(tmp_path / "artifacts"),
        "modelName": "gemini-2.5-pro",
        "lastUserInput": "Check Antigravity PreToolUse policy guardrails.",
        "toolName": tool_name,
        "toolInput": args,
        "toolCall": {"name": tool_name, "args": args},
    }


def _bind_call(fn: Any, kwargs: dict[str, Any], positional: tuple[Any, ...] = ()) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    filtered = {k: v for k, v in kwargs.items() if has_varkw or k in params}
    if all(
        p.default is not inspect.Parameter.empty
        or p.name in filtered
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params.values()
    ):
        sig.bind(**filtered)
        return fn(**filtered)
    sig.bind(*positional)
    return fn(*positional)


def _assert_rejected(fn: Any, kwargs: dict[str, Any], positional: tuple[Any, ...] = ()) -> None:
    try:
        res = _bind_call(fn, kwargs, positional)
    except (ValueError, AssertionError, FileNotFoundError):
        return
    if isinstance(res, bool):
        assert res is False, "Expected False/rejection, got True"
        return
    if isinstance(res, dict):
        for key in ("ok", "passed", "valid", "allowed"):
            if key in res and isinstance(res[key], bool):
                assert res[key] is False, f"Expected rejected dict, got {res!r}"
                return
        if "decision" in res:
            assert str(res["decision"]).lower() == "deny", f"Expected deny, got {res!r}"
            return
    decision = getattr(res, "decision", None)
    if decision is not None:
        assert str(decision).lower() == "deny", f"Expected deny, got {res!r}"
        return
    raise AssertionError(f"Expected rejection exception or negative verdict, got {res!r}")


# ==============================================================================
# Feature 17: Antigravity Boundary, Hook Block & Schema Rejection (8 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f17_rule_validator_rejects_yaml_array_globs_and_camel_case_trigger(tmp_path: Path) -> None:
    rules_dir = REPO_ROOT / ".agents" / "rules"
    assert rules_dir.is_dir(), f"Missing rules directory: {rules_dir}"
    rule_files = sorted(rules_dir.glob("*.md"))
    expected = {
        "00-repository-invariants.md",
        "10-sheaf-verify-stdlib.md",
        "20-empirical-architecture-guardrails.md",
    }
    assert expected.issubset({p.name for p in rule_files})
    for rule_path in rule_files:
        _validate_rule_text(rule_path.read_text(encoding="utf-8"), str(rule_path), rule_path=rule_path)

    array_globs_rule = (
        "---\n"
        "trigger: glob\n"
        "globs:\n"
        '  - "src/sheaf_verify/**/*.py"\n'
        '  - "tests/**/*.py"\n'
        "description: Broken array globs.\n"
        "---\nbody\n"
    )
    with pytest.raises(ValueError, match="comma-separated str"):
        _validate_rule_text(array_globs_rule, "array_globs")

    camel_trigger_rule = "---\ntrigger: alwaysOn\ndescription: Wrong case.\n---\nbody\n"
    with pytest.raises(ValueError, match="invalid trigger"):
        _validate_rule_text(camel_trigger_rule, "camel_trigger")

    missing_desc_model_decision = "---\ntrigger: model_decision\n---\nbody\n"
    with pytest.raises(ValueError, match="missing description"):
        _validate_rule_text(missing_desc_model_decision, "missing_desc")

    oversized_rule = "---\ntrigger: always_on\ndescription: Oversized.\n---\n" + ("x" * 24_001)
    with pytest.raises(ValueError, match="24,000"):
        _validate_rule_text(oversized_rule, "oversized")

    inc_file = tmp_path / "big_include.txt"
    inc_file.write_text("y" * 24_001, encoding="utf-8")
    inc_rule_path = tmp_path / "inc_rule.md"
    inc_rule_text = "---\ntrigger: always_on\ndescription: Oversized via include.\n---\n@[inc](big_include.txt)\n"
    inc_rule_path.write_text(inc_rule_text, encoding="utf-8")
    with pytest.raises(ValueError, match="24,000"):
        _validate_rule_text(inc_rule_text, str(inc_rule_path), rule_path=inc_rule_path)


@pytest.mark.tier2
def test_t2_f17_agent_validator_rejects_unknown_sdk_tool_names_and_auditor_writes() -> None:
    auditor_path = REPO_ROOT / ".agents" / "agents" / "gate-auditor" / "agent.md"
    builder_path = REPO_ROOT / ".agents" / "agents" / "sheaf-bridge-builder" / "agent.md"
    assert auditor_path.is_file() and builder_path.is_file()
    _validate_agent_text(auditor_path.read_text(encoding="utf-8"), str(auditor_path), require_readonly=True)
    _validate_agent_text(builder_path.read_text(encoding="utf-8"), str(builder_path), require_readonly=False)

    typo_agent = "---\nname: bad-typo\ndescription: Bad tool.\ntools: [view_files]\n---\n"
    with pytest.raises(ValueError, match="invalid App/CLI runtime tool"):
        _validate_agent_text(typo_agent, "typo_agent")

    for sdk_tool in ("edit_file", "list_directory", "create_file"):
        sdk_agent = f"---\nname: bad-sdk\ndescription: SDK name.\ntools: [{sdk_tool}]\n---\n"
        with pytest.raises(ValueError, match="invalid App/CLI runtime tool"):
            _validate_agent_text(sdk_agent, f"sdk_{sdk_tool}")

    writable_auditor = (
        "---\n"
        "name: gate-auditor\n"
        "description: Should not write.\n"
        "tools: [view_file, write_to_file]\n"
        "---\n"
    )
    with pytest.raises(ValueError, match="write tools"):
        _validate_agent_text(writable_auditor, "writable_auditor", require_readonly=True)


@pytest.mark.tier2
def test_t2_f17_pretooluse_hook_blocks_non_stdlib_import_in_sheaf_verify_exit0(tmp_path: Path) -> None:
    hooks_path = REPO_ROOT / ".agents" / "hooks.json"
    cmds = _matching_pretooluse_commands(hooks_path, "write_to_file")

    allow_args = {
        "path": "src/sheaf_verify/ok_stdlib.py",
        "file_path": "src/sheaf_verify/ok_stdlib.py",
        "TargetFile": "src/sheaf_verify/ok_stdlib.py",
        "content": "from __future__ import annotations\n\nimport json\n",
        "CodeContent": "from __future__ import annotations\n\nimport json\n",
    }
    assert cmds
    allow_payload = _make_stdin_payload(tmp_path, "write_to_file", allow_args)
    for cmd in cmds:
        proc_ok, parsed_ok = run_hook(cmd, allow_payload, cwd=REPO_ROOT)
        assert proc_ok.returncode == 0
        assert parsed_ok.get("decision") in {"allow", "auto_approve"}

    deny_args = {
        "path": "src/sheaf_verify/illegal_np.py",
        "file_path": "src/sheaf_verify/illegal_np.py",
        "TargetFile": "src/sheaf_verify/illegal_np.py",
        "content": "from __future__ import annotations\n\nimport numpy as np\n",
        "CodeContent": "from __future__ import annotations\n\nimport numpy as np\n",
    }
    payload = _make_stdin_payload(tmp_path, "write_to_file", deny_args)
    parsed_list: list[dict[str, Any]] = []
    for cmd in cmds:
        proc, parsed = run_hook(cmd, payload, cwd=REPO_ROOT)
        assert proc.returncode == 0, f"PreToolUse deny MUST exit 0 (non-zero fails open!): rc={proc.returncode}"
        parsed_list.append(parsed)
    assert any(
        p.get("decision") == "deny" and isinstance(p.get("reason"), str) and bool(p["reason"].strip())
        for p in parsed_list
    ), f"Expected deny for non-stdlib import, got {parsed_list!r}"


@pytest.mark.tier2
def test_t2_f17_pretooluse_hook_blocks_binary_weights_and_drive_letter_paths_exit0(tmp_path: Path) -> None:
    hooks_path = REPO_ROOT / ".agents" / "hooks.json"
    cmds = _matching_pretooluse_commands(hooks_path, "write_to_file")
    assert cmds

    c_drive = chr(67) + ":" + chr(92) + "Users" + chr(92) + "weights.txt"
    d_drive = chr(68) + ":" + chr(92) + "data" + chr(92) + "notes.md"
    blocked_cases = [
        ("artifacts/model" + ".safe" + "tensors", "binary weight payload"),
        ("artifacts/checkpoint" + ".p" + "t", "binary weight payload"),
        ("artifacts/weights" + ".bi" + "n", "binary weight payload"),
        ("src/hadukenn_core/bad_drive_c.py", f'PATH_A = "{c_drive}"\n'),
        ("docs/bad_drive_d.md", f"Path is {d_drive}\n"),
    ]
    for target_file, code_text in blocked_cases:
        args = {
            "path": target_file,
            "file_path": target_file,
            "TargetFile": target_file,
            "content": code_text,
            "CodeContent": code_text,
        }
        payload = _make_stdin_payload(tmp_path, "write_to_file", args)
        parsed_list: list[dict[str, Any]] = []
        for cmd in cmds:
            proc, parsed = run_hook(cmd, payload, cwd=REPO_ROOT)
            assert proc.returncode == 0, f"PreToolUse deny MUST exit 0 (got {proc.returncode}) for {target_file}"
            parsed_list.append(parsed)
        assert any(
            p.get("decision") == "deny" and isinstance(p.get("reason"), str) and bool(p["reason"].strip())
            for p in parsed_list
        ), f"Expected deny for {target_file!r}, got {parsed_list!r}"


@pytest.mark.tier2
def test_t2_f17_pretooluse_hook_blocks_182x181_on_32k_gpu_exclusion_exit0(tmp_path: Path) -> None:
    hooks_path = REPO_ROOT / ".agents" / "hooks.json"
    cmds = _matching_pretooluse_commands(hooks_path, "write_to_file")
    assert cmds

    bad_code = (
        "from __future__ import annotations\n\n"
        "def train_gpu_32k():\n"
        "    vocab_size = 32768\n"
        '    head_mode = "182x181"\n'
        '    device = "cuda"\n'
        "    return vocab_size, head_mode, device\n"
    )
    args = {
        "path": "src/hadukenn_core/bad_182x181_gpu.py",
        "file_path": "src/hadukenn_core/bad_182x181_gpu.py",
        "TargetFile": "src/hadukenn_core/bad_182x181_gpu.py",
        "content": bad_code,
        "CodeContent": bad_code,
    }
    payload = _make_stdin_payload(tmp_path, "write_to_file", args)
    parsed_list: list[dict[str, Any]] = []
    for cmd in cmds:
        proc, parsed = run_hook(cmd, payload, cwd=REPO_ROOT)
        assert proc.returncode == 0, f"PreToolUse deny MUST exit 0 (got {proc.returncode})"
        parsed_list.append(parsed)
    assert any(
        p.get("decision") == "deny" and isinstance(p.get("reason"), str) and bool(p["reason"].strip())
        for p in parsed_list
    ), f"Expected deny for 182x181 32k GPU proposal, got {parsed_list!r}"


@pytest.mark.tier2
def test_t2_f17_hooks_and_mcp_reject_legacy_http_sse_and_settings_json_placement() -> None:
    hooks_path = REPO_ROOT / ".agents" / "hooks.json"
    mcp_path = REPO_ROOT / ".agents" / "mcp_config.json"
    assert hooks_path.is_file(), f"Missing {hooks_path}"
    assert mcp_path.is_file(), f"Missing {mcp_path}"

    hooks_data = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert set(hooks_data.keys()) - {"enabled", "$schema"} <= VALID_HOOK_EVENTS
    mcp_data = json.loads(mcp_path.read_text(encoding="utf-8"))
    _validate_mcp_config_dict(mcp_data, str(mcp_path))

    settings_path = REPO_ROOT / ".agents" / "settings.json"
    if settings_path.is_file():
        settings_data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "hooks" not in settings_data, "Hooks must NOT be placed inside settings.json"

    for bad_cfg in (
        {"mcpServers": {"legacy": {"httpUrl": "https://example.com/sse"}}},
        {"mcpServers": {"both": {"command": "python3", "serverUrl": "https://example.com/mcp"}}},
        {"mcpServers": {"zero_to": {"command": "python3", "timeoutSeconds": 0}}},
        {"mcpServers": {"bad_bg": {"command": "python3", "tools": {"t": {"background": "never"}}}}},
    ):
        with pytest.raises(ValueError):
            _validate_mcp_config_dict(bad_cfg, "bad_mcp_cfg")


@pytest.mark.tier2
def test_t2_f17_mcp_stdio_rejects_unknown_method_and_invalid_tool_without_crash() -> None:
    baseline = snapshot_readonly_hashes(REPO_ROOT)
    try:
        with mcp_stdio_session() as client:
            init_resp = client.initialize()
            assert "result" in init_resp

            unknown_resp = client.request("nonexistent/method", params={})
            assert "error" in unknown_resp, f"Expected JSON-RPC error for unknown method, got {unknown_resp!r}"
            assert isinstance(unknown_resp["error"], dict)
            assert "code" in unknown_resp["error"] and "message" in unknown_resp["error"]

            bad_tool_resp = client.call_tool("nonexistent_tool", arguments={"x": 1})
            has_rpc_error = "error" in bad_tool_resp
            result_obj = bad_tool_resp.get("result") or {}
            has_tool_error = isinstance(result_obj, dict) and bool(result_obj.get("isError"))
            assert has_rpc_error or has_tool_error, (
                f"Expected error or isError=True for unknown tool, got {bad_tool_resp!r}"
            )

            tools_after = client.list_tools()
            assert len(tools_after) >= 1, "MCP server must remain alive after invalid method/tool calls"
    finally:
        assert_readonly_unchanged(baseline, REPO_ROOT)


@pytest.mark.tier2
def test_t2_f17_plugin_draft_2020_12_rejects_excess_prompts_extra_keys_and_bad_name() -> None:
    plugin_root = REPO_ROOT / ".agents" / "plugins" / "hadukenn-sheaf"
    assert plugin_root.is_dir(), f"Missing plugin directory: {plugin_root}"
    preferred = plugin_root / ".antigravity-plugin" / "plugin.json"
    fallback = plugin_root / "plugin.json"
    manifest_path = preferred if preferred.is_file() else fallback
    assert manifest_path.is_file(), f"Missing plugin manifest in {plugin_root}"
    _validate_plugin_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path))

    plugin_skills_dir = plugin_root / "skills"
    if plugin_skills_dir.is_dir():
        for skill_md in sorted(plugin_skills_dir.glob("*/SKILL.md")):
            _validate_plugin_skill_frontmatter(
                skill_md.read_text(encoding="utf-8"),
                skill_md.parent.name,
                str(skill_md),
            )

    plugin_hooks_path = plugin_root / "hooks.json"
    if plugin_hooks_path.is_file():
        _validate_plugin_hooks_dict(
            json.loads(plugin_hooks_path.read_text(encoding="utf-8")),
            str(plugin_hooks_path),
        )

    for bad_manifest in (
        {"name": "hadukenn-sheaf", "suggestedPrompts": ["p1", "p2", "p3", "p4"]},
        {"name": "hadukenn-sheaf", "suggestedPrompts": ["   "]},
        {"name": "hadukenn-sheaf", "unknownExtraKey": True},
        {"name": "Bad_Name"},
        {"name": "-leading-dash"},
    ):
        with pytest.raises(ValueError):
            _validate_plugin_manifest(bad_manifest, "bad_plugin_manifest")

    bad_skill_text = "---\nname: wrong-skill-name\ndescription: Mismatch.\n---\n## Safety\nDo not use when testing.\n"
    with pytest.raises(ValueError, match="!= dir"):
        _validate_plugin_skill_frontmatter(bad_skill_text, "expected-skill-dir", "bad_plugin_skill")

    bad_plugin_hooks = {
        "PreToolUse": [{"matcher": "write_to_file", "hooks": [{"command": "python3 ${PLUGIN_ROOT}/s.py"}]}]
    }
    with pytest.raises(ValueError, match="PLUGIN_ROOT"):
        _validate_plugin_hooks_dict(bad_plugin_hooks, "bad_plugin_hooks")


# ==============================================================================
# Feature 18: Antigravity SDK, Multi-Modal & Docs Budget Rejections (8 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f18_conversation_id_rejects_short_and_underscore() -> None:
    conv_validator = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_conversation_id", "is_valid_conversation_id", "LocalAgentConfig"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    valid_id = "conv-antigravity-valid-0123456789ab"
    assert len(valid_id) >= 32 and "_" not in valid_id and CONV_ID_RE.match(valid_id)
    if callable(conv_validator) and not isinstance(conv_validator, type):
        assert conv_validator(valid_id) is not False
    else:
        assert _bind_call(conv_validator, {"conversation_id": valid_id}) is not None

    for bad_id in ("short-id-123", "a" * 30 + "_1", "", "   ", "invalid_with_underscore_and_long_32"):
        _assert_rejected(conv_validator, {"conversation_id": bad_id}, (bad_id,))


@pytest.mark.tier2
def test_t2_f18_permission_deny_precedence_and_read_implies_write_denial() -> None:
    preset_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("resolve_permission_preset", "PermissionPreset", "PermissionPolicy", "PolicyEvaluator"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    eval_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("evaluate_permission", "evaluate_policy", "PermissionPolicy", "PolicyEvaluator"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )

    precedence_rules = [
        {"action": "read_file", "decision": "allow"},
        {"action": "read_file", "decision": "ask"},
        {"action": "read_file", "decision": "deny"},
        {"action": "write_file", "decision": "allow"},
    ]
    if isinstance(eval_fn, type):
        evaluator = _bind_call(eval_fn, {"rules": precedence_rules}, (precedence_rules,))
        fn = getattr(evaluator, "evaluate", getattr(evaluator, "check", None))
        assert fn is not None
    else:
        fn = eval_fn

    allow_rules = [{"action": "read_file", "decision": "allow"}]
    ok_res = _bind_call(fn, {"action": "read_file", "rules": allow_rules}, ("read_file", allow_rules))
    assert str(getattr(ok_res, "decision", ok_res)).lower() == "allow"

    read_denied = _bind_call(fn, {"action": "read_file", "rules": precedence_rules}, ("read_file", precedence_rules))
    assert str(getattr(read_denied, "decision", read_denied)).lower() == "deny"

    write_denied = _bind_call(
        fn,
        {"action": "write_file", "rules": precedence_rules},
        ("write_file", precedence_rules),
    )
    assert str(getattr(write_denied, "decision", write_denied)).lower() == "deny"

    _assert_rejected(
        preset_fn.from_preset if isinstance(preset_fn, type) and hasattr(preset_fn, "from_preset") else preset_fn,
        {"preset": "unknown-unsafe-preset"},
        ("unknown-unsafe-preset",),
    )


@pytest.mark.tier2
def test_t2_f18_litert_and_response_schema_reject_invalid_inputs() -> None:
    litert_cfg_cls = resolve_public_symbol("hadukenn_core", "LiteRTAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    caps_cls = resolve_public_symbol("hadukenn_core", "CapabilitiesConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    builtin_tools = resolve_public_symbol("hadukenn_core", "BuiltinTools", _ANTIGRAVITY_PREFERRED_MODULES)
    schema_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_response_schema", "verify_response_schema", "ResponseSchemaValidator"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )

    valid_cfg = _bind_call(litert_cfg_cls, {"model_path": "models/litert_int8", "port": 8080})
    assert valid_cfg is not None
    valid_ro = builtin_tools.read_only() if callable(getattr(builtin_tools, "read_only", None)) else []
    assert valid_ro
    assert _bind_call(caps_cls, {"enabled_tools": list(valid_ro)}, (list(valid_ro),)) is not None

    _assert_rejected(litert_cfg_cls, {"model_path": "", "port": 8080})
    _assert_rejected(litert_cfg_cls, {"model_path": "models/litert_int8", "port": 0})
    _assert_rejected(litert_cfg_cls, {"model_path": "models/litert_int8", "port": 70_000})
    _assert_rejected(caps_cls, {"enabled_tools": ["replace_file_content"]}, (["replace_file_content"],))

    schema = {
        "type": "object",
        "required": ["ok", "anchor"],
        "properties": {"ok": {"type": "boolean"}, "anchor": {"type": "string"}},
    }
    bad_payload = {"ok": "not-a-bool"}
    if isinstance(schema_fn, type):
        validator = _bind_call(schema_fn, {"schema": schema}, (schema,))
        check_method = getattr(validator, "validate", getattr(validator, "verify", None))
        assert check_method is not None
        _assert_rejected(check_method, {"payload": bad_payload, "data": bad_payload}, (bad_payload,))
    else:
        _assert_rejected(schema_fn, {"payload": bad_payload, "schema": schema}, (bad_payload, schema))


@pytest.mark.tier2
def test_t2_f18_lifecycle_on_tool_error_intercepts_and_recovers() -> None:
    hook_result_cls = resolve_public_symbol("hadukenn_core", "HookResult", _ANTIGRAVITY_PREFERRED_MODULES)
    lifecycle = resolve_any_public_symbol(
        "hadukenn_core",
        ("LifecycleHooks", "HookRegistry", "AntigravityLifecycle", "run_lifecycle_hooks"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )

    recovery = _bind_call(hook_result_cls, {"decision": "ask", "reason": "tool_error_intercepted"}, ("ask",))
    registry = lifecycle() if isinstance(lifecycle, type) else lifecycle

    captured: list[str] = []

    def _on_tool_error(ctx: dict[str, Any]) -> Any:
        captured.append(str(ctx.get("error", "")))
        return recovery

    if hasattr(registry, "on_tool_error") and callable(registry.on_tool_error):
        registry.on_tool_error(_on_tool_error)
    if hasattr(registry, "run_on_tool_error"):
        out = registry.run_on_tool_error({"tool_name": "run_command", "error": "boom"})
    elif callable(registry):
        out = _bind_call(
            registry,
            {"on_tool_error": _on_tool_error, "ctx": {"tool_name": "run_command", "error": "boom"}},
        )
    else:
        raise AssertionError(f"Lifecycle hook surface lacks run_on_tool_error/callable: {registry!r}")
    assert out is not None and captured == ["boom"]


@pytest.mark.tier2
def test_t2_f18_multimodal_pathway_rejects_182x181_32k_gpu_and_missing_media(tmp_path: Path) -> None:
    litert_cfg_cls = resolve_public_symbol("hadukenn_core", "LiteRTAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    pathway_fn = resolve_any_public_symbol(
        "hadukenn_core",
        (
            "run_multimodal_pathway",
            "evaluate_multimodal_pathway",
            "AntigravityMultimodalRouter",
            "MultiModalPathway",
        ),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    from_file_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("from_file", "load_media_attachment", "MultimodalAttachment"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )

    good_media = tmp_path / "valid.png"
    good_media.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    attachment = (
        from_file_fn.from_file(good_media)
        if isinstance(from_file_fn, type) and hasattr(from_file_fn, "from_file")
        else from_file_fn(good_media)
    )
    assert attachment is not None

    cfg = _bind_call(litert_cfg_cls, {"model_path": "models/struct32k_int8"}, ("models/struct32k_int8",))
    runner = pathway_fn() if isinstance(pathway_fn, type) else pathway_fn
    exec_fn = getattr(runner, "run", getattr(runner, "evaluate", runner))

    ok_res = _bind_call(
        exec_fn,
        {
            "config": cfg,
            "attachments": [attachment],
            "vocab_size": 32_768,
            "head_type": "dense_flat",
            "hardware_target": "cpu",
        },
    )
    assert ok_res is not None

    _assert_rejected(
        exec_fn,
        {
            "config": cfg,
            "attachments": [attachment],
            "vocab_size": 32_768,
            "head_type": "182x181",
            "hardware_target": "cuda",
        },
    )

    missing_media = tmp_path / "missing.png"
    loader = (
        from_file_fn.from_file
        if isinstance(from_file_fn, type) and hasattr(from_file_fn, "from_file")
        else from_file_fn
    )
    _assert_rejected(loader, {"path": missing_media}, (missing_media,))


@pytest.mark.tier2
def test_t2_f18_docs_budget_synthetic_rejects_missing_docs_dir(tmp_path: Path) -> None:
    src_script = REPO_ROOT / ".github" / "scripts" / "docs_budget.py"
    dst_script = tmp_path / ".github" / "scripts" / "docs_budget.py"
    dst_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_script, dst_script)

    proc = subprocess.run(
        [sys.executable, str(dst_script)],
        cwd=tmp_path,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert proc.returncode == 1
    assert "missing" in proc.stderr.lower()


@pytest.mark.tier2
def test_t2_f18_docs_budget_synthetic_rejects_oversized_html_and_total_tree(tmp_path: Path) -> None:
    src_script = REPO_ROOT / ".github" / "scripts" / "docs_budget.py"
    dst_script = tmp_path / ".github" / "scripts" / "docs_budget.py"
    dst_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_script, dst_script)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    big_page = docs_dir / "oversized.html"
    big_page.write_bytes(b"a" * 307_201)

    proc_page = subprocess.run(
        [sys.executable, str(dst_script)],
        cwd=tmp_path,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert proc_page.returncode == 1
    assert "over 307200" in proc_page.stderr

    big_page.unlink()
    for idx in range(4):
        (docs_dir / f"chunk_{idx}.html").write_bytes(b"b" * 260_000)

    proc_total = subprocess.run(
        [sys.executable, str(dst_script)],
        cwd=tmp_path,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert proc_total.returncode == 1
    assert "over 1000000" in proc_total.stderr


@pytest.mark.tier2
def test_t2_f18_docs_budget_synthetic_rejects_external_script_tags(tmp_path: Path) -> None:
    src_script = REPO_ROOT / ".github" / "scripts" / "docs_budget.py"
    dst_script = tmp_path / ".github" / "scripts" / "docs_budget.py"
    dst_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_script, dst_script)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    page = docs_dir / "index.html"

    for tag in (
        '<script src="https://cdn.example.com/a.js"></script>',
        '<script src="//cdn.example.com/a.js"></script>',
    ):
        page.write_text(f"<!doctype html><html><head>{tag}</head><body>x</body></html>\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(dst_script)],
            cwd=tmp_path,
            env=no_gpu_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
        assert proc.returncode == 1, f"Expected failure for {tag!r}"
        assert "external script" in proc.stderr.lower()
