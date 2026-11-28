"""Tier 1 isolated happy-path Antigravity platform & SDK integration E2E tests (Features 17-18).

Pre-M4 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, `torch`, or `yaml`. All target symbols and `.agents/` files
are resolved at test execution time.
"""

from __future__ import annotations

import ast
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

CANONICAL_ANCHOR = "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"

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

SDK_BUILTIN_TOOL_NAMES = frozenset(
    {
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "create_file",
        "edit_file",
        "run_command",
        "ask_question",
        "start_subagent",
        "generate_image",
        "search_web",
        "read_url_content",
        "finish",
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
VALID_PRETOOL_DECISIONS = frozenset(
    {"allow", "auto_approve", "deny", "ask", "force_ask", "deny_unless_prior_grant"}
)
VALID_COMMAND_POLICIES = frozenset({"off", "auto", "eager", "sandbox"})
VALID_RESTART_POLICIES = frozenset({"always", "on-failure", "never"})
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


def _parse_frontmatter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    assert normalized.startswith("---\n"), f"{source} missing opening YAML frontmatter fence"
    lines = normalized.split("\n")
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    assert end_idx is not None, f"{source} missing closing YAML frontmatter fence"

    meta: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.lstrip().startswith("- ") and current_list_key is not None:
            item_raw = raw_line.lstrip()[2:].strip()
            existing = meta.get(current_list_key)
            assert isinstance(existing, list), f"{source} invalid YAML list under {current_list_key!r}"
            existing.append(_parse_yaml_scalar(item_raw))
            continue
        assert ":" in raw_line, f"{source} malformed YAML line: {raw_line!r}"
        key, raw_val = raw_line.split(":", 1)
        key = key.strip()
        assert key, f"{source} empty YAML key in {raw_line!r}"
        val_stripped = raw_val.strip()
        if val_stripped == "":
            meta[key] = []
            current_list_key = key
        else:
            meta[key] = _parse_yaml_scalar(val_stripped)
            current_list_key = None

    body = "\n".join(lines[end_idx + 1 :])
    return meta, body


def _expand_rule_includes(body: str, rule_path: Path) -> str:
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
            _, included = _parse_frontmatter(included, candidate)
        return included

    return INCLUDE_LINK_RE.sub(_replace, body)


def _check_python_ast_and_stdlib(script_path: Path, *, allow_sheaf_verify: bool = True) -> None:
    source = script_path.read_text(encoding="utf-8")
    assert "\t" not in source, f"Tab indentation forbidden in {script_path}"
    tree = ast.parse(source, filename=str(script_path))
    allowed = set(sys.stdlib_module_names)
    if allow_sheaf_verify:
        allowed.add("sheaf_verify")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                assert top in allowed, f"Non-stdlib import {top!r} in {script_path}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".", 1)[0]
            assert top in allowed, f"Non-stdlib import {top!r} in {script_path}"


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
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    groups = data.get("PreToolUse")
    assert isinstance(groups, list) and groups, f"Missing PreToolUse groups in {hooks_path}"
    cmds: list[list[str]] = []
    for group in groups:
        assert isinstance(group, dict)
        matcher = group.get("matcher", "")
        assert isinstance(matcher, str) and matcher
        if re.search(matcher, tool_name):
            for handler in group.get("hooks", []):
                assert isinstance(handler, dict)
                raw_cmd = handler.get("command")
                assert isinstance(raw_cmd, str) and raw_cmd.strip()
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
        "lastUserInput": "Run deterministic Antigravity E2E verification.",
        "toolName": tool_name,
        "toolInput": args,
        "toolCall": {"name": tool_name, "args": args},
    }


def _validate_plugin_skill(skill_md: Path, expected_dir_name: str) -> None:
    content = skill_md.read_text(encoding="utf-8")
    meta, _ = _parse_frontmatter(content, skill_md)
    desc = meta.get("description")
    assert isinstance(desc, str) and 1 <= len(desc.strip()) <= 1024, f"{skill_md} bad description"
    name = meta.get("name")
    assert isinstance(name, str) and 1 <= len(name) <= 64 and KEBAB_RE.match(name), f"{skill_md} bad name"
    assert name == expected_dir_name, f"{skill_md} name {name!r} != dir {expected_dir_name!r}"
    assert "## Safety" in content, f"{skill_md} missing ## Safety"
    assert "Do not use when" in content, f"{skill_md} missing Do not use when"


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
        return fn(**filtered)
    return fn(*positional)


# ==============================================================================
# Feature 17: Antigravity .agents/ Artifacts, Hooks, MCP, Plugin & Sidecar (8 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f17_rules_valid_frontmatter_comma_globs_and_24kb_budget() -> None:
    rules_dir = REPO_ROOT / ".agents" / "rules"
    assert rules_dir.is_dir(), f"Missing rules directory: {rules_dir}"
    rule_files = sorted(rules_dir.glob("*.md"))
    assert len(rule_files) >= 3, f"Expected >= 3 rules in {rules_dir}"

    expected = {
        "00-repository-invariants.md",
        "10-sheaf-verify-stdlib.md",
        "20-empirical-architecture-guardrails.md",
    }
    assert expected.issubset({p.name for p in rule_files})

    for rule_path in rule_files:
        raw = rule_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw, rule_path)
        trigger = meta.get("trigger")
        assert trigger in VALID_TRIGGERS, f"{rule_path} invalid trigger: {trigger!r}"
        if trigger == "model_decision":
            desc = meta.get("description")
            assert isinstance(desc, str) and desc.strip(), f"{rule_path} missing model_decision description"
        if trigger == "glob":
            glob_val = meta.get("globs") if "globs" in meta else meta.get("glob")
            assert isinstance(glob_val, str), (
                f"{rule_path} glob/globs MUST be comma-separated str, got {type(glob_val).__name__}: {glob_val!r}"
            )
            patterns = [item.strip() for item in glob_val.split(",") if item.strip()]
            assert patterns, f"{rule_path} empty glob patterns"
        expanded = _expand_rule_includes(body, rule_path)
        post_include_bytes = len(expanded.encode("utf-8"))
        assert post_include_bytes <= 24_000, f"{rule_path} exceeded 24KB budget: {post_include_bytes}"


@pytest.mark.tier1
def test_t1_f17_skills_frontmatter_safety_markers_and_scripts() -> None:
    skills_dir = REPO_ROOT / ".agents" / "skills"
    assert skills_dir.is_dir(), f"Missing skills directory: {skills_dir}"
    skill_mds = sorted(skills_dir.glob("*/SKILL.md"))
    assert len(skill_mds) >= 3

    for skill_md in skill_mds:
        raw = skill_md.read_text(encoding="utf-8")
        meta, _ = _parse_frontmatter(raw, skill_md)
        desc = meta.get("description")
        assert isinstance(desc, str) and 1 <= len(desc.strip()) <= 1024, f"{skill_md} bad description"
        if "name" in meta:
            name = meta["name"]
            assert isinstance(name, str) and 1 <= len(name) <= 64 and KEBAB_RE.match(name), (
                f"{skill_md} invalid kebab-case name: {name!r}"
            )
        if "disable-model-invocation" in meta:
            assert isinstance(meta["disable-model-invocation"], bool)
        assert "## Safety" in raw, f"{skill_md} missing ## Safety"
        assert "Do not use when" in raw, f"{skill_md} missing boundary clause"

        scripts_dir = skill_md.parent / "scripts"
        if scripts_dir.is_dir():
            py_scripts = sorted(scripts_dir.glob("*.py"))
            assert py_scripts, f"{scripts_dir} has no .py scripts"
            for script in py_scripts:
                ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


@pytest.mark.tier1
def test_t1_f17_agents_runtime_tool_allowlist_and_gate_auditor_readonly() -> None:
    agents_dir = REPO_ROOT / ".agents" / "agents"
    assert agents_dir.is_dir(), f"Missing agents directory: {agents_dir}"
    agent_files = sorted([*agents_dir.glob("*.md"), *agents_dir.glob("*/agent.md")])
    assert len(agent_files) >= 2

    parsed_by_name: dict[str, dict[str, Any]] = {}
    for agent_md in agent_files:
        meta, _ = _parse_frontmatter(agent_md.read_text(encoding="utf-8"), agent_md)
        name = meta.get("name")
        desc = meta.get("description")
        tools = meta.get("tools")
        assert isinstance(name, str) and name.strip(), f"{agent_md} missing name"
        assert isinstance(desc, str) and desc.strip(), f"{agent_md} missing description"
        assert isinstance(tools, list) and tools, f"{agent_md} must define explicit non-empty tools allowlist"
        for tool in tools:
            assert isinstance(tool, str) and tool in APP_CLI_RUNTIME_TOOLS, (
                f"{agent_md} invalid App/CLI runtime tool {tool!r}"
            )
        assert isinstance(meta.get("subagent"), bool), f"{agent_md} missing boolean subagent"
        assert isinstance(meta.get("mainAgent"), bool), f"{agent_md} missing boolean mainAgent"
        policy = meta.get("commandExecutionPolicy")
        assert policy in VALID_COMMAND_POLICIES, f"{agent_md} invalid commandExecutionPolicy: {policy!r}"
        if "model" in meta:
            assert meta["model"] in {"inherit", "flash_lite", "flash", "pro"}
        parsed_by_name[name] = meta

    assert "gate-auditor" in parsed_by_name
    assert "sheaf-bridge-builder" in parsed_by_name
    auditor = parsed_by_name["gate-auditor"]
    assert auditor["subagent"] is True
    assert auditor["mainAgent"] is False
    assert auditor["commandExecutionPolicy"] in {"sandbox", "off"}
    assert WRITE_RUNTIME_TOOLS.isdisjoint(set(auditor["tools"])), (
        f"gate-auditor must be read-only, found write tools in {auditor['tools']}"
    )


@pytest.mark.tier1
def test_t1_f17_hooks_json_schema_and_pretooluse_allow_exit0(tmp_path: Path) -> None:
    hooks_path = REPO_ROOT / ".agents" / "hooks.json"
    assert hooks_path.is_file(), f"Missing Antigravity hooks manifest: {hooks_path}"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    if "enabled" in data:
        assert isinstance(data["enabled"], bool)
        assert data["enabled"] is True

    present_events = set(data.keys()) - {"enabled", "$schema"}
    assert present_events and present_events.issubset(VALID_HOOK_EVENTS), f"Invalid events: {present_events}"
    assert "PreToolUse" in data

    for event_name in present_events:
        entries = data[event_name]
        assert isinstance(entries, list) and entries, f"{event_name} must be non-empty list"
        if event_name in {"PreToolUse", "PostToolUse"}:
            for group in entries:
                assert isinstance(group, dict)
                matcher = group.get("matcher")
                assert isinstance(matcher, str) and matcher
                re.compile(matcher)
                handlers = group.get("hooks")
                assert isinstance(handlers, list) and handlers
                for handler in handlers:
                    assert isinstance(handler, dict)
                    assert handler.get("type", "command") == "command"
                    assert isinstance(handler.get("command"), str) and handler["command"].strip()
                    timeout = handler.get("timeout", 30)
                    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0

    cmds = _matching_pretooluse_commands(hooks_path, "write_to_file")
    benign_cases = [
        ("src/sheaf_verify/ok_stdlib.py", "from __future__ import annotations\n\nimport hashlib\n"),
        ("src/hadukenn_core/ok_np.py", "from __future__ import annotations\n\nimport numpy as np\n"),
    ]
    for target_file, code_text in benign_cases:
        args = {
            "path": target_file,
            "file_path": target_file,
            "TargetFile": target_file,
            "content": code_text,
            "CodeContent": code_text,
        }
        payload = _make_stdin_payload(tmp_path, "write_to_file", args)
        for cmd in cmds:
            proc, parsed = run_hook(cmd, payload, cwd=REPO_ROOT)
            assert proc.returncode == 0, f"Allow hook must exit 0 (got {proc.returncode}): {proc.stderr}"
            decision = parsed.get("decision")
            assert decision in {"allow", "auto_approve"}, f"Expected allow/auto_approve, got {parsed!r}"


@pytest.mark.tier1
def test_t1_f17_mcp_config_and_stdio_initialize_tools_list() -> None:
    mcp_cfg_path = REPO_ROOT / ".agents" / "mcp_config.json"
    assert mcp_cfg_path.is_file(), f"Missing MCP config: {mcp_cfg_path}"
    cfg = json.loads(mcp_cfg_path.read_text(encoding="utf-8"))
    servers = cfg.get("mcpServers")
    assert isinstance(servers, dict) and servers

    stdio_count = 0
    for srv_name, entry in servers.items():
        assert isinstance(entry, dict), f"Server {srv_name!r} must be object"
        assert "httpUrl" not in entry, f"Legacy httpUrl forbidden in {srv_name!r}"
        has_cmd = bool(entry.get("command"))
        has_http = bool(entry.get("serverUrl") or entry.get("url"))
        assert has_cmd ^ has_http, f"Server {srv_name!r} must specify stdio command XOR Streamable HTTP serverUrl/url"
        if "timeoutSeconds" in entry:
            ts = entry["timeoutSeconds"]
            assert isinstance(ts, int) and not isinstance(ts, bool) and (ts == -1 or ts > 0)
        if "tools" in entry:
            assert isinstance(entry["tools"], dict)
            for t_cfg in entry["tools"].values():
                if isinstance(t_cfg, dict) and "background" in t_cfg:
                    assert t_cfg["background"] in {"off", "always"}
        if has_cmd:
            stdio_count += 1
            for arg in entry.get("args", []):
                if isinstance(arg, str) and arg.endswith(".py"):
                    script = (REPO_ROOT / arg).resolve()
                    if script.is_file():
                        _check_python_ast_and_stdlib(script, allow_sheaf_verify=True)
    assert stdio_count >= 1

    with mcp_stdio_session() as client:
        init_resp = client.initialize()
        assert "result" in init_resp, f"MCP initialize failed: {init_resp!r}"
        result = init_resp["result"]
        assert "protocolVersion" in result
        assert "serverInfo" in result

        tools = client.list_tools()
        assert len(tools) >= 1
        for tool in tools:
            assert isinstance(tool.get("name"), str) and tool["name"]
            assert isinstance(tool.get("description"), str) and tool["description"]
            assert isinstance(tool.get("inputSchema"), dict)
            meta = tool.get("_meta")
            if isinstance(meta, dict) and "ui" in meta:
                assert isinstance(meta["ui"], dict)
                assert isinstance(meta["ui"].get("resourceUri"), str)


@pytest.mark.tier1
def test_t1_f17_mcp_stdio_tools_call_read_only_verification() -> None:
    baseline = snapshot_readonly_hashes(REPO_ROOT)
    try:
        with mcp_stdio_session() as client:
            init_resp = client.initialize()
            assert "result" in init_resp
            tools = client.list_tools()
            names = [t["name"] for t in tools]
            assert names

            anchor_tools = [t for t in tools if "anchor" in t["name"].lower()]
            target_tool = anchor_tools[0] if anchor_tools else tools[0]
            props = target_tool.get("inputSchema", {}).get("properties", {})
            args: dict[str, Any] = {}
            for key in ("store", "store_path", "path"):
                if key in props:
                    args[key] = "fixtures/passing"
                    break

            call_resp = client.call_tool(target_tool["name"], args)
            assert "result" in call_resp, f"tools/call returned error: {call_resp!r}"
            res = call_resp["result"]
            assert res.get("isError") is not True
            content = res.get("content")
            assert isinstance(content, list) and content
            rendered = json.dumps(res, sort_keys=True)
            assert CANONICAL_ANCHOR in rendered, f"Expected canonical anchor in MCP result: {rendered}"
    finally:
        assert_readonly_unchanged(baseline, REPO_ROOT)


@pytest.mark.tier1
def test_t1_f17_plugin_bundle_draft_2020_12_and_variable_substitution() -> None:
    plugin_root = REPO_ROOT / ".agents" / "plugins" / "hadukenn-sheaf"
    assert plugin_root.is_dir(), f"Missing plugin directory: {plugin_root}"

    preferred_manifest = plugin_root / ".antigravity-plugin" / "plugin.json"
    fallback_manifest = plugin_root / "plugin.json"
    manifest_path = preferred_manifest if preferred_manifest.is_file() else fallback_manifest
    assert manifest_path.is_file(), f"Missing plugin.json in {plugin_root}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    unknown_keys = set(manifest.keys()) - PLUGIN_ALLOWED_KEYS
    assert not unknown_keys, f"Draft 2020-12 additionalProperties violation: {sorted(unknown_keys)}"
    name = manifest.get("name")
    assert isinstance(name, str) and KEBAB_RE.match(name), f"Invalid plugin name: {name!r}"
    if "disabled" in manifest:
        assert isinstance(manifest["disabled"], bool)
    if "suggestedPrompts" in manifest:
        prompts = manifest["suggestedPrompts"]
        assert isinstance(prompts, list) and len(prompts) <= 3
        for item in prompts:
            assert isinstance(item, str) and re.search(r"\S", item)
    if "logo" in manifest:
        logo_rel = manifest["logo"]
        assert isinstance(logo_rel, str)
        logo_path = (plugin_root / logo_rel).resolve()
        assert logo_path.is_file(), f"Missing plugin logo: {logo_path}"
        assert logo_path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"}

    plugin_skills_dir = plugin_root / "skills"
    if plugin_skills_dir.is_dir():
        for skill_md in sorted(plugin_skills_dir.glob("*/SKILL.md")):
            _validate_plugin_skill(skill_md, skill_md.parent.name)

    plugin_mcp = plugin_root / "mcp_config.json"
    if plugin_mcp.is_file():
        mcp_text = plugin_mcp.read_text(encoding="utf-8")
        assert "${PLUGIN_ROOT}" in mcp_text or "${PLUGIN_DATA}" in mcp_text
        mcp_data = json.loads(mcp_text)
        assert isinstance(mcp_data.get("mcpServers"), dict) and mcp_data["mcpServers"]

    plugin_hooks = plugin_root / "hooks.json"
    if plugin_hooks.is_file():
        hooks_text = plugin_hooks.read_text(encoding="utf-8")
        assert "${PLUGIN_ROOT}" not in hooks_text, "Plugin hooks.json must use relative ./scripts/... paths"
        assert "./scripts/" in hooks_text
        hooks_data = json.loads(hooks_text)
        assert isinstance(hooks_data, dict)

    agy_bin = shutil.which("agy")
    if agy_bin:
        proc = subprocess.run(
            [agy_bin, "plugin", "validate", "--strict", "--json", str(plugin_root)],
            cwd=REPO_ROOT,
            env=no_gpu_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        assert proc.returncode == 0, f"agy plugin validate failed: {proc.stdout}\n{proc.stderr}"
        report = json.loads(proc.stdout)
        errors = [f for f in report.get("findings", []) if f.get("severity") == "error"]
        assert not errors, f"agy plugin validate findings: {errors}"


@pytest.mark.tier1
def test_t1_f17_sidecar_automation_spec_valid() -> None:
    candidates = sorted(
        [
            *(REPO_ROOT / ".agents" / "sidecars").glob("**/sidecar.json"),
            *(REPO_ROOT / ".agents" / "plugins" / "hadukenn-sheaf" / "sidecars").glob("**/sidecar.json"),
        ]
    )
    assert candidates, "Expected at least one Antigravity sidecar.json spec under .agents/"

    for sidecar_path in candidates:
        spec = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert isinstance(spec, dict)
        has_cmd = isinstance(spec.get("command"), str) and bool(spec["command"].strip())
        has_builtin = spec.get("builtin") == "schedule"
        assert has_cmd or has_builtin, f"{sidecar_path} must define command or builtin='schedule'"
        assert spec.get("restart_policy") in VALID_RESTART_POLICIES, f"{sidecar_path} bad restart_policy"
        desc = spec.get("description")
        assert isinstance(desc, str) and desc.strip(), f"{sidecar_path} missing description"
        if "args" in spec:
            assert isinstance(spec["args"], list) and all(isinstance(a, str) for a in spec["args"])
        if "arguments" in spec:
            assert isinstance(spec["arguments"], dict)
        if "agent_permissions" in spec:
            perms = spec["agent_permissions"]
            assert isinstance(perms, dict)
            if "workspace_uris" in perms:
                assert isinstance(perms["workspace_uris"], list)
            if "access_grants" in perms:
                assert isinstance(perms["access_grants"], list)


# ==============================================================================
# Feature 18: Antigravity Python SDK Adapter, Multi-Modal & Docs Budget (8 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f18_sdk_configs_and_stdlib_fallback_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")

    local_cfg_cls = resolve_public_symbol("hadukenn_core", "LocalAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    litert_cfg_cls = resolve_public_symbol("hadukenn_core", "LiteRTAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    openai_cfg_cls = resolve_any_public_symbol(
        "hadukenn_core",
        ("LocalOpenAIAgentConfig", "LocalAgentConfig"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    caps_cls = resolve_public_symbol("hadukenn_core", "CapabilitiesConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    builtin_tools = resolve_public_symbol("hadukenn_core", "BuiltinTools", _ANTIGRAVITY_PREFERRED_MODULES)
    subagent_cls = resolve_public_symbol("hadukenn_core", "SubagentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    mcp_stdio_cls = resolve_public_symbol("hadukenn_core", "McpStdioServer", _ANTIGRAVITY_PREFERRED_MODULES)

    read_only_tools = builtin_tools.read_only() if callable(getattr(builtin_tools, "read_only", None)) else []
    assert read_only_tools, "BuiltinTools.read_only() must return non-empty read-only SDK tools"
    ro_names = {getattr(t, "value", t) for t in read_only_tools}
    assert {"create_file", "edit_file", "run_command"}.isdisjoint(ro_names)

    sdk_members = {getattr(m, "value", m) for m in builtin_tools} if hasattr(builtin_tools, "__iter__") else set()
    if sdk_members:
        assert "edit_file" in sdk_members and "list_directory" in sdk_members
        assert "replace_file_content" not in sdk_members

    caps = _bind_call(
        caps_cls,
        {"enabled_tools": list(read_only_tools), "enable_subagents": True},
        (list(read_only_tools),),
    )
    assert caps is not None

    litert_cfg = (
        litert_cfg_cls.lightweight()
        if callable(getattr(litert_cfg_cls, "lightweight", None))
        else _bind_call(litert_cfg_cls, {"model_path": "models/litert_int8"}, ("models/litert_int8",))
    )
    assert litert_cfg is not None

    openai_cfg = (
        openai_cfg_cls.lightweight(model="hadukenn-struct32k", base_url="http://127.0.0.1:8080/v1")
        if callable(getattr(openai_cfg_cls, "lightweight", None))
        else _bind_call(
            openai_cfg_cls,
            {"model": "hadukenn-struct32k", "base_url": "http://127.0.0.1:8080/v1"},
        )
    )
    assert openai_cfg is not None

    local_cfg = _bind_call(
        local_cfg_cls,
        {
            "conversation_id": "conv-antigravity-e2e-0123456789abcdef",
            "capabilities": caps,
        },
    )
    assert local_cfg is not None

    subagent = _bind_call(
        subagent_cls,
        {"name": "gate-auditor", "description": "Read-only gate auditor", "tools": list(read_only_tools)},
        ("gate-auditor", "Read-only gate auditor"),
    )
    assert subagent is not None

    mcp_srv = _bind_call(
        mcp_stdio_cls,
        {"command": sys.executable, "args": ["-m", "sheaf_verify"]},
        (sys.executable, ["-m", "sheaf_verify"]),
    )
    assert mcp_srv is not None
    assert "edit_file" in SDK_BUILTIN_TOOL_NAMES and "replace_file_content" in APP_CLI_RUNTIME_TOOLS


@pytest.mark.tier1
def test_t1_f18_sdk_session_id_response_schema_and_lifecycle_hooks() -> None:
    conv_validator = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_conversation_id", "is_valid_conversation_id", "LocalAgentConfig"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    valid_conv_id = "conv-antigravity-e2e-0123456789abcdef"
    assert len(valid_conv_id) >= 32 and "_" not in valid_conv_id and CONV_ID_RE.match(valid_conv_id)
    if callable(conv_validator) and not isinstance(conv_validator, type):
        res = conv_validator(valid_conv_id)
        assert res is not False
    else:
        cfg = _bind_call(conv_validator, {"conversation_id": valid_conv_id})
        assert cfg is not None

    schema_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_response_schema", "verify_response_schema", "ResponseSchemaValidator"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    schema = {
        "type": "object",
        "required": ["ok", "anchor"],
        "properties": {"ok": {"type": "boolean"}, "anchor": {"type": "string"}},
    }
    sample = {"ok": True, "anchor": CANONICAL_ANCHOR}
    if isinstance(schema_fn, type):
        validator = _bind_call(schema_fn, {"schema": schema}, (schema,))
        check_method = getattr(validator, "validate", getattr(validator, "verify", None))
        assert check_method is not None
        assert check_method(sample) is not False
    else:
        schema_params = inspect.signature(schema_fn).parameters
        schema_kwargs: dict[str, Any] = {"schema": schema}
        if "data" in schema_params and "payload" not in schema_params:
            schema_kwargs["data"] = sample
        else:
            schema_kwargs["payload"] = sample
        assert _bind_call(schema_fn, schema_kwargs, (sample, schema)) is not False

    hook_result_cls = resolve_public_symbol("hadukenn_core", "HookResult", _ANTIGRAVITY_PREFERRED_MODULES)
    lifecycle = resolve_any_public_symbol(
        "hadukenn_core",
        ("LifecycleHooks", "HookRegistry", "AntigravityLifecycle", "run_lifecycle_hooks"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    hr = _bind_call(hook_result_cls, {"decision": "allow", "reason": "ok"}, ("allow",))
    assert hr is not None

    registry = lifecycle() if isinstance(lifecycle, type) else lifecycle
    seen: list[str] = []

    def _pre_turn(ctx: dict[str, Any]) -> Any:
        seen.append("pre_turn")
        return hr

    def _on_tool_error(ctx: dict[str, Any]) -> Any:
        seen.append("on_tool_error")
        return hr

    if hasattr(registry, "pre_turn") and callable(registry.pre_turn):
        registry.pre_turn(_pre_turn)
    if hasattr(registry, "on_tool_error") and callable(registry.on_tool_error):
        registry.on_tool_error(_on_tool_error)
    if hasattr(registry, "run_pre_turn"):
        out = registry.run_pre_turn({"conversation_id": valid_conv_id})
        assert out is not None and "pre_turn" in seen
    elif callable(registry):
        out = _bind_call(
            registry,
            {"pre_turn": _pre_turn, "on_tool_error": _on_tool_error, "ctx": {"conversation_id": valid_conv_id}},
        )
        assert out is not None and "pre_turn" in seen
    else:
        raise AssertionError(f"Lifecycle surface lacks run_pre_turn/callable: {registry!r}")


@pytest.mark.tier1
def test_t1_f18_sdk_presets_and_permission_precedence_rules() -> None:
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

    presets = (
        "request-review",
        "default",
        "accept-edits",
        "auto",
        "plan",
        "strict",
        "always-proceed",
        "proceed-in-sandbox",
    )
    for preset in presets:
        if isinstance(preset_fn, type):
            resolved = preset_fn.from_preset(preset) if hasattr(preset_fn, "from_preset") else preset_fn(preset)
        else:
            resolved = preset_fn(preset)
        assert resolved is not None, f"Failed to resolve preset {preset!r}"

    rules = [
        {"action": "read_file", "decision": "allow"},
        {"action": "write_file", "decision": "allow"},
        {"action": "command", "pattern": "pytest", "decision": "ask"},
    ]
    if isinstance(eval_fn, type):
        evaluator = _bind_call(eval_fn, {"rules": rules}, (rules,))
        fn = getattr(evaluator, "evaluate", getattr(evaluator, "check", None))
        assert fn is not None
    else:
        fn = eval_fn

    write_res = _bind_call(fn, {"action": "write_file", "rules": rules}, ("write_file", rules))
    read_res = _bind_call(fn, {"action": "read_file", "rules": rules}, ("read_file", rules))
    assert str(getattr(write_res, "decision", write_res)).lower() == "allow"
    assert str(getattr(read_res, "decision", read_res)).lower() == "allow"


@pytest.mark.tier1
def test_t1_f18_multimodal_pathway_graph_kernel_hardware_formal_integration(tmp_path: Path) -> None:
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

    media_path = tmp_path / "telemetry_sample.png"
    media_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    attachment = (
        from_file_fn.from_file(media_path)
        if isinstance(from_file_fn, type) and hasattr(from_file_fn, "from_file")
        else from_file_fn(media_path)
    )
    assert attachment is not None

    cfg = _bind_call(
        litert_cfg_cls,
        {
            "model_path": "models/struct32k_int8",
            "vision_backend": "cpu",
            "audio_backend": "cpu",
        },
        ("models/struct32k_int8",),
    )
    runner = pathway_fn() if isinstance(pathway_fn, type) else pathway_fn
    exec_fn = getattr(runner, "run", getattr(runner, "evaluate", runner))
    result = _bind_call(
        exec_fn,
        {
            "config": cfg,
            "attachments": [attachment],
            "vocab_size": 32_768,
            "head_type": "dense_flat",
            "hardware_target": "cpu",
        },
    )
    assert result is not None
    if isinstance(result, bool):
        assert result is True
    elif isinstance(result, dict):
        status_str = str(result.get("status", "")).lower()
        assert result.get("ok") is not False, f"Multimodal pathway failed: {result!r}"
        assert result.get("passed") is not False, f"Multimodal pathway failed: {result!r}"
        assert result.get("valid") is not False, f"Multimodal pathway failed: {result!r}"
        assert result.get("approved") is not False, f"Multimodal pathway failed: {result!r}"
        assert status_str not in {"error", "failed", "fail", "deny", "rejected"}, (
            f"Multimodal pathway failed status: {result!r}"
        )
        assert (
            result.get("ok") is True
            or result.get("passed") is True
            or result.get("valid") is True
            or result.get("approved") is True
            or status_str in {"ok", "pass", "passed", "success", "allow", "approved"}
        ), f"Multimodal pathway missing explicit positive signal: {result!r}"


@pytest.mark.tier1
def test_t1_f18_antigravity_cli_and_ide_surface_contracts() -> None:
    cli_contract = resolve_any_public_symbol(
        "hadukenn_core",
        ("AntigravityCliContract", "build_headless_cli_args", "HEADLESS_OUTPUT_FORMATS"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    settings_url = "https://antigravity.google/schemas/v1/settings.json"
    handoff_text = (REPO_ROOT / "ANTIGRAVITY_HANDOFF.md").read_text(encoding="utf-8")
    assert "Antigravity" in handoff_text and "agy" in handoff_text

    if isinstance(cli_contract, (set, frozenset, tuple, list)):
        assert {"text", "json", "stream-json"}.issubset(set(cli_contract))
    elif isinstance(cli_contract, type):
        instance = cli_contract()
        formats = getattr(instance, "output_formats", getattr(cli_contract, "OUTPUT_FORMATS", ()))
        statuses = getattr(instance, "terminal_statuses", getattr(cli_contract, "TERMINAL_STATUSES", ()))
        assert {"text", "json", "stream-json"}.issubset(set(formats))
        assert {"SUCCESS", "ERROR", "CANCELED", "INTERRUPTED", "INVALID", "WAITING", "RUNNING"}.issubset(
            set(statuses)
        )
        schema_val = getattr(instance, "settings_schema", getattr(cli_contract, "SETTINGS_SCHEMA", ""))
        assert schema_val == settings_url
    else:
        argv = _bind_call(
            cli_contract,
            {"prompt": "verify gates", "output_format": "stream-json"},
            ("verify gates",),
        )
        assert isinstance(argv, list) and "--output-format" in argv

    agy_bin = shutil.which("agy")
    if agy_bin:
        proc = subprocess.run(
            [agy_bin, "--version"],
            cwd=REPO_ROOT,
            env=no_gpu_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
        assert proc.returncode == 0


@pytest.mark.tier1
def test_t1_f18_docs_budget_live_tree_passes_and_under_limits() -> None:
    script = REPO_ROOT / ".github" / "scripts" / "docs_budget.py"
    assert script.is_file()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert proc.returncode == 0, f"docs_budget.py failed: {proc.stderr}"
    match = re.search(r"docs/\s+(\d+)\s+bytes", proc.stdout)
    assert match is not None, f"Unexpected docs_budget stdout: {proc.stdout!r}"
    assert int(match.group(1)) <= 1_000_000
    for html_path in sorted((REPO_ROOT / "docs").rglob("*.html")):
        assert html_path.stat().st_size <= 307_200


@pytest.mark.tier1
def test_t1_f18_docs_budget_local_relative_script_allowed(tmp_path: Path) -> None:
    src_script = REPO_ROOT / ".github" / "scripts" / "docs_budget.py"
    dst_script = tmp_path / ".github" / "scripts" / "docs_budget.py"
    dst_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_script, dst_script)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "index.html").write_text(
        '<!doctype html><html><head><script src="app.js"></script></head><body>ok</body></html>\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(dst_script)],
        cwd=tmp_path,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert "no external scripts" in proc.stdout


@pytest.mark.tier1
def test_t1_f18_repo_export_hygiene_and_v080_markers_preserved() -> None:
    baseline_paths = (
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "ANTIGRAVITY_HANDOFF.md",
        REPO_ROOT / ".agents" / "rules" / "00-repository-invariants.md",
        REPO_ROOT / ".agents" / "rules" / "10-sheaf-verify-stdlib.md",
        REPO_ROOT / ".agents" / "rules" / "20-empirical-architecture-guardrails.md",
        REPO_ROOT / ".agents" / "skills" / "verify-all-gates" / "SKILL.md",
        REPO_ROOT / ".agents" / "skills" / "verify-all-gates" / "scripts" / "verify_all.py",
        REPO_ROOT / ".agents" / "skills" / "run-sheaf-bridge" / "SKILL.md",
        REPO_ROOT / ".agents" / "skills" / "run-sheaf-bridge" / "scripts" / "run_bridge_benchmark.py",
        REPO_ROOT / ".agents" / "skills" / "prepare-iteration-pr" / "SKILL.md",
        REPO_ROOT / ".agents" / "skills" / "prepare-iteration-pr" / "scripts" / "check_pr_hygiene.py",
        REPO_ROOT / ".agents" / "agents" / "gate-auditor" / "agent.md",
        REPO_ROOT / ".agents" / "agents" / "sheaf-bridge-builder" / "agent.md",
    )
    for req in baseline_paths:
        assert req.is_file(), f"Missing baseline Antigravity handoff file: {req}"

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".so", ".dll"}
    banned_phrases = (
        "Permission is hereby " + "granted",
        "well" + "bore",
        "drill" + "ing",
        "DEVn" + "-1",
        "DESKTOP" + "-",
        "train" + "_paper",
    )
    drive_markers = tuple(chr(code) + ":" + chr(92) for code in (67, 68, 69, 72))
    scanned_suffixes = {".py", ".md", ".toml", ".yml", ".json", ".html", ".css", ""}

    for folder in (
        REPO_ROOT / ".agents",
        REPO_ROOT / "src" / "hadukenn_core",
        REPO_ROOT / "docs",
        REPO_ROOT / "tests_e2e",
    ):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            assert path.suffix.lower() not in forbidden_suffixes, f"Forbidden binary file: {path}"
            if path.suffix.lower() in scanned_suffixes:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for phrase in banned_phrases:
                    assert phrase not in text, f"Banned phrase in {path}"
                for drive in drive_markers:
                    assert drive not in text, f"Windows drive path in {path}"
