"""Deterministic pure-stdlib smoke, schema, subprocess, and SDK tests for Antigravity integration.

Designed to run identically under both real ``pytest`` and ``python3 -S -m sheaf_verify.selftest``
without third-party dependencies or custom non-autouse fixtures.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hadukenn_core.antigravity_sdk import (
    AntigravityCliContract,
    BuiltinTools,
    CapabilitiesConfig,
    HookResult,
    LifecycleHooks,
    LiteRTAgentConfig,
    LocalAgentConfig,
    LocalOpenAIAgentConfig,
    McpStdioServer,
    McpStreamableHttpServer,
    MultimodalAttachment,
    SubagentConfig,
    build_headless_cli_args,
    evaluate_permission,
    from_file,
    resolve_permission_preset,
    run_multimodal_pathway,
    validate_conversation_id,
    validate_response_schema,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ANCHOR = "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"
APP_CLI_TOOLS = frozenset(
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
WRITE_TOOLS = frozenset({"write_to_file", "replace_file_content", "multi_replace_file_content"})


def _parse_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert raw.startswith("---\n"), f"{path} missing frontmatter"
    lines = raw.split("\n")
    end_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    meta: dict[str, object] = {}
    for line in lines[1:end_idx]:
        if not line.strip() or line.strip().startswith("#"):
            continue
        assert ":" in line, f"{path} invalid YAML line: {line!r}"
        key, val = line.split(":", 1)
        key = key.strip()
        val_s = val.strip()
        if len(val_s) >= 2 and val_s[0] == val_s[-1] and val_s[0] in {'"', "'"}:
            meta[key] = val_s[1:-1]
        elif val_s.lower() == "true":
            meta[key] = True
        elif val_s.lower() == "false":
            meta[key] = False
        elif val_s.startswith("[") and val_s.endswith("]"):
            inner = val_s[1:-1].strip()
            meta[key] = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
        else:
            meta[key] = val_s
    return meta, "\n".join(lines[end_idx + 1 :])


def test_rules_agents_and_plugin_schema_contracts() -> None:
    rule_10, body_10 = _parse_frontmatter(ROOT / ".agents" / "rules" / "10-sheaf-verify-stdlib.md")
    assert rule_10["trigger"] == "glob"
    assert isinstance(rule_10["globs"], str)
    assert "src/sheaf_verify/**/*.py" in rule_10["globs"]
    assert len(body_10.encode("utf-8")) <= 24_000

    auditor, _ = _parse_frontmatter(ROOT / ".agents" / "agents" / "gate-auditor" / "agent.md")
    assert auditor["subagent"] is True
    assert auditor["mainAgent"] is False
    assert auditor["commandExecutionPolicy"] == "sandbox"
    assert isinstance(auditor["tools"], list) and set(auditor["tools"]).issubset(APP_CLI_TOOLS)
    assert WRITE_TOOLS.isdisjoint(set(auditor["tools"]))

    builder, _ = _parse_frontmatter(ROOT / ".agents" / "agents" / "sheaf-bridge-builder" / "agent.md")
    assert builder["subagent"] is True
    assert builder["mainAgent"] is False
    assert isinstance(builder["tools"], list) and set(builder["tools"]).issubset(APP_CLI_TOOLS)

    plugin_root = ROOT / ".agents" / "plugins" / "hadukenn-sheaf"
    manifest_path = plugin_root / ".antigravity-plugin" / "plugin.json"
    assert manifest_path.is_file()
    assert not (plugin_root / "plugin.json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_keys = {
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
    assert set(manifest).issubset(allowed_keys)
    assert manifest["name"] == "hadukenn-sheaf"
    assert len(manifest["suggestedPrompts"]) <= 3
    assert (plugin_root / manifest["logo"]).is_file()

    plugin_hooks_text = (plugin_root / "hooks.json").read_text(encoding="utf-8")
    assert "${PLUGIN_ROOT}" not in plugin_hooks_text
    assert "./scripts/" in plugin_hooks_text

    plugin_mcp_text = (plugin_root / "mcp_config.json").read_text(encoding="utf-8")
    assert "${PLUGIN_ROOT}" in plugin_mcp_text

    for sidecar_path in (
        ROOT / ".agents" / "sidecars" / "sheaf-nightly-audit" / "sidecar.json",
        plugin_root / "sidecars" / "sheaf-nightly-audit" / "sidecar.json",
    ):
        spec = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert spec["builtin"] == "schedule"
        assert spec["restart_policy"] in {"always", "on-failure", "never"}
        assert isinstance(spec["description"], str) and spec["description"].strip()

    agy_bin = shutil.which("agy")
    if agy_bin is not None:
        proc = subprocess.run(
            [agy_bin, "plugin", "validate", "--strict", "--json", str(plugin_root)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_hook_scripts_executable_pure_stdlib_and_exit0_decisions() -> None:
    hooks_cfg = json.loads((ROOT / ".agents" / "hooks.json").read_text(encoding="utf-8"))
    assert hooks_cfg["enabled"] is True
    assert not (ROOT / ".agents" / "settings.json").exists()

    hook_script = ROOT / ".agents" / "hooks" / "pre_tool_use_guard.py"
    assert stat.S_IMODE(hook_script.stat().st_mode) & 0o111

    def _run_hook(payload: dict[str, object], event: str = "PreToolUse") -> dict[str, object]:
        proc = subprocess.run(
            [sys.executable, str(hook_script), "--event", event],
            cwd=ROOT,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip())

    # Allow cases
    ok_sheaf = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {"TargetFile": "src/sheaf_verify/ok.py", "CodeContent": "import json\n"},
        }
    )
    assert ok_sheaf["decision"] == "allow"

    ok_core = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {"path": "src/hadukenn_core/ok.py", "content": "import numpy as np\n"},
        }
    )
    assert ok_core["decision"] == "allow"

    # Deny cases (all must exit 0)
    bad_np = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {"TargetFile": "src/sheaf_verify/bad.py", "CodeContent": "import numpy as np\n"},
        }
    )
    assert bad_np["decision"] == "deny" and bad_np["reason"]

    bad_bin = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {"path": "artifacts/weights.safetensors", "content": "x"},
        }
    )
    assert bad_bin["decision"] == "deny"

    drive_str = chr(67) + ":" + chr(92) + "Users" + chr(92) + "weights.txt"
    bad_drive = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {"path": "docs/note.md", "content": f"path={drive_str}\n"},
        }
    )
    assert bad_drive["decision"] == "deny"

    bad_182 = _run_hook(
        {
            "toolName": "write_to_file",
            "toolInput": {
                "path": "src/hadukenn_core/bad_head.py",
                "content": 'vocab = 32768\nhead = "182x181"\ndevice = "cuda"\n',
            },
        }
    )
    assert bad_182["decision"] == "deny"

    bad_fixture = _run_hook(
        {
            "toolName": "replace_file_content",
            "toolInput": {"TargetFile": "fixtures/passing/ledger/head.json", "ReplacementContent": "{}"},
        }
    )
    assert bad_fixture["decision"] == "deny"

    assert _run_hook({}, event="Stop")["decision"] == "allow"
    assert _run_hook({}, event="PostToolUse") == {}


def test_mcp_server_ast_purity_and_stdio_roundtrip() -> None:
    mcp_cfg = json.loads((ROOT / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
    srv = mcp_cfg["mcpServers"]["hadukenn-sheaf"]
    script = ROOT / srv["args"][0]
    assert stat.S_IMODE(script.stat().st_mode) & 0o111

    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))
    allowed = set(sys.stdlib_module_names) | {"sheaf_verify"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] in allowed
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".", 1)[0] in allowed

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "sheaf_anchor", "arguments": {"store": "fixtures/passing"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "ui://hadukenn-sheaf/dashboard"}},
        {"jsonrpc": "2.0", "id": 5, "method": "unknown/method", "params": {}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "missing_tool", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}},
    ]
    stdin_text = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    by_id = {r["id"]: r for r in responses}
    assert set(by_id) == {1, 2, 3, 4, 5, 6, 7}
    caps = by_id[1]["result"]["capabilities"]
    assert "io.modelcontextprotocol/ui" in caps["extensions"]
    assert "com.google.antigravity/ui" in caps["extensions"]
    tools = by_id[2]["result"]["tools"]
    assert any(t["name"] == "sheaf_anchor" for t in tools)
    assert all(t["_meta"]["ui"]["resourceUri"] == "ui://hadukenn-sheaf/dashboard" for t in tools)
    anchor_res = by_id[3]["result"]
    assert anchor_res["isError"] is False
    assert CANONICAL_ANCHOR in json.dumps(anchor_res)
    html_text = by_id[4]["result"]["contents"][0]["text"]
    assert len(html_text.encode("utf-8")) < 100_000
    assert not re.search(r"<script\b", html_text, re.IGNORECASE)
    assert "error" in by_id[5]
    assert by_id[6]["result"]["isError"] is True
    assert len(by_id[7]["result"]["tools"]) == len(tools)

    # Also verify Content-Length LSP-style stdio framing round-trip
    lsp_req = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}}).encode("utf-8")
    lsp_frame = f"Content-Length: {len(lsp_req)}\r\n\r\n".encode("ascii") + lsp_req
    lsp_proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        input=lsp_frame,
        capture_output=True,
        check=False,
    )
    assert lsp_proc.returncode == 0
    assert lsp_proc.stdout.startswith(b"Content-Length:")
    _, body_bytes = lsp_proc.stdout.split(b"\r\n\r\n", 1)
    assert json.loads(body_bytes.decode("utf-8")) == {"jsonrpc": "2.0", "id": 99, "result": {}}


def test_antigravity_sdk_configs_permissions_hooks_and_multimodal(tmp_path: Path) -> None:
    ro = BuiltinTools.read_only()
    assert ro and {"create_file", "edit_file", "run_command"}.isdisjoint({t.value for t in ro})
    caps = CapabilitiesConfig(enabled_tools=ro, enable_subagents=True)
    assert caps.enable_subagents is True

    with pytest.raises(ValueError):
        CapabilitiesConfig(enabled_tools=["replace_file_content"])

    valid_id = "conv-antigravity-smoke-0123456789abcdef"
    assert validate_conversation_id(valid_id) is True
    for bad_id in ("short", "conv_with_underscore_0123456789abcd", ""):
        with pytest.raises(ValueError):
            validate_conversation_id(bad_id)

    local_cfg = LocalAgentConfig(conversation_id=valid_id, capabilities=caps)
    assert local_cfg.backend_mode in {"stdlib_fallback", "google_antigravity"}
    litert_cfg = LiteRTAgentConfig.lightweight()
    openai_cfg = LocalOpenAIAgentConfig.lightweight()
    assert litert_cfg.port == 8080 and openai_cfg.model == "hadukenn-struct32k"
    assert SubagentConfig(name="auditor", description="Read-only", tools=ro).name == "auditor"
    assert McpStdioServer(command=sys.executable, args=["-m", "sheaf_verify"]).command == sys.executable
    assert McpStreamableHttpServer(url="http://127.0.0.1:8080/mcp").timeout_seconds == 180

    with pytest.raises(ValueError):
        LiteRTAgentConfig(model_path="")
    with pytest.raises(ValueError):
        LiteRTAgentConfig(port=0)

    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    assert validate_response_schema({"ok": True}, schema) is True
    with pytest.raises(ValueError):
        validate_response_schema({"ok": "nope"}, schema)

    hooks = LifecycleHooks()
    hooks.pre_turn(lambda ctx: HookResult("allow", f"ok:{ctx.get('id', '')}"))
    hooks.on_tool_error(lambda ctx: HookResult("ask", str(ctx.get("error", ""))))
    assert hooks.run_pre_turn({"id": "1"}).decision == "allow"
    assert hooks.run_on_tool_error({"error": "boom"}).reason == "boom"

    for preset in (
        "request-review",
        "default",
        "accept-edits",
        "auto",
        "plan",
        "strict",
        "always-proceed",
        "proceed-in-sandbox",
    ):
        assert resolve_permission_preset(preset)
    with pytest.raises(ValueError):
        resolve_permission_preset("unknown-unsafe-preset")

    deny_rules = [
        {"action": "read_file", "decision": "allow"},
        {"action": "read_file", "decision": "deny"},
        {"action": "write_file", "decision": "allow"},
    ]
    assert evaluate_permission("read_file", deny_rules).decision == "deny"
    assert evaluate_permission("write_file", deny_rules).decision == "deny"

    img = tmp_path / "sample.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    att = from_file(img)
    assert isinstance(att, MultimodalAttachment) and att.media_type == "image/png"
    with pytest.raises(FileNotFoundError):
        from_file(tmp_path / "missing.png")

    ok_mm = run_multimodal_pathway(litert_cfg, [att], vocab_size=32_768, head_type="dense_flat", hardware_target="cpu")
    assert ok_mm["ok"] is True
    with pytest.raises(ValueError):
        run_multimodal_pathway(litert_cfg, [att], vocab_size=32_768, head_type="182x181", hardware_target="cuda")

    cli = AntigravityCliContract()
    assert "stream-json" in cli.output_formats
    assert cli.settings_schema == "https://antigravity.google/schemas/v1/settings.json"
    argv = build_headless_cli_args("verify", conversation_id=valid_id)
    assert argv[:3] == ["agy", "-p", "verify"]


def test_hook_hardening_path_norm_camelcase_ast_fallback_and_t38_rtx() -> None:
    root_hook = ROOT / ".agents" / "hooks" / "pre_tool_use_guard.py"
    plugin_hook = ROOT / ".agents" / "plugins" / "hadukenn-sheaf" / "scripts" / "pre_tool_use_guard.py"
    assert root_hook.read_bytes() == plugin_hook.read_bytes()
    assert stat.S_IMODE(plugin_hook.stat().st_mode) & 0o111

    def _run(payload: dict[str, object]) -> dict[str, object]:
        proc = subprocess.run(
            [sys.executable, str(root_hook), "--event", "PreToolUse"],
            cwd=ROOT,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip())

    # H-1: path normalization with ./, .., and workspace root prefix
    for bad_path in (
        "fixtures/./passing/ledger/head.json",
        "fixtures/sub/../test_seal.key",
        "src/../fixtures/test_seal.key",
        f"{ROOT}/src/../fixtures/test_seal.key",
    ):
        res = _run({"toolInput": {"TargetFile": bad_path, "CodeContent": "{}"}})
        assert res["decision"] == "deny", f"Expected deny for {bad_path}: {res}"

    evil_norm = _run(
        {"toolInput": {"TargetFile": "src/./sheaf_verify/evil.py", "CodeContent": "import torch\n"}}
    )
    assert evil_norm["decision"] == "deny"

    # H-2: empty toolInput {} must not shadow toolCall.args, and camelCase keys are honored
    shadowed = _run(
        {
            "toolInput": {},
            "toolCall": {
                "name": "write_to_file",
                "args": {"TargetFile": "src/sheaf_verify/x.py", "CodeContent": "import torch\n"},
            },
        }
    )
    assert shadowed["decision"] == "deny"

    for camel_input in (
        {"targetFile": "src/sheaf_verify/x.py", "codeContent": "import torch\n"},
        {
            "targetFile": "src/sheaf_verify/x.py",
            "replacementChunks": [{"replacementContent": "import torch\n"}],
        },
        {
            "targetFile": "src/sheaf_verify/x.py",
            "oldString": "import json",
            "newString": "import torch",
        },
    ):
        res = _run({"toolInput": camel_input})
        assert res["decision"] == "deny", f"Expected deny for camelCase payload {camel_input}: {res}"

    # H-3: SyntaxError / IndentationError semicolon/compound imports & dynamic imports
    for bad_py in (
        "    x = 1\ny = 2; import torch\n",
        "def f(:\n    if True: import numpy as np\n",
        "import importlib\ntorch = importlib.import_module('torch')\n",
        "x = __import__('numpy')\n",
    ):
        res = _run({"toolInput": {"TargetFile": "src/sheaf_verify/x.py", "ReplacementContent": bad_py}})
        assert res["decision"] == "deny", f"Expected deny for snippet: {bad_py!r} -> {res}"

    # H-3 false-positive guard: strings, comments, relative imports, and stdlib import_module allowed
    for ok_py in (
        "def f(:\n    # import torch\n    s = 'import numpy'\n",
        "from .hashing import sha256_bytes\n",
        "import importlib\nj = importlib.import_module('json')\n",
    ):
        res = _run({"toolInput": {"TargetFile": "src/sheaf_verify/ok.py", "ReplacementContent": ok_py}})
        assert res["decision"] == "allow", f"Unexpected deny for benign snippet: {ok_py!r} -> {res}"

    # H-4: T-38 RTX and 32,768 formatted checks
    for bad_t38 in (
        "vocab = 32768\nhead = '182x181'\ntarget = 'RTX 5070'\n",
        "182x181 on V=32,768 CUDA\n",
        "182x181 on V=32,768 rtx5070\n",
    ):
        res = _run({"toolInput": {"TargetFile": "src/hadukenn_core/head.py", "CodeContent": bad_t38}})
        assert res["decision"] == "deny", f"Expected T-38 deny for {bad_t38!r}"


def test_mcp_lsp_framing_headers_zero_length_and_plugin_skill_path() -> None:
    root_mcp = ROOT / ".agents" / "mcp" / "sheaf_mcp_server.py"
    plugin_mcp = ROOT / ".agents" / "plugins" / "hadukenn-sheaf" / "scripts" / "sheaf_mcp_server.py"
    assert root_mcp.read_bytes() == plugin_mcp.read_bytes()
    assert stat.S_IMODE(plugin_mcp.stat().st_mode) & 0o111

    # M-1(a): Content-Type before Content-Length must not emit spurious Parse error
    req_100 = json.dumps({"jsonrpc": "2.0", "id": 100, "method": "ping", "params": {}}).encode("utf-8")
    frame_ct = (
        b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        + f"Content-Length: {len(req_100)}\r\n\r\n".encode("ascii")
        + req_100
    )
    proc_ct = subprocess.run(
        [sys.executable, str(root_mcp)],
        cwd=ROOT,
        input=frame_ct,
        capture_output=True,
        check=False,
    )
    assert proc_ct.returncode == 0
    assert b"-32700" not in proc_ct.stdout
    _, body_100 = proc_ct.stdout.split(b"\r\n\r\n", 1)
    assert json.loads(body_100.decode("utf-8")) == {"jsonrpc": "2.0", "id": 100, "result": {}}

    # M-1(b): Content-Length <= 0 emits JSON-RPC error and keeps server alive for subsequent frames
    stream_bad_len = (
        b'Content-Length: 0\r\n\r\n{"jsonrpc":"2.0","id":3,"method":"ping"}\n'
        b'Content-Length: -5\r\n\r\n{"jsonrpc":"2.0","id":4,"method":"tools/call",'
        b'"params":{"name":"kernel_pipeline_summary","arguments":{}}}\n'
    )
    proc_zero = subprocess.run(
        [sys.executable, str(root_mcp)],
        cwd=ROOT,
        input=stream_bad_len,
        capture_output=True,
        check=False,
    )
    assert proc_zero.returncode == 0
    raw_out = proc_zero.stdout.decode("utf-8")
    assert '"id":3,"result":{}' in raw_out or '"id": 3' in raw_out
    assert '"id":4' in raw_out
    assert "hadukenn_core.antigravity_sdk" in raw_out
    assert "hadukenn_core.rlm_engine" in raw_out

    # P-1: Plugin SKILL.md references an existing workspace-relative script path
    skill_md = ROOT / ".agents" / "plugins" / "hadukenn-sheaf" / "skills" / "verify-all-gates" / "SKILL.md"
    skill_text = skill_md.read_text(encoding="utf-8")
    assert "python3 scripts/verify_all.py" not in skill_text
    assert ".agents/plugins/hadukenn-sheaf/skills/verify-all-gates/scripts/verify_all.py" in skill_text
    assert (ROOT / ".agents/plugins/hadukenn-sheaf/skills/verify-all-gates/scripts/verify_all.py").is_file()


def test_sdk_hardening_permissions_lifecycle_schema_and_t38_npu() -> None:
    # S-1(a): invalid regex raises ValueError cleanly (never uncaught re.error)
    with pytest.raises(ValueError):
        evaluate_permission(
            "command",
            [{"action": "command", "pattern": "regex:[unclosed", "decision": "deny"}],
            command="ls",
        )

    # S-1(b): deny regex rule still matches dangerous metacharacter commands over allow rules
    cmd_rules = [
        {"action": "command", "pattern": "regex:rm.*", "decision": "deny"},
        {"action": "command", "pattern": "rm $(echo hi)", "decision": "allow"},
    ]
    assert evaluate_permission("command", cmd_rules, command="rm $(echo hi)").decision == "deny"
    # Non-deny allow regex still refuses metacharacter bypass
    allow_only = [{"action": "command", "pattern": "regex:pytest.*", "decision": "allow"}]
    assert evaluate_permission("command", allow_only, command="pytest $(id)").decision == "ask"

    # S-2: LifecycleHooks.run_pre_turn enforces Deny > Ask > Allow precedence
    hooks = LifecycleHooks()
    hooks.pre_turn(lambda _ctx: HookResult("deny", "security block"))
    hooks.pre_turn(lambda _ctx: HookResult("allow", "telemetry ok"))
    res_deny = hooks.run_pre_turn({"id": "1"})
    assert res_deny.decision == "deny"
    assert "security block" in res_deny.reason

    hooks_ask = LifecycleHooks()
    hooks_ask.pre_turn(lambda _ctx: HookResult("allow", "ok"))
    hooks_ask.pre_turn(lambda _ctx: HookResult("force_ask", "needs review"))
    hooks_ask.pre_turn(lambda _ctx: HookResult("auto_approve", "fast"))
    assert hooks_ask.run_pre_turn().decision == "force_ask"

    # S-3: validate_response_schema bool vs int in enum & additionalProperties=False without properties
    with pytest.raises(ValueError):
        validate_response_schema({"v": True}, {"type": "object", "properties": {"v": {"enum": [1, 2]}}})
    with pytest.raises(ValueError):
        validate_response_schema({"v": 1}, {"type": "object", "properties": {"v": {"enum": [True]}}})
    with pytest.raises(ValueError):
        validate_response_schema({"extra": 1}, {"type": "object", "additionalProperties": False})
    assert validate_response_schema({}, {"type": "object", "additionalProperties": False}) is True

    # S-4: T-38 rejects rtx5070 and NPU batch_size > 1 at 32k, allows NPU batch_size=1
    cfg = LiteRTAgentConfig.lightweight()
    with pytest.raises(ValueError):
        run_multimodal_pathway(cfg, vocab_size=32_768, head_type="182x181", hardware_target="rtx5070")
    with pytest.raises(ValueError):
        run_multimodal_pathway(cfg, vocab_size=32_768, head_type="182x181", hardware_target="npu", batch_size=8)
    ok_npu_b1 = run_multimodal_pathway(
        cfg, vocab_size=32_768, head_type="182x181", hardware_target="edge_npu", batch_size=1
    )
    assert ok_npu_b1["ok"] is True
