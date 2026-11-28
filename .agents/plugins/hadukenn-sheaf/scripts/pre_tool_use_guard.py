#!/usr/bin/env python3
"""Pure-stdlib Antigravity lifecycle & PreToolUse guardrail hook.

Always exits 0 and emits a single valid JSON object on stdout so blocks are
enforced deterministically (non-zero hook exits fail open in Antigravity).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import posixpath
import re
import sys
import textwrap
import tokenize
from pathlib import Path, PurePosixPath
from typing import Any

FORBIDDEN_BINARY_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".so", ".dll"}
)
PROTECTED_PREFIXES = (
    "fixtures/passing/",
    "fixtures/passing",
    "fixtures/test_seal.key",
    "iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.json",
    "iterations/t49-hst-foundation-stage/T49_HST_FOUNDATION_RESULTS.json",
)
DRIVE_BACKSLASH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:" + (chr(92) * 2))
FROM_CLAUSE_RE = re.compile(r"(?:^|[;\s:])from\s+([A-Za-z_][A-Za-z0-9_]*)")
IMPORT_CLAUSE_RE = re.compile(r"(?:^|[;\s:])import\s+([A-Za-z_][A-Za-z0-9_]*)")
DYNAMIC_IMPORT_RE = re.compile(
    r"(?:importlib\s*\.\s*import_module|import_module|__import__)\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]"
)
ALLOWED_SHEAF_IMPORTS = frozenset(sys.stdlib_module_names) | frozenset({"sheaf_verify", "__future__"})

_THIS_FILE = Path(__file__).resolve()
_DEFAULT_WORKSPACE_ROOTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        posixpath.normpath(str(p).replace(chr(92), "/"))
        for p in (
            Path.cwd().resolve(),
            *(
                parent
                for parent in _THIS_FILE.parents
                if (parent / "src" / "sheaf_verify").is_dir() or (parent / ".agents").is_dir()
            ),
        )
    )
)


def _allow(reason: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"decision": "allow"}
    if reason:
        out["reason"] = reason
    return out


def _deny(reason: str) -> dict[str, Any]:
    return {"decision": "deny", "reason": reason}


def _is_nonempty_value(val: Any) -> bool:
    return val is not None and val != "" and val != [] and val != {}


def _extract_args(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    tool_call = payload.get("toolCall")
    if isinstance(tool_call, dict) and isinstance(tool_call.get("args"), dict):
        merged.update(tool_call["args"])
    tool_input = payload.get("toolInput")
    if isinstance(tool_input, dict):
        for key, val in tool_input.items():
            if _is_nonempty_value(val) or key not in merged:
                merged[key] = val
    if merged:
        return merged
    return payload


def _extract_target_path(args: dict[str, Any]) -> str:
    for key in (
        "TargetFile",
        "targetFile",
        "file_path",
        "filePath",
        "path",
        "target_file",
        "FilePath",
    ):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_content(args: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "content",
        "CodeContent",
        "codeContent",
        "ReplacementContent",
        "replacementContent",
        "newString",
        "new_string",
    ):
        val = args.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    if not parts:
        for key in ("oldString", "old_string"):
            val = args.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
    for chunks_key in ("ReplacementChunks", "replacementChunks"):
        chunks = args.get(chunks_key)
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    rep = chunk.get("ReplacementContent") or chunk.get("replacementContent")
                    if isinstance(rep, str) and rep:
                        parts.append(rep)
    edits = args.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict):
                rep = (
                    edit.get("replacement")
                    or edit.get("ReplacementContent")
                    or edit.get("replacementContent")
                    or edit.get("newString")
                    or edit.get("new_string")
                )
                if isinstance(rep, str) and rep:
                    parts.append(rep)
    return "\n".join(parts)


def _normalize_repo_rel(raw_path: str, workspace_paths: list[str] | None = None) -> str:
    cleaned = raw_path.strip()
    if not cleaned:
        return ""
    norm = cleaned.replace(chr(92), "/")
    while "//" in norm:
        norm = norm.replace("//", "/")
    norm = posixpath.normpath(norm)

    roots = list(_DEFAULT_WORKSPACE_ROOTS)
    if isinstance(workspace_paths, list):
        for ws in workspace_paths:
            if isinstance(ws, str) and ws.strip():
                ws_norm = posixpath.normpath(ws.strip().replace(chr(92), "/"))
                if ws_norm not in roots:
                    roots.append(ws_norm)
    for root in sorted(roots, key=len, reverse=True):
        if root and root != "/" and norm.startswith(f"{root}/"):
            norm = norm[len(root) + 1 :]
            break
        if norm == root:
            norm = ""
            break
    while norm.startswith("./"):
        norm = norm[2:]
    return "" if norm == "." else norm


def _is_sheaf_verify_py(norm_path: str) -> bool:
    if not norm_path.endswith(".py"):
        return False
    return norm_path.startswith("src/sheaf_verify/") or "/src/sheaf_verify/" in norm_path


def _check_dynamic_import_call(node: ast.Call) -> str | None:
    is_dunder_import = isinstance(node.func, ast.Name) and node.func.id == "__import__"
    is_import_module = (
        (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        or (isinstance(node.func, ast.Name) and node.func.id == "import_module")
    )
    if not (is_dunder_import or is_import_module):
        return None

    target_arg: ast.expr | None = node.args[0] if node.args else None
    if target_arg is None:
        for kw in node.keywords:
            if kw.arg == "name":
                target_arg = kw.value
                break

    if isinstance(target_arg, ast.Constant) and isinstance(target_arg.value, str):
        top = target_arg.value.split(".", 1)[0]
        if top and top not in ALLOWED_SHEAF_IMPORTS:
            return f"Non-stdlib dynamic import {top!r} is forbidden inside src/sheaf_verify/"
        return None

    if is_dunder_import:
        return "Dynamic __import__() call is forbidden inside src/sheaf_verify/"
    return None


def _scan_tokens_for_imports(tokens: list[tuple[int, str]]) -> str | None:
    filtered = [
        (tok_type, tok_str)
        for tok_type, tok_str in tokens
        if tok_type
        not in {
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        }
    ]
    n = len(filtered)
    i = 0
    while i < n:
        tok_type, tok_str = filtered[i]
        if tok_type == tokenize.NAME and tok_str == "from":
            j = i + 1
            if j < n:
                nt, ns = filtered[j]
                if nt == tokenize.OP and ns in {".", "..."}:
                    # Relative import: skip until after the imported names clause
                    while j < n and not (filtered[j][0] == tokenize.NAME and filtered[j][1] == "import"):
                        j += 1
                    i = j + 1
                    continue
                if nt == tokenize.NAME:
                    if ns not in ALLOWED_SHEAF_IMPORTS:
                        return f"Non-stdlib import {ns!r} is forbidden inside src/sheaf_verify/"
                    while j < n and not (filtered[j][0] == tokenize.NAME and filtered[j][1] == "import"):
                        j += 1
                    # Skip the `import a, b` clause on this statement
                    if j < n and filtered[j] == (tokenize.NAME, "import"):
                        j += 1
                        if j < n and filtered[j] == (tokenize.OP, "("):
                            depth = 1
                            j += 1
                            while j < n and depth > 0:
                                if filtered[j] == (tokenize.OP, "("):
                                    depth += 1
                                elif filtered[j] == (tokenize.OP, ")"):
                                    depth -= 1
                                j += 1
                        else:
                            while j < n and filtered[j] != (tokenize.OP, ";"):
                                if filtered[j][0] == tokenize.NAME and filtered[j][1] in {
                                    "import",
                                    "from",
                                    "def",
                                    "class",
                                    "if",
                                    "for",
                                    "while",
                                    "return",
                                }:
                                    break
                                j += 1
                    i = j
                    continue
        elif tok_type == tokenize.NAME and tok_str == "import":
            j = i + 1
            expect_mod = True
            while j < n:
                nt, ns = filtered[j]
                if expect_mod:
                    if nt == tokenize.NAME:
                        if ns not in ALLOWED_SHEAF_IMPORTS:
                            return f"Non-stdlib import {ns!r} is forbidden inside src/sheaf_verify/"
                        expect_mod = False
                        j += 1
                        continue
                    break
                if nt == tokenize.OP and ns == ".":
                    j += 1
                    if j < n and filtered[j][0] == tokenize.NAME:
                        j += 1
                    continue
                if nt == tokenize.NAME and ns == "as":
                    j += 2
                    continue
                if nt == tokenize.OP and ns == ",":
                    expect_mod = True
                    j += 1
                    continue
                break
            i = j
            continue
        elif tok_type == tokenize.NAME and tok_str in {"__import__", "import_module"}:
            if i + 2 < n and filtered[i + 1] == (tokenize.OP, "("):
                arg_type, arg_str = filtered[i + 2]
                if arg_type == tokenize.STRING:
                    try:
                        mod_val = ast.literal_eval(arg_str)
                    except Exception:  # noqa: BLE001
                        mod_val = ""
                    if isinstance(mod_val, str):
                        top = mod_val.split(".", 1)[0]
                        if top and top not in ALLOWED_SHEAF_IMPORTS:
                            return f"Non-stdlib dynamic import {top!r} is forbidden inside src/sheaf_verify/"
                elif tok_str == "__import__":
                    return "Dynamic __import__() call is forbidden inside src/sheaf_verify/"
        i += 1
    return None


def _tokenize_safe(snippet: str) -> tuple[list[tuple[int, str]], int]:
    tokens: list[tuple[int, str]] = []
    last_line = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(snippet).readline):
            tokens.append((tok.type, tok.string))
            last_line = max(last_line, tok.end[0])
    except (tokenize.TokenError, IndentationError):
        pass
    return tokens, last_line


def _check_sheaf_verify_stdlib(content: str) -> str | None:
    if not content.strip():
        return None
    tree: ast.AST | None = None
    for candidate in (content, textwrap.dedent(content)):
        try:
            tree = ast.parse(candidate)
            break
        except SyntaxError:
            tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top not in ALLOWED_SHEAF_IMPORTS:
                        return f"Non-stdlib import {top!r} is forbidden inside src/sheaf_verify/"
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".", 1)[0]
                if top not in ALLOWED_SHEAF_IMPORTS:
                    return f"Non-stdlib import {top!r} is forbidden inside src/sheaf_verify/"
            elif isinstance(node, ast.Call):
                dyn_violation = _check_dynamic_import_call(node)
                if dyn_violation is not None:
                    return dyn_violation
        return None

    # Fallback for partial/SyntaxError/IndentationError replacement snippets:
    # 1. Tokenize full snippet as far as possible (preserves string/comment boundaries)
    tokens, last_line = _tokenize_safe(content)
    violation = _scan_tokens_for_imports(tokens)
    if violation is not None:
        return violation

    # 2. If full tokenization aborted early (e.g., IndentationError on line 2+),
    # tokenize each remaining line independently (stripping leading indentation)
    lines = content.splitlines()
    for idx in range(max(0, last_line - 1), len(lines)):
        line = lines[idx]
        line_tokens, _ = _tokenize_safe(line.lstrip())
        if line_tokens:
            violation = _scan_tokens_for_imports(line_tokens)
            if violation is not None:
                return violation
            continue
        # 3. Last-resort regex scan on untokenizable line
        stripped = line.split("#", 1)[0]
        for regex in (FROM_CLAUSE_RE, DYNAMIC_IMPORT_RE):
            for match in regex.finditer(stripped):
                top = match.group(1).split(".", 1)[0]
                if top not in ALLOWED_SHEAF_IMPORTS:
                    return f"Non-stdlib import {top!r} is forbidden inside src/sheaf_verify/"
        if not FROM_CLAUSE_RE.search(stripped):
            for match in IMPORT_CLAUSE_RE.finditer(stripped):
                top = match.group(1).split(".", 1)[0]
                if top not in ALLOWED_SHEAF_IMPORTS:
                    return f"Non-stdlib import {top!r} is forbidden inside src/sheaf_verify/"
    return None


def evaluate_pre_tool_use(payload: dict[str, Any]) -> dict[str, Any]:
    args = _extract_args(payload)
    raw_path = _extract_target_path(args)
    content = _extract_content(args)

    if DRIVE_BACKSLASH_RE.search(raw_path):
        return _deny(f"Windows drive-letter backslash path is forbidden in target path: {raw_path!r}")
    if DRIVE_BACKSLASH_RE.search(content):
        return _deny("Windows drive-letter backslash path is forbidden in repository content.")

    workspace_paths = payload.get("workspacePaths") if isinstance(payload.get("workspacePaths"), list) else None
    norm_path = _normalize_repo_rel(raw_path, workspace_paths=workspace_paths)
    if norm_path:
        for protected in PROTECTED_PREFIXES:
            if protected.endswith("/"):
                if norm_path.startswith(protected) or f"/{protected}" in norm_path:
                    return _deny(f"Protected canonical/sealed path cannot be modified: {norm_path!r}")
            else:
                if (
                    norm_path == protected
                    or norm_path.startswith(f"{protected}/")
                    or f"/{protected}" in norm_path
                ):
                    return _deny(f"Protected canonical/sealed path cannot be modified: {norm_path!r}")

        suffix = PurePosixPath(norm_path).suffix.lower()
        if suffix in FORBIDDEN_BINARY_SUFFIXES:
            return _deny(f"Forbidden binary weight or shared-library suffix {suffix!r} for {norm_path!r}")

        if _is_sheaf_verify_py(norm_path):
            violation = _check_sheaf_verify_stdlib(content)
            if violation is not None:
                return _deny(violation)

    lowered = content.lower()
    has_182x181 = "182x181" in lowered or (
        "182" in lowered
        and "181" in lowered
        and ("goodman" in lowered or "cluster_is" in lowered or "factored" in lowered)
    )
    has_32k = any(tok in lowered for tok in ("32768", "32k", "32,768", "32_768"))
    has_gpu = any(tok in lowered for tok in ("cuda", "gpu", "rtx"))
    if has_182x181 and has_32k and has_gpu:
        return _deny("Empirical exclusion (AGENTS.md T-38): 182x181 factored head is banned on 32k GPU targets.")

    return _allow()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event", default="PreToolUse")
    args, _ = parser.parse_known_args(argv)

    try:
        raw_stdin = sys.stdin.read()
        payload = json.loads(raw_stdin) if raw_stdin.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:  # noqa: BLE001
        payload = {}

    event = str(payload.get("hookEvent") or args.event)
    if event == "PostToolUse":
        out: dict[str, Any] = {}
    elif event in {"SessionStart", "PreInvocation", "PostInvocation"}:
        out = {"injectSteps": [], "terminationBehavior": ""}
    elif event == "Stop":
        out = {"decision": "allow", "reason": "ok"}
    else:
        try:
            out = evaluate_pre_tool_use(payload)
        except Exception as exc:  # noqa: BLE001
            out = _deny(f"Guardrail evaluation error: {exc}")

    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
