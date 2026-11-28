#!/usr/bin/env python3
"""Pure-stdlib stdio MCP server with MCP Apps UI metadata for HaDukeNN_LLM.

AST purity invariant: top-level and function-level AST imports MUST remain
strictly inside Python 3.11+ stdlib | {"sheaf_verify"}. Optional M3 inspection
uses importlib.import_module inside try/except so AST checks stay 100% pure.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = (
    Path.cwd().resolve()
    if (Path.cwd() / "src" / "sheaf_verify").is_dir()
    else next(
        (p for p in _THIS_FILE.parents if (p / "src" / "sheaf_verify").is_dir()),
        _THIS_FILE.parents[2],
    )
)
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sheaf_verify.control_graph import problems as control_graph_problems  # noqa: E402
from sheaf_verify.gate_coverage import assess as assess_gate_coverage  # noqa: E402
from sheaf_verify.gate_coverage import load_events as load_gate_events  # noqa: E402
from sheaf_verify.gates import all_gates  # noqa: E402
from sheaf_verify.hashing import SheafError, is_sha256  # noqa: E402
from sheaf_verify.rlm_guards import RLMBudgetPolicy, RLMBudgetState, SiipVerifyGate  # noqa: E402
from sheaf_verify.seal import load_key  # noqa: E402
from sheaf_verify.store import Store  # noqa: E402
from sheaf_verify.verify import verify_store  # noqa: E402

UI_RESOURCE_URI = "ui://hadukenn-sheaf/dashboard"

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HaDukeNN Sheaf Verification Dashboard</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 16px; color: #111827; background: #f9fafb; }
  .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  h1 { font-size: 16px; margin: 0 0 8px 0; }
  code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
  <div class="card">
    <h1>HaDukeNN Sheaf Verification Control Plane</h1>
    <p>Pure-stdlib MCP server exposing read-only <code>sheaf_verify</code> gates, canonical anchor checks, RLM budget guards, and SIIP mode validation.</p>
    <p>Canonical Anchor: <code>307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4</code></p>
  </div>
</body>
</html>
"""

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "sheaf_anchor",
        "description": "Read-only lookup of the canonical ledger head anchor for a sheaf_verify store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "store": {
                    "type": "string",
                    "description": "Path to the artifact store directory.",
                    "default": "fixtures/passing",
                },
                "store_path": {
                    "type": "string",
                    "description": "Alias for store path.",
                },
                "path": {
                    "type": "string",
                    "description": "Alias for store path.",
                },
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
    {
        "name": "sheaf_verify_store",
        "description": "Read-only verification of manifests, payloads, hash-chained ledger, and gates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "store": {"type": "string", "default": "fixtures/passing"},
                "seal_key_file": {"type": "string", "default": "fixtures/test_seal.key"},
                "allow_unsigned": {"type": "boolean", "default": False},
                "strict": {"type": "boolean", "default": False},
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
    {
        "name": "sheaf_gate_coverage",
        "description": "Inspect control-graph gate definitions and optional gate-trace coverage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_path": {
                    "type": "string",
                    "description": "Optional JSONL gate trace path to assess coverage.",
                },
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
    {
        "name": "rlm_budget_check",
        "description": "Validate recursive LM depth, batch floor, token cap, and thinking-reserve headroom.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "default": 1},
                "max_subcalls": {"type": "integer", "default": 16},
                "max_total_tokens": {"type": "integer", "default": 100000},
                "min_batch_chars": {"type": "integer", "default": 10000},
                "thinking_reserve": {"type": "integer", "default": 1024},
                "depth": {"type": "integer", "default": 1},
                "chars": {"type": "integer", "default": 10000},
                "tokens": {"type": "integer", "default": 1000},
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
    {
        "name": "siip_gate_check",
        "description": "Evaluate SIIP dual-mode speculative/formal commit acceptance and optional cosine damping.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "default": "formal"},
                "f_k": {"type": "number", "default": 1.0},
                "f_cand": {"type": "number", "default": 0.5},
                "x_cand": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [0.1, 0.2],
                },
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
    {
        "name": "kernel_pipeline_summary",
        "description": "Inspect available hadukenn_core kernel pipeline and sheaf_verify gate metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_steps": {"type": "integer", "default": 4},
                "m_streams": {"type": "integer", "default": 4},
            },
        },
        "_meta": {"ui": {"resourceUri": UI_RESOURCE_URI}},
    },
]


def _resolve_repo_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _handle_sheaf_anchor(args: dict[str, Any]) -> dict[str, Any]:
    raw_store = args.get("store") or args.get("store_path") or args.get("path") or "fixtures/passing"
    store_dir = _resolve_repo_path(str(raw_store))
    head_path = store_dir / "ledger" / "head.json"
    if not head_path.is_file():
        return _tool_result({"error": f"ledger head is missing at {head_path}"}, is_error=True)
    head = json.loads(head_path.read_text(encoding="utf-8"))
    digest = head.get("head_sha256") if isinstance(head, dict) else None
    if not is_sha256(digest):
        return _tool_result({"error": f"invalid ledger head hash at {head_path}"}, is_error=True)
    return _tool_result(
        {
            "ok": True,
            "store": str(raw_store),
            "anchor_line": f"sheaf-verify.anchor.v1 {digest}",
            "head_sha256": digest,
        }
    )


def _handle_sheaf_verify_store(args: dict[str, Any]) -> dict[str, Any]:
    raw_store = str(args.get("store") or "fixtures/passing")
    allow_unsigned = bool(args.get("allow_unsigned", False))
    strict = bool(args.get("strict", False))
    raw_key = args.get("seal_key_file", "fixtures/test_seal.key")
    key = None if allow_unsigned else load_key(_resolve_repo_path(str(raw_key)))
    report = verify_store(Store(_resolve_repo_path(raw_store)), key=key, allow_unsigned=allow_unsigned, strict=strict)
    return _tool_result(
        {
            "ok": report.ok,
            "head": report.head,
            "gates_total": len(report.gate_rows),
            "gates_passed": sum(1 for row in report.gate_rows if row.get("status") == "pass"),
            "problems": list(report.problems),
        },
        is_error=not report.ok,
    )


def _handle_sheaf_gate_coverage(args: dict[str, Any]) -> dict[str, Any]:
    gates = all_gates()
    cg_problems = control_graph_problems(_REPO_ROOT)
    payload: dict[str, Any] = {
        "ok": len(cg_problems) == 0,
        "gate_count": len(gates),
        "gate_ids": [g.id for g in gates],
        "control_graph_problems": cg_problems,
    }
    trace_raw = args.get("trace_path")
    if isinstance(trace_raw, str) and trace_raw.strip():
        events = load_gate_events(_resolve_repo_path(trace_raw.strip()))
        rep = assess_gate_coverage(events)
        payload["trace_coverage_ok"] = rep.ok
        payload["trace_problems"] = rep.problems
        payload["ok"] = payload["ok"] and rep.ok
    return _tool_result(payload, is_error=not payload["ok"])


def _handle_rlm_budget_check(args: dict[str, Any]) -> dict[str, Any]:
    policy = RLMBudgetPolicy(
        max_depth=int(args.get("max_depth", 1)),
        max_subcalls=int(args.get("max_subcalls", 16)),
        max_total_tokens=int(args.get("max_total_tokens", 100_000)),
        min_batch_chars=int(args.get("min_batch_chars", 10_000)),
        thinking_reserve=int(args.get("thinking_reserve", 1024)),
    )
    state = RLMBudgetState()
    res = policy.check_subcall(
        state,
        depth=int(args.get("depth", 1)),
        chars=int(args.get("chars", 10_000)),
        tokens=int(args.get("tokens", 1000)),
    )
    return _tool_result(res)


def _handle_siip_gate_check(args: dict[str, Any]) -> dict[str, Any]:
    gate = SiipVerifyGate(mode=str(args.get("mode", "formal")))
    x_cand = args.get("x_cand", [0.1, 0.2])
    res = gate.verify_commit(
        f_k=float(args.get("f_k", 1.0)),
        f_cand=float(args.get("f_cand", 0.5)),
        x_cand=[float(v) for v in x_cand],
    )
    return _tool_result(res)


def _handle_kernel_pipeline_summary(args: dict[str, Any]) -> dict[str, Any]:
    n_steps = int(args.get("n_steps", 4))
    m_streams = int(args.get("m_streams", 4))
    available_modules: list[str] = []
    for mod_name in (
        "hadukenn_core.kernel_state",
        "hadukenn_core.rlm_engine",
        "hadukenn_core.rtx5070_frontier",
        "hadukenn_core.c0_universal_kernel",
        "hadukenn_core.universal_kernel",
        "hadukenn_core.sheaf_bridge",
        "hadukenn_core.antigravity_sdk",
    ):
        try:
            importlib.import_module(mod_name)
            available_modules.append(mod_name)
        except Exception:  # noqa: BLE001
            continue
    return _tool_result(
        {
            "ok": True,
            "n_steps": n_steps,
            "m_streams": m_streams,
            "gate_count": len(all_gates()),
            "available_modules": available_modules,
        }
    )


_TOOL_HANDLERS = {
    "sheaf_anchor": _handle_sheaf_anchor,
    "sheaf_verify_store": _handle_sheaf_verify_store,
    "sheaf_gate_coverage": _handle_sheaf_gate_coverage,
    "rlm_budget_check": _handle_rlm_budget_check,
    "siip_gate_check": _handle_siip_gate_check,
    "kernel_pipeline_summary": _handle_kernel_pipeline_summary,
}


def _rpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    if "id" not in msg:
        return None
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

    if method == "initialize":
        return _rpc_result(
            req_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "extensions": {
                        "io.modelcontextprotocol/ui": {"version": "2025-03-26"},
                        "com.google.antigravity/ui": {"entrypoints": ["aux_pane"]},
                    },
                },
                "serverInfo": {"name": "hadukenn-sheaf-mcp", "version": "1.2.0"},
            },
        )
    if method == "ping":
        return _rpc_result(req_id, {})
    if method == "tools/list":
        return _rpc_result(req_id, {"tools": TOOL_DEFINITIONS})
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        handler = _TOOL_HANDLERS.get(str(tool_name))
        if handler is None:
            return _rpc_result(req_id, _tool_result({"error": f"Unknown tool: {tool_name}"}, is_error=True))
        try:
            return _rpc_result(req_id, handler(arguments))
        except (SheafError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return _rpc_result(req_id, _tool_result({"error": str(exc)}, is_error=True))
    if method == "resources/list":
        return _rpc_result(
            req_id,
            {
                "resources": [
                    {
                        "uri": UI_RESOURCE_URI,
                        "name": "HaDukeNN Sheaf Verification Dashboard",
                        "description": "Self-contained MCP Apps UI summary for sheaf_verify gates and canonical anchor.",
                        "mimeType": "text/html;profile=mcp-app",
                    }
                ]
            },
        )
    if method == "resources/read":
        uri = params.get("uri")
        if uri == UI_RESOURCE_URI:
            return _rpc_result(
                req_id,
                {
                    "contents": [
                        {
                            "uri": UI_RESOURCE_URI,
                            "mimeType": "text/html;profile=mcp-app",
                            "text": DASHBOARD_HTML,
                        }
                    ]
                },
            )
        return _rpc_error(req_id, -32002, f"Resource not found: {uri}")

    return _rpc_error(req_id, -32601, f"Method not found: {method}")


_HEADER_LINE_RE = re.compile(rb"^[A-Za-z0-9-]+:\s*.*$")


class _FrameError(Exception):
    def __init__(self, code: int, message: str, framing: str = "ndjson") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.framing = framing


def _read_exact(stream: Any, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stdin_buf: Any) -> tuple[dict[str, Any] | None, str]:
    while True:
        raw_line = stdin_buf.readline()
        if not raw_line:
            return None, "ndjson"
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not stripped.startswith((b"{", b"[")) and _HEADER_LINE_RE.match(stripped):
            header_lines: list[bytes] = [stripped]
            while True:
                hdr = stdin_buf.readline()
                if not hdr or hdr.strip() == b"":
                    break
                header_lines.append(hdr.strip())
            headers: dict[str, str] = {}
            for hline in header_lines:
                key, sep, val = hline.partition(b":")
                if sep:
                    headers[key.strip().lower().decode("ascii", errors="replace")] = val.strip().decode(
                        "ascii", errors="replace"
                    )
            raw_len = headers.get("content-length")
            if raw_len is None:
                raise _FrameError(-32600, "Missing Content-Length header", "lsp")
            try:
                length = int(raw_len)
            except ValueError as exc:
                raise _FrameError(-32700, f"Invalid Content-Length header {raw_len!r}: {exc}", "lsp") from exc
            if length <= 0:
                raise _FrameError(-32600, f"Invalid Content-Length: {length}", "lsp")
            body = _read_exact(stdin_buf, length)
            if not body:
                return None, "lsp"
            if len(body) < length:
                raise _FrameError(-32700, f"Truncated LSP payload ({len(body)}/{length} bytes)", "lsp")
            try:
                return json.loads(body.decode("utf-8")), "lsp"
            except Exception as exc:  # noqa: BLE001
                raise _FrameError(-32700, f"Parse error: {exc}", "lsp") from exc
        try:
            return json.loads(stripped.decode("utf-8")), "ndjson"
        except Exception as exc:  # noqa: BLE001
            raise _FrameError(-32700, f"Parse error: {exc}", "ndjson") from exc


def _write_frame(stdout_buf: Any, msg: dict[str, Any], framing: str) -> None:
    payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    if framing == "lsp":
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        stdout_buf.write(header + payload)
    else:
        stdout_buf.write(payload + b"\n")
    stdout_buf.flush()


def main() -> int:
    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer
    while True:
        try:
            msg, framing = _read_frame(stdin_buf)
        except _FrameError as exc:
            _write_frame(stdout_buf, _rpc_error(None, exc.code, exc.message), exc.framing)
            continue
        except Exception as exc:  # noqa: BLE001
            _write_frame(stdout_buf, _rpc_error(None, -32700, f"Parse error: {exc}"), "ndjson")
            continue
        if msg is None:
            break
        if not isinstance(msg, dict):
            _write_frame(stdout_buf, _rpc_error(None, -32600, "Invalid Request"), framing)
            continue
        response = handle_message(msg)
        if response is not None:
            _write_frame(stdout_buf, response, framing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
