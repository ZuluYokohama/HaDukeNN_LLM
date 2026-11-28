"""Shared opaque-box E2E test infrastructure for HaDukeNN_LLM (E2E-M1 through E2E-M4).

Zero target-package (`sheaf_verify.*`, `hadukenn_core.*`, `scripts.*`, `numpy`, `scipy`, `torch`)
imports appear at module top level so `pytest --collect-only tests_e2e/` succeeds cleanly
prior to M2-M4 implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import pkgutil
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_READONLY_FILES = (
    Path("fixtures/test_seal.key"),
    Path("iterations/t46-tri-machine-hal/T46_TRI_MACHINE_RESULTS.json"),
    Path("iterations/t49-hst-foundation-stage/T49_HST_FOUNDATION_RESULTS.json"),
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the 4 E2E tier markers."""
    config.addinivalue_line("markers", "tier1: Tier 1 isolated happy-path feature tests")
    config.addinivalue_line("markers", "tier2: Tier 2 boundary, negative, and exclusion tests")
    config.addinivalue_line("markers", "tier3: Tier 3 cross-feature pairwise combination tests")
    config.addinivalue_line("markers", "tier4: Tier 4 real-world end-to-end scenario tests")


def _build_pythonpath(existing: str | None) -> str:
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if existing:
        for item in existing.split(os.pathsep):
            if item and item not in entries:
                entries.append(item)
    return os.pathsep.join(entries)


def no_gpu_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a clean GPU-disabled subprocess environment with repo PYTHONPATH."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = _build_pythonpath(env.get("PYTHONPATH"))
    if extra:
        env.update(extra)
    return env


def run_sheaf_cli(
    args: Sequence[str],
    *,
    store: Path | None = None,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Invoke `python -m sheaf_verify ...`, appending `--store` after the subcommand."""
    cmd_args = list(args)
    has_store_flag = any(a == "--store" or a.startswith("--store=") for a in cmd_args)
    if store is not None and not has_store_flag:
        cmd_args.extend(["--store", str(store)])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = _build_pythonpath(env.get("PYTHONPATH"))
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "sheaf_verify", *cmd_args],
        cwd=cwd or REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _normalize_module_name(package_name: str, mod_name: str) -> str:
    if mod_name == package_name or mod_name.startswith(f"{package_name}."):
        return mod_name
    return f"{package_name}.{mod_name}"


def _has_private_segment(package_name: str, full_mod: str) -> bool:
    if full_mod == package_name:
        return False
    prefix = f"{package_name}."
    rel = full_mod[len(prefix) :] if full_mod.startswith(prefix) else full_mod
    return any(part.startswith("_") for part in rel.split("."))


def resolve_public_symbol(
    package_name: str,
    symbol_name: str,
    preferred_modules: tuple[str, ...] = (),
) -> Any:
    """Resolve a public symbol at test runtime, skipping `_`-prefixed submodules (e.g., `__main__`)."""
    if not symbol_name or symbol_name.startswith("_"):
        raise AssertionError(f"Refusing to resolve private/empty symbol {symbol_name!r}")

    tried: list[str] = []

    for raw_mod in preferred_modules:
        full_mod = _normalize_module_name(package_name, raw_mod)
        if _has_private_segment(package_name, full_mod):
            continue
        tried.append(full_mod)
        try:
            mod = importlib.import_module(full_mod)
        except ImportError:
            continue
        if hasattr(mod, symbol_name):
            return getattr(mod, symbol_name)

    try:
        pkg = importlib.import_module(package_name)
    except ImportError as exc:
        raise AssertionError(f"Cannot import package {package_name!r} for symbol {symbol_name!r}: {exc}") from exc

    if package_name not in tried:
        tried.append(package_name)
    if hasattr(pkg, symbol_name):
        return getattr(pkg, symbol_name)

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is not None:
        for _importer, mod_name, _ispkg in pkgutil.walk_packages(pkg_path, prefix=f"{package_name}."):
            if _has_private_segment(package_name, mod_name):
                continue
            if mod_name in tried:
                continue
            tried.append(mod_name)
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            if hasattr(mod, symbol_name):
                return getattr(mod, symbol_name)

    raise AssertionError(f"Missing public symbol {symbol_name!r} in {package_name!r} (checked: {tried})")


def resolve_any_public_symbol(
    package_name: str,
    candidate_names: tuple[str, ...],
    preferred_modules: tuple[str, ...] = (),
) -> Any:
    """Resolve the first matching public symbol from `candidate_names` or fail with AssertionError."""
    if not candidate_names:
        raise AssertionError("candidate_names must be non-empty")
    for name in candidate_names:
        if not name or name.startswith("_"):
            raise AssertionError(f"Refusing private/empty candidate symbol {name!r}")

    errors: list[str] = []
    for name in candidate_names:
        try:
            return resolve_public_symbol(package_name, name, preferred_modules=preferred_modules)
        except AssertionError as exc:
            errors.append(str(exc))

    raise AssertionError(
        f"Missing any public symbol from {candidate_names!r} in {package_name!r}; details: {errors[-1]}"
    )


def snapshot_readonly_hashes(root: Path | None = None) -> dict[str, str]:
    """Compute SHA-256 digests of all immutable fixture and sealed campaign files."""
    base = (root or REPO_ROOT).resolve()
    hashes: dict[str, str] = {}

    passing_dir = base / "fixtures" / "passing"
    if not passing_dir.is_dir():
        raise AssertionError(f"Missing immutable directory: {passing_dir}")
    passing_files = sorted(p for p in passing_dir.rglob("*") if p.is_file())
    if not passing_files:
        raise AssertionError(f"Empty immutable directory: {passing_dir}")

    for fpath in passing_files:
        rel = fpath.relative_to(base).as_posix()
        hashes[rel] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    for rel_path in _READONLY_FILES:
        fpath = base / rel_path
        if not fpath.is_file():
            raise AssertionError(f"Missing read-only file: {fpath}")
        hashes[rel_path.as_posix()] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    return hashes


def assert_readonly_unchanged(baseline: dict[str, str], root: Path | None = None) -> None:
    """Assert that all immutable fixtures and sealed files match `baseline` SHA-256 digests."""
    current = snapshot_readonly_hashes(root)
    if current == baseline:
        return
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    modified = sorted(k for k in set(current) & set(baseline) if current[k] != baseline[k])
    raise AssertionError(
        f"Read-only fixtures/sealed artifacts mutated! added={added}, removed={removed}, modified={modified}"
    )


def _resolve_command_argv(cmd: str | Path | Sequence[str]) -> list[str]:
    if isinstance(cmd, Path):
        return [sys.executable, str(cmd)] if cmd.suffix == ".py" else [str(cmd)]
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    else:
        parts = list(cmd)
    if not parts:
        raise AssertionError("Hook or server command must be non-empty")
    if Path(parts[0]).name in {"python", "python3"}:
        parts[0] = sys.executable
    return parts


def run_hook(
    hook_cmd: str | Path | Sequence[str],
    payload: dict[str, Any],
    *,
    cwd: Path | None = None,
    env_extra: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run an Antigravity hook subprocess with JSON stdin and return `(CompletedProcess, parsed_json)`."""
    argv = _resolve_command_argv(hook_cmd)
    proc = subprocess.run(
        argv,
        input=json.dumps(payload),
        cwd=cwd or REPO_ROOT,
        env=no_gpu_env(env_extra),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    raw_out = proc.stdout.strip()
    if not raw_out:
        raise AssertionError(f"Hook {argv!r} produced empty stdout (rc={proc.returncode}, stderr={proc.stderr!r})")
    try:
        parsed = json.loads(raw_out)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Hook {argv!r} emitted invalid JSON {raw_out!r} (rc={proc.returncode}, stderr={proc.stderr!r})"
        ) from exc
    if not isinstance(parsed, dict):
        raise AssertionError(f"Hook {argv!r} JSON output must be an object, got {type(parsed).__name__}")
    return proc, parsed


class McpStdioClient:
    """Bounded JSON-RPC 2.0 client supporting newline-delimited and Content-Length stdio framing."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc
        self._messages: queue.Queue[dict[str, Any] | Exception | None] = queue.Queue()
        self._next_id = 1
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    @staticmethod
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

    def _pump_stdout(self) -> None:
        assert self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            while True:
                raw_line = stdout.readline()
                if not raw_line:
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith(b"content-length:"):
                    length = int(stripped.split(b":", 1)[1].strip())
                    while True:
                        hdr = stdout.readline()
                        if not hdr:
                            raise EOFError("MCP server closed stdout during Content-Length headers")
                        if hdr.strip() == b"":
                            break
                    payload_bytes = self._read_exact(stdout, length)
                    if len(payload_bytes) < length:
                        raise EOFError(
                            f"MCP server closed stdout after {len(payload_bytes)}/{length} Content-Length bytes"
                        )
                    parsed = json.loads(payload_bytes.decode("utf-8"))
                else:
                    parsed = json.loads(stripped.decode("utf-8"))
                if isinstance(parsed, dict):
                    self._messages.put(parsed)
        except Exception as exc:  # noqa: BLE001
            self._messages.put(exc)
        finally:
            self._messages.put(None)

    def _read_stderr_nonblocking(self) -> str:
        if self._proc.stderr is not None and self._proc.poll() is not None:
            return self._proc.stderr.read().decode("utf-8", errors="replace")
        return ""

    def _read_message(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stderr_out = self._read_stderr_nonblocking()
                raise AssertionError(f"Timed out waiting for MCP JSON-RPC response (stderr={stderr_out!r})")
            try:
                item = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                stderr_out = self._read_stderr_nonblocking()
                raise AssertionError(
                    f"Timed out waiting for MCP JSON-RPC stdout message (stderr={stderr_out!r})"
                ) from exc
            if isinstance(item, Exception):
                raise AssertionError(f"MCP stdout framing/JSON error: {item}") from item
            if item is None:
                stderr_out = (
                    self._proc.stderr.read().decode("utf-8", errors="replace")
                    if self._proc.stderr is not None
                    else ""
                )
                raise AssertionError(f"MCP server closed stdout unexpectedly (rc={self._proc.poll()}, {stderr_out!r})")
            return item

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self._proc.stdin is not None
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        req_id: int | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        assert self._proc.stdin is not None
        rid = req_id if req_id is not None else self._next_id
        self._next_id = max(self._next_id + 1, rid + 1)
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

        deadline = time.monotonic() + timeout
        while True:
            resp = self._read_message(max(0.1, deadline - time.monotonic()))
            if resp.get("id") == rid:
                return resp

    def initialize(self, *, client_name: str = "tests_e2e", timeout: float = 15.0) -> dict[str, Any]:
        resp = self.request(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1.0.0"},
            },
            timeout=timeout,
        )
        self.notify("notifications/initialized", params={})
        return resp

    def list_tools(self, *, timeout: float = 15.0) -> list[dict[str, Any]]:
        resp = self.request("tools/list", params={}, timeout=timeout)
        assert "result" in resp, f"tools/list returned error: {resp!r}"
        tools = resp["result"].get("tools")
        assert isinstance(tools, list), f"tools/list missing tools list: {resp!r}"
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self.request(
            "tools/call",
            params={"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )


def _load_default_mcp_spec() -> tuple[list[str], Path, dict[str, str]]:
    cfg_path = REPO_ROOT / ".agents" / "mcp_config.json"
    if not cfg_path.is_file():
        raise AssertionError(f"Missing Antigravity MCP config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise AssertionError(f"Missing non-empty mcpServers in {cfg_path}")
    for entry in servers.values():
        if isinstance(entry, dict) and entry.get("command"):
            argv = _resolve_command_argv([entry["command"], *(entry.get("args") or [])])
            raw_cwd = entry.get("cwd")
            srv_cwd = (REPO_ROOT / raw_cwd).resolve() if isinstance(raw_cwd, str) else REPO_ROOT
            raw_env = entry.get("env")
            srv_env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}
            return argv, srv_cwd, srv_env
    raise AssertionError(f"No stdio command entry found in {cfg_path}")


@contextlib.contextmanager
def mcp_stdio_session(
    server_cmd: Sequence[str] | None = None,
    *,
    cwd: Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> Iterator[McpStdioClient]:
    """Launch a stdio MCP server and yield a connected `McpStdioClient`."""
    if server_cmd is None:
        argv, cfg_cwd, cfg_env = _load_default_mcp_spec()
        effective_cwd = cwd or cfg_cwd
        merged_env = dict(cfg_env)
        if env_extra:
            merged_env.update(env_extra)
    else:
        argv = _resolve_command_argv(server_cmd)
        effective_cwd = cwd or REPO_ROOT
        merged_env = dict(env_extra or {})

    proc = subprocess.Popen(
        argv,
        cwd=effective_cwd,
        env=no_gpu_env(merged_env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = McpStdioClient(proc)
    try:
        yield client
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if proc.stdout is not None and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr is not None and not proc.stderr.closed:
            proc.stderr.close()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(name="no_gpu_env")
def no_gpu_env_fixture() -> dict[str, str]:
    return no_gpu_env()


@pytest.fixture(name="run_sheaf_cli")
def run_sheaf_cli_fixture() -> Callable[..., subprocess.CompletedProcess[str]]:
    return run_sheaf_cli


@pytest.fixture(name="resolve_public_symbol")
def resolve_public_symbol_fixture() -> Callable[..., Any]:
    return resolve_public_symbol


@pytest.fixture(name="resolve_any_public_symbol")
def resolve_any_public_symbol_fixture() -> Callable[..., Any]:
    return resolve_any_public_symbol


@pytest.fixture
def readonly_snapshot(repo_root: Path) -> Iterator[dict[str, str]]:
    baseline = snapshot_readonly_hashes(repo_root)
    yield baseline
    assert_readonly_unchanged(baseline, repo_root)


@pytest.fixture(name="run_hook")
def run_hook_fixture() -> Callable[..., tuple[subprocess.CompletedProcess[str], dict[str, Any]]]:
    return run_hook


def scaffold_mini_selftest_repo(
    repo_root: Path,
    mini: Path,
    *,
    create_default_test: bool = False,
) -> Path:
    """Create an isolated temporary repo copy suitable for `python -S -m sheaf_verify.selftest`."""
    mini.mkdir(parents=True, exist_ok=True)
    for rel in ("docs", ".github", "fixtures", "src", ".agents"):
        shutil.copytree(repo_root / rel, mini / rel, dirs_exist_ok=True)
    shutil.copytree(
        repo_root / "iterations" / "exec-hypergraph-v01",
        mini / "iterations" / "exec-hypergraph-v01",
        dirs_exist_ok=True,
    )
    for fname in ("AGENTS.md", "ANTIGRAVITY_HANDOFF.md", "GATES.md"):
        shutil.copy2(repo_root / fname, mini / fname)
    (mini / "tests").mkdir(exist_ok=True)
    if create_default_test:
        (mini / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return mini


@pytest.fixture(name="mcp_stdio_session")
def mcp_stdio_session_fixture() -> Callable[..., contextlib.AbstractContextManager[McpStdioClient]]:
    return mcp_stdio_session
