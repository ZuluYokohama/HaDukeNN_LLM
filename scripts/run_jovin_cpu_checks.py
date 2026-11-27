#!/usr/bin/env python3
"""Run receipted stdlib CPU-only HaDukeNN checks inside the jovin-harness sandbox.

The sandbox image is python:3.12-slim with no numpy, no CUDA, no repo mount,
a 256 KiB input cap, and a 30 s timeout ceiling. This driver therefore does
*not* run the host pytest suite or the GPU HAL build. It verifies a stdlib
CPU slice: the float64 CCE reference, HAL claim shape, and recomputation of
parity / timings / guards from raw payloads. When live sidecars are present,
those documents are embedded in the tests file (the sandbox cannot see the
host filesystem).

GPU work stays on the host. This script never passes --gpus or an nvidia
runtime. It does not overwrite the sealed T46_TRI_MACHINE_RESULTS.json and
does not print the seal key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "scripts" / "jovin_cpu_payload.py"
PAYLOAD_TESTS = ROOT / "scripts" / "jovin_cpu_payload_tests.py"
SEALED_T46 = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"
DEFAULT_SIDECARS = Path.home() / "hadukenn" / "logs" / "live-gated-sidecars"
DEFAULT_OUT = Path.home() / "hadukenn" / "logs" / "jovin-cpu-checks.json"
DEFAULT_JOVIN = Path.home() / "hadukenn" / "tools" / "jovin-harness" / ".venv" / "bin" / "jovin-harness"
MAX_INPUT_BYTES = 262144
MAX_TIMEOUT_S = 30
LIVE_FILES = ("claim.json", "parity.json", "timings.json", "guards.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_jovin() -> Path:
    env = os.environ.get("JOVIN_HARNESS")
    if env:
        return Path(env).expanduser()
    if DEFAULT_JOVIN.is_file():
        return DEFAULT_JOVIN
    found = shutil.which("jovin-harness")
    if found:
        return Path(found)
    return DEFAULT_JOVIN


def resolve_sidecars(explicit: str | None, no_live: bool) -> Path | None:
    if no_live:
        return None
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise SystemExit(f"live sidecars directory is missing: {path}")
        return path
    if DEFAULT_SIDECARS.is_dir() and all((DEFAULT_SIDECARS / name).is_file() for name in LIVE_FILES):
        return DEFAULT_SIDECARS
    return None


def load_sidecars(directory: Path) -> tuple[dict, dict[str, str]]:
    documents = {}
    hashes = {}
    for name in LIVE_FILES:
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"required sidecar is missing: {path}")
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = sha256_file(path)
    return documents, hashes


def _json_loads_stmt(name: str, document: object) -> str:
    """Emit a Python assignment that loads exact JSON (not a JSON literal with true/false)."""
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return f"{name} = json.loads({json.dumps(raw)})"


def render_live_section(documents: dict) -> str:
    """Embed live sidecar JSON in the tests file. The sandbox has no host mounts."""
    parts = [
        "",
        "# --- live sidecar slice, embedded by run_jovin_cpu_checks.py ---",
        "import json",
        _json_loads_stmt("LIVE_CLAIM", documents["claim.json"]),
        _json_loads_stmt("LIVE_PARITY", documents["parity.json"]),
        _json_loads_stmt("LIVE_TIMINGS", documents["timings.json"]),
        _json_loads_stmt("LIVE_GUARDS", documents["guards.json"]),
        "",
        "def test_live_claim_criteria():",
        "    failed = _failed(evaluate_claim(LIVE_CLAIM))",
        "    assert failed == [], failed",
        "",
        "def test_live_parity_recomputed_from_float64():",
        "    failed = _failed(evaluate_parity(LIVE_PARITY, LIVE_CLAIM))",
        "    assert failed == [], failed",
        "",
        "def test_live_timings_recomputed_from_raw_repeats():",
        "    failed = _failed(evaluate_timings(LIVE_TIMINGS, LIVE_CLAIM))",
        "    assert failed == [], failed",
        "",
        "def test_live_guards_from_recorded_samples():",
        "    failed = _failed(evaluate_guards(LIVE_GUARDS))",
        "    assert failed == [], failed",
        "",
    ]
    return "\n".join(parts)


def assemble_tests(sidecars: Path | None) -> tuple[str, dict[str, str] | None]:
    text = PAYLOAD_TESTS.read_text(encoding="utf-8")
    hashes = None
    if sidecars is not None:
        documents, hashes = load_sidecars(sidecars)
        text = text.rstrip() + "\n" + render_live_section(documents)
    return text, hashes


def check_size(label: str, data: bytes) -> None:
    if len(data) > MAX_INPUT_BYTES:
        raise SystemExit(f"{label} is {len(data)} bytes, above the {MAX_INPUT_BYTES} sandbox input cap")


def refuse_sealed_out(path: Path) -> None:
    if path.resolve() == SEALED_T46.resolve():
        raise SystemExit("refusing to overwrite the sealed T46_TRI_MACHINE_RESULTS.json")


def run_json(argv: list[str]) -> dict:
    proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    text = proc.stdout.strip() or proc.stderr.strip()
    try:
        document = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError:
        document = None
    if document is None:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(argv)}\n{text}")
    if proc.returncode != 0 and not isinstance(document, dict):
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(argv)}\n{text}")
    return document


def doctor(jovin: Path) -> dict:
    document = run_json([str(jovin), "doctor"])
    if not document.get("ready"):
        problems = document.get("problems") or ["doctor reported ready=false"]
        raise SystemExit("jovin-harness is not ready: " + "; ".join(map(str, problems)))
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jovin", type=Path, default=None, help="path to the jovin-harness CLI")
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="sandbox timeout in seconds (cannot exceed the installed jovin-harness limit; stock config is 10s, hard ceiling is 30s)",
    )
    parser.add_argument("--live-sidecars", type=str, default=None, help="directory of live HAL sidecar JSON")
    parser.add_argument("--no-live", action="store_true", help="do not embed live sidecars even if present")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="summary JSON path (outside the repo)")
    parser.add_argument(
        "--dry-run", action="store_true", help="assemble payloads and print sizes; do not invoke Docker"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Fail closed if a GPU flag is ever added to this driver.
    raw = list(sys.argv[1:] if argv is None else argv)
    lowered = [item.lower() for item in raw]
    forbidden = ("--gpus", "--gpu", "--runtime=nvidia", "nvidia")
    if any(item.startswith(flag) or item == flag for flag in forbidden for item in lowered):
        raise SystemExit("refusing GPU flags: this check is CPU-only; GPU work stays on the host")

    args = build_parser().parse_args(raw)
    if not PAYLOAD.is_file() or not PAYLOAD_TESTS.is_file():
        raise SystemExit(f"payload files are missing under {ROOT / 'scripts'}")

    timeout = args.timeout
    if timeout <= 0 or timeout > MAX_TIMEOUT_S:
        raise SystemExit(f"timeout must be in 1..{MAX_TIMEOUT_S}")

    out = args.out.expanduser()
    refuse_sealed_out(out)

    sidecars = resolve_sidecars(args.live_sidecars, args.no_live)
    tests_text, sidecar_hashes = assemble_tests(sidecars)
    code_bytes = PAYLOAD.read_bytes()
    tests_bytes = tests_text.encode("utf-8")
    check_size("payload", code_bytes)
    check_size("tests", tests_bytes)

    summary = {
        "format": "hadukenn.jovin-cpu-checks.v1",
        "gpu": False,
        "note": "stdlib CPU slice only; the host pytest suite and GPU/HAL .so work stay on the host",
        "code_bytes": len(code_bytes),
        "tests_bytes": len(tests_bytes),
        "code_sha256": sha256_bytes(code_bytes),
        "tests_sha256": sha256_bytes(tests_bytes),
        "live_sidecars": sidecars is not None,
        "live_sidecars_dir": str(sidecars) if sidecars else None,
        "live_sidecar_sha256": sidecar_hashes,
        "timeout_s": timeout,
        "isolation": "process",
    }

    if args.dry_run:
        summary["status"] = "dry_run"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    jovin = (args.jovin or default_jovin()).expanduser()
    if not jovin.is_file():
        raise SystemExit(f"jovin-harness CLI not found: {jovin}")

    health = doctor(jovin)
    configured = int((health.get("limits") or {}).get("timeout_s") or 10)
    if timeout > configured:
        timeout = configured
    with tempfile.TemporaryDirectory(prefix="jovin-cpu-checks-") as tmp:
        tests_path = Path(tmp) / "jovin_cpu_payload_tests.py"
        tests_path.write_text(tests_text, encoding="utf-8")
        argv_verify = [
            str(jovin),
            "verify",
            "--tests",
            str(tests_path),
            "--timeout",
            str(timeout),
            "--isolation",
            "process",
            str(PAYLOAD),
        ]
        # GPU must stay off the argv: no runtime, no --gpus.
        result = run_json(argv_verify)
    summary["timeout_s"] = timeout

    receipt_id = result.get("receipt_id")
    receipt_sha = result.get("receipt_sha256")
    chain = run_json([str(jovin), "receipts", "verify"])
    status = result.get("status")
    passed = bool(result.get("passed")) and status == "passed" and chain.get("valid") is True

    summary.update(
        {
            "status": "passed" if passed else (status or "failed"),
            "receipt_id": receipt_id,
            "receipt_sha256": receipt_sha,
            "chain_valid": bool(chain.get("valid")),
            "chain_count": chain.get("count"),
            "passed_count": result.get("passed_count"),
            "total": result.get("total"),
            "duration_s": result.get("duration_s"),
            "integrity": result.get("integrity"),
            "image": (health.get("image_info") or {}).get("id") or health.get("image"),
            "harness_version": health.get("harness_version"),
            "failures": result.get("failures") or [],
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
