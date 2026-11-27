#!/usr/bin/env python3
"""Live LapBeast gated run: measure the HAL, register a sheaf-verify store, run/verify/anchor.

Follows the README trainer flow via the sheaf-verify CLI. Does not overwrite
the sealed T46_TRI_MACHINE_RESULTS.json. Does not print the seal key.
Gates that need training corpus, checkpoints, or G0-style metrics are left
unregistered and are expected to fail or block honestly.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEALED_T46 = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"
LIVE_T46 = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.lapbeast-live.json"
HAL_C = ROOT / "src" / "hadukenn_core" / "csrc" / "tri_machine_hal.c"
CUDA_KERNEL = ROOT / "src" / "hadukenn_core" / "csrc" / "cuda_blackwell_turing_kernels.cu"
HOST_REF = ROOT / "scripts" / "bench_cce_cpu.py"
DEFAULT_STORE = ROOT / ".sheaf"
DEFAULT_SEAL = Path.home() / ".sheaf-seal-key"
DEFAULT_ANCHOR = Path.home() / "sheaf-anchors.txt"
DEFAULT_LOGDIR = Path.home() / "hadukenn" / "logs"

KERNEL_ENTRY = "hadukenn_tiled_cce_mup_fwd_bwd_f32"
CUDA_ENTRY = "hadukenn_cuda_mup_cce_fwd_bwd_f32"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_seal_key(path: Path) -> Path:
    """Create a local production HMAC key if missing. Never logs the bytes."""
    path = path.expanduser()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"seal key path is not a regular file: {path}")
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            path.chmod(0o600)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, os.urandom(32))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


def probe_environment() -> dict[str, str]:
    py = platform.python_version()
    plat = platform.platform()
    cuda = "unavailable"
    driver = "unavailable"
    gpu = "unavailable"
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        proc = subprocess.run(
            [
                nvsmi,
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu, driver, cap = parts[0], parts[1], parts[2]
                cuda = cap
            elif len(parts) == 2:
                gpu, driver = parts
        # nvcc version if present
        nvcc = shutil.which("nvcc")
        if nvcc:
            nv = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
            for line in nv.stdout.splitlines():
                if "release" in line.lower():
                    # e.g. Cuda compilation tools, release 12.4, V12.4.131
                    cuda = line.strip()
                    break
    return {"python": py, "cuda": cuda, "driver": driver, "gpu": gpu, "platform": plat}


def write_live_json(stdout_text: str, dest: Path) -> Path:
    if dest.resolve() == SEALED_T46.resolve():
        raise SystemExit("refusing to overwrite the sealed T46_TRI_MACHINE_RESULTS.json")
    start = stdout_text.find("{")
    if start < 0:
        raise SystemExit("build_tri_machine.py produced no JSON object")
    doc = json.loads(stdout_text[start:])
    doc.get("regulatory_compliance", {}).pop("bundle", None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def _sv(*args: str, key: Path | None = None) -> dict:
    cmd = [sys.executable, "-m", "sheaf_verify.cli", *args, "--json"]
    if key is not None:
        cmd.extend(["--seal-key-file", str(key)])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode not in (0, 1):
        raise SystemExit(
            f"sheaf-verify {' '.join(args)} failed (rc={proc.returncode}):\n{proc.stderr or proc.stdout}"
        )
    text = proc.stdout.strip()
    start = text.find("{")
    payload = json.loads(text[start:]) if start >= 0 else {"raw": text, "returncode": proc.returncode}
    payload["_returncode"] = proc.returncode
    payload["_stderr"] = proc.stderr
    return payload


def prepare_store(store: Path, *, fresh: bool) -> Path:
    store = store.resolve()
    if store.exists() and any(store.iterdir()):
        if not fresh:
            raise SystemExit(f"store {store} is not empty; pass --fresh to rotate it aside")
        backup = store.with_name(store.name + ".prev-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
        store.rename(backup)
    store.mkdir(parents=True, exist_ok=True)
    return store


def write_sidecar_payloads(work: Path, live_json: Path) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "sheaf-verify.config.v1",
        "study": "t46-tri-machine-hal-lapbeast-live",
        "machine": "lapbeast",
        "note": "HAL / CCE microbench on this host. Not a training run.",
    }
    seeds = {
        "format": "sheaf-verify.seeds.v1",
        "design": "fixed_suite_seed",
        "arms": {"cce_hal_suite": [20261126]},
        "note": "Seed used by run_tri_machine_verification_suite. Not a paper-plan seed ladder.",
    }
    claim = {
        "format": "sheaf-verify.claim.v1",
        "hypothesis_id": "lapbeast-c11-cce-vs-python-row",
        "description": (
            "On this LapBeast host, the compiled C11 tiled CCE kernel is faster than the "
            "serial Python row loop at the suite shape, under isolated OpenBLAS/OpenMP pools. "
            "This is not a WikiText / G0–G6 training claim."
        ),
        "primary_metric": "c11_speedup_over_py_loop",
        "primary_contrast": "c11_vs_python_row",
        "direction": "greater",
        "alpha": 0.05,
        "claimed": False,
    }
    paths = {
        "config": work / "config.json",
        "seeds": work / "seeds.json",
        "claim": work / "claim.json",
        "metrics": live_json,
    }
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["seeds"].write_text(json.dumps(seeds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["claim"].write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def register_all(
    store: Path,
    *,
    env: dict[str, str],
    payloads: dict[str, Path],
    log_paths: list[Path],
) -> dict[str, str]:
    st = ["--store", str(store)]
    shas: dict[str, str] = {}

    def reg(*args: str) -> str:
        out = _sv(*args)
        digest = out.get("manifest_sha256")
        if not isinstance(digest, str):
            raise SystemExit(f"registration did not return a manifest hash: {out}")
        return digest

    shas["code"] = reg("register-tree", *st, "--role", "code", "--path", str(ROOT))
    shas["environment"] = reg(
        "register-env",
        *st,
        "--python",
        env["python"],
        "--cuda",
        env["cuda"],
        "--driver",
        env["driver"],
        "--gpu",
        env["gpu"],
        "--platform",
        env["platform"],
    )
    # Host reference first (NumPy/Python CCE bench), then C11 and CUDA kernels bind to it.
    host_out = _sv(
        "register",
        *st,
        "--kind",
        "kernel",
        "--role",
        "kernel.host",
        "--path",
        str(HOST_REF),
        "--metadata",
        json.dumps({"backend": "cpu", "entry": "numpy_gemm_cce"}),
    )
    shas["kernel.host"] = host_out["manifest_sha256"]
    host_payload_sha = (host_out.get("manifest") or {}).get("payload_sha256")
    if not host_payload_sha:
        man_path = store / "manifests" / f"{shas['kernel.host']}.json"
        host_payload_sha = json.loads(man_path.read_text())["payload_sha256"]

    shas["kernel"] = reg(
        "register",
        *st,
        "--kind",
        "kernel",
        "--role",
        "kernel",
        "--path",
        str(HAL_C),
        "--input",
        shas["kernel.host"],
        "--metadata",
        json.dumps(
            {
                "backend": "c",
                "entry": KERNEL_ENTRY,
                "host_reference_sha256": host_payload_sha,
            }
        ),
    )
    shas["kernel.cuda"] = reg(
        "register",
        *st,
        "--kind",
        "kernel",
        "--role",
        "kernel.cuda",
        "--path",
        str(CUDA_KERNEL),
        "--input",
        shas["kernel.host"],
        "--metadata",
        json.dumps(
            {
                "backend": "cuda",
                "entry": CUDA_ENTRY,
                "host_reference_sha256": host_payload_sha,
            }
        ),
    )
    shas["config"] = reg("register", *st, "--kind", "config", "--role", "config", "--path", str(payloads["config"]))
    shas["seeds"] = reg("register", *st, "--kind", "seeds", "--role", "seeds", "--path", str(payloads["seeds"]))
    shas["claim"] = reg("register", *st, "--kind", "claim", "--role", "claim", "--path", str(payloads["claim"]))
    metric_inputs = ["--input", shas["seeds"], "--input", shas["code"], "--input", shas["kernel"]]
    shas["metrics"] = reg(
        "register",
        *st,
        "--kind",
        "metrics",
        "--role",
        "metrics",
        "--path",
        str(payloads["metrics"]),
        *metric_inputs,
    )
    for log in log_paths:
        role = "log." + log.stem.replace("-", "_").replace(".", "_").lower()
        # role must match ROLE_RE: log.bench_cce_cpu etc.
        shas[role] = reg(
            "register",
            *st,
            "--kind",
            "log",
            "--role",
            role,
            "--path",
            str(log),
            "--input",
            shas["metrics"],
        )
    return shas


def gate_table(run_payload: dict) -> list[dict]:
    return list(run_payload.get("gates") or [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live LapBeast HAL run + sheaf-verify store (README trainer flow).")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--seal-key-file", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--anchor-out", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--live-json", type=Path, default=LIVE_T46)
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    parser.add_argument("--fresh", action="store_true", help="rotate a non-empty store aside and start empty")
    parser.add_argument("--skip-suite", action="store_true", help="reuse --live-json; do not rebuild the HAL")
    args = parser.parse_args(argv)

    if args.live_json.resolve() == SEALED_T46.resolve():
        print("refusing to use the sealed T46 path as the live output", file=sys.stderr)
        return 2

    seal = ensure_seal_key(args.seal_key_file)
    env = probe_environment()
    print("environment:", json.dumps(env), flush=True)

    if not args.skip_suite:
        cmd = [sys.executable, str(ROOT / "scripts" / "build_tri_machine.py"), "--machine", "lapbeast"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(ROOT))
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        write_live_json(proc.stdout, args.live_json)
        print("wrote", args.live_json, flush=True)
    elif not args.live_json.is_file():
        print(f"missing live JSON {args.live_json}", file=sys.stderr)
        return 2

    store = prepare_store(args.store, fresh=args.fresh)
    # Sidecar payloads must sit outside the store: sheaf-verify refuses in-store paths.
    work = (args.logdir / "live-gated-sidecars").resolve()
    payloads = write_sidecar_payloads(work, args.live_json)
    logs: list[Path] = []
    for name in ("bench-cce-cpu.json", "bench-cuda-sm89.json"):
        candidate = args.logdir / name
        if candidate.is_file():
            logs.append(candidate)

    shas = register_all(store, env=env, payloads=payloads, log_paths=logs)
    print("registered:", json.dumps(shas, indent=2), flush=True)

    run_out = _sv("run", "--store", str(store), key=seal)
    print("run rc=", run_out["_returncode"], "failed=", run_out.get("failed"), flush=True)
    for row in gate_table(run_out):
        print(f"  {row.get('gate_id')}: {row.get('status')}", flush=True)

    ver = _sv("verify", "--store", str(store), key=seal)
    print("verify ok=", ver.get("ok"), "head=", ver.get("head"), "problems=", len(ver.get("problems") or []), flush=True)
    if ver.get("problems"):
        for problem in ver["problems"]:
            print("  problem:", problem, flush=True)

    anc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sheaf_verify.cli",
            "anchor",
            "--store",
            str(store),
            "--out",
            str(args.anchor_out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    sys.stdout.write(anc.stdout)
    if anc.returncode != 0:
        print(anc.stderr, file=sys.stderr)
        return anc.returncode

    report = {
        "store": str(store),
        "live_json": str(args.live_json),
        "environment": env,
        "manifests": shas,
        "run": {"returncode": run_out["_returncode"], "failed": run_out.get("failed"), "gates": gate_table(run_out)},
        "verify": {"ok": ver.get("ok"), "head": ver.get("head"), "problems": ver.get("problems")},
        "anchor": anc.stdout.strip(),
        "generated_at": utc_now(),
        "note": (
            "Corpus, vocabulary, hypergraph, checkpoints, causality trace, and figure "
            "were not registered: this is a HAL microbench, not a training run. "
            "Those gates should fail or block honestly."
        ),
    }
    (store / "live_gated_run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # run may exit 1 when gates fail; that is expected and not a script failure
    return 0 if ver.get("ok") and anc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
