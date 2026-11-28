#!/usr/bin/env python3
"""Live LapBeast gated run: measure the HAL, register a sheaf-verify store, run/verify/anchor.

Follows the README trainer flow via the sheaf-verify CLI. Does not overwrite
the sealed T46_TRI_MACHINE_RESULTS.json. Does not print the seal key.
Registers a HAL study and evaluates it with `sheaf-verify run --profile hal`.
Training G/H/P gates are not walked and are not bypassed.
"""
from __future__ import annotations

import argparse
import json
import os
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
    from sheaf_verify.hal import probe_environment as _probe

    live = _probe()
    return {field: live[field] for field in ("python", "cuda", "driver", "gpu", "platform")}


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
        "format": "sheaf-verify.hal-claim.v1",
        "study_profile": "hal",
        "hypothesis_id": "lapbeast-c11-cce-vs-serial-python-loop",
        "description": (
            "On this LapBeast host, the compiled C11 tiled CCE kernel is faster than a "
            "serial Python row loop at the recorded shapes, with OpenBLAS isolated. "
            "Packed NumPy is reported as a fair baseline and is not the claim baseline."
        ),
        "performance": {
            "metric": "speedup",
            "direction": "greater",
            "baseline": "serial_python_loop",
            "baselines_reported": ["serial_python_loop", "packed_numpy"],
            "min_speedup": 1.5,
            "subject": "c11_tiled_cce",
        },
        "parity": {"max_abs": 0.0001, "ops": ["cce_mup_fwd"]},
        "guards": {"thermal_stop_celsius": 85},
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



def _nvidia_sample() -> dict:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"t": utc_now(), "raw": "", "rc": -1, "error": f"nvidia-smi unavailable: {exc}"}
    sample = {"t": utc_now(), "raw": (proc.stdout or "").strip(), "rc": proc.returncode}
    if proc.returncode == 0 and proc.stdout.strip():
        parts = [item.strip() for item in proc.stdout.strip().splitlines()[0].split(",")]
        if parts:
            try:
                sample["celsius"] = float(parts[0])
            except ValueError:
                pass
        if len(parts) > 1:
            try:
                sample["watts"] = float(parts[1])
            except ValueError:
                pass
    else:
        sample["error"] = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
    return sample


def _time_repeats(fn, repeats: int = 5, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = __import__("time").perf_counter()
        fn()
        times.append((__import__("time").perf_counter() - start) * 1000.0)
    return times


def collect_hal_evidence(work: Path) -> dict[str, Path]:
    """Run a small parity case and raw-repeat timings. Recompute happens in the gates."""
    import math

    import numpy as np

    from hadukenn_core.tri_machine import _set_openblas_threads, hal_tiled_cce_mup, probe_host_hardware

    work.mkdir(parents=True, exist_ok=True)
    samples = [_nvidia_sample()]
    rng = np.random.default_rng(20261126)
    hidden = rng.standard_normal((8, 16)).astype(np.float32)
    embedding = (rng.standard_normal((32, 16)) / math.sqrt(16)).astype(np.float32)
    targets = rng.integers(0, 32, size=8, dtype=np.int32)
    loss, _, _ = hal_tiled_cce_mup(
        hidden, embedding, targets, tile_V=32, mup_base_d=16, compute_grad=False, num_threads=1
    )
    parity = {
        "format": "sheaf-verify.hal-parity.v1",
        "cases": [
            {
                "name": "cce_mup_small",
                "op": "cce_mup_fwd",
                "inputs": {
                    "H": hidden.astype(float).tolist(),
                    "E": embedding.astype(float).tolist(),
                    "y": [int(v) for v in targets],
                    "mup_base_d": 16.0,
                },
                "kernel_outputs": {"loss": [float(v) for v in loss]},
            }
        ],
    }
    threads = len(probe_host_hardware().get("p_cores") or []) or 8
    cases = []
    for n, d, v in ((32, 256, 1024), (64, 384, 2048)):
        h = rng.standard_normal((n, d)).astype(np.float32)
        e = (rng.standard_normal((v, d)) / math.sqrt(d)).astype(np.float32)
        y = rng.integers(0, v, size=n, dtype=np.int32)
        scale = 384.0 / float(d)

        def py_row(h=h, e=e, y=y, scale=scale) -> None:
            for idx in range(h.shape[0]):
                row = (h[idx] @ e.T) * scale
                peak = float(np.max(row))
                _ = peak + math.log(float(np.sum(np.exp(row - peak)))) - float(row[y[idx]])

        def np_gemm(h=h, e=e, y=y, scale=scale) -> None:
            logits = (h @ e.T) * scale
            peak = np.max(logits, axis=1, keepdims=True)
            _ = (peak.squeeze(-1) + np.log(np.sum(np.exp(logits - peak), axis=1))) - logits[np.arange(h.shape[0]), y]

        def c11(h=h, e=e, y=y) -> None:
            hal_tiled_cce_mup(h, e, y, tile_V=512, mup_base_d=384, compute_grad=False, num_threads=threads)

        _set_openblas_threads(1)
        serial = _time_repeats(py_row)
        _set_openblas_threads(threads)
        packed = _time_repeats(np_gemm)
        _set_openblas_threads(1)
        subject = _time_repeats(c11)
        samples.append(_nvidia_sample())
        cases.append({
            "name": f"cce_{n}x{d}x{v}",
            "subject_ms": subject,
            "baselines_ms": {"serial_python_loop": serial, "packed_numpy": packed},
        })
    samples.append(_nvidia_sample())
    timings = {"format": "sheaf-verify.hal-timings.v1", "cases": cases}
    watts = [item["watts"] for item in samples if "watts" in item]
    temps = [{"t": item["t"], "celsius": item["celsius"]} for item in samples if "celsius" in item]
    if watts:
        power = {"observed": True, "draw_watts": watts[-1], "samples": watts}
    else:
        reason = next((item.get("error") or "power.draw was not numeric") for item in samples)
        power = {
            "observed": False,
            "probe": "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits",
            "reason": reason,
        }
    if temps:
        thermal = {"stop_celsius": 85, "unit": "C", "samples": temps}
    else:
        reason_t = next((item.get("error") or "nvidia-smi thermal probe returned no numeric readings") for item in samples)
        thermal = {
            "stop_celsius": 85,
            "unit": "C",
            "samples": temps,
            "observed": False,
            "probe": "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits",
            "reason": reason_t,
        }
    guards = {
        "format": "sheaf-verify.hal-guards.v1",
        "thermal": thermal,
        "power": power,
    }
    paths = {
        "parity": work / "parity.json",
        "timings": work / "timings.json",
        "guards": work / "guards.json",
    }
    for key, payload in (("parity", parity), ("timings", timings), ("guards", guards)):
        paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def classify_gates(store: Path) -> list[dict]:
    from sheaf_verify.gate_trace import trace_outcome
    from sheaf_verify.ledger import Ledger
    from sheaf_verify.store import Store

    rows = []
    latest = {}
    for record in Ledger(Store(store).root).records():
        latest[record["gate_id"]] = record
    for record in latest.values():
        own = [item for item in record["criteria"] if not str(item.get("id", "")).startswith("upstream.")]
        upstream = {
            str(item["id"]).removeprefix("upstream."): bool(item.get("passed"))
            for item in record["criteria"]
            if str(item.get("id", "")).startswith("upstream.")
        }
        own_failed = any(not item.get("passed") for item in own)
        failed = [item["id"] + ": " + item["detail"] for item in own if not item.get("passed")]
        rows.append({
            "gate_id": record["gate_id"],
            "ledger_status": record["status"],
            "outcome": trace_outcome(record["status"], bool(own), own_failed, upstream),
            "profile": record.get("profile"),
            "reasons": failed,
        })
    order = ["HAL0", "HAL1", "HAL2", "HAL3", "HAL4", "HAL5", "HAL6"]
    rows.sort(key=lambda row: order.index(row["gate_id"]) if row["gate_id"] in order else 99)
    return rows


def register_all(
    store: Path,
    *,
    env: dict[str, str],
    payloads: dict[str, Path],
    log_paths: list[Path],
    claim_sha: str,
) -> dict[str, str]:
    st = ["--store", str(store)]
    shas: dict[str, str] = {"claim": claim_sha}

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
    # Claim is registered by main() before evidence is collected.
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
        stem = log.stem.lower()
        if stem == "parity":
            role = "log.parity"
        elif stem == "timings":
            role = "log.timings"
        elif stem == "guards":
            role = "log.guards"
        else:
            role = "log." + stem.replace("-", "_").replace(".", "_")
        extra = ["--input", shas["claim"]] if role in {"log.parity", "log.timings"} else ["--input", shas["metrics"]]
        shas[role] = reg(
            "register",
            *st,
            "--kind",
            "log",
            "--role",
            role,
            "--path",
            str(log),
            *extra,
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
    # Freeze the claim before metrics and evidence exist.
    claim_when = utc_now()
    claim_sha = _sv(
        "register",
        "--store",
        str(store),
        "--kind",
        "claim",
        "--role",
        "claim",
        "--path",
        str(payloads["claim"]),
        "--registered-at",
        claim_when,
    )["manifest_sha256"]
    print("registered claim at", claim_when, claim_sha, flush=True)

    evidence = collect_hal_evidence(work)
    logs: list[Path] = [evidence["parity"], evidence["timings"], evidence["guards"]]
    for name in ("bench-cce-cpu.json", "bench-cuda-sm89.json"):
        candidate = args.logdir / name
        if candidate.is_file():
            logs.append(candidate)

    shas = register_all(store, env=env, payloads=payloads, log_paths=logs, claim_sha=claim_sha)
    print("registered:", json.dumps(shas, indent=2), flush=True)

    run_out = _sv("run", "--store", str(store), "--profile", "hal", key=seal)
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

    table = classify_gates(store)
    print("HAL gate outcomes:", flush=True)
    for row in table:
        detail = "; ".join(row["reasons"]) if row["reasons"] else "ok"
        print(f"  {row['gate_id']}: {row['outcome']} ({detail})", flush=True)
    report = {
        "store": str(store),
        "live_json": str(args.live_json),
        "profile": "hal",
        "environment": env,
        "manifests": shas,
        "run": {"returncode": run_out["_returncode"], "failed": run_out.get("failed"), "gates": gate_table(run_out)},
        "gate_table": table,
        "verify": {"ok": ver.get("ok"), "head": ver.get("head"), "problems": ver.get("problems")},
        "anchor": anc.stdout.strip(),
        "generated_at": utc_now(),
        "note": (
            "This run used --profile hal. Training G/H/P gates were not evaluated and were not bypassed. "
            "Parity and performance were recomputed from registered payloads, not from T46 JSON summaries."
        ),
    }
    report_path = args.logdir / "live_gated_run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", report_path, flush=True)
    # run may exit 1 when gates fail; that is expected and not a script failure
    return 0 if ver.get("ok") and anc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
