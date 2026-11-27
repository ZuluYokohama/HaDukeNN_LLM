"""Single-command local E2E runner for Snapdragon X Plus deployment verification."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from hadukenn_core.hst_foundation_lm import verify_oryon_neon_tiled_cce
from hadukenn_core.npu_qnn_backend import close_all_htp_sessions
from hadukenn_core.snapdragon_local import (
    _sanitize_obj,
    canonical_json_bytes,
    compute_manifest_sha256,
    plan_hst_foundation_snapdragon,
    run_t46_snapdragon_local_suite,
)
from hadukenn_core.tri_machine import run_tri_machine_verification_suite

_BANNED_DRIVE_PREFIXES = tuple(f"{d}:" + "\\" for d in ("C", "D", "E", "H", "c", "d", "e", "h"))


def _sanitize_error(text: str) -> str:
    cleaned = text
    for prefix in _BANNED_DRIVE_PREFIXES:
        cleaned = cleaned.replace(prefix, "[win-path]/")
    return cleaned


def _validate_output_path_early(out_path: Path) -> Path:
    """Fail fast before running heavy kernels if ``out_path`` cannot be written."""
    if out_path.is_symlink():
        raise OSError(f"Output path must not be a symlink: {out_path}")
    if out_path.exists():
        if out_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {out_path}")
        if not os.access(out_path, os.W_OK):
            raise PermissionError(f"Output path is not writable: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix=".e2e_probe_", delete=True):
        pass
    return out_path


def run_local_e2e(
    json_out: str | Path | None = None,
    *,
    dcvs_mode: str = "burst",
    include_tri_machine: bool = False,
    include_hst_foundation: bool = False,
) -> dict[str, Any]:
    """Execute full Snapdragon X Plus local E2E pipeline and optionally write canonical JSON."""
    out_path = _validate_output_path_early(Path(json_out)) if json_out is not None else None
    try:
        report = run_t46_snapdragon_local_suite(
            dcvs_mode=dcvs_mode,
            close_npu_sessions=True,
            include_tiled_cce_51m=True,
        )
        if include_tri_machine:
            tri_report = run_tri_machine_verification_suite()
            reg = tri_report.get("regulatory_compliance")
            if isinstance(reg, dict):
                reg.pop("bundle", None)
            report["tri_machine_hal"] = _sanitize_obj(tri_report)
        if include_hst_foundation:
            report["hst_foundation"] = _sanitize_obj(
                {
                    "schema": "sheaf-verify.hst-snapdragon-integration.v1",
                    "dcvs_mode": dcvs_mode,
                    "ladder_plan": plan_hst_foundation_snapdragon(),
                    "oryon_neon_tiled_cce": verify_oryon_neon_tiled_cce(),
                }
            )
    finally:
        close_all_htp_sessions()

    report["manifest_sha256"] = compute_manifest_sha256(report)
    if out_path is not None:
        out_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hadukenn-local-e2e",
        description="Run Snapdragon X Plus local E2E verification and emit T-46 report JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write canonical T-46 JSON report.",
    )
    parser.add_argument(
        "--dcvs-mode",
        choices=("burst", "power_saver"),
        default="burst",
        help="Hexagon HTP DCVS performance mode.",
    )
    parser.add_argument(
        "--include-tri-machine",
        action="store_true",
        help="Include Tri-Machine HAL v0.9.0 verification summary in the T-46 report.",
    )
    parser.add_argument(
        "--include-hst-foundation",
        action="store_true",
        help="Include HST-v1 Snapdragon ladder plan and Oryon NEON CCE check in the T-46 report.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_local_e2e(
            json_out=args.json_out,
            dcvs_mode=args.dcvs_mode,
            include_tri_machine=args.include_tri_machine,
            include_hst_foundation=args.include_hst_foundation,
        )
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"error: {_sanitize_error(str(exc))}", file=sys.stderr)
        return 1

    if args.json_out is None:
        sys.stdout.write(canonical_json_bytes(report).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
