#!/usr/bin/env python3
"""CLI builder, profiler, and receipt generator for the Tri-Machine HAL (v0.9.0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadukenn_core.tri_machine import (  # noqa: E402
    apply_thread_affinity,
    detect_machine_profile,
    get_tri_machine_lib,
    run_tri_machine_verification_suite,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify HaDukeNN Tri-Machine HAL.")
    parser.add_argument(
        "--machine",
        default="auto",
        choices=("auto", "wbeast", "lapbeast", "zenbook"),
        help="Target machine profile",
    )
    parser.add_argument(
        "--write-iteration",
        action="store_true",
        help="Write verified T-46 results and regulatory bundle into iterations/t46-tri-machine-hal/",
    )
    args = parser.parse_args(argv)

    lib = get_tri_machine_lib(force_rebuild=True)
    if lib is None:
        print("ERROR: Failed to compile C11 tri_machine_hal shared library.", file=sys.stderr)
        return 1

    profile = detect_machine_profile(None if args.machine == "auto" else args.machine)
    affinity_compute = apply_thread_affinity(profile, role="compute")
    report = run_tri_machine_verification_suite()
    report["selected_profile"] = profile.machine_id
    report["affinity_probe"] = affinity_compute

    if args.write_iteration:
        iter_dir = ROOT / "iterations" / "t46-tri-machine-hal"
        iter_dir.mkdir(parents=True, exist_ok=True)
        bundle = report["regulatory_compliance"].pop("bundle")
        (iter_dir / "T46_REGULATORY_BUNDLE.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (iter_dir / "T46_TRI_MACHINE_RESULTS.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        report["regulatory_compliance"].pop("bundle", None)

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
