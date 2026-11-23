"""Pass conditions taken from the architecture spec and the paper plan.

G0 and G1 are the go/no-go lines of stages H0 and H1. G2–G6 are the gates
of H2–H6. H7 has no numeric gate. P0–P6 are the paper-plan phases. A study
cannot replace any of these thresholds by writing a smaller number into
its metrics. Seed minima are per gate: pilots use ``PILOT_SEEDS``, the
headline uses ``HEADLINE_SEEDS``.
"""

from __future__ import annotations

import math

from sheaf_verify.checks import Criterion, bad, ok
from sheaf_verify.hashing import SheafError, as_int, as_real, fmt_float
from sheaf_verify.policy import (
    ALPHA,
    G0_MAX_BUILD_CPU_HOURS,
    G0_MAX_BUILD_RAM_GB,
    G0_MAX_BYTE_FALLBACK_PERCENT,
    G0_MIN_UNIT_REDUCTION,
    G0_RECORDED_VOCAB,
    G0_VOCAB_CAP,
    G1_SD_MARGIN,
    G3_MAX_REL_ERR,
    G3_MAX_SHEAFHEAT_FRACTION,
    G4_MIN_BASELINE_SD,
    HEADLINE_SEEDS,
    MAX_FLOP_RELATIVE_TOLERANCE,
    P4_SEED_SCHEDULE,
    PILOT_SEEDS,
)
from sheaf_verify.stats import mean, one_sided_less, sample_variance, slope_interval, welch

G0_FORMAT = "sheaf-verify.g0.v1"
G1_ARMS = ("hnt64k", "struct32k", "sw64k")
G2_CONTROLS = {
    "a2_spectral": "shuffled_degree",
    "a3_compositional": "random_dag",
}
G4_COMPONENTS = ("attention_bias", "compositional", "sheaf_heat", "spectral")
G6_ARMS = ("corpus_memory", "motif_head")
P1_ARMS = (
    "enhanced_wallmatch",
    "hybrid",
    "random_rewire",
    "s4d",
    "shape_matched",
    "sheaf_reg",
    "spectral_init",
)
P2_PE_CONTROLS = ("alibi", "heat", "heat_rope", "nope", "rope", "sinusoidal")
STAGES = {
    "H0": ("Corpus stats", "G0"),
    "H1": ("Units-only pilot", "G1"),
    "H2": ("Spectral and compositional embeddings", "G2"),
    "H3": ("Kernels K2/K4 and CUDA graph", "G3"),
    "H4": ("A4-A7 and leave-one-out", "G4"),
    "H5": ("Scaling 15-120M and FineWeb-Edu", "G5"),
    "H6": ("Hyperedge output head and corpus-memory arm", "G6"),
    "H7": ("lm-eval and write-up", None),
}


def recorded_g0() -> dict:
    """The feasibility build's G0 measurements.

    These are recorded metadata. They are not WikiText bytes, not a
    training run, and not model weights. ``vocab_size`` is 65,536, which
    the vocab-cap check flags against the written cap of 64,000.
    """
    return {
        "build_cpu_hours": 0.093,
        "byte_fallback_percent": 0.048,
        "comparison": "gpt2_bpe",
        "fewer_units_percent": 27.35,
        "format": G0_FORMAT,
        "gpt2_bpe_valid_units": 249887,
        "note": "Recorded measurements from the feasibility build. Not corpus bytes, not a training run, and not model weights.",
        "not_data": True,
        "peak_ram_gb": 4.83,
        "record_class": "recorded_metadata",
        "split": "valid",
        "valid_units": 181544,
        "vocab_size": G0_RECORDED_VOCAB,
    }


def g0_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict) or document.get("format") != G0_FORMAT:
        return [bad("g0.format", "G0 record is not sheaf-verify.g0.v1")]
    criteria = [_g0_label(document)]
    if document.get("split") != "valid":
        criteria.append(bad("g0.split", "G0 counts units on the WT103 valid split"))
    else:
        criteria.append(ok("g0.split", "valid split"))
    if document.get("comparison") != "gpt2_bpe":
        criteria.append(bad("g0.comparison", "the unit comparison is GPT-2 BPE"))
    else:
        criteria.append(ok("g0.comparison", "GPT-2 BPE"))
    try:
        units = as_int(document.get("valid_units"), "valid_units")
        gpt2 = as_int(document.get("gpt2_bpe_valid_units"), "gpt2_bpe_valid_units")
        vocab = as_int(document.get("vocab_size"), "vocab_size")
        fallback = as_real(document.get("byte_fallback_percent"), "byte_fallback_percent")
        cpu = as_real(document.get("build_cpu_hours"), "build_cpu_hours")
        ram = as_real(document.get("peak_ram_gb"), "peak_ram_gb")
    except SheafError as error:
        return criteria + [bad("g0.fields", str(error))]
    criteria.append(_fewer_units(units, gpt2))
    reported = document.get("fewer_units_percent")
    if reported is not None and gpt2 > 0:
        criteria.append(_reported_percent(reported, units, gpt2))
    if fallback <= G0_MAX_BYTE_FALLBACK_PERCENT:
        criteria.append(ok("g0.fallback", f"{fmt_float(fallback)}% byte fallback"))
    else:
        criteria.append(bad("g0.fallback", f"{fmt_float(fallback)}% byte fallback exceeds 1%"))
    criteria.append(_vocab_cap(vocab))
    if cpu <= G0_MAX_BUILD_CPU_HOURS:
        criteria.append(ok("g0.build_cpu", f"{fmt_float(cpu)} CPU-h"))
    else:
        criteria.append(bad("g0.build_cpu", f"{fmt_float(cpu)} CPU-h exceeds {G0_MAX_BUILD_CPU_HOURS}"))
    if ram <= G0_MAX_BUILD_RAM_GB:
        criteria.append(ok("g0.build_ram", f"{fmt_float(ram)} GB peak"))
    else:
        criteria.append(bad("g0.build_ram", f"{fmt_float(ram)} GB exceeds {G0_MAX_BUILD_RAM_GB}"))
    return criteria


def g1_criteria(document, train_size: int, test_size: int) -> list[Criterion]:
    arms = document.get("arms") if isinstance(document, dict) else None
    if not isinstance(arms, dict):
        return [bad("g1.arms", "G1 arms are missing")]
    missing = [name for name in G1_ARMS if name not in arms]
    if missing:
        return [bad("g1.arms", "missing " + ", ".join(missing))]
    criteria = []
    parsed = {}
    for name in G1_ARMS:
        try:
            parsed[name] = _g1_arm(arms[name], name)
        except SheafError as error:
            criteria.append(bad(f"g1.arm.{name}", str(error)))
    if len(parsed) != len(G1_ARMS):
        return criteria
    for name, arm in parsed.items():
        count = len(arm["bpb"])
        if count < PILOT_SEEDS:
            criteria.append(bad(f"g1.seeds.{name}", f"{count} seeds is below the pilot minimum {PILOT_SEEDS}"))
        else:
            criteria.append(ok(f"g1.seeds.{name}", f"{count} seeds meets the pilot minimum {PILOT_SEEDS}"))
    hnt = parsed["hnt64k"]
    struct = parsed["struct32k"]
    sw = parsed["sw64k"]
    if len(struct["bpb"]) >= 2:
        center = mean(struct["bpb"])
        sd = math.sqrt(sample_variance(struct["bpb"]))
        limit = center + G1_SD_MARGIN * sd
        hnt_mean = mean(hnt["bpb"])
        if hnt_mean <= limit:
            criteria.append(ok(
                "g1.bpb",
                f"hnt64k mean {fmt_float(hnt_mean)} <= struct32k mean + {G1_SD_MARGIN} sd ({fmt_float(limit)})",
            ))
        else:
            criteria.append(bad(
                "g1.bpb",
                f"hnt64k mean {fmt_float(hnt_mean)} exceeds struct32k + {G1_SD_MARGIN} sd ({fmt_float(limit)})",
            ))
    else:
        criteria.append(bad("g1.bpb", "struct32k needs two seeds to compute an sd"))
    for other in ("struct32k", "sw64k"):
        if hnt["flops"] < parsed[other]["flops"]:
            criteria.append(ok(
                f"g1.train_flops.{other}",
                f"hnt64k train FLOPs/byte {fmt_float(hnt['flops'])} < {other} {fmt_float(parsed[other]['flops'])}",
            ))
        else:
            criteria.append(bad(
                f"g1.train_flops.{other}",
                f"hnt64k train FLOPs/byte is not lower than {other}",
            ))
    if hnt["vocab"] == sw["vocab"]:
        criteria.append(ok("g1.vocab", f"sw64k matches hnt64k at vocab size {hnt['vocab']}"))
    else:
        criteria.append(bad("g1.vocab", "sw64k is not the same vocabulary size as hnt64k"))
    if hnt["graph"] > 0:
        criteria.append(ok("g1.graph_build", f"hnt64k records graph_build_flops {hnt['graph']}"))
    else:
        criteria.append(bad("g1.graph_build", "hnt64k graph_build_flops must be positive"))
    for name, arm in parsed.items():
        if arm["train_bytes"] != train_size:
            criteria.append(bad(f"g1.bytes.{name}", f"train_bytes does not equal the train corpus ({train_size})"))
        elif arm["denominator"] != "raw_corpus_bytes":
            criteria.append(bad(f"g1.denominator.{name}", "BPB_corpus denominator is not raw corpus bytes"))
        elif arm["scored"] != test_size or arm["raw"] != test_size:
            criteria.append(bad(f"g1.scored.{name}", "scored bytes do not equal the test corpus"))
        else:
            criteria.append(ok(f"g1.bytes.{name}", f"train bytes {train_size}, scored test bytes {test_size}"))
    return criteria


def g2_criteria(document) -> list[Criterion]:
    rows = document.get("comparisons") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        return [bad("g2.comparisons", "A2 and A3 comparisons are missing")]
    by_name = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            by_name[row["name"]] = row
    if set(G2_CONTROLS) - set(by_name):
        return [bad("g2.comparisons", "A2 spectral and A3 compositional comparisons are both required")]
    prepared = []
    criteria = []
    for name, control in G2_CONTROLS.items():
        row = by_name[name]
        if row.get("control") != control:
            criteria.append(bad(f"g2.control.{name}", f"control must be {control}"))
            continue
        try:
            primary = _series(row.get("bpb_per_seed"), f"{name} bpb")
            other = _series(row.get("control_bpb_per_seed"), f"{name} control bpb")
        except SheafError as error:
            criteria.append(bad(f"g2.values.{name}", str(error)))
            continue
        if len(primary) < PILOT_SEEDS or len(other) < PILOT_SEEDS:
            criteria.append(bad(f"g2.seeds.{name}", f"pilot minimum is {PILOT_SEEDS}"))
            continue
        criteria.append(ok(f"g2.seeds.{name}", f"pilot minimum {PILOT_SEEDS}"))
        try:
            test = welch(primary, other)
        except SheafError as error:
            criteria.append(bad(f"g2.values.{name}", str(error)))
            continue
        prepared.append((name, one_sided_less(test), mean(primary) < mean(other)))
    criteria.extend(_holm(prepared, "g2", ALPHA))
    return criteria


def g3_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("g3.fields", "kernel microbench record is missing")]
    criteria = []
    if document.get("dtype") == "bf16":
        criteria.append(ok("g3.dtype", "bf16"))
    else:
        criteria.append(bad("g3.dtype", "G3's relative-error gate is for BF16"))
    try:
        rel = as_real(document.get("rel_err"), "rel_err")
        fraction = as_real(document.get("sheafheat_step_fraction"), "sheafheat_step_fraction")
    except SheafError as error:
        return criteria + [bad("g3.fields", str(error))]
    if rel <= G3_MAX_REL_ERR:
        criteria.append(ok("g3.rel_err", f"rel err {fmt_float(rel)}"))
    else:
        criteria.append(bad("g3.rel_err", f"rel err {fmt_float(rel)} exceeds {G3_MAX_REL_ERR}"))
    if fraction <= G3_MAX_SHEAFHEAT_FRACTION:
        criteria.append(ok("g3.sheafheat", f"SheafHeat fraction {fmt_float(fraction)}"))
    else:
        criteria.append(bad("g3.sheafheat", f"SheafHeat fraction {fmt_float(fraction)} exceeds {G3_MAX_SHEAFHEAT_FRACTION}"))
    return criteria


def g4_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("g4.fields", "G4 record is missing")]
    headline = document.get("headline")
    baseline = document.get("baseline")
    components = document.get("components")
    if not isinstance(headline, dict) or not isinstance(baseline, dict) or not isinstance(components, list):
        return [bad("g4.fields", "headline, baseline, and components are required")]
    criteria = []
    try:
        hnt = _series(headline.get("bpb_per_seed"), "headline bpb")
        base = _series(baseline.get("bpb_per_seed"), "baseline bpb")
        hnt_all = as_int(headline.get("all_in_flops"), "headline all_in_flops")
        base_all = as_int(baseline.get("all_in_flops"), "baseline all_in_flops")
        hnt_train = as_int(headline.get("train_flops"), "headline train_flops")
        hnt_graph = as_int(headline.get("graph_build_flops"), "headline graph_build_flops")
    except SheafError as error:
        return [bad("g4.fields", str(error))]
    if len(hnt) < HEADLINE_SEEDS or len(base) < HEADLINE_SEEDS:
        criteria.append(bad(
            "g4.seeds.headline",
            f"headline minimum is {HEADLINE_SEEDS}; hnt has {len(hnt)}, baseline has {len(base)}",
        ))
    else:
        criteria.append(ok("g4.seeds.headline", f"headline minimum {HEADLINE_SEEDS}"))
    if len(hnt) >= 2 and len(base) >= 2:
        gain = mean(base) - mean(hnt)
        sd = math.sqrt(sample_variance(base))
        if gain > G4_MIN_BASELINE_SD * sd:
            criteria.append(ok("g4.baseline_sd", f"gain {fmt_float(gain)} > {G4_MIN_BASELINE_SD} baseline sd ({fmt_float(sd)})"))
        else:
            criteria.append(bad("g4.baseline_sd", f"gain {fmt_float(gain)} is not above {G4_MIN_BASELINE_SD} baseline sd"))
    if hnt_graph > 0 and hnt_train + hnt_graph == hnt_all:
        criteria.append(ok("g4.graph_build", "headline all-in FLOPs include a positive graph build"))
    else:
        criteria.append(bad("g4.graph_build", "headline all-in FLOPs must equal train plus a positive graph build"))
    if _matched(hnt_all, base_all):
        criteria.append(ok("g4.matched_flops", "headline and baseline all-in FLOPs match"))
    else:
        criteria.append(bad("g4.matched_flops", "all-in FLOPs are not matched"))
    by_name = {}
    for row in components:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            by_name[row["name"]] = row
    prepared = []
    for name in G4_COMPONENTS:
        row = by_name.get(name)
        if row is None:
            criteria.append(bad(f"g4.component.{name}", "component is missing"))
            continue
        if row.get("control") != "random_hypergraph":
            criteria.append(bad(f"g4.control.{name}", "control must be random_hypergraph"))
            continue
        try:
            primary = _series(row.get("bpb_per_seed"), f"{name} bpb")
            other = _series(row.get("control_bpb_per_seed"), f"{name} control bpb")
        except SheafError as error:
            criteria.append(bad(f"g4.values.{name}", str(error)))
            continue
        if len(primary) < PILOT_SEEDS or len(other) < PILOT_SEEDS:
            criteria.append(bad(f"g4.seeds.{name}", f"component minimum is the pilot count {PILOT_SEEDS}"))
            continue
        criteria.append(ok(f"g4.seeds.{name}", f"pilot minimum {PILOT_SEEDS}"))
        try:
            test = welch(primary, other)
        except SheafError as error:
            criteria.append(bad(f"g4.values.{name}", str(error)))
            continue
        prepared.append((name, one_sided_less(test), mean(primary) < mean(other)))
    criteria.extend(_holm(prepared, "g4", ALPHA))
    return criteria


def g5_criteria(document) -> list[Criterion]:
    sizes = document.get("sizes") if isinstance(document, dict) else None
    if not isinstance(sizes, list):
        return [bad("g5.sizes", "scaling sizes are missing")]
    found = {}
    criteria = []
    for row in sizes:
        if not isinstance(row, dict):
            return [bad("g5.sizes", "a size row is not an object")]
        try:
            params = as_int(row.get("params"), "params")
            gain = as_real(row.get("gain"), "gain")
        except SheafError as error:
            return [bad("g5.sizes", str(error))]
        found[params] = gain
    if set(found) != set(P4_SEED_SCHEDULE):
        return [bad("g5.sizes", "sizes must be 15M, 30M, 60M, and 120M")]
    criteria.append(ok("g5.sizes", "15M, 30M, 60M, 120M"))
    ordered = sorted(found)
    gains = [found[item] for item in ordered]
    if all(item > 0 for item in gains):
        criteria.append(ok("g5.gains", "gain is positive at every size"))
    else:
        criteria.append(bad("g5.gains", "a size has already lost the gain"))
    try:
        slope, low, high = slope_interval([math.log(float(item)) for item in ordered], gains)
    except SheafError as error:
        return criteria + [bad("g5.slope", str(error))]
    excludes = high < 0 or low > 0
    if excludes:
        criteria.append(ok("g5.slope_ci", f"slope {fmt_float(slope)} interval ({fmt_float(low)}, {fmt_float(high)}) excludes 0"))
    else:
        criteria.append(bad("g5.slope_ci", f"slope interval ({fmt_float(low)}, {fmt_float(high)}) includes 0"))
    if slope >= 0:
        criteria.append(ok("g5.not_shrinking", f"slope {fmt_float(slope)} is not negative"))
    else:
        criteria.append(bad("g5.not_shrinking", f"slope {fmt_float(slope)} says the gain is shrinking"))
    return criteria


def g6_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("g6.fields", "G6 record is missing")]
    criteria = []
    arm = document.get("arm")
    if arm in G6_ARMS:
        criteria.append(ok("g6.arm", str(arm)))
    else:
        criteria.append(bad("g6.arm", "arm must be motif_head or corpus_memory"))
    try:
        bpb = as_real(document.get("bpb"), "bpb")
        other = as_real(document.get("a7_bpb"), "a7_bpb")
        flops = as_int(document.get("all_in_flops"), "all_in_flops")
        a7_flops = as_int(document.get("a7_all_in_flops"), "a7_all_in_flops")
    except SheafError as error:
        return criteria + [bad("g6.fields", str(error))]
    if bpb < other:
        criteria.append(ok("g6.bpb", f"{fmt_float(bpb)} < A7 {fmt_float(other)}"))
    else:
        criteria.append(bad("g6.bpb", "the arm does not beat A7 on BPB"))
    if _matched(flops, a7_flops):
        criteria.append(ok("g6.matched", "compute matches A7"))
    else:
        criteria.append(bad("g6.matched", "compute is not matched to A7"))
    return criteria


def stage_criteria(metrics: dict, stage_id: str) -> list[Criterion]:
    work, gate = STAGES[stage_id]
    spec = metrics.get("spec") if isinstance(metrics, dict) else None
    stages = spec.get("stages") if isinstance(spec, dict) else None
    card = stages.get(stage_id) if isinstance(stages, dict) else None
    if not isinstance(card, dict) or card.get("work") != work or card.get("gate") != gate:
        return [bad(f"stage.{stage_id}", f"stage card must say work {work!r} and gate {gate!r}")]
    shown = "none" if gate is None else gate
    return [ok(f"stage.{stage_id}", f"{work}; gate {shown}")]


def h1_criteria(document) -> list[Criterion]:
    arms = document.get("arms") if isinstance(document, dict) else None
    gpt2 = arms.get("gpt2_bpe") if isinstance(arms, dict) else None
    hnt = arms.get("hnt64k") if isinstance(arms, dict) else None
    if not isinstance(gpt2, dict) or not isinstance(hnt, dict):
        return [bad("h1.gpt2_bpe", "the units-only pilot must include GPT-2 BPE beside hnt64k")]
    if gpt2.get("train_bytes") != hnt.get("train_bytes"):
        return [bad("h1.gpt2_bpe", "GPT-2 BPE is not byte-matched to hnt64k")]
    return [ok("h1.gpt2_bpe", "GPT-2 BPE is present and byte-matched")]


def h7_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("h7.fields", "lm-eval record is missing")]
    criteria = []
    if document.get("headline") is False:
        criteria.append(ok("h7.headline", "lm-eval is recorded and is not the headline"))
    else:
        criteria.append(bad("h7.headline", "lm-eval at this scale is reported, not headlined"))
    tasks = document.get("tasks")
    if isinstance(tasks, list) and tasks and all(isinstance(item, str) and item.strip() for item in tasks):
        criteria.append(ok("h7.tasks", ", ".join(tasks)))
    else:
        criteria.append(bad("h7.tasks", "lm-eval tasks are missing"))
    return criteria


def p0_fairness(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("p0.fields", "fairness record is missing")]
    criteria = []
    try:
        bytes_per_s = as_real(document.get("bytes_per_s"), "bytes_per_s")
        flops = as_real(document.get("flops"), "flops")
        mfu = as_real(document.get("mfu"), "mfu")
    except SheafError as error:
        return [bad("p0.fields", str(error))]
    if bytes_per_s > 0 and flops > 0 and 0 < mfu <= 1:
        criteria.append(ok("p0.logged", "bytes/s, FLOPs, and MFU are logged"))
    else:
        criteria.append(bad("p0.logged", "bytes/s, FLOPs, and MFU must be logged and positive"))
    if document.get("schedule_complete") is True:
        criteria.append(ok("p0.schedule", "the baseline schedule runs to its own end"))
    else:
        criteria.append(bad("p0.schedule", "the baseline schedule is not complete"))
    if document.get("baseline_wall_interpolated") is False:
        criteria.append(ok("p0.wall", "the comparison is not a mid-schedule interpolation"))
    else:
        criteria.append(bad("p0.wall", "a mid-schedule wall interpolation is not a fairness result"))
    return criteria


def p1_criteria(document) -> list[Criterion]:
    arms = document.get("arms") if isinstance(document, dict) else None
    if not isinstance(arms, list) or set(arms) != set(P1_ARMS):
        return [bad("p1.arms", "P1 must list S4D, hybrid, sheaf reg, random-rewire, spectral init, shape matched, and enhanced wall-match")]
    return [ok("p1.arms", "the written ablation list is present")]


def p2_criteria(document, headline_seeds: int) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("p2.fields", "strong-baseline record is missing")]
    criteria = []
    if document.get("modern_baseline") is True:
        criteria.append(ok("p2.baseline", "a tuned modern baseline is named"))
    else:
        criteria.append(bad("p2.baseline", "a tuned modern baseline is required"))
    controls = document.get("pe_controls")
    if isinstance(controls, list) and set(controls) == set(P2_PE_CONTROLS):
        criteria.append(ok("p2.pe", "sinusoidal, RoPE, ALiBi, NoPE, heat, and heat+RoPE"))
    else:
        criteria.append(bad("p2.pe", "the PE control list is incomplete"))
    declared = document.get("headline_seeds")
    try:
        declared_count = None if declared is None else as_int(declared, "headline_seeds")
    except SheafError as error:
        return criteria + [bad("p2.headline_seeds", str(error))]
    if declared_count is not None and declared_count < HEADLINE_SEEDS:
        criteria.append(bad("p2.headline_seeds", f"declared {declared_count} is below the headline minimum {HEADLINE_SEEDS}"))
    elif headline_seeds < HEADLINE_SEEDS:
        criteria.append(bad("p2.headline_seeds", f"{headline_seeds} headline seeds is below {HEADLINE_SEEDS}"))
    else:
        criteria.append(ok("p2.headline_seeds", f"headline minimum {HEADLINE_SEEDS}"))
    return criteria


def p3_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("p3.fields", "data record is missing")]
    criteria = []
    if document.get("wt103_full_epoch") is True:
        criteria.append(ok("p3.wt103", "a full WikiText-103 epoch is recorded"))
    else:
        criteria.append(bad("p3.wt103", "a full WikiText-103 epoch is required"))
    record_class = document.get("record_class")
    if record_class not in ("synthetic", "recorded_metadata") or document.get("not_data") is not True:
        criteria.append(bad("p3.label", "the data record must say whether it is synthetic or recorded metadata"))
    else:
        criteria.append(ok("p3.label", str(record_class)))
    try:
        bpb = as_real(document.get("fineweb_edu_val_bpb"), "fineweb_edu_val_bpb")
    except SheafError as error:
        return criteria + [bad("p3.fineweb", str(error))]
    if bpb > 0:
        criteria.append(ok("p3.fineweb", f"FineWeb-Edu val BPB {fmt_float(bpb)}"))
    else:
        criteria.append(bad("p3.fineweb", "FineWeb-Edu val BPB must be positive"))
    return criteria


def p4_criteria(document) -> list[Criterion]:
    sizes = document.get("sizes") if isinstance(document, dict) else None
    if not isinstance(sizes, list):
        return [bad("p4.seeds", "scaling seed counts are missing")]
    found = {}
    for row in sizes:
        if not isinstance(row, dict):
            return [bad("p4.seeds", "a size row is not an object")]
        try:
            found[as_int(row.get("params"), "params")] = as_int(row.get("seeds"), "seeds")
        except SheafError as error:
            return [bad("p4.seeds", str(error))]
    criteria = []
    if set(found) != set(P4_SEED_SCHEDULE):
        return [bad("p4.seeds", "sizes must be 15M, 30M, 60M, and 120M")]
    for params, minimum in P4_SEED_SCHEDULE.items():
        count = found[params]
        label = f"p4.seeds.{params}"
        if count < minimum:
            criteria.append(bad(label, f"{count} seeds is below the paper-plan minimum {minimum}"))
        else:
            criteria.append(ok(label, f"{count} seeds meets the paper-plan minimum {minimum}"))
    return criteria


def p5_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("p5.fields", "speed record is missing")]
    criteria = []
    if document.get("cuda_graph_all_arms") is True:
        criteria.append(ok("p5.cuda_graph", "the whole-step CUDA graph is applied to every arm"))
    else:
        criteria.append(bad("p5.cuda_graph", "the CUDA graph must be on for every arm"))
    try:
        seconds = as_real(document.get("time_to_target_bpb_s"), "time_to_target_bpb_s")
    except SheafError as error:
        return criteria + [bad("p5.time", str(error))]
    if seconds > 0:
        criteria.append(ok("p5.time", f"time to target {fmt_float(seconds)} s"))
    else:
        criteria.append(bad("p5.time", "time to target BPB must be positive"))
    return criteria


def p6_criteria(document) -> list[Criterion]:
    if not isinstance(document, dict):
        return [bad("p6.fields", "replication record is missing")]
    note = document.get("replication_note")
    if document.get("replication_recorded") is True and isinstance(note, str) and len(note.strip()) >= 20:
        return [ok("p6.replication", "a replication note is recorded")]
    return [bad("p6.replication", "replication needs a recorded note of at least 20 characters")]


def _g0_label(document: dict) -> Criterion:
    record_class = document.get("record_class")
    note = document.get("note")
    if document.get("not_data") is not True:
        return bad("g0.label", "G0 inputs must set not_data true")
    if record_class == "recorded_metadata":
        if isinstance(note, str) and "not corpus" in note.lower():
            return ok("g0.label", "recorded metadata, not corpus bytes")
        return bad("g0.label", "recorded metadata must say it is not corpus bytes")
    if record_class == "synthetic":
        return ok("g0.label", "synthetic G0 inputs, not a corpus")
    return bad("g0.label", "record_class must be synthetic or recorded_metadata")


def _fewer_units(units: int, gpt2: int) -> Criterion:
    if gpt2 <= 0 or units < 0:
        return bad("g0.fewer_units", "unit counts must be non-negative and the GPT-2 count positive")
    fewer = 1.0 - units / gpt2
    percent = fewer * 100.0
    detail = f"{percent:.2f}% fewer units on valid ({units} vs {gpt2})"
    if fewer + 1e-12 >= G0_MIN_UNIT_REDUCTION:
        return ok("g0.fewer_units", detail)
    return bad("g0.fewer_units", detail + " is below 15%")


def _reported_percent(reported, units: int, gpt2: int) -> Criterion:
    try:
        value = as_real(reported, "fewer_units_percent")
    except SheafError as error:
        return bad("g0.fewer_units_reported", str(error))
    actual = round((1.0 - units / gpt2) * 100.0 + 1e-12, 2)
    if abs(value - actual) <= 0.001:
        return ok("g0.fewer_units_reported", f"reported {fmt_float(value)}% matches the unit counts")
    return bad("g0.fewer_units_reported", f"reported {fmt_float(value)}% does not match {actual:.2f}%")


def _vocab_cap(vocab: int) -> Criterion:
    if vocab == G0_RECORDED_VOCAB:
        return bad(
            "g0.vocab_cap",
            f"V={vocab} exceeds the written cap V <= 64k ({G0_VOCAB_CAP}). "
            f"The recorded build used V={G0_RECORDED_VOCAB}, which is 2^16. "
            "That reading of 64k is inconsistent with the written cap and is flagged.",
        )
    if vocab > G0_VOCAB_CAP:
        return bad("g0.vocab_cap", f"V={vocab} exceeds the written cap {G0_VOCAB_CAP}")
    return ok("g0.vocab_cap", f"V={vocab} is within the written cap {G0_VOCAB_CAP}")


def _g1_arm(arm, name: str) -> dict:
    if not isinstance(arm, dict):
        raise SheafError(f"{name} is not an object")
    evaluation = arm.get("eval")
    if not isinstance(evaluation, dict):
        raise SheafError(f"{name} eval is missing")
    return {
        "bpb": _series(arm.get("bpb_per_seed"), f"{name} bpb"),
        "denominator": evaluation.get("denominator"),
        "flops": as_real(arm.get("train_flops_per_byte"), f"{name} train_flops_per_byte"),
        "graph": as_int(arm.get("graph_build_flops"), f"{name} graph_build_flops"),
        "raw": as_int(evaluation.get("raw_bytes"), f"{name} raw_bytes"),
        "scored": as_int(evaluation.get("scored_bytes"), f"{name} scored_bytes"),
        "train_bytes": as_int(arm.get("train_bytes"), f"{name} train_bytes"),
        "vocab": as_int(arm.get("vocab_size"), f"{name} vocab_size"),
    }


def _series(values, label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise SheafError(f"{label} is missing")
    return [as_real(item, label) for item in values]


def _holm(prepared: list[tuple[str, float, bool]], prefix: str, alpha: float) -> list[Criterion]:
    if not prepared:
        return []
    order = sorted(range(len(prepared)), key=lambda index: prepared[index][1])
    surviving = set()
    for rank, index in enumerate(order):
        threshold = alpha / (len(prepared) - rank)
        if prepared[index][1] <= threshold:
            surviving.add(prepared[index][0])
        else:
            break
    criteria = []
    for name, p_value, lower in prepared:
        if name in surviving and lower:
            criteria.append(ok(f"{prefix}.p.{name}", f"one-sided p {fmt_float(p_value)} survives Holm at {alpha}"))
        elif not lower:
            criteria.append(bad(f"{prefix}.p.{name}", "mean is not lower than the control"))
        else:
            criteria.append(bad(f"{prefix}.p.{name}", f"one-sided p {fmt_float(p_value)} does not survive Holm at {alpha}"))
    return criteria


def _matched(left: int, right: int) -> bool:
    peak = max(left, right)
    if peak < 0 or left < 0 or right < 0:
        return False
    if peak == 0:
        return True
    return (peak - min(left, right)) / peak <= MAX_FLOP_RELATIVE_TOLERANCE
