"""Pure-stdlib verification and scaling-law audit for live T-38 Stages 1-3 (v1.0.0).

Audits the recorded FineWeb-Edu 32k BPE scaling campaign across:
  - Stage 1 Headline (rung M, 4 epochs, 5 seeds: 13, 21, 34, 55, 89)
  - Stage 2 Complete Ladder (XS, S, M, M+ across 1-epoch and 4-epoch cells)
  - Holm-Bonferroni family-wise multiplicity gate (alpha = 0.01)
  - Spearman rank-order stage3_stop gate over 4-epoch gaps
  - Deterministic seed-bootstrap N_body power-law exponent fit
  - Stage 3 Rung L (d=768, L=12) 0.5-epoch LR-sweep muP drift diagnostic & L-1ep-P anchor
  - IETF SCITT RFC 9943 regulatory receipt generation and verification
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from sheaf_verify.compliance import emit_regulatory_bundle, verify_regulatory_bundle
from sheaf_verify.hashing import canonical_bytes

RECORDED_T38_CELLS: dict[str, dict[str, Any]] = {
    "XS_1ep": {
        "rung": "XS",
        "epochs": 1.0,
        "d_model": 256,
        "n_layer": 4,
        "d_ff": 704,
        "n_body": 3_213_568,
        "n_tot": 11_602_176,
        "seeds": [13, 21, 34],
        "p_test": [1.3867784930785616, 1.384955807880293, 1.3825170277908632],
        "h_test": [1.4256031283816377, 1.423042896971388, 1.4205445137152666],
        "p_tflops": [28.581020, 31.317456, 31.334272],
        "h_tflops": [9.972473, 12.044760, 12.094574],
    },
    "XS_4ep": {
        "rung": "XS",
        "epochs": 4.0,
        "d_model": 256,
        "n_layer": 4,
        "d_ff": 704,
        "n_body": 3_213_568,
        "n_tot": 11_602_176,
        "seeds": [13, 21, 34],
        "p_test": [1.334516076783616, 1.3331833338837384, 1.3321368274101302],
        "h_test": [1.3648836567018499, 1.365863981495008, 1.3652332838001955],
        "p_tflops": [32.501794, 32.509756, 32.497517],
        "h_tflops": [12.544659, 12.486180, 12.557951],
    },
    "S_1ep": {
        "rung": "S",
        "epochs": 1.0,
        "d_model": 384,
        "n_layer": 6,
        "d_ff": 1024,
        "n_body": 10_621_824,
        "n_tot": 23_204_736,
        "seeds": [13, 21, 34],
        "p_test": [1.3187162481548356, 1.3230348675416228, 1.3233269390341882],
        "h_test": [1.3539595808909444, 1.3566409199914296, 1.3577214275513843],
        "p_tflops": [35.444633, 35.471774, 35.486784],
        "h_tflops": [21.801045, 21.805527, 21.803007],
    },
    "S_4ep": {
        "rung": "S",
        "epochs": 4.0,
        "d_model": 384,
        "n_layer": 6,
        "d_ff": 1024,
        "n_body": 10_621_824,
        "n_tot": 23_204_736,
        "seeds": [13, 21, 34],
        "p_test": [1.2597916630116457, 1.2590023930565752, 1.2616290361027078],
        "h_test": [1.285064498399597, 1.2849632439038912, 1.2845629855267837],
        "p_tflops": [36.373146, 36.377203, 36.384049],
        "h_tflops": [22.384559, 22.386219, 22.382067],
    },
    "M_1ep": {
        "rung": "M",
        "epochs": 1.0,
        "d_model": 576,
        "n_layer": 8,
        "d_ff": 1536,
        "n_body": 31_860_288,
        "n_tot": 50_734_656,
        "seeds": [13, 21, 34],
        "p_test": [1.277709410500094, 1.276255060042071, 1.278716535451258],
        "h_test": [1.3047528727601823, 1.305100792654188, 1.3051181511913592],
        "p_tflops": [40.599154, 40.652379, 40.637446],
        "h_tflops": [30.923461, 30.950163, 30.959525],
    },
    "M_4ep": {
        "rung": "M",
        "epochs": 4.0,
        "d_model": 576,
        "n_layer": 8,
        "d_ff": 1536,
        "n_body": 31_860_288,
        "n_tot": 50_734_656,
        "seeds": [13, 21, 34, 55, 89],
        "p_test": [
            1.2085737111564947,
            1.2094372022902276,
            1.2082487734280616,
            1.2087573674284897,
            1.2118152515441942,
        ],
        "h_test": [
            1.2276103524976905,
            1.2278304579731147,
            1.2316744930005001,
            1.2269044083651732,
            1.2267104604962311,
        ],
        "p_tflops": [41.462204, 41.371085, 41.310005, 41.331736, 41.173048],
        "h_tflops": [31.478848, 31.466149, 31.319475, 31.476473, 31.453948],
    },
    "Mp_1ep": {
        "rung": "M+",
        "epochs": 1.0,
        "d_model": 640,
        "n_layer": 10,
        "d_ff": 1728,
        "n_body": 49_575_040,
        "n_tot": 70_546_560,
        "seeds": [13, 21, 34],
        "p_test": [1.2665388499600985, 1.266631012731945, 1.2665568256311848],
        "h_test": [1.290502649367647, 1.288483139428422, 1.2896583622936884],
        "p_tflops": [40.280558, 40.292398, 40.299501],
        "h_tflops": [32.892881, 32.894748, 32.882177],
    },
    "Mp_4ep": {
        "rung": "M+",
        "epochs": 4.0,
        "d_model": 640,
        "n_layer": 10,
        "d_ff": 1728,
        "n_body": 49_575_040,
        "n_tot": 70_546_560,
        "seeds": [13, 21, 34],
        "p_test": [1.1940823415424733, 1.1938191330475543, 1.1956155014800465],
        "h_test": [1.2092269916576592, 1.2100043945112697, 1.2100843353560584],
        "p_tflops": [40.839986, 40.718439, 40.832537],
        "h_tflops": [33.206366, 33.308263, 33.308290],
    },
}

RECORDED_STAGE3_L_PARTIAL: dict[str, Any] = {
    "rung": "L",
    "d_model": 768,
    "n_layer": 12,
    "d_ff": 2048,
    "n_body": 84_953_856,
    "n_tot": 110_119_680,
    "sweep_0p5ep_s13": {
        "P": {
            "0.0005": {"val_bpb": 1.2841672461857112, "test_bpb": 1.3345427439040256, "tflops": 42.499573},
            "0.001": {"val_bpb": 1.2844158411554536, "test_bpb": 1.3353045107387385, "tflops": 43.975366},
            "0.002": {"val_bpb": 1.500160871569283, "test_bpb": 1.5610546649209704, "tflops": 44.674937},
        },
        "H": {
            "0.0005": {"val_bpb": 1.3115918076086486, "test_bpb": 1.3625025685650145, "tflops": 35.757652},
            "0.001": {"val_bpb": 1.3086194376520452, "test_bpb": 1.3586740432781836, "tflops": 38.498408},
            "0.002": {"val_bpb": 1.4916829697837566, "test_bpb": 1.5505198246235001, "tflops": 38.538161},
        },
    },
    "L_1ep_P": {
        "seeds": [13, 21, 34],
        "lr_used": 0.001,
        "p_test": [1.265382255350709, 1.2647870351814483, 1.2620752175865553],
        "p_val": [1.2179882543897986, 1.2173386348348048, 1.215022752191131],
        "p_tflops": [45.345340, 45.351381, 45.376234],
    },
}


@dataclass(frozen=True)
class CellContrast:
    cell_id: str
    rung: str
    epochs: float
    n_seeds: int
    mean_p: float
    mean_h: float
    delta_h_minus_p: float
    welch_se: float
    paired_se: float
    t_stat: float
    df: float
    p_one_sided_h_worse: float
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "delta_h_minus_p": round(self.delta_h_minus_p, 8),
            "df": round(self.df, 4),
            "epochs": self.epochs,
            "mean_h": round(self.mean_h, 8),
            "mean_p": round(self.mean_p, 8),
            "n_seeds": self.n_seeds,
            "p_one_sided_h_worse": self.p_one_sided_h_worse,
            "paired_se": round(self.paired_se, 8),
            "rung": self.rung,
            "t_stat": round(self.t_stat, 4),
            "verdict": self.verdict,
            "welch_se": round(self.welch_se, 8),
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sample_var(xs: list[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _student_t_sf(t_val: float, df: float) -> float:
    """Accurate upper-tail P(T >= t_val) for Student-t via Simpson quadrature."""
    if t_val <= 0.0:
        return 1.0 - _student_t_sf(-t_val, df)
    steps = 400
    a = 0.0
    b = math.pi / 2.0 - 1e-9
    h = (b - a) / steps
    log_norm = math.lgamma((df + 1.0) / 2.0) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2.0)

    def integrand(theta: float) -> float:
        tan_th = math.tan(theta)
        sec2 = 1.0 + tan_th * tan_th
        x = t_val + tan_th
        log_pdf = log_norm - 0.5 * (df + 1.0) * math.log(1.0 + (x * x) / df)
        return math.exp(log_pdf) * sec2

    total = integrand(a) + integrand(b)
    for i in range(1, steps):
        theta = a + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * integrand(theta)
    area = total * h / 3.0
    return max(0.0, min(0.5, area))


def compute_cell_contrast(cell_id: str, cell: dict[str, Any]) -> CellContrast:
    p_vals = list(cell["p_test"])
    h_vals = list(cell["h_test"])
    n = len(p_vals)
    mp = _mean(p_vals)
    mh = _mean(h_vals)
    delta = mh - mp
    vp = _sample_var(p_vals)
    vh = _sample_var(h_vals)
    welch_se = math.sqrt(vp / n + vh / n)
    diffs = [h - p for h, p in zip(h_vals, p_vals, strict=True)]
    paired_se = math.sqrt(_sample_var(diffs) / n)
    num = (vp / n + vh / n) ** 2
    den = ((vp / n) ** 2) / (n - 1) + ((vh / n) ** 2) / (n - 1)
    df = num / den
    t_stat = delta / welch_se
    p_worse = _student_t_sf(t_stat, df)
    verdict = "KILL" if (delta >= 2.0 * welch_se and p_worse < 0.01) else "INCONCLUSIVE"
    return CellContrast(
        cell_id=cell_id,
        rung=str(cell["rung"]),
        epochs=float(cell["epochs"]),
        n_seeds=n,
        mean_p=mp,
        mean_h=mh,
        delta_h_minus_p=delta,
        welch_se=welch_se,
        paired_se=paired_se,
        t_stat=t_stat,
        df=df,
        p_one_sided_h_worse=p_worse,
        verdict=verdict,
    )


def holm_bonferroni_gate(contrasts: list[CellContrast], alpha: float = 0.01) -> dict[str, Any]:
    ordered = sorted(contrasts, key=lambda c: c.p_one_sided_h_worse)
    m = len(ordered)
    passed_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    still_rejecting = True
    for idx, c in enumerate(ordered):
        threshold = alpha / (m - idx)
        reject = still_rejecting and (c.p_one_sided_h_worse <= threshold) and (c.delta_h_minus_p >= 2.0 * c.welch_se)
        if not reject:
            still_rejecting = False
        else:
            passed_ids.append(c.cell_id)
        rows.append(
            {
                "cell_id": c.cell_id,
                "holm_threshold": threshold,
                "p_one_sided_h_worse": c.p_one_sided_h_worse,
                "rejected_in_favor_of_P": reject,
            }
        )
    return {
        "alpha_family": alpha,
        "cells_killing_H": len(passed_ids),
        "rows": rows,
        "total_cells": m,
    }


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Spearman requires equal-length sequences of size >= 2")

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = _mean(rx)
    my = _mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


def fit_power_law_grid(ns: list[float], losses: list[float]) -> tuple[float, float, float]:
    """Fit L(N) = E + A * N^(-alpha) by profile least-squares over E."""
    min_loss = min(losses)
    log_ns = [math.log(n) for n in ns]
    mean_x = _mean(log_ns)
    sxx = sum((x - mean_x) ** 2 for x in log_ns)

    def sse_for_e(e_val: float) -> tuple[float, float, float]:
        ys = [math.log(l_val - e_val) for l_val in losses]
        mean_y = _mean(ys)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ns, ys, strict=True)) / sxx
        alpha = -slope
        log_a = mean_y - slope * mean_x
        a_val = math.exp(log_a)
        sse = sum((l_val - (e_val + a_val * (n_val ** (-alpha)))) ** 2 for n_val, l_val in zip(ns, losses, strict=True))
        return sse, alpha, a_val

    best_e = 0.90
    best_sse = float("inf")
    best_alpha = 0.0
    best_a = 0.0
    upper_e = min_loss - 0.015
    steps = 300
    for i in range(steps + 1):
        e_try = 0.85 + (upper_e - 0.85) * (i / steps)
        sse, alpha, a_val = sse_for_e(e_try)
        if sse < best_sse:
            best_sse, best_e, best_alpha, best_a = sse, e_try, alpha, a_val

    lo = max(0.75, best_e - 0.01)
    hi = min(min_loss - 0.002, best_e + 0.01)
    for i in range(201):
        e_try = lo + (hi - lo) * (i / 200)
        sse, alpha, a_val = sse_for_e(e_try)
        if sse < best_sse:
            best_sse, best_e, best_alpha, best_a = sse, e_try, alpha, a_val
    return best_e, best_alpha, best_a


def bootstrap_scaling_exponent(
    cells_4ep: tuple[str, ...] = ("XS_4ep", "S_4ep", "M_4ep", "Mp_4ep"),
    n_boot: int = 400,
    seed: int = 20261126,
    param_key: str = "n_body",
) -> dict[str, Any]:
    ns = [float(RECORDED_T38_CELLS[cid][param_key]) / 1e6 for cid in cells_4ep]
    p_means = [_mean(RECORDED_T38_CELLS[cid]["p_test"]) for cid in cells_4ep]
    h_means = [_mean(RECORDED_T38_CELLS[cid]["h_test"]) for cid in cells_4ep]
    ep, alpha_p, ap = fit_power_law_grid(ns, p_means)
    eh, alpha_h, ah = fit_power_law_grid(ns, h_means)
    point_diff = alpha_h - alpha_p

    p_excess = [math.log(y - ep) for y in p_means]
    h_excess = [math.log(y - ep) for y in h_means]
    mean_shift = _mean([hx - px for hx, px in zip(h_excess, p_excess, strict=True)])
    k_eff = math.exp(-mean_shift / alpha_p)

    rng = random.Random(seed)
    boot_diffs: list[float] = []
    for _ in range(n_boot):
        p_draw: list[float] = []
        h_draw: list[float] = []
        for cid in cells_4ep:
            ps = RECORDED_T38_CELLS[cid]["p_test"]
            hs = RECORDED_T38_CELLS[cid]["h_test"]
            k = len(ps)
            idxs = [rng.randrange(k) for _ in range(k)]
            p_draw.append(_mean([ps[i] for i in idxs]))
            h_draw.append(_mean([hs[i] for i in idxs]))
        _, ap_b, _ = fit_power_law_grid(ns, p_draw)
        _, ah_b, _ = fit_power_law_grid(ns, h_draw)
        boot_diffs.append(ah_b - ap_b)

    boot_diffs.sort()
    lo_idx = int(0.025 * (n_boot - 1))
    hi_idx = int(0.975 * (n_boot - 1))
    ci_lo = boot_diffs[lo_idx]
    ci_hi = boot_diffs[hi_idx]
    pass_scaling = ci_lo > 0.0
    return {
        "A_H": round(ah, 6),
        "A_P": round(ap, 6),
        "E_H": round(eh, 6),
        "E_P": round(ep, 6),
        "alpha_H": round(alpha_h, 6),
        "alpha_H_minus_alpha_P": round(point_diff, 6),
        "alpha_P": round(alpha_p, 6),
        "ci95_high": round(ci_hi, 6),
        "ci95_low": round(ci_lo, 6),
        "effective_capacity_ratio_H_over_P": round(k_eff, 4),
        "n_boot": n_boot,
        "param_key": param_key,
        "pass_scaling_in_favor_of_H": pass_scaling,
    }


def run_t38_live_audit(n_boot: int = 400) -> dict[str, Any]:
    ordered_ids = ["XS_1ep", "XS_4ep", "S_1ep", "S_4ep", "M_1ep", "M_4ep", "Mp_1ep", "Mp_4ep"]
    contrasts = [compute_cell_contrast(cid, RECORDED_T38_CELLS[cid]) for cid in ordered_ids]
    by_id = {c.cell_id: c for c in contrasts}

    holm = holm_bonferroni_gate(contrasts, alpha=0.01)
    four_ep_ids = ["XS_4ep", "S_4ep", "M_4ep", "Mp_4ep"]
    ns_4ep = [float(RECORDED_T38_CELLS[cid]["n_body"]) for cid in four_ep_ids]
    gaps_4ep = [by_id[cid].delta_h_minus_p for cid in four_ep_ids]
    rho = spearman_rank_correlation(ns_4ep, gaps_4ep)
    stage3_stop_fired = by_id["M_4ep"].verdict == "KILL" and by_id["M_4ep"].delta_h_minus_p > 0.0117 and rho >= 0.0

    fit_nbody = bootstrap_scaling_exponent(n_boot=n_boot, param_key="n_body")
    fit_ntot = bootstrap_scaling_exponent(n_boot=max(200, n_boot // 2), param_key="n_tot")

    l_sweep_p = RECORDED_STAGE3_L_PARTIAL["sweep_0p5ep_s13"]["P"]
    l_sweep_h = RECORDED_STAGE3_L_PARTIAL["sweep_0p5ep_s13"]["H"]
    p_best_full_val_lr = min(l_sweep_p, key=lambda lr: l_sweep_p[lr]["val_bpb"])
    h_best_full_val_lr = min(l_sweep_h, key=lambda lr: l_sweep_h[lr]["val_bpb"])
    l_1ep_p_mean = _mean(RECORDED_STAGE3_L_PARTIAL["L_1ep_P"]["p_test"])
    l_1ep_p_tflops = _mean(RECORDED_STAGE3_L_PARTIAL["L_1ep_P"]["p_tflops"])

    payload_core = {
        "cells": [c.as_dict() for c in contrasts],
        "headline_M_4ep": by_id["M_4ep"].as_dict(),
        "holm_bonferroni_alpha_0p01": holm,
        "scaling_fit_n_body": fit_nbody,
        "scaling_fit_n_tot": fit_ntot,
        "spearman_rho_4ep_vs_N": round(rho, 6),
        "stage3_L_diagnostic": {
            "L_1ep_P_lr_used": RECORDED_STAGE3_L_PARTIAL["L_1ep_P"]["lr_used"],
            "L_1ep_P_mean_test_bpb": round(l_1ep_p_mean, 8),
            "L_1ep_P_mean_tflops": round(l_1ep_p_tflops, 4),
            "P_lr0p0005_minus_lr0p001_val_bpb": round(
                l_sweep_p["0.0005"]["val_bpb"] - l_sweep_p["0.001"]["val_bpb"], 8
            ),
            "P_lr0p002_spike_test_bpb": round(l_sweep_p["0.002"]["test_bpb"], 6),
            "h_best_full_val_lr": h_best_full_val_lr,
            "mup_width_drift_detected": p_best_full_val_lr == "0.0005",
            "p_best_full_val_lr": p_best_full_val_lr,
        },
        "stage3_stop_fired": stage3_stop_fired,
    }
    digest = hashlib.sha256(canonical_bytes(payload_core)).hexdigest()
    policy_sha = hashlib.sha256(b"hadukenn.t38.prereg.v1").hexdigest()
    signing_key = b"hadukenn-t38-audit-signing-key-v1"
    events = [
        {
            "approved": True,
            "cell_id": c.cell_id,
            "delta_bpb": round(c.delta_h_minus_p, 6),
            "status": "ok",
            "verdict": "VALID",
        }
        for c in contrasts
    ]
    bundle = emit_regulatory_bundle(
        system_id="hadukenn-t38-scaling-auditor",
        trace_head_sha256=digest,
        state_digest_sha256=digest,
        policy_sha256=policy_sha,
        events=events,
        created_at="2026-11-26T19:55:00-06:00",
        signing_key=signing_key,
        retention_months=84,
    )
    scitt_ok, scitt_problems = verify_regulatory_bundle(bundle, signing_key=signing_key)
    return {
        **payload_core,
        "audit_commitment_sha256": digest,
        "scitt_rfc9943_bundle": bundle,
        "scitt_verified": scitt_ok and not scitt_problems,
    }
