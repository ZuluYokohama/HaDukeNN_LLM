#!/usr/bin/env python3
"""T-49-REAL-WIKI103: equal-bytes HST-v1 vs Byte-BPE on WikiText-103 raw.

This is not the 48 KB toy campaign. The binding protocol is
iterations/t49-hst-foundation-stage/T49_REALCORPUS_PREREG.json.
Official seeds must not start until that document is registered and committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PREREG_PATH = REPO / "iterations" / "t49-hst-foundation-stage" / "T49_REALCORPUS_PREREG.json"
DEFAULT_DATA = Path.home() / "hadukenn" / "data" / "wikitext-103-raw"
DEFAULT_WORK = Path.home() / "hadukenn" / "logs" / "t49-realcorpus"

PAUSE_C = 83.0
RESUME_C = 80.0
STOP_C = 85.0
MIN_RUN_BATTERY = 20


def decide_guard(
    temp_c: float,
    ac_online: bool,
    battery_pct: int | None,
    *,
    is_running: bool,
) -> tuple[str, str]:
    """Return (run|pause|stop, reason). Pure function, safe to unit test."""
    if temp_c >= STOP_C:
        return "stop", f"gpu {temp_c:.1f}C >= stop {STOP_C:.0f}C"
    if temp_c >= PAUSE_C:
        return "pause", f"gpu {temp_c:.1f}C >= pause {PAUSE_C:.0f}C"
    if not ac_online:
        if is_running and battery_pct is not None and battery_pct < MIN_RUN_BATTERY:
            return "stop", f"AC offline and battery {battery_pct}% < {MIN_RUN_BATTERY}%"
        return "pause", "AC offline"
    return "run", "ok"


def lock_train_bytes(
    *,
    bytes_per_sec: float,
    target: int = 200_000_000,
    minimum: int = 20_000_000,
    quantize: int = 10_000_000,
    wall_hours: float = 9.0,
    n_runs: int = 10,
    safety: float = 0.85,
) -> int:
    """Apply the pre-registered wall-clock budget rule."""
    if bytes_per_sec <= 0:
        raise ValueError("bytes_per_sec must be positive")
    affordable = (wall_hours * 3600.0 * bytes_per_sec * safety) / n_runs
    locked = min(float(target), affordable)
    locked = int(locked // quantize) * quantize
    if locked < minimum:
        raise RuntimeError(f"projected locked_bytes {locked} < minimum {minimum}; refusing to start official seeds")
    return locked


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def read_ac_battery() -> tuple[bool, int | None]:
    ac = False
    for name in ("ADP0", "ACAD", "AC0", "AC"):
        path = Path("/sys/class/power_supply") / name / "online"
        if path.is_file():
            ac = path.read_text().strip() == "1"
            break
    battery: int | None = None
    bat = Path("/sys/class/power_supply/BAT0/capacity")
    if bat.is_file():
        try:
            battery = int(bat.read_text().strip())
        except ValueError:
            battery = None
    return ac, battery


def read_gpu_temp() -> float:
    import subprocess

    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return float(raw.splitlines()[0])


def log_line(work: Path, message: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} CST {message}"
    print(line, flush=True)
    with (work / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def verify_corpus(data_dir: Path, prereg: dict) -> dict[str, Path]:
    expected = prereg["corpus"]["raw_sha256"]
    sizes = prereg["corpus"]["raw_bytes"]
    paths = {}
    for name in ("wiki.train.raw", "wiki.validation.raw", "wiki.test.raw"):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus file {path}")
        if path.stat().st_size != sizes[name]:
            raise RuntimeError(f"{name} size {path.stat().st_size} != {sizes[name]}")
        digest = sha256_file(path)
        if digest != expected[name]:
            raise RuntimeError(f"{name} sha256 {digest} != {expected[name]}")
        paths[name] = path
    return paths


def tokenizer_prefix(train_path: Path, raw_cap: int = 8 * 1024 * 1024) -> bytes:
    data = train_path.read_bytes()[:raw_cap]
    cut = data.rfind(b"\n")
    if cut <= 0:
        raise RuntimeError("tokenizer prefix has no newline")
    return data[: cut + 1]


def texts_from_prefix(prefix: bytes) -> list[str]:
    return prefix.decode("utf-8").split("\n")


def train_or_load_tokenizers(work: Path, train_path: Path, prereg: dict, log) -> dict:
    import pickle

    tok_dir = work / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)
    prefix = tokenizer_prefix(train_path)
    prefix_hash = sha256_bytes(prefix)
    meta_path = tok_dir / "manifest.json"
    hst_path = tok_dir / "hst_v1.pkl"
    bpe_path = tok_dir / "byte_bpe.pkl"
    if hst_path.is_file() and bpe_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("prefix_sha256") == prefix_hash and meta.get("vocab_size") == 8192:
            log("reusing cached tokenizers")
            return {
                "hst_v1": pickle.loads(hst_path.read_bytes()),
                "byte_bpe": pickle.loads(bpe_path.read_bytes()),
                "manifest": meta,
            }
    log(f"training tokenizers on {len(prefix)} UTF-8 bytes (prefix sha256 {prefix_hash})")
    texts = texts_from_prefix(prefix)
    from hadukenn_core.tokenization import ByteBPETokenizer
    from sheaf_verify.hst_standard import HSTTokenizer

    t0 = time.time()
    cfg = prereg["tokenizer_training"]
    hst = HSTTokenizer.train(
        texts,
        vocab_size=int(cfg["vocab_size"]),
        stage_a_fraction=float(cfg["hst"]["stage_a_fraction"]),
        min_pair_count=int(cfg["hst"]["min_pair_count"]),
        min_superspan_pmi=float(cfg["hst"]["min_superspan_pmi"]),
        max_span_bytes=int(cfg["hst"]["max_span_bytes"]),
    )
    log(f"HST tokenizer ready in {time.time() - t0:.1f}s vocab={hst.vocab_size}")
    t1 = time.time()
    bpe = ByteBPETokenizer.train(
        texts,
        vocab_size=int(cfg["vocab_size"]),
        min_pair_count=int(cfg["byte_bpe"]["min_pair_count"]),
    )
    log(f"Byte-BPE tokenizer ready in {time.time() - t1:.1f}s vocab={bpe.vocab_size}")
    meta = {
        "prefix_sha256": prefix_hash,
        "prefix_bytes": len(prefix),
        "vocab_size": int(cfg["vocab_size"]),
        "hst_vocab": int(hst.vocab_size),
        "bpe_vocab": int(bpe.vocab_size),
    }
    hst_path.write_bytes(pickle.dumps(hst, protocol=4))
    bpe_path.write_bytes(pickle.dumps(bpe, protocol=4))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return {"hst_v1": hst, "byte_bpe": bpe, "manifest": meta}


def encode_bytes(tokenizer, raw: bytes, arm: str) -> list[int]:
    text = raw.decode("utf-8", errors="strict")
    if arm == "hst_v1":
        return [int(i) for i in tokenizer.encode(text)]
    ids = tokenizer.encode(text)
    return [int(i) for i in ids]


def token_byte_lens(tokenizer, arm: str, vocab: int) -> list[int]:
    if arm == "hst_v1":
        lens = list(tokenizer.token_byte_lengths())
    else:
        lens = [len(b) for b in tokenizer.token_bytes]
    if len(lens) < vocab:
        lens = lens + [1] * (vocab - len(lens))
    return lens[:vocab]


# --- model ---
def build_model(cfg: dict, vocab: int, device, torch, nn, F):
    d = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])
    d_ff = int(cfg["d_ff"])
    if d % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = d // n_heads

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    def apply_rope(q, k):
        # q,k: (B, H, T, D)
        t = q.shape[2]
        half = head_dim // 2
        device = q.device
        dtype = torch.float32
        inv = 10000.0 ** (-torch.arange(half, device=device, dtype=dtype) / half)
        pos = torch.arange(t, device=device, dtype=dtype)
        freqs = torch.outer(pos, inv)
        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]

        def rot(x):
            xf = x.float()
            x1f, x2f = xf[..., :half], xf[..., half:]
            out1 = x1f * cos - x2f * sin
            out2 = x1f * sin + x2f * cos
            return torch.cat([out1, out2], dim=-1).type_as(x)

        return rot(q), rot(k)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = RMSNorm(d)
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            self.wo = nn.Linear(d, d, bias=False)
            self.n2 = RMSNorm(d)
            self.w1 = nn.Linear(d, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d, bias=False)
            self.w3 = nn.Linear(d, d_ff, bias=False)

        def forward(self, x):
            b, t, _ = x.shape
            h = self.n1(x)
            q = self.wq(h).view(b, t, n_heads, head_dim).transpose(1, 2)
            k = self.wk(h).view(b, t, n_heads, head_dim).transpose(1, 2)
            v = self.wv(h).view(b, t, n_heads, head_dim).transpose(1, 2)
            q, k = apply_rope(q, k)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).contiguous().view(b, t, d)
            x = x + self.wo(att)
            h = self.n2(x)
            x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
            return x

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.norm = RMSNorm(d)
            self.vocab = vocab

        def forward(self, idx, targets=None):
            x = self.tok(idx)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            logits = F.linear(x, self.tok.weight)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.float().reshape(-1, self.vocab),
                    targets.reshape(-1),
                )
            return logits, loss

    model = CausalLM()
    # GPT-style init
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(param)
    nn.init.normal_(model.tok.weight, mean=0.0, std=0.02)
    return model.to(device)


class ByteWindowStream:
    """Stream packed (x, y) windows while charging raw UTF-8 bytes once."""

    def __init__(self, raw_path: Path, tokenizer, arm: str, seq_len: int, byte_budget: int):
        self.raw_path = raw_path
        self.tokenizer = tokenizer
        self.arm = arm
        self.seq_len = seq_len
        self.byte_budget = byte_budget
        self.bytes_charged = 0
        self.eof = False
        self._token_buf: list[int] = []
        self._file = raw_path.open("rb")
        self._pending = b""

    def close(self) -> None:
        self._file.close()

    def _fill(self) -> None:
        if self.eof or self.bytes_charged >= self.byte_budget:
            return
        want = min(1 << 20, self.byte_budget - self.bytes_charged)
        chunk = self._file.read(want)
        if not chunk:
            self.eof = True
            if self._pending:
                self._token_buf.extend(encode_bytes(self.tokenizer, self._pending, self.arm))
                self.bytes_charged += len(self._pending)
                self._pending = b""
            return
        data = self._pending + chunk
        cut = data.rfind(b"\n")
        if cut < 0:
            self._pending = data
            return
        usable, self._pending = data[: cut + 1], data[cut + 1 :]
        if usable:
            self._token_buf.extend(encode_bytes(self.tokenizer, usable, self.arm))
            self.bytes_charged += len(usable)

    def next_batch(self, batch: int):
        need = batch * (self.seq_len + 1)
        while len(self._token_buf) < need and not self.eof and self.bytes_charged < self.byte_budget:
            self._fill()
        if len(self._token_buf) < self.seq_len + 1:
            return None
        take = min(batch, (len(self._token_buf) - 1) // self.seq_len)
        if take < 1:
            return None
        used = take * self.seq_len
        # we need used+1 tokens to form y
        window = self._token_buf[: used + 1]
        self._token_buf = self._token_buf[used:]
        import torch

        ids = torch.tensor(window, dtype=torch.long)
        # reshape to take x y of shape (take, seq_len)
        x = torch.empty((take, self.seq_len), dtype=torch.long)
        y = torch.empty((take, self.seq_len), dtype=torch.long)
        for i in range(take):
            sl = slice(i * self.seq_len, i * self.seq_len + self.seq_len)
            x[i] = ids[sl]
            y[i] = ids[i * self.seq_len + 1 : i * self.seq_len + self.seq_len + 1]
        return x, y


def eval_bpb(model, tokenizer, arm: str, path: Path, seq_len: int, device, torch) -> float:
    raw = path.read_bytes()
    ids = encode_bytes(tokenizer, raw, arm)
    lens = token_byte_lens(tokenizer, arm, model.vocab)
    if len(ids) < seq_len + 1:
        raise RuntimeError(f"{path.name} too short to evaluate")
    n_win = (len(ids) - 1) // seq_len
    usable = n_win * seq_len
    xs = ids[:usable]
    ys = ids[1 : usable + 1]
    total_nats = 0.0
    total_bytes = 0.0
    model.eval()
    bs = 16
    with torch.no_grad():
        for start in range(0, n_win, bs):
            end = min(n_win, start + bs)
            x = torch.tensor(
                [xs[i * seq_len : (i + 1) * seq_len] for i in range(start, end)],
                dtype=torch.long,
                device=device,
            )
            y = torch.tensor(
                [ys[i * seq_len : (i + 1) * seq_len] for i in range(start, end)],
                dtype=torch.long,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, model.vocab),
                y.reshape(-1),
                reduction="none",
            )
            total_nats += float(loss.sum().item())
            for tok in y.reshape(-1).tolist():
                total_bytes += float(lens[int(tok)])
    model.train()
    return total_nats / (max(1.0, total_bytes) * math.log(2.0))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def enforce_guard(work: Path, log, *, is_running: bool) -> None:
    while True:
        temp = read_gpu_temp()
        ac, battery = read_ac_battery()
        action, reason = decide_guard(temp, ac, battery, is_running=is_running)
        if action == "run":
            return
        if action == "stop":
            log(f"WATCHDOG STOP: {reason}")
            raise SystemExit(143)
        log(f"WATCHDOG PAUSE: {reason} (resume at <= {RESUME_C:.0f}C, AC on)")
        time.sleep(15)


def set_seeds(seed: int, torch) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_status(work: Path, payload: dict) -> None:
    (work / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_arm(
    *,
    arm: str,
    seed: int,
    byte_budget: int,
    tokenizers: dict,
    prereg: dict,
    paths: dict[str, Path],
    work: Path,
    device,
    torch,
    nn,
    F,
    log,
    probe: bool,
    probe_seconds: float | None,
) -> dict:
    cfg = prereg["model"]
    opt_cfg = prereg["optimization"]
    tokenizer = tokenizers[arm]
    vocab = int(prereg["tokenizer_training"]["vocab_size"])
    seq_len = int(cfg["seq_len"])
    set_seeds(seed, torch)
    model = build_model(cfg, vocab, device, torch, nn, F)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    micro = int(opt_cfg["microbatch_sequences"])
    accum = int(opt_cfg["grad_accum_steps"])
    warmup = int(opt_cfg["warmup_steps"])
    stream = ByteWindowStream(paths["wiki.train.raw"], tokenizer, arm, seq_len, byte_budget)
    ckpt_dir = work / "ckpt" / f"seed{seed}_{arm}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    tokens = 0
    t0 = time.time()
    first_eval = None
    first_eval_at_bytes = 2_000_000
    model.train()
    peak_lr = float(opt_cfg["lr"])
    log(f"start arm={arm} seed={seed} params={n_params} budget={byte_budget} probe={probe}")
    try:
        while stream.bytes_charged < byte_budget and not stream.eof:
            enforce_guard(work, log, is_running=True)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            got = 0
            for _ in range(accum):
                batch = stream.next_batch(micro)
                if batch is None:
                    break
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    (loss / accum).backward()
                step_loss += float(loss.item())
                got += 1
                tokens += x.numel()
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip_norm"]))
            step += 1
            # lr schedule: need estimated total steps; use bytes fraction after warmup
            if step <= warmup:
                lr = peak_lr * step / max(1, warmup)
            else:
                frac = stream.bytes_charged / max(1, byte_budget)
                lr = peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if not math.isfinite(step_loss):
                log(f"NON-FINITE loss at step {step}; arm failed")
                result = {
                    "arm": arm,
                    "seed": seed,
                    "ok": False,
                    "reason": "non_finite_loss",
                    "steps": step,
                    "bytes_charged": stream.bytes_charged,
                    "params": n_params,
                }
                append_jsonl(work / "metrics.jsonl", result)
                return result
            elapsed = time.time() - t0
            if probe and probe_seconds and elapsed >= probe_seconds:
                bps = stream.bytes_charged / max(elapsed, 1e-6)
                result = {
                    "arm": arm,
                    "seed": seed,
                    "ok": True,
                    "probe": True,
                    "seconds": elapsed,
                    "bytes_charged": stream.bytes_charged,
                    "tokens": tokens,
                    "steps": step,
                    "bytes_per_sec": bps,
                    "tokens_per_sec": tokens / max(elapsed, 1e-6),
                    "train_loss": step_loss / got,
                    "params": n_params,
                }
                append_jsonl(work / "metrics.jsonl", result)
                log(
                    f"probe arm={arm} {result['bytes_per_sec']:.0f} B/s "
                    f"{result['tokens_per_sec']:.0f} tok/s loss={result['train_loss']:.4f} "
                    f"bytes={stream.bytes_charged}"
                )
                return result
            do_eval = first_eval is None and stream.bytes_charged >= first_eval_at_bytes
            if do_eval or (step % 200 == 0):
                row = {
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "bytes_charged": stream.bytes_charged,
                    "tokens": tokens,
                    "train_loss": step_loss / got,
                    "lr": lr,
                    "seconds": elapsed,
                    "gpu_temp_c": read_gpu_temp(),
                    "params": n_params,
                }
                if do_eval:
                    row["valid_bpb"] = eval_bpb(
                        model, tokenizer, arm, paths["wiki.validation.raw"], seq_len, device, torch
                    )
                    first_eval = row["valid_bpb"]
                    row["first_eval"] = True
                    log(
                        f"FIRST EVAL arm={arm} seed={seed} valid_bpb={first_eval:.4f} "
                        f"bytes={stream.bytes_charged} step={step} loss={row['train_loss']:.4f}"
                    )
                append_jsonl(work / "metrics.jsonl", row)
                write_status(work, {"phase": "probe" if probe else "official", **row})
            if step % 50 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "bytes_charged": stream.bytes_charged,
                        "arm": arm,
                        "seed": seed,
                    },
                    ckpt_dir / "latest.pt",
                )
        # official end-of-arm test
        result = {
            "arm": arm,
            "seed": seed,
            "ok": True,
            "probe": probe,
            "steps": step,
            "bytes_charged": stream.bytes_charged,
            "tokens": tokens,
            "seconds": time.time() - t0,
            "first_valid_bpb": first_eval,
            "params": n_params,
        }
        if not probe:
            result["valid_bpb"] = eval_bpb(model, tokenizer, arm, paths["wiki.validation.raw"], seq_len, device, torch)
            result["test_bpb"] = eval_bpb(model, tokenizer, arm, paths["wiki.test.raw"], seq_len, device, torch)
            log(
                f"DONE arm={arm} seed={seed} test_bpb={result['test_bpb']:.4f} "
                f"valid_bpb={result['valid_bpb']:.4f} steps={step} bytes={stream.bytes_charged}"
            )
        append_jsonl(work / "metrics.jsonl", result)
        torch.save({"result": result, "model": model.state_dict()}, ckpt_dir / "final.pt")
        return result
    finally:
        stream.close()


def cmd_status(work: Path) -> int:
    status = work / "status.json"
    logp = work / "run.log"
    print(f"work_dir={work}")
    if status.is_file():
        print(status.read_text())
    else:
        print("no status.json yet")
    if logp.is_file():
        lines = logp.read_text(encoding="utf-8", errors="replace").splitlines()
        print("---- last 20 log lines ----")
        print("\n".join(lines[-20:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T-49 real-corpus WikiText-103 trainer")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--phase", choices=("prepare", "probe", "official", "status", "all"), default="all")
    parser.add_argument("--probe-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == "status":
        return cmd_status(work)

    def log(msg: str) -> None:
        log_line(work, msg)

    prereg = load_prereg()
    paths = verify_corpus(args.data_dir, prereg)
    log("corpus hashes verified")

    # Tokenizers are a derived artifact of the prereg, not official LM training.
    tokenizers = train_or_load_tokenizers(work, paths["wiki.train.raw"], prereg, log)

    if args.phase == "prepare":
        log("prepare complete")
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        log("CUDA is not available; refusing to run")
        return 2
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    log(
        f"torch {torch.__version__} device={torch.cuda.get_device_name(0)} cap={cap} bf16={torch.cuda.is_bf16_supported()}"
    )

    ac, battery = read_ac_battery()
    if not ac:
        log("AC is offline; refusing to start (keep the laptop on AC)")
        return 75
    log(f"AC online battery={battery}% gpu={read_gpu_temp():.0f}C")

    lock_path = work / "budget_lock.json"
    if args.phase in ("probe", "all") and not lock_path.is_file():
        log("starting discarded seed-0 throughput probe")
        rates = []
        for arm in ("byte_bpe", "hst_v1"):
            row = run_arm(
                arm=arm,
                seed=0,
                byte_budget=int(prereg["equal_bytes_protocol"]["target_train_bytes_per_arm_seed"]),
                tokenizers=tokenizers,
                prereg=prereg,
                paths=paths,
                work=work,
                device=device,
                torch=torch,
                nn=nn,
                F=F,
                log=log,
                probe=True,
                probe_seconds=args.probe_seconds,
            )
            rates.append(float(row["bytes_per_sec"]))
        bps = min(rates)
        locked = lock_train_bytes(bytes_per_sec=bps)
        lock = {
            "locked_train_bytes_per_arm_seed": locked,
            "min_probe_bytes_per_sec": bps,
            "probe_bytes_per_sec": {"byte_bpe": rates[0], "hst_v1": rates[1]},
            "locked_at_unix": time.time(),
            "rule": "T49_REALCORPUS_PREREG.json equal_bytes_protocol.lock_rule",
        }
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
        log(f"WROTE budget lock {locked} bytes/arm-seed (min probe {bps:.0f} B/s)")
        if args.phase == "probe":
            return 0

    if args.phase == "official" or args.phase == "all":
        if not lock_path.is_file():
            log("budget_lock.json missing; run --phase probe first and commit the lock before official seeds")
            return 2
        lock = json.loads(lock_path.read_text())
        locked = int(lock["locked_train_bytes_per_arm_seed"])
        marker = work / "OFFICIAL_STARTED"
        # Official start is gated by this file existing only after the caller
        # has committed the prereg + lock. The launcher creates it.
        if not marker.is_file():
            log(
                "refusing official seeds: work_dir/OFFICIAL_STARTED is absent "
                "(create it only after the prereg store and budget lock are committed)"
            )
            return 3
        official = []
        for seed in prereg["seeds"]["official"]:
            for arm in ("byte_bpe", "hst_v1"):
                done = work / "ckpt" / f"seed{seed}_{arm}" / "result.json"
                if done.is_file():
                    official.append(json.loads(done.read_text()))
                    log(f"skip completed seed={seed} arm={arm}")
                    continue
                row = run_arm(
                    arm=arm,
                    seed=seed,
                    byte_budget=locked,
                    tokenizers=tokenizers,
                    prereg=prereg,
                    paths=paths,
                    work=work,
                    device=device,
                    torch=torch,
                    nn=nn,
                    F=F,
                    log=log,
                    probe=False,
                    probe_seconds=None,
                )
                (work / "ckpt" / f"seed{seed}_{arm}" / "result.json").write_text(json.dumps(row, indent=2) + "\n")
                official.append(row)
        (work / "official_results.json").write_text(json.dumps(official, indent=2) + "\n")
        log("official schedule finished (see official_results.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
