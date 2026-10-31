"""Training loop Aegis-Core (v0.1.0, prototipe untuk uji mekanika; CPU saja).

- Batch dari data_loader.get_batch(seed, step): deterministik terhadap nomor langkah.
- Checkpoint atomik; resume menolak hash tokenizer/data, config, seed, atau jadwal LR berbeda.
- Berhenti bila loss/grad-norm NaN atau inf.
- Validation loss hanya dari split eval; None bila tidak tersedia.
- Tidak mencetak atau menyimpan teks dokumen.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import data_loader as dl  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

__version__ = "0.1.0"
CKPT, LOG = "ckpt.pt", "log.jsonl"
SCHED_KEYS = ("seed", "batch_size", "block_size", "max_steps", "lr", "min_lr",
              "warmup", "weight_decay", "grad_clip")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/tokens_trial")
    ap.add_argument("--out-dir", default="data/runs_trial")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="berhenti setelah langkah ini (jadwal LR tetap memakai --max-steps)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--ckpt-interval", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def lr_at(step, lr, min_lr, warmup, max_steps):
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_ckpt(path, payload):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def evaluate(model, a, tok_sha):
    mp = Path(a.data_dir) / "meta_eval.json"
    if not mp.is_file():
        return None
    shards, _ = dl.load_split(a.data_dir, "eval", tok_sha)
    if not any(len(s) > a.block_size for s in shards):
        return None
    model.eval()
    try:
        tot = 0.0
        with torch.no_grad():
            for i in range(a.eval_batches):
                x, y = dl.get_batch(a.data_dir, "eval", a.batch_size, a.block_size,
                                    seed=a.seed, step=i, expected_tokenizer_sha256=tok_sha)
                _, loss = model(x, y)
                tot += loss.item()
        return tot / a.eval_batches
    finally:
        model.train()


def run(a):
    if a.max_steps < 1 or a.log_interval < 1 or a.ckpt_interval < 1 or a.warmup < 0:
        raise ValueError("parameter tidak valid")
    if a.resume and a.overwrite:
        raise ValueError("--resume dan --overwrite tidak boleh bersamaan")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck, lg = out / CKPT, out / LOG
    if a.overwrite:
        for p in (ck, lg):
            p.unlink(missing_ok=True)
    if a.resume:
        if not ck.is_file():
            raise FileNotFoundError(f"tidak ada checkpoint: {ck}")
    elif ck.exists() or lg.exists():
        raise FileExistsError(f"sudah ada di {out}; pakai --resume atau --overwrite")

    mpath = Path(a.data_dir) / "meta_train.json"
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    tok_sha, vocab = meta["tokenizer_json_sha256"], int(meta["vocab_size"])
    data_sha = sha256_file(mpath)
    cfg = AegisConfig(vocab_size=vocab, block_size=a.block_size, n_layer=a.n_layer,
                      n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)
    sched = {k: getattr(a, k) for k in SCHED_KEYS}

    torch.manual_seed(a.seed)
    model = AegisNano(cfg)
    params = list(model.parameters())
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": a.weight_decay},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))

    start = 0
    if a.resume:
        c = torch.load(ck, weights_only=True)
        if c["tokenizer_sha256"] != tok_sha:
            raise ValueError("hash tokenizer di checkpoint tidak cocok dengan data")
        if c["data_meta_sha256"] != data_sha:
            raise ValueError("hash meta data di checkpoint tidak cocok")
        if c["model_cfg"] != asdict(cfg):
            raise ValueError("config model di checkpoint berbeda")
        if c["sched"] != sched:
            raise ValueError("seed/jadwal/ukuran batch di checkpoint berbeda")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        torch.set_rng_state(c["torch_rng"])
        start = int(c["step"])

    end = a.max_steps if a.stop_after is None else min(a.max_steps, a.stop_after)
    if start >= end:
        raise ValueError(f"tidak ada langkah tersisa (mulai={start}, akhir={end})")

    model.train()
    losses, eval_loss = [], None
    with open(lg, "a", encoding="utf-8") as fl:
        for step in range(start, end):
            lr = lr_at(step, a.lr, a.min_lr, a.warmup, a.max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = dl.get_batch(a.data_dir, "train", a.batch_size, a.block_size,
                                seed=a.seed, step=step, expected_tokenizer_sha256=tok_sha)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss tidak hingga pada langkah {step}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            if not torch.isfinite(gn):
                raise FloatingPointError(f"grad-norm tidak hingga pada langkah {step}")
            opt.step()
            losses.append(loss.item())
            last = step + 1 == end
            if (step + 1) % a.log_interval == 0 or last:
                fl.write(json.dumps({"step": step, "loss": losses[-1], "lr": lr,
                                     "grad_norm": float(gn)}) + "\n")
                fl.flush()
            if (a.eval_interval > 0 and (step + 1) % a.eval_interval == 0) or last:
                eval_loss = evaluate(model, a, tok_sha)
                if eval_loss is not None:
                    fl.write(json.dumps({"step": step, "eval_loss": eval_loss}) + "\n")
                    fl.flush()
            if (step + 1) % a.ckpt_interval == 0 or last:
                save_ckpt(ck, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                               "step": step + 1, "model_cfg": asdict(cfg), "sched": sched,
                               "torch_rng": torch.get_rng_state(), "tokenizer_sha256": tok_sha,
                               "data_meta_sha256": data_sha, "script_version": __version__})
    return {"losses": losses, "start_step": start, "end_step": end,
            "eval_loss": eval_loss, "vocab": vocab, "ckpt": str(ck)}


def main(argv=None):
    a = parse_args(argv)
    try:
        r = run(a)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    n = min(10, len(r["losses"]))
    first = sum(r["losses"][:n]) / n
    last = sum(r["losses"][-n:]) / n
    print(f"langkah {r['start_step']}..{r['end_step']} dari {a.max_steps}")
    print(f"loss rata-rata {n} langkah pertama={first:.4f} terakhir={last:.4f} "
          f"ln(V)={math.log(r['vocab']):.4f}")
    print("eval_loss:", "tidak tersedia" if r["eval_loss"] is None else f"{r['eval_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
