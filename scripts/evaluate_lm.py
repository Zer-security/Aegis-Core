"""Evaluasi loss/perplexity Aegis-Core pada split eval (v0.1.0, prototipe).

- Jendela tanpa tumpang tindih sepanjang T; target digeser satu token; pada stride 1
  setiap token (kecuali yang pertama) menjadi target tepat sekali (EOS ikut dihitung).
- Hanya membuka meta_eval.json; split lain tidak bisa dievaluasi.
- Hash tokenizer di meta data harus sama dengan di checkpoint; ukuran vocab harus cocok.
- Keluaran hanya angka; tidak ada teks atau ID dokumen.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import data_loader as dl  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

__version__ = "0.1.0"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"tidak ada checkpoint: {p}")
    c = torch.load(p, map_location="cpu", weights_only=True)
    for k in ("model", "model_cfg", "tokenizer_sha256", "step"):
        if k not in c:
            raise ValueError(f"checkpoint tidak memuat '{k}'")
    model = AegisNano(AegisConfig(**c["model_cfg"]))
    model.load_state_dict(c["model"])
    model.eval()
    return model, c


def _sum_ce(model, x, y):
    logits, _ = model(x)
    v = logits.size(-1)
    s = F.cross_entropy(logits.reshape(-1, v).float(), y.reshape(-1), reduction="none")
    return s.double().sum().item(), int(y.numel())


def eval_shard(model, arr, T, batch, stride):
    if stride < 1 or batch < 1 or T < 1:
        raise ValueError("stride, batch, dan T harus >= 1")
    if T > model.cfg.block_size:
        raise ValueError(f"T={T} melebihi block_size={model.cfg.block_size}")
    n = len(arr)
    starts = list(range(0, n - 1, T))[::stride]
    full = [s for s in starts if n - 1 - s >= T]
    part = [s for s in starts if n - 1 - s < T]
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(full), batch):
            ss = full[i:i + batch]
            x = torch.from_numpy(np.stack([np.asarray(arr[s:s + T], dtype=np.int64) for s in ss]))
            y = torch.from_numpy(np.stack([np.asarray(arr[s + 1:s + T + 1], dtype=np.int64)
                                           for s in ss]))
            a, b = _sum_ce(model, x, y)
            total += a
            count += b
        for s in part:
            m = n - 1 - s
            x = torch.from_numpy(np.asarray(arr[s:s + m], dtype=np.int64))[None]
            y = torch.from_numpy(np.asarray(arr[s + 1:s + 1 + m], dtype=np.int64))
            a, b = _sum_ce(model, x, y)
            total += a
            count += b
    return total, count


def _entry(loss_sum, tokens):
    loss = loss_sum / tokens
    return {"tokens": tokens, "loss": loss, "perplexity": math.exp(loss)}


def evaluate_dataset(model, ckpt_tok_sha, data_dir, T, batch, stride):
    shards, meta = dl.load_split(data_dir, "eval", expected_tokenizer_sha256=ckpt_tok_sha)
    if int(meta.get("vocab_size", -1)) != model.cfg.vocab_size:
        raise ValueError("ukuran vocab di meta data tidak cocok dengan model")
    names = [n for n, i in sorted(meta["shards"].items()) if i["tokens"] > 0]
    out, tl, tt = {}, 0.0, 0
    for name, arr in zip(names, shards):
        lang = name.split("_")[0]
        ls, nt = eval_shard(model, arr, T, batch, stride)
        if nt == 0:
            raise ValueError(f"{name}: tidak ada token yang dievaluasi")
        out[lang] = _entry(ls, nt)
        tl += ls
        tt += nt
    if tt == 0:
        raise ValueError("tidak ada shard eval")
    out["_all"] = _entry(tl, tt)
    return out


def parse_kv(s, what):
    if "=" not in s:
        raise ValueError(f"{what} harus berbentuk NAMA=...: {s}")
    k, v = s.split("=", 1)
    if not k or not v:
        raise ValueError(f"{what} tidak valid: {s}")
    return k, v


def parse_set(s, default_stride):
    name, rest = parse_kv(s, "--eval-set")
    stride = default_stride
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if tail.isdigit():
            rest, stride = head, int(tail)
    if stride < 1:
        raise ValueError("stride harus >= 1")
    return name, rest, stride


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAMA=path")
    ap.add_argument("--eval-set", nargs="+", required=True,
                    help="NAMA=direktori[:stride] (direktori berisi meta_eval.json)")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--out", default="data/eval_results/results.json")
    a = ap.parse_args(argv)
    try:
        ckpts = [parse_kv(s, "--ckpt") for s in a.ckpt]
        sets = [parse_set(s, a.window_stride) for s in a.eval_set]
        if len({n for n, _ in ckpts}) != len(ckpts) or len({n for n, _, _ in sets}) != len(sets):
            raise ValueError("nama checkpoint atau eval-set duplikat")
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(f"sudah ada: {out}")
        results = {"script_version": __version__, "block_size": a.block_size,
                   "batch_size": a.batch_size, "torch_threads": torch.get_num_threads(),
                   "eval_sets": {}, "runs": {}}
        for name, d, stride in sets:
            mp = Path(d) / "meta_eval.json"
            if not mp.is_file():
                raise FileNotFoundError(f"tidak ada: {mp}")
            results["eval_sets"][name] = {"dir": str(d), "meta_eval_sha256": sha256_file(mp),
                                          "window_stride": stride}
        for cname, cpath in ckpts:
            model, c = load_checkpoint(cpath)
            entry = {"path": str(cpath), "sha256": sha256_file(cpath), "step": int(c["step"]),
                     "params": model.num_params(), "model_cfg": c["model_cfg"], "datasets": {}}
            for sname, d, stride in sets:
                t0 = time.time()
                r = evaluate_dataset(model, c["tokenizer_sha256"], d, a.block_size,
                                     a.batch_size, stride)
                sec = round(time.time() - t0, 1)
                entry["datasets"][sname] = {"stride": stride, "seconds": sec, "per_lang": r}
                parts = " | ".join(f"{k}: tok={v['tokens']} loss={v['loss']:.4f} ppl={v['perplexity']:.1f}"
                                   for k, v in r.items() if k != "_all")
                al = r["_all"]
                print(f"[{cname} langkah={entry['step']}] set={sname} stride={stride} {parts} | "
                      f"semua: tok={al['tokens']} loss={al['loss']:.4f} ppl={al['perplexity']:.1f} "
                      f"({sec}s)", flush=True)
            results["runs"][cname] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)
        print(f"hasil: {out}")
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
