#!/usr/bin/env python3
"""Read-only repetition probe for an Aegis-Nano checkpoint.

Loads a checkpoint through generate.load_model (never writes to it), decodes a
fixed prompt set with several strategies and seeds, and writes per-generation
records plus a summary to a NEW json file. No training, no data download.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as gen  # noqa: E402

__version__ = "0.1.0"

PROMPTS = {
    "en": ["The history of the city begins",
           "Water boils at a temperature of",
           "In order to learn a new language, it is important to",
           "The scientists announced that",
           "Education is important because"],
    "id": ["Sejarah kota ini dimulai",
           "Air mendidih pada suhu",
           "Untuk belajar bahasa baru, penting untuk",
           "Para ilmuwan mengumumkan bahwa",
           "Pendidikan itu penting karena"],
}

STRATEGIES = {
    "greedy":          dict(temperature=0.0, top_k=None, top_p=None, rep=1.0),
    "greedy_rep1.2":   dict(temperature=0.0, top_k=None, top_p=None, rep=1.2),
    "t1.0_pure":       dict(temperature=1.0, top_k=None, top_p=None, rep=1.0),
    "t0.8_k40":        dict(temperature=0.8, top_k=40, top_p=None, rep=1.0),
    "t0.8_k40_rep1.2": dict(temperature=0.8, top_k=40, top_p=None, rep=1.2),
    "t0.8_p0.9":       dict(temperature=0.8, top_k=None, top_p=0.9, rep=1.0),
    "t0.8_p0.9_rep1.2": dict(temperature=0.8, top_k=None, top_p=0.9, rep=1.2),
}


def next_token(logits, ctx_ids, s, g):
    logits = logits.clone()
    if s["rep"] != 1.0:  # CTRL-style penalty over prompt + generated ids
        idx = torch.tensor(sorted(set(ctx_ids)), dtype=torch.long)
        sel = logits[idx]
        logits[idx] = torch.where(sel > 0, sel / s["rep"], sel * s["rep"])
    if s["temperature"] == 0.0:
        return int(torch.argmax(logits))
    logits = logits / s["temperature"]
    if s["top_k"]:
        v, _ = torch.topk(logits, min(s["top_k"], logits.numel()))
        logits[logits < v[-1]] = float("-inf")
    if s["top_p"]:
        sl, si = torch.sort(logits, descending=True)
        pr = F.softmax(sl, dim=-1)
        remove = (torch.cumsum(pr, dim=-1) - pr) >= s["top_p"]
        logits[si[remove]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=g))


@torch.no_grad()
def decode(model, ids, eos, s, max_new, seed):
    g = torch.Generator().manual_seed(seed)
    seq, new, stopped = list(ids), [], False
    bs = model.cfg.block_size
    for _ in range(max_new):
        logits, _ = model(torch.tensor([seq[-bs:]], dtype=torch.long))
        t = next_token(logits[0, -1].float(), seq, s, g)
        if t == eos:
            stopped = True
            break
        seq.append(t)
        new.append(t)
    return new, stopped


def distinct(toks, n):
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def first_repeat(toks, n=4):
    seen = set()
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        if g in seen:
            return i
        seen.add(g)
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer-dir", default="data/tokenizer_cmp/v16384")
    ap.add_argument("--out", default="data/gen_eval/gen_eval.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"STOP: {out} sudah ada (tidak menimpa)")
    if any(p.startswith("runs") for p in out.parts):
        sys.exit("STOP: --out tidak boleh berada di dalam folder runs*")
    if a.max_new_tokens < 1 or a.seeds < 1:
        sys.exit("STOP: argumen tidak valid")

    model, tok, eos, step = gen.load_model(a.ckpt, a.tokenizer_dir)
    ckpt_sha = gen.sha256_file(a.ckpt)
    print(f"ckpt step={step} sha256={ckpt_sha[:12]}... torch_threads={torch.get_num_threads()}")

    runs, t0 = [], time.time()
    for sname, s in STRATEGIES.items():
        seeds = [1337] if s["temperature"] == 0.0 else [1337 + i for i in range(a.seeds)]
        for lang, plist in PROMPTS.items():
            for pi, prompt in enumerate(plist):
                ids = tok.encode(prompt).ids
                for seed in seeds:
                    new, stopped = decode(model, ids, eos, s, a.max_new_tokens, seed)
                    fr = first_repeat(new)
                    runs.append({"strategy": sname, "lang": lang, "prompt_idx": pi,
                                 "seed": seed, "n_tokens": len(new), "stopped_eos": stopped,
                                 "distinct": {n: distinct(new, n) for n in (1, 2, 3, 4)},
                                 "first_repeat_4gram": fr,
                                 "text": tok.decode(new, skip_special_tokens=False)})
        print(f"{sname}: done ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for sname in STRATEGIES:
        for lang in PROMPTS:
            r = [x for x in runs if x["strategy"] == sname and x["lang"] == lang]
            frs = [x["first_repeat_4gram"] for x in r if x["first_repeat_4gram"] is not None]
            summary[f"{sname}|{lang}"] = {
                "n_generations": len(r),
                "mean_tokens": mean([x["n_tokens"] for x in r]),
                "share_stopped_eos": mean([1.0 if x["stopped_eos"] else 0.0 for x in r]),
                **{f"distinct{n}": mean([x["distinct"][n] for x in r]) for n in (1, 2, 3, 4)},
                "share_with_4gram_repeat": round(len(frs) / len(r), 4),
                "mean_first_repeat_pos": mean(frs)}
    meta = {"script": "scripts/eval_generation.py", "version": __version__,
            "ckpt": str(a.ckpt), "ckpt_sha256": ckpt_sha, "step": step,
            "tokenizer_dir": a.tokenizer_dir, "max_new_tokens": a.max_new_tokens,
            "seeds": a.seeds, "strategies": STRATEGIES, "prompts": PROMPTS,
            "python": platform.python_version(), "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "summary": summary, "runs": runs},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'strategy|lang':24s} d2     d3     d4     rep4%  firstrep  tokens")
    for k, v in summary.items():
        print(f"{k:24s} {v['distinct2']}  {v['distinct3']}  {v['distinct4']}  "
              f"{v['share_with_4gram_repeat']}  {v['mean_first_repeat_pos']}  {v['mean_tokens']}")
    print("WROTE", out)


if __name__ == "__main__":
    main()
