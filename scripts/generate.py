"""Sampling Aegis-Core dari checkpoint (v0.1.0, prototipe).

- Checkpoint dimuat dengan weights_only=True; hash tokenizer.json harus cocok.
- sample() memanggil model.eval() sendiri (dropout mati).
- Seed diset ulang per prompt; teks dipotong di EOS pertama.
- Prompt dengan token khusus literal atau terlalu panjang ditolak.
- Keluaran dicetak sebagai JSON (karakter kontrol di-escape).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import AegisConfig, AegisNano  # noqa: E402

__version__ = "0.1.0"
EOS, PAD = "<|endoftext|>", "<|pad|>"
DEFAULT_PROMPTS = ["The history of", "Education is", "Indonesia adalah",
                   "Pada zaman dahulu", ""]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path, tok_dir):
    ckpt_path = Path(ckpt_path)
    tj = Path(tok_dir) / "tokenizer.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"tidak ada checkpoint: {ckpt_path}")
    if not tj.is_file():
        raise FileNotFoundError(f"tidak ada tokenizer: {tj}")
    c = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for k in ("model", "model_cfg", "tokenizer_sha256", "step"):
        if k not in c:
            raise ValueError(f"checkpoint tidak memuat '{k}'")
    if sha256_file(tj) != c["tokenizer_sha256"]:
        raise ValueError("hash tokenizer.json tidak cocok dengan checkpoint")
    tok = Tokenizer.from_file(str(tj))
    cfg = AegisConfig(**c["model_cfg"])
    if tok.get_vocab_size() != cfg.vocab_size:
        raise ValueError("ukuran vocab tokenizer tidak cocok dengan model")
    eos = tok.token_to_id(EOS)
    if eos is None:
        raise ValueError("token EOS tidak ada di tokenizer")
    model = AegisNano(cfg)
    model.load_state_dict(c["model"])
    model.eval()
    return model, tok, eos, int(c["step"])


def sample(model, tok, eos, prompt, max_new_tokens, temperature, top_k, seed):
    if temperature <= 0 or max_new_tokens < 1 or (top_k is not None and top_k < 1):
        raise ValueError("parameter sampling tidak valid")
    if EOS in prompt or PAD in prompt:
        raise ValueError("prompt memuat token khusus")
    ids = tok.encode(prompt).ids if prompt else [eos]
    if len(ids) >= model.cfg.block_size:
        raise ValueError(f"prompt terlalu panjang ({len(ids)} token, block_size "
                         f"{model.cfg.block_size})")
    model.eval()
    torch.manual_seed(seed)
    out = model.generate(torch.tensor([ids], dtype=torch.long), max_new_tokens,
                         temperature=temperature, top_k=top_k)[0].tolist()
    new = out[len(ids):]
    stopped = eos in new
    if stopped:
        new = new[:new.index(eos)]
    return tok.decode(new, skip_special_tokens=False), len(new), stopped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/runs_v1/ckpt.pt")
    ap.add_argument("--tokenizer-dir", default="data/tokenizer_cmp/v16384")
    ap.add_argument("--prompt", nargs="*", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args(argv)
    prompts = DEFAULT_PROMPTS if a.prompt is None else a.prompt
    try:
        model, tok, eos, step = load_model(a.ckpt, a.tokenizer_dir)
        print(f"checkpoint langkah={step} param={model.num_params()} "
              f"temperature={a.temperature} top_k={a.top_k} seed={a.seed} "
              f"mode_train={model.training}")
        for i, p in enumerate(prompts):
            text, n, stopped = sample(model, tok, eos, p, a.max_new_tokens,
                                      a.temperature, a.top_k, a.seed)
            print(f"[{i}] prompt={json.dumps(p, ensure_ascii=False)} -> "
                  f"{json.dumps(text, ensure_ascii=False)} (token={n}, berhenti_EOS={stopped})")
    except (ValueError, FileNotFoundError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
