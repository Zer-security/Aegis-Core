"""Pelatihan tokenizer BPE byte-level untuk Aegis-Core (v0.1.0, prototipe).

- Hanya membaca berkas bernama *_train.jsonl; nama berisi eval/test/holdout ditolak
  SEBELUM berkas apa pun dibuka.
- Tidak mencetak teks atau ID dokumen; hanya statistik.
- Tanpa normalisasi Unicode: encode -> decode harus eksak.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

__version__ = "0.1.0"
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]
FORBIDDEN_IN_NAME = ("eval", "test", "holdout")


def check_train_file(path):
    p = Path(path)
    name = p.name.lower()
    if any(w in name for w in FORBIDDEN_IN_NAME):
        raise ValueError(f"DITOLAK (bukan data latih): {p.name}")
    if not name.endswith("_train.jsonl"):
        raise ValueError(f"DITOLAK (nama harus *_train.jsonl): {p.name}")
    if not p.is_file():
        raise FileNotFoundError(f"tidak ada: {p}")
    return p


def iter_texts(files, stats):
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p.name}: baris {n}: JSON tidak valid ({e.msg})") from None
                if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                    keys = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
                    raise ValueError(f"{p.name}: baris {n}: field 'text' (str) tidak ada; kunci: {keys}")
                t = rec["text"]
                stats["docs"] += 1
                stats["chars"] += len(t)
                stats["utf8_bytes"] += len(t.encode("utf-8"))
                yield t


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(files, vocab_size, out_dir, overwrite=False):
    files = [check_train_file(f) for f in files]
    if not files:
        raise ValueError("tidak ada berkas latih")
    out_dir = Path(out_dir)
    tj, mj = out_dir / "tokenizer.json", out_dir / "meta.json"
    if (tj.exists() or mj.exists()) and not overwrite:
        raise FileExistsError(f"sudah ada di {out_dir}; pakai --overwrite")

    stats = {"docs": 0, "chars": 0, "utf8_bytes": 0}
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(iter_texts(files, stats), trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tj))
    meta = {
        "script_version": __version__,
        "tokenizers_version": tokenizers.__version__,
        "python": platform.python_version(),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
        "inputs": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files},
        "stats": stats,
        "tokenizer_json_sha256": sha256_file(tj),
        "catatan": "Prototipe uji alat pada data trial; BUKAN tokenizer final.",
    }
    mj.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+",
                    default=["data/sample_trial/en_train.jsonl", "data/sample_trial/id_train.jsonl"])
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--out-dir", default="data/tokenizer_trial")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        meta = train(a.files, a.vocab_size, a.out_dir, a.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    print(f"vocab diminta={meta['requested_vocab_size']} aktual={meta['actual_vocab_size']}")
    print(f"dokumen={meta['stats']['docs']} karakter={meta['stats']['chars']} "
          f"byte_utf8={meta['stats']['utf8_bytes']}")
    print(f"tokenizer.json sha256={meta['tokenizer_json_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
