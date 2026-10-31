"""Persiapan token Aegis-Core (v0.1.0, prototipe).

- Hanya menerima *_train.jsonl (split train) atau *_eval.jsonl (split eval, eksplisit).
  Nama berisi test/holdout ditolak sebelum berkas apa pun dibuka.
- Hash tokenizer.json harus cocok dengan meta.json tokenizer.
- Dokumen dengan token khusus literal atau kosong dilewati dan dihitung.
- Tidak mencetak teks atau ID dokumen.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

__version__ = "0.1.0"
EOS, PAD = "<|endoftext|>", "<|pad|>"
SPECIALS = (EOS, PAD)
MAX_VOCAB = 65536  # batas uint16


def check_input(path, split):
    p = Path(path)
    name = p.name.lower()
    if split not in ("train", "eval"):
        raise ValueError(f"split tidak valid: {split}")
    if any(w in name for w in ("test", "holdout")):
        raise ValueError(f"DITOLAK (dicadangkan): {p.name}")
    if not name.endswith(f"_{split}.jsonl"):
        raise ValueError(f"DITOLAK (nama harus *_{split}.jsonl): {p.name}")
    if split == "train" and "eval" in name:
        raise ValueError(f"DITOLAK (bukan data latih): {p.name}")
    if split == "eval" and "train" in name:
        raise ValueError(f"DITOLAK (bukan data eval): {p.name}")
    if not p.is_file():
        raise FileNotFoundError(f"tidak ada: {p}")
    return p


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(tok_dir):
    d = Path(tok_dir)
    tj, mj = d / "tokenizer.json", d / "meta.json"
    if not tj.is_file() or not mj.is_file():
        raise FileNotFoundError(f"tokenizer.json/meta.json tidak lengkap di {d}")
    meta = json.loads(mj.read_text(encoding="utf-8"))
    h = sha256_file(tj)
    if h != meta.get("tokenizer_json_sha256"):
        raise ValueError("hash tokenizer.json tidak cocok dengan meta.json")
    tok = Tokenizer.from_file(str(tj))
    vocab = tok.get_vocab_size()
    if vocab > MAX_VOCAB:
        raise ValueError(f"vocab {vocab} > {MAX_VOCAB} (tidak muat uint16)")
    ids = {s: tok.token_to_id(s) for s in SPECIALS}
    if any(v is None for v in ids.values()):
        raise ValueError("token khusus tidak ada di tokenizer")
    return tok, h, ids, vocab


def _write_shard(path, tok, eos, tmp, vocab):
    h = hashlib.sha256()
    docs = skipped_special = skipped_empty = tokens = 0
    with open(path, "r", encoding="utf-8") as fin, open(tmp, "wb") as fout:
        for n, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: baris {n}: JSON tidak valid ({e.msg})") from None
            if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                raise ValueError(f"{path.name}: baris {n}: field 'text' (str) tidak ada")
            t = rec["text"]
            if any(s in t for s in SPECIALS):
                skipped_special += 1
                continue
            ids = tok.encode(t).ids if t else []
            if not ids:
                skipped_empty += 1
                continue
            if max(ids) >= vocab:
                raise ValueError(f"{path.name}: baris {n}: id di luar vocab")
            b = np.asarray(ids + [eos], dtype="<u2").tobytes()
            fout.write(b)
            h.update(b)
            docs += 1
            tokens += len(ids) + 1
    return {"tokens": tokens, "docs": docs, "skipped_special": skipped_special,
            "skipped_empty": skipped_empty, "sha256": h.hexdigest(),
            "input": path.name, "input_sha256": sha256_file(path),
            "input_bytes": path.stat().st_size}


def prepare(files, split, tok_dir, out_dir, overwrite=False):
    files = [check_input(f, split) for f in files]
    if not files:
        raise ValueError("tidak ada berkas")
    tok, tok_sha, ids, vocab = load_tokenizer(tok_dir)
    names = [p.stem + ".bin" for p in files]
    if len(set(names)) != len(names):
        raise ValueError("nama berkas duplikat")
    out = Path(out_dir)
    mp = out / f"meta_{split}.json"
    if not overwrite and any(t.exists() for t in [out / n for n in names] + [mp]):
        raise FileExistsError(f"sudah ada di {out}; pakai --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    shards, partials = {}, []
    try:
        for p, n in zip(files, names):
            tmp = out / (n + ".partial")
            partials.append(tmp)
            shards[n] = _write_shard(p, tok, ids[EOS], tmp, vocab)
        for n in names:
            os.replace(out / (n + ".partial"), out / n)
    except BaseException:
        for t in partials:
            t.unlink(missing_ok=True)
        raise
    meta = {"script_version": __version__, "split": split, "dtype": "uint16",
            "tokenizer_json_sha256": tok_sha, "vocab_size": vocab, "eos_id": ids[EOS],
            "shards": shards,
            "catatan": "Prototipe uji alat pada data trial; BUKAN data final."}
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--tokenizer-dir", default="data/tokenizer_trial")
    ap.add_argument("--out-dir", default="data/tokens_trial")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    files = a.files
    if files is None:
        if a.split != "train":
            print("GALAT: split eval wajib --files eksplisit", file=sys.stderr)
            return 2
        files = ["data/sample_trial/en_train.jsonl", "data/sample_trial/id_train.jsonl"]
    try:
        meta = prepare(files, a.split, a.tokenizer_dir, a.out_dir, a.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    for n, s in meta["shards"].items():
        print(f"{n}: dokumen={s['docs']} token={s['tokens']} "
              f"lewati_khusus={s['skipped_special']} lewati_kosong={s['skipped_empty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
