"""Perbandingan ukuran vocab BPE Aegis-Core (v0.1.0).

- Tokenizer dilatih HANYA dari *_train.jsonl lewat train_tokenizer.train()
  (penjaga nama berkas berlaku). Semua pelatihan selesai SEBELUM berkas eval dibuka.
- Berkas eval harus *_eval.jsonl; nama berisi train/test/holdout ditolak sebelum dibuka.
- Hanya angka yang dicetak dan disimpan; tidak ada teks atau ID dokumen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tokenizer"))
import train_tokenizer as tt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

__version__ = "0.1.0"
MIN_VOCAB = 260


def check_eval_file(path):
    p = Path(path)
    name = p.name.lower()
    if any(w in name for w in ("train", "test", "holdout")):
        raise ValueError(f"DITOLAK (bukan data eval): {p.name}")
    if not name.endswith("_eval.jsonl"):
        raise ValueError(f"DITOLAK (nama harus *_eval.jsonl): {p.name}")
    if not p.is_file():
        raise FileNotFoundError(f"tidak ada: {p}")
    return p


def lang_of(p):
    return Path(p).name.split("_")[0]


def measure(tok, path, max_docs=None):
    m = {"docs": 0, "skipped_special": 0, "skipped_empty": 0, "chars": 0, "bytes": 0,
         "words": 0, "tokens": 0, "roundtrip_failures": 0}
    used = set()
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{Path(path).name}: baris {n}: JSON tidak valid ({e.msg})") from None
            if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                raise ValueError(f"{Path(path).name}: baris {n}: field 'text' (str) tidak ada")
            t = rec["text"]
            if any(s in t for s in tt.SPECIAL_TOKENS):
                m["skipped_special"] += 1
                continue
            if not t:
                m["skipped_empty"] += 1
                continue
            ids = tok.encode(t).ids
            if tok.decode(ids, skip_special_tokens=False) != t:
                m["roundtrip_failures"] += 1
            m["docs"] += 1
            m["chars"] += len(t)
            m["bytes"] += len(t.encode("utf-8"))
            m["words"] += len(t.split())
            m["tokens"] += len(ids)
            used.update(ids)
            if max_docs is not None and m["docs"] >= max_docs:
                break
    m["bytes_per_token"] = m["bytes"] / m["tokens"] if m["tokens"] else None
    m["tokens_per_word"] = m["tokens"] / m["words"] if m["words"] else None
    return m, used


def compare(train_files, eval_files, vocab_sizes, out_dir, n_embd=384, probe_docs=2000,
            overwrite=False):
    train_files = [tt.check_train_file(f) for f in train_files]
    eval_files = [check_eval_file(f) for f in eval_files]
    if not train_files or not eval_files:
        raise ValueError("berkas train dan eval wajib ada")
    langs = [lang_of(p) for p in eval_files]
    if len(set(langs)) != len(langs):
        raise ValueError("bahasa eval duplikat")
    sizes = sorted({int(v) for v in vocab_sizes})
    if not sizes or sizes[0] < MIN_VOCAB:
        raise ValueError(f"vocab minimal {MIN_VOCAB}")
    out = Path(out_dir)
    if not overwrite and any(p.exists() for p in [out / "results.json"] + [out / f"v{v}" for v in sizes]):
        raise FileExistsError(f"sudah ada di {out}; pakai --overwrite")

    metas, seconds = {}, {}
    for v in sizes:  # fase 1: pelatihan, hanya data train
        t0 = time.time()
        metas[v] = tt.train(train_files, v, out / f"v{v}", overwrite)
        seconds[v] = round(time.time() - t0, 1)

    results = {"script_version": __version__, "n_embd": n_embd, "probe_docs": probe_docs,
               "inputs": {p.name: {"sha256": tt.sha256_file(p), "bytes": p.stat().st_size}
                          for p in train_files + eval_files},
               "vocab": {}}
    for v in sizes:  # fase 2: pengukuran, eval baru dibuka di sini
        tok = Tokenizer.from_file(str(out / f"v{v}" / "tokenizer.json"))
        actual = tok.get_vocab_size()
        entry = {"requested": v, "actual": actual, "seconds_train": seconds[v],
                 "tokenizer_json_sha256": metas[v]["tokenizer_json_sha256"],
                 "embedding_params": actual * n_embd, "eval": {}, "train_probe": {}}
        used_all = set()
        for p, lang in zip(eval_files, langs):
            m, used = measure(tok, p)
            m["ids_used_fraction"] = len(used) / actual
            entry["eval"][lang] = m
            used_all |= used
        entry["eval_ids_used_fraction_all"] = len(used_all) / actual
        for p in train_files:
            m, _ = measure(tok, p, max_docs=probe_docs)
            entry["train_probe"][lang_of(p)] = m
        results["vocab"][str(v)] = entry

    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out / "results.json")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-files", nargs="+",
                    default=["data/sample/en_train.jsonl", "data/sample/id_train.jsonl"])
    ap.add_argument("--eval-files", nargs="+",
                    default=["data/sample/en_eval.jsonl", "data/sample/id_eval.jsonl"])
    ap.add_argument("--vocab-sizes", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--out-dir", default="data/tokenizer_cmp")
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--probe-docs", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        r = compare(a.train_files, a.eval_files, a.vocab_sizes, a.out_dir, a.n_embd,
                    a.probe_docs, a.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    for v, e in r["vocab"].items():
        print(f"== vocab diminta={v} aktual={e['actual']} latih={e['seconds_train']}s "
              f"embedding={e['embedding_params'] / 1e6:.2f}M param "
              f"id_terpakai_eval={e['eval_ids_used_fraction_all']:.3f} ==")
        for lang, m in e["eval"].items():
            tp = e["train_probe"].get(lang)
            probe = f" | in-sample B/tok={tp['bytes_per_token']:.3f}" if tp and tp["bytes_per_token"] else ""
            print(f"  {lang} eval: dok={m['docs']} lewati_khusus={m['skipped_special']} "
                  f"B/tok={m['bytes_per_token']:.3f} tok/kata={m['tokens_per_word']:.3f} "
                  f"roundtrip_gagal={m['roundtrip_failures']}{probe}")
    print(f"hasil: {Path(a.out_dir) / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
