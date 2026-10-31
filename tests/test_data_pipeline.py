"""Tes pipeline data Aegis-Core. Korpus sintetis; tanpa data nyata, tanpa jaringan."""
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "tokenizer", "scripts"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))
import data_loader as dl  # noqa: E402
import prepare_tokens as pt  # noqa: E402
import train_tokenizer as tt  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

MARKER = "RAHASIA_UJI_9f3a"
VOCAB = 300
WORDS = ["saya", "makan", "nasi", "di", "rumah", "yang", "dengan", "untuk", "belajar",
         "model", "bahasa", "kecil", "the", "learns", "from", "data", "with", "small",
         "language", "and", "tests", "café", "naïve", "日本語", "emoji🙂", "Ωmega"]


def make_docs(n=300, seed=0):
    rnd = random.Random(seed)
    return [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 40))) for _ in range(n)]


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": f"doc{i}", "text": t}, ensure_ascii=False) + "\n")


def write_manual(d, split, name, arr, tok_sha="abc"):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype="<u2")
    (d / name).write_bytes(arr.tobytes())
    mp = d / f"meta_{split}.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {
        "split": split, "tokenizer_json_sha256": tok_sha, "shards": {}}
    meta["shards"][name] = {"tokens": int(arr.size)}
    mp.write_text(json.dumps(meta))


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    texts = make_docs()
    f = d / "en_train.jsonl"
    write_jsonl(f, texts)
    tokdir = d / "tok"
    tt.train([f], VOCAB, tokdir)
    out = d / "tokens"
    meta = pt.prepare([f], "train", tokdir, out)
    return {"dir": d, "texts": texts, "file": f, "tokdir": tokdir, "out": out, "meta": meta,
            "tok": Tokenizer.from_file(str(tokdir / "tokenizer.json"))}


def test_shard_sama_dengan_encode_plus_eos(prep):
    tok = prep["tok"]
    expected = []
    for t in prep["texts"]:
        expected += tok.encode(t).ids + [0]
    got = np.fromfile(prep["out"] / "en_train.bin", dtype="<u2").tolist()
    assert got == expected
    assert prep["meta"]["shards"]["en_train.bin"]["tokens"] == len(expected)
    assert got.count(0) == len(prep["texts"])  # EOS hanya di batas dokumen


def test_rentang_id(prep):
    got = np.fromfile(prep["out"] / "en_train.bin", dtype="<u2")
    assert got.max() < prep["meta"]["vocab_size"] == VOCAB


def test_dokumen_khusus_dan_kosong_dilewati(prep, tmp_path):
    texts = ["saya makan nasi", "ada <|endoftext|> di tengah", "", "model bahasa kecil",
             "x <|pad|> y"]
    f = tmp_path / "id_train.jsonl"
    write_jsonl(f, texts)
    meta = pt.prepare([f], "train", prep["tokdir"], tmp_path / "o")
    s = meta["shards"]["id_train.bin"]
    assert (s["docs"], s["skipped_special"], s["skipped_empty"]) == (2, 2, 1)
    got = np.fromfile(tmp_path / "o" / "id_train.bin", dtype="<u2").tolist()
    exp = prep["tok"].encode(texts[0]).ids + [0] + prep["tok"].encode(texts[3]).ids + [0]
    assert got == exp


@pytest.mark.parametrize("name,split", [
    ("en_eval.jsonl", "train"), ("x_test_train.jsonl", "train"), ("en_train.txt", "train"),
    ("en_holdout_train.jsonl", "train"), ("x_eval_train.jsonl", "train"),
    ("en_train.jsonl", "eval"), ("en_test_eval.jsonl", "eval"), ("en_train_eval.jsonl", "eval"),
])
def test_tolak_input_salah_split(prep, tmp_path, name, split):
    with pytest.raises(ValueError):
        pt.prepare([tmp_path / name], split, prep["tokdir"], tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_hash_tokenizer_tidak_cocok_menghentikan(prep, tmp_path):
    bad = tmp_path / "tok_bad"
    shutil.copytree(prep["tokdir"], bad)
    tj = bad / "tokenizer.json"
    tj.write_bytes(tj.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        pt.prepare([prep["file"]], "train", bad, tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_tolak_timpa_tanpa_overwrite(prep):
    with pytest.raises(FileExistsError):
        pt.prepare([prep["file"]], "train", prep["tokdir"], prep["out"])


def test_stdout_dan_meta_tanpa_teks(prep, tmp_path, capsys):
    f = tmp_path / "en_train.jsonl"
    write_jsonl(f, [f"saya makan {MARKER}"] + make_docs(5))
    rc = pt.main(["--files", str(f), "--tokenizer-dir", str(prep["tokdir"]),
                  "--out-dir", str(tmp_path / "o")])
    cap = capsys.readouterr()
    assert rc == 0
    assert MARKER not in cap.out + cap.err and "doc0" not in cap.out + cap.err
    assert MARKER not in (tmp_path / "o" / "meta_train.json").read_text(encoding="utf-8")


def test_pergeseran_dan_kontiguitas(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(1000))
    x, y = dl.get_batch(tmp_path, "train", B=8, T=32, seed=0)
    assert x.shape == y.shape == (8, 32)
    assert x.dtype == y.dtype == torch.int64
    assert x.is_contiguous() and y.is_contiguous()
    assert torch.equal(y, x + 1)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_determinisme_get_batch(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(5000) % 997)
    a = dl.get_batch(tmp_path, "train", 4, 16, seed=1, step=3)
    b = dl.get_batch(tmp_path, "train", 4, 16, seed=1, step=3)
    c = dl.get_batch(tmp_path, "train", 4, 16, seed=1, step=4)
    d = dl.get_batch(tmp_path, "train", 4, 16, seed=2, step=3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])
    assert not torch.equal(a[0], d[0])


def test_loader_train_tidak_membaca_eval(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(500))
    write_manual(tmp_path, "eval", "en_eval.bin", np.arange(500) + 1000)
    for step in range(200):
        x, y = dl.get_batch(tmp_path, "train", 8, 16, seed=0, step=step)
        assert int(y.max()) < 1000
    x, y = dl.get_batch(tmp_path, "eval", 8, 16, seed=0)
    assert int(x.min()) >= 1000  # kontrol: split eval memang membaca shard eval


def test_split_tidak_valid_atau_meta_tidak_ada(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(500))
    for bad in ("test", "val", "holdout"):
        with pytest.raises(ValueError):
            dl.get_batch(tmp_path, bad, 2, 8, seed=0)
    with pytest.raises(FileNotFoundError):
        dl.get_batch(tmp_path, "eval", 2, 8, seed=0)


def test_loader_tolak_nama_shard_split_salah(tmp_path):
    write_manual(tmp_path, "train", "en_eval.bin", np.arange(500))
    with pytest.raises(ValueError):
        dl.get_batch(tmp_path, "train", 2, 8, seed=0)


def test_loader_tolak_hash_tokenizer_dan_ukuran(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(500), tok_sha="abc")
    dl.get_batch(tmp_path, "train", 2, 8, seed=0, expected_tokenizer_sha256="abc")
    with pytest.raises(ValueError):
        dl.get_batch(tmp_path, "train", 2, 8, seed=0, expected_tokenizer_sha256="zzz")
    p = tmp_path / "en_train.bin"
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ValueError):
        dl.get_batch(tmp_path, "train", 2, 8, seed=0)


def test_semua_shard_terlalu_pendek(tmp_path):
    write_manual(tmp_path, "train", "en_train.bin", np.arange(10))
    with pytest.raises(ValueError):
        dl.get_batch(tmp_path, "train", 2, 16, seed=0)


def test_integrasi_loader_model(prep):
    torch.manual_seed(0)
    m = AegisNano(AegisConfig(vocab_size=VOCAB, block_size=16, n_layer=1, n_head=2,
                              n_embd=32, dropout=0.0)).eval()
    x, y = dl.get_batch(prep["out"], "train", 4, 16, seed=0,
                        expected_tokenizer_sha256=prep["meta"]["tokenizer_json_sha256"])
    _, loss = m(x, y)  # targets kontigu: temuan #2 tidak muncul
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5
