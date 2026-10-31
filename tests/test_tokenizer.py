"""Tes tokenizer Aegis-Core. Korpus sintetis; tanpa data nyata, tanpa jaringan.

Tes berawalan test_karakterisasi_ merekam perilaku yang ada (bukan yang diinginkan).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tokenizer"))
import train_tokenizer as tt  # noqa: E402

MARKER = "RAHASIA_UJI_9f3a"
VOCAB = 300


def make_corpus(path, n=300, seed=0):
    rnd = random.Random(seed)
    words = ["saya", "makan", "nasi", "di", "rumah", "yang", "dengan", "untuk", "belajar",
             "model", "bahasa", "kecil", "the", "learns", "from", "data", "with", "small",
             "language", "and", "tests", "café", "naïve", "日本語", "emoji🙂", "Ωmega"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ws = [rnd.choice(words) for _ in range(rnd.randint(8, 40))]
            if i == 0:
                ws.append(MARKER)
            f.write(json.dumps({"id": f"doc{i}", "text": " ".join(ws)}, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    f = d / "en_train.jsonl"
    make_corpus(f)
    out = d / "out"
    meta = tt.train([f], VOCAB, out)
    return {"dir": d, "file": f, "out": out, "meta": meta,
            "tok": Tokenizer.from_file(str(out / "tokenizer.json"))}


FIXED = ["", " ", "  awal", "akhir  ", "a\tb\nc\r\nd", "saya makan nasi", "café naïve",
         "日本語のテキスト", "emoji 🙂👍", "Ωmega", "\u0000\u0001", "e\u0301", "x" * 500]


@pytest.mark.parametrize("s", FIXED)
def test_roundtrip_eksak(trained, s):
    tok = trained["tok"]
    assert tok.decode(tok.encode(s).ids, skip_special_tokens=False) == s


def test_roundtrip_unicode_acak(trained):
    tok = trained["tok"]
    rnd = random.Random(1)
    for _ in range(200):
        s = "".join(
            chr(rnd.choice([rnd.randint(0x20, 0x7E), rnd.randint(0xA0, 0x2FFF),
                            rnd.randint(0x1F300, 0x1F64F)]))
            for _ in range(rnd.randint(0, 60)))
        assert tok.decode(tok.encode(s).ids, skip_special_tokens=False) == s


def test_id_dalam_rentang_vocab(trained):
    tok = trained["tok"]
    ids = tok.encode(" ".join(FIXED)).ids
    assert ids and 0 <= min(ids) and max(ids) < tok.get_vocab_size()


def test_token_khusus_dan_ukuran_vocab(trained):
    tok, meta = trained["tok"], trained["meta"]
    assert tok.token_to_id("<|endoftext|>") == 0
    assert tok.token_to_id("<|pad|>") == 1
    assert tok.get_vocab_size() == VOCAB == meta["actual_vocab_size"]


def test_determinisme_pelatihan(trained, tmp_path):
    out2 = tmp_path / "out2"
    tt.train([trained["file"]], VOCAB, out2)
    assert sha(out2 / "tokenizer.json") == sha(trained["out"] / "tokenizer.json")


def test_meta_dan_stdout_tanpa_teks(trained, tmp_path, capsys):
    meta_txt = (trained["out"] / "meta.json").read_text(encoding="utf-8")
    assert MARKER not in meta_txt and "doc0" not in meta_txt
    assert trained["meta"]["stats"]["docs"] == 300
    rc = tt.main(["--files", str(trained["file"]), "--vocab-size", str(VOCAB),
                  "--out-dir", str(tmp_path / "o3")])
    cap = capsys.readouterr()
    assert rc == 0
    assert MARKER not in cap.out + cap.err and "doc0" not in cap.out + cap.err


@pytest.mark.parametrize("name", ["en_eval.jsonl", "id_eval.jsonl", "x_test.jsonl",
                                  "en_holdout_train.jsonl", "en_train.txt"])
def test_tolak_berkas_bukan_latih(tmp_path, name):
    with pytest.raises(ValueError):
        tt.train([tmp_path / name], VOCAB, tmp_path / "o")


def test_berkas_eval_tidak_pernah_dibuka(tmp_path, monkeypatch):
    tr, ev = tmp_path / "en_train.jsonl", tmp_path / "en_eval.jsonl"
    make_corpus(tr)
    make_corpus(ev, seed=5)
    opened = []
    real_open = open

    def rec_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", rec_open)
    with pytest.raises(ValueError):
        tt.train([tr, ev], VOCAB, tmp_path / "o")
    assert not any("eval" in Path(p).name.lower() for p in opened)
    assert not (tmp_path / "o" / "tokenizer.json").exists()


def test_tolak_timpa_tanpa_overwrite(trained):
    with pytest.raises(FileExistsError):
        tt.train([trained["file"]], VOCAB, trained["out"])


def test_galat_tanpa_field_text_tidak_membocorkan_nilai(tmp_path):
    f = tmp_path / "en_train.jsonl"
    f.write_text(json.dumps({"foo": MARKER}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        tt.train([f], VOCAB, tmp_path / "o")
    assert "text" in str(e.value) and MARKER not in str(e.value)


def test_karakterisasi_token_khusus_dalam_teks_menjadi_eos(trained):
    """Teks yang memuat string '<|endoftext|>' dikodekan sebagai token khusus (id 0).
    Pipeline data harus menanganinya (risiko injeksi EOS)."""
    ids = trained["tok"].encode("a <|endoftext|> b").ids
    assert 0 in ids
