"""Tes compare_vocab. Korpus sintetis; tanpa data nyata, tanpa jaringan."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tokenizer"))
import compare_vocab as cv  # noqa: E402
import train_tokenizer as tt  # noqa: E402
from test_tokenizer import MARKER, make_corpus  # noqa: E402

SIZES = [270, 300]


def mk(d):
    d.mkdir(parents=True, exist_ok=True)
    make_corpus(d / "en_train.jsonl", n=200, seed=0)
    make_corpus(d / "id_train.jsonl", n=200, seed=1)
    make_corpus(d / "en_eval.jsonl", n=40, seed=7)
    make_corpus(d / "id_eval.jsonl", n=40, seed=8)
    return d


def run(d, out="out", sizes=SIZES, **kw):
    return cv.compare([d / "en_train.jsonl", d / "id_train.jsonl"],
                      [d / "en_eval.jsonl", d / "id_eval.jsonl"],
                      sizes, d / out, probe_docs=30, **kw)


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    d = mk(tmp_path_factory.mktemp("cmp"))
    return d, run(d)


def test_struktur_dan_roundtrip(res):
    d, r = res
    for v in SIZES:
        e = r["vocab"][str(v)]
        assert (d / "out" / f"v{v}" / "tokenizer.json").is_file()
        assert e["actual"] <= v and e["embedding_params"] == e["actual"] * 384
        for lang in ("en", "id"):
            m = e["eval"][lang]
            assert m["docs"] == 40 and m["roundtrip_failures"] == 0
            assert m["bytes_per_token"] > 1 and 0 < m["ids_used_fraction"] <= 1
            assert e["train_probe"][lang]["docs"] == 30


def test_vocab_lebih_besar_token_tidak_bertambah(res):
    _, r = res
    for lang in ("en", "id"):
        assert r["vocab"]["300"]["eval"][lang]["tokens"] <= r["vocab"]["270"]["eval"][lang]["tokens"]


def test_hasil_dan_stdout_tanpa_teks(tmp_path, capsys):
    d = mk(tmp_path)
    rc = cv.main(["--train-files", str(d / "en_train.jsonl"), str(d / "id_train.jsonl"),
                  "--eval-files", str(d / "en_eval.jsonl"), str(d / "id_eval.jsonl"),
                  "--vocab-sizes", "270", "300", "--out-dir", str(d / "o"), "--probe-docs", "30"])
    cap = capsys.readouterr()
    txt = (d / "o" / "results.json").read_text(encoding="utf-8")
    assert rc == 0
    for s in (cap.out + cap.err, txt):
        assert MARKER not in s and "doc0" not in s


def test_eval_tidak_dibuka_sebelum_pelatihan_selesai(tmp_path, monkeypatch):
    d = mk(tmp_path)
    events, real_open, real_train = [], open, tt.train

    def rec(file, *a, **k):
        events.append(("open", Path(str(file)).name))
        return real_open(file, *a, **k)

    def wrapped(*a, **k):
        r = real_train(*a, **k)
        events.append(("train_done", None))
        return r

    monkeypatch.setattr("builtins.open", rec)
    monkeypatch.setattr(tt, "train", wrapped)
    run(d)
    last_train = max(i for i, e in enumerate(events) if e[0] == "train_done")
    eval_idx = [i for i, e in enumerate(events) if e[0] == "open" and "eval" in e[1]]
    assert eval_idx and min(eval_idx) > last_train


def test_tolak_eval_di_daftar_train(tmp_path):
    d = mk(tmp_path)
    with pytest.raises(ValueError):
        cv.compare([d / "en_eval.jsonl"], [d / "id_eval.jsonl"], SIZES, d / "o")
    assert not (d / "o").exists()


def test_tolak_train_di_daftar_eval(tmp_path):
    d = mk(tmp_path)
    with pytest.raises(ValueError):
        cv.compare([d / "en_train.jsonl"], [d / "id_train.jsonl"], SIZES, d / "o")
    assert not (d / "o").exists()


def test_tolak_timpa_dan_vocab_kecil(tmp_path):
    d = mk(tmp_path)
    run(d)
    with pytest.raises(FileExistsError):
        run(d)
    with pytest.raises(ValueError):
        run(d, out="o2", sizes=[100])
    assert not (d / "o2").exists()


def test_dokumen_token_khusus_dilewati(tmp_path):
    d = mk(tmp_path)
    with open(d / "en_eval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "text": "a <|endoftext|> b"}) + "\n")
    r = run(d)
    m = r["vocab"]["270"]["eval"]["en"]
    assert m["skipped_special"] == 1 and m["docs"] == 40
