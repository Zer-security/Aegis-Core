"""Tes evaluate_lm. Data dan checkpoint sintetis; tanpa data nyata, tanpa jaringan."""
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "scripts"):
    sys.path.insert(0, str(p))
import evaluate_lm as ev  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

V, T = 64, 16


def cfg():
    return AegisConfig(vocab_size=V, block_size=T, n_layer=1, n_head=2, n_embd=32, dropout=0.1)


def make_model(seed=0):
    torch.manual_seed(seed)
    return AegisNano(cfg()).eval()


def rand_arr(n, seed=0):
    return np.random.default_rng(seed).integers(0, V, n)


def make_data(d, arrs, split="eval", tok_sha="tokX", vocab=V):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    shards = {}
    for name, arr in arrs.items():
        a = np.asarray(arr, dtype="<u2")
        (d / name).write_bytes(a.tobytes())
        shards[name] = {"tokens": int(a.size)}
    (d / f"meta_{split}.json").write_text(json.dumps(
        {"split": split, "tokenizer_json_sha256": tok_sha, "vocab_size": vocab,
         "shards": shards}))


def ref_loss(model, arr):
    tot, n = 0.0, len(arr)
    for s in range(0, n - 1, T):
        m = min(T, n - 1 - s)
        x = torch.tensor(arr[s:s + m], dtype=torch.long)[None]
        y = torch.tensor(arr[s + 1:s + 1 + m], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        tot += F.cross_entropy(logits[0], y, reduction="sum").item()
    return tot


def test_cakupan_penuh_dan_sama_dengan_referensi():
    m, arr = make_model(), rand_arr(3 * T + 7)
    ls, nt = ev.eval_shard(m, arr, T, batch=2, stride=1)
    assert nt == len(arr) - 1
    assert ls == pytest.approx(ref_loss(m, arr), rel=1e-4)


def test_ukuran_batch_tidak_memengaruhi_hasil():
    m, arr = make_model(), rand_arr(5 * T + 3, seed=1)
    a = ev.eval_shard(m, arr, T, batch=1, stride=1)
    b = ev.eval_shard(m, arr, T, batch=3, stride=1)
    assert a[1] == b[1] and a[0] == pytest.approx(b[0], rel=1e-4)


def test_satu_jendela_sama_dengan_loss_model():
    m, arr = make_model(), rand_arr(T + 1, seed=2)
    ls, nt = ev.eval_shard(m, arr, T, batch=4, stride=1)
    x = torch.tensor(arr[:T], dtype=torch.long)[None]
    y = torch.tensor(arr[1:T + 1], dtype=torch.long)[None]
    with torch.no_grad():
        _, loss = m(x, y)
    assert nt == T and ls / nt == pytest.approx(loss.item(), rel=1e-5)


def test_stride_mengurangi_token_dan_tetap_deterministik():
    m, arr = make_model(), rand_arr(8 * T + 1, seed=3)
    full = ev.eval_shard(m, arr, T, 4, 1)
    half = ev.eval_shard(m, arr, T, 4, 2)
    assert 0 < half[1] < full[1] and half[1] == 4 * T
    assert half == ev.eval_shard(m, arr, T, 4, 2)


def test_loss_model_baru_dekat_ln_v():
    m = make_model()
    ls, nt = ev.eval_shard(m, rand_arr(2000, seed=4), T, 8, 1)
    assert abs(ls / nt - math.log(V)) < 0.3


def test_parameter_tidak_valid():
    m = make_model()
    with pytest.raises(ValueError):
        ev.eval_shard(m, rand_arr(100), T + 1, 2, 1)
    with pytest.raises(ValueError):
        ev.eval_shard(m, rand_arr(100), T, 2, 0)


def test_dataset_tolak_hash_vocab_dan_split_train(tmp_path):
    m = make_model()
    make_data(tmp_path / "ok", {"en_eval.bin": rand_arr(200)})
    with pytest.raises(ValueError):
        ev.evaluate_dataset(m, "tokBEDA", tmp_path / "ok", T, 4, 1)
    make_data(tmp_path / "v", {"en_eval.bin": rand_arr(200)}, vocab=V + 1)
    with pytest.raises(ValueError):
        ev.evaluate_dataset(m, "tokX", tmp_path / "v", T, 4, 1)
    make_data(tmp_path / "tr", {"en_train.bin": rand_arr(200)}, split="train")
    with pytest.raises(FileNotFoundError):
        ev.evaluate_dataset(m, "tokX", tmp_path / "tr", T, 4, 1)


def test_dataset_per_bahasa_dan_total(tmp_path):
    m = make_model()
    make_data(tmp_path, {"en_eval.bin": rand_arr(300, 1), "id_eval.bin": rand_arr(500, 2)})
    r = ev.evaluate_dataset(m, "tokX", tmp_path, T, 4, 1)
    assert set(r) == {"en", "id", "_all"}
    assert r["en"]["tokens"] == 299 and r["id"]["tokens"] == 499 and r["_all"]["tokens"] == 798
    mix = (r["en"]["loss"] * 299 + r["id"]["loss"] * 499) / 798
    assert r["_all"]["loss"] == pytest.approx(mix, rel=1e-9)
    assert r["id"]["perplexity"] == pytest.approx(math.exp(r["id"]["loss"]))


def _ckpt(path, tok_sha="tokX"):
    m = make_model()
    torch.save({"model": m.state_dict(), "model_cfg": asdict(cfg()), "step": 7,
                "tokenizer_sha256": tok_sha}, path)


def test_main_ujung_ke_ujung_dan_tolak_timpa(tmp_path, capsys):
    _ckpt(tmp_path / "a.pt")
    make_data(tmp_path / "d", {"en_eval.bin": rand_arr(400, 1), "id_eval.bin": rand_arr(400, 2)})
    out = tmp_path / "res" / "r.json"
    argv = ["--ckpt", f"a={tmp_path / 'a.pt'}", "--eval-set", f"s={tmp_path / 'd'}:2",
            "--block-size", str(T), "--out", str(out)]
    assert ev.main(argv) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["runs"]["a"]["step"] == 7 and r["runs"]["a"]["datasets"]["s"]["stride"] == 2
    assert r["runs"]["a"]["datasets"]["s"]["per_lang"]["_all"]["tokens"] > 0
    assert "loss=" in capsys.readouterr().out
    assert ev.main(argv) == 2


def test_main_hash_checkpoint_tidak_cocok(tmp_path, capsys):
    _ckpt(tmp_path / "a.pt", tok_sha="tokBEDA")
    make_data(tmp_path / "d", {"en_eval.bin": rand_arr(400)})
    rc = ev.main(["--ckpt", f"a={tmp_path / 'a.pt'}", "--eval-set", f"s={tmp_path / 'd'}",
                  "--block-size", str(T), "--out", str(tmp_path / "r.json")])
    assert rc == 2 and "GALAT" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()
