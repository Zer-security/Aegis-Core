"""Tes training loop Aegis-Core. Data sintetis; tanpa data nyata, tanpa jaringan."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "scripts"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))
import data_loader as dl  # noqa: E402
import train as tr  # noqa: E402

V = 64
PERIODIC = np.tile(np.arange(1, 17), 400)


def make_data(d, tokens, split="train", tok_sha="tokX", vocab=V):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype="<u2")
    name = f"en_{split}.bin"
    (d / name).write_bytes(arr.tobytes())
    (d / f"meta_{split}.json").write_text(json.dumps({
        "split": split, "tokenizer_json_sha256": tok_sha, "vocab_size": vocab,
        "shards": {name: {"tokens": int(arr.size)}}}))


def targs(tmp_path, *extra):
    return tr.parse_args(["--data-dir", str(tmp_path / "data"), "--out-dir", str(tmp_path / "run"),
                          "--block-size", "16", "--batch-size", "8", "--n-layer", "2",
                          "--n-head", "2", "--n-embd", "32", "--log-interval", "1",
                          "--ckpt-interval", "1000", *extra])


@pytest.fixture
def data(tmp_path):
    make_data(tmp_path / "data", PERIODIC)
    return tmp_path


def test_jadwal_lr():
    f = lambda s: tr.lr_at(s, 1e-3, 1e-4, 10, 100)  # noqa: E731
    assert f(0) == pytest.approx(1e-4)
    assert f(9) == pytest.approx(1e-3) and f(10) == pytest.approx(1e-3)
    assert f(100) == pytest.approx(1e-4) and f(500) == pytest.approx(1e-4)
    vals = [f(s) for s in range(10, 101)]
    assert all(a >= b - 1e-15 for a, b in zip(vals, vals[1:]))


def test_loss_turun_pada_data_terpelajari(data):
    r = tr.run(targs(data, "--max-steps", "200", "--warmup", "10"))
    ls = r["losses"]
    first, last = sum(ls[:10]) / 10, sum(ls[-10:]) / 10
    assert abs(ls[0] - math.log(V)) < 0.5
    assert last < 0.5 * first  # ambang ditetapkan sebelum dijalankan


def test_resume_identik_dengan_run_tanpa_henti(data):
    a = targs(data, "--out-dir", str(data / "run_a"), "--max-steps", "30")
    ra = tr.run(a)
    b1 = targs(data, "--out-dir", str(data / "run_b"), "--max-steps", "30", "--stop-after", "12")
    rb1 = tr.run(b1)
    assert rb1["end_step"] == 12
    b2 = targs(data, "--out-dir", str(data / "run_b"), "--max-steps", "30", "--resume")
    rb2 = tr.run(b2)
    assert rb2["start_step"] == 12
    assert rb1["losses"] + rb2["losses"] == ra["losses"]
    ca = torch.load(data / "run_a" / "ckpt.pt", weights_only=True)
    cb = torch.load(data / "run_b" / "ckpt.pt", weights_only=True)
    assert ca["step"] == cb["step"] == 30
    assert all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])


def test_resume_dan_run_baru_menolak_kondisi_salah(data):
    tr.run(targs(data, "--max-steps", "10", "--stop-after", "5"))
    with pytest.raises(FileExistsError):
        tr.run(targs(data, "--max-steps", "10"))
    with pytest.raises(ValueError):
        tr.run(targs(data, "--max-steps", "10", "--resume", "--seed", "7"))
    with pytest.raises(ValueError):
        tr.run(targs(data, "--max-steps", "10", "--resume", "--n-embd", "64"))
    with pytest.raises(FileNotFoundError):
        tr.run(targs(data, "--out-dir", str(data / "kosong"), "--max-steps", "10", "--resume"))
    make_data(data / "data", PERIODIC, tok_sha="tokBEDA")
    with pytest.raises(ValueError):
        tr.run(targs(data, "--max-steps", "10", "--resume"))


def test_berhenti_bila_loss_nan(data, monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "cross_entropy",
                        lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError):
        tr.run(targs(data, "--max-steps", "5"))


def test_eval_tidak_tersedia_tanpa_meta_eval(data):
    r = tr.run(targs(data, "--max-steps", "3"))
    assert r["eval_loss"] is None


def test_eval_terpisah_dari_train(data, monkeypatch):
    make_data(data / "data", np.tile(np.arange(40, 56), 100), split="eval")
    calls, orig = [], dl.get_batch

    def spy(data_dir, split, *a, **k):
        x, y = orig(data_dir, split, *a, **k)
        calls.append(split)
        if split == "train":
            assert int(x.max()) < 40 and int(y.max()) < 40  # tidak ada token eval di batch train
        else:
            assert int(x.min()) >= 40
        return x, y

    monkeypatch.setattr(dl, "get_batch", spy)
    r = tr.run(targs(data, "--max-steps", "5", "--eval-batches", "3"))
    assert calls.count("train") == 5 and calls.count("eval") == 3
    assert r["eval_loss"] is not None and math.isfinite(r["eval_loss"])


def test_eval_shard_terlalu_pendek_dilaporkan_tidak_tersedia(data):
    make_data(data / "data", np.arange(10), split="eval")
    r = tr.run(targs(data, "--max-steps", "3"))
    assert r["eval_loss"] is None


def test_checkpoint_atomik_dan_log_tanpa_teks(data):
    tr.run(targs(data, "--max-steps", "6", "--ckpt-interval", "2"))
    run = data / "run"
    assert not list(run.glob("*.tmp"))
    c = torch.load(run / "ckpt.pt", weights_only=True)
    assert c["step"] == 6 and c["tokenizer_sha256"] == "tokX"
    assert {"model", "optimizer", "torch_rng", "model_cfg", "sched", "data_meta_sha256"} <= set(c)
    rows = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(set(r) <= {"step", "loss", "lr", "grad_norm", "eval_loss"} for r in rows)
