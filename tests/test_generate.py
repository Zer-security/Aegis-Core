"""Tes generate.py. Checkpoint dan tokenizer sintetis; tanpa data nyata, tanpa jaringan."""
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "scripts", HERE.parent / "tokenizer"):
    sys.path.insert(0, str(p))
import generate as gen  # noqa: E402
import train_tokenizer as tt  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402
from test_tokenizer import make_corpus  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("gen")
    make_corpus(d / "en_train.jsonl", n=200)
    meta = tt.train([d / "en_train.jsonl"], 300, d / "tok")
    cfg = AegisConfig(vocab_size=300, block_size=32, n_layer=1, n_head=2, n_embd=32,
                      dropout=0.1)
    torch.manual_seed(0)
    m = AegisNano(cfg)
    ck = d / "ckpt.pt"
    torch.save({"model": m.state_dict(), "model_cfg": asdict(cfg), "step": 5,
                "tokenizer_sha256": meta["tokenizer_json_sha256"]}, ck)
    return SimpleNamespace(d=d, ck=ck, tok=d / "tok", meta=meta)


def args(model, tok, eos, prompt="saya makan", **kw):
    p = dict(max_new_tokens=20, temperature=1.0, top_k=None, seed=1)
    p.update(kw)
    return gen.sample(model, tok, eos, prompt, **p)


def test_load_model_mode_eval(env):
    model, tok, eos, step = gen.load_model(env.ck, env.tok)
    assert model.training is False and step == 5 and eos == 0


def test_sample_deterministik_terhadap_seed(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    a, b, c = args(model, tok, eos), args(model, tok, eos), args(model, tok, eos, seed=2)
    assert a == b
    assert a[0] != c[0]


def test_top_k_1_tidak_bergantung_seed(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    assert args(model, tok, eos, top_k=1, seed=1) == args(model, tok, eos, top_k=1, seed=2)


def test_sample_memaksa_eval(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    model.train()
    args(model, tok, eos, max_new_tokens=2)
    assert model.training is False


def test_hash_tokenizer_tidak_cocok(env):
    c = torch.load(env.ck, weights_only=True)
    c["tokenizer_sha256"] = "0" * 64
    bad = env.d / "bad.pt"
    torch.save(c, bad)
    with pytest.raises(ValueError):
        gen.load_model(bad, env.tok)


def test_prompt_token_khusus_ditolak(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    for p in ("a <|endoftext|> b", "x <|pad|>"):
        with pytest.raises(ValueError):
            args(model, tok, eos, prompt=p)


def test_prompt_terlalu_panjang(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    with pytest.raises(ValueError):
        args(model, tok, eos, prompt="saya makan " * 40)


def test_prompt_kosong_memakai_eos(env):
    model, tok, eos, _ = gen.load_model(env.ck, env.tok)
    text, n, _ = args(model, tok, eos, prompt="", max_new_tokens=5)
    assert isinstance(text, str) and n <= 5


def test_potong_di_eos(env):
    _, tok, eos, _ = gen.load_model(env.ck, env.tok)

    class Fake:
        cfg = SimpleNamespace(block_size=32)

        def eval(self):
            pass

        def generate(self, x, n, temperature=1.0, top_k=None):
            return torch.cat([x, torch.tensor([[5, 6, 0, 7]])], dim=1)

    text, n, stopped = gen.sample(Fake(), tok, eos, "saya", 4, 1.0, None, 0)
    assert stopped and n == 2 and text == tok.decode([5, 6], skip_special_tokens=False)


def test_main_ok_dan_galat(env, capsys):
    rc = gen.main(["--ckpt", str(env.ck), "--tokenizer-dir", str(env.tok),
                   "--prompt", "saya makan", "--max-new-tokens", "5"])
    cap = capsys.readouterr()
    assert rc == 0 and "mode_train=False" in cap.out and "[0]" in cap.out
    rc = gen.main(["--ckpt", str(env.d / "tidak_ada.pt"), "--tokenizer-dir", str(env.tok)])
    assert rc == 2 and "GALAT" in capsys.readouterr().err
