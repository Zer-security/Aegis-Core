"""Tes kebenaran Aegis-Nano (model.py).

Dua kategori:
- Tes kebenaran: gagal berarti cacat implementasi.
- Tes karakterisasi (awalan test_karakterisasi_): merekam perilaku yang saat ini
  ada dan merupakan keterbatasan yang diketahui, BUKAN perilaku yang diinginkan.
  Lulus berarti prediksi benar; tidak berarti perilakunya baik.

Semua tes memakai config kecil dan token acak: tanpa data, tanpa jaringan.
"""
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as aegis_model  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

V, T = 64, 16  # vocab dan block_size untuk config kecil


def small_cfg(**kw):
    base = dict(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=32, dropout=0.1)
    base.update(kw)
    return AegisConfig(**base)


def make_model(seed=0, **kw):
    torch.manual_seed(seed)
    return AegisNano(small_cfg(**kw))


def rand_idx(B=2, length=T, seed=123):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, length), generator=g)


# ---------------------------------------------------------------- kebenaran

def test_forward_bentuk_dan_tanpa_loss():
    m = make_model().eval()
    idx = rand_idx()
    logits, loss = m(idx)
    assert logits.shape == (2, T, V)
    assert loss is None
    logits, loss = m(idx, rand_idx(seed=7))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_awal_mendekati_uniform():
    m = make_model().eval()
    _, loss = m(rand_idx(), rand_idx(seed=7))
    assert abs(loss.item() - math.log(V)) < 0.3


@pytest.mark.parametrize("vocab,expected", [(50304, 30_036_864), (16384, 17_011_584)])
def test_jumlah_parameter(vocab, expected):
    # Angka expected dihitung manual dari kode sebelum tes dijalankan.
    m = AegisNano(AegisConfig(vocab_size=vocab))
    assert m.num_params() == expected


def test_input_melebihi_block_size():
    m = make_model().eval()
    with pytest.raises(AssertionError):
        m(torch.zeros(1, T + 1, dtype=torch.long))


@pytest.mark.parametrize("t", [1, 5, 15])
def test_causal_mask_tanpa_kebocoran(t):
    m = make_model().eval()
    a = rand_idx(B=2)
    b = a.clone()
    b[:, t] = (b[:, t] + 1) % V  # ubah token di posisi t
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[:, :t], lb[:, :t], atol=1e-6, rtol=0)  # masa lalu tetap
    assert (la[:, t:] - lb[:, t:]).abs().max() > 1e-4  # kontrol: posisi >= t berubah


def test_kontrol_tes_causal_mendeteksi_kebocoran(monkeypatch):
    """Mutasi: paksa attention non-kausal. Tes kausal harus bisa melihat kebocoran."""
    orig = F.scaled_dot_product_attention

    def noncausal(q, k, v, **kw):
        kw["is_causal"] = False
        return orig(q, k, v, **kw)

    monkeypatch.setattr(aegis_model.F, "scaled_dot_product_attention", noncausal)
    m = make_model().eval()
    t = 8
    a = rand_idx(B=2)
    b = a.clone()
    b[:, t] = (b[:, t] + 1) % V
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert (la[:, :t] - lb[:, :t]).abs().max() > 1e-4


def test_loss_sama_dengan_cross_entropy_manual():
    m = make_model().eval()
    idx, tg = rand_idx(), rand_idx(seed=7)
    logits, loss = m(idx, tg)
    ref = F.cross_entropy(logits.reshape(-1, V), tg.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-6)


def test_ignore_index_diabaikan():
    m = make_model().eval()
    idx = rand_idx()
    tg = rand_idx(seed=7).clone()
    tg[:, 10:] = -1
    tg[0, 3] = -1
    logits, loss = m(idx, tg)
    valid = tg != -1
    assert valid.sum() > 0
    logp = F.log_softmax(logits, dim=-1)
    picked = logp.gather(-1, tg.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ref = -(picked[valid]).mean()
    assert torch.allclose(loss, ref, atol=1e-6)


def test_kontrak_pergeseran_target():
    """Model TIDAK menggeser target; pemanggil harus melakukannya."""
    m = make_model().eval()
    idx = rand_idx()
    logits, loss_same = m(idx, idx)
    ref_same = F.cross_entropy(logits.reshape(-1, V), idx.reshape(-1))
    assert torch.allclose(loss_same, ref_same, atol=1e-6)  # tanpa pergeseran internal

    x = idx[:, :-1].contiguous()
    y = idx[:, 1:].contiguous()
    _, loss_shift = m(x, y)
    full_logits, _ = m(idx)
    ref_shift = F.cross_entropy(full_logits[:, :-1].reshape(-1, V), y.reshape(-1))
    assert torch.allclose(loss_shift, ref_shift, atol=1e-5)


def test_right_padding_tidak_mengubah_loss():
    m = make_model().eval()
    n = 10
    seq = rand_idx(B=1, length=n + 1, seed=5)
    x_short = seq[:, :n].contiguous()
    y_short = seq[:, 1:n + 1].contiguous()
    _, loss_short = m(x_short, y_short)

    x_pad = torch.zeros(1, T, dtype=torch.long)
    x_pad[:, :n] = x_short
    y_pad = torch.full((1, T), -1, dtype=torch.long)
    y_pad[:, :n] = y_short
    _, loss_pad = m(x_pad, y_pad)
    assert torch.allclose(loss_short, loss_pad, atol=1e-5)


def test_weight_tying():
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    _, loss = m(rand_idx(), rand_idx(seed=7))
    loss.backward()
    assert m.tok_emb.weight.grad is not None
    assert m.tok_emb.weight.grad.abs().sum() > 0


def test_determinisme_seed():
    a, b, c = make_model(seed=1), make_model(seed=1), make_model(seed=2)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)  # kontrol: seed beda -> beda
    a.eval()
    b.eval()
    idx = rand_idx()
    la, _ = a(idx)
    lb, _ = b(idx)
    la2, _ = a(idx)
    assert torch.equal(la, lb)
    assert torch.equal(la, la2)  # mode eval: dropout mati, hasil berulang


def test_skala_inisialisasi():
    m = make_model()
    sd = m.state_dict()
    n_layer = m.cfg.n_layer
    expected_proj = 0.02 / (2 * n_layer) ** 0.5
    for name in ("blocks.0.attn.proj.weight", "blocks.1.mlp.proj.weight"):
        assert sd[name].std().item() == pytest.approx(expected_proj, rel=0.15)
    assert sd["blocks.0.attn.qkv.weight"].std().item() == pytest.approx(0.02, rel=0.15)
    assert sd["tok_emb.weight"].std().item() == pytest.approx(0.02, rel=0.15)


def test_checkpoint_roundtrip(tmp_path):
    a = make_model(seed=1).eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": a.state_dict(), "cfg": asdict(a.cfg)}, path)
    ckpt = torch.load(path, weights_only=True)

    torch.manual_seed(99)
    b = AegisNano(AegisConfig(**ckpt["cfg"])).eval()
    idx = rand_idx()
    with torch.no_grad():
        la, _ = a(idx)
        lb_before, _ = b(idx)
    assert not torch.equal(la, lb_before)  # kontrol: sebelum load, hasil beda
    b.load_state_dict(ckpt["model"])  # strict
    with torch.no_grad():
        lb_after, _ = b(idx)
    assert torch.equal(la, lb_after)
    assert b.lm_head.weight is b.tok_emb.weight  # tying bertahan setelah load


def test_overfit_satu_batch():
    m = make_model(seed=0, dropout=0.0)
    m.train()
    data = rand_idx(B=4, length=T + 1, seed=11)
    x = data[:, :-1].contiguous()
    y = data[:, 1:].contiguous()  # target digeser dengan benar
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3, weight_decay=0.0)
    _, loss0 = m(x, y)
    assert abs(loss0.item() - math.log(V)) < 0.3
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        _, loss = m(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        _, final = m(x, y)
    assert final.item() < 0.25 * math.log(V)  # ambang ditetapkan sebelum dijalankan


def test_generate_bentuk_dan_rentang():
    m = make_model().eval()
    prompt = rand_idx(B=2, length=3)
    out = m.generate(prompt, max_new_tokens=5)
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], prompt)
    assert out.min() >= 0 and out.max() < V


def test_generate_top_k_1_sama_dengan_greedy():
    m = make_model().eval()
    prompt = rand_idx(B=2, length=3)
    out1 = m.generate(prompt, 6, top_k=1)
    out2 = m.generate(prompt, 6, top_k=1)
    assert torch.equal(out1, out2)
    cur = prompt.clone()
    with torch.no_grad():
        for _ in range(6):
            logits, _ = m(cur[:, -T:])
            cur = torch.cat([cur, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out1, cur)


def test_generate_melebihi_block_size():
    m = make_model().eval()
    out = m.generate(rand_idx(B=1, length=3), max_new_tokens=20)  # 23 > T=16
    assert out.shape == (1, 23)


def test_generate_reproducible_dengan_seed():
    m = make_model().eval()
    prompt = rand_idx(B=2, length=3)
    torch.manual_seed(42)
    a = m.generate(prompt, 8)
    torch.manual_seed(42)
    b = m.generate(prompt, 8)
    assert torch.equal(a, b)


# ------------------------------------------------------------- karakterisasi

def test_karakterisasi_targets_non_contiguous():
    """Temuan #2: targets.view(-1) gagal untuk irisan data[:, 1:] (B>1)."""
    m = make_model().eval()
    data = rand_idx(B=2, length=T + 1)
    x = data[:, :-1].contiguous()
    y = data[:, 1:]
    assert not y.is_contiguous()
    with pytest.raises(RuntimeError):
        m(x, y)
    _, loss = m(x, y.contiguous())  # jalan keluar di sisi pemanggil
    assert torch.isfinite(loss)


def test_karakterisasi_semua_target_ignore_nan():
    """Temuan #3: batch yang seluruh targetnya -1 menghasilkan loss NaN."""
    m = make_model().eval()
    tg = torch.full((2, T), -1, dtype=torch.long)
    _, loss = m(rand_idx(), tg)
    assert torch.isnan(loss)


def test_karakterisasi_generate_tidak_memanggil_eval():
    """Temuan #5: generate() tidak mengubah mode; dropout tetap aktif bila train()."""
    m = make_model()
    m.train()
    m.generate(rand_idx(B=1, length=3), max_new_tokens=2)
    assert m.training is True
