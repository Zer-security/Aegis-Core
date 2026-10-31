"""Tes sampler (scripts/sample_corpus.py v0.1.2). Parquet sintetis di memori; tanpa jaringan."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sample_corpus as sc  # noqa: E402

MARKER = "RAHASIA_UJI_9f3a"


def make_parquet(seed, n=400, n_hold=40):
    hold, train, i = [], [], 0
    while len(hold) < n_hold or len(train) < n - n_hold:
        d = f"id-{seed}-{i}"
        i += 1
        (hold if sc.is_holdout(d) else train).append(d)
    hi, ti = iter(hold), iter(train)
    ids = [next(hi) if k % 10 == 9 else next(ti) for k in range(n)]
    texts = [f"{MARKER} seed {seed} dokumen {k} " + "lorem ipsum dolor sit amet " * 8
             for k in range(n)]
    buf = io.BytesIO()
    pq.write_table(pa.table({"id": ids, "text": texts, "url": [f"u{k}" for k in range(n)]}),
                   buf, row_group_size=50, write_page_index=False)
    return buf.getvalue()


class FakeSource:
    def __init__(self, blob):
        self.blob, self.calls = blob, 0

    def list_files(self, subdir):
        return [] if subdir.endswith("test") else [(f"{subdir}/part-0.parquet", len(self.blob))]

    def read_range(self, path, start, end):
        self.calls += 1
        return self.blob[start:end]


def boom(lang):
    raise AssertionError("jaringan/sumber disentuh sebelum gerbang")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "DATA_DIR", tmp_path / "data")
    srcs = {"en": FakeSource(make_parquet(1)), "id": FakeSource(make_parquet(2))}
    return SimpleNamespace(srcs=srcs, factory=lambda lang: srcs[lang],
                           trial=tmp_path / "data" / "sample_trial",
                           full=tmp_path / "data" / "sample")


def trial(env, *extra):
    return sc.main(["--profile", "trial", "--run", *extra], source_factory=env.factory)


def tree(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).iterdir())}


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]


def test_trial_run_dasar(env, capsys):
    assert trial(env) == 0
    cap = capsys.readouterr()
    assert sorted(tree(env.trial)) == ["MANIFEST.json", "en_eval.jsonl", "en_train.jsonl",
                                       "id_eval.jsonl", "id_train.jsonl"]
    for lang in ("en", "id"):
        tr, ev = rows(env.trial / f"{lang}_train.jsonl"), rows(env.trial / f"{lang}_eval.jsonl")
        assert len(tr) == 100 and len(ev) > 0
        assert not any(sc.is_holdout(r["id"]) for r in tr)
        assert all(sc.is_holdout(r["id"]) for r in ev)
    out = cap.out + cap.err + (env.trial / "MANIFEST.json").read_text(encoding="utf-8")
    assert MARKER not in out and "id-1-" not in out and "id-2-" not in out


def test_gerbang_full_tanpa_manifest_sebelum_jaringan(env):
    assert sc.main(["--profile", "full", "--run"], source_factory=boom) == 2


def _m_profile(m): m["profile"] = "full"
def _m_rev(m): m["languages"]["en"]["revision"] = "x"
def _m_lang(m): del m["languages"]["id"]
def _m_stats(m): m["languages"]["en"]["stats"]["train_docs"] = 0
def _m_out(m): m["outputs"]["en_train.jsonl"]["sha256"] = "0" * 64


@pytest.mark.parametrize("mut", [_m_profile, _m_rev, _m_lang, _m_stats, _m_out])
def test_manifest_trial_tidak_valid_ditolak(env, mut):
    assert trial(env) == 0
    mp = env.trial / "MANIFEST.json"
    assert len(sc.check_trial_manifest(mp)) == 64
    m = json.loads(mp.read_text(encoding="utf-8"))
    mut(m)
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(sc.PlanError):
        sc.check_trial_manifest(mp)
    assert sc.main(["--profile", "full", "--run"], source_factory=boom) == 2


def test_isi_output_trial_diubah_ditolak(env):
    assert trial(env) == 0
    with open(env.trial / "en_train.jsonl", "a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(sc.PlanError):
        sc.check_trial_manifest(env.trial / "MANIFEST.json")


def test_full_manifest_valid_lolos_gerbang_lalu_berhenti_tidak_layak(env):
    assert trial(env) == 0
    before = env.srcs["en"].calls
    assert sc.main(["--profile", "full", "--run"], source_factory=env.factory) == 2
    assert env.srcs["en"].calls > before  # gerbang lolos, rencana berjalan
    assert not env.full.exists()  # tidak layak: tidak ada keluaran


def test_overwrite_dan_sisa_dicek_sebelum_jaringan(env):
    assert trial(env) == 0
    assert sc.main(["--profile", "trial", "--run"], source_factory=boom) == 2
    (env.trial / "x.partial").write_text("sisa", encoding="utf-8")
    assert sc.main(["--profile", "trial", "--run", "--overwrite"], source_factory=boom) == 2
    assert (env.trial / "x.partial").read_text(encoding="utf-8") == "sisa"  # tidak dihapus


def test_interupsi_membersihkan_partial(env, monkeypatch):
    def fake_run(plan, prof, source, meter, train_path, eval_path):
        Path(train_path).write_text("setengah", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(sc, "run_language", fake_run)
    with pytest.raises(KeyboardInterrupt):
        trial(env)
    assert list(env.trial.iterdir()) == []


def test_promosi_gagal_mengembalikan_berkas_lama(env, monkeypatch):
    assert trial(env) == 0
    for p in env.trial.iterdir():
        p.write_text("LAMA " + p.name, encoding="utf-8")
    before = tree(env.trial)
    orig = Path.replace

    def failing(self, target):
        if self.name.endswith(".partial") and Path(target).name == "id_train.jsonl":
            raise OSError("gagal uji")
        return orig(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    assert trial(env, "--overwrite") == 3
    assert tree(env.trial) == before  # tanpa .bak/.partial, isi lama utuh


def test_batas_trafik(env):
    assert sc.main(["--max-traffic-mb", "999"], source_factory=boom) == 2  # tidak bisa dinaikkan
    assert sc.main(["--run", "--max-traffic-mb", "0.001"], source_factory=env.factory) == 2
    assert env.srcs["en"].calls == 0 and env.srcs["id"].calls == 0
    assert not env.trial.exists()


def test_reservasi_sebelum_permintaan():
    s = FakeSource(b"x" * 200)
    m = sc.Meter(100)
    with pytest.raises(sc.BudgetExceeded):
        sc.fetch_range(s, "p", 0, 101, m)
    assert s.calls == 0 and m.used == 0
