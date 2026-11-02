"""Tes sample_eval. Parquet sintetis di memori; tanpa jaringan."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import sample_corpus as sc  # noqa: E402
import sample_eval as se  # noqa: E402
from test_sample_corpus import MARKER, make_parquet  # noqa: E402

USED = "sample/10BT/000.parquet"
EN_B, EN_C = "sample/10BT/001.parquet", "sample/10BT/002.parquet"
ID_T = "data/ind_Latn/test/000.parquet"
TARGET = ["--target-mb", "0.02"]


class Src:
    def __init__(self, blobs, listing):
        self.blobs, self.listing, self.reads = blobs, listing, []

    def list_files(self, subdir):
        return sorted((p, len(self.blobs[p])) for p in self.listing.get(subdir, []))

    def read_range(self, path, start, end):
        self.reads.append(path)
        return self.blobs[path][start:end]


def boom(lang):
    raise AssertionError("jaringan/sumber disentuh sebelum gerbang")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    sample = data / "sample"
    sample.mkdir(parents=True)
    monkeypatch.setattr(sc, "DATA_DIR", data)
    blobs = {USED: make_parquet(5), EN_B: make_parquet(6), EN_C: make_parquet(8), ID_T: make_parquet(7)}
    dup = pq.read_table(io.BytesIO(blobs[EN_B])).column("text")[3].as_py()
    rows = {"en_train.jsonl": [{"id": "t0", "text": "train doc zero"}, {"id": "t1", "text": dup}],
            "id_train.jsonl": [{"id": "t2", "text": "nothing here"}]}
    for name, rs in rows.items():
        (sample / name).write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
    man = {"profile": "full",
           "outputs": {n: {"sha256": sha(sample / n), "bytes": (sample / n).stat().st_size} for n in rows},
           "languages": {l: {"revision": sc.SOURCES[l]["revision"],
                             "files_chosen": [{"path": USED if l == "en" else "data/ind_Latn/train/x.parquet"}]}
                         for l in ("en", "id")}}
    (sample / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    src = Src(blobs, {"sample/10BT": [USED, EN_B, EN_C], "data/ind_Latn/test": [ID_T]})
    return SimpleNamespace(src=src, factory=lambda lang: src, data=data, sample=sample,
                           out=data / "eval_indep")


def run(env, *extra):
    return se.main(["--run", *TARGET, *extra], source_factory=env.factory)


def test_plan_only_tanpa_keluaran_dan_tanpa_berkas_terpakai(env):
    assert se.main(TARGET, source_factory=env.factory) == 0
    assert not env.out.exists()
    assert USED not in env.src.reads


def test_run_dasar_dedup_dan_tanpa_teks(env, capsys):
    assert run(env) == 0
    cap = capsys.readouterr()
    assert sorted(p.name for p in env.out.iterdir()) == ["MANIFEST.json", "en_eval.jsonl", "id_eval.jsonl"]
    m = json.loads((env.out / "MANIFEST.json").read_text(encoding="utf-8"))
    for n, info in m["outputs"].items():
        assert sha(env.out / n) == info["sha256"]
    en = m["languages"]["en"]["stats"]
    assert en["skipped_train_duplicate"] == 1
    assert en["eval_docs"] + en["skipped_train_duplicate"] == 200
    assert USED not in env.src.reads
    txt = cap.out + cap.err + (env.out / "MANIFEST.json").read_text(encoding="utf-8")
    assert MARKER not in txt and not any(f"id-{s}-" in txt for s in (5, 6, 7, 8))
    assert m["train_manifest_sha256"] == sha(env.sample / "MANIFEST.json")


def test_gerbang_manifest_hilang_dan_train_berubah(env):
    (env.sample / "en_train.jsonl").write_text("{}\n", encoding="utf-8")
    assert se.main(["--run", *TARGET], source_factory=boom) == 2
    (env.sample / "MANIFEST.json").unlink()
    assert se.main(["--run", *TARGET], source_factory=boom) == 2


def test_keluaran_ada_dan_sisa_dicek_sebelum_jaringan(env):
    assert run(env) == 0
    assert se.main(["--run", *TARGET], source_factory=boom) == 2
    for p in env.out.iterdir():
        p.unlink()
    (env.out / "x.partial").write_text("sisa", encoding="utf-8")
    assert se.main(["--run", *TARGET], source_factory=boom) == 2
    assert (env.out / "x.partial").read_text(encoding="utf-8") == "sisa"


def test_interupsi_membersihkan(env, monkeypatch):
    def fake(plan, source, meter, digests, out_path, target):
        Path(out_path).write_text("setengah", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(se, "run_language", fake)
    with pytest.raises(KeyboardInterrupt):
        run(env)
    assert list(env.out.iterdir()) == []


def test_promosi_gagal_rollback(env, monkeypatch):
    orig = Path.replace

    def failing(self, target):
        if self.name == "MANIFEST.json.partial":
            raise OSError("gagal uji")
        return orig(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    assert run(env) == 3
    assert list(env.out.iterdir()) == []


def test_target_terlalu_besar_tidak_layak_tanpa_keluaran(env):
    assert se.main(["--run", "--target-mb", "5"], source_factory=env.factory) == 2
    assert not env.out.exists()


def test_batas_trafik_hanya_turun(env):
    assert se.main(["--max-traffic-mb", "999"], source_factory=boom) == 2
    assert se.main(["--run", *TARGET, "--max-traffic-mb", "0.001"], source_factory=env.factory) == 2
    assert env.src.reads == [] and not env.out.exists()
