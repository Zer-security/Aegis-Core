"""Tes percobaan ulang Range (sample_corpus v0.1.3). Sumber palsu; tanpa jaringan."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import sample_corpus as sc  # noqa: E402
from test_sample_corpus import FakeSource, make_parquet  # noqa: E402

BLOB = bytes(range(100))


class Scripted:
    """cuts: per panggilan, jumlah byte yang tiba sebelum terpotong (None = utuh)."""

    def __init__(self, blob, cuts):
        self.blob, self.cuts, self.calls = blob, list(cuts), []

    def read_range(self, path, start, end):
        self.calls.append((start, end))
        cut = self.cuts.pop(0) if self.cuts else None
        data = self.blob[start:end]
        if cut is not None and cut < len(data):
            raise sc.ShortRead(data[:cut], end - start)
        return data


def test_pulih_dengan_range_sisa():
    s, m = Scripted(BLOB, [40]), sc.Meter(1000)
    assert sc.fetch_range(s, "p", 0, 100, m) == BLOB
    assert s.calls == [(0, 100), (40, 100)]
    assert m.used == 100


def test_pulih_setelah_dua_pemotongan_termasuk_nol_byte():
    s, m = Scripted(BLOB, [0, 30]), sc.Meter(1000)
    assert sc.fetch_range(s, "p", 10, 100, m) == BLOB[10:100]
    assert s.calls == [(10, 100), (10, 100), (40, 100)]
    assert m.used == 90


def test_batas_percobaan_dan_hitungan_byte():
    s, m = Scripted(BLOB, [10, 10, 10, 10, 10]), sc.Meter(1000)
    with pytest.raises(RuntimeError) as e:
        sc.fetch_range(s, "data/x.parquet", 0, 100, m)
    assert "incomplete" in str(e.value) and "data/x.parquet" in str(e.value)
    assert len(s.calls) == 1 + sc.MAX_RETRIES
    assert m.used == 10 * (1 + sc.MAX_RETRIES)  # byte yang tiba tetap dihitung


def test_panjang_salah_tanpa_shortread_tidak_diulang():
    class Short:
        calls = 0

        def read_range(self, path, start, end):
            self.calls += 1
            return BLOB[start:end - 5]

    s = Short()
    with pytest.raises(RuntimeError):
        sc.fetch_range(s, "p", 0, 100, sc.Meter(1000))
    assert s.calls == 1


def test_anggaran_dicadangkan_sebelum_setiap_permintaan():
    s = Scripted(BLOB, [])
    with pytest.raises(sc.BudgetExceeded):
        sc.fetch_range(s, "p", 0, 100, sc.Meter(50))
    assert s.calls == []


class Resp:
    status = 206

    def __init__(self, chunks, headers):
        self.chunks, self.headers = list(chunks), headers

    def read(self, amt=-1):
        if not self.chunks:
            return b""
        c = self.chunks.pop(0)
        if isinstance(c, Exception):
            raise c
        return c

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


HDR = {"Content-Range": "bytes 0-99/1000"}


def patch_open(monkeypatch, resp):
    monkeypatch.setattr(sc.urllib.request, "urlopen", lambda req, timeout=None: resp)


def test_http_body_terpotong_menjadi_shortread(monkeypatch):
    patch_open(monkeypatch, Resp([b"x" * 60], HDR))
    with pytest.raises(sc.ShortRead) as e:
        sc.HFSource("r", "rev").read_range("p", 0, 100)
    assert e.value.data == b"x" * 60


def test_http_koneksi_putus_di_tengah_menyimpan_byte_parsial(monkeypatch):
    patch_open(monkeypatch, Resp([b"y" * 40, ConnectionResetError("putus")], HDR))
    with pytest.raises(sc.ShortRead) as e:
        sc.HFSource("r", "rev").read_range("p", 0, 100)
    assert e.value.data == b"y" * 40 and "ConnectionResetError" in str(e.value)


def test_http_body_utuh(monkeypatch):
    patch_open(monkeypatch, Resp([b"z" * 100], HDR))
    assert sc.HFSource("r", "rev").read_range("p", 0, 100) == b"z" * 100


def test_http_status_bukan_206_tetap_ditolak(monkeypatch):
    r = Resp([b"z" * 100], HDR)
    r.status = 200
    patch_open(monkeypatch, r)
    with pytest.raises(RuntimeError):
        sc.HFSource("r", "rev").read_range("p", 0, 100)
    assert r.chunks  # body tidak dibaca


class TruncOnce(FakeSource):
    def __init__(self, blob):
        super().__init__(blob)
        self.done = False

    def read_range(self, path, start, end):
        data = super().read_range(path, start, end)
        if not self.done and end - start > 100:
            self.done = True
            raise sc.ShortRead(data[: len(data) // 2], end - start)
        return data


def test_run_trial_dengan_pembacaan_terpotong(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "DATA_DIR", tmp_path / "data")
    srcs = {"en": TruncOnce(make_parquet(1)), "id": TruncOnce(make_parquet(2))}
    rc = sc.main(["--profile", "trial", "--run"], source_factory=lambda lang: srcs[lang])
    assert rc == 0 and all(s.done for s in srcs.values())
    out = tmp_path / "data" / "sample_trial"
    for lang in ("en", "id"):
        with open(out / f"{lang}_train.jsonl", encoding="utf-8") as f:
            assert sum(1 for _ in f) == 100
