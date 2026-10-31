"""Data loader Aegis-Core (v0.1.0, prototipe).

Daftar shard HANYA dari meta_{split}.json; split selain train/eval ditolak.
x dan y selalu int64, kontigu, dengan y[t] = x[t+1].
"""
import json
from pathlib import Path

import numpy as np
import torch

__version__ = "0.1.0"
SPLITS = ("train", "eval")


def load_split(data_dir, split, expected_tokenizer_sha256=None):
    if split not in SPLITS:
        raise ValueError(f"split tidak valid: {split!r}")
    d = Path(data_dir)
    mp = d / f"meta_{split}.json"
    if not mp.is_file():
        raise FileNotFoundError(f"tidak ada: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("split") != split:
        raise ValueError(f"meta bukan split {split}")
    if expected_tokenizer_sha256 is not None and \
            meta.get("tokenizer_json_sha256") != expected_tokenizer_sha256:
        raise ValueError("hash tokenizer di meta tidak cocok dengan yang diharapkan")
    shards = []
    for name, info in sorted(meta["shards"].items()):
        if name != Path(name).name or not name.endswith(f"_{split}.bin"):
            raise ValueError(f"nama shard tidak valid untuk split {split}: {name}")
        p = d / name
        if p.stat().st_size != int(info["tokens"]) * 2:
            raise ValueError(f"ukuran shard tidak cocok dengan meta: {name}")
        if info["tokens"] > 0:
            shards.append(np.memmap(p, dtype="<u2", mode="r"))
    return shards, meta


def get_batch(data_dir, split, B, T, seed, step=0, expected_tokenizer_sha256=None):
    shards, _ = load_split(data_dir, split, expected_tokenizer_sha256)
    usable = [s for s in shards if len(s) > T]
    if not usable:
        raise ValueError(f"tidak ada shard dengan panjang > {T} pada split {split}")
    w = np.array([len(s) - T for s in usable], dtype=np.float64)
    rng = np.random.default_rng([int(seed), int(step)])
    which = rng.choice(len(usable), size=B, p=w / w.sum())
    xs, ys = [], []
    for k in which:
        s = usable[int(k)]
        i = int(rng.integers(0, len(s) - T))
        xs.append(np.asarray(s[i:i + T], dtype=np.int64))
        ys.append(np.asarray(s[i + 1:i + T + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))
    assert x.is_contiguous() and y.is_contiguous()
    return x, y
