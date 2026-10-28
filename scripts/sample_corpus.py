#!/usr/bin/env python3
"""Aegis-Core: build the tokenizer training sample and a held-out eval set.

PLAN FIRST. Without --run this script only reads parquet footers (metadata)
and prints a plan. No document text is read in plan mode.

Sources are pinned in SOURCES (see DATA.md). Both are ODC-By 1.0 datasets
derived from Common Crawl and are also subject to Common Crawl's Terms of Use.

Profiles
--------
trial : about 100 training documents per language, traffic cap 150 MB per language.
full  : 100 MB of training text per language, traffic cap 400 MB per language.
        A full run is refused until a trial run has produced its manifest.
(1 MB = 1,000,000 bytes everywhere in this file.)

Traffic accounting
------------------
Every download goes through fetch_range(), which reserves the bytes against a
per-language budget BEFORE the request is sent. The counter measures response
bytes requested by this program. It is an estimate, not a guarantee about
network traffic: HTTP/TLS headers, DNS, redirects, proxies and any retries
made by the operating system are not counted. The effective budget is
therefore SAFETY_MARGIN (90%) of the agreed cap. Cross-check with
/proc/net/dev before and after a run. Caps are never raised automatically.

How the traffic is kept small
-----------------------------
Parquet files are several GB each, but are stored in row groups and columns.
The script reads each chosen file's footer, then downloads only the byte
ranges of the "text" and "id" column chunks of selected row groups, using HTTP
Range requests. Other columns are never downloaded. The smallest unit that can
be downloaded is one column chunk of one row group, so the plan prints the
worst-case traffic before anything else is fetched. If a single row group does
not fit in the budget, the script stops and reports it.

Holdout rule
------------
A document goes to the evaluation set when
    int.from_bytes(sha256(utf8(id)).digest()[:8], "big") % 100 == 0
The decision depends only on the document id, so it is reproducible, and an
evaluation document never appears in the training file.
Evaluation documents are collected from the same row groups and are not counted
toward the training target.

Output (data/ is git-ignored, nothing here is ever committed)
-------------------------------------------------------------
data/sample_trial/ (trial) or data/sample/ (full):
    en_train.jsonl, en_eval.jsonl, id_train.jsonl, id_eval.jsonl, MANIFEST.json
MANIFEST.json holds statistics, pinned revisions, files and row groups used,
and checksums. It contains no document text and no document ids.
This script never prints document text or document ids.

The official FineWeb-2 test split is NOT read by this script. It is only listed
in the plan (metadata) and stays reserved for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MB = 1_000_000
SCRIPT_VERSION = "0.1.1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

SAFETY_MARGIN = 0.90      # effective budget = 90% of the agreed cap
PLAN_SAFETY = 1.30        # plan for 30% more text than the target
FOOTER_TAIL = 64 * 1024   # first guess for the parquet footer size
HOLDOUT_MOD = 100         # 1% of documents
HTTP_TIMEOUT = 120
HOLDOUT_RULE = (
    "eval if int.from_bytes(sha256(utf8(id)).digest()[:8], 'big') % 100 == 0"
)

SOURCES = {
    "en": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "train_dir": "sample/10BT",
        "test_dir": None,
    },
    "id": {
        "repo": "HuggingFaceFW/fineweb-2",
        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9",
        "train_dir": "data/ind_Latn/train",
        "test_dir": "data/ind_Latn/test",
    },
}

PROFILES = {
    "trial": {
        "cap": 150 * MB,
        "n_files": {"en": 1, "id": 1},
        "target_docs": 100,
        "target_bytes": None,
        "out": "sample_trial",
    },
    "full": {
        "cap": 400 * MB,
        "n_files": {"en": 3, "id": 4},
        "target_docs": None,
        "target_bytes": 100 * MB,
        "out": "sample",
    },
}


class BudgetExceeded(RuntimeError):
    pass


class PlanError(RuntimeError):
    pass


# ---------------------------------------------------------------- traffic


class Meter:
    """Counts response bytes requested by this program (see module docstring)."""

    def __init__(self, cap_bytes: int):
        self.cap = cap_bytes
        self.used = 0

    def reserve(self, n: int) -> None:
        if self.used + n > self.cap:
            raise BudgetExceeded(
                f"a request of {n / MB:.1f} MB would exceed the budget "
                f"({self.used / MB:.1f} of {self.cap / MB:.1f} MB already used)"
            )

    def add(self, n: int) -> None:
        self.used += n


def fetch_range(source, path: str, start: int, end: int, meter: Meter) -> bytes:
    n = end - start
    if n <= 0:
        raise ValueError("empty byte range")
    meter.reserve(n)
    data = source.read_range(path, start, end)
    meter.add(len(data))
    if len(data) != n:
        raise RuntimeError(f"expected {n} bytes, received {len(data)}")
    return data


# ---------------------------------------------------------------- sources


class HFSource:
    """Pinned Hugging Face dataset revision, read with HTTP Range requests.

    No token is sent. Redirects keep the Range header. If the server does not
    answer 206 Partial Content, the body is never read.
    """

    def __init__(self, repo: str, revision: str):
        self.repo = repo
        self.revision = revision

    def list_files(self, subdir: str):
        from huggingface_hub import HfApi

        items = HfApi().list_repo_tree(
            self.repo, path_in_repo=subdir, repo_type="dataset", revision=self.revision
        )
        files = [
            (f.path, int(f.size))
            for f in items
            if hasattr(f, "size") and f.path.endswith(".parquet")
        ]
        return sorted(files)

    def read_range(self, path: str, start: int, end: int) -> bytes:
        from huggingface_hub import hf_hub_url

        n = end - start
        url = hf_hub_url(self.repo, path, repo_type="dataset", revision=self.revision)
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={start}-{end - 1}",
                "User-Agent": f"aegis-core-sample/{SCRIPT_VERSION}",
            },
        )
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"server answered HTTP {response.status} instead of 206; "
                    "refusing to read the body"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end - 1}/"):
                raise RuntimeError("unexpected Content-Range header")
            body = response.read(n + 1)
        if len(body) != n:
            raise RuntimeError(f"expected {n} bytes, received {len(body)}")
        return body


def make_hf_source(lang: str) -> HFSource:
    src = SOURCES[lang]
    return HFSource(src["repo"], src["revision"])


# ---------------------------------------------------------------- parquet


class SparseFile(io.RawIOBase):
    """Read-only file object that serves only byte ranges that were downloaded.

    Reading anything else raises OSError, so a missed range fails loudly
    instead of silently triggering a download.
    """

    def __init__(self, size: int):
        self._size = size
        self._pos = 0
        self._chunks = []  # (start, data, permanent)

    def add(self, start: int, data: bytes, permanent: bool = False) -> None:
        self._chunks.append((start, data, permanent))

    def clear_temporary(self) -> None:
        self._chunks = [c for c in self._chunks if c[2]]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError("invalid whence")
        return self._pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self._size - self._pos)
        if want <= 0:
            return 0
        pos, filled = self._pos, 0
        while filled < want:  # a read may span adjacent downloaded chunks
            for start, data, _ in self._chunks:
                if start <= pos < start + len(data):
                    take = min(want - filled, start + len(data) - pos)
                    buffer[filled : filled + take] = data[pos - start : pos - start + take]
                    pos += take
                    filled += take
                    break
            else:
                raise OSError(
                    f"not downloaded: first missing byte {pos}; "
                    f"the read started at {self._pos} and ends at {self._pos + want}"
                )
        self._pos += want
        return want


def read_footer(source, path: str, size: int, meter: Meter):
    import pyarrow.parquet as pq

    tail_len = min(size, FOOTER_TAIL)
    tail = fetch_range(source, path, size - tail_len, size, meter)
    if len(tail) < 8 or tail[-4:] != b"PAR1":
        raise ValueError(f"{path}: not a parquet file")
    need = int.from_bytes(tail[-8:-4], "little") + 8
    if need > size:
        raise ValueError(f"{path}: corrupt footer length")
    if need > tail_len:
        head = fetch_range(source, path, size - need, size - tail_len, meter)
        tail = head + tail
    sparse = SparseFile(size)
    sparse.add(0, b"PAR1", permanent=True)
    sparse.add(size - len(tail), tail, permanent=True)
    # pre_buffer=False: read exactly each column chunk. The default would merge nearby
    # ranges into one read that also covers the small gap between column chunks.
    return pq.ParquetFile(sparse, pre_buffer=False), sparse


def column_range(column) -> tuple:
    offsets = [column.data_page_offset]
    if column.has_dictionary_page:
        offsets.append(column.dictionary_page_offset)
    start = min(o for o in offsets if o)
    return start, start + column.total_compressed_size


def rowgroup_stats(pf) -> list:
    metadata = pf.metadata
    stats = []
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        columns = {rg.column(j).path_in_schema: rg.column(j) for j in range(rg.num_columns)}
        if "text" not in columns or "id" not in columns:
            raise PlanError("parquet file has no 'text' or 'id' column")
        text, ident = columns["text"], columns["id"]
        stats.append(
            {
                "idx": i,
                "rows": rg.num_rows,
                "text_c": text.total_compressed_size,
                "text_u": text.total_uncompressed_size,
                "id_c": ident.total_compressed_size,
                "ranges": [column_range(text), column_range(ident)],
            }
        )
    return stats


# ---------------------------------------------------------------- planning


@dataclass
class FilePlan:
    path: str
    size: int
    footer_bytes: int
    pf: object
    sparse: SparseFile
    rgs: list


@dataclass
class LangPlan:
    lang: str
    total_files: int
    files: list
    selected: list
    est_text: int
    est_rows: int
    est_traffic: int
    footer_traffic: int
    cap_eff: int
    feasible: bool
    reason: str
    test_split: object


def choose_files(entries: list, k: int) -> list:
    n = len(entries)
    if k >= n:
        return list(entries)
    positions = sorted({int((j + 0.5) * n / k) for j in range(k)})
    return [entries[i] for i in positions]


def spread_order(n: int) -> list:
    """Row-group indices ordered so that early picks are spread out."""
    if n == 0:
        return []
    first = n // 2
    chosen = [first]
    remaining = set(range(n)) - {first}
    while remaining:
        best = max(remaining, key=lambda i: (min(abs(i - c) for c in chosen), -i))
        chosen.append(best)
        remaining.discard(best)
    return chosen


def rg_cost(rg: dict) -> int:
    return rg["text_c"] + rg["id_c"]


def satisfied(prof: dict, est_text: int, est_rows: int) -> bool:
    if prof["target_bytes"]:
        return est_text >= PLAN_SAFETY * prof["target_bytes"]
    return est_rows >= PLAN_SAFETY * prof["target_docs"]


def make_plan(lang: str, source, prof: dict, cap_eff: int, meter: Meter) -> LangPlan:
    src = SOURCES[lang]
    entries = source.list_files(src["train_dir"])
    if not entries:
        raise PlanError(f"[{lang}] no parquet files found in {src['train_dir']}")

    files = []
    for path, size in choose_files(entries, prof["n_files"][lang]):
        before = meter.used
        pf, sparse = read_footer(source, path, size, meter)
        files.append(FilePlan(path, size, meter.used - before, pf, sparse, rowgroup_stats(pf)))
    footer_traffic = meter.used

    if prof["target_docs"]:  # trial: cheapest row groups first
        sequence = sorted(
            ((fi, rg["idx"]) for fi, f in enumerate(files) for rg in f.rgs),
            key=lambda t: (rg_cost(files[t[0]].rgs[t[1]]), t),
        )
    else:  # full: spread across files and across each file's row groups
        orders = [spread_order(len(f.rgs)) for f in files]
        sequence = []
        for pos in range(max(len(o) for o in orders)):
            for fi, order in enumerate(orders):
                if pos < len(order):
                    sequence.append((fi, order[pos]))

    selected, est_text, est_rows, est_traffic = [], 0, 0, footer_traffic
    feasible, reason = False, "source exhausted before the target was reached"
    for fi, ri in sequence:
        rg = files[fi].rgs[ri]
        if est_traffic + rg_cost(rg) > cap_eff:
            reason = (
                f"next row group costs {rg_cost(rg) / MB:.1f} MB; worst-case traffic would be "
                f"{(est_traffic + rg_cost(rg)) / MB:.1f} MB, above the effective cap "
                f"{cap_eff / MB:.1f} MB"
            )
            break
        selected.append((fi, ri))
        est_traffic += rg_cost(rg)
        est_text += rg["text_u"]
        est_rows += rg["rows"]
        if satisfied(prof, est_text, est_rows):
            feasible, reason = True, "ok"
            break

    test_split = None
    if src["test_dir"]:
        try:
            test_files = source.list_files(src["test_dir"])
            test_split = {"files": len(test_files), "bytes": sum(s for _, s in test_files)}
        except Exception as exc:  # metadata only; failure is informational
            test_split = {"error": type(exc).__name__}

    return LangPlan(
        lang, len(entries), files, selected, est_text, est_rows,
        est_traffic, footer_traffic, cap_eff, feasible, reason, test_split,
    )


def print_plan(plan: LangPlan, prof: dict) -> None:
    src = SOURCES[plan.lang]
    print(f"\n== {plan.lang}: {src['repo']} @ {src['revision'][:12]} ==")
    print(f"  source files: {plan.total_files}; chosen: {len(plan.files)}; "
          f"footer traffic: {plan.footer_traffic / MB:.2f} MB")
    for f in plan.files:
        costs = sorted(rg_cost(rg) for rg in f.rgs)
        print(f"  - {f.path}: {f.size / MB:.0f} MB, {len(f.rgs)} row groups, "
              f"text+id per row group min/median/max = "
              f"{costs[0] / MB:.1f}/{costs[len(costs) // 2] / MB:.1f}/{costs[-1] / MB:.1f} MB")
    if prof["target_bytes"]:
        target = f"{prof['target_bytes'] / MB:.0f} MB text (x{PLAN_SAFETY} planned)"
    else:
        target = f"{prof['target_docs']} documents (x{PLAN_SAFETY} planned)"
    print(f"  target: {target}")
    print(f"  selected row groups: {len(plan.selected)}; est. rows {plan.est_rows}; "
          f"est. text {plan.est_text / MB:.1f} MB")
    print(f"  worst-case traffic: {plan.est_traffic / MB:.1f} MB "
          f"(effective cap {plan.cap_eff / MB:.1f} MB)")
    if plan.test_split is not None:
        print(f"  official test split (metadata only, not read): {plan.test_split}")
    print(f"  RESULT: {'FEASIBLE' if plan.feasible else 'NOT FEASIBLE'} - {plan.reason}")


# ---------------------------------------------------------------- run


def is_holdout(doc_id: str) -> bool:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % HOLDOUT_MOD == 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_language(plan: LangPlan, prof: dict, source, meter: Meter, train_path: Path, eval_path: Path) -> dict:
    stats = {
        "rows_scanned": 0, "train_docs": 0, "train_bytes": 0, "eval_docs": 0,
        "eval_bytes": 0, "skipped_empty": 0, "skipped_duplicate": 0, "row_groups_read": [],
    }
    seen = set()
    done = False
    with open(train_path, "w", encoding="utf-8", newline="\n") as f_train, \
            open(eval_path, "w", encoding="utf-8", newline="\n") as f_eval:
        for fi, ri in plan.selected:
            if done:
                break
            fp = plan.files[fi]
            for start, end in fp.rgs[ri]["ranges"]:
                fp.sparse.add(start, fetch_range(source, fp.path, start, end, meter))
            stats["row_groups_read"].append([fp.path, ri])
            batches = fp.pf.iter_batches(
                batch_size=512, row_groups=[ri], columns=["id", "text"], use_threads=False
            )
            for batch in batches:
                ids = batch.column("id").to_pylist()
                texts = batch.column("text").to_pylist()
                for doc_id, text in zip(ids, texts):
                    stats["rows_scanned"] += 1
                    if not doc_id or not text:
                        stats["skipped_empty"] += 1
                        continue
                    key = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                    if key in seen:
                        stats["skipped_duplicate"] += 1
                        continue
                    seen.add(key)
                    nbytes = len(text.encode("utf-8"))
                    line = json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n"
                    if is_holdout(doc_id):
                        f_eval.write(line)
                        stats["eval_docs"] += 1
                        stats["eval_bytes"] += nbytes
                    else:
                        f_train.write(line)
                        stats["train_docs"] += 1
                        stats["train_bytes"] += nbytes
                        if (prof["target_bytes"] and stats["train_bytes"] >= prof["target_bytes"]) or \
                                (prof["target_docs"] and stats["train_docs"] >= prof["target_docs"]):
                            done = True
                            break
                if done:
                    break
            fp.sparse.clear_temporary()
    if not done:
        raise RuntimeError(
            "planned row groups were exhausted before the target was reached "
            "(the estimate was too optimistic); nothing was extended automatically"
        )
    stats["traffic_bytes_requested"] = meter.used
    return stats


# ---------------------------------------------------------------- main


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Plan (default) or build the tokenizer sample. See the module docstring.",
    )
    p.add_argument("--profile", choices=sorted(PROFILES), default="trial")
    p.add_argument("--langs", default="en,id", help="comma-separated: en,id")
    p.add_argument("--run", action="store_true",
                   help="download and write the sample (default: plan only)")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--max-traffic-mb", type=float, default=None,
                   help="lower the traffic cap; it can never be raised above the profile cap")
    return p.parse_args(argv)


def main(argv=None, source_factory=None) -> int:
    args = parse_args(argv)
    try:
        import pyarrow
    except ImportError:
        print("ERROR: pyarrow is required (pip install pyarrow).")
        return 3

    prof = PROFILES[args.profile]
    cap = prof["cap"]
    if args.max_traffic_mb is not None:
        lowered = int(args.max_traffic_mb * MB)
        if lowered > cap:
            print(f"ERROR: --max-traffic-mb cannot exceed the {cap / MB:.0f} MB cap of this profile.")
            return 2
        cap = lowered
    cap_eff = int(cap * SAFETY_MARGIN)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    unknown = [x for x in langs if x not in SOURCES]
    if unknown or not langs:
        print(f"ERROR: unknown language(s): {unknown or args.langs}")
        return 2

    factory = source_factory or make_hf_source
    mode = "RUN" if args.run else "PLAN ONLY (no document text is read)"
    print(f"Aegis-Core sample builder v{SCRIPT_VERSION} | profile={args.profile} | mode={mode}")
    print(f"Traffic cap per language: {cap / MB:.0f} MB (effective {cap_eff / MB:.0f} MB). "
          "The byte counter is an estimate, not a network guarantee.")

    sources = {lang: factory(lang) for lang in langs}
    meters = {lang: Meter(cap_eff) for lang in langs}
    plans = {}
    try:
        for lang in langs:
            plans[lang] = make_plan(lang, sources[lang], prof, cap_eff, meters[lang])
    except (BudgetExceeded, PlanError) as exc:
        print(f"STOP: {exc}")
        return 2
    except Exception as exc:  # network or format problem while reading metadata
        print(f"ERROR while planning: {type(exc).__name__}: {exc}")
        return 3

    for lang in langs:
        print_plan(plans[lang], prof)

    if not all(plans[lang].feasible for lang in langs):
        print("\nSTOP: at least one language is not feasible within its traffic cap. "
              "Nothing was downloaded beyond footers, and the cap was not raised.")
        return 2
    if not args.run:
        print("\nPlan only. Re-run with --run to download (requires separate approval).")
        return 0

    trial_manifest = DATA_DIR / PROFILES["trial"]["out"] / "MANIFEST.json"
    if args.profile == "full" and not trial_manifest.exists():
        print("STOP: run the trial profile first (data/sample_trial/MANIFEST.json is missing).")
        return 2

    out_dir = DATA_DIR / prof["out"]
    names = [f"{lang}_{kind}.jsonl" for lang in langs for kind in ("train", "eval")] + ["MANIFEST.json"]
    if any((out_dir / n).exists() for n in names) and not args.overwrite:
        print(f"STOP: {out_dir} already has output files; use --overwrite to replace them.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    partials = {n: out_dir / (n + ".partial") for n in names if n != "MANIFEST.json"}
    results = {}
    try:
        for lang in langs:
            results[lang] = run_language(
                plans[lang], prof, sources[lang], meters[lang],
                partials[f"{lang}_train.jsonl"], partials[f"{lang}_eval.jsonl"],
            )
            r = results[lang]
            print(f"[{lang}] train docs {r['train_docs']} ({r['train_bytes'] / MB:.1f} MB), "
                  f"eval docs {r['eval_docs']} ({r['eval_bytes'] / MB:.2f} MB), "
                  f"traffic requested {r['traffic_bytes_requested'] / MB:.1f} MB")
        outputs = {}
        for name, partial in partials.items():
            final = out_dir / name
            partial.replace(final)
            outputs[name] = {"bytes": final.stat().st_size, "sha256": sha256_file(final)}
    except Exception as exc:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        print(f"ERROR during run: {type(exc).__name__}: {exc}. Partial files were removed.")
        return 3

    manifest = {
        "script": {"name": "scripts/sample_corpus.py", "version": SCRIPT_VERSION,
                   "sha256": sha256_file(Path(__file__))},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": args.profile,
        "targets": {"train_bytes": prof["target_bytes"], "train_docs": prof["target_docs"]},
        "holdout_rule": HOLDOUT_RULE,
        "traffic": {
            "unit": "response bytes requested by this program; not a network guarantee",
            "cap_agreed_bytes": cap, "safety_margin": SAFETY_MARGIN, "cap_effective_bytes": cap_eff,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "languages": {
            lang: {
                "repo": SOURCES[lang]["repo"], "revision": SOURCES[lang]["revision"],
                "source_files_total": plans[lang].total_files,
                "files_chosen": [{"path": f.path, "size_bytes": f.size} for f in plans[lang].files],
                "stats": results[lang],
            }
            for lang in langs
        },
        "outputs": outputs,
    }
    manifest["environment"]["pyarrow"] = pyarrow.__version__
    if args.profile == "full":
        manifest["trial_manifest_sha256"] = sha256_file(trial_manifest)
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Done. Output in {out_dir}. Nothing was committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
