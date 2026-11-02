"""Aegis-Core: independent evaluation sample (v0.1.0, prototype).

PLAN FIRST. Without --run only parquet footers (metadata) are read.

- id: the official FineWeb-2 ind_Latn `test` split (never read by sample_corpus.py).
- en: FineWeb-Edu sample/10BT files that are NOT in data/sample/MANIFEST.json files_chosen.
  Independence for en is at file level only (same Common Crawl origin).
- Exact duplicates (BLAKE2b-64 of the text) of the training files are dropped and counted.
- Reuses fetch_range/Meter/read_footer from sample_corpus.py (traffic reserved before
  every request; the cap can only be lowered).
- Never prints document text or ids; the manifest contains neither.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_corpus as sc  # noqa: E402

__version__ = "0.1.0"
CAP = 100 * sc.MB
N_FILES = {"en": 2, "id": 1}
MIN_ROW_GROUPS = 4
FLOOR = 0.8          # run fails (nothing written) if eval text < 80% of target
OUT_SUBDIR = "eval_indep"
TRAIN_NAMES = ("en_train.jsonl", "id_train.jsonl")


@dataclass
class FilePlan:
    path: str
    size: int
    pf: object
    sparse: object
    rgs: list


@dataclass
class LangPlan:
    lang: str
    candidates: int
    files: list
    selected: list
    est_text: int
    est_traffic: int
    footer_traffic: int
    cap_eff: int
    feasible: bool
    reason: str


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def check_train_manifest(sample_dir: Path):
    mp = sample_dir / "MANIFEST.json"
    if not mp.is_file():
        raise sc.PlanError("data/sample/MANIFEST.json is missing (run the full profile first)")
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise sc.PlanError("data/sample/MANIFEST.json is unreadable") from None
    if not isinstance(m, dict) or m.get("profile") != "full":
        raise sc.PlanError("data/sample/MANIFEST.json is not a full-profile manifest")
    outputs = m.get("outputs")
    if not isinstance(outputs, dict) or any(n not in outputs for n in TRAIN_NAMES):
        raise sc.PlanError("manifest does not list the training files")
    for name, info in outputs.items():
        f = sample_dir / name
        if (name != Path(name).name or not isinstance(info, dict) or not f.is_file()
                or sc.sha256_file(f) != info.get("sha256")):
            raise sc.PlanError(f"output does not match the manifest: {name}")
    langs = m.get("languages")
    if not isinstance(langs, dict):
        raise sc.PlanError("manifest has no languages section")
    for lang, src in sc.SOURCES.items():
        e = langs.get(lang)
        if (not isinstance(e, dict) or e.get("revision") != src["revision"]
                or not isinstance(e.get("files_chosen"), list)):
            raise sc.PlanError(f"manifest [{lang}]: missing revision or files_chosen")
    return m, sc.sha256_file(mp)


def load_train_digests(path: Path) -> set:
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(digest(json.loads(line)["text"]))
    return out


def make_plan(lang, source, used, cap_eff, target, meter) -> LangPlan:
    src = sc.SOURCES[lang]
    if lang == "id":
        if not src["test_dir"]:
            raise sc.PlanError("[id] no official test split configured")
        entries = source.list_files(src["test_dir"])
    else:
        entries = [e for e in source.list_files(src["train_dir"]) if e[0] not in used]
    if not entries:
        raise sc.PlanError(f"[{lang}] no candidate parquet files")
    files = []
    for path, size in sc.choose_files(entries, N_FILES[lang]):
        pf, sparse = sc.read_footer(source, path, size, meter)
        files.append(FilePlan(path, size, pf, sparse, sc.rowgroup_stats(pf)))
    footer = meter.used
    orders = [sc.spread_order(len(f.rgs)) for f in files]
    sequence = []
    for pos in range(max(len(o) for o in orders)):
        for fi, order in enumerate(orders):
            if pos < len(order):
                sequence.append((fi, order[pos]))
    selected, est_text, est_traffic = [], 0, footer
    feasible, reason = False, "source exhausted before the target was reached"
    for fi, ri in sequence:
        rg = files[fi].rgs[ri]
        if est_traffic + sc.rg_cost(rg) > cap_eff:
            reason = (f"next row group would raise worst-case traffic to "
                      f"{(est_traffic + sc.rg_cost(rg)) / sc.MB:.1f} MB, above the cap "
                      f"{cap_eff / sc.MB:.1f} MB")
            break
        selected.append((fi, ri))
        est_traffic += sc.rg_cost(rg)
        est_text += rg["text_u"]
        if est_text >= target and len(selected) >= MIN_ROW_GROUPS:
            feasible, reason = True, "ok"
            break
    return LangPlan(lang, len(entries), files, selected, est_text, est_traffic, footer,
                    cap_eff, feasible, reason)


def print_plan(p: LangPlan, target: int) -> None:
    src = sc.SOURCES[p.lang]
    print(f"\n== {p.lang}: {src['repo']} @ {src['revision'][:12]} ==")
    print(f"  candidate files: {p.candidates}; chosen: {len(p.files)}; "
          f"footer traffic: {p.footer_traffic / sc.MB:.2f} MB")
    for f in p.files:
        costs = sorted(sc.rg_cost(rg) for rg in f.rgs)
        print(f"  - {f.path}: {f.size / sc.MB:.0f} MB, {len(f.rgs)} row groups, "
              f"text+id per row group min/median/max = "
              f"{costs[0] / sc.MB:.2f}/{costs[len(costs) // 2] / sc.MB:.2f}/{costs[-1] / sc.MB:.2f} MB")
    print(f"  target: {target / sc.MB:.2f} MB text; selected row groups: {len(p.selected)}; "
          f"est. text {p.est_text / sc.MB:.2f} MB")
    print(f"  worst-case traffic: {p.est_traffic / sc.MB:.1f} MB (effective cap {p.cap_eff / sc.MB:.1f} MB)")
    print(f"  RESULT: {'FEASIBLE' if p.feasible else 'NOT FEASIBLE'} - {p.reason}")


def run_language(plan, source, meter, train_digests, out_path, target) -> dict:
    stats = {"rows_scanned": 0, "eval_docs": 0, "eval_bytes": 0, "skipped_empty": 0,
             "skipped_train_duplicate": 0, "skipped_internal_duplicate": 0,
             "row_groups_read": []}
    seen = set()
    with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        for fi, ri in plan.selected:
            fp = plan.files[fi]
            for start, end in fp.rgs[ri]["ranges"]:
                fp.sparse.add(start, sc.fetch_range(source, fp.path, start, end, meter))
            stats["row_groups_read"].append([fp.path, ri])
            for batch in fp.pf.iter_batches(batch_size=512, row_groups=[ri],
                                            columns=["id", "text"], use_threads=False):
                for doc_id, text in zip(batch.column("id").to_pylist(),
                                        batch.column("text").to_pylist()):
                    stats["rows_scanned"] += 1
                    if not doc_id or not text:
                        stats["skipped_empty"] += 1
                        continue
                    k = digest(text)
                    if k in train_digests:
                        stats["skipped_train_duplicate"] += 1
                        continue
                    if k in seen:
                        stats["skipped_internal_duplicate"] += 1
                        continue
                    seen.add(k)
                    fout.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
                    stats["eval_docs"] += 1
                    stats["eval_bytes"] += len(text.encode("utf-8"))
            fp.sparse.clear_temporary()
    if stats["eval_bytes"] < FLOOR * target:
        raise RuntimeError(f"eval text {stats['eval_bytes'] / sc.MB:.2f} MB is below "
                           f"{FLOOR:.0%} of the target; nothing was extended automatically")
    stats["traffic_bytes_requested"] = meter.used
    return stats


def parse_args(argv):
    p = argparse.ArgumentParser(description="Plan (default) or build the independent eval sample.")
    p.add_argument("--langs", default="en,id")
    p.add_argument("--run", action="store_true", help="download and write (default: plan only)")
    p.add_argument("--target-mb", type=float, default=10.0, help="eval text per language")
    p.add_argument("--max-traffic-mb", type=float, default=None,
                   help="lower the traffic cap; it can never exceed 100 MB")
    return p.parse_args(argv)


def main(argv=None, source_factory=None) -> int:
    a = parse_args(argv)
    try:
        import pyarrow
    except ImportError:
        print("ERROR: pyarrow is required.")
        return 3
    cap = CAP
    if a.max_traffic_mb is not None:
        low = int(a.max_traffic_mb * sc.MB)
        if low > cap:
            print(f"ERROR: --max-traffic-mb cannot exceed {cap / sc.MB:.0f} MB.")
            return 2
        cap = low
    cap_eff = int(cap * sc.SAFETY_MARGIN)
    if a.target_mb <= 0:
        print("ERROR: --target-mb must be positive.")
        return 2
    target = int(a.target_mb * sc.MB)
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    unknown = [x for x in langs if x not in sc.SOURCES]
    if unknown or not langs:
        print(f"ERROR: unknown language(s): {unknown or a.langs}")
        return 2

    sample_dir = sc.DATA_DIR / "sample"
    out_dir = sc.DATA_DIR / OUT_SUBDIR
    names = [f"{lang}_eval.jsonl" for lang in langs] + ["MANIFEST.json"]
    try:  # local checks first: every refusal here happens BEFORE any network access
        tm, tm_sha = check_train_manifest(sample_dir)
    except sc.PlanError as exc:
        print(f"STOP: {exc}.")
        return 2
    if a.run:
        if any((out_dir / n).exists() for n in names):
            print(f"STOP: {out_dir} already has output files; nothing was changed.")
            return 2
        left = sorted(p.name for p in out_dir.iterdir()
                      if p.name.endswith((".partial", ".bak"))) if out_dir.is_dir() else []
        if left:
            print(f"STOP: leftover files in {out_dir}: {left}. Inspect and remove manually.")
            return 2

    factory = source_factory or sc.make_hf_source
    print(f"Aegis-Core eval sampler v{__version__} | mode={'RUN' if a.run else 'PLAN ONLY (no document text is read)'}")
    print(f"Traffic cap per language: {cap / sc.MB:.0f} MB (effective {cap_eff / sc.MB:.0f} MB); "
          "the counter is an estimate, not a network guarantee.")
    sources = {lang: factory(lang) for lang in langs}
    meters = {lang: sc.Meter(cap_eff) for lang in langs}
    plans = {}
    try:
        for lang in langs:
            used = {c["path"] for c in tm["languages"][lang]["files_chosen"]}
            plans[lang] = make_plan(lang, sources[lang], used, cap_eff, target, meters[lang])
    except (sc.BudgetExceeded, sc.PlanError) as exc:
        print(f"STOP: {exc}")
        return 2
    except Exception as exc:
        print(f"ERROR while planning: {type(exc).__name__}: {exc}")
        return 3
    for lang in langs:
        print_plan(plans[lang], target)
    if not all(plans[lang].feasible for lang in langs):
        print("\nSTOP: at least one language is not feasible. Nothing was downloaded beyond footers.")
        return 2
    if not a.run:
        print("\nPlan only. Re-run with --run to download (requires separate approval).")
        return 0

    train_digests = {lang: load_train_digests(sample_dir / f"{lang}_train.jsonl") for lang in langs}
    out_dir.mkdir(parents=True, exist_ok=True)
    partials = {n: out_dir / (n + ".partial") for n in names}
    results, outputs, promoted = {}, {}, []
    try:
        for lang in langs:
            results[lang] = run_language(plans[lang], sources[lang], meters[lang],
                                         train_digests[lang], partials[f"{lang}_eval.jsonl"], target)
            r = results[lang]
            print(f"[{lang}] eval docs {r['eval_docs']} ({r['eval_bytes'] / sc.MB:.2f} MB), "
                  f"dup_vs_train {r['skipped_train_duplicate']}, "
                  f"traffic requested {r['traffic_bytes_requested'] / sc.MB:.1f} MB")
        for n in names[:-1]:
            outputs[n] = {"bytes": partials[n].stat().st_size, "sha256": sc.sha256_file(partials[n])}
        manifest = {
            "script": {"name": "scripts/sample_eval.py", "version": __version__,
                       "sha256": sc.sha256_file(Path(__file__))},
            "sampler_dependency": {"name": "scripts/sample_corpus.py", "version": sc.SCRIPT_VERSION},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target_bytes_per_language": target,
            "train_manifest_sha256": tm_sha,
            "independence": {"id": "official FineWeb-2 test split (separate file)",
                             "en": "FineWeb-Edu files not used for training (file level only)"},
            "traffic": {"cap_agreed_bytes": cap, "cap_effective_bytes": cap_eff,
                        "unit": "response bytes requested by this program; not a network guarantee"},
            "environment": {"python": platform.python_version(), "pyarrow": pyarrow.__version__},
            "languages": {lang: {"repo": sc.SOURCES[lang]["repo"], "revision": sc.SOURCES[lang]["revision"],
                                 "files_chosen": [{"path": f.path, "size_bytes": f.size}
                                                  for f in plans[lang].files],
                                 "stats": results[lang]} for lang in langs},
            "outputs": outputs,
        }
        partials["MANIFEST.json"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        for n in names:  # MANIFEST.json last
            partials[n].replace(out_dir / n)
            promoted.append(n)
    except BaseException as exc:
        for n in reversed(promoted):
            (out_dir / n).unlink(missing_ok=True)
        for p in partials.values():
            p.unlink(missing_ok=True)
        print(f"ERROR during run: {type(exc).__name__}: {exc}. Partial and promoted files were removed.")
        if isinstance(exc, Exception):
            return 3
        raise
    print(f"Done. Output in {out_dir}. Nothing was committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
