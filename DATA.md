# Data Sources and Licenses

Status: PLANNED. The full tokenizer sample has not been downloaded yet.
A small trial run (2026-10-25, see "Trial run") tested the sampling script;
it is not used for training or evaluation.
This file is updated before any data is used for training or evaluation.

## Tokenizer training sample (planned)

| Source | Language | License | Config |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | English | ODC-By 1.0 | sample-10BT |
| HuggingFaceFW/fineweb-2 | Indonesian | ODC-By 1.0, subject to Common Crawl Terms of Use | ind_Latn |

Only a small slice of each source is used (streamed, not fully downloaded).
Revisions are pinned in the next section.

## Pinned revisions

Checked on 2026-10-25 through the Hugging Face API (metadata only, no data downloaded).

| Dataset | Config / path | Revision (commit) | Last modified (UTC) | License tag |
|---|---|---|---|---|
| HuggingFaceFW/fineweb-edu | sample-10BT | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 | 2025-07-11 | odc-by |
| HuggingFaceFW/fineweb-2 | ind_Latn | af9c13333eb981300149d5ca60a8e9d659b276b9 | 2025-10-27 | odc-by |

The dataset cards list FineWeb-Edu v1.4.0 (2025-07-11) and FineWeb-2 v2.1.1
(2025-10-27) as their latest changelog entries. These dates match the
last-modified dates above. Loading code must pass these commit hashes as the
`revision` argument so every contributor gets identical data.

Server-side file sizes (metadata, not downloaded): `sample/10BT` has 14 files
(about 28.5 GB); `data/ind_Latn/train` has 32 files (about 152.2 GB).

## Rules

- Raw data is NOT stored in this repository. The `data/` folder is git-ignored.
- Attribution to the sources above is required by ODC-By and is kept in this file.
- Evaluation documents are kept separate from tokenizer and model training data.
- The FineWeb-2 `test` split is reserved for evaluation only. It is never used
  to train the tokenizer or any model. The dataset card says most languages
  include a small test split that should not be trained on. Its presence for
  `ind_Latn` was verified on 2026-10-25 (metadata only): `data/ind_Latn/test`
  has 1 file of 58,341,633 bytes. It was not read or downloaded.
- FineWeb-Edu has only a `train` split on its card. English evaluation
  documents are held out from `sample-10BT` by a deterministic rule on the
  document `id` (see "Holdout rule") and excluded from training.
- Deduplication: exact duplicate texts are removed inside the sample (see
  "Sampling method"). Results for the full sample will be documented here.

## Sampling method

The sample is built by `scripts/sample_corpus.py` (version 0.1.1 at the time of
writing). It plans first and downloads only when run with `--run`.

- Plan mode reads only parquet footers (metadata) and prints the worst-case
  traffic. No document text is read.
- With `--run`, only the `text` and `id` column chunks of the selected row
  groups are downloaded, using HTTP Range requests on the pinned revisions.
- Traffic caps per language: 150 MB for the trial profile and 400 MB for the
  full profile. The effective budget is 90% of the cap. The script never raises
  a cap by itself and stops if the plan does not fit. Its byte counter counts
  requested response bytes only; it is an estimate, not a measure of network
  traffic.
- Targets: trial = about 100 training documents per language; full = 100 MB of
  training text per language (Indonesian:English 50:50).
- Exact duplicate texts are removed within the sample (64-bit BLAKE2b digest of
  the text). Near-duplicates are not removed.
- Output goes to `data/`, which is git-ignored. `MANIFEST.json` records the
  script version and SHA-256, the pinned revisions, the files and row groups
  used, statistics, and checksums. It contains no document text and no
  document ids.

## Holdout rule

About 1% of the sampled documents (both languages) are set aside for evaluation.
A document is an evaluation document when

    int.from_bytes(sha256(utf8(id)).digest()[:8], "big") % 100 == 0

The decision depends only on the document `id`. Evaluation documents are written
to separate `*_eval.jsonl` files, are not counted toward the training target,
and are never used to train the tokenizer or any model. The official FineWeb-2
`test` split is a separate, additional evaluation source and is not read by the
script.

## Trial run (2026-10-25)

A trial run of version 0.1.1 tested the script on the real sources. It
downloaded one row group per language (100 training documents each; 0 English
and 2 Indonesian evaluation documents) with about 20 MB of counted traffic
(about 21 MB measured on the network interface). The trial data was downloaded
before this section was written. It is for testing only: the cheapest row
groups were chosen, so it is not representative, and it is not used for
training or evaluation.

## Provenance (from the dataset cards)

- Both datasets come from Common Crawl web pages. Use is also subject to
  Common Crawl's Terms of Use.
- FineWeb-Edu keeps pages from FineWeb that scored 3 or higher (scale 0 to 5)
  with an educational-quality classifier. The classifier was trained on
  annotations produced by Llama-3-70B-Instruct. `sample-10BT` is a random
  sample of about 10B GPT-2 tokens.
- FineWeb-2 identifies languages with GlotLID, deduplicates per language
  across all snapshots, and applies per-language filters.

## Privacy

- FineWeb-2 anonymizes email addresses and public IP addresses. Phone numbers
  are not anonymized. The card states that other personal information is very
  likely still present.
- Both cards refer to a removal / opt-out process for personal data and for
  webmasters. Common Crawl respects robots.txt at crawl time.

## Bias and known limitations

- Adult and toxic content was filtered only at URL level. A significant number
  of documents that could be considered toxic or harmful remain, and web biases
  may be reproduced.
- FineWeb-Edu: code content is likely not prevalent, and processing was not
  tailored to individual websites.
- FineWeb-2: the language classifier can confuse closely related languages, and
  some filters may remove certain content disproportionately (the card gives
  poetry as an example).
- These are web-crawled datasets. They may contain copyrighted text, personal
  information, and social biases. The dataset license does not clear the rights
  of every underlying page.
- This is not legal advice.

## Citations

- Lozhkov, Ben Allal, von Werra, Wolf. FineWeb-Edu: the Finest Collection of
  Educational Content. 2024. doi:10.57967/hf/2497
- Penedo et al. FineWeb2: One Pipeline to Scale Them All. 2025. arXiv:2506.20920
