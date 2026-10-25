# Data Sources and Licenses

Status: PLANNED. No corpus has been downloaded yet.
This file is updated before any data is downloaded or used.

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
  include a small test split that should not be trained on; its presence for
  `ind_Latn` is still to be verified when the sample is taken.
- FineWeb-Edu has only a `train` split on its card. English evaluation
  documents will be held out from `sample-10BT` by a deterministic rule on the
  document `id` (rule to be documented here before use) and excluded from training.
- Deduplication method and results will be documented here.

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
