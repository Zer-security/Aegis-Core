# Data Sources and Licenses

Status: PLANNED. No corpus has been downloaded yet.
This file is updated before any data is downloaded or used.

## Tokenizer training sample (planned)

| Source | Language | License | Config |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | English | ODC-By 1.0 | sample-10BT |
| HuggingFaceFW/fineweb-2 | Indonesian | ODC-By 1.0, subject to Common Crawl Terms of Use | ind_Latn |

Only a small slice of each source is used (streamed, not fully downloaded).
Exact dataset revisions (commit hashes) will be pinned here once chosen.

## Rules

- Raw data is NOT stored in this repository. The `data/` folder is git-ignored.
- Attribution to the sources above is required by ODC-By and is kept in this file.
- Evaluation documents are kept separate from tokenizer and model training data.
- Deduplication method and results will be documented here.

## Known limitations

- These are web-crawled datasets. They may contain copyrighted text, personal
  information, and social biases. The dataset license does not clear the rights
  of every underlying page.
- This is not legal advice.
