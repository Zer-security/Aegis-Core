# Aegis-Nano (v3): experimental small language model

**Status: research artifact.** Small, CPU-trained, not fluent, and not evaluated
on any task benchmark. Not suitable for any production or user-facing use.
Weights are **not published**; see "License".

## Model

- Decoder-only, pre-LayerNorm Transformer; GELU MLP (4x); learned position
  embeddings; tied input/output embeddings.
- 6 layers, 6 heads, embedding width 384, context 256 tokens, dropout 0.1,
  no bias terms.
- Tokenizer: byte-level BPE (Hugging Face `tokenizers`), vocabulary 16,384,
  trained on the training split only. Tokenizer sha256 `a31e3dcf2c889f7b44ced454e560676767fcba0e871fc9bb9c4c3cb017cf4153`.
- Parameters: 17,011,584.
- Source code: Apache-2.0, https://github.com/Zer-security/Aegis-Core

## Training data

English (FineWeb-Edu `sample/10BT`) and Indonesian (FineWeb-2 `ind_Latn`) at
pinned revisions. Training split: 20,633 English and 26,623 Indonesian
documents, 46,651,978 BPE tokens in total. Licenses: ODC-By 1.0 and Common Crawl
Terms of Use; the dataset license does not clear the rights of every
underlying page. Provenance, holdout rule, dedup and limitations: see
[DATA.md](DATA.md). Only exact duplicates were removed.

## Training

Single run, CPU only, seed 1337. 22,800 steps, batch 8, block 256
(46,694,400 tokens, about one epoch-equivalent; windows are sampled at random
with replacement, so not a single full pass). AdamW (betas 0.9/0.95), weight
decay 0.1 on tensors with 2+ dimensions, learning rate 1e-3 to 1e-4 (cosine),
warmup 400, gradient clip 1.0. The learning rate was not tuned.
Checkpoint sha256 `5802ebe849b81ce31f1fb6fd623d1e25e199ebe323ca97d8444fcde760de3647`.
Environment: Python 3.14.4, torch 2.14.1+cpu.

## Evaluation

Metric: cross-entropy loss (nats) per BPE-16K token on non-overlapping
256-token windows. Perplexity depends on the tokenizer and is not comparable
with models that use a different one. One seed per configuration, no
confidence intervals.

| Checkpoint | Steps | Loss, independent set | Loss, earlier holdout |
|---|---|---|---|
| v1 | 2,000 | 6.361 | 6.376 |
| v2 | 10,000 | 5.380 | 5.367 |
| v3 | 22,800 | 4.788 | 4.759 |

v3 on the independent set: perplexity 120.1 (English loss 4.740, perplexity
114.4; Indonesian loss 4.857, perplexity 128.6). The independent set was
evaluated with window stride 4 (1,893,120 of 7,571,884 tokens). English
independence is at source-file level only; near-duplicates were not checked.
The earlier holdout shares source files with the training data.

No task-level benchmark, factuality, safety or memorization evaluation has
been run.

## Limitations and risks

- Generated text is not fluent: informal inspection of a few samples showed
  repetition, invented words and loops under greedy decoding. This is an
  anecdote, not a measurement.
- Training data is web text that may contain personal information,
  copyrighted material and bias. Memorization of such data was not tested.
- No instruction tuning, safety tuning or output filtering.
- Provenance note: the `catatan` field in the token metadata files says
  "trial data, not final". It is stale; the shards contain the full sample
  described above. The files are left unchanged because their hashes are
  bound to the checkpoint.

## License

Code: Apache-2.0. Weights: **UNDECIDED**, and not distributed. This will be
decided after the dataset licenses, distribution terms and the implications
for derived weights are reviewed.
