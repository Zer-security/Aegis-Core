# Aegis-Core

**A small language-model project, built in the open.**

Aegis-Core is an early-stage effort to build and understand a language model from its foundations. It starts with **Aegis-Nano**, a decoder-only Transformer implemented in PyTorch. The work is deliberately incremental: establish a sound model, test its behaviour, build the tokenizer and data pipeline, then train and evaluate it with results that can be checked.

This is a work in progress. The repository is public so people can follow the work and review it while it develops—not because the model is finished.

## Current state

| Component | Status |
| --- | --- |
| Aegis-Nano model implementation | Present |
| Model unit tests | Present; 25 tests passed in the reported local run |
| Data-sampling script | Experimental; see [DATA.md](DATA.md) |
| Tokenizer | Implemented (byte-level BPE, 16,384 vocabulary); see `tokenizer/` |
| Training loop and checkpoint workflow | Implemented (CPU, resumable); see `scripts/train.py` |
| Evaluation | Perplexity evaluator on held-out data (`scripts/evaluate_lm.py`); no task benchmarks yet |
| Trained model checkpoint | Not included |

The default configuration has about 30 million parameters because its vocabulary size (50,304) is a placeholder. Training runs use a 16,384-token vocabulary (17,011,584 parameters). Models trained so far are small, CPU-trained and not fluent. No claim is made about language-model quality: loss values alone are not evidence of capability. Trained checkpoints are not included.

## Model at a glance

Aegis-Nano currently uses:

- A decoder-only, pre-LayerNorm Transformer.
- Causal self-attention through PyTorch's scaled dot-product attention.
- A GELU feed-forward block with a 4× expansion.
- Learned token and position embeddings.
- Tied token-embedding and output-projection weights.
- GPT-2-style residual projection initialization.

Default configuration:

| Setting | Value |
| --- | ---: |
| Layers | 6 |
| Attention heads | 6 |
| Embedding width | 384 |
| Context length | 256 tokens |
| Vocabulary size (placeholder) | 50,304 |
| Parameters | 30,036,864 |

These values describe the implementation, not a measure of capability. The vocabulary setting is a placeholder default; training runs use the 16,384-token vocabulary of the project tokenizer.

## Install

~~~bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
~~~

`requirements.txt` covers the model, training, evaluation and tokenizer; `requirements-data.txt` adds the data-sampling scripts; `requirements-dev.txt` adds pytest. A clean install from `requirements-dev.txt` was checked in a fresh virtual environment (Python 3.14.4, CPU-only PyTorch); the test suite was only collected there, not run.

## Run the model tests

Use an environment with Python, PyTorch and pytest installed. From the repository root, run:

~~~bash
python -m pytest tests/test_model.py -v
~~~

The current test suite checks forward-pass shapes, causal masking, loss calculation, ignored targets, parameter count, tied weights, seeded initialization, checkpoint round-tripping, a small overfit test and generation behaviour. It also contains characterization tests that document known limitations; a passing characterization test does **not** mean the behaviour is desirable.

**Development result:** the reported local run completed with **25 passed** using Python 3.14.4, PyTorch 2.14.1+cpu and pytest 9.2.0. This is a local result, not a published CI guarantee. Tests use small synthetic inputs; they do not validate training quality or model performance.

## Data and reproducibility

The experimental sampler in **scripts/sample_corpus.py** plans a small English/Indonesian sample using pinned revisions of FineWeb-Edu and FineWeb-2. Its design includes traffic limits and a separate holdout rule. Read [DATA.md](DATA.md) for the source references, provenance, sampling design, privacy considerations and known dataset limitations before using the script.

Raw sample data is not included in this repository. The **data/** directory is ignored by Git. A plan-only run can still contact the source to inspect metadata; downloading documents requires the script's explicit run mode.

The sampler is still experimental. Its presence should not be taken as a claim that the full data pipeline has been independently validated.

## Development priorities

The next substantial milestones are:

1. Add a model card and threat model (privacy, memorization, harmful output) before any weights are published.
2. Run controlled experiments (more data, learning-rate search) and publish the results, including failures and limitations.
3. Add task-level evaluation beyond perplexity.

Changes should be small enough to review and accompanied by tests or evidence where practical. Loss curves, parameter counts and successful execution alone are not proof that a model has become more capable.

## Contributing and discussion

Aegis-Core is at an early stage, and constructive review is welcome. Issues that identify a reproducible bug, a missing test, a data-provenance concern or a concrete improvement are especially useful. Please include the relevant environment and reproduction steps; avoid posting private data or credentials.

## License

The source code is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE). This covers source code only. Model weights and dataset-derived artifacts are not covered, and no weights are distributed from this repository. Training data is subject to its own terms (see [DATA.md](DATA.md)).

---

Aegis-Core is being developed in public, one verifiable step at a time. The goal is to make the engineering and evidence visible as the model grows—not to claim results that have not been demonstrated.
