# Hotel Review Classification (HRAST BERT)

Two independently fine-tuned `bert-base-uncased` models over hotel-review
sentences from the HRAST dataset, served through a small Flask page:

1. **Sentiment classifier** — one label per sentence: `negative`, `neutral`,
   or `positive`.
2. **Aspect classifier** — zero or more of 21 hotel aspects (`Staff`,
   `Breakfast`, `Wi-Fi`, ...), using a tuned sigmoid threshold per label.

This is sentence-level sentiment plus multi-label aspect detection, **not**
full aspect-based sentiment analysis: a sentence can be tagged with several
aspects, but it always gets one overall sentiment. See [plan.md](plan.md) for
the full design rationale and the mandatory coding rules every file in this
project follows (simple control flow, small named functions, comments that
explain *why*, no unnecessary abstractions).

## Repository layout

```text
dataset/HRAST.csv            immutable input data
data/processed/               cleaned data, splits, audit.json
configs/train.yaml             shared training hyperparameters
src/data/                      labels.py, prepare.py (audit + cleaning + split)
src/training/                  train_sentiment.py, train_aspects.py, tune_thresholds.py
src/inference/                 predictor.py (shared by CLI and Flask), cli.py
src/web/                       app.py, templates/, static/
artifacts/                     trained checkpoints + metrics (not committed)
tests/                         pytest suite
```

## 1. Environment

Requires Python 3.11+ and, for real training, a CUDA-capable GPU (the plan
targets an RTX 3090). Install pinned dependencies:

```bash
pip install -r requirements.txt
```

Then record the environment and confirm CUDA is visible:

```bash
python scripts/check_environment.py
```

`src/training/common.py` also lets every script fall back to CPU
(`--device cpu`), which is enough to smoke-test the pipeline but far too slow
for a full 5-epoch run over ~16k training sentences.

## 2. Data preparation

Already run once; the outputs are committed under `data/processed/`. To
reproduce it from scratch:

```bash
cd src/data
python prepare.py --input ../../dataset/HRAST.csv --output-dir ../../data/processed
```

This validates every sentiment/aspect cell, quarantines malformed or
conflicting rows (with reasons recorded in `audit.json`), deduplicates
normalized review text, and writes a shared 70/15/15 multilabel-stratified
split. The script asserts the cleaned row count is exactly 23,095 and stops
with an error otherwise.

Key audit numbers from the current `data/processed/audit.json`:

- Raw rows: 23,113 → clean rows: 23,095
- 5 rows quarantined for malformed labels (2 with more than one sentiment,
  1 with a non-binary `σ` value, 2 from a conflicting duplicate pair)
- 11 duplicate-text extras collapsed into one row each
- Split sizes: train 16,159 / validation 3,467 / test 3,469, each keeping the
  same positive/negative/neutral ratio (roughly 51% / 45% / 3%)

## 3. Training

Train each model separately (never both on the GPU at once):

```bash
python src/training/train_sentiment.py --loss baseline
python src/training/train_aspects.py --loss baseline
```

**No local GPU?** Use the two self-contained notebooks under `notebooks/`
(no dependency on this repo's `src/` code — plain, single-purpose cells):

- [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) — cleans the
  data, trains both models, tunes the aspect threshold, and tries a few
  example sentences. Upload `dataset/HRAST.csv` as a Kaggle Dataset, attach
  it, turn on GPU + internet, and run top to bottom.
- [notebooks/kaggle_infer.ipynb](notebooks/kaggle_infer.ipynb) — a separate,
  much smaller notebook for later sessions: attach the first notebook's
  saved Output as an input dataset, point it at the `artifacts/sentiment`
  and `artifacts/aspects` folders, and run inference without retraining.

Add `--smoke` first to run a 1-epoch, ~64-example pass and confirm the
pipeline works before committing GPU time to a full run. If per-class recall
or rare-aspect recall is weak on the baseline, run the weighted comparison
(`--loss weighted`) once, compare `metrics.json` for both runs, and keep the
run with the better validation macro-F1.

Each run saves the best checkpoint (by validation macro-F1), its tokenizer,
and a `metrics.json` with per-epoch history, final test metrics, the data
audit hash, and the seed. Once you have picked the winning run for each task,
copy (or re-run directly into) the final locations the predictor expects:

```text
artifacts/sentiment/   <- winning sentiment checkpoint
artifacts/aspects/     <- winning aspect checkpoint
```

Then tune the aspect decision threshold on validation data only:

```bash
python src/training/tune_thresholds.py --model-dir artifacts/aspects
```

This searches a `0.20`–`0.80` grid in steps of `0.05`, compares the best
result against the plain `0.5` baseline, and only adopts the grid winner if
it meaningfully improves validation macro-F1. The result is saved to
`artifacts/aspects/thresholds.json`, which the predictor loads automatically.

## 4. Running inference

Command-line smoke test over the four manual-acceptance sentences from the
plan (or a custom `--text`):

```bash
python src/inference/cli.py
```

Flask app:

```bash
python -m flask --app src.web.app run
# or: python src/web/app.py
```

Then open `http://127.0.0.1:5000/`. Routes:

- `GET /` — the review form.
- `POST /` — validates input, runs inference, renders the result.
- `POST /api/predict` — same operation as JSON (`{"review": "..."}`).
- `GET /health` — `200` with `{"status": "ok"}` once both models are loaded,
  `503` with an error detail otherwise.

Model locations and device are configurable via environment variables:
`SENTIMENT_MODEL_DIR`, `ASPECT_MODEL_DIR` (defaults: `artifacts/sentiment`,
`artifacts/aspects`), and `DEVICE` (`auto`, `cpu`, or `cuda`).

## 5. Tests

```bash
pytest
```

The data-preparation and label tests are pure Python and always run. The
predictor's integration test and the CLI both look for trained checkpoints
under `artifacts/`; the Flask tests mock the predictor entirely, so the full
suite runs without loading BERT once at least the smoke-test checkpoints
exist. Run the real inference check once both final checkpoints are trained:

```bash
python src/inference/cli.py --sentiment-model-dir artifacts/sentiment --aspect-model-dir artifacts/aspects
```

## 6. Known limitation

For a sentence like *"Breakfast was poor but the staff were excellent,"* the
aspect model can correctly flag both `Breakfast` and `Staff`, but the
sentiment model still produces a single sentence-level sentiment — it cannot
say the sentiment was negative for breakfast and positive for staff
separately. The Flask page states this explicitly next to every result.

## 7. Status

- [x] Environment, pinned requirements, CUDA smoke-test script
- [x] Canonical labels and deterministic data audit/cleaning
- [x] Shared 70/15/15 split, verified for overlap
- [x] Sentiment and aspect training scripts, smoke-tested end to end on CPU
- [x] Threshold-tuning script for the aspect model
- [x] Shared `Predictor` class and CLI smoke test
- [x] Flask page and `/api/predict` JSON endpoint
- [x] Automated tests (data validation, labels, predictor, mocked Flask)
- [ ] Full GPU training runs and final frozen artifacts under `artifacts/`
      (this repository was completed in an environment with no CUDA GPU;
      run the commands in section 3 on the target RTX 3090 machine to
      produce the real checkpoints, then re-run `tune_thresholds.py` and the
      real inference check)
- [ ] Final metrics/confusion-matrix write-up once the runs above exist
