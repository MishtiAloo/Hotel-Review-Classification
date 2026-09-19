# HRAST BERT Project Plan

## 1. Goal and scope

Build a small, reproducible NLP project with two independent `bert-base-uncased`
models:

1. **Sentiment classifier** — predicts exactly one sentence-level label:
   `positive`, `negative`, or `neutral`.
2. **Aspect classifier** — predicts zero or more of the 21 hotel aspects in the
   dataset.
3. **Flask inference interface** — accepts one hotel-review sentence and displays
   the sentiment, its confidence, and all detected aspects with their scores.

This project is **sentence-level sentiment classification plus multi-label aspect
detection**. It is not full aspect-based sentiment analysis: for a sentence such as
“Breakfast was poor but the staff were excellent,” the system may detect both
`Breakfast` and `Staff`, but it produces only one overall sentiment and cannot assign
a separate sentiment to each aspect.

Keep the first version deliberately simple: two separately fine-tuned BERT models,
one local Flask app, no database, authentication, cloud deployment, model serving
framework, or JavaScript frontend framework.

## Mandatory coding rules (must follow)

These rules apply to every Python, HTML, CSS, configuration, and test file in the
project:

1. Keep the code as simple and direct as possible.
2. Do not introduce complex processing steps, unnecessary abstractions, design
   patterns, helper layers, or extra dependencies.
3. Avoid complex Python syntax. Prefer ordinary loops, clear `if` statements,
   small named functions, and readable intermediate variables over clever one-liners,
   deeply nested comprehensions, decorators, metaprogramming, or advanced language
   features.
4. Each function should do one clear job and have a descriptive name.
5. Add useful comments that explain the purpose of each important step, especially
   data cleaning, label conversion, loss selection, thresholding, and inference.
6. Comments should explain **why** a step is needed when that is not obvious. Do not
   clutter the code with comments that merely repeat a self-explanatory line.
7. Use docstrings for modules and non-trivial functions, written in plain language.
8. Prefer a little repetition over an abstraction that makes the code harder for a
   student or evaluator to follow.
9. Every implementation review must reject code that is unnecessarily complicated,
   insufficiently commented, or difficult to explain during the lab presentation.

## 2. Dataset facts and audit findings

Source: `dataset/HRAST.csv` (23,113 rows).

### Target columns

- Sentiment: `positive`, `negative`, `neutral`
- Aspects (21): `Clean`, `Comfort`, `Facilities/Amenities`, `Location`,
  `Restaurant (dinner)`, `Staff`, `View (Balcony)`, `Breakfast`, `Room`, `Pool`,
  `Beach`, `Bathroom/Shower (toilet)`, `Bar`, `Bed`, `Parking`, `Noise`,
  `Reception-checkin`, `Lift`, `Value for money`, `Wi-Fi`, `Generic`

### Observed distribution in the raw file

- Sentiment positives: 11,819
- Sentiment negatives: 10,504
- Sentiment neutrals: 794
- Most frequent aspects: `Room` 5,004; `Location` 4,729; `Staff` 3,804
- Rarest aspects: `Beach` 188; `Wi-Fi` 268; `Lift` 291;
  `Restaurant (dinner)` 300; `Bar` 322
- Mean aspects per sentence: about 1.60
- Review length: mean 10.3 words, 95th percentile 24, maximum 123

These figures justify `max_length=128`, macro-averaged metrics, and explicit handling
of class imbalance.

### Data problems that must be handled

- Drop the unnamed empty CSV column between `review` and `positive`.
- There are four rows with more than one active sentiment label: IDs `4557`,
  `11602`, `15917`, and `18969`. Quarantine them rather than guessing a label.
- Row `4634` contains `σ` in the binary `Location` field. Quarantine it rather than
  silently coercing it.
- After whitespace normalization, matching duplicate groups account for 11 extra
  rows and should be reduced to one row per group. The duplicate text at IDs
  `13869` and `13870` has conflicting aspect labels; quarantine both rows.
- The final `Aspect` column must **not** be treated as a 22nd aspect or as a reliable
  “has any aspect” target. Its values disagree with the OR of the 21 topic columns
  on 20,233 rows. Retain it only in the audit output and exclude it from training
  until its true semantics are established.
- Never overwrite the source CSV. Produce a cleaned file and an audit report under
  `data/processed/`. With the rules above, the expected cleaned row count is 23,095;
  the preparation script must assert this count or stop with a useful error.

## 3. Proposed repository layout

```text
proj/
├── dataset/
│   ├── HRAST.csv                 # immutable input
│   └── README.md
├── data/
│   └── processed/
│       ├── hrast_clean.csv
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── audit.json
├── configs/
│   └── train.yaml                # shared seeds and hyperparameters
├── src/
│   ├── data/
│   │   ├── prepare.py
│   │   └── labels.py             # canonical label order and display names
│   ├── training/
│   │   ├── train_sentiment.py
│   │   ├── train_aspects.py
│   │   └── metrics.py
│   ├── inference/
│   │   └── predictor.py          # shared by CLI, tests, and Flask
│   └── web/
│       ├── app.py
│       ├── templates/index.html
│       └── static/style.css
├── artifacts/
│   ├── sentiment/                # model, tokenizer, label mapping, metrics
│   └── aspects/                  # model, tokenizer, thresholds, metrics
├── tests/
│   ├── test_prepare.py
│   ├── test_predictor.py
│   └── test_web.py
├── requirements.txt
├── README.md
└── plan.md
```

Generated datasets and checkpoints should normally be excluded from Git, except for
small JSON metadata/metric files if they are useful for the lab submission.

## 4. Environment

Use Python 3.11 and a CUDA-enabled PyTorch build compatible with the installed
NVIDIA driver. Core dependencies:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `scikit-learn`
- `iterative-stratification`
- `pandas`, `numpy`, `pyyaml`
- `flask`
- `pytest`

Before training, record the Python, PyTorch, CUDA, GPU, and package versions and run
a short CUDA smoke test. Pin working versions in `requirements.txt` so the result can
be reproduced.

## 5. Data preparation and splitting

Implement one deterministic preparation command with seed `42`.

1. Load the CSV with UTF-8 handling and remove the unnamed column.
2. Preserve the original `id`; normalize only leading/trailing and repeated
   whitespace in `review`.
3. Do **not** remove punctuation, stopwords, or casing, and do not stem or lemmatize.
4. Validate that every sentiment/aspect cell is exactly integer `0` or `1`.
5. Validate that every retained row has exactly one sentiment label.
6. Quarantine malformed rows and record their IDs and reasons in `audit.json`.
7. Deduplicate normalized review text before splitting. Keep one copy of a group
   only when all model labels agree; quarantine every member of a conflicting group.
8. Create a single shared 70/15/15 train/validation/test split for both tasks.
9. Use iterative multi-label stratification over a combined matrix of the 21 aspect
   labels and the three one-hot sentiment labels. This keeps rare aspects and the
   neutral class represented while ensuring both models use identical examples.
10. Assert that normalized review texts do not overlap between splits, and write
    split-level sentiment/aspect counts to the audit report.

Only training-split statistics may determine class weights or decision thresholds.
The test set remains untouched until final evaluation.

## 6. Tokenization

Use `AutoTokenizer.from_pretrained("bert-base-uncased")` for both tasks.

- `max_length: 128`
- truncation enabled
- dynamic batch padding with `DataCollatorWithPadding`
- no fixed padding during preprocessing
- retain raw text in the split CSVs for error analysis

Log the number of examples truncated at 128 tokens. A nonzero count is acceptable,
but it should be reported.

## 7. Sentiment model

Use `AutoModelForSequenceClassification` with three output labels and an explicit,
saved mapping:

```text
0 -> negative
1 -> neutral
2 -> positive
```

Training objective: multiclass cross-entropy. Because neutral is much rarer, train
the initial baseline without weights, inspect per-class recall/macro-F1, and run one
weighted-loss experiment if neutral performance is poor. Any class weights must be
calculated from the training split only.

Primary model-selection metric: validation macro-F1.

Report:

- accuracy and macro/weighted F1
- per-class precision, recall, F1, and support
- confusion matrix
- final test loss

## 8. Aspect model

Use a separate `AutoModelForSequenceClassification` with 21 outputs and
`problem_type="multi_label_classification"`. The canonical output order must live in
one shared labels module and be saved with the checkpoint.

Training objective: `BCEWithLogitsLoss`. Establish an unweighted baseline first.
Because aspect prevalence ranges from 188 to 5,004 positives, also evaluate a
training-only `pos_weight = negatives / positives` configuration if rare-label
recall is weak. If raw weights make training unstable, cap the weights (for example,
at 20) and document that choice; do not tune this using the test set.

Tune sigmoid decision thresholds on the validation split:

1. Start with a global threshold of `0.5` as the baseline.
2. Search a small fixed grid (for example `0.20` to `0.80` in steps of `0.05`).
3. Prefer one global threshold for simplicity; use per-label thresholds only if they
   produce a meaningful macro-F1 improvement.
4. Save the chosen threshold(s) in `artifacts/aspects/thresholds.json` and load them
   during inference.

Primary model-selection metric: validation macro-F1.

Report:

- micro-F1 and macro-F1
- per-aspect precision, recall, F1, and support
- micro/macro average precision (PR-AUC)
- Hamming loss
- exact-match accuracy as a secondary, deliberately strict metric
- results at both threshold `0.5` and the validation-tuned threshold(s)

## 9. RTX 3090 training configuration

Train the models sequentially, not at the same time. A practical starting
configuration for the 24 GB RTX 3090 is:

```yaml
base_model: bert-base-uncased
max_length: 128
seed: 42
epochs: 5
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.10
train_batch_size: 32
eval_batch_size: 64
gradient_accumulation_steps: 1
fp16: true
gradient_checkpointing: false
max_grad_norm: 1.0
early_stopping_patience: 2
save_total_limit: 2
```

Start with batch size 32 for reliability. After a one-batch smoke test, increase it
to 64 only if memory usage is comfortably below the limit. Dynamic padding should
make this likely, but the plan must not depend on it. Use automatic mixed precision;
do not enable gradient checkpointing unless memory measurements show it is needed.

For each model:

1. Run a tiny overfit/smoke test on 32–64 examples.
2. Run the full baseline and select the best checkpoint by validation macro-F1.
3. Run at most one justified imbalance-handling comparison.
4. Evaluate the selected configuration once on the test split.
5. Save model/tokenizer with `save_pretrained`, plus metrics, config, label mappings,
   thresholds, data-audit hash, and seed.

Do not load both models onto the GPU during training. For inference both will fit,
but the predictor should support `DEVICE=cpu`, `DEVICE=cuda`, or automatic choice.

## 10. Shared inference layer

Create one `Predictor` class that loads both checkpoints once at application startup.
It must:

- set both models to evaluation mode;
- use `torch.inference_mode()`;
- apply exactly the same tokenizer settings as training;
- use softmax for sentiment and sigmoid for aspects;
- apply saved aspect thresholds rather than a hard-coded web-only value;
- return plain Python values suitable for JSON serialization;
- preserve stable label ordering from checkpoint metadata.

Suggested response shape:

```json
{
  "text": "The room was clean but the Wi-Fi was terrible.",
  "sentiment": {
    "label": "negative",
    "confidence": 0.81,
    "scores": {"negative": 0.81, "neutral": 0.03, "positive": 0.16}
  },
  "aspects": [
    {"label": "Clean", "score": 0.91},
    {"label": "Room", "score": 0.87},
    {"label": "Wi-Fi", "score": 0.84}
  ]
}
```

The CLI and Flask app must call this class rather than reimplementing inference.

## 11. Flask interface

Keep the interface server-rendered and minimal.

### Routes

- `GET /` — render a form with one review textarea and an example sentence.
- `POST /` — validate input, run inference, and render the result on the same page.
- `POST /api/predict` — optional JSON form of the same operation for testing/demo use.
- `GET /health` — report whether both models loaded successfully.

### Result display

- Overall sentiment label, confidence, and three class probabilities.
- Detected aspects sorted by score, shown as simple badges/bars.
- A clear “No aspect passed the configured threshold” message when appropriate.
- A short note that sentiment is for the whole sentence, not separately for each
  detected aspect.

### Safety and usability

- Reject empty/whitespace input and cap input length (for example, 2,000 characters).
- Let Jinja auto-escape user text; never render it with `|safe`.
- Show friendly validation/model errors without a stack trace in the page.
- Load models once, not per request.
- Avoid Flask's development reloader when using CUDA because it can load the models
  twice. Bind to localhost by default; the development server is sufficient for the
  lab demo.
- Use a clean responsive CSS layout, but do not introduce Bootstrap or a build step.

## 12. Verification

### Automated tests

- Data validation rejects non-binary labels and non-exclusive sentiment rows.
- Deduplication keeps matching annotations and quarantines conflicts.
- Split files have no review-text overlap and contain all labels.
- Label index/name round trips are stable.
- Predictor outputs three sentiment probabilities summing approximately to one.
- Aspect scores are in `[0, 1]` and saved thresholds are applied.
- Flask accepts a valid request, rejects blank/oversized input, and returns the
  documented JSON structure.

Mock the predictor in web tests so routine tests do not require loading BERT. Keep
one optional CUDA integration test for the real artifacts.

### Manual acceptance checks

Try at least these cases and record the outputs in the report:

1. `The staff were friendly and the room was spotless.`
2. `The Wi-Fi was slow and the room was noisy.`
3. `Breakfast was poor but the staff were excellent.`
4. `The hotel is located three kilometres from the airport.`

The third case should be used to demonstrate the system's sentence-level sentiment
limitation rather than presented as aspect-specific sentiment.

## 13. Implementation order

- [ ] Create environment, pinned requirements, and CUDA smoke test.
- [ ] Implement canonical labels and deterministic data audit/cleaning.
- [ ] Generate and verify the shared 70/15/15 split.
- [ ] Train/evaluate the sentiment baseline and one weighted comparison if needed.
- [ ] Train/evaluate the aspect baseline and one weighted comparison if needed.
- [ ] Tune aspect threshold(s) using validation data and freeze final artifacts.
- [ ] Implement the shared predictor and a small command-line smoke test.
- [ ] Implement the Flask page and optional JSON endpoint.
- [ ] Add automated tests and run a real two-model GPU inference check.
- [ ] Write the final README/report with dataset audit, hyperparameters, metrics,
      confusion matrix, per-aspect results, limitations, and launch commands.

## 14. Definition of done

The project is complete when a fresh environment can reproduce preprocessing and
training from documented commands; both held-out test reports are saved; the Flask
app loads the frozen artifacts and returns correct, thresholded predictions; tests
pass; and the report accurately describes the task as overall sentence sentiment
plus multi-label aspect detection. All submitted code must also satisfy the mandatory
coding rules: it must remain simple, use straightforward Python syntax, and include
clear comments and docstrings for important logic.
