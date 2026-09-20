# Testing the trained models (inference guide)

This guide shows how to check that the two models trained in
`notebooks/kaggle_train_manual.ipynb` work on your computer: first in the
terminal, then in the web page.

Everything below was run on the extracted models in `hotel_models/`, and the
expected outputs shown are the real results.

---

## 0. What you should have

After unzipping `hotel_models.zip` you have this layout:

```text
hotel_models/
└── artifacts/
    ├── sentiment/   config.json, model.safetensors, tokenizer files, metrics.json
    └── aspects/     config.json, model.safetensors, tokenizer files, metrics.json, thresholds.json
```

Check that `model.safetensors` is about **440 MB** in both folders. If it is
much smaller, the download was cut off and you must download the zip again.

The project code looks for the models in `artifacts/sentiment` and
`artifacts/aspects`. You have two ways to point it at your folders. Pick one.

**Option A (no moving files).** Tell the code where the models are. Used in the
commands below.

**Option B (move once).** Move the two folders so the defaults just work, then
you can drop all the path options below:

```powershell
Move-Item hotel_models\artifacts\sentiment artifacts\sentiment
Move-Item hotel_models\artifacts\aspects   artifacts\aspects
```

---

## 1. Install the requirements (once)

Use Python 3.11 or newer, from the project root:

```powershell
pip install torch transformers flask numpy pandas pytest
```

(`pip install -r requirements.txt` also works, but it asks for a CUDA build of
PyTorch. The line above is enough to run inference on a normal CPU.)

---

## 2. Test 1: command line (fastest check)

From the project root:

```powershell
python src/inference/cli.py --sentiment-model-dir hotel_models/artifacts/sentiment --aspect-model-dir hotel_models/artifacts/aspects --device cpu
```

It loads both models and predicts four example sentences. **Expected result**
(numbers can differ in the last decimal):

| Sentence | Sentiment | Aspects found |
|---|---|---|
| The staff were friendly and the room was spotless. | positive (0.9999) | Staff 0.96, Clean 0.95, Room 0.37 |
| The Wi-Fi was slow and the room was noisy. | negative (0.9999) | Wi-Fi 0.93, Noise 0.77 |
| Breakfast was poor but the staff were excellent. | positive (0.77) | Staff 0.98, Breakfast 0.98 |
| The hotel is located three kilometres from the airport. | positive (0.85) | Location 0.99 |

Test your own sentence:

```powershell
python src/inference/cli.py --sentiment-model-dir hotel_models/artifacts/sentiment --aspect-model-dir hotel_models/artifacts/aspects --device cpu --text "The bed was uncomfortable and parking was expensive."
```

**Test 1 passes if:** no error appears, sentiment names are real words
(`negative` / `neutral` / `positive`, not `LABEL_0`), and the aspects make sense
for the sentence.

---

## 3. Test 2: the web page

Set the model folders, then start the app (PowerShell):

```powershell
$env:SENTIMENT_MODEL_DIR = "hotel_models/artifacts/sentiment"
$env:ASPECT_MODEL_DIR    = "hotel_models/artifacts/aspects"
$env:DEVICE              = "cpu"
python src/web/app.py
```

Leave that window open, then in your browser go to:

- <http://127.0.0.1:5000/health> should show `{"status":"ok"}`. If it shows
  `error`, the message says what failed (usually a wrong folder path).
- <http://127.0.0.1:5000/> is the review page. Type a review, press submit, and
  you should see the sentiment and the detected aspects.

Stop the app with `Ctrl+C`.

### Test the JSON endpoint (optional)

With the app running, open a second PowerShell window:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/predict -Method Post -ContentType "application/json" -Body '{"review": "The pool was dirty and the lift was broken."}' | ConvertTo-Json -Depth 5
```

**Expected:** sentiment `negative`, and `Lift` (about 0.88) among the aspects.
An empty review (`{"review": "   "}`) must return an error (HTTP 400).

---

## 4. Test 3: does it score as well as it did in training?

The notebook measured the models on the held-out test set and saved the results.
Read them:

```powershell
python -c "import json; d=json.load(open('hotel_models/artifacts/sentiment/metrics.json')); print('accuracy', d['accuracy'], 'macro_f1', d['macro_f1'])"
python -c "import json; d=json.load(open('hotel_models/artifacts/aspects/metrics.json')); print('threshold', d['threshold'], 'macro_f1', d['macro_f1'], 'micro_f1', d['micro_f1'])"
```

Your models' test scores:

| Model | Accuracy | Macro-F1 | Micro-F1 |
|---|---|---|---|
| Sentiment | 0.939 | 0.726 | - |
| Aspects (threshold 0.3) | - | 0.883 | 0.894 |

Sentiment macro-F1 is lower than its accuracy because the `neutral` class has
very few sentences, so the model is weaker on it. Look at
`plots/sentiment_confusion_matrix.png` from the notebook output to see this.

---

## 5. Test 4: sentences that should be easy or hard

Use these to see the behaviour for yourself. Run each with `--text`.

| Sentence | What to expect |
|---|---|
| The breakfast was delicious and the staff were so kind. | positive (0.9999), Staff 0.99, Breakfast 0.97 |
| The room was dirty and the bathroom smelled. | negative (0.9998), Bathroom 0.97, Clean 0.89, Room 0.52 |
| The wifi kept dropping. | negative (0.9997), Wi-Fi 0.93 |
| There is a lift to the third floor. | neutral (0.96), Lift 0.93 |
| Great location, but very noisy at night. | neutral (0.96), Location 0.99, Noise 0.96 |
| The bed was uncomfortable and parking was expensive. | negative (0.9999), Parking 0.98, Comfort 0.93, Value for money 0.57, Bed 0.42 |

The model gives **one overall sentiment per sentence** (not one per aspect). A
sentence with good and bad parts ("Great location, but very noisy") therefore
gets a single label, here `neutral`. That is expected for this project.

---

## 6. Test 5: the automatic test suite

```powershell
python -m pytest -q
```

**Expected:** `19 passed`. The prediction test uses `artifacts/sentiment` and
`artifacts/aspects` if they exist (so after Option B it tests your real models).
Otherwise it falls back to the small `artifacts/smoke_*` checkpoints.

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Model checkpoints not found` on the page or `/health` | The environment variables are not set in this window, or the paths are wrong. Set them again (section 3), or use Option B. |
| Labels look like `LABEL_0`, `LABEL_1` | The models were not saved by the new notebook. Re-run the notebook and download the zip again. |
| Weaker aspects (e.g. Room, Bed) are missing from results | `thresholds.json` is missing from `aspects/`, so the app falls back to 0.5 instead of the tuned 0.3. Check the file exists. |
| `OSError` / `safetensors` error while loading | The `model.safetensors` file is incomplete. Download the zip again. |
| First request is slow | Normal. Models load once at start; on CPU each prediction takes a fraction of a second. |
| `ModuleNotFoundError: src` | Run commands from the project root (`Hotel-Review-Classification`), not from inside `src`. |
