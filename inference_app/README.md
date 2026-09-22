# inference_app

A small web page that uses the two trained models (sentiment + aspects).

## Run it

```powershell
python inference_app/run.py
```

or double-click `inference_app/run.bat` (it installs Streamlit first if needed).
Your browser opens at <http://localhost:8501>. Stop with `Ctrl+C`.

## Where the models must be

`app.py` looks in these places, in order, for a folder that holds
`sentiment/` and `aspects/`:

1. the folder named in the environment variable `HOTEL_MODELS_DIR`
2. `artifacts/` (project root)
3. `hotel_models/artifacts/` (what you get by unzipping `hotel_models.zip` in the project root)

If you unzipped in the project root, nothing else is needed.

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit page (finds models, loads them, predicts, shows results) |
| `run.py` | starts the page |
| `run.bat` | same, by double-click on Windows |
| `requirements.txt` | packages needed |

## About the model files and GitHub

The trained weights (`model.safetensors`, about 440 MB each) are **not stored in
this repository**, because GitHub rejects files over 100 MB. After cloning:

1. Get `hotel_models.zip` (the output of `notebooks/kaggle_train_manual.ipynb`)
   from whoever trained it (for example a GitHub Release, Google Drive or Kaggle link).
2. Unzip it in the project root, or put the `sentiment` and `aspects` folders in
   `inference_app/models/`.
3. Run `python inference_app/run.py`.
