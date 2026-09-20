"""Hotel review web page (Streamlit).

Type a hotel review sentence and this page shows:
  1. its sentiment (negative / neutral / positive)
  2. which hotel aspects it mentions (Staff, Breakfast, Wi-Fi, ...)

Start it with:  python run.py   (or double-click run.bat)
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 128

EXAMPLE_SENTENCES = [
    "The staff were friendly and the room was spotless.",
    "The Wi-Fi was slow and the room was noisy.",
    "Breakfast was poor but the staff were excellent.",
    "The hotel is located three kilometres from the airport.",
    "The pool was dirty and the lift was broken.",
]

THIS_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = THIS_FOLDER.parent


# ---------------------------------------------------------------- finding the models

def find_models_folder():
    """Return the folder that contains the 'sentiment' and 'aspects' model folders."""
    places_to_look = []

    # 1. A folder you name yourself (set HOTEL_MODELS_DIR before running).
    if os.environ.get("HOTEL_MODELS_DIR"):
        places_to_look.append(Path(os.environ["HOTEL_MODELS_DIR"]))

    # 2. inference_app/models, 3. artifacts, 4. the folder made by unzipping hotel_models.zip.
    places_to_look.append(THIS_FOLDER / "models")
    places_to_look.append(PROJECT_FOLDER / "artifacts")
    places_to_look.append(PROJECT_FOLDER / "hotel_models" / "artifacts")

    for place in places_to_look:
        has_sentiment = (place / "sentiment" / "config.json").exists()
        has_aspects = (place / "aspects" / "config.json").exists()
        if has_sentiment and has_aspects:
            return place

    return None


# ---------------------------------------------------------------- loading (done once)

@st.cache_resource
def load_everything(models_folder):
    """Load both models, their tokenizers, and the aspect threshold. Cached between clicks."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sentiment_folder = Path(models_folder) / "sentiment"
    aspect_folder = Path(models_folder) / "aspects"

    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_folder)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_folder)
    sentiment_model.to(device)
    sentiment_model.eval()

    aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_folder)
    aspect_model = AutoModelForSequenceClassification.from_pretrained(aspect_folder)
    aspect_model.to(device)
    aspect_model.eval()

    # The threshold picked on the validation set during training (0.5 if the file is missing).
    threshold = 0.5
    threshold_file = aspect_folder / "thresholds.json"
    if threshold_file.exists():
        with open(threshold_file) as f:
            threshold = json.load(f).get("global_threshold", 0.5)

    return {
        "device": device,
        "sentiment_tokenizer": sentiment_tokenizer,
        "sentiment_model": sentiment_model,
        "aspect_tokenizer": aspect_tokenizer,
        "aspect_model": aspect_model,
        "default_threshold": threshold,
    }


# ---------------------------------------------------------------- predicting

def run_model(tokenizer, model, device, text):
    """Tokenize one text, run the model, and return the raw scores (logits)."""
    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    return output.logits[0].cpu()


def predict_sentiment(models, text):
    """Return (best label, {label: probability})."""
    logits = run_model(models["sentiment_tokenizer"], models["sentiment_model"], models["device"], text)
    probabilities = torch.softmax(logits, dim=0).tolist()   # softmax: the 3 numbers add up to 1

    label_names = models["sentiment_model"].config.id2label
    scores = {}
    for i in range(len(probabilities)):
        scores[label_names[i]] = probabilities[i]

    best_label = max(scores, key=scores.get)
    return best_label, scores


def predict_aspects(models, text):
    """Return {aspect: probability} for all aspects."""
    logits = run_model(models["aspect_tokenizer"], models["aspect_model"], models["device"], text)
    probabilities = torch.sigmoid(logits).tolist()           # sigmoid: each aspect is its own yes/no

    label_names = models["aspect_model"].config.id2label
    scores = {}
    for i in range(len(probabilities)):
        scores[label_names[i]] = probabilities[i]
    return scores


# ---------------------------------------------------------------- the page

st.set_page_config(page_title="Hotel Review Analyzer", page_icon="🏨")
st.title("🏨 Hotel Review Analyzer")
st.write("Type one hotel review sentence. The two BERT models will find its **sentiment** "
         "and the **aspects** it talks about.")

models_folder = find_models_folder()
if models_folder is None:
    st.error("Could not find the trained models.")
    st.write("Put the `sentiment` and `aspects` folders inside `inference_app/models/`, "
             "or set the environment variable `HOTEL_MODELS_DIR` to the folder that contains them.")
    st.stop()

with st.spinner("Loading models (only the first time)..."):
    models = load_everything(str(models_folder))

# Sidebar: information and the one setting you can change.
st.sidebar.header("Settings")
st.sidebar.write("Models folder:")
st.sidebar.code(str(models_folder))
st.sidebar.write("Running on:", str(models["device"]).upper())
threshold = st.sidebar.slider(
    "Aspect threshold", min_value=0.05, max_value=0.95,
    value=float(models["default_threshold"]), step=0.05,
    help="An aspect is reported when its probability is at least this value. "
         "The default is the value found best during training.",
)

# Input: pick an example or write your own.
choice = st.selectbox("Start from an example (or write your own below)",
                      ["Write my own"] + EXAMPLE_SENTENCES)
if choice == "Write my own":
    starting_text = ""
else:
    starting_text = choice
review = st.text_area("Review sentence", value=starting_text, key=choice, height=100)

if st.button("Analyze", type="primary"):
    if review.strip() == "":
        st.warning("Please type a review first.")
        st.stop()

    with st.spinner("Thinking..."):
        sentiment_label, sentiment_scores = predict_sentiment(models, review.strip())
        aspect_scores = predict_aspects(models, review.strip())

    # ----- sentiment -----
    st.subheader("Sentiment")
    confidence = sentiment_scores[sentiment_label]
    message = f"**{sentiment_label.upper()}**  (confidence {confidence:.1%})"
    if sentiment_label == "positive":
        st.success(message)
    elif sentiment_label == "negative":
        st.error(message)
    else:
        st.info(message)
    st.bar_chart(pd.Series(sentiment_scores, name="probability"))

    # ----- aspects -----
    st.subheader("Aspects mentioned")
    found = []
    for aspect, probability in aspect_scores.items():
        if probability >= threshold:
            found.append((aspect, probability))
    found.sort(key=lambda item: item[1], reverse=True)

    if len(found) == 0:
        st.write("No aspect passed the threshold. Try lowering it in the sidebar.")
    for aspect, probability in found:
        st.write(f"**{aspect}**  {probability:.1%}")
        st.progress(min(probability, 1.0))

    with st.expander("Show the probability of every aspect"):
        all_scores = pd.Series(aspect_scores, name="probability").sort_values(ascending=False)
        st.bar_chart(all_scores)
