"""Canonical labels shared by data preparation, training, and inference."""

# The order of these columns must never change after models have been trained.
SENTIMENT_COLUMNS = ["positive", "negative", "neutral"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

SENTIMENT_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ASPECT_COLUMNS = [
    "Clean",
    "Comfort",
    "Facilities/Amenities",
    "Location",
    "Restaurant (dinner)",
    "Staff",
    "View (Balcony)",
    "Breakfast",
    "Room",
    "Pool",
    "Beach",
    "Bathroom/Shower (toilet)",
    "Bar",
    "Bed",
    "Parking",
    "Noise",
    "Reception-checkin",
    "Lift",
    "Value for money",
    "Wi-Fi",
    "Generic",
]


def get_sentiment_name(row):
    """Convert the three one-hot columns in a validated row to one label."""
    if int(row["negative"]) == 1:
        return "negative"
    if int(row["neutral"]) == 1:
        return "neutral"
    return "positive"

