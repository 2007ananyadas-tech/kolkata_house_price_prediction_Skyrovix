from pathlib import Path
import json

import joblib
import pandas as pd


# =========================================================
# PROJECT INFORMATION
# =========================================================

AUTHOR = "Ananya Das"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model_artifacts"
    / "kolkata_house_price_model.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "model_artifacts"
    / "metadata.json"
)


# =========================================================
# CHECK FILES
# =========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "Trained model was not found.\n"
        "Please run 'train model.py' first."
    )


if not METADATA_PATH.exists():

    raise FileNotFoundError(
        "Metadata file was not found.\n"
        "Please run 'train model.py' first."
    )


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    MODEL_PATH
)


# =========================================================
# LOAD METADATA
# =========================================================

metadata = json.loads(
    METADATA_PATH.read_text(
        encoding="utf-8"
    )
)


features = metadata[
    "features"
]


# =========================================================
# CREATE TEST INPUT
# =========================================================

area_median = (
    metadata[
        "input_ranges"
    ][
        "Area"
    ][
        "median"
    ]
)


bedroom_median = (
    metadata[
        "input_ranges"
    ][
        "No. of Bedrooms"
    ][
        "median"
    ]
)


resale_default = (
    metadata[
        "input_ranges"
    ][
        "Resale"
    ][
        "default"
    ]
)


locations = metadata[
    "locations"
]


if not locations:

    raise ValueError(
        "No Kolkata locations were found "
        "in the dataset."
    )


sample_location = (
    locations[0]
)


# Build one sample row
sample_input = pd.DataFrame(
    [
        {
            "Area":
                area_median,

            "Location":
                sample_location,

            "No. of Bedrooms":
                bedroom_median,

            "Resale":
                resale_default
        }
    ],
    columns=features
)


# =========================================================
# PREDICTION TEST
# =========================================================

prediction = model.predict(
    sample_input
)


predicted_price = float(
    prediction[0]
)


# =========================================================
# VALIDATION
# =========================================================

assert len(prediction) == 1

assert predicted_price >= 0


# =========================================================
# OUTPUT
# =========================================================

print()
print("=" * 65)

print(
    "        KOLKATA HOUSE PRICE MODEL TEST"
)

print("=" * 65)

print()

print(
    f"Developer       : {AUTHOR}"
)

print(
    f"Selected Model  : "
    f"{metadata['best_model']}"
)

print(
    f"Test Location   : "
    f"{sample_location}"
)

print(
    f"Test Area       : "
    f"{area_median:,.0f} sq ft"
)

print(
    f"Bedrooms        : "
    f"{bedroom_median}"
)

print(
    f"Resale          : "
    f"{'Yes' if resale_default == 1 else 'No'}"
)

print(
    f"Sample Price    : "
    f"₹{predicted_price:,.0f}"
)

print()

print(
    "✅ Model loaded successfully."
)

print(
    "✅ Metadata loaded successfully."
)

print(
    "✅ Sample input created successfully."
)

print(
    "✅ Prediction generated successfully."
)

print(
    "✅ Prediction value is valid."
)

print()

print(
    "TEST PASSED"
)

print("=" * 65)