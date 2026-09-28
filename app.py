from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PROJECT SETTINGS
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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Kolkata HomeValue",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD PROJECT
# =========================================================

@st.cache_resource
def load_project():
    if not MODEL_PATH.exists():
        return None, None

    if not METADATA_PATH.exists():
        return None, None

    try:
        model = joblib.load(MODEL_PATH)

        metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        return model, metadata

    except Exception:
        return None, None


model, metadata = load_project()


# =========================================================
# CHECK PROJECT
# =========================================================

if model is None or metadata is None:

    st.error("The prediction service is not ready.")

    st.info(
        "Please run 'train model.py' once and open the app again."
    )

    st.stop()


# =========================================================
# HELPERS
# =========================================================

def format_inr(value):
    value = int(round(float(value)))

    number = str(abs(value))

    if len(number) <= 3:
        formatted = number

    else:
        last_three = number[-3:]
        remaining = number[:-3]

        groups = []

        while len(remaining) > 2:
            groups.insert(
                0,
                remaining[-2:]
            )

            remaining = remaining[:-2]

        if remaining:
            groups.insert(
                0,
                remaining
            )

        formatted = (
            ",".join(groups)
            + ","
            + last_three
        )

    if value < 0:
        return f"-₹{formatted}"

    return f"₹{formatted}"


# =========================================================
# GET INPUT INFORMATION
# =========================================================

locations = metadata.get(
    "locations",
    []
)

area_range = metadata.get(
    "input_ranges",
    {}
).get(
    "Area",
    {}
)

bedroom_range = metadata.get(
    "input_ranges",
    {}
).get(
    "No. of Bedrooms",
    {}
)


if not locations:
    st.error("Kolkata location data is unavailable.")
    st.stop()


# =========================================================
# TOP HEADER
# =========================================================

st.write("")

top_left, top_right = st.columns(
    [5, 1]
)

with top_left:

    st.title("🏠 Kolkata HomeValue")

    st.caption(
        "A simple way to estimate the value of a property in Kolkata."
    )

with top_right:

    st.write("")
    st.write("")
    st.caption(
        f"Developed by {AUTHOR}"
    )


st.divider()


# =========================================================
# INTRO
# =========================================================

st.subheader("Find your estimated property value")

st.write(
    "Enter a few basic details about the property and get "
    "an estimated Kolkata house price."
)

st.write("")


# =========================================================
# PROPERTY FORM
# =========================================================

with st.container(border=True):

    st.subheader("Property Details")

    location = st.selectbox(
        "Kolkata Location",
        options=locations
    )

    area_col, bedroom_col = st.columns(
        2
    )

    with area_col:

        area = st.number_input(
            "Area (sq ft)",

            min_value=float(
                area_range.get(
                    "min",
                    300
                )
            ),

            max_value=float(
                area_range.get(
                    "max",
                    10000
                )
            ),

            value=float(
                area_range.get(
                    "median",
                    1000
                )
            ),

            step=10.0,

            help="Enter the property area in square feet."
        )

    with bedroom_col:

        bedrooms = st.number_input(
            "Bedrooms",

            min_value=int(
                bedroom_range.get(
                    "min",
                    1
                )
            ),

            max_value=int(
                bedroom_range.get(
                    "max",
                    5
                )
            ),

            value=int(
                bedroom_range.get(
                    "median",
                    2
                )
            ),

            step=1
        )

    st.write("")

    property_type = st.radio(
        "Property Status",
        [
            "New Property",
            "Resale Property"
        ],
        horizontal=True
    )

    resale = 1 if property_type == "Resale Property" else 0

    st.write("")

    calculate = st.button(
        "🏠  Estimate Property Value",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if calculate:

    try:

        input_data = pd.DataFrame(
            [
                {
                    "Area": area,
                    "Location": location,
                    "No. of Bedrooms": bedrooms,
                    "Resale": resale
                }
            ]
        )

        prediction = model.predict(
            input_data
        )

        estimated_price = max(
            float(prediction[0]),
            0.0
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.write("")
        st.success("Your estimate is ready.")

        st.metric(
            label="Estimated Kolkata House Price",
            value=format_inr(
                estimated_price
            )
        )


        # -------------------------------------------------
        # PROPERTY SUMMARY
        # -------------------------------------------------

        st.write("")

        st.subheader("Property Summary")

        summary_1, summary_2, summary_3 = st.columns(3)

        with summary_1:
            st.info(
                f"📍 Location\n\n{location}"
            )

        with summary_2:
            st.info(
                f"📐 Area\n\n{area:,.0f} sq ft"
            )

        with summary_3:
            st.info(
                f"🛏 Bedrooms\n\n{int(bedrooms)}"
            )


        st.caption(
            "This is an estimated value for informational purposes "
            "and should not be treated as a professional valuation."
        )


    except Exception as error:

        st.error(
            f"Unable to generate the estimate: {error}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    f"🏠 Kolkata HomeValue  •  Developed by {AUTHOR}"
)