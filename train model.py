from pathlib import Path
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# PROJECT INFORMATION
# =========================================================

AUTHOR = "Ananya Das"
PROJECT_NAME = "Kolkata HousePrice AI"

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model_artifacts"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# =========================================================
# DATASET
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "anish2105/House-Price-Prediction/"
    "main/Kolkata.csv"
)

DATA_PATH = DATA_DIR / "Kolkata.csv"

MODEL_PATH = (
    MODEL_DIR
    / "kolkata_house_price_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "metadata.json"
)


# =========================================================
# FEATURES
# =========================================================

TARGET_COLUMN = "Price"

FEATURE_COLUMNS = [
    "Area",
    "Location",
    "No. of Bedrooms",
    "Resale"
]

NUMERIC_FEATURES = [
    "Area",
    "No. of Bedrooms",
    "Resale"
]

CATEGORICAL_FEATURES = [
    "Location"
]


# =========================================================
# HELPER
# =========================================================

def download_dataset():
    """
    Download the Kolkata CSV automatically
    the first time the project is trained.
    """

    if DATA_PATH.exists():

        print(
            "Kolkata dataset already exists."
        )

        return

    print(
        "Downloading Kolkata housing dataset..."
    )

    try:

        urllib.request.urlretrieve(
            DATA_URL,
            DATA_PATH
        )

        print(
            f"Dataset saved to: {DATA_PATH}"
        )

    except Exception as error:

        raise RuntimeError(
            "Could not download the Kolkata dataset. "
            "Please check your internet connection "
            "and try again."
        ) from error


def create_one_hot_encoder():
    """
    Support both newer and older
    scikit-learn versions.
    """

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                create_one_hot_encoder()
            )
        ]
    )


    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )


def clean_dataset(data):

    # Clean column names
    data.columns = [
        str(column).strip()
        for column in data.columns
    ]


    # Check required columns
    required_columns = [
        TARGET_COLUMN,
        *FEATURE_COLUMNS
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )


    # Keep only required columns
    data = data[
        required_columns
    ].copy()


    # Remove duplicate records
    data = data.drop_duplicates()


    # Convert numeric columns
    for column in [
        TARGET_COLUMN,
        "Area",
        "No. of Bedrooms",
        "Resale"
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


    # Clean location
    data["Location"] = (
        data["Location"]
        .astype(str)
        .str.strip()
    )


    # The source dataset uses 9 for missing values.
    # Convert it to NaN in numeric columns.
    for column in [
        TARGET_COLUMN,
        "Area",
        "No. of Bedrooms",
        "Resale"
    ]:

        data.loc[
            data[column] == 9,
            column
        ] = np.nan


    # Remove invalid critical rows
    data = data.dropna(
        subset=[
            TARGET_COLUMN,
            "Area",
            "Location",
            "No. of Bedrooms",
            "Resale"
        ]
    )


    # Keep realistic values
    data = data[
        (data[TARGET_COLUMN] > 0)
        & (data["Area"] > 0)
        & (data["No. of Bedrooms"] > 0)
        & (
            data["Resale"].isin([0, 1])
        )
    ]


    # Remove blank locations
    data = data[
        data["Location"].str.len() > 0
    ]


    return data.reset_index(drop=True)


# =========================================================
# MAIN TRAINING
# =========================================================

def main():

    print()
    print("=" * 65)
    print("          KOLKATA HOUSE PRICE PREDICTION")
    print("                 Developed by Ananya Das")
    print("=" * 65)
    print()


    # -----------------------------------------------------
    # 1. Download dataset
    # -----------------------------------------------------

    download_dataset()


    # -----------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------

    print(
        "Loading Kolkata housing dataset..."
    )

    raw_data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Original rows: {len(raw_data):,}"
    )


    # -----------------------------------------------------
    # 3. Clean dataset
    # -----------------------------------------------------

    data = clean_dataset(
        raw_data
    )

    print(
        f"Rows after cleaning: "
        f"{len(data):,}"
    )


    if len(data) < 100:

        raise ValueError(
            "Not enough valid data rows "
            "for model training."
        )


    # -----------------------------------------------------
    # 4. Split features and target
    # -----------------------------------------------------

    X = data[
        FEATURE_COLUMNS
    ]

    y = data[
        TARGET_COLUMN
    ]


    # -----------------------------------------------------
    # 5. Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )


    print()
    print(
        f"Training records: {len(X_train):,}"
    )

    print(
        f"Testing records : {len(X_test):,}"
    )


    # -----------------------------------------------------
    # 6. Preprocessor
    # -----------------------------------------------------

    preprocessor = build_preprocessor()


    # -----------------------------------------------------
    # 7. Machine learning models
    # -----------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=250,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }


    # -----------------------------------------------------
    # 8. Train and evaluate
    # -----------------------------------------------------

    results = {}

    best_model_name = None
    best_pipeline = None
    best_r2 = float("-inf")


    for model_name, model in models.items():

        print()
        print(
            f"Training: {model_name}"
        )


        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        results[model_name] = {

            "mae":
                float(mae),

            "rmse":
                float(rmse),

            "r2":
                float(r2)
        }


        print(
            f"MAE  : ₹{mae:,.0f}"
        )

        print(
            f"RMSE : ₹{rmse:,.0f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )


        # Select best model by R²
        if r2 > best_r2:

            best_r2 = r2

            best_model_name = model_name

            best_pipeline = pipeline


    # -----------------------------------------------------
    # 9. Save best model
    # -----------------------------------------------------

    joblib.dump(
        best_pipeline,
        MODEL_PATH,
        compress=3
    )


    # -----------------------------------------------------
    # 10. Metadata
    # -----------------------------------------------------

    location_list = sorted(
        data["Location"]
        .astype(str)
        .unique()
        .tolist()
    )


    metadata = {

        "author":
            AUTHOR,

        "project":
            PROJECT_NAME,

        "dataset":
            "Kolkata.csv",

        "dataset_source":
            DATA_URL,

        "target":
            TARGET_COLUMN,

        "features":
            FEATURE_COLUMNS,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "best_model":
            best_model_name,

        "test_size":
            0.20,

        "random_state":
            42,

        "row_count":
            int(len(data)),

        "locations":
            location_list,

        "input_ranges": {

            "Area": {

                "min":
                    float(
                        data["Area"].min()
                    ),

                "max":
                    float(
                        data["Area"].max()
                    ),

                "median":
                    float(
                        data["Area"].median()
                    )
            },

            "No. of Bedrooms": {

                "min":
                    int(
                        data[
                            "No. of Bedrooms"
                        ].min()
                    ),

                "max":
                    int(
                        data[
                            "No. of Bedrooms"
                        ].max()
                    ),

                "median":
                    int(
                        round(
                            data[
                                "No. of Bedrooms"
                            ].median()
                        )
                    )
            },

            "Resale": {

                "default":
                    int(
                        round(
                            data[
                                "Resale"
                            ].mode()[0]
                        )
                    )
            }
        },

        "models":
            results
    }


    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4
        ),
        encoding="utf-8"
    )


    # -----------------------------------------------------
    # 11. Final output
    # -----------------------------------------------------

    print()
    print("=" * 65)

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print("=" * 65)

    print()
    print(
        f"Best Model : {best_model_name}"
    )

    print(
        f"Best R²    : {best_r2:.4f}"
    )

    print()

    print(
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        f"Metadata saved to:"
    )

    print(
        METADATA_PATH
    )

    print()
    print(
        f"Developer: {AUTHOR}"
    )

    print()


if __name__ == "__main__":
    main()