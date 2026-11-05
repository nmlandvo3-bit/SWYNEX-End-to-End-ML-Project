
"""Dataset loading, validation, and feature preparation."""

from pathlib import Path

import numpy as np
import pandas as pd


NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

TARGET_MAPPING = {"No": 0, "Yes": 1}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the original customer churn CSV file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw customer records for a fitted ML pipeline.

    Accepts the original dataset or new customer records.
    customerID and Churn, when present, are excluded.

    Preprocessing statistics are NOT fitted here. Numerical
    imputation, scaling, and categorical encoding are handled
    by the scikit-learn pipeline.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input contains no customer records.")

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    features = data[FEATURE_COLUMNS].copy()

    # Convert numerical fields safely.
    for column in NUMERICAL_COLUMNS:
        features[column] = pd.to_numeric(
            features[column].astype("string").str.strip(),
            errors="coerce",
        )

    # In the original dataset, blank TotalCharges occurs
    # for customers whose tenure is zero.
    zero_tenure_blank = (
        features["TotalCharges"].isna()
        & features["tenure"].eq(0)
    )

    features.loc[
        zero_tenure_blank, "TotalCharges"
    ] = 0.0

    # SeniorCitizen is a categorical 0/1 indicator.
    senior = pd.to_numeric(
        features["SeniorCitizen"],
        errors="coerce",
    )

    invalid_senior = (
        senior.notna() & ~senior.isin([0, 1])
    )

    if invalid_senior.any():
        raise ValueError(
            "SeniorCitizen must be 0 or 1."
        )

    features["SeniorCitizen"] = (
        senior.astype("Int64")
        .astype("string")
    )

    # Normalize categorical values and represent missing
    # entries using np.nan for scikit-learn imputation.
    for column in CATEGORICAL_COLUMNS:
        cleaned = (
            features[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        features[column] = (
            cleaned.astype(object)
            .where(cleaned.notna(), np.nan)
        )

    return features


def prepare_target(data: pd.DataFrame) -> pd.Series:
    """Encode customer churn labels as 0 and 1."""
    if "Churn" not in data.columns:
        raise ValueError("Missing target column: Churn")

    target = (
        data["Churn"]
        .astype("string")
        .str.strip()
        .map(TARGET_MAPPING)
    )

    if target.isna().any():
        raise ValueError(
            "Churn must contain only 'No' or 'Yes'."
        )

    return target.astype(int).rename("Churn")
