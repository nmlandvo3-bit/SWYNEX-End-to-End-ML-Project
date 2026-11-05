
"""Predict customer churn using the saved Task 4 ML pipeline.

Examples:

    python -m src.predict --input data/sample/customer.json

    python -m src.predict --input data/sample/customer.json --output results/prediction.json

Only load model artifacts created by trusted code.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    prepare_features,
)
from src.train import MODEL_PATH


def load_customer_input(path: str | Path) -> pd.DataFrame:
    """Load one customer or a list of customers from JSON."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Customer input file not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not data:
        raise ValueError(
            "JSON must contain one customer object "
            "or a non-empty list of customer objects."
        )

    if not all(isinstance(row, dict) for row in data):
        raise ValueError(
            "Every customer record must be a JSON object."
        )

    return pd.DataFrame(data)


def validate_customer_input(
    customers: pd.DataFrame,
) -> None:
    """Check the input schema before prediction."""

    missing = [
        name
        for name in FEATURE_COLUMNS
        if name not in customers.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required customer features: {missing}"
        )

    for column in FEATURE_COLUMNS:
        if customers[column].isna().any():
            raise ValueError(
                f"Missing customer value in: {column}"
            )

    for column in NUMERICAL_COLUMNS:
        values = pd.to_numeric(
            customers[column],
            errors="coerce",
        )

        # A blank TotalCharges value is allowed
        # only when tenure is zero.
        if column == "TotalCharges":
            tenure = pd.to_numeric(
                customers["tenure"],
                errors="coerce",
            )

            invalid = (
                values.isna()
                & ~(
                    tenure.eq(0)
                    & customers[column]
                    .astype("string")
                    .str.strip()
                    .eq("")
                )
            )
        else:
            invalid = values.isna()

        if invalid.any():
            raise ValueError(
                f"Invalid numerical value in: {column}"
            )

    tenure = pd.to_numeric(customers["tenure"])
    monthly = pd.to_numeric(customers["MonthlyCharges"])
    total = pd.to_numeric(
        customers["TotalCharges"],
        errors="coerce",
    )

    if (tenure < 0).any():
        raise ValueError("tenure cannot be negative.")

    if (monthly < 0).any():
        raise ValueError(
            "MonthlyCharges cannot be negative."
        )

    if (total.dropna() < 0).any():
        raise ValueError(
            "TotalCharges cannot be negative."
        )


def predict_customers(
    customers: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> list[dict]:
    """Return churn predictions for customer records."""

    validate_customer_input(customers)

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run 'python -m src.train' first."
        )

    artifact = joblib.load(model_path)

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid saved model threshold.")

    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "Saved model feature schema does not match."
        )

    features = prepare_features(customers)

    probabilities = pipeline.predict_proba(
        features
    )[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model returned invalid probabilities."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = []

    for index, (probability, prediction) in enumerate(
        zip(probabilities, predictions)
    ):
        customer_id = (
            customers.iloc[index]["customerID"]
            if "customerID" in customers.columns
            else None
        )

        if pd.isna(customer_id):
            customer_id = None

        result = {
            "customer_id": (
                str(customer_id)
                if customer_id is not None
                else None
            ),
            "predicted_churn": (
                "Yes" if prediction == 1 else "No"
            ),
            "churn_probability": round(
                float(probability),
                6,
            ),
            "no_churn_probability": round(
                1.0 - float(probability),
                6,
            ),
            "classification_threshold": threshold,
        }

        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict customer churn using the trained "
            "SWYNEX Task 4 model."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to customer JSON input file.",
    )

    parser.add_argument(
        "--output",
        help="Optional JSON output file path.",
    )

    args = parser.parse_args()

    try:
        customers = load_customer_input(args.input)

        results = predict_customers(customers)

        output_data = {
            "model": "Logistic Regression",
            "customer_count": len(results),
            "predictions": results,
        }

        output_json = json.dumps(
            output_data,
            indent=4,
        )

        print("\n" + "=" * 60)
        print("SWYNEX TASK 4 - CUSTOMER CHURN PREDICTION")
        print("=" * 60)
        print(output_json)
        print("\nPASS: Customer predictions generated.")

        if args.output:
            output_path = Path(args.output)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                output_json + "\n",
                encoding="utf-8",
            )

            print(f"Saved predictions: {output_path}")

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
