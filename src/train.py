
"""Train and save the SWYNEX Task 4 customer churn model.

Run from the project root:

    python -m src.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.pipeline import (
    build_model_pipeline,
    CLASSIFICATION_THRESHOLD,
    RANDOM_STATE,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    load_dataset,
    prepare_features,
    prepare_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_customer_churn.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR / "customer_churn_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "training_metadata.json"
)


def train_model():
    """Train and save a complete customer churn ML pipeline."""

    print("=" * 60)
    print("SWYNEX TASK 4 - MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------

    print("\n[1/6] Loading dataset...")

    df = load_dataset(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # 2. Prepare features and target
    # --------------------------------------------------

    print("\n[2/6] Preparing features and target...")

    X = prepare_features(df)
    y = prepare_target(df)

    print(f"Feature matrix: {X.shape}")
    print(f"Target records: {len(y)}")

    assert len(X) == len(y)
    assert X.shape[1] == 19

    # --------------------------------------------------
    # 3. Train/test split
    # --------------------------------------------------

    print("\n[3/6] Creating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    assert len(X_train) == 5634
    assert len(X_test) == 1409

    # --------------------------------------------------
    # 4. Build and train pipeline
    # --------------------------------------------------

    print("\n[4/6] Training Logistic Regression pipeline...")

    pipeline = build_model_pipeline()

    pipeline.fit(X_train, y_train)

    transformed_feature_count = len(
        pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()
    )

    print("Algorithm: Logistic Regression")
    print("Regularization C: 0.1")
    print("Class weight: balanced")
    print(
        "Classification threshold:",
        CLASSIFICATION_THRESHOLD,
    )

    print(
        "Transformed feature count:",
        transformed_feature_count,
    )

    assert transformed_feature_count == 46

    # --------------------------------------------------
    # 5. Save fitted pipeline
    # --------------------------------------------------

    print("\n[5/6] Saving trained model...")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_artifact = {
        "pipeline": pipeline,
        "threshold": CLASSIFICATION_THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {
            "No": 0,
            "Yes": 1,
        },
        "random_state": RANDOM_STATE,
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "transformed_feature_count": transformed_feature_count,
    }

    joblib.dump(
        model_artifact,
        MODEL_PATH,
    )

    print(
        "Saved model:",
        MODEL_PATH.relative_to(PROJECT_ROOT),
    )

    # --------------------------------------------------
    # 6. Save metadata and verify model
    # --------------------------------------------------

    print("\n[6/6] Verifying saved model...")

    metadata = {
        "project": "SWYNEX Task 4 - End-to-End ML",
        "algorithm": "Logistic Regression",
        "dataset": DATA_PATH.name,
        "total_records": len(df),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "original_features": len(FEATURE_COLUMNS),
        "transformed_features": transformed_feature_count,
        "regularization_C": 0.1,
        "class_weight": "balanced",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "split_strategy": "stratified",
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    loaded_artifact = joblib.load(MODEL_PATH)

    loaded_pipeline = loaded_artifact["pipeline"]

    # Verify serialization using held-out feature inputs.
    # No test labels or evaluation metrics are used for training.
    original_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    loaded_probabilities = loaded_pipeline.predict_proba(
        X_test
    )[:, 1]

    assert np.allclose(
        original_probabilities,
        loaded_probabilities,
    )

    assert (
        loaded_artifact["threshold"]
        == CLASSIFICATION_THRESHOLD
    )

    assert (
        loaded_artifact["feature_columns"]
        == FEATURE_COLUMNS
    )

    print("PASS: Saved model successfully reloaded.")
    print("PASS: Predictions match original pipeline.")
    print("PASS: Feature schema verified.")

    print("\nTraining metadata saved.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return model_artifact


if __name__ == "__main__":
    train_model()
