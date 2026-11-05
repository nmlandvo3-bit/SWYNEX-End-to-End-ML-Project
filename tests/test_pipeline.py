
"""Automated tests for SWYNEX Task 4.

Run from the project root:

    python -m unittest discover -s tests -p "test_*.py" -v

Run training first to generate the saved model.
"""

import json
import unittest

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.pipeline import (
    build_model_pipeline,
    CLASSIFICATION_THRESHOLD,
    RANDOM_STATE,
)
from src.predict import (
    load_customer_input,
    predict_customers,
    validate_customer_input,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    load_dataset,
    prepare_features,
    prepare_target,
)
from src.train import (
    DATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)


SAMPLE_PATH = (
    PROJECT_ROOT / "data" / "sample" / "customer.json"
)

METRICS_PATH = (
    PROJECT_ROOT / "results" / "evaluation_metrics.json"
)


class TestCustomerChurnPipeline(unittest.TestCase):
    """Test preprocessing, model loading, and predictions."""

    @classmethod
    def setUpClass(cls):
        """Load the dataset and saved model once."""

        cls.df = load_dataset(DATA_PATH)

        cls.X = prepare_features(cls.df)
        cls.y = prepare_target(cls.df)

        _, cls.X_test, _, cls.y_test = train_test_split(
            cls.X,
            cls.y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=cls.y,
        )

        cls.artifact = joblib.load(MODEL_PATH)
        cls.pipeline = cls.artifact["pipeline"]

        cls.sample = load_customer_input(SAMPLE_PATH)

    def test_01_dataset_dimensions(self):
        """Verify the original dataset dimensions."""

        self.assertEqual(self.df.shape, (7043, 21))
        self.assertEqual(self.X.shape, (7043, 19))
        self.assertEqual(len(self.y), 7043)

    def test_02_feature_schema(self):
        """Verify expected model features."""

        self.assertEqual(
            len(FEATURE_COLUMNS),
            19,
        )

        self.assertEqual(
            set(self.X.columns),
            set(FEATURE_COLUMNS),
        )

        self.assertNotIn("customerID", self.X.columns)
        self.assertNotIn("Churn", self.X.columns)

    def test_03_target_encoding(self):
        """Verify binary churn target encoding."""

        self.assertEqual(
            set(self.y.unique().tolist()),
            {0, 1},
        )

        self.assertEqual(
            int((self.y == 0).sum()),
            5174,
        )

        self.assertEqual(
            int((self.y == 1).sum()),
            1869,
        )

    def test_04_missing_total_charges(self):
        """Verify zero-tenure TotalCharges handling."""

        self.assertFalse(
            self.X["TotalCharges"].isna().any()
        )

        zero_tenure = self.X["tenure"].eq(0)

        self.assertTrue(
            self.X.loc[
                zero_tenure, "TotalCharges"
            ].eq(0).all()
        )

    def test_05_split_dimensions(self):
        """Verify the reproducible test partition."""

        self.assertEqual(
            self.X_test.shape,
            (1409, 19),
        )

        self.assertEqual(
            len(self.y_test),
            1409,
        )

    def test_06_saved_model_configuration(self):
        """Verify the saved trained pipeline."""

        self.assertEqual(
            self.artifact["threshold"],
            CLASSIFICATION_THRESHOLD,
        )

        self.assertEqual(
            self.artifact["feature_columns"],
            FEATURE_COLUMNS,
        )

        classifier = self.pipeline.named_steps[
            "classifier"
        ]

        self.assertEqual(
            type(classifier).__name__,
            "LogisticRegression",
        )

        self.assertAlmostEqual(
            classifier.C,
            0.1,
        )

        self.assertEqual(
            classifier.class_weight,
            "balanced",
        )

    def test_07_transformed_feature_count(self):
        """Verify the fitted feature transformation."""

        transformed = self.pipeline.named_steps[
            "preprocessing"
        ].transform(self.X_test)

        self.assertEqual(
            transformed.shape,
            (1409, 46),
        )

        self.assertTrue(
            np.isfinite(transformed).all()
        )

    def test_08_prediction_probabilities(self):
        """Verify valid model probabilities."""

        probabilities = self.pipeline.predict_proba(
            self.X_test
        )[:, 1]

        self.assertEqual(
            len(probabilities),
            1409,
        )

        self.assertTrue(
            np.isfinite(probabilities).all()
        )

        self.assertTrue(
            ((probabilities >= 0)
             & (probabilities <= 1)).all()
        )

    def test_09_evaluation_reproducibility(self):
        """Recalculate and compare saved test metrics."""

        with METRICS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        probabilities = self.pipeline.predict_proba(
            self.X_test
        )[:, 1]

        predictions = (
            probabilities >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        calculated = {
            "accuracy": accuracy_score(
                self.y_test,
                predictions,
            ),
            "recall": recall_score(
                self.y_test,
                predictions,
            ),
            "f1_score": f1_score(
                self.y_test,
                predictions,
            ),
        }

        for metric, value in calculated.items():
            self.assertAlmostEqual(
                value,
                report["metrics"][metric],
                places=6,
            )

    def test_10_single_customer_prediction(self):
        """Verify prediction on a new customer."""

        results = predict_customers(self.sample)

        self.assertEqual(len(results), 1)

        prediction = results[0]

        self.assertEqual(
            prediction["customer_id"],
            "DEMO-001",
        )

        self.assertIn(
            prediction["predicted_churn"],
            ["Yes", "No"],
        )

        self.assertGreaterEqual(
            prediction["churn_probability"],
            0,
        )

        self.assertLessEqual(
            prediction["churn_probability"],
            1,
        )

        expected_label = (
            "Yes"
            if prediction["churn_probability"]
            >= CLASSIFICATION_THRESHOLD
            else "No"
        )

        self.assertEqual(
            prediction["predicted_churn"],
            expected_label,
        )

    def test_11_multiple_customer_predictions(self):
        """Verify batch prediction support."""

        customers = pd.concat(
            [self.sample, self.sample],
            ignore_index=True,
        )

        customers.loc[
            1, "customerID"
        ] = "DEMO-002"

        results = predict_customers(customers)

        self.assertEqual(len(results), 2)

        self.assertEqual(
            results[0]["customer_id"],
            "DEMO-001",
        )

        self.assertEqual(
            results[1]["customer_id"],
            "DEMO-002",
        )

    def test_12_missing_feature_rejected(self):
        """Verify incomplete customer validation."""

        invalid = self.sample.drop(
            columns=["MonthlyCharges"]
        )

        with self.assertRaises(ValueError):
            validate_customer_input(invalid)

    def test_13_negative_charge_rejected(self):
        """Verify negative charges are rejected."""

        invalid = self.sample.copy()

        invalid.loc[
            0, "MonthlyCharges"
        ] = -25.0

        with self.assertRaises(ValueError):
            validate_customer_input(invalid)

    def test_14_unfitted_pipeline_creation(self):
        """Verify a new pipeline is initially unfitted."""

        pipeline = build_model_pipeline()

        self.assertEqual(
            len(pipeline.steps),
            2,
        )

        self.assertFalse(
            hasattr(
                pipeline.named_steps["classifier"],
                "coef_",
            )
        )


if __name__ == "__main__":
    unittest.main()
