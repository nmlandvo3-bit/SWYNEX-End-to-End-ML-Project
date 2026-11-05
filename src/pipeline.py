
"""Build the customer churn machine-learning pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.preprocessing import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
)


RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.54


def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted preprocessing transformer."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_model_pipeline() -> Pipeline:
    """Create an unfitted Logistic Regression pipeline."""

    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )
