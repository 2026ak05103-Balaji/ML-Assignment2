"""Train and save the assignment classifiers and a held-out evaluation CSV."""

import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"


def build_models() -> dict[str, Pipeline]:
    """Create the five classifiers, scaling only where it is useful."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline([("classifier", GaussianNB())]),
        "Support Vector Machine": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=SVC(random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=5,
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    labels = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    metric_rows: dict[str, dict[str, float]] = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]

        metric_rows[model_name] = {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "AUC": float(roc_auc_score(y_test, probabilities)),
            "Precision": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "Recall": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "F1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "MCC": float(matthews_corrcoef(y_test, predictions)),
        }
        model_file = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        dump(model, model_file)

    metrics_file = PROJECT_DIR / "metrics.json"
    metrics_file.write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    print(f"Dataset: {len(features)} rows, {features.shape[1]} features")
    print(f"Training rows: {len(x_train)}; test rows: {len(x_test)}")
    print(f"Test CSV: {PROJECT_DIR / 'test_data.csv'}")
    print(f"Saved models: {MODEL_DIR}")
    print(pd.DataFrame.from_dict(metric_rows, orient="index").round(4).to_string())


if __name__ == "__main__":
    main()
