"""Streamlit interface for evaluating the saved breast cancer classifiers."""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
TARGET_COLUMN = "target"
CLASS_NAMES = {0: "malignant", 1: "benign"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_models() -> dict:
    missing = [
        filename
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Saved model files are missing: "
            + ", ".join(missing)
            + ". Run `python train_models.py` before starting the app."
        )
    return {
        name: load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


def evaluate_model(model, features: pd.DataFrame, actual: pd.Series) -> dict:
    predictions = model.predict(features)
    scores = model.predict_proba(features)[:, 1]
    auc = (
        float(roc_auc_score(actual, scores))
        if actual.nunique() == 2
        else None
    )
    return {
        "Accuracy": float(accuracy_score(actual, predictions)),
        "AUC": auc,
        "Precision": float(
            precision_score(actual, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(actual, predictions, average="weighted", zero_division=0)
        ),
        "F1": float(
            f1_score(actual, predictions, average="weighted", zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(actual, predictions)),
        "predictions": predictions,
    }


def main() -> None:
    st.set_page_config(page_title="Wisconsin Breast Cancer Classifiers", page_icon="🔬")
    st.title("Wisconsin Breast Cancer: Model Comparison")
    st.write(
        "Compare five saved classifiers on labeled test data. The app evaluates "
        "your CSV with the existing models; it does not train or refit them."
    )
    st.caption(
        "Dataset: 569 samples, 30 numeric features, binary target "
        "(0 = malignant, 1 = benign)."
    )

    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a labeled test CSV",
        type=["csv"],
        help="Include all 30 features and the target column.",
    )
    if uploaded_file is None:
        st.info("Upload `test_data.csv` to view the model metrics and confusion matrix.")
        return

    try:
        data = pd.read_csv(uploaded_file)
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
    ) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return

    required_features = list(next(iter(models.values())).feature_names_in_)
    missing_columns = [
        column
        for column in required_features + [TARGET_COLUMN]
        if column not in data.columns
    ]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        return
    if data.empty:
        st.error("The uploaded CSV contains no rows.")
        return

    features = data[required_features].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        st.error("Feature columns must contain valid numeric values and no missing cells.")
        return
    if not np.isfinite(features.to_numpy()).all():
        st.error("Feature columns must not contain infinite values.")
        return

    actual = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if actual.isna().any() or not actual.isin([0, 1]).all():
        st.error("The target column must contain only 0 (malignant) or 1 (benign).")
        return
    actual = actual.astype(int)

    results = {
        name: evaluate_model(model, features, actual)
        for name, model in models.items()
    }
    comparison = pd.DataFrame(
        {
            name: {
                metric: value
                for metric, value in result.items()
                if metric != "predictions"
            }
            for name, result in results.items()
        }
    ).T
    st.subheader(f"Evaluation on {len(data)} uploaded rows")
    st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
    if actual.nunique() != 2:
        st.warning("AUC is unavailable because the uploaded target contains only one class.")

    selected_model = st.selectbox("Model for confusion matrix", list(models))
    selected_predictions = results[selected_model]["predictions"]
    matrix = confusion_matrix(actual, selected_predictions, labels=[0, 1])
    st.subheader(f"{selected_model}: confusion matrix")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[f"Actual {CLASS_NAMES[label]}" for label in [0, 1]],
            columns=[f"Predicted {CLASS_NAMES[label]}" for label in [0, 1]],
        ),
        width="stretch",
    )
    st.caption("Rows are actual classes; columns are predicted classes.")


if __name__ == "__main__":
    main()
