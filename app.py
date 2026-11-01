"""Interactive Streamlit dashboard for the saved breast cancer classifiers."""

from pathlib import Path

import numpy as np
import pandas as pd
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
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
TARGET_COLUMN = "target"
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
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
    return {
        "Accuracy": float(accuracy_score(actual, predictions)),
        "AUC": (
            float(roc_auc_score(actual, scores))
            if actual.nunique() == 2
            else None
        ),
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
        "probabilities": scores,
    }


def validate_data(data: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = feature_names + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("The CSV contains no rows.")

    features = data[feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Feature columns must contain numeric values without missing cells.")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature columns must not contain infinite values.")

    actual = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if actual.isna().any() or not actual.isin([0, 1]).all():
        raise ValueError("The target column must contain only 0 (malignant) or 1 (benign).")
    return features, actual.astype(int)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(35, 122, 114, .11), transparent 34%),
                #f5f7fa;
            color: #1f2937;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #f5f7fa;
            color: #1f2937;
        }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e9ef;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 3px 12px rgba(27, 44, 64, .04);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] div {
            color: #263445 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #edf2f6;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #263445;
        }
        [data-testid="stAlert"] {
            background-color: #ffffff;
            border: 1px solid #d5e0e8;
            color: #263445;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
            color: #263445 !important;
        }
        [data-testid="stAlert"][kind="success"] {
            background-color: #e8f5ee;
            border-color: #8bc7a2;
        }
        [data-testid="stAlert"][kind="success"] p,
        [data-testid="stAlert"][kind="success"] [data-testid="stMarkdownContainer"] {
            color: #174d30 !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background-color: #ffffff;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: #263445;
        }
        .hero {
            padding: 1.65rem 1.8rem;
            margin-bottom: 1.2rem;
            border-radius: 20px;
            color: #f7fbff;
            background: linear-gradient(118deg, #153b4a, #176b70 70%, #348b7c);
        }
        .hero h1 { margin: 0 0 .45rem; color: white; font-size: 2.15rem; }
        .hero p { margin: 0; color: #deeeee; font-size: 1.02rem; }
        .eyebrow {
            color: #a9ddd1; font-size: .76rem; font-weight: 700;
            letter-spacing: .13em; text-transform: uppercase;
        }
        div[data-testid="stTabs"] button { font-weight: 600; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem 1rem 2rem; }
            .hero { padding: 1.2rem; }
            .hero h1 { font-size: 1.65rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Breast Cancer Classifier Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Machine learning · Assignment-2 · ROLL NO:2026AK05103 · NAME:BALAJI.S</div>
            <h1>Wisconsin Breast Cancer Classifier</h1>
            <p>Explore five trained models, compare their test-set performance,
            and inspect individual predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    feature_names = list(next(iter(models.values())).feature_names_in_)
    with st.sidebar:
        st.header("Evaluation setup")
        data_source = st.radio(
            "Choose evaluation data",
            ["Bundled test sample", "Upload your own CSV"],
            help="Uploaded files must include the 30 model features and a labeled target column.",
        )
        st.divider()
        st.markdown("**Dataset at a glance**")
        st.write("569 observations · 30 numeric features · 2 classes")
        st.caption("0 = malignant · 1 = benign")
        st.info("The saved models are only evaluated here. No retraining takes place.")

    data: pd.DataFrame | None = None
    data_label = "Bundled test sample"
    if data_source == "Bundled test sample":
        if not TEST_DATA_PATH.is_file():
            st.error(
                "`test_data.csv` was not found. Run `python train_models.py` "
                "to generate the sample evaluation data."
            )
            st.stop()
        try:
            data = pd.read_csv(TEST_DATA_PATH)
        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
            st.error(f"Could not read the bundled test CSV: {error}")
            st.stop()
    else:
        st.subheader("Bring your own labeled test data")
        st.write(
            "Upload a CSV with the exact training feature columns and a `target` "
            "column. The target must use `0` for malignant and `1` for benign."
        )
        uploaded_file = st.file_uploader(
            "Choose a test CSV",
            type=["csv"],
            help="All 30 input features and the target column are required.",
        )
        if uploaded_file is None:
            st.info("Choose a CSV to run the five saved models on your data.")
            return
        data_label = uploaded_file.name
        try:
            data = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            return

    try:
        features, actual = validate_data(data, feature_names)
    except ValueError as error:
        st.error(str(error))
        return

    results = {
        name: evaluate_model(model, features, actual)
        for name, model in models.items()
    }
    comparison = pd.DataFrame(
        {
            name: {
                metric: value
                for metric, value in result.items()
                if metric in {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
            }
            for name, result in results.items()
        }
    ).T
    ranked_models = comparison.sort_values(
        ["Accuracy", "AUC"], ascending=False, na_position="last"
    )
    winner = ranked_models.index[0]
    best_accuracy = comparison["Accuracy"].max()
    best_auc = comparison["AUC"].max()
    class_counts = actual.value_counts()

    st.caption(f"Evaluating **{data_label}** · {len(data):,} labeled rows")
    for metric_row in (
        [
            ("Test rows", f"{len(data):,}"),
            ("Input features", f"{len(feature_names)}"),
        ],
        [
            ("Top accuracy", f"{best_accuracy:.1%}"),
            ("Top AUC", f"{best_auc:.3f}" if pd.notna(best_auc) else "N/A"),
        ],
    ):
        summary_columns = st.columns(2)
        for column, (label, value) in zip(summary_columns, metric_row):
            column.metric(label, value)
    st.success(
        f"**Current leader: {winner}** — ranked by accuracy, with AUC used to break ties. "
        "Scores describe this evaluation file only."
    )
    if actual.nunique() != 2:
        st.warning("AUC is unavailable because the evaluation file contains only one target class.")

    overview_tab, detail_tab, data_tab = st.tabs(
        ["📊 Model comparison", "🔎 Prediction explorer", "🧾 Test data"]
    )
    with overview_tab:
        left, right = st.columns([1.45, 1])
        with left:
            st.subheader("Scorecard")
            st.dataframe(
                comparison.style.format("{:.4f}", na_rep="N/A").highlight_max(
                    axis=0, color="#d9f2e8"
                ),
                width="stretch",
            )
        with right:
            st.subheader("Compare the models")
            st.bar_chart(
                comparison[["Accuracy", "AUC", "F1"]].sort_values("Accuracy"),
                height=330,
                y_label="Score",
                stack=False,
            )

        st.subheader("How the test labels are distributed")
        class_columns = st.columns(2)
        for column, label in zip(class_columns, [0, 1]):
            count = int(class_counts.get(label, 0))
            column.metric(CLASS_NAMES[label], f"{count:,}", f"{count / len(actual):.1%} of rows")
        st.caption(
            "Metrics use weighted precision, recall and F1. AUC uses the predicted "
            "probability for the benign class."
        )

    with detail_tab:
        st.subheader("Inspect one model")
        selected_model = st.selectbox("Choose a classifier", list(models))
        selected_result = results[selected_model]
        selected_metrics = comparison.loc[selected_model]
        for metric_row in (
            ["Accuracy", "AUC", "Precision"],
            ["Recall", "F1", "MCC"],
        ):
            metric_columns = st.columns(3)
            for column, metric in zip(metric_columns, metric_row):
                value = selected_metrics[metric]
                column.metric(metric, f"{value:.4f}" if pd.notna(value) else "N/A")

        detail_left, detail_right = st.columns([1, 1.2])
        with detail_left:
            st.markdown("**Confusion matrix**")
            matrix = confusion_matrix(actual, selected_result["predictions"], labels=[0, 1])
            matrix_frame = pd.DataFrame(
                matrix,
                index=[f"Actual {CLASS_NAMES[label]}" for label in [0, 1]],
                columns=[f"Predicted {CLASS_NAMES[label]}" for label in [0, 1]],
            )
            st.dataframe(
                matrix_frame.style.background_gradient(cmap="GnBu", axis=None),
                width="stretch",
            )
            st.caption("Rows are actual classes; columns are predicted classes.")
        with detail_right:
            st.markdown("**What the scores mean**")
            st.markdown(
                "- **Accuracy:** share of all predictions that are correct.\n"
                "- **AUC:** how well benign cases rank above malignant cases.\n"
                "- **Precision / Recall / F1:** weighted summaries across both classes.\n"
                "- **MCC:** balanced correlation between actual and predicted labels; "
                "1 is perfect and 0 is no better than chance."
            )

        prediction_table = pd.DataFrame(
            {
                "Actual class": actual.map(CLASS_NAMES).to_numpy(),
                "Predicted class": pd.Series(selected_result["predictions"]).map(CLASS_NAMES),
                "Benign probability": selected_result["probabilities"],
            },
            index=data.index,
        )
        prediction_table["Result"] = np.where(
            prediction_table["Actual class"] == prediction_table["Predicted class"],
            "Correct",
            "Needs review",
        )
        with st.expander("Browse row-by-row predictions"):
            result_filter = st.selectbox(
                "Filter predictions",
                ["All rows", "Incorrect predictions only"],
                key="prediction_filter",
            )
            visible_predictions = prediction_table
            if result_filter == "Incorrect predictions only":
                visible_predictions = prediction_table[
                    prediction_table["Result"] == "Needs review"
                ]
            st.dataframe(visible_predictions, width="stretch")
            st.download_button(
                "Download prediction results",
                prediction_table.to_csv(index=True).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )

    with data_tab:
        st.subheader("Explore the evaluation data")
        st.write("Preview of the labeled rows supplied to the saved models.")
        st.dataframe(data.head(10), width="stretch")
        st.caption(
            f"Showing 10 of {len(data):,} rows and {data.shape[1]} columns, "
            "including the target label."
        )


if __name__ == "__main__":
    main()
