from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Bank Marketing | ML Model Explorer", page_icon="📊", layout="wide")
st.title("Bank Marketing: Classification Model Explorer")
st.caption("BITS Pilani • Machine Learning Assignment 2 | Evaluate five trained models on labeled test data")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
TARGET = "y"
@st.cache_resource
def load_model(filename):
    return joblib.load(Path(__file__).parent / "model" / filename)

st.sidebar.header("Evaluate a model")
model_name = st.sidebar.selectbox("Select classification model", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload labeled test CSV", type=["csv"])
st.sidebar.caption("Use test_data.csv. The CSV must include the target column 'y' with yes/no values.")
if uploaded is None:
    st.info("Upload your labeled test_data.csv using the sidebar to calculate metrics and show a confusion matrix.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if TARGET not in data.columns:
    st.error("Missing target column 'y'. Please upload labeled test_data.csv.")
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
labels = data[TARGET].astype(str).str.strip().str.lower()
if not labels.isin(["yes", "no"]).all():
    st.error("Target 'y' must contain only 'yes' and 'no' values.")
    st.stop()
y_true = labels.map({"no": 0, "yes": 1})
if y_true.nunique() < 2:
    st.warning("Only one target class is present. AUC cannot be calculated.")

try:
    model = load_model(MODEL_FILES[model_name])
    features = data.drop(columns=[TARGET])
    expected = list(model.feature_names_in_)
    missing = [name for name in expected if name not in features.columns]
    if missing:
        st.error("Missing input columns: " + ", ".join(missing))
        st.stop()
    features = features[expected]
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    else:
        scores = model.decision_function(features)
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

auc = roc_auc_score(y_true, scores) if y_true.nunique() == 2 else None
metrics = {
    "Accuracy": accuracy_score(y_true, predictions),
    "AUC": auc,
    "Precision": precision_score(y_true, predictions, zero_division=0),
    "Recall": recall_score(y_true, predictions, zero_division=0),
    "F1 Score": f1_score(y_true, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_true, predictions),
}
st.subheader(model_name)
st.write(f"Evaluated **{len(data):,}** labeled test records.")
cols = st.columns(3)
for i, (name, value) in enumerate(metrics.items()):
    cols[i % 3].metric(name, f"{value:.4f}" if value is not None else "N/A")

left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, predictions, labels=[0, 1], display_labels=["No", "Yes"],
        cmap="Blues", values_format="d", ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_true, predictions, labels=[0, 1],
                                   target_names=["No", "Yes"], zero_division=0)
    st.code(report, language="text")

with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(20), use_container_width=True)

st.caption("Evaluation uses uploaded labeled test data only; no training occurs in this app.")
