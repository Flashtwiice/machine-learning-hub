import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe to run inside Flask's request threads

import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CSV_PATH = "data/diabetes.csv"
FEATURE = "glucose_level"
TARGET = "diabetic"
CLASS_NAMES = {0: "Non-Diabetic", 1: "Diabetic"}

df = pd.read_csv(CSV_PATH)
RECORD_COUNT = len(df)

X = df[[FEATURE]]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
TRAIN_COUNT = len(X_train)
TEST_COUNT = len(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

_y_pred_test = model.predict(X_test)


def predict_class(glucose_level):
    """Return the predicted class name and probability for a new glucose reading."""
    input_df = pd.DataFrame({FEATURE: [glucose_level]})
    predicted_label = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][predicted_label])
    return {
        "label": predicted_label,
        "class_name": CLASS_NAMES[predicted_label],
        "probability": round(probability * 100, 2),
    }


def get_metrics():
    """Confusion matrix and classification metrics computed on the 20% test set."""
    tn, fp, fn, tp = confusion_matrix(y_test, _y_pred_test, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": round(accuracy_score(y_test, _y_pred_test) * 100, 2),
        "precision": round(precision_score(y_test, _y_pred_test) * 100, 2),
        "recall": round(recall_score(y_test, _y_pred_test) * 100, 2),
        "f1": round(f1_score(y_test, _y_pred_test) * 100, 2),
    }


def generate_plot():
    """Scatter of glucose readings by class, with the model's fitted probability curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(7)
    for label, color, name in [(0, "#0d6efd", CLASS_NAMES[0]), (1, "#dc3545", CLASS_NAMES[1])]:
        subset = df[df[TARGET] == label]
        jitter = rng.normal(0, 0.03, size=len(subset))
        ax.scatter(subset[FEATURE], label + jitter, alpha=0.5, color=color, label=name)

    x_curve = pd.DataFrame({FEATURE: np.linspace(df[FEATURE].min(), df[FEATURE].max(), 200)})
    probability_curve = model.predict_proba(x_curve)[:, 1]
    ax.plot(x_curve[FEATURE], probability_curve, color="#212529", linewidth=2, label="Predicted probability")

    ax.set_title("Diabetes Classification by Glucose Level")
    ax.set_xlabel("Glucose Level (mg/dL)")
    ax.set_ylabel("Class (0 = Non-Diabetic, 1 = Diabetic) / Probability")
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
