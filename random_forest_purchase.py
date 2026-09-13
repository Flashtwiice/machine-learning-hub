import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe to run inside Flask's request threads

import io
import base64

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CSV_PATH = "data/customer_purchase.csv"
FEATURES = ["age", "annual_income", "time_on_site"]
TARGET = "purchased"
CLASS_NAMES = {0: "Will Not Purchase", 1: "Will Purchase"}

df = pd.read_csv(CSV_PATH)
RECORD_COUNT = len(df)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
TRAIN_COUNT = len(X_train)
TEST_COUNT = len(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

_y_pred_test = model.predict(X_test)


def predict_class(age, annual_income, time_on_site):
    """Return the predicted class name and probability for a new customer."""
    input_df = pd.DataFrame(
        {"age": [age], "annual_income": [annual_income], "time_on_site": [time_on_site]}
    )
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
    """Scatter of income vs. time on site by class, plus feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    for label, color, name in [(0, "#0d6efd", CLASS_NAMES[0]), (1, "#dc3545", CLASS_NAMES[1])]:
        subset = df[df[TARGET] == label]
        ax.scatter(subset["annual_income"], subset["time_on_site"], alpha=0.5, color=color, label=name)
    ax.set_title("Customer Purchases by Income and Time on Site")
    ax.set_xlabel("Annual Income (thousands USD)")
    ax.set_ylabel("Time on Site (minutes)")
    ax.legend()

    ax2 = axes[1]
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    ax2.barh(importances.index, importances.values, color="#6610f2")
    ax2.set_title("Feature Importance (Random Forest)")
    ax2.set_xlabel("Relative Importance")

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
