"""
train_model.py
---------------
Trains and evaluates classification models to predict heart disease
presence from patient records. Compares Logistic Regression,
Random Forest, and K-Nearest Neighbors; tunes the best-performing
model; saves the confusion matrix, ROC curve, feature importance
plot, and a metrics summary (JSON) for the final report.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from data_loader import load_data

VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuals"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VISUALS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    return metrics, y_pred, y_proba


def compare_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    all_metrics = {}
    fitted_models = {}
    print("=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics, _, _ = evaluate_model(name, model, X_test_scaled, y_test)
        all_metrics[name] = metrics
        fitted_models[name] = model

    return all_metrics, fitted_models


def tune_random_forest(X_train_scaled, y_train):
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING: Random Forest (GridSearchCV, 5-fold CV)")
    print("=" * 60)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    print("Best params:", grid.best_params_)
    print("Best CV F1-score: {:.4f}".format(grid.best_score_))
    return grid.best_estimator_


def plot_confusion_matrix(y_test, y_pred, model_name="Best Model"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, model_name="Best Model"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "roc_curve.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(7, 6))
    importances.plot(kind="barh", color="#3498db")
    plt.title("Feature Importance — Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def run_training():
    df = load_data()
    X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)

    # Step 1: Compare baseline models
    all_metrics, fitted_models = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 2: Tune the strongest baseline model (Random Forest)
    best_rf = tune_random_forest(X_train_scaled, y_train)
    tuned_metrics, y_pred, y_proba = evaluate_model("Tuned Random Forest", best_rf, X_test_scaled, y_test)
    all_metrics["Tuned Random Forest"] = tuned_metrics

    # Step 3: 5-fold cross-validation score for the tuned model (robustness check)
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy (Tuned RF): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Step 4: Visualizations for the final report
    plot_confusion_matrix(y_test, y_pred, model_name="Tuned Random Forest")
    plot_roc_curve(y_test, y_proba, model_name="Tuned Random Forest")
    plot_feature_importance(best_rf, X.columns)

    # Step 5: Classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT — Tuned Random Forest")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Step 6: Save metrics summary as JSON
    summary = {
        "model_comparison": all_metrics,
        "cross_validation": {
            "mean_accuracy": round(cv_scores.mean(), 4),
            "std_accuracy": round(cv_scores.std(), 4),
        },
        "best_model": "Tuned Random Forest",
    }
    with open(RESULTS_DIR / "model_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary to: {RESULTS_DIR / 'model_metrics.json'}")

    return summary


if __name__ == "__main__":
    run_training()
