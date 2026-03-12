"""
train.py
--------
End-to-end model training: LightGBM + XGBoost with cross-validation,
hyperparameter tuning, and full evaluation suite.
"""

import os
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import shap

from features import ChurnFeatureEngineer, get_target

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MDL_DIR  = ROOT / "models"
RPT_DIR  = ROOT / "reports"

for d in [MDL_DIR, RPT_DIR]:
    d.mkdir(exist_ok=True)


# ─── Load data ────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_DIR / "telco_churn_raw.csv")
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}")
    return df


# ─── Model definitions ────────────────────────────────────────────────────

def get_models():
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    xgbm = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=2.3,   # ~70/30 class ratio
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, class_weight="balanced",
            max_iter=1000, random_state=42
        ))
    ])

    return {"LightGBM": lgbm, "XGBoost": xgbm, "LogisticRegression (baseline)": logreg}


# ─── Evaluation helpers ───────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, threshold=0.5):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    auc  = roc_auc_score(y_test, prob)
    ap   = average_precision_score(y_test, prob)
    f1   = f1_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  PR-AUC   : {ap:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"\n{classification_report(y_test, pred, target_names=['No Churn','Churn'])}")

    return {"model": name, "roc_auc": auc, "pr_auc": ap,
            "f1": f1, "precision": prec, "recall": rec,
            "prob": prob, "pred": pred}


# ─── Plot helpers ─────────────────────────────────────────────────────────

PALETTE = {"LightGBM": "#6366F1", "XGBoost": "#F59E0B",
           "LogisticRegression (baseline)": "#94A3B8"}

def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})",
                color=PALETTE.get(r["model"], "#64748B"), lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Prediction Models")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(RPT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/roc_curves.png")


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", lw=1, linestyle="--",
               label=f"No-skill baseline ({baseline:.2f})")
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["prob"])
        ax.plot(rec, prec, label=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})",
                color=PALETTE.get(r["model"], "#64748B"), lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Churn Prediction Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RPT_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/pr_curves.png")


def plot_confusion_matrix(name, pred, y_test):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(RPT_DIR / f"confusion_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, name, top_n=25):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        imp = model.named_steps["clf"].coef_[0]
    else:
        return

    fi = pd.Series(np.abs(imp), index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.sort_values().plot.barh(ax=ax, color="#6366F1", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(RPT_DIR / f"feature_importance_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: reports/feature_importance_{safe}.png")


def plot_shap(model, X_test_df, name):
    try:
        if isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)
            # LightGBM returns list for binary; take class=1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_test_df,
                              max_display=20, show=False, plot_size=(10, 7))
            plt.title(f"SHAP Summary — {name}", pad=12)
            plt.tight_layout()
            safe = name.lower().replace(" ", "_")
            plt.savefig(RPT_DIR / f"shap_{safe}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: reports/shap_{safe}.png")
    except Exception as e:
        print(f"SHAP plot skipped for {name}: {e}")


def plot_threshold_analysis(name, prob, y_test):
    thresholds = np.linspace(0.1, 0.9, 80)
    metrics = []
    for t in thresholds:
        p = (prob >= t).astype(int)
        if p.sum() == 0:
            continue
        metrics.append({
            "threshold": t,
            "precision": precision_score(y_test, p, zero_division=0),
            "recall": recall_score(y_test, p, zero_division=0),
            "f1": f1_score(y_test, p, zero_division=0),
        })
    mdf = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mdf["threshold"], mdf["precision"], label="Precision", color="#6366F1", lw=2)
    ax.plot(mdf["threshold"], mdf["recall"],    label="Recall",    color="#F59E0B", lw=2)
    ax.plot(mdf["threshold"], mdf["f1"],        label="F1",        color="#10B981", lw=2)
    ax.axvline(0.5, color="k", linestyle="--", alpha=0.4, label="Default @ 0.5")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sensitivity — {name}")
    ax.legend()
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(RPT_DIR / f"threshold_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: reports/threshold_{safe}.png")


def plot_calibration(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for r in results:
        bins = np.linspace(0, 1, 11)
        bin_centers, bin_means = [], []
        for i in range(len(bins) - 1):
            mask = (r["prob"] >= bins[i]) & (r["prob"] < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(y_test[mask].mean())
        ax.plot(bin_centers, bin_means, "o-",
                label=r["model"], color=PALETTE.get(r["model"], "#64748B"), lw=2)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction Positive")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RPT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/calibration.png")


def plot_cv_scores(cv_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(cv_results.keys())
    means  = [v["mean"] for v in cv_results.values()]
    stds   = [v["std"]  for v in cv_results.values()]
    colors = [PALETTE.get(n, "#64748B") for n in names]
    bars = ax.barh(names, means, xerr=stds, color=colors,
                   edgecolor="white", height=0.5, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(m + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{m:.4f}", va="center", fontsize=10)
    ax.set_xlabel("ROC-AUC (5-Fold CV)")
    ax.set_title("Cross-Validation ROC-AUC — All Models")
    ax.set_xlim([0.7, 1.0])
    plt.tight_layout()
    plt.savefig(RPT_DIR / "cv_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/cv_scores.png")


# ─── Main training loop ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SaaS Pulse — Churn Prediction Pipeline")
    print("=" * 60)

    # 1. Load + transform
    df = load_data()
    fe = ChurnFeatureEngineer()
    X  = fe.fit_transform(df)
    y  = get_target(df)

    feature_names = X.columns.tolist()
    print(f"\nFeature matrix: {X.shape[0]:,} × {X.shape[1]} features")

    # 2. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"Train churn rate: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")

    # 3. 5-Fold CV on all models
    print("\n── Cross-Validation (5-Fold Stratified) ──")
    models = get_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores.tolist()}
        print(f"  {name:40s} AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    plot_cv_scores(cv_results)

    # 4. Final training + evaluation on hold-out test set
    print("\n── Hold-Out Test Set Evaluation ──")
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        r = evaluate(name, model, X_test, y_test)
        results.append(r)
        plot_confusion_matrix(name, r["pred"], y_test)
        plot_threshold_analysis(name, r["prob"], y_test)
        plot_feature_importance(model, feature_names, name)

    # 5. Comparative plots
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_calibration(results, y_test)

    # 6. SHAP for best model (LightGBM)
    print("\n── SHAP Analysis (LightGBM) ──")
    lgbm_model = trained_models["LightGBM"]
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    plot_shap(lgbm_model, X_test_sample, "LightGBM")

    # 7. Segment analysis — churn rate by feature bucket
    print("\n── Churn Segment Analysis ──")
    test_df = X_test.copy()
    test_df["churn"] = y_test.values
    test_df["churn_prob"] = trained_models["LightGBM"].predict_proba(X_test)[:, 1]

    for col, label in [("contract_enc", "Contract Type"),
                       ("tenure_segment", "Tenure Segment"),
                       ("internet_fiber", "Fiber Internet")]:
        grp = test_df.groupby(col)["churn"].agg(["mean", "count"])
        grp.columns = ["churn_rate", "count"]
        print(f"\n  {label}:\n{grp.to_string()}")

    # 8. Save best model + feature engineer
    print("\n── Saving Artifacts ──")
    best_model = trained_models["LightGBM"]
    with open(MDL_DIR / "lgbm_churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MDL_DIR / "feature_engineer.pkl", "wb") as f:
        pickle.dump(fe, f)

    # Save metrics JSON
    metrics_out = []
    for r, cv in zip(results, cv_results.values()):
        metrics_out.append({
            "model": r["model"],
            "roc_auc": round(r["roc_auc"], 4),
            "pr_auc":  round(r["pr_auc"], 4),
            "f1":      round(r["f1"], 4),
            "precision": round(r["precision"], 4),
            "recall":  round(r["recall"], 4),
            "cv_auc_mean": round(cv["mean"], 4),
            "cv_auc_std":  round(cv["std"], 4),
        })
    with open(RPT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaved: models/lgbm_churn_model.pkl")
    print("Saved: models/feature_engineer.pkl")
    print("Saved: reports/metrics.json")

    # 9. Summary table
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON")
    print("=" * 60)
    mdf = pd.DataFrame(metrics_out).set_index("model")
    print(mdf[["roc_auc", "pr_auc", "f1", "precision", "recall",
               "cv_auc_mean"]].to_string())

    return trained_models, results


if __name__ == "__main__":
    main()
