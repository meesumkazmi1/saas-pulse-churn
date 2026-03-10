"""
features.py
-----------
Feature engineering pipeline for SaaS Pulse Churn Prediction.
Transforms raw Telco data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ─── Raw column groups ──────────────────────────────────────────────────────

BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
]

ORDINAL_MAPS = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
}

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

MULTI_LINE_MAP = {"No phone service": 0, "No": 0, "Yes": 1}
ADDON_MAP      = {"No internet service": 0, "No": 0, "Yes": 1}
GENDER_MAP     = {"Female": 0, "Male": 1}


# ─── Transformer ────────────────────────────────────────────────────────────

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer. Produces a dense numeric DataFrame
    with business-meaningful engineered features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── 1. Basic encodings ────────────────────────────────────────────
        df["gender_enc"]    = df["gender"].map(GENDER_MAP)
        df["contract_enc"]  = df["Contract"].map(ORDINAL_MAPS["Contract"])

        for col in BINARY_YES_NO:
            if col in df.columns:
                df[f"{col}_enc"] = (df[col] == "Yes").astype(int)

        df["multiple_lines_enc"] = df["MultipleLines"].map(MULTI_LINE_MAP).fillna(0).astype(int)

        for col in INTERNET_ADDONS:
            df[f"{col}_enc"] = df[col].map(ADDON_MAP).fillna(0).astype(int)

        # Internet service dummies
        df["internet_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["internet_dsl"]   = (df["InternetService"] == "DSL").astype(int)
        df["internet_none"]  = (df["InternetService"] == "No").astype(int)

        # Payment method dummies
        for pm in ["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"]:
            safe = pm.lower().replace(" ", "_").replace("(", "").replace(")", "")
            df[f"pay_{safe}"] = (df["PaymentMethod"] == pm).astype(int)

        # ── 2. TotalCharges cleaning ──────────────────────────────────────
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

        # ── 3. Engineered features (the good stuff) ───────────────────────

        # Revenue per month of tenure — proxy for ARPU trajectory
        df["avg_monthly_spend"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"]
        )

        # Monthly charges vs. cohort average — above/below average spender
        df["monthly_vs_median"] = df["MonthlyCharges"] - df["MonthlyCharges"].median()

        # Tenure segments (early / growth / mature / loyal)
        df["tenure_segment"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 24, 48, 72],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Count of add-on services purchased
        addon_enc_cols = [f"{c}_enc" for c in INTERNET_ADDONS]
        df["addon_count"] = df[addon_enc_cols].sum(axis=1)

        # Service depth score: phone + internet type + addons (0–9 scale)
        df["service_depth"] = (
            df["PhoneService_enc"] +
            df["multiple_lines_enc"] +
            df["internet_fiber"] * 2 +
            df["internet_dsl"] * 1 +
            df["addon_count"]
        )

        # Engagement index: paperless billing + auto-pay = digitally engaged
        auto_pay = df["pay_bank_transfer_automatic"] + df["pay_credit_card_automatic"]
        df["digital_engagement"] = df["PaperlessBilling_enc"] + auto_pay.clip(0, 1)

        # Risk flag: month-to-month AND fiber AND no security (high-churn combo)
        df["high_risk_combo"] = (
            (df["contract_enc"] == 0) &
            (df["internet_fiber"] == 1) &
            (df["OnlineSecurity_enc"] == 0)
        ).astype(int)

        # Spend-to-contract ratio: high monthly on short contract = at-risk
        df["spend_contract_ratio"] = df["MonthlyCharges"] / (df["contract_enc"] + 1)

        # Lifetime value proxy: tenure * monthly
        df["ltv_proxy"] = df["tenure"] * df["MonthlyCharges"]

        # Streaming subscriber (both TV + movies)
        df["dual_streamer"] = (
            (df["StreamingTV_enc"] == 1) & (df["StreamingMovies_enc"] == 1)
        ).astype(int)

        # Support / security bundle (both tech support + security)
        df["security_bundle"] = (
            (df["TechSupport_enc"] == 1) & (df["OnlineSecurity_enc"] == 1)
        ).astype(int)

        # ── 4. Select final feature set ───────────────────────────────────
        feature_cols = [
            # Demographics
            "gender_enc", "SeniorCitizen", "Partner_enc", "Dependents_enc",
            # Account basics
            "tenure", "tenure_segment", "contract_enc",
            "PaperlessBilling_enc", "digital_engagement",
            # Financials
            "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
            "monthly_vs_median", "spend_contract_ratio", "ltv_proxy",
            # Services
            "PhoneService_enc", "multiple_lines_enc",
            "internet_fiber", "internet_dsl", "internet_none",
            "addon_count", "service_depth",
            "OnlineSecurity_enc", "OnlineBackup_enc", "DeviceProtection_enc",
            "TechSupport_enc", "StreamingTV_enc", "StreamingMovies_enc",
            "dual_streamer", "security_bundle",
            # Payment
            "pay_electronic_check", "pay_mailed_check",
            "pay_bank_transfer_automatic", "pay_credit_card_automatic",
            # Composite risk
            "high_risk_combo",
        ]

        return df[feature_cols]


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return binary churn target (1 = churned)."""
    return (df["Churn"] == "Yes").astype(int)
