"""
predict.py
----------
Load trained model and score new customers.
Outputs a risk-ranked DataFrame with churn probability, risk tier, and
recommended intervention for each customer.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
MDL_DIR = ROOT / "models"
RPT_DIR = ROOT / "reports"


def load_artifacts():
    with open(MDL_DIR / "lgbm_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MDL_DIR / "feature_engineer.pkl", "rb") as f:
        fe = pickle.load(f)
    return model, fe


def risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "Critical"
    elif prob >= 0.45:
        return "High"
    elif prob >= 0.25:
        return "Medium"
    return "Low"


def intervention(row) -> str:
    p = row["churn_prob"]
    contract = row.get("contract_enc", -1)

    if p < 0.25:
        return "Monitor — no action needed"
    if contract == 0 and p >= 0.45:
        return "Offer annual contract upgrade + discount"
    if row.get("internet_fiber", 0) == 1 and row.get("OnlineSecurity_enc", 0) == 0:
        return "Bundle security add-on — proactive outreach"
    if row.get("digital_engagement", 2) < 1:
        return "Enroll in paperless billing + auto-pay"
    if row.get("tenure", 100) < 12 and p >= 0.45:
        return "Onboarding check-in call"
    return "Loyalty reward or retention offer"


def score_customers(df_raw: pd.DataFrame) -> pd.DataFrame:
    model, fe = load_artifacts()

    customer_ids = df_raw.get("customerID", pd.RangeIndex(len(df_raw)))
    X = fe.transform(df_raw)
    probs = model.predict_proba(X)[:, 1]

    out = X.copy()
    out["customerID"]  = customer_ids.values
    out["churn_prob"]  = probs.round(4)
    out["risk_tier"]   = [risk_tier(p) for p in probs]
    out["intervention"] = out.apply(intervention, axis=1)

    out = out.sort_values("churn_prob", ascending=False).reset_index(drop=True)
    return out[["customerID", "churn_prob", "risk_tier", "intervention",
                "tenure", "contract_enc", "MonthlyCharges", "service_depth",
                "high_risk_combo", "ltv_proxy"]]


def generate_churn_report(df_raw: pd.DataFrame) -> None:
    scored = score_customers(df_raw)

    print("\n" + "=" * 65)
    print("  CHURN RISK REPORT — SaaS Pulse")
    print("=" * 65)

    tier_counts = scored["risk_tier"].value_counts()
    print(f"\n  Portfolio Risk Distribution ({len(scored):,} customers):")
    for tier in ["Critical", "High", "Medium", "Low"]:
        n = tier_counts.get(tier, 0)
        pct = n / len(scored)
        bar = "█" * int(pct * 30)
        print(f"    {tier:10s}  {n:4d}  ({pct:5.1%})  {bar}")

    critical = scored[scored["risk_tier"] == "Critical"]
    at_risk_ltv = scored[scored["churn_prob"] >= 0.45]["ltv_proxy"].sum()
    print(f"\n  Critical-tier customers : {len(critical):,}")
    print(f"  At-risk LTV exposure    : ${at_risk_ltv:,.0f}")
    print(f"\n  Top 10 At-Risk Customers:")
    print(scored.head(10).to_string(index=False))

    print(f"\n  Top Interventions Recommended:")
    print(scored["intervention"].value_counts().head(5).to_string())

    scored.to_csv(RPT_DIR / "churn_scores.csv", index=False)
    print(f"\nScores saved → reports/churn_scores.csv")


if __name__ == "__main__":
    df_raw = pd.read_csv(ROOT / "data" / "telco_churn_raw.csv")
    generate_churn_report(df_raw)
