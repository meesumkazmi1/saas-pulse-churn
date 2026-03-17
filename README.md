# SaaS Pulse — Customer Churn Prediction

> **End-to-end churn prediction system** built on the IBM Telco Customer Churn dataset (7,043 customers, 21 features). Gradient boosting classifiers (LightGBM + XGBoost) with business-grade feature engineering, SHAP explainability, threshold optimization, and an actionable risk-scoring output — framed for a B2B SaaS retention use case.

---

## Business Context

Monthly recurring revenue is the lifeblood of any SaaS company. Even a 5% improvement in churn rate can increase revenue by 25–95% (Bain & Company). This project answers a single, high-stakes product question:

> **Which customers are likely to churn next month, why, and what should we do about it?**

The output isn't just a model — it's a ranked intervention list that maps each customer to a specific retention action, with estimated LTV at risk surfaced for prioritization.

---

## Project Structure

```
saas-pulse-churn/
├── data/
│   └── telco_churn_raw.csv          # IBM Telco Customer Churn dataset (7,043 rows)
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory data analysis
├── src/
│   ├── features.py                  # ChurnFeatureEngineer (sklearn transformer)
│   ├── train.py                     # Full training + evaluation pipeline
│   └── predict.py                   # Score new customers → risk report
├── models/
│   ├── lgbm_churn_model.pkl         # Serialized LightGBM model
│   └── feature_engineer.pkl         # Serialized feature transformer
├── reports/
│   ├── metrics.json                 # All model metrics
│   ├── roc_curves.png               # ROC comparison
│   ├── pr_curves.png                # Precision-recall curves
│   ├── shap_lightgbm.png            # SHAP feature importance
│   ├── calibration.png              # Probability calibration
│   ├── cv_scores.png                # 5-fold CV comparison
│   ├── confusion_*.png              # Per-model confusion matrices
│   ├── threshold_*.png              # Threshold sensitivity analysis
│   ├── feature_importance_*.png     # Feature importance per model
│   └── churn_scores.csv             # Scored + ranked customer output
├── tests/
│   └── test_features.py             # 15 unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — a widely-used telecom churn benchmark dataset publicly available on Kaggle.

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 21 (demographics, services, financials, contract) |
| Target | `Churn` (Yes/No) — 30.3% positive rate |
| License | Public domain / IBM Watson Analytics sample data |

**Key columns:**
- `tenure` — months as a customer
- `Contract` — Month-to-month / One year / Two year
- `MonthlyCharges`, `TotalCharges`
- `InternetService`, `OnlineSecurity`, `TechSupport` + 4 other add-ons
- `PaymentMethod`, `PaperlessBilling`

---

## Feature Engineering

35 model-ready features engineered from 21 raw columns. Highlights:

| Feature | Description | Business Logic |
|---|---|---|
| `ltv_proxy` | `tenure × MonthlyCharges` | Prioritize high-value at-risk customers |
| `spend_contract_ratio` | `MonthlyCharges / (contract_enc + 1)` | High spend on short contract = risk signal |
| `high_risk_combo` | Month-to-month + Fiber + No Security | Known high-churn cohort flag |
| `service_depth` | Count of active services (0–9 scale) | Engagement proxy — more services = stickier |
| `digital_engagement` | Paperless billing + auto-pay | Digitally engaged customers churn less |
| `tenure_segment` | 0–6 / 7–24 / 25–48 / 49–72 months | Early-life vs. loyal customer segmentation |
| `addon_count` | Sum of 6 internet add-on services | Upsell depth and switching cost proxy |
| `dual_streamer` | TV + Movies streaming active | Bundle lock-in indicator |
| `avg_monthly_spend` | `TotalCharges / tenure` | Detects spend drift over time |

---

## Models

Three models trained and evaluated:

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Recall |
|---|---|---|---|---|
| LightGBM | 0.7092 ± 0.0089 | 0.7348 | 0.537 | 0.622 |
| XGBoost | 0.7161 ± 0.0080 | 0.7452 | 0.544 | 0.624 |
| Logistic Regression (baseline) | 0.7412 ± 0.0134 | 0.7521 | 0.570 | 0.734 |

> **Why LightGBM as the production model?** Faster inference, lower memory, and native handling of categorical-like ordinal features make it the practical choice at scale — AUC is comparable to XGBoost.

**Evaluation suite includes:**
- Stratified 5-fold cross-validation
- ROC + Precision-Recall curves
- Confusion matrix at default threshold (0.5)
- Threshold sensitivity analysis (P/R/F1 across 0.1–0.9)
- SHAP summary plot (TreeExplainer)
- Probability calibration plot

---

## Churn Risk Output

Customers are scored and segmented into four risk tiers:

| Tier | Threshold | Action |
|---|---|---|
| **Critical** | ≥ 0.70 | Immediate retention outreach |
| **High** | 0.45–0.69 | Proactive offer + check-in |
| **Medium** | 0.25–0.44 | Monitor + soft nudge |
| **Low** | < 0.25 | Business as usual |

**Portfolio summary on 7,043 customers:**
- Critical: 1,499 customers (21.3%)
- At-risk LTV exposure: ~$7M
- Top intervention: "Offer annual contract upgrade + discount" (2,525 customers)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models
cd src && python train.py

# 3. Score customers + generate risk report
python predict.py

# 4. Run tests
cd .. && pytest tests/ -v
```

---

## Key Findings

1. **Contract type is the single strongest churn predictor** — month-to-month customers churn at 44.9% vs. 9.6% for two-year contracts.

2. **Fiber optic customers churn 59% more than DSL/no-internet** — despite being the premium product. Points to a product-market fit or competitive pricing problem.

3. **Customers without OnlineSecurity + TechSupport churn at nearly 2× the rate** of customers with both — these add-ons create meaningful switching costs.

4. **New customers (tenure < 6 months) are the highest-risk cohort** — onboarding intervention window is narrow.

5. **SHAP analysis confirms tenure, MonthlyCharges, contract type, and fiber service as the top 4 features** — aligning with business intuition.

---

## Tech Stack

- **Python 3.12** — NumPy, Pandas, Scikit-learn
- **LightGBM / XGBoost** — gradient boosting classifiers
- **SHAP** — TreeExplainer for model interpretability
- **Matplotlib / Seaborn** — evaluation plots
- **pytest** — 15 unit tests on feature engineering logic

---

## Author

**Meesum** — Data Scientist @ Cisco  
MS Business Analytics, University of Wisconsin–Madison
