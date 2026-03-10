"""
test_features.py
----------------
Unit tests for the feature engineering pipeline.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import ChurnFeatureEngineer, get_target


@pytest.fixture
def sample_df():
    """Minimal valid rows covering all code paths."""
    rows = [
        {
            "customerID": "A1", "gender": "Male", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 24,
            "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.5, "TotalCharges": 2292.0, "Churn": "Yes"
        },
        {
            "customerID": "B2", "gender": "Female", "SeniorCitizen": 1,
            "Partner": "No", "Dependents": "Yes", "tenure": 60,
            "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 45.0, "TotalCharges": 2700.0, "Churn": "No"
        },
        {
            "customerID": "C3", "gender": "Male", "SeniorCitizen": 0,
            "Partner": "No", "Dependents": "No", "tenure": 0,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 20.0, "TotalCharges": 0, "Churn": "No"
        }
    ]
    return pd.DataFrame(rows)


class TestChurnFeatureEngineer:

    def test_output_shape(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.shape[0] == 3
        assert X.shape[1] > 30, "Expected 30+ features"

    def test_no_nulls(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.isnull().sum().sum() == 0, "Output contains NaN values"

    def test_all_numeric(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert all(X.dtypes != object), "Non-numeric columns in output"

    def test_high_risk_combo(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        # Row 0: month-to-month + fiber + no security → high_risk_combo = 1
        assert X.iloc[0]["high_risk_combo"] == 1
        # Row 1: two-year contract → high_risk_combo = 0
        assert X.iloc[1]["high_risk_combo"] == 0

    def test_tenure_segment_values(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert set(X["tenure_segment"].unique()).issubset({0, 1, 2, 3})

    def test_addon_count_range(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert (X["addon_count"] >= 0).all()
        assert (X["addon_count"] <= 6).all()

    def test_no_internet_addons_zero(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        # Row 2 has no internet — all addon encs should be 0
        addon_cols = ["OnlineSecurity_enc", "OnlineBackup_enc", "DeviceProtection_enc",
                      "TechSupport_enc", "StreamingTV_enc", "StreamingMovies_enc"]
        assert X.iloc[2][addon_cols].sum() == 0

    def test_ltv_proxy(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        # LTV = tenure * monthly
        expected_0 = 24 * 95.5
        assert abs(X.iloc[0]["ltv_proxy"] - expected_0) < 0.01

    def test_dual_streamer(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.iloc[0]["dual_streamer"] == 1   # both streaming
        assert X.iloc[1]["dual_streamer"] == 0   # neither

    def test_security_bundle(self, sample_df):
        fe = ChurnFeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.iloc[1]["security_bundle"] == 1  # has both
        assert X.iloc[0]["security_bundle"] == 0  # missing tech support

    def test_fit_returns_self(self, sample_df):
        fe = ChurnFeatureEngineer()
        result = fe.fit(sample_df)
        assert result is fe

    def test_fit_transform_equals_transform(self, sample_df):
        fe = ChurnFeatureEngineer()
        X1 = fe.fit_transform(sample_df)
        X2 = fe.transform(sample_df)
        pd.testing.assert_frame_equal(X1, X2)


class TestGetTarget:

    def test_binary_output(self, sample_df):
        y = get_target(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_correct_values(self, sample_df):
        y = get_target(sample_df)
        assert y.iloc[0] == 1   # "Yes"
        assert y.iloc[1] == 0   # "No"
        assert y.iloc[2] == 0   # "No"

    def test_length(self, sample_df):
        y = get_target(sample_df)
        assert len(y) == len(sample_df)
