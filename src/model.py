"""
Multi-Target ML Prediction Model Pipeline.

Trains models for 4 outputs:
1. Facility Reference (Classifier)
2. Usage Day (Classifier: 0=Mon, 6=Sun)
3. Usage Hour (Classifier: 6..21)
4. Notification / Nudge Time (Regressor for Lead Time -> Nudge Timestamp)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib


class FacilityPredictorPipeline:
    def __init__(self):
        self.facility_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.day_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.hour_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.lead_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        self.facility_encoder = LabelEncoder()
        self.lag1_encoder = LabelEncoder()
        self.lag2_encoder = LabelEncoder()
        self.top_fac_encoder = LabelEncoder()
        
        self.feature_cols = [
            "hist_count", "avg_usage_hour", "std_usage_hour", "avg_lead_hours",
            "days_since_last_booking", "top_day", "lag1_usage_day", "lag1_usage_hour",
            "top_facility_enc", "lag1_facility_enc", "lag2_facility_enc",
            "ratio_Gym", "ratio_Swimming Pool", "ratio_Badminton Court", "ratio_Tennis Court", "ratio_Clubhouse",
            "ratio_day_0", "ratio_day_1", "ratio_day_2", "ratio_day_3", "ratio_day_4", "ratio_day_5", "ratio_day_6"
        ]

    def _prepare_matrix(self, feat_df: pd.DataFrame, is_training: bool = False):
        df = feat_df.copy()
        
        all_facs = ["Gym", "Swimming Pool", "Badminton Court", "Tennis Court", "Clubhouse"]
        if is_training:
            self.top_fac_encoder.fit(all_facs)
            self.lag1_encoder.fit(all_facs)
            self.lag2_encoder.fit(all_facs)

        df["top_facility_enc"] = self.top_fac_encoder.transform(df["top_facility"])
        df["lag1_facility_enc"] = self.lag1_encoder.transform(df["lag1_facility"])
        df["lag2_facility_enc"] = self.lag2_encoder.transform(df["lag2_facility"])

        X = df[self.feature_cols]
        return X

    def fit(self, train_df: pd.DataFrame):
        X = self._prepare_matrix(train_df, is_training=True)
        
        # Fit encoders & targets
        self.facility_encoder.fit(["Gym", "Swimming Pool", "Badminton Court", "Tennis Court", "Clubhouse"])
        y_facility = self.facility_encoder.transform(train_df["target_facility"])
        y_day = train_df["target_usage_day"]
        y_hour = train_df["target_usage_hour"]
        y_lead = train_df["target_lead_hours"]

        self.facility_model.fit(X, y_facility)
        self.day_model.fit(X, y_day)
        self.hour_model.fit(X, y_hour)
        self.lead_model.fit(X, y_lead)
        
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_matrix(test_df, is_training=False)
        
        pred_fac_enc = self.facility_model.predict(X)
        pred_facilities = self.facility_encoder.inverse_transform(pred_fac_enc)
        
        pred_days = self.day_model.predict(X)
        pred_hours = self.hour_model.predict(X)
        pred_leads = self.lead_model.predict(X)
        
        results = test_df.copy()
        results["pred_facility"] = pred_facilities
        results["pred_usage_day"] = pred_days
        results["pred_usage_hour"] = pred_hours
        results["pred_lead_hours"] = pred_leads
        
        # Compute predicted usage timestamp and notification (nudge) timestamp
        pred_nudge_times = []
        pred_usage_times = []

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for idx, row in results.iterrows():
            booking_dt = pd.to_datetime(row["booking_timestamp"])
            
            # Predict next usage date based on predicted day of week
            target_day_idx = int(row["pred_usage_day"])
            days_ahead = (target_day_idx - booking_dt.weekday()) % 7
            if days_ahead == 0 and row["pred_usage_hour"] <= booking_dt.hour:
                days_ahead = 7 # Next week if hour has passed
            
            pred_usage_dt = (booking_dt + timedelta(days=days_ahead)).replace(
                hour=int(row["pred_usage_hour"]), minute=0, second=0
            )
            
            # Nudge time is predicted usage time minus predicted lead time
            lead_hrs = float(row["pred_lead_hours"])
            pred_nudge_dt = pred_usage_dt - timedelta(hours=lead_hrs)
            
            pred_usage_times.append(pred_usage_dt.strftime("%Y-%m-%d %H:%M"))
            pred_nudge_times.append(pred_nudge_dt.strftime("%Y-%m-%d %H:%M"))

        results["pred_usage_timestamp"] = pred_usage_times
        results["pred_nudge_timestamp"] = pred_nudge_times
        
        return results
