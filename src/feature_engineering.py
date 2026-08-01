"""
Leakage-Safe Feature Engineering Module.

Ensures no look-ahead temporal data leakage by constructing feature representations
for each booking strictly from historical logs preceding the booking's creation timestamp.
"""

import pandas as pd
import numpy as np
from datetime import datetime


FACILITIES = ["Gym", "Swimming Pool", "Badminton Court", "Tennis Court", "Clubhouse"]


def extract_leakage_safe_features(df: pd.DataFrame, min_history: int = 2) -> pd.DataFrame:
    """
    Constructs leakage-safe features for each booking record.
    Filters out initial cold-start records with history < min_history.
    """
    df = df.copy()
    df["booking_dt"] = pd.to_datetime(df["booking_timestamp"])
    df["usage_dt"] = pd.to_datetime(df["usage_timestamp"])
    
    # Target values
    df["target_facility"] = df["facility_name"]
    df["target_usage_day"] = df["usage_dt"].dt.dayofweek # 0=Mon, 6=Sun
    df["target_usage_hour"] = df["usage_dt"].dt.hour
    
    # Calculate lead time in hours (usage - booking)
    df["target_lead_hours"] = (df["usage_dt"] - df["booking_dt"]).dt.total_seconds() / 3600.0
    
    # Sort chronologically by booking timestamp
    df = df.sort_values("booking_dt").reset_index(drop=True)
    
    features_list = []
    valid_indices = []

    # Group bookings by resident to construct rolling historical stats
    resident_groups = df.groupby("resident_id")

    for resident_id, group in resident_groups:
        group_records = group.to_dict("records")
        
        # Keep track of history prior to each booking
        history = []

        for record in group_records:
            cur_booking_dt = record["booking_dt"]
            
            # Filter history strictly before cur_booking_dt
            past_history = [h for h in history if h["booking_dt"] < cur_booking_dt]

            if len(past_history) >= min_history:
                # Calculate features from past_history
                hist_len = len(past_history)
                
                # Facility counts & ratios
                fac_counts = {f: 0 for f in FACILITIES}
                for h in past_history:
                    fac_counts[h["facility_name"]] += 1
                fac_ratios = {f"ratio_{f}": fac_counts[f] / hist_len for f in FACILITIES}
                top_facility = max(fac_counts, key=fac_counts.get)
                
                # Day of week frequencies
                day_counts = [0] * 7
                for h in past_history:
                    day_counts[h["target_usage_day"]] += 1
                day_ratios = {f"ratio_day_{d}": day_counts[d] / hist_len for d in range(7)}
                top_day = int(np.argmax(day_counts))
                
                # Hour frequencies & statistics
                hours = [h["target_usage_hour"] for h in past_history]
                avg_hour = np.mean(hours)
                std_hour = np.std(hours) if hist_len > 1 else 0.0
                
                # Lead time statistics
                leads = [h["target_lead_hours"] for h in past_history]
                avg_lead = np.mean(leads)
                
                # Recent past items (lag 1 & lag 2)
                last_1 = past_history[-1]
                last_2 = past_history[-2] if hist_len >= 2 else last_1
                
                time_since_last_booking_days = (cur_booking_dt - last_1["booking_dt"]).total_seconds() / 86400.0
                
                feat = {
                    "resident_id": resident_id,
                    "record_index": record["index"] if "index" in record else len(valid_indices),
                    "booking_timestamp": record["booking_timestamp"],
                    "usage_timestamp": record["usage_timestamp"],
                    "target_facility": record["target_facility"],
                    "target_usage_day": record["target_usage_day"],
                    "target_usage_hour": record["target_usage_hour"],
                    "target_lead_hours": record["target_lead_hours"],
                    # Historical aggregates
                    "hist_count": hist_len,
                    "top_facility": top_facility,
                    "top_day": top_day,
                    "avg_usage_hour": avg_hour,
                    "std_usage_hour": std_hour,
                    "avg_lead_hours": avg_lead,
                    "days_since_last_booking": time_since_last_booking_days,
                    "lag1_facility": last_1["facility_name"],
                    "lag1_usage_day": last_1["target_usage_day"],
                    "lag1_usage_hour": last_1["target_usage_hour"],
                    "lag2_facility": last_2["facility_name"],
                }
                
                feat.update(fac_ratios)
                feat.update(day_ratios)
                features_list.append(feat)

            # Append current record to history for future iterations
            history.append(record)

    feat_df = pd.DataFrame(features_list)
    return feat_df


if __name__ == "__main__":
    import os
    raw_path = "c:/Users/jaisw/OneDrive/Resume/Technical writer/projects/facility-usage-prediction-system/data/raw_bookings.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        feat_df = extract_leakage_safe_features(df_raw)
        print(f"Extracted features for {len(feat_df)} instances.")
        print(feat_df.head())
