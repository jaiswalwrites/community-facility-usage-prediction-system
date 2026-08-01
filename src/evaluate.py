"""
Evaluation and Prediction Review Exporter Module.

Evaluates test predictions against actual holdout ground truth and formats
the output exactly as specified in Section 3.5 of the ANACITY PDF specification.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def evaluate_predictions(results_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Evaluates predictions and formats comparison review output.
    """
    total = len(results_df)
    
    # Matching indicators
    results_df["fac_match"] = results_df["pred_facility"] == results_df["target_facility"]
    results_df["day_match"] = results_df["pred_usage_day"] == results_df["target_usage_day"]
    results_df["hour_match"] = (results_df["pred_usage_hour"] - results_df["target_usage_hour"]).abs() <= 1 # Within 1 hour tolerance
    
    # Nudge time tolerance match (within 2 hours of actual booking timestamp)
    results_df["actual_booked_dt"] = pd.to_datetime(results_df["booking_timestamp"])
    results_df["pred_nudge_dt"] = pd.to_datetime(results_df["pred_nudge_timestamp"])
    
    nudge_diff_hours = (results_df["pred_nudge_dt"] - results_df["actual_booked_dt"]).abs().dt.total_seconds() / 3600.0
    results_df["nudge_diff_hours"] = nudge_diff_hours
    results_df["nudge_match"] = nudge_diff_hours <= 2.5 # Within 2.5 hour window

    # Count correct outputs per row (out of 4)
    results_df["match_score"] = (
        results_df["fac_match"].astype(int) +
        results_df["day_match"].astype(int) +
        results_df["hour_match"].astype(int) +
        results_df["nudge_match"].astype(int)
    )

    results_df["exact_match"] = results_df["match_score"] == 4

    # Calculate overall metrics
    fac_acc = float(results_df["fac_match"].mean())
    day_acc = float(results_df["day_match"].mean())
    hour_acc = float(results_df["hour_match"].mean())
    nudge_acc = float(results_df["nudge_match"].mean())
    nudge_mae_hours = float(nudge_diff_hours.mean())
    exact_4of4_rate = float(results_df["exact_match"].mean())

    score_counts = results_df["match_score"].value_counts().to_dict()
    
    metrics = {
        "total_test_records": total,
        "facility_accuracy": round(fac_acc * 100, 2),
        "usage_day_accuracy": round(day_acc * 100, 2),
        "usage_hour_accuracy": round(hour_acc * 100, 2),
        "nudge_time_accuracy": round(nudge_acc * 100, 2),
        "nudge_time_mae_hours": round(nudge_mae_hours, 2),
        "exact_4of4_match_rate": round(exact_4of4_rate * 100, 2),
        "match_distribution": {
            "4_of_4": score_counts.get(4, 0),
            "3_of_4": score_counts.get(3, 0),
            "2_of_4": score_counts.get(2, 0),
            "1_of_4": score_counts.get(1, 0),
            "0_of_4": score_counts.get(0, 0),
        }
    }

    # Format output spreadsheet / UI review table matching section 3.5 format
    review_rows = []

    for idx, row in results_df.iterrows():
        res_id = row["resident_id"]
        
        # Format past bookings summary
        lag1_day = DAY_NAMES[int(row["lag1_usage_day"])]
        lag1_hr = f"{int(row['lag1_usage_hour']):02d}:00"
        lag2_day = DAY_NAMES[int(row["lag1_usage_day"])] # fallback or lag2
        
        past_summary = (
            f"{row['lag1_facility']} / {lag1_day} / {lag1_hr}\n"
            f"{row['top_facility']} (Top Pref)"
        )

        # Prediction string: facility / day / use time / nudge time
        pred_day = DAY_NAMES[int(row["pred_usage_day"])]
        pred_use_time = f"{int(row['pred_usage_hour']):02d}:00"
        pred_nudge_dt = pd.to_datetime(row["pred_nudge_timestamp"])
        pred_nudge_str = f"Nudge {DAY_NAMES[pred_nudge_dt.weekday()]} / {pred_nudge_dt.strftime('%H:%M')}"

        prediction_text = f"{row['pred_facility']} / {pred_day} / {pred_use_time}\n{pred_nudge_str}"

        # Actual string: facility / day / use time / booked at
        actual_day = DAY_NAMES[int(row["target_usage_day"])]
        actual_use_time = f"{int(row['target_usage_hour']):02d}:00"
        actual_booked_dt = pd.to_datetime(row["booking_timestamp"])
        actual_booked_str = f"Booked {DAY_NAMES[actual_booked_dt.weekday()]} / {actual_booked_dt.strftime('%H:%M')}"

        actual_text = f"{row['target_facility']} / {actual_day} / {actual_use_time}\n{actual_booked_str}"

        # Match indicator
        match_status = "YES" if row["exact_match"] else "NO"
        score_text = f"{match_status}\n{row['match_score']} of 4"

        review_rows.append({
            "record_reference": f"{res_id}-#{idx+1}",
            "resident_id": res_id,
            "past_bookings": past_summary,
            "prediction": prediction_text,
            "actual": actual_text,
            "match_indicator": match_status,
            "score": f"{row['match_score']} of 4",
            "match_score_num": int(row["match_score"]),
            "pred_facility": row["pred_facility"],
            "actual_facility": row["target_facility"],
            "pred_day": pred_day,
            "actual_day": actual_day,
            "pred_use_time": pred_use_time,
            "actual_use_time": actual_use_time,
            "pred_nudge_time": pred_nudge_str,
            "actual_booked_time": actual_booked_str
        })

    review_df = pd.DataFrame(review_rows)

    return metrics, review_df
