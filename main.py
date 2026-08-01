"""
Main End-to-End Orchestrator Script for Facility Usage Prediction System.

Runs the complete workflow:
1. Synthetic data generation
2. Leakage-safe feature extraction
3. Chronological train/test splitting
4. Multi-output pipeline model fitting
5. Unseen holdout prediction
6. Metric reporting & prediction review output export
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_generator import generate_synthetic_bookings
from src.feature_engineering import extract_leakage_safe_features
from src.model import FacilityPredictorPipeline
from src.evaluate import evaluate_predictions


def run_pipeline():
    print("=" * 70)
    print("      ANACITY Facility Usage Prediction System - Execution Pipeline")
    print("=" * 70)
    
    # Ensure data directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    web_dir = os.path.join(base_dir, "web")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)
    
    # Step 1: Data Generation
    print("\n[Step 1/5] Generating synthetic booking dataset for 120 residents over 180 days...")
    raw_df = generate_synthetic_bookings(num_residents=120, num_days=180, random_seed=42)
    raw_path = os.path.join(data_dir, "raw_bookings.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f" -> Successfully generated {len(raw_df)} booking records in '{raw_path}'")
    
    # Step 2: Leakage-Safe Feature Engineering
    print("\n[Step 2/5] Extracting leakage-safe rolling historical features...")
    feat_df = extract_leakage_safe_features(raw_df, min_history=2)
    print(f" -> Constructed feature matrix with {len(feat_df)} instances and {len(feat_df.columns)} columns.")

    # Step 3: Chronological Train / Test Split (Strict Holdout, No Leakage)
    print("\n[Step 3/5] Performing chronological train/test split...")
    feat_df["booking_dt"] = pd.to_datetime(feat_df["booking_timestamp"])
    feat_df = feat_df.sort_values("booking_dt").reset_index(drop=True)
    
    split_idx = int(len(feat_df) * 0.80) # 80% train, 20% test holdout
    train_df = feat_df.iloc[:split_idx].copy()
    test_df = feat_df.iloc[split_idx:].copy()

    train_path = os.path.join(data_dir, "train_bookings.csv")
    test_path = os.path.join(data_dir, "test_bookings.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f" -> Chronological Split: Train = {len(train_df)} records, Test Holdout = {len(test_df)} records")
    print(f" -> Train Date Range: {train_df['booking_timestamp'].min()} to {train_df['booking_timestamp'].max()}")
    print(f" -> Test Date Range:  {test_df['booking_timestamp'].min()} to {test_df['booking_timestamp'].max()}")

    # Step 4: Model Pipeline Training & Prediction
    print("\n[Step 4/5] Fitting multi-output prediction model pipeline...")
    pipeline = FacilityPredictorPipeline()
    pipeline.fit(train_df)
    
    print(" -> Generating predictions on unseen test holdout set...")
    results_df = pipeline.predict(test_df)

    # Step 5: Evaluation & Review Export
    print("\n[Step 5/5] Evaluating performance and formatting prediction review table...")
    metrics, review_df = evaluate_predictions(results_df)

    review_path = os.path.join(data_dir, "prediction_review.csv")
    metrics_path = os.path.join(data_dir, "metrics_summary.json")
    web_json_path = os.path.join(web_dir, "dashboard_data.json")

    review_df.to_csv(review_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    dashboard_export = {
        "metrics": metrics,
        "predictions": review_df.to_dict("records"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(web_json_path, "w") as f:
        json.dump(dashboard_export, f, indent=2)

    print("\n" + "=" * 70)
    print("                         SUMMARY OF RESULTS")
    print("=" * 70)
    print(f" Total Unseen Test Records  : {metrics['total_test_records']}")
    print(f" Facility Accuracy           : {metrics['facility_accuracy']}%")
    print(f" Usage Day Accuracy          : {metrics['usage_day_accuracy']}%")
    print(f" Usage Hour Accuracy (±1h)   : {metrics['usage_hour_accuracy']}%")
    print(f" Nudge Time Accuracy (±2.5h) : {metrics['nudge_time_accuracy']}%")
    print(f" Nudge Time MAE              : {metrics['nudge_time_mae_hours']} hours")
    print(f" -------------------------------------------------------------------")
    print(f" OVERALL EXACT 4/4 MATCH RATE: {metrics['exact_4of4_match_rate']}%")
    print(f" Match Score Breakdown       : {metrics['match_distribution']}")
    print("=" * 70)
    print(f"\nOutputs written to:")
    print(f" 1. Review Table CSV : {review_path}")
    print(f" 2. Metrics JSON     : {metrics_path}")
    print(f" 3. Dashboard Data   : {web_json_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
