# Technical Documentation: Facility Usage Prediction System

**Author / Project**: ANACITY Residential Community Analytics  
**System**: Facility Usage & Notification Prediction Engine  
**Version**: 1.0 (Production Candidate)

---

## 1. Executive Summary

Residential communities offer shared amenities such as gyms, swimming pools, badminton courts, tennis courts, and clubhouses. Predicting resident booking behavior allows property management systems to proactively optimize facility resource allocation, prevent overcrowding, and deliver personalized push notifications (nudges) at the exact moment a resident is most likely to make a booking.

This document details the end-to-end design, synthetic dataset synthesis, leakage-safe feature engineering pipeline, machine learning modeling strategy, holdout evaluation performance, and operational limitations of the **Facility Usage Prediction System**.

---

## 2. Synthetic Dataset Generation (`src/data_generator.py`)

To evaluate the predictive workflow under realistic operational conditions, a synthetic dataset generator was built to simulate 6 months of historical booking activity across 120 residential community members.

### Core Data Schema
| Data Field | Type | Description |
| :--- | :--- | :--- |
| `resident_id` | String | Unique resident identifier (e.g. `R-101` to `R-220`) |
| `facility_name` | String | Booked facility (`Gym`, `Swimming Pool`, `Badminton Court`, `Tennis Court`, `Clubhouse`) |
| `booking_timestamp` | Timestamp | Exact date and time when the resident initiated the booking |
| `usage_timestamp` | Timestamp | Scheduled date and time when the facility will be used |

### Realistic Data Synthesis Mechanics
1. **Resident Behavioral Archetypes**: Residents are assigned distinct preference distributions (e.g. *Morning Gym Enthusiasts*, *Evening Badminton Players*, *Weekend Pool & Social*, *Tennis Specialists*, *Casual Multi-Facility Users*).
2. **Temporal & Day Patterns**: Peak usage windows (e.g. Gym: 06:00-08:00 AM; Badminton: 18:00-20:00 PM; Pool: Sat-Sun afternoons).
3. **Variable Lead Times**: Logarithmic/Gaussian lead time distributions between `booking_timestamp` and `usage_timestamp` ranging from 2 hours to 48 hours.
4. **Behavioral Noise & Sparsity**: 10-15% random noise added to capture spontaneous bookings, inactive periods, and shifting personal routines.

---

## 3. Feature Engineering & Temporal Leakage Controls (`src/feature_engineering.py`)

### Temporal Data Leakage Prevention Strategy
To guarantee zero look-ahead bias (data leakage), the system enforces strict chronological ordering. For any target booking record occurring at timestamp $T_{\text{booking}}$, feature extraction is restricted strictly to historical bookings by that resident where:
$$T_{\text{past\_booking}} < T_{\text{booking}}$$

No future or concurrent information is accessible during feature construction.

### Engineered Feature Taxonomy
1. **Historical Frequency Ratios**: Proportion of total past bookings allocated to each of the 5 facilities ($Ratio_{\text{Gym}}, Ratio_{\text{Pool}}, \dots$).
2. **Temporal Distribution Metrics**: Historical day-of-week ratio ($Ratio_{\text{Day } 0..6}$), mean usage hour ($\mu_{\text{hour}}$), and standard deviation ($\sigma_{\text{hour}}$).
3. **Lead Time Historical Mean**: Average historical lead time ($\bar{L}_{\text{hours}}$) used to project nudge notification triggers.
4. **Sequential Lag Features**: Immediately preceding facility choices ($Lag_1, Lag_2$), previous usage hour, and day gap since last reservation ($t - t_{\text{last}}$).

---

## 4. Modeling Approach (`src/model.py`)

Predicting next facility usage involves a multi-target learning objective. A multi-output ensemble classifier/regressor architecture was implemented:

```
                          ┌──────────────────────────┐
                          │  Leakage-Safe Features   │
                          └────────────┬─────────────┘
                                       │
           ┌───────────────────┬───────┴───────────┬───────────────────┐
           ▼                   ▼                   ▼                   ▼
 ┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐ ┌───────────────┐
 │  Facility Model   │ │   Day Model   │ │    Hour Model     │ │  Lead Model   │
 │ (RandomForestClf) │ │(RandomForest) │ │ (RandomForestClf) │ │(RandomForest) │
 └─────────┬─────────┘ └───────┬───────┘ └─────────┬─────────┘ └───────┬───────┘
           │                   │                   │                   │
           ▼                   ▼                   ▼                   ▼
    Facility Name          Usage Day           Usage Hour          Lead Time
 (Gym, Pool, Court)       (Mon - Sun)           (06:00)             (Hours)
                               │                   │                   │
                               └─────────┬─────────┴───────────────────┘
                                         ▼
                             Predicted Usage Timestamp
                                         │
                                         ▼ (Subtract Lead Time)
                             Optimal Nudge Notification
```

- **Output 1 (Facility Name)**: Multi-class Random Forest Classifier predicting 1 of 5 facilities.
- **Output 2 (Usage Day)**: 7-class Random Forest Classifier predicting day of week (Monday–Sunday).
- **Output 3 (Usage Hour)**: Multi-class Classifier predicting hour of day (06:00 to 21:00).
- **Output 4 (Notification Nudge Time)**: Random Forest Regressor predicting lead time prior to usage, subtracted from predicted usage timestamp to derive $Timestamp_{\text{Nudge}}$.

---

## 5. Evaluation & Results (`src/evaluate.py`)

### Chronological Holdout Setup
- **Total Dataset**: 10,233 total bookings.
- **Train Set (80%)**: 7,994 records (Jan 2, 2026 – May 25, 2026).
- **Unseen Test Holdout (20%)**: 1,999 records (May 25, 2026 – Jun 29, 2026).

### Performance Metrics
| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Facility Accuracy** | **55.63%** | Accurately predicts amenity based on resident preference weights |
| **Usage Day Accuracy** | **73.14%** | Captures recurring weekly schedules (e.g. Mon/Wed/Fri Gym) |
| **Usage Hour Accuracy** | **54.08%** | Predicts peak daily usage windows within $\pm 1$ hour |
| **Nudge Time Accuracy** | **31.12%** | Notification window within $\pm 2.5$ hours of actual booking creation |
| **Nudge Time MAE** | **23.79 hrs** | Mean absolute error of lead time estimation |
| **Exact 4/4 Match Rate** | **13.26%** | Percentage of test cases where ALL 4 outputs matched ground truth |

### Match Score Breakdown across Unseen Test Records
- **4 of 4 Match (Perfect)**: 265 records
- **3 of 4 Match**: 514 records
- **2 of 4 Match**: 627 records
- **1 of 4 Match**: 421 records
- **0 of 4 Match**: 172 records

---

## 6. Limitations & Future Work

1. **Cold-Start Problem**: New residents with fewer than 2 historical bookings cannot generate rolling statistical features. Current fallback uses community-wide facility averages.
2. **Behavioral Drift**: Resident habits shift seasonally (e.g., higher swimming pool usage in summer months). Future iterations will introduce exponential time-decay weighting on historical features.
3. **Sequential Deep Learning**: Replacing Random Forests with a Recurrent Neural Network (LSTM / GRU) or Transformer-based sequence model to better model long-term temporal dependencies.
