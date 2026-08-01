# ANACITY Facility Usage Prediction System

[![Open In Colab](https://img.shields.io/badge/Google%20Colab-Run%20Instantly-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/jaiswalwrites/community-facility-usage-prediction-system/blob/main/Facility_Usage_Prediction_System.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/jaiswalwrites/community-facility-usage-prediction-system)

A machine learning solution for predicting residential community amenity usage, optimal booking day/time, and intelligent notification push timing (nudges), built according to the **ANACITY / ANAROCK** technical assignment specification.

---

## ⚡ 1-Click Instant Online Execution (No Setup Needed!)

You can run the entire synthetic data generation, ML training, evaluation, and review table formatting directly in your browser with **zero installation or cloning**:

👉 **[Click Here to Run on Google Colab](https://colab.research.google.com/github/jaiswalwrites/community-facility-usage-prediction-system/blob/main/Facility_Usage_Prediction_System.ipynb)**

---

## 🌟 Deliverables Overview

1. **Synthetic Dataset Generator (`src/data_generator.py`)**:
   Generates 10,000+ realistic booking logs for 120 residents over 6 months, capturing resident preferences, facility popularity, day/time distributions, lead times, and noise.
2. **Leakage-Safe Prediction Pipeline (`src/model.py`, `src/feature_engineering.py`)**:
   Multi-target ensemble pipeline that predicts:
   - **Facility Reference** (Gym, Pool, Badminton, Tennis, Clubhouse)
   - **Usage Day** (Monday – Sunday)
   - **Usage Hour** (06:00 – 21:00)
   - **Notification / Nudge Timestamp** (Optimal reminder push time)
3. **Prediction Review Output & Web Dashboard (`web/`)**:
   Interactive glassmorphism web dashboard featuring:
   - Side-by-side **Prediction vs. Actual** review table formatted per Section 3.5 of the spec.
   - Overall & per-output performance metrics.
   - **Resident Prediction Simulator** for testing live inference.
   - One-click **CSV Exporter** for spreadsheet analysis (`prediction_review.csv`).
4. **Technical Documentation (`TECHNICAL_DOCUMENTATION.md`)**:
   Comprehensive document detailing data synthesis, feature choices, leakage prevention, holdout metrics, error analysis, and production limitations.

---

## 📁 Repository Structure

```
community-facility-usage-prediction-system/
├── Facility_Usage_Prediction_System.ipynb # 🚀 1-Click Google Colab Notebook
├── data/
│   ├── raw_bookings.csv          # Generated synthetic booking dataset (10,233 records)
│   ├── train_bookings.csv        # Chronological train split (7,994 records)
│   ├── test_bookings.csv         # Chronological test holdout split (1,999 records)
│   ├── prediction_review.csv     # Final prediction review output CSV
│   └── metrics_summary.json      # Evaluated test metrics summary
├── src/
│   ├── data_generator.py         # Synthetic dataset generator
│   ├── feature_engineering.py    # Rolling leakage-safe feature builder
│   ├── model.py                  # Multi-output prediction pipeline
│   └── evaluate.py               # Evaluation & review table formatter
├── web/
│   ├── index.html                # Interactive Prediction Review Web Dashboard
│   ├── styles.css                # Premium modern dark-theme styles
│   ├── app.js                    # Web dashboard interactivity & live simulator
│   └── dashboard_data.json       # Exported test predictions & metrics for Web UI
├── main.py                       # Single command pipeline orchestrator
├── requirements.txt              # Dependencies (pandas, scikit-learn, numpy)
├── TECHNICAL_DOCUMENTATION.md    # In-depth technical report
└── README.md                     # Quickstart guide
```

---

## 📊 Summary of Prediction Review Results

Formated output comparison preview (matching Section 3.5 format):

| Record Reference | Past Bookings | Prediction | Actual | Match Indicator |
| :--- | :--- | :--- | :--- | :--- |
| **R-104-#1** | Gym / Mon / 07:00<br>Gym (Top Pref) | Gym / Fri / 07:00<br>Nudge Thu / 18:15 | Gym / Fri / 07:00<br>Booked Thu / 18:15 | **YES**<br>(4 of 4) |
| **R-118-#2** | Pool / Sat / 18:00<br>Pool (Top Pref) | Pool / Sat / 18:00<br>Nudge Fri / 09:20 | Pool / Sat / 18:00<br>Booked Fri / 09:20 | **YES**<br>(4 of 4) |
| **R-126-#3** | Badminton / Thu / 20:00<br>Badminton (Top Pref) | Badminton / Fri / 20:00<br>Nudge Thu / 12:30 | Gym / Fri / 19:00<br>Booked Thu / 12:30 | **NO**<br>(2 of 4) |

---

## 📜 License
Confidential - For ANACITY / ANAROCK intended review.
