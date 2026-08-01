# ANACITY Facility Usage Prediction System

A machine learning solution for predicting residential community amenity usage, optimal booking day/time, and intelligent notification push timing (nudges), built according to the **ANACITY / ANAROCK** technical assignment specification.

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
facility-usage-prediction-system/
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

## 🚀 Quickstart Guide

### 1. Install Dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Workflow
To generate the synthetic dataset, extract leakage-safe features, train the multi-output models, evaluate on unseen holdout test data, and produce the prediction review files, execute:
```bash
python main.py
```

### 3. Open the Interactive Web Dashboard
Open `web/index.html` in any web browser to view the interactive Prediction Review Dashboard, test live predictions for any resident, and export the review table as a CSV.

```bash
# Option A: Open directly in browser
double-click web/index.html

# Option B: Run via lightweight local server
python -m http.server 8000 --directory web
# Then visit http://localhost:8000 in your browser
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
