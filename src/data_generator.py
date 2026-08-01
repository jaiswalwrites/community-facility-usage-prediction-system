"""
Synthetic Facility Booking Data Generator for Residential Community.

Generates realistic historical facility-booking data including:
- Resident preferences (Gym, Pool, Badminton, Tennis, Clubhouse)
- Time & day of week patterns (Morning gym, Evening badminton, Weekend pool)
- Variable booking lead times (nudge/booking timestamp prior to usage timestamp)
- Sparsity, imbalance, noise, and behavioral variance over time.
"""

import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_synthetic_bookings(
    num_residents: int = 120,
    start_date_str: str = "2026-01-01",
    num_days: int = 180,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic booking logs.
    
    Fields:
    - resident_id: Resident reference (e.g. R-101)
    - facility_name: Facility reference (Gym, Swimming Pool, Badminton Court, Tennis Court, Clubhouse)
    - booking_timestamp: When the booking was created (ISO format)
    - usage_timestamp: When the facility was scheduled to be used (ISO format)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    facilities = ["Gym", "Swimming Pool", "Badminton Court", "Tennis Court", "Clubhouse"]
    
    # Define resident archetypes with specific preference probabilities and typical hours
    archetypes = [
        {
            "name": "Morning Gym Enthusiast",
            "weights": [0.70, 0.10, 0.10, 0.05, 0.05],
            "preferred_hours": [6, 7, 8],
            "preferred_days": [0, 1, 2, 3, 4], # Mon-Fri
            "avg_lead_hours": 14.0, # Books night before
            "frequency": 0.65 # High activity
        },
        {
            "name": "Evening Badminton Player",
            "weights": [0.10, 0.05, 0.70, 0.10, 0.05],
            "preferred_hours": [18, 19, 20],
            "preferred_days": [0, 1, 2, 3, 4],
            "avg_lead_hours": 28.0, # Books 1 day in advance
            "frequency": 0.50
        },
        {
            "name": "Weekend Swimming & Social",
            "weights": [0.10, 0.55, 0.05, 0.10, 0.20],
            "preferred_hours": [10, 11, 15, 16, 17],
            "preferred_days": [5, 6], # Sat-Sun
            "avg_lead_hours": 6.0, # Same day or short notice
            "frequency": 0.40
        },
        {
            "name": "Tennis Specialist",
            "weights": [0.10, 0.05, 0.10, 0.70, 0.05],
            "preferred_hours": [7, 8, 17, 18],
            "preferred_days": [1, 3, 5, 6],
            "avg_lead_hours": 42.0, # Books well in advance
            "frequency": 0.35
        },
        {
            "name": "Clubhouse & Leisure User",
            "weights": [0.05, 0.15, 0.10, 0.10, 0.60],
            "preferred_hours": [14, 15, 18, 19, 20],
            "preferred_days": [4, 5, 6],
            "avg_lead_hours": 20.0,
            "frequency": 0.30
        },
        {
            "name": "Casual Multi-Facility Resident",
            "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
            "preferred_hours": [7, 8, 11, 17, 19],
            "preferred_days": [0, 1, 2, 3, 4, 5, 6],
            "avg_lead_hours": 12.0,
            "frequency": 0.25
        }
    ]

    residents = []
    for i in range(num_residents):
        res_id = f"R-{100 + i}"
        arch = random.choice(archetypes)
        residents.append({
            "id": res_id,
            "archetype": arch
        })

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    bookings = []

    for day_offset in range(num_days):
        current_day = start_date + timedelta(days=day_offset)
        day_of_week = current_day.weekday() # 0 = Mon, 6 = Sun

        for res in residents:
            arch = res["archetype"]
            
            # Day preference modifier
            day_mult = 1.6 if day_of_week in arch["preferred_days"] else 0.4
            prob = arch["frequency"] * day_mult * random.uniform(0.7, 1.3)
            
            if random.random() < prob:
                # Select facility based on archetype preferences (with 10% random noise)
                if random.random() < 0.10:
                    facility = random.choice(facilities)
                else:
                    facility = np.random.choice(facilities, p=arch["weights"])
                
                # Select hour based on preferred hours (with 15% random noise)
                if random.random() < 0.15:
                    usage_hour = random.randint(6, 21)
                else:
                    usage_hour = random.choice(arch["preferred_hours"])
                
                usage_minute = random.choice([0, 15, 30, 45])
                usage_dt = current_day.replace(hour=usage_hour, minute=usage_minute, second=0, microsecond=0)
                
                # Calculate lead time (booking timestamp before usage)
                lead_std = arch["avg_lead_hours"] * 0.25
                lead_hours = max(0.5, np.random.normal(arch["avg_lead_hours"], lead_std))
                
                booking_dt = usage_dt - timedelta(hours=lead_hours)
                
                bookings.append({
                    "resident_id": res["id"],
                    "facility_name": facility,
                    "booking_timestamp": booking_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "usage_timestamp": usage_dt.strftime("%Y-%m-%d %H:%M:%S"),
                })

    df = pd.DataFrame(bookings)
    # Sort chronologically by booking timestamp
    df["booking_dt"] = pd.to_datetime(df["booking_timestamp"])
    df = df.sort_values("booking_dt").drop(columns=["booking_dt"]).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_bookings(num_residents=120, num_days=180)
    print(f"Generated {len(df)} bookings.")
    print(df.head(10))
    df.to_csv("c:/Users/jaisw/OneDrive/Resume/Technical writer/projects/facility-usage-prediction-system/data/raw_bookings.csv", index=False)
    print("Saved to raw_bookings.csv")
