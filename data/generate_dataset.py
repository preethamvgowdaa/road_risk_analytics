"""
Synthetic road sensor dataset generator.

Every civic body that tried to publish this kind of dataset would pull it
from IoT sensors embedded in road segments, weather station feeds, and
traffic cameras. Since no public India wide feed of that kind exists for a
student project, this script builds a physically plausible synthetic
equivalent: it samples repeated sensor readings across named road segments
in six Indian cities, across a full monsoon and post monsoon cycle, and
derives a risk label from a domain informed (not purely random) formula so
that the downstream machine learning model has real structure to learn.
"""

import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.config import CITY_METADATA, N_RECORDS, RANDOM_STATE, RAW_DATA_PATH


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def seasonal_rainfall(day_of_year: int, city: str) -> float:
    """Approximate a monsoon curve rather than uniform noise.

    Peninsular and western coastal cities (Mumbai) see a sharper June to
    September peak than the northern plains (Delhi), so each city gets its
    own peak day and amplitude before random daily variation is layered on.
    """
    city_profile = {
        "Mumbai": (180, 55, 25),
        "Bangalore": (200, 30, 12),
        "Chennai": (300, 28, 14),
        "Delhi": (200, 35, 10),
        "Hyderabad": (210, 32, 12),
        "Pune": (190, 34, 14),
    }
    peak_day, amplitude, base = city_profile.get(city, (190, 30, 10))
    seasonal_component = amplitude * np.exp(-((day_of_year - peak_day) ** 2) / (2 * 40 ** 2))
    daily_noise = np.random.gamma(shape=1.5, scale=4.0)
    return max(0.0, base * 0.2 + seasonal_component + daily_noise)


def compute_drainage_quality(road_age_years: float, ward_seed: int) -> float:
    """Older wards were built before modern stormwater drainage norms.

    A ward level random offset represents municipal maintenance quality
    that does not depend purely on the road's own age, mirroring how two
    roads of the same age in different wards drain very differently.
    """
    rng = np.random.RandomState(ward_seed)
    ward_baseline = rng.uniform(35, 90)
    age_penalty = min(30.0, road_age_years * 1.1)
    noise = np.random.normal(0, 4)
    score = ward_baseline - age_penalty + noise
    return float(np.clip(score, 5, 100))


def derive_risk_label(row: pd.Series) -> str:
    """Domain weighted composite score converted into four risk tiers.

    The weights below reflect civil engineering intuition rather than an
    arbitrary random assignment: water exposure (rainfall combined with
    poor drainage) and structural fatigue (age combined with heavy vehicle
    load) dominate real world road failure, so they are weighted highest.
    """
    rainfall_term = 0.28 * (row["rainfall_mm"] / 120.0)
    humidity_term = 0.10 * (row["humidity_percent"] / 100.0)
    traffic_term = 0.16 * (row["traffic_density_vph"] / 3000.0)
    age_term = 0.20 * (row["road_age_years"] / 30.0)
    load_term = 0.16 * (row["vehicle_load_tons"] / 40.0)
    drainage_term = 0.10 * (1 - row["drainage_quality_score"] / 100.0)

    composite = rainfall_term + humidity_term + traffic_term + age_term + load_term + drainage_term
    composite += np.random.normal(0, 0.006)
    composite = float(np.clip(composite, 0, 1.4))

    if composite < 0.32:
        return "Low"
    elif composite < 0.42:
        return "Medium"
    elif composite < 0.50:
        return "High"
    else:
        return "Critical"


def inject_missing_values(df: pd.DataFrame, fraction: float = 0.03) -> pd.DataFrame:
    """Simulate real sensor dropouts so the cleaning stage has actual work to do."""
    df = df.copy()
    sensor_cols = ["rainfall_mm", "humidity_percent", "traffic_density_vph", "vehicle_load_tons"]
    n_rows = len(df)
    for col in sensor_cols:
        drop_idx = np.random.choice(n_rows, size=int(n_rows * fraction), replace=False)
        df.loc[drop_idx, col] = np.nan
    return df


def generate_dataset(n_records: int = N_RECORDS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    seed_everything(seed)

    road_segments = []
    for city, meta in CITY_METADATA.items():
        for road_name, ward, lat, lon in meta["roads"]:
            road_segments.append(
                {
                    "city": city,
                    "road_name": road_name,
                    "ward": ward,
                    "base_lat": lat,
                    "base_lon": lon,
                }
            )

    start_date = datetime(2025, 1, 1)
    records = []

    readings_per_segment = max(1, n_records // len(road_segments))

    for segment in road_segments:
        ward_seed = abs(hash(segment["ward"])) % (2 ** 31)
        road_age_years = np.random.RandomState(ward_seed + 1).uniform(2, 28)
        drainage_quality_score = compute_drainage_quality(road_age_years, ward_seed)

        for _ in range(readings_per_segment):
            day_offset = random.randint(0, 364)
            reading_date = start_date + timedelta(days=day_offset)
            day_of_year = reading_date.timetuple().tm_yday

            rainfall_mm = seasonal_rainfall(day_of_year, segment["city"])
            humidity_percent = float(np.clip(45 + rainfall_mm * 0.6 + np.random.normal(0, 6), 20, 100))
            hour_weight = random.choice([0.6, 0.8, 1.0, 1.0, 1.2, 1.3])
            traffic_density_vph = float(np.clip(np.random.normal(1400, 550) * hour_weight, 100, 4000))
            vehicle_load_tons = float(np.clip(np.random.gamma(shape=4.0, scale=3.2), 2, 45))

            jitter_lat = segment["base_lat"] + np.random.normal(0, 0.006)
            jitter_lon = segment["base_lon"] + np.random.normal(0, 0.006)

            record = {
                "record_id": f"{segment['city'][:3].upper()}-{len(records) + 1:05d}",
                "date": reading_date.strftime("%Y-%m-%d"),
                "city": segment["city"],
                "ward": segment["ward"],
                "road_name": segment["road_name"],
                "latitude": round(jitter_lat, 6),
                "longitude": round(jitter_lon, 6),
                "rainfall_mm": round(rainfall_mm, 2),
                "humidity_percent": round(humidity_percent, 2),
                "traffic_density_vph": round(traffic_density_vph, 1),
                "road_age_years": round(road_age_years, 1),
                "vehicle_load_tons": round(vehicle_load_tons, 2),
                "drainage_quality_score": round(drainage_quality_score, 2),
            }
            records.append(record)

    df = pd.DataFrame(records)

    df["risk_label"] = df.apply(derive_risk_label, axis=1)

    df = inject_missing_values(df, fraction=0.03)

    anomaly_idx = np.random.choice(len(df), size=max(1, int(len(df) * 0.01)), replace=False)
    df.loc[anomaly_idx, "traffic_density_vph"] = df.loc[anomaly_idx, "traffic_density_vph"] * np.random.uniform(3, 5)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["record_id"] = [f"RRA-{i + 1:05d}" for i in range(len(df))]

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(dataset)} records across {dataset['city'].nunique()} cities.")
    print(f"Saved to {RAW_DATA_PATH}")
    print(dataset["risk_label"].value_counts())
