"""
Central configuration for the Road Risk Analytics pipeline.
Keeping every tunable constant in a single module avoids magic numbers
scattered across the data generation, modeling, and reporting code.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "road_sensor_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")
CIVIC_CONTACTS_PATH = os.path.join(BASE_DIR, "civic_contacts.json")

for directory in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)

RANDOM_STATE = 42
N_RECORDS = 5000
TRAIN_TEST_SPLIT = 0.20

RISK_LABELS = ["Low", "Medium", "High", "Critical"]

RISK_LABEL_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#EF6C00",
    "Critical": "#C62828",
}

FEATURE_COLUMNS = [
    "rainfall_mm",
    "humidity_percent",
    "traffic_density_vph",
    "road_age_years",
    "vehicle_load_tons",
    "monsoon_intensity_index",
    "structural_fatigue_index",
    "traffic_stress_index",
    "drainage_quality_score",
]

CITY_METADATA = {
    "Bangalore": {
        "center_lat": 12.9716,
        "center_lon": 77.5946,
        "roads": [
            ("Outer Ring Road", "Marathahalli Ward", 12.9569, 77.7011),
            ("Hosur Road", "Electronic City Ward", 12.8452, 77.6602),
            ("Bannerghatta Road", "Bannerghatta Ward", 12.8988, 77.5975),
            ("Sarjapur Road", "Bellandur Ward", 12.9121, 77.6857),
            ("Old Airport Road", "HAL Ward", 12.9605, 77.6486),
            ("Mysore Road", "Kengeri Ward", 12.9081, 77.4855),
            ("Tumkur Road", "Peenya Ward", 13.0298, 77.5202),
            ("Whitefield Main Road", "Whitefield Ward", 12.9698, 77.7500),
        ],
    },
    "Mumbai": {
        "center_lat": 19.0760,
        "center_lon": 72.8777,
        "roads": [
            ("Western Express Highway", "Andheri Ward", 19.1197, 72.8468),
            ("Eastern Express Highway", "Chembur Ward", 19.0522, 72.8994),
            ("Link Road", "Malad Ward", 19.1863, 72.8493),
            ("Sion Panvel Highway", "Vashi Ward", 19.0748, 73.0007),
            ("LBS Marg", "Ghatkopar Ward", 19.0863, 72.9081),
            ("SV Road", "Bandra Ward", 19.0596, 72.8295),
            ("Ghodbunder Road", "Thane Ward", 19.2403, 72.9781),
        ],
    },
    "Delhi": {
        "center_lat": 28.7041,
        "center_lon": 77.1025,
        "roads": [
            ("Outer Ring Road", "Dhaula Kuan Ward", 28.5921, 77.1631),
            ("NH48 Highway", "Dwarka Ward", 28.5921, 77.0460),
            ("Mathura Road", "Nizamuddin Ward", 28.5877, 77.2534),
            ("Rohtak Road", "Nangloi Ward", 28.6817, 77.0563),
            ("GT Karnal Road", "Azadpur Ward", 28.7128, 77.1734),
            ("Vikas Marg", "Laxmi Nagar Ward", 28.6304, 77.2760),
        ],
    },
    "Chennai": {
        "center_lat": 13.0827,
        "center_lon": 80.2707,
        "roads": [
            ("Old Mahabalipuram Road", "Sholinganallur Ward", 12.9010, 80.2279),
            ("GST Road", "Tambaram Ward", 12.9249, 80.1000),
            ("Anna Salai", "Teynampet Ward", 13.0418, 80.2434),
            ("East Coast Road", "Neelankarai Ward", 12.9591, 80.2570),
            ("Poonamallee High Road", "Koyambedu Ward", 13.0694, 80.1948),
        ],
    },
    "Hyderabad": {
        "center_lat": 17.3850,
        "center_lon": 78.4867,
        "roads": [
            ("Outer Ring Road", "Gachibowli Ward", 17.4435, 78.3772),
            ("Tank Bund Road", "Hussain Sagar Ward", 17.4239, 78.4738),
            ("Banjara Hills Road No 12", "Banjara Hills Ward", 17.4156, 78.4347),
            ("Uppal Road", "Uppal Ward", 17.4030, 78.5590),
        ],
    },
    "Pune": {
        "center_lat": 18.5204,
        "center_lon": 73.8567,
        "roads": [
            ("Katraj Dehu Road Bypass", "Katraj Ward", 18.4515, 73.8642),
            ("Pune Satara Road", "Bibwewadi Ward", 18.4667, 73.8624),
            ("Baner Road", "Baner Ward", 18.5590, 73.7868),
            ("Nagar Road", "Vadgaon Sheri Ward", 18.5580, 73.9143),
        ],
    },
}
