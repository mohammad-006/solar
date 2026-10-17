"""Single source of truth for paths, location, units and feature definitions.
The app, the validation layer, the training script and the tests all read from here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
DATA_PATH = BASE_DIR / "solar_data.csv"

# Dataset definition (PVGIS-SARAH3, Baghdad)
LATITUDE, LONGITUDE = 33.315, 44.366
UTC_OFFSET_HOURS = 3  # Baghdad local time = UTC + 3 ; dataset timestamps are UTC

TARGET = "pv_power_W"
TARGET_LABEL = "Simulated PV power for a 1 kWp c-Si system"
TARGET_UNIT = "W"
SYSTEM_KWP = 1.0  # the target is defined for a 1 kWp system

# Optional columns in an uploaded CSV
ACTUAL_COLUMN = TARGET            # if present, used to compare predicted vs reference values
TIMESTAMP_COLUMN = "datetime_utc"  # if present, used to detect duplicate timestamps

# name -> label (Arabic), unit, hard_min, hard_max, default, step, is_integer
# hard_min / hard_max are physical/engineering limits, NOT the training range.
FEATURES = {
    "irradiance_plane_Wm2": dict(label="الإشعاع الشمسي على سطح اللوح", unit="W/m²", hard_min=0.0, hard_max=1500.0, default=800.0, step=10.0),
    "air_temp_C": dict(label="درجة حرارة الهواء", unit="°C", hard_min=-50.0, hard_max=60.0, default=30.0, step=0.5),
    "wind_speed_ms": dict(label="سرعة الرياح", unit="m/s", hard_min=0.0, hard_max=60.0, default=3.0, step=0.5),
    "hour": dict(label="الساعة (UTC)", unit="0-23", hard_min=0, hard_max=23, default=9, step=1, integer=True),
    "month": dict(label="الشهر", unit="1-12", hard_min=1, hard_max=12, default=6, step=1, integer=True),
    # defined for future models; they are only shown if the loaded model actually needs them
    "sun_height_deg": dict(label="ارتفاع الشمس", unit="°", hard_min=-90.0, hard_max=90.0, default=45.0, step=1.0),
    "humidity_pct": dict(label="الرطوبة النسبية", unit="%", hard_min=0.0, hard_max=100.0, default=30.0, step=1.0),
}

# Features the project agreed on initially (used only by the demo training script)
DEFAULT_FEATURES = ["irradiance_plane_Wm2", "air_temp_C", "wind_speed_ms", "hour", "month"]

MONTH_NAMES_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                  "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]


def utc_to_local_hour(utc_hour):
    return (utc_hour + UTC_OFFSET_HOURS) % 24
