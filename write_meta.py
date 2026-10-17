"""Creates model_meta.json for a model trained by the team (so the app can verify and describe it).

Usage (after putting the team's model.joblib in this folder):
    python write_meta.py --train-years 2020 2021 2022 --test-year 2023 \
        --description "RandomForest, أعدّته زهراء علي"

You must give the years that were REALLY used for training and for testing. The script recomputes
the test metrics from the model and solar_data.csv; it cannot know whether the model saw the
test year during training, so that remains the trainer's responsibility.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

import config
import data_utils
import model_utils


def build_meta(model_path, data_path, train_years, test_year, description, is_demo=False):
    if test_year in train_years or test_year <= max(train_years):
        raise ValueError("سنة الاختبار يجب أن تأتي بعد كل سنوات التدريب (تقسيم زمني بلا تداخل).")
    bundle = model_utils.load_bundle(model_path, Path(model_path).with_name("__no_meta__.json"))
    if not bundle.usable:
        raise ValueError(bundle.message)
    df = pd.read_csv(data_path)
    missing = [c for c in bundle.features + [config.TARGET, "year", "month", "hour"] if c not in df.columns]
    if missing:
        raise ValueError("أعمدة غير موجودة في ملف البيانات: " + ", ".join(missing))
    train, test = df[df.year.isin(train_years)], df[df.year == test_year]
    if train.empty or test.empty:
        raise ValueError("لا توجد بيانات لسنوات التدريب أو الاختبار المحددة.")

    pred = model_utils.predict_power(bundle, test)
    m = data_utils.evaluate(test[config.TARGET], pred)
    gmax = train.groupby(["month", "hour"]).irradiance_plane_Wm2.max().round(1)
    meta = {
        "is_demo": bool(is_demo),
        "description": description,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "features": bundle.features,
        "target": config.TARGET,
        "target_unit": config.TARGET_UNIT,
        "target_definition": "قدرة PV ساعية (W) لنظام 1 kWp من نوع c-Si بميل 30° وخسائر 14%، محسوبة بنموذج PVGIS من الإشعاع والحرارة والرياح (محاكاة وليست قياساً).",
        "train_years": sorted(int(y) for y in train_years),
        "test_year": int(test_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "metrics_test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "feature_ranges": {f: {"min": float(train[f].min()), "max": float(train[f].max())}
                           for f in bundle.features},
        "irradiance_max_by_month_hour": {f"{mo}-{h}": float(v) for (mo, h), v in gmax.items()},
        "sklearn_version": sklearn.__version__,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "model_sha256": model_utils.sha256_file(model_path),
    }
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-years", type=int, nargs="+", required=True)
    p.add_argument("--test-year", type=int, required=True)
    p.add_argument("--description", required=True)
    a = p.parse_args()
    meta = build_meta(config.MODEL_PATH, config.DATA_PATH, a.train_years, a.test_year, a.description)
    config.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", config.META_PATH.name, "| test metrics:", meta["metrics_test"])


if __name__ == "__main__":
    main()
