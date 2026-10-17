"""Trains the DEMO RandomForest used to test the app, and writes model.joblib + model_meta.json.

This is NOT the project team's model. The metadata marks it with is_demo = true so the app
shows it as a test model. When the real model is ready, replace model.joblib and model_meta.json
(train it with the same scikit-learn version listed in requirements.txt).

Evaluation uses time-ordered splits (never a random split): the model never sees the test year.
"""
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import config
import data_utils

FEATURES = config.DEFAULT_FEATURES
TARGET = config.TARGET
PARAMS = dict(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
TRAIN_YEARS, TEST_YEAR = (2020, 2021, 2022), 2023


def scores(y_true, y_pred):
    m = data_utils.evaluate(y_true, y_pred)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def evaluate_split(df, train_years, test_year):
    tr, te = df[df.year.isin(train_years)], df[df.year == test_year]
    rf = RandomForestRegressor(**PARAMS).fit(tr[FEATURES], tr[TARGET])
    pred = rf.predict(te[FEATURES])
    day = te.irradiance_plane_Wm2 > 0
    res = {"random_forest": scores(te[TARGET], pred),
           "random_forest_daylight_only": scores(te.loc[day, TARGET], pred[day.to_numpy()])}
    res["baselines"] = {
        "train_mean": scores(te[TARGET], np.full(len(te), tr[TARGET].mean())),
        "linear_irradiance_only": scores(
            te[TARGET], LinearRegression().fit(tr[["irradiance_plane_Wm2"]], tr[TARGET])
            .predict(te[["irradiance_plane_Wm2"]])),
        "linear_irradiance_temp_wind": scores(
            te[TARGET], LinearRegression().fit(tr[FEATURES[:3]], tr[TARGET]).predict(te[FEATURES[:3]])),
    }
    return res


def main():
    df = pd.read_csv(config.DATA_PATH)
    train = df[df.year.isin(TRAIN_YEARS)]
    test = df[df.year == TEST_YEAR]
    assert not set(train.datetime_utc) & set(test.datetime_utc), "train/test overlap"

    final = evaluate_split(df, TRAIN_YEARS, TEST_YEAR)
    folds = {f"train<= {y - 1} -> test {y}": evaluate_split(df, tuple(range(2020, y)), y)["random_forest"]
             for y in (2021, 2022, 2023)}

    model = RandomForestRegressor(**PARAMS).fit(train[FEATURES], train[TARGET])
    joblib.dump(model, config.MODEL_PATH, compress=3)

    gmax = (train.groupby(["month", "hour"]).irradiance_plane_Wm2.max().round(1))
    meta = {
        "is_demo": True,
        "description": "نموذج RandomForest تجريبي لاختبار الواجهة فقط. ليس نموذج الفريق ولا يُعتمد في نتائج المشروع.",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "features": FEATURES,
        "target": TARGET,
        "target_unit": config.TARGET_UNIT,
        "target_definition": "قدرة PV ساعية (W) لنظام 1 kWp من نوع c-Si بميل 30° وخسائر 14%، محسوبة بنموذج PVGIS من الإشعاع والحرارة والرياح (محاكاة وليست قياساً).",
        "train_years": list(TRAIN_YEARS),
        "test_year": TEST_YEAR,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "metrics_test": final["random_forest"],
        "metrics_test_daylight_only": final["random_forest_daylight_only"],
        "baselines_test": final["baselines"],
        "rolling_origin_folds": folds,
        "feature_ranges": {f: {"min": float(train[f].min()), "max": float(train[f].max())} for f in FEATURES},
        "irradiance_max_by_month_hour": {f"{m}-{h}": float(v) for (m, h), v in gmax.items()},
        "hyperparameters": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
        "sklearn_version": sklearn.__version__,
        "data_sha256": hashlib.sha256(config.DATA_PATH.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),
    }
    config.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TEST", TEST_YEAR, "(model trained on", TRAIN_YEARS, ")")
    print("  RandomForest         :", final["random_forest"])
    print("  RandomForest daylight:", final["random_forest_daylight_only"])
    for name, s in final["baselines"].items():
        print(f"  baseline {name:<28}:", s)
    print("rolling-origin:", {k: (v["mae"], v["rmse"], v["r2"]) for k, v in folds.items()})
    print("saved", config.MODEL_PATH.name, "and", config.META_PATH.name)


if __name__ == "__main__":
    main()
