"""Loading the saved model, checking it against its metadata, and running predictions.

SECURITY: joblib files are pickles and can run code when loaded. Only load model files
that came from the project team.
"""
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import config

UNAVAILABLE = "unavailable"   # no model file
INCOMPATIBLE = "incompatible"  # file cannot be read / features not understood
UNVERIFIED = "unverified"      # model loads, but metadata missing or does not match the file
DEMO = "demo"                  # test model with matching metadata (is_demo = true)
PROJECT = "project"            # project model with matching metadata (is_demo = false)
USABLE = {UNVERIFIED, DEMO, PROJECT}


@dataclass
class ModelBundle:
    state: str
    message: str
    model: object = None
    features: list = field(default_factory=list)
    use_names: bool = True          # pass a DataFrame (True) or a numpy array (False) to the model
    meta: dict = field(default_factory=dict)  # only metadata that was verified against the file
    notes: list = field(default_factory=list)

    @property
    def usable(self):
        return self.state in USABLE


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_meta(path):
    """Returns (meta_dict | None, problem | None)."""
    path = Path(path)
    if not path.is_file():
        return None, "ملف model_meta.json غير موجود."
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"تعذّر قراءة model_meta.json ({type(exc).__name__})."
    if not isinstance(meta, dict):
        return None, "model_meta.json ليس بالصيغة المتوقعة."
    return meta, None


def load_bundle(model_path=config.MODEL_PATH, meta_path=config.META_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        return ModelBundle(UNAVAILABLE, f"ملف النموذج ({model_path.name}) غير موجود. ضعه بجانب app.py.")

    meta, meta_problem = read_meta(meta_path)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(model_path)
    except Exception as exc:  # pickle/joblib can raise many error types for bad files
        return ModelBundle(
            INCOMPATIBLE,
            f"تعذّر قراءة ملف النموذج ({type(exc).__name__}). قد يكون تالفاً أو محفوظاً بإصدار مختلف من scikit-learn.",
        )
    notes = [str(w.message) for w in caught if "version" in str(w.message).lower()]

    if not callable(getattr(model, "predict", None)):
        return ModelBundle(INCOMPATIBLE, "الملف المحمّل ليس نموذجاً يملك دالة predict.")

    names = getattr(model, "feature_names_in_", None)
    use_names = names is not None
    if use_names:
        features = [str(n) for n in names]
    elif meta and isinstance(meta.get("features"), list):
        features = [str(n) for n in meta["features"]]
    else:
        return ModelBundle(
            INCOMPATIBLE,
            "النموذج لا يحمل أسماء الميزات ولا يوجد model_meta.json يحددها. درّب النموذج على DataFrame له أسماء أعمدة.",
        )

    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != len(features):
        return ModelBundle(INCOMPATIBLE, f"النموذج يتوقع {n_in} ميزة بينما المعرّف {len(features)}.")
    unknown = [f for f in features if f not in config.FEATURES]
    if unknown:
        return ModelBundle(
            INCOMPATIBLE,
            "ميزات غير معرّفة في config.py: " + ", ".join(unknown) + ". أضف تعريفها (الاسم، الوحدة، الحدود) ثم أعد التشغيل.",
        )

    problems = []
    if meta is None:
        problems.append(meta_problem)
    else:
        if meta.get("model_sha256") != sha256_file(model_path):
            problems.append("بصمة ملف النموذج لا تطابق model_meta.json (استُبدل النموذج دون تحديث الوصف).")
        if meta.get("features") != features:
            problems.append("ميزات model_meta.json تختلف عن ميزات النموذج أو ترتيبها.")
        if not isinstance(meta.get("is_demo"), bool):
            problems.append("الحقل is_demo غير محدد في model_meta.json.")

    if problems:
        return ModelBundle(
            UNVERIFIED, " ".join(problems) + " لذلك لا تُعرض مقاييس الدقة ولا حدود التدريب.",
            model=model, features=features, use_names=use_names, notes=notes,
        )

    if meta.get("sklearn_version") and meta["sklearn_version"] != sklearn.__version__:
        notes.append(f"النموذج دُرّب بـ scikit-learn {meta['sklearn_version']} والمثبّت {sklearn.__version__}.")
    state = DEMO if meta["is_demo"] else PROJECT
    return ModelBundle(
        state, meta.get("description", ""), model=model, features=features,
        use_names=use_names, meta=meta, notes=notes,
    )


def predict_power(bundle, X):
    """X: DataFrame that contains every model feature. Returns predicted power in W (not clipped)."""
    if not bundle.usable:
        raise RuntimeError("لا يوجد نموذج صالح للاستخدام.")
    missing = [f for f in bundle.features if f not in X.columns]
    if missing:
        raise ValueError("أعمدة مطلوبة للنموذج غير موجودة: " + ", ".join(missing))
    data = X[bundle.features]
    if data.isna().any().any() or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("المدخلات تحتوي قيماً فارغة أو غير محدودة؛ تحقق منها قبل التنبؤ.")
    arg = data if bundle.use_names else data.to_numpy(dtype=float)
    out = np.asarray(bundle.model.predict(arg), dtype=float).reshape(-1)
    if len(out) != len(data):
        raise ValueError("عدد التنبؤات لا يطابق عدد الصفوف.")
    return out


def sweep_prediction(bundle, base_row, feature, values):
    """Predicted power while one input changes and the others stay fixed."""
    rows = pd.DataFrame([{**base_row, feature: v} for v in values])
    return predict_power(bundle, rows)
