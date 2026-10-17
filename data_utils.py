"""Reading and validating input data. No Streamlit and no model code here.

Rule: nothing is silently changed. Empty or invalid values are reported, never replaced by 0,
and rows are never dropped without being listed in the results.
"""
import csv
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MISSING_TOKENS = {"", "nan", "na", "n/a", "null", "none"}

ERROR, WARNING = "error", "warning"
MESSAGES = {
    "EMPTY": "قيمة فارغة",
    "NOT_NUMERIC": "قيمة غير رقمية",
    "NOT_FINITE": "قيمة غير محدودة (Infinity)",
    "OUT_OF_BOUNDS": "خارج الحدود الفيزيائية/الهندسية",
    "NOT_INTEGER": "يجب أن تكون عدداً صحيحاً",
    "OUT_OF_TRAINING_RANGE": "خارج نطاق بيانات التدريب (النتيجة قد تكون أقل موثوقية)",
    "IRRADIANCE_ABOVE_HISTORY": "الإشعاع أعلى من أي قيمة مسجلة في التدريب لنفس الشهر والساعة",
    "DUP_TIMESTAMP": "الختم الزمني مكرر",
    "DUP_ROW": "صف مطابق لصف سابق في كل الميزات (قد يكون مشروعاً)",
}


class DataError(ValueError):
    """The file cannot be used at all (message is meant for the user)."""


def read_uploaded_csv(raw):
    """bytes -> DataFrame of strings (nothing is converted yet). Raises DataError."""
    if raw is None or len(raw.strip()) == 0:
        raise DataError("الملف فارغ.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DataError("حجم الملف أكبر من الحد المسموح (50 MB).")
    text = None
    for enc in ("utf-8-sig", "cp1256"):  # UTF-8 (with/without BOM), then Arabic Windows
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None or "\x00" in text:
        raise DataError("الملف ليس ملف CSV نصياً. احفظه بترميز UTF-8.")
    kwargs = dict(dtype=str, keep_default_na=False)
    try:
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", **kwargs)
        except csv.Error:  # delimiter cannot be sniffed (e.g. a single column)
            df = pd.read_csv(io.StringIO(text), sep=",", **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataError(f"تعذّرت قراءة الملف كـ CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    dup = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if dup:
        raise DataError("أسماء أعمدة مكررة: " + ", ".join(dup))
    if len(df) == 0:
        raise DataError("الملف يحتوي على الترويسة فقط، ولا توجد صفوف بيانات.")
    return df


def parse_numeric(series):
    """Returns (float values, Series of problem codes or None). Never fills or guesses values."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(float)
        codes = pd.Series(np.nan, index=series.index, dtype=object)
        codes[vals.isna()] = "EMPTY"
    else:
        s = series.astype("string").str.strip().fillna("")
        empty = s.str.lower().isin(MISSING_TOKENS)
        vals = pd.to_numeric(s.mask(empty, "nan"), errors="coerce").astype(float)
        codes = pd.Series(np.nan, index=series.index, dtype=object)
        codes[empty] = "EMPTY"
        codes[vals.isna() & ~empty] = "NOT_NUMERIC"
    codes[np.isinf(vals)] = "NOT_FINITE"
    return vals, codes


@dataclass
class Validation:
    values: pd.DataFrame            # parsed float values of the model features
    issues: pd.DataFrame            # row_id, level, code, column, message
    valid_mask: pd.Series           # rows that can be sent to the model
    missing_columns: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    hints: list = field(default_factory=list)

    @property
    def errors(self):
        return self.issues[self.issues.level == ERROR]

    @property
    def warnings(self):
        return self.issues[self.issues.level == WARNING]


def validate_frame(raw, features, ranges=None, irradiance_max=None, timestamp_col=None):
    """Validate a table of raw values (strings from a file, or numbers from the UI).

    ranges:         {feature: {"min": x, "max": y}} from training data (warnings only)
    irradiance_max: {"month-hour": max irradiance seen in training}
    Errors make a row unusable; warnings never block a prediction.
    """
    rows = []

    def add(mask, level, code, column, extra=""):
        for pos in np.flatnonzero(np.asarray(mask, dtype=bool)):
            rows.append((int(pos) + 1, level, code, column, MESSAGES[code] + extra))

    missing = [f for f in features if f not in raw.columns]
    extra_cols = [c for c in raw.columns if c not in features]
    norm = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}
    hints = [f"العمود «{f}» غير موجود، لكن يوجد «{norm[f.lower()]}» (الأسماء حساسة لحالة الأحرف)."
             for f in missing if f.lower() in norm]

    values = pd.DataFrame(index=raw.index)
    for f in features:
        if f not in raw.columns:
            continue
        spec = config.FEATURES[f]
        vals, codes = parse_numeric(raw[f])
        values[f] = vals
        for code in ("EMPTY", "NOT_NUMERIC", "NOT_FINITE"):
            add(codes == code, ERROR, code, f)
        ok = vals.notna() & np.isfinite(vals)
        out = ok & ((vals < spec["hard_min"]) | (vals > spec["hard_max"]))
        add(out, ERROR, "OUT_OF_BOUNDS", f, f" [{spec['hard_min']}, {spec['hard_max']}] {spec['unit']}")
        if spec.get("integer"):
            add(ok & ~out & (vals != np.round(vals)), ERROR, "NOT_INTEGER", f)
        if ranges and f in ranges:
            lo, hi = ranges[f]["min"], ranges[f]["max"]
            add(ok & ~out & ((vals < lo) | (vals > hi)), WARNING, "OUT_OF_TRAINING_RANGE", f,
                f" [{lo:g}, {hi:g}]")

    irr, month, hour = "irradiance_plane_Wm2", "month", "hour"
    if irradiance_max and all(c in values.columns for c in (irr, month, hour)):
        usable = values[[irr, month, hour]].apply(lambda c: c.notna() & np.isfinite(c)).all(axis=1)
        usable &= values[month].between(1, 12) & values[hour].between(0, 23)
        keys = (values[month].where(usable, 0).astype(int).astype(str) + "-"
                + values[hour].where(usable, 0).astype(int).astype(str))
        limit = keys.map(irradiance_max).astype(float) * 1.1 + 10
        add(usable & (values[irr] > limit), WARNING, "IRRADIANCE_ABOVE_HISTORY", irr)

    if timestamp_col and timestamp_col in raw.columns:
        ts = raw[timestamp_col].astype("string").str.strip()
        add(ts.duplicated(keep="first") & ts.notna() & (ts != ""), WARNING, "DUP_TIMESTAMP", timestamp_col)
    elif not missing and len(values.columns):
        add(values.duplicated(keep="first") & values.notna().all(axis=1), WARNING, "DUP_ROW", "")

    issues = pd.DataFrame(rows, columns=["row_id", "level", "code", "column", "message"])
    bad_ids = set(issues.loc[issues.level == ERROR, "row_id"])
    valid = pd.Series([(i + 1) not in bad_ids for i in range(len(raw))], index=raw.index)
    if missing:
        valid[:] = False
    return Validation(values, issues, valid, missing, extra_cols, hints)


def assemble_results(raw, validation, power_w, interval_hours=1.0, features=()):
    """One output row per input row. Rows with errors are kept, listed, and not predicted.

    power_w: predictions for the valid rows only (same order as the valid rows).
    """
    n = len(raw)
    out = pd.DataFrame({"row_id": np.arange(1, n + 1)}, index=raw.index)
    for f in features:
        out[f] = raw[f]
    for extra in (config.TIMESTAMP_COLUMN, config.ACTUAL_COLUMN):
        if extra in raw.columns and extra not in out.columns:
            out[extra] = raw[extra]

    iss = validation.issues
    err_ids = set(iss.loc[iss.level == ERROR, "row_id"])
    warn_ids = set(iss.loc[iss.level == WARNING, "row_id"])
    out["status"] = [
        "invalid" if i in err_ids else ("warning" if i in warn_ids else "valid") for i in out["row_id"]
    ]
    notes = (iss.assign(text=iss["column"].where(iss["column"] != "", "") + ": " + iss["message"])
             .groupby("row_id")["text"].agg(" | ".join))
    out["notes"] = out["row_id"].map(notes).fillna("").str.lstrip(": ")

    power = pd.Series(np.nan, index=raw.index, dtype=float)
    power[validation.valid_mask] = np.asarray(power_w, dtype=float)
    out["predicted_power_W"] = power.round(1)
    out["predicted_energy_kWh"] = (power / 1000 * interval_hours).round(4)
    return out


def evaluate(actual, predicted):
    """MAE, RMSE, R2 on rows where both values are finite. Returns dict (or None if < 2 rows)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    keep = np.isfinite(a) & np.isfinite(p)
    a, p = a[keep], p[keep]
    if len(a) < 2:
        return None
    err = p - a
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return {
        "n": int(len(a)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
    }


def results_to_csv_bytes(df):
    """UTF-8 with BOM so that Excel shows Arabic text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def template_csv_bytes(features):
    cols = list(features) + [config.TIMESTAMP_COLUMN, config.ACTUAL_COLUMN]
    return (",".join(cols) + "\n").encode("utf-8-sig")
