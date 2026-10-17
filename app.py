
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

import charts
import config
import data_utils
import model_utils

st.set_page_config(
    page_title="تحليلات الطاقة الشمسية - بغداد",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMarkdownContainer"],
[data-testid="stAlert"],
label,
[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}

/* بطاقات المقاييس */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px 18px;
    color: var(--card-text);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    transition: border-color .2s ease, box-shadow .2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #bfdbfe;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.10);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] p,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] span {
    color: var(--card-text) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--card-muted) !important;
    font-size: 0.9rem;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: var(--accent) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* تنسيق عام للعناوين */
h1, h2, h3 {
    color: var(--card-text);
}

/* تنسيق التنبيهات */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--card-border);
}

/* تحسين الشريط الجانبي */
[data-testid="stSidebar"] {
    background: var(--bg-soft);
    border-left: 1px solid var(--card-border);
}
</style>
""",
unsafe_allow_html=True,

)

POWER_COL = config.TARGET
IRR_COL = "irradiance_plane_Wm2"
TEMP_COL = "air_temp_C"
WIND_COL = "wind_speed_ms"
YEAR_COL = "year"
MONTH_COL = "month"
HOUR_COL = "hour"


def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0


def fmt(v, d=0):
    return f"{v:,.{d}f}"


def spec_label(f):
    s = config.FEATURES[f]
    return f"{s['label']} ({s['unit']})"


@dataclass
class DatasetLoad:
    df: pd.DataFrame | None = None
    error: str | None = None
    warnings: list = field(default_factory=list)
    parse_ok: bool = False


@st.cache_data(show_spinner="جاري تحميل البيانات التاريخية...")
def load_historical_dataset(csv_mtime):
    path = config.DATA_PATH
    if not path.is_file():
        return DatasetLoad(
            error=f"ملف البيانات ({path.name}) غير موجود. ضع solar_data.csv بجانب app.py.",
        )
    try:
        raw = path.read_bytes()
        if not raw.strip():
            return DatasetLoad(error="ملف البيانات فارغ.")
        df = data_utils.read_uploaded_csv(raw)
    except data_utils.DataError as exc:
        return DatasetLoad(error=str(exc))
    except OSError as exc:
        return DatasetLoad(error=f"تعذّر قراءة ملف البيانات ({type(exc).__name__}).")

    warnings = []
    df = df.copy()
    for col in df.columns:
        if col in (config.TIMESTAMP_COLUMN,):
            continue
        if col in config.FEATURES or col in (POWER_COL, IRR_COL, TEMP_COL, WIND_COL, "sun_height_deg",
                                              YEAR_COL, MONTH_COL, HOUR_COL):
            vals, codes = data_utils.parse_numeric(df[col])
            df[col] = vals
            n_bad = int(codes.notna().sum())
            if n_bad:
                warnings.append(f"العمود «{col}»: {n_bad} قيمة غير صالحة أو فارغة (تُستبعد من الحسابات).")

    parse_ok = False
    if config.TIMESTAMP_COLUMN in df.columns:
        ts = pd.to_datetime(df[config.TIMESTAMP_COLUMN], errors="coerce", utc=False)
        n_bad = int(ts.isna().sum())
        if n_bad == len(df):
            warnings.append(
                f"تعذّر تحليل عمود التاريخ «{config.TIMESTAMP_COLUMN}»؛ لن تُعرض المخططات الزمنية."
            )
        else:
            parse_ok = True
            if n_bad:
                warnings.append(f"«{config.TIMESTAMP_COLUMN}»: {n_bad} ختم زمني غير قابل للتحليل.")
            df["_ts"] = ts
    else:
        warnings.append(
            f"عمود «{config.TIMESTAMP_COLUMN}» غير موجود؛ المخططات الزمنية غير متاحة. "
            "يمكنك الاستمرار باستخدام year / month / hour إن وُجدت."
        )

    if YEAR_COL not in df.columns and parse_ok:
        df[YEAR_COL] = df["_ts"].dt.year
    if MONTH_COL not in df.columns and parse_ok:
        df[MONTH_COL] = df["_ts"].dt.month
    if HOUR_COL not in df.columns and parse_ok:
        df[HOUR_COL] = df["_ts"].dt.hour

    return DatasetLoad(df=df, warnings=warnings, parse_ok=parse_ok)


@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def get_bundle(model_mtime, meta_mtime):
    return model_utils.load_bundle()


def available_years(df):
    if YEAR_COL not in df.columns:
        return []
    yrs = df[YEAR_COL].dropna()
    yrs = yrs[np.isfinite(yrs)].astype(int)
    return sorted(y for y in yrs.unique() if y >= 2020)


def filter_period(df, year, month, hour=None):
    if df is None or len(df) == 0:
        return df
    out = df
    if YEAR_COL in out.columns:
        out = out[out[YEAR_COL] == year]
    if month is not None and MONTH_COL in out.columns:
        out = out[out[MONTH_COL] == month]
    if hour is not None and HOUR_COL in out.columns:
        out = out[out[HOUR_COL] == hour]
    return out


def safe_numeric_series(df, col):
    if df is None or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s[np.isfinite(s)]


def sidebar_selectors(ds, df):
    st.sidebar.header("تصفية البيانات التاريخية")
    st.sidebar.caption(
        f"الموقع: بغداد ({config.LATITUDE}°N, {config.LONGITUDE}°E) — "
        f"التوقيت المحلي = UTC+{config.UTC_OFFSET_HOURS}"
    )
    years = available_years(df) if df is not None else []
    if not years:
        st.sidebar.warning("لا توجد سنوات ≥ 2020 في البيانات. تحقق من ملف CSV أو عمود السنة.")
        year = month = hour = None
    else:
        year = st.sidebar.selectbox("السنة", years, index=len(years) - 1, key="sel_year")
        month = st.sidebar.selectbox(
            "الشهر",
            list(range(1, 13)),
            format_func=lambda m: f"{m} — {config.MONTH_NAMES_AR[m - 1]}",
            index=5,
            key="sel_month",
        )
        hour = st.sidebar.slider("الساعة (UTC)", 0, 23, 9, key="sel_hour")
        local_h = config.utc_to_local_hour(hour)
        st.sidebar.info(f"الساعة {hour}:00 UTC ≈ {local_h}:00 بتوقيت بغداد.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**ملاحظة:** السنة تُستخدم لتصفية **البيانات التاريخية** فقط. "
        "مدخلات النموذج هي الشهر والساعة (UTC) مع ظروف الطقس، دون سنة."
    )
    if ds.warnings:
        with st.sidebar.expander("تحذيرات البيانات"):
            for w in ds.warnings:
                st.caption("• " + w)
    return year, month, hour


def render_overview(df, year, month, hour, parse_ok):
    st.subheader("نظرة عامة على الفترة المختارة")
    if df is None:
        st.error("البيانات غير متاحة.")
        return
    if YEAR_COL not in df.columns:
        st.warning(
            "عمود «year» غير موجود في CSV؛ لا يمكن عزل سنة واحدة. "
            "تُعرض كل السجلات المطابقة للشهر (عبر سنوات متعددة)."
        )

    month_df = filter_period(df, year, month)
    slot_df = filter_period(df, year, month, hour)

    if len(month_df) == 0:
        st.warning(
            f"لا توجد سجلات تاريخية للسنة {year} والشهر {config.MONTH_NAMES_AR[month - 1]}. "
            "جرّب سنة أو شهراً آخر."
        )
        return

    p_month = safe_numeric_series(month_df, POWER_COL)
    p_slot = safe_numeric_series(slot_df, POWER_COL)
    irr_slot = safe_numeric_series(slot_df, IRR_COL)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سجلات الشهر", fmt(len(month_df)))
    c2.metric("سجلات الساعة المختارة", fmt(len(slot_df)))
    c3.metric(
        "متوسط القدرة (الشهر)",
        f"{fmt(p_month.mean())} W" if p_month is not None and len(p_month) else "—",
    )
    c4.metric(
        f"متوسط القدرة عند {hour}:00 UTC",
        f"{fmt(p_slot.mean())} W" if p_slot is not None and len(p_slot) else "—",
    )

    if len(slot_df) == 0:
        st.info(
            f"لا توجد سجلات عند الساعة {hour}:00 UTC في {config.MONTH_NAMES_AR[month - 1]} {year}. "
            "اختر ساعة أخرى أو راجع البيانات."
        )

    left, right = st.columns(2)

    if parse_ok and "_ts" in month_df.columns and POWER_COL in month_df.columns:
        plot_df = month_df.dropna(subset=["_ts", POWER_COL]).sort_values("_ts")
        if len(plot_df):
            left.plotly_chart(
                charts.historical_line(
                    plot_df["_ts"],
                    plot_df[POWER_COL],
                    f"القدرة المرجعية (PVGIS) — {config.MONTH_NAMES_AR[month - 1]} {year}",
                    f"{config.TARGET_LABEL} ({config.TARGET_UNIT})",
                ),
                use_container_width=True,
                key="ov_ts_power",
            )
        else:
            left.info("لا توجد قيم صالحة للقدرة في هذا الشهر لرسم المخطط الزمني.")
    else:
        left.info("المخطط الزمني يتطلب عمود تاريخ صالحاً وعمود القدرة.")

    if HOUR_COL in month_df.columns and POWER_COL in month_df.columns:
        g = month_df.dropna(subset=[HOUR_COL, POWER_COL]).groupby(HOUR_COL)[POWER_COL].mean()
        if len(g):
            right.plotly_chart(
                charts.hourly_profile(
                    g.index.astype(int),
                    g.values,
                    f"متوسط القدرة حسب الساعة — {config.MONTH_NAMES_AR[month - 1]} {year}",
                    f"متوسط {POWER_COL} (W)",
                ),
                use_container_width=True,
                key="ov_hour_profile",
            )
        else:
            right.info("لا توجد بيانات كافية لملف الساعات.")
    else:
        right.info("ملف الساعات يتطلب أعمدة hour و pv_power_W.")

    if (
        parse_ok
        and "_ts" in month_df.columns
        and IRR_COL in month_df.columns
        and TEMP_COL in month_df.columns
    ):
        wdf = month_df.dropna(subset=["_ts", IRR_COL, TEMP_COL]).sort_values("_ts")
        if len(wdf):
            st.plotly_chart(
                charts.dual_axis_weather(
                    wdf["_ts"],
                    wdf[IRR_COL],
                    wdf[TEMP_COL],
                    f"الإشعاع ودرجة الحرارة — {config.MONTH_NAMES_AR[month - 1]} {year}",
                ),
                use_container_width=True,
                key="ov_weather",
            )

    if len(slot_df) > 0:
        with st.expander(f"سجلات الساعة {hour}:00 UTC في الشهر ({len(slot_df)} صف)"):
            show_cols = [c for c in (
                config.TIMESTAMP_COLUMN, YEAR_COL, MONTH_COL, HOUR_COL,
                IRR_COL, TEMP_COL, WIND_COL, POWER_COL,
            ) if c in slot_df.columns]
            st.dataframe(slot_df[show_cols].head(500), hide_index=True)


def defaults_from_history(slot_df, bundle):
    """Suggested weather inputs from historical rows at the selected slot (not sent to model automatically)."""
    values = {}
    for f in bundle.features:
        spec = config.FEATURES[f]
        if f in (MONTH_COL, HOUR_COL):
            continue
        if slot_df is not None and len(slot_df) and f in slot_df.columns:
            s = safe_numeric_series(slot_df, f)
            if s is not None and len(s):
                values[f] = float(s.median())
                continue
        values[f] = float(spec["default"]) if not spec.get("integer") else int(spec["default"])
    return values


def render_predict(bundle, year, month, hour, df, model_ok):
    st.subheader("تنبؤ بالقدرة من ظروف الطقس")
    st.caption(
        f"المدخلات الزمنية للنموذج: شهر {month} ({config.MONTH_NAMES_AR[month - 1]}) "
        f"وساعة {hour}:00 UTC (≈ {config.utc_to_local_hour(hour)}:00 بغداد). "
        f"السنة {year} للسياق التاريخي فقط."
    )

    if not model_ok:
        st.error(
            "لا يمكن إنشاء تنبؤ لأن النموذج غير متاح أو غير متوافق. "
            "ضع model.joblib و model_meta.json الصحيحين بجانب app.py، ثم حدّث الصفحة."
        )
        st.info("يمكنك متابعة استكشاف البيانات التاريخية من تبويب «لوحة البيانات».")
        return

    slot_df = filter_period(df, year, month, hour) if df is not None else None
    hist_defaults = defaults_from_history(slot_df, bundle)

    if slot_df is not None and len(slot_df):
        st.success(
            f"وُجد {len(slot_df)} سجل تاريخي عند هذه الساعة؛ الزر أدناه يملأ الظروف بقيم median منها."
        )
        if st.button("ملء الظروف من البيانات التاريخية لهذه الساعة", key="fill_from_hist"):
            for f, v in hist_defaults.items():
                st.session_state[f"pred_{f}"] = v
            if MONTH_COL in bundle.features:
                st.session_state[f"pred_{MONTH_COL}"] = month
            if HOUR_COL in bundle.features:
                st.session_state[f"pred_{HOUR_COL}"] = hour

    meta = bundle.meta
    ranges = meta.get("feature_ranges")
    irr_max = meta.get("irradiance_max_by_month_hour")

    cols = st.columns(3)
    values = {}
    for i, f in enumerate(bundle.features):
        s = config.FEATURES[f]
        help_txt = f"الحدود الفيزيائية: [{s['hard_min']}, {s['hard_max']}]"
        if ranges and f in ranges:
            help_txt += f" | نطاق التدريب: [{ranges[f]['min']:g}, {ranges[f]['max']:g}]"
        key = f"pred_{f}"
        if f == MONTH_COL:
            init = month
        elif f == HOUR_COL:
            init = hour
        elif key not in st.session_state:
            init = hist_defaults.get(f, s["default"])
        else:
            init = st.session_state[key]
        with cols[i % 3]:
            if s.get("integer"):
                values[f] = st.number_input(
                    spec_label(f), int(s["hard_min"]), int(s["hard_max"]),
                    int(init), 1, help=help_txt, key=key,
                )
            else:
                values[f] = st.number_input(
                    spec_label(f), float(s["hard_min"]), float(s["hard_max"]),
                    float(init), float(s["step"]), help=help_txt, key=key,
                )

    hours = st.number_input("المدة (ساعات) بافتراض ثبات الظروف", 0.25, 24.0, 1.0, 0.25, key="pred_hours")
    run = st.button("توليد التنبؤ", type="primary", key="run_pred")

    check = data_utils.validate_frame(pd.DataFrame([values]), bundle.features, ranges, irr_max)
    for msg in check.errors.itertuples():
        st.error(f"{msg.column}: {msg.message}")
    for w in check.warnings.itertuples():
        st.warning(f"{w.column}: {w.message}")

    if not run:
        st.info("عدّل الظروف ثم اضغط «توليد التنبؤ».")
        return
    if len(check.errors):
        st.error("صحّح أخطاء المدخلات قبل التنبؤ.")
        return

    try:
        power = float(model_utils.predict_power(bundle, check.values)[0])
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    if power < 0:
        st.warning("أنتج النموذج قيمة سالبة؛ القدرة لا تكون سالبة فيزيائياً. راجع النموذج أو المدخلات.")

    r1, r2, r3 = st.columns(3)
    r1.metric("القدرة المتوقعة", f"{fmt(power)} W")
    r2.metric(f"الطاقة المتوقعة خلال {hours:g} ساعة", f"{fmt(power / 1000 * hours, 3)} kWh")
    st.caption(f"القيم لنظام {config.SYSTEM_KWP:g} kWp كما عُرّف الهدف في البيانات ({config.TARGET_UNIT}).")

    if slot_df is not None and POWER_COL in slot_df.columns:
        actual = safe_numeric_series(slot_df, POWER_COL)
        if actual is not None and len(actual):
            r3.metric("متوسط القدرة المرجعية (نفس الساعة)", f"{fmt(actual.mean())} W")
            st.caption("المقارنة على السجلات التاريخية عند نفس السنة/الشهر/الساعة — وليس على المدخلات الحالية.")

    with st.expander("المدخلات المستخدمة"):
        st.dataframe(pd.DataFrame([values]), hide_index=True)

    with st.expander("حساسية النموذج لكل متغير"):
        st.caption("يتغير متغير واحد وتبقى بقية المدخلات كما أدخلتها.")
        for f in ("irradiance_plane_Wm2", "air_temp_C", "wind_speed_ms"):
            if f not in bundle.features:
                continue
            lo, hi = (
                (ranges[f]["min"], ranges[f]["max"])
                if ranges and f in ranges
                else (config.FEATURES[f]["hard_min"], config.FEATURES[f]["hard_max"])
            )
            xs = np.linspace(lo, hi, 25)
            try:
                ys = model_utils.sweep_prediction(bundle, values, f, xs)
            except (RuntimeError, ValueError) as exc:
                st.warning(f"تعذّر رسم حساسية {f}: {exc}")
                continue
            st.plotly_chart(
                charts.sensitivity(xs, ys, spec_label(f), values[f], power),
                use_container_width=True,
                key=f"sens_{f}",
            )


def render_explore(df, year):
    st.subheader("استكشاف السنة")
    if df is None:
        return
    if YEAR_COL in df.columns:
        year_df = df[df[YEAR_COL] == year]
    else:
        year_df = df
    if len(year_df) == 0:
        st.warning(f"لا توجد سجلات للسنة {year}.")
        return

    if MONTH_COL in year_df.columns and POWER_COL in year_df.columns:
        g = year_df.groupby(MONTH_COL)[POWER_COL].mean()
        labels = [config.MONTH_NAMES_AR[int(m) - 1] for m in g.index]
        st.plotly_chart(
            charts.monthly_bars(g.index, g.values, labels, f"متوسط القدرة الشهري — {year}", "متوسط القدرة (W)"),
            use_container_width=True,
            key="ex_monthly",
        )
    else:
        st.info("المخطط الشهري يتطلب أعمدة month و pv_power_W.")

    c1, c2 = st.columns(2)
    for col, title, box in (
        (IRR_COL, "الإشعاع", c1),
        (TEMP_COL, "درجة الحرارة", c2),
    ):
        s = safe_numeric_series(year_df, col)
        if s is not None and len(s):
            box.plotly_chart(
                charts.distribution_histogram(s, f"توزيع {title} — {year}", title),
                use_container_width=True,
                key=f"hist_{col}_{year}",
            )
        else:
            box.info(f"العمود {col} غير متاح أو فارغ.")


def render_bulk(bundle, model_ok):
    st.subheader("تنبؤ جماعي من ملف CSV")
    if not model_ok:
        st.error("التنبؤ الجماعي يتطلب نموذجاً صالحاً.")
        return

    meta = bundle.meta
    ranges = meta.get("feature_ranges")
    irr_max = meta.get("irradiance_max_by_month_hour")

    st.markdown("الأعمدة المطلوبة (بالاسم نفسه): `" + "`, `".join(bundle.features) + "`")
    st.caption(
        f"اختيارياً: `{config.TIMESTAMP_COLUMN}` لكشف تكرار الأوقات، و`{config.ACTUAL_COLUMN}` "
        "لمقارنة المتوقع بالمرجعي. الساعة hour بتوقيت UTC، والشهر month من 1 إلى 12."
    )
    st.download_button(
        "تنزيل قالب CSV (ترويسة فقط)",
        data_utils.template_csv_bytes(bundle.features),
        file_name="template.csv",
        mime="text/csv",
    )

    up = st.file_uploader("ارفع ملف CSV", type=["csv"])
    interval = st.number_input("كل صف يمثل فترة بطول (ساعات)", 0.25, 24.0, 1.0, 0.25, key="bulk_interval")
    policy = st.radio(
        "عند وجود صفوف غير صالحة",
        ["استبعادها من التنبؤ مع إبقائها في النتائج بحالتها وسببها", "إيقاف التنبؤ حتى يُصحَّح الملف"],
    )
    if up is None:
        st.info("لم يُرفع ملف بعد.")
        return

    try:
        raw = data_utils.read_uploaded_csv(up.getvalue())
    except data_utils.DataError as exc:
        st.error(str(exc))
        return

    val = data_utils.validate_frame(raw, bundle.features, ranges, irr_max, config.TIMESTAMP_COLUMN)
    if val.missing_columns:
        st.error("أعمدة مطلوبة غير موجودة: " + ", ".join(val.missing_columns))
        for h in val.hints:
            st.info(h)
        st.caption("الأعمدة الموجودة في ملفك: " + ", ".join(raw.columns))
        return
    if val.extra_columns:
        st.caption("أعمدة تجاهلها التنبؤ (غير مستخدمة بالنموذج): " + ", ".join(val.extra_columns))

    n, n_valid = len(raw), int(val.valid_mask.sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("عدد الصفوف", fmt(n))
    k2.metric("صالحة", fmt(n_valid))
    k3.metric("غير صالحة", fmt(n - n_valid))
    k4.metric("صفوف بتحذيرات", fmt(val.warnings.row_id.nunique()))

    if len(val.issues):
        summary = (
            val.issues.groupby(["level", "code", "column"])
            .agg(rows=("row_id", "nunique"), message=("message", "first"))
            .reset_index()
        )
        st.markdown("**ملخص مشكلات البيانات**")
        st.dataframe(summary, hide_index=True)

    if n_valid < n and policy.startswith("إيقاف"):
        st.error("تم إيقاف التنبؤ لأن الملف يحتوي صفوفاً غير صالحة.")
        return
    if n_valid == 0:
        st.error("لا توجد صفوف صالحة للتنبؤ.")
        return

    try:
        power = model_utils.predict_power(bundle, val.values[val.valid_mask])
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    res = data_utils.assemble_results(raw, val, power, interval, bundle.features)
    st.subheader("النتائج")
    st.dataframe(res, hide_index=True)
    ok = res[res.predicted_power_W.notna()]
    if len(ok):
        s = ok.predicted_power_W.describe()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("متوسط القدرة", f"{fmt(s['mean'])} W")
        a2.metric("أعلى قدرة", f"{fmt(s['max'])} W")
        a3.metric("أدنى قدرة", f"{fmt(s['min'])} W")
        a4.metric("إجمالي الطاقة", f"{fmt(ok.predicted_energy_kWh.sum(), 3)} kWh")

    st.download_button(
        "تنزيل النتائج (CSV)",
        data_utils.results_to_csv_bytes(res),
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.subheader("الرسوم البيانية")
    st.plotly_chart(charts.bar_by_record(res), use_container_width=True, key="bulk_bar")
    left, right = st.columns(2)
    left.plotly_chart(charts.histogram(res), use_container_width=True, key="bulk_hist")
    if "hour" in res.columns:
        right.plotly_chart(charts.by_hour(res), use_container_width=True, key="bulk_hour")

    if config.ACTUAL_COLUMN in raw.columns:
        actual, _ = data_utils.parse_numeric(raw[config.ACTUAL_COLUMN])
        both = res.predicted_power_W.notna() & actual.notna() & np.isfinite(actual)
        if both.sum() >= 2:
            m = data_utils.evaluate(actual[both], res.predicted_power_W[both])
            st.subheader("المقارنة مع القيم المرجعية")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("MAE", f"{m['mae']:.2f} W")
            e2.metric("RMSE", f"{m['rmse']:.2f} W")
            e3.metric("R²", f"{m['r2']:.4f}")
            e4.metric("عدد الصفوف", fmt(m["n"]))
            st.plotly_chart(
                charts.actual_vs_predicted(actual[both], res.predicted_power_W[both], config.ACTUAL_COLUMN),
                use_container_width=True,
                key="bulk_avp",
            )


def render_about(bundle, model_ok):
    st.subheader("حول النموذج والبيانات")
    meta = bundle.meta
    ranges = meta.get("feature_ranges") if model_ok else None
    rows = []
    feats = bundle.features if bundle.features else config.DEFAULT_FEATURES
    for f in feats:
        r = ranges.get(f) if ranges else None
        rows.append({
            "الميزة": f,
            "الوصف": config.FEATURES[f]["label"],
            "الوحدة": config.FEATURES[f]["unit"],
            "أدنى قيمة بالتدريب": r["min"] if r else "غير موثّق",
            "أعلى قيمة بالتدريب": r["max"] if r else "غير موثّق",
        })
    st.markdown("**ميزات النموذج (بالترتيب المتوقع):**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if meta:
        st.markdown(f"**الهدف:** `{meta.get('target', config.TARGET)}` ({meta.get('target_unit', config.TARGET_UNIT)})")
        if meta.get("target_definition"):
            st.markdown(meta["target_definition"])
        if meta.get("train_years"):
            st.markdown(
                f"**التدريب:** سنوات {meta['train_years']} ({fmt(meta.get('n_train', 0))} سجلاً)، "
                f"**الاختبار:** سنة {meta.get('test_year')} ({fmt(meta.get('n_test', 0))} سجلاً)."
            )
        with st.expander("الوصف الكامل (model_meta.json)"):
            st.json(meta)

    st.markdown(
        """**قيود يجب ذكرها عند العرض**
- الهدف ناتج محاكاة PVGIS وليس قياساً من محطة.
- التنبؤ المستقبلي يتطلب توقعات إشعاع وطقس غير متوفرة هنا.
- ملفات joblib قد تنفّذ كوداً عند التحميل؛ استخدم فقط ملفات من الفريق."""
    )


# ------------------------------------------------------------------ main
ds = load_historical_dataset(_mtime(config.DATA_PATH))
df = ds.df

st.title("☀️ توقعات الطاقة الشمسية")

if ds.error:
    st.error(ds.error)

bundle = get_bundle(_mtime(config.MODEL_PATH), _mtime(config.META_PATH))
model_ok = bundle.usable

year, month, hour = sidebar_selectors(ds, df)
if year is None:
    year, month, hour = 2020, 6, 9

tab_dash, tab_pred, tab_explore, tab_bulk, tab_about = st.tabs(
    ["لوحة البيانات", "التنبؤ", "استكشاف السنة", "تنبؤ جماعي (CSV)", "حول"]
)

with tab_dash:
    if df is not None and year is not None:
        render_overview(df, year, month, hour, ds.parse_ok)
    else:
        st.info("حمّل البيانات التاريخية لعرض اللوحة.")

with tab_pred:
    if year is not None:
        render_predict(bundle, year, month, hour, df, model_ok)

with tab_explore:
    if df is not None and year is not None:
        render_explore(df, year)

with tab_bulk:
    render_bulk(bundle, model_ok)

with tab_about:
    render_about(bundle, model_ok)
