"""Data audit for solar_data.csv (run: python audit_data.py).
Checks structure, units/ranges, time axis, and how sun_height_deg relates to time and location."""
from pathlib import Path

import numpy as np
import pandas as pd

import config

df = pd.read_csv(config.DATA_PATH)
t = pd.to_datetime(df["datetime_utc"])
print(f"rows={len(df)}  columns={list(df.columns)}")
print(f"period (UTC): {t.min()} -> {t.max()}")
print("missing values:", int(df.isna().sum().sum()), "| duplicated timestamps:", int(t.duplicated().sum()))
expected = pd.date_range(t.min().floor("h"), t.max().floor("h"), freq="h")
print("missing hourly steps:", len(expected.difference(t.dt.floor("h"))))

print("\nRanges:")
print(df[["pv_power_W", "irradiance_plane_Wm2", "sun_height_deg", "air_temp_C", "wind_speed_ms"]]
      .agg(["min", "max"]).round(2).to_string())

# 1) Is the target a deterministic function of the inputs? (leakage / tautology check)
day = df[df.irradiance_plane_Wm2 > 50]
ratio = day.pv_power_W / day.irradiance_plane_Wm2
print(f"\nP/G(i) when G(i) > 50 W/m2: min={ratio.min():.3f} median={ratio.median():.3f} max={ratio.max():.3f}")
print("corr(P, G(i)) =", round(df.pv_power_W.corr(df.irradiance_plane_Wm2), 5))
print("P > 0 while G(i) = 0:", int(((df.pv_power_W > 0) & (df.irradiance_plane_Wm2 == 0)).sum()))

# 2) sun_height_deg: can it be computed from timestamp + location only?
lat, lon = config.LATITUDE, config.LONGITUDE


def solar_elevation(ts_utc, lat, lon):
    """NOAA low-precision solar elevation in degrees (no refraction)."""
    doy = ts_utc.dt.dayofyear.to_numpy()
    hrs = (ts_utc.dt.hour + ts_utc.dt.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hrs - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hrs * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cos_z = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_z, -1, 1)))


calc = solar_elevation(t, lat, lon)
up = df.sun_height_deg > 0
err = (calc - df.sun_height_deg)[up]
print(f"\nsun_height_deg vs NOAA formula (daylight rows): MAE={err.abs().mean():.2f} deg, max |err|={err.abs().max():.2f} deg")

# approximation using only month + hour (mid-month day) - what the app would have
mid = pd.to_datetime(dict(year=2021, month=df.month, day=15, hour=df.hour, minute=9))
err2 = (solar_elevation(mid, lat, lon) - df.sun_height_deg)[up]
print(f"using only (month, hour) with day=15: MAE={err2.abs().mean():.2f} deg, max |err|={err2.abs().max():.2f} deg")
