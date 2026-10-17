"""Plotly figures. Each function only draws data it is given; nothing is invented."""
import numpy as np
import plotly.graph_objects as go

BLUE, ORANGE, GREY = "#1f77b4", "#e8710a", "#6b7280"


def _layout(fig, title, xtitle, ytitle):
    fig.update_layout(
        title=title, xaxis_title=xtitle, yaxis_title=ytitle,
        template="plotly_white", margin=dict(l=50, r=20, t=60, b=50), height=380,
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


def sensitivity(x, y, label, cur_x, cur_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="تنبؤ النموذج", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=[cur_x], y=[cur_y], mode="markers", name="القيمة الحالية",
                             marker=dict(color=ORANGE, size=11)))
    return _layout(fig, f"حساسية النموذج: القدرة مقابل {label}", label, "القدرة المتوقعة (W)")


def bar_by_record(res, max_rows=200):
    d = res[res.predicted_power_W.notna()].head(max_rows)
    fig = go.Figure(go.Bar(x=d.row_id.astype(str), y=d.predicted_power_W, marker_color=BLUE))
    title = "مقارنة الإنتاج المتوقع بين السجلات"
    if res.predicted_power_W.notna().sum() > max_rows:
        title += f" (أول {max_rows} سجلاً صالحاً)"
    return _layout(fig, title, "رقم الصف في الملف", "القدرة المتوقعة (W)")


def histogram(res):
    fig = go.Figure(go.Histogram(x=res.predicted_power_W.dropna(), nbinsx=30, marker_color=BLUE))
    return _layout(fig, "توزيع القدرة المتوقعة", "القدرة المتوقعة (W)", "عدد السجلات")


def by_hour(res):
    d = res[res.predicted_power_W.notna()].assign(h=lambda t: t["hour"].astype(float))
    g = d.groupby("h").predicted_power_W.mean()
    fig = go.Figure(go.Scatter(x=g.index, y=g.values, mode="lines+markers", line=dict(color=BLUE)))
    fig.update_xaxes(dtick=2)
    return _layout(fig, "  متوسط القدرة المتوقعة لكل ساعة", "الساعة (UTC)", " القدرة المتوقعة (W)")


def actual_vs_predicted(actual, predicted, label):
    hi = float(max(np.nanmax(actual), np.nanmax(predicted)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="السجلات",
                             marker=dict(color=BLUE, size=5, opacity=0.5)))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="تطابق تام",
                             line=dict(color=GREY, dash="dash")))
    return _layout(fig, f"القيم المرجعية مقابل المتوقعة", f"{label} (W)", "القدرة المتوقعة (W)")


def error_histogram(errors):
    fig = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color=ORANGE))
    fig = _layout(fig, "توزيع أخطاء التنبؤ (المتوقع − المرجعي)", "الخطأ (W)", "عدد السجلات")
    fig.update_xaxes(tickformat=",.0f")
    return fig


def historical_line(x, y, title, ytitle, xtitle="التاريخ (UTC)"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BLUE, width=1.5), name=ytitle))
    fig.update_xaxes(tickformat="%Y-%m-%d\n%H:%M")
    return _layout(fig, title, xtitle, ytitle)


def hourly_profile(hours, values, title, ytitle):
    fig = go.Figure(go.Scatter(x=hours, y=values, mode="lines+markers", line=dict(color=BLUE)))
    fig.update_xaxes(dtick=2, range=[-0.5, 23.5])
    return _layout(fig, title, "الساعة (UTC)", ytitle)


def monthly_bars(months, values, month_labels, title, ytitle):
    fig = go.Figure(go.Bar(x=month_labels, y=values, marker_color=BLUE))
    return _layout(fig, title, "الشهر", ytitle)


def distribution_histogram(values, title, xtitle):
    fig = go.Figure(go.Histogram(x=values, nbinsx=30, marker_color=BLUE))
    return _layout(fig, title, xtitle, "عدد الساعات")


def dual_axis_weather(x, irradiance, temperature, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=irradiance, name="الإشعاع (W/m²)", line=dict(color=ORANGE), yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=temperature, name="درجة الحرارة (°C)", line=dict(color=BLUE), yaxis="y2"))
    fig.update_layout(
        title=title, template="plotly_white", margin=dict(l=50, r=50, t=60, b=50), height=400,
        yaxis=dict(title="الإشعاع (W/m²)", tickformat=",.0f"),
        yaxis2=dict(title="درجة الحرارة (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d\n%H:%M")
    return fig
