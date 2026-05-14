"""
Дашборд по всему стаду: текущий статус, тревоги, рейтинги, графики.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_loaded

df, bundle = ensure_loaded()

st.title("📊 Дашборд стада")
st.markdown("Общий обзор состояния всего поголовья и животных, требующих внимания.")

# ====== Фильтр периода ======
min_date = df["date"].min().date()
max_date = df["date"].max().date()

col_filter1, col_filter2 = st.columns([2, 5])
with col_filter1:
    date_range = st.date_input(
        "Период анализа:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if len(date_range) == 2:
    period_df = df[
        (df["date"].dt.date >= date_range[0]) &
        (df["date"].dt.date <= date_range[1])
    ]
else:
    period_df = df

st.divider()

# ====== Список тревог: коровы с риском/болезнью в последний день ======
latest = df.sort_values("date").groupby("cow_id").tail(1)
alerts = latest[latest["health_status"].isin(["at_risk", "sick"])].sort_values(
    "health_status", ascending=False
)

st.subheader(f"🚨 Тревоги — {len(alerts)} коров требуют внимания")

if len(alerts) > 0:
    alerts_display = alerts.copy()
    alerts_display["date"] = alerts_display["date"].dt.strftime("%d.%m.%Y")
    alerts_display = alerts_display[[
        "cow_id", "date", "cbt_mean", "thi_mean", "milk_yield_kg",
        "activity", "rumination", "hr", "health_status",
    ]]
    alerts_display.columns = [
        "ID коровы", "Дата", "Темп., °C", "THI", "Удой, кг",
        "Активность", "Руминация", "ЧСС", "Статус",
    ]

    # Подсветка по статусу
    def color_status(row):
        if row["Статус"] == "sick":
            return ["background-color: #ffcdd2"] * len(row)
        elif row["Статус"] == "at_risk":
            return ["background-color: #fff9c4"] * len(row)
        return [""] * len(row)

    st.dataframe(
        alerts_display.style.apply(color_status, axis=1),
        use_container_width=True,
        hide_index=True,
        height=min(400, 50 + 35 * len(alerts_display)),
    )
else:
    st.success("✅ Все коровы в норме!")

st.divider()

# ====== Распределение статусов и среднего по показателям ======
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Распределение статусов в стаде")
    status_counts = latest["health_status"].value_counts().reindex(
        ["healthy", "at_risk", "sick"], fill_value=0
    )
    colors = ["#4CAF50", "#FFB300", "#E53935"]
    fig = go.Figure(go.Pie(
        labels=["Здоровые", "В зоне риска", "Больные"],
        values=status_counts.values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo="label+percent+value",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📈 Тренд числа больных коров по дням")
    daily_status = period_df.groupby([period_df["date"].dt.date, "health_status"]).size().unstack(fill_value=0)
    for cls in ["healthy", "at_risk", "sick"]:
        if cls not in daily_status.columns:
            daily_status[cls] = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_status.index, y=daily_status["sick"],
        mode="lines+markers", name="Больные",
        line=dict(color="#E53935", width=2.5),
        fill="tozeroy", fillcolor="rgba(229,57,53,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=daily_status.index, y=daily_status["at_risk"],
        mode="lines+markers", name="В зоне риска",
        line=dict(color="#FFB300", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,179,0,0.12)",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Дата",
        yaxis_title="Количество коров",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ====== Средние показатели по группам ======
st.subheader("📊 Средние значения показателей по группам здоровья")

agg = period_df.groupby("health_status")[
    ["cbt_mean", "thi_mean", "milk_yield_kg", "activity", "rumination", "hr"]
].mean().round(2)
agg = agg.reindex(["healthy", "at_risk", "sick"])
agg.index = ["Здоровые", "В зоне риска", "Больные"]
agg.columns = ["Темп., °C", "THI", "Удой, кг", "Активность", "Руминация, мин", "ЧСС"]

st.dataframe(
    agg.style.background_gradient(cmap="YlOrRd", axis=0),
    use_container_width=True,
)

st.caption(
    f"Период анализа: с {date_range[0].strftime('%d.%m.%Y')} по {date_range[1].strftime('%d.%m.%Y')}. "
    f"Всего наблюдений в периоде: {len(period_df):,}"
)
