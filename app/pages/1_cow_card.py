"""
Карточка отдельного животного: выбор коровы, графики показателей за период,
текущий статус и предсказание модели.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Чтобы импорты соседних модулей работали при прямом переходе по URL
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_loaded

df, bundle = ensure_loaded()

st.title("🐄 Карточка животного")
st.markdown("Детальная информация о состоянии конкретной коровы за период наблюдений.")

# ====== Селектор коровы ======
cow_ids = sorted(df["cow_id"].unique())

col_select, col_info = st.columns([1, 3])
with col_select:
    selected_cow = st.selectbox("Выберите корову:", cow_ids)

# Данные по выбранной корове
cow_df = df[df["cow_id"] == selected_cow].sort_values("date")
latest = cow_df.iloc[-1]

# ====== Текущий статус ======
with col_info:
    status = latest["health_status"]
    if status == "healthy":
        st.success(f"✅ Текущий статус: **ЗДОРОВА** (последнее измерение: {latest['date'].strftime('%d.%m.%Y')})")
    elif status == "at_risk":
        st.warning(f"⚠️ Текущий статус: **В ЗОНЕ РИСКА** (последнее измерение: {latest['date'].strftime('%d.%m.%Y')})")
    else:
        st.error(f"🚨 Текущий статус: **БОЛЬНА** (последнее измерение: {latest['date'].strftime('%d.%m.%Y')})")

st.divider()

# ====== Текущие показатели ======
st.subheader("📊 Последние измерения")

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_with_warning(col, label, value, unit, lo, hi, fmt="{:.1f}"):
    """Метрика с цветовой индикацией нормы."""
    in_norm = lo <= value <= hi
    delta = None if in_norm else "вне нормы"
    delta_color = "off" if in_norm else "inverse"
    col.metric(label, fmt.format(value) + f" {unit}", delta=delta, delta_color=delta_color)

metric_with_warning(c1, "🌡️ Температура", latest["cbt_mean"], "°C", 38.0, 39.5)
metric_with_warning(c2, "💧 THI", latest["thi_mean"], "", 0, 72)
metric_with_warning(c3, "🥛 Удой", latest["milk_yield_kg"], "кг", 25, 50)
metric_with_warning(c4, "🚶 Активность", latest["activity"], "шагов", 5000, 12000, fmt="{:.0f}")
metric_with_warning(c5, "🌾 Руминация", latest["rumination"], "мин", 420, 540, fmt="{:.0f}")
metric_with_warning(c6, "💓 ЧСС", latest["hr"], "уд/мин", 60, 80)

st.divider()

# ====== Графики динамики ======
st.subheader("📈 Динамика показателей за весь период")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Температура тела (°C)",
        "Индекс температуры-влажности (THI)",
        "Удой (кг/сутки)",
        "Активность (шагов/сутки)",
        "Время руминации (мин/сутки)",
        "ЧСС (уд/мин)",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

# Цвета для статусов
status_colors = {"healthy": "#4CAF50", "at_risk": "#FFB300", "sick": "#E53935"}
colors_array = [status_colors[s] for s in cow_df["health_status"]]

features_to_plot = [
    ("cbt_mean", 1, 1, [(38.0, 39.5)]),
    ("thi_mean", 1, 2, [(0, 72)]),
    ("milk_yield_kg", 2, 1, [(25, 50)]),
    ("activity", 2, 2, [(5000, 12000)]),
    ("rumination", 3, 1, [(420, 540)]),
    ("hr", 3, 2, [(60, 80)]),
]

for feat, r, c, norms in features_to_plot:
    # Линия
    fig.add_trace(
        go.Scatter(
            x=cow_df["date"], y=cow_df[feat],
            mode="lines+markers",
            marker=dict(color=colors_array, size=7),
            line=dict(color="#1f77b4", width=1.5),
            showlegend=False,
            hovertemplate="%{x|%d.%m.%Y}<br>%{y:.2f}<extra></extra>",
        ),
        row=r, col=c,
    )
    # Зоны нормы (горизонтальные полосы)
    for lo, hi in norms:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor="green", opacity=0.07,
            line_width=0, row=r, col=c,
        )

fig.update_layout(
    height=750,
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
)
fig.update_xaxes(showgrid=True, gridcolor="lightgray", gridwidth=0.5)
fig.update_yaxes(showgrid=True, gridcolor="lightgray", gridwidth=0.5)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🟢 Зелёные точки — здоровая, 🟡 жёлтые — в зоне риска, 🔴 красные — больная. "
    "Зелёные полосы — диапазоны нормы."
)

st.divider()

# ====== Таблица последних измерений ======
st.subheader("📋 Последние 10 измерений")
display_df = cow_df.tail(10).copy()
display_df["date"] = display_df["date"].dt.strftime("%d.%m.%Y")
display_df = display_df[[
    "date", "cbt_mean", "thi_mean", "milk_yield_kg",
    "activity", "rumination", "hr", "behavior_mode", "health_status",
]]
display_df.columns = [
    "Дата", "Температура", "THI", "Удой",
    "Активность", "Руминация", "ЧСС", "Поведение", "Статус",
]
st.dataframe(display_df, use_container_width=True, hide_index=True)
