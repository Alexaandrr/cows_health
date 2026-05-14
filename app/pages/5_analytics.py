"""
Разведочный анализ данных (EDA): распределения, корреляции, выбросы.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_loaded, FIGURES_DIR

df, bundle = ensure_loaded()

st.title("📈 Аналитика данных (EDA)")
st.markdown(
    "Разведочный анализ датасета: распределения признаков, корреляции, "
    "сравнение групп здоровья."
)

# ====== Общая статистика ======
st.subheader("📋 Описательная статистика")
features = ["cbt_mean", "thi_mean", "milk_yield_kg", "activity", "rumination", "hr"]
feature_names_ru = {
    "cbt_mean": "Температура, °C",
    "thi_mean": "THI",
    "milk_yield_kg": "Удой, кг",
    "activity": "Активность",
    "rumination": "Руминация, мин",
    "hr": "ЧСС, уд/мин",
}

stats = df[features].describe().round(2)
stats.index = ["Кол-во", "Среднее", "Ст.откл.", "Мин", "25%", "50%", "75%", "Макс"]
stats.columns = [feature_names_ru[f] for f in features]
st.dataframe(stats, use_container_width=True)

st.divider()

# ====== Распределение классов ======
st.subheader("⚖️ Баланс классов")
st.image(str(FIGURES_DIR / "01_class_balance.png"), use_container_width=True)

st.warning(
    "⚠️ Датасет имеет естественный дисбаланс классов "
    "(больные коровы встречаются реже здоровых). "
    "Для борьбы с этим в моделях применяется `class_weight='balanced'`, "
    "а основной метрикой выбран F1-macro, чувствительный к редким классам."
)

st.divider()

# ====== Распределения признаков по классам ======
st.subheader("📊 Распределения признаков по группам здоровья")
st.markdown(
    "Видно, что классы хорошо различимы по таким показателям как **температура тела** "
    "и **время руминации** — это подтверждается анализом важности признаков."
)
st.image(str(FIGURES_DIR / "02_feature_distributions.png"), use_container_width=True)

st.divider()

# ====== Корреляционная матрица ======
st.subheader("🔗 Корреляционная матрица признаков")
st.image(str(FIGURES_DIR / "03_correlation_matrix.png"), use_container_width=True)

with st.expander("💡 Интерпретация"):
    st.markdown(
        """
- **Активность и руминация** — слабая положительная корреляция (здоровые коровы и больше двигаются, и дольше жуют)
- **Температура и ЧСС** — заметная положительная связь (повышение температуры ведёт к учащению пульса)
- **Удой и активность** — положительная связь (продуктивные коровы активнее)
- **THI и температура тела** — связь через тепловой стресс

Отсутствие сильных мультиколлинеарных связей (>0.9) делает признаки независимо полезными для модели.
        """
    )

st.divider()

# ====== Boxplots ======
st.subheader("📦 Сравнительный анализ показателей (Boxplots)")
st.image(str(FIGURES_DIR / "04_boxplots.png"), use_container_width=True)

st.divider()

# ====== Интерактивная гистограмма ======
st.subheader("🎚️ Интерактивный анализ распределения")

col1, col2 = st.columns([1, 3])
with col1:
    selected_feature = st.selectbox(
        "Выберите показатель:",
        options=features,
        format_func=lambda x: feature_names_ru[x],
    )
    group_by_status = st.checkbox("Разбить по статусу", value=True)

with col2:
    if group_by_status:
        fig = px.histogram(
            df, x=selected_feature, color="health_status",
            barmode="overlay", opacity=0.6, nbins=40,
            color_discrete_map={
                "healthy": "#4CAF50",
                "at_risk": "#FFB300",
                "sick": "#E53935",
            },
            labels={selected_feature: feature_names_ru[selected_feature]},
        )
    else:
        fig = px.histogram(
            df, x=selected_feature, nbins=40,
            labels={selected_feature: feature_names_ru[selected_feature]},
        )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ====== Анализ поведения ======
st.subheader("🐄 Распределение поведения по статусам здоровья")

behavior_status = df.groupby(["health_status", "behavior_mode"]).size().unstack(fill_value=0)
behavior_status_pct = behavior_status.div(behavior_status.sum(axis=1), axis=0) * 100
behavior_status_pct = behavior_status_pct.reindex(["healthy", "at_risk", "sick"])

fig = go.Figure()
colors_beh = {"standing": "#5C6BC0", "lying": "#7E57C2", "grazing": "#43A047", "walking": "#FB8C00"}
for behavior in behavior_status_pct.columns:
    fig.add_trace(go.Bar(
        x=["Здоровые", "В зоне риска", "Больные"],
        y=behavior_status_pct[behavior].values,
        name=behavior,
        marker_color=colors_beh.get(behavior, "#888"),
    ))
fig.update_layout(
    barmode="stack",
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis_title="Доля наблюдений (%)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "🔍 Больные коровы значительно чаще находятся в положении **лёжа** — "
    "это согласуется с клиническими наблюдениями: ослабленные животные меньше двигаются."
)
