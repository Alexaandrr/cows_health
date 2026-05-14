"""
Интерактивное предсказание здоровья по введённым биометрическим данным.

Гибридная схема: ML-модель + клинический guard (rule-based override).
ML отлично работает в пределах тренировочного распределения, но линейная модель
экстраполирует за его границы (например, T=36.5°C даёт «healthy 99.9%»).
Чтобы это исправить, перед показом ML-вердикта запускаем проверку клинических
правил: при критических отклонениях ставим вердикт «КРИТИЧЕСКОЕ СОСТОЯНИЕ»
и помечаем ML-предсказание как ненадёжное (out-of-distribution).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_loaded

df, bundle = ensure_loaded()
model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
feature_names = bundle["features"]
classes = bundle["classes"]

# Клинически критические пороги (вне их — состояние требует ветеринара
# независимо от вывода ML; модель в этих регионах данных не видела).
# Формат: (label, unit, crit_low, crit_high, name_low, name_high)
CRITICAL = {
    "cbt_mean":      ("Температура тела", "°C",     37.0, 41.0, "гипотермия",                       "гипертермия / лихорадка"),
    "thi_mean":      ("Индекс THI",       "",       None, 84,   None,                                "экстремальный тепловой стресс"),
    "milk_yield_kg": ("Удой",             "кг",     3,    None, "критическое падение продуктивности", None),
    "activity":      ("Активность",       "шагов",  1000, None, "полное отсутствие двигательной активности", None),
    "rumination":    ("Руминация",        "мин",    180,  None, "критическое снижение жвачки",       None),
    "hr":            ("ЧСС",              "уд/мин", 50,   115,  "брадикардия",                       "тяжёлая тахикардия"),
}

# Границы тренировочного распределения — для мягкого OOD-флага
TRAIN_MIN = df[list(CRITICAL.keys())].min()
TRAIN_MAX = df[list(CRITICAL.keys())].max()


def clinical_alarms(inputs: dict) -> list[dict]:
    """Возвращает список критических клинических отклонений."""
    alarms = []
    for key, val in inputs.items():
        lbl, unit, lo, hi, name_lo, name_hi = CRITICAL[key]
        if lo is not None and val < lo:
            alarms.append({"feature": lbl, "value": val, "unit": unit,
                           "sign": name_lo, "threshold": f"< {lo}"})
        elif hi is not None and val > hi:
            alarms.append({"feature": lbl, "value": val, "unit": unit,
                           "sign": name_hi, "threshold": f"> {hi}"})
    return alarms


def ood_features(inputs: dict) -> list[str]:
    """Признаки, лежащие за пределами тренировочного распределения."""
    out = []
    for key, val in inputs.items():
        if val < TRAIN_MIN[key] or val > TRAIN_MAX[key]:
            lbl = CRITICAL[key][0]
            out.append(f"{lbl} = {val} (обучение видело {TRAIN_MIN[key]:.1f}–{TRAIN_MAX[key]:.1f})")
    return out


st.title("🤖 Предсказание состояния здоровья")
st.markdown(
    "Введите биометрические показатели коровы — система выдаст вердикт на основе "
    "**комбинации ML-модели и клинических правил** (rule-based override). "
    "При критических отклонениях правила перекрывают ML, так как модель обучалась "
    "только на физиологически реалистичных диапазонах и не умеет распознавать, например, гипотермию."
)

# ====== Форма ввода ======
st.subheader("📝 Ввод биометрических данных")

col1, col2, col3 = st.columns(3)

with col1:
    cbt = st.slider("🌡️ Температура тела (°C)", 36.5, 42.0, 38.7, 0.05,
                    help="Норма: 38.0–39.5 °C · Критично: <37.0 или >41.0")
    thi = st.slider("💧 Индекс THI", 55.0, 85.0, 68.0, 0.5,
                    help="Норма: <72 (комфорт), 72–78 (тепловой стресс) · Критично: >84")

with col2:
    milk = st.slider("🥛 Удой (кг/сутки)", 0.0, 50.0, 32.0, 0.5,
                     help="Норма: 25–40 кг · Критично: <3")
    activity = st.slider("🚶 Активность (шагов/сутки)", 0, 12000, 7800, 100,
                         help="Норма: 5000–10000 · Критично: <1000")

with col3:
    rumination = st.slider("🌾 Время руминации (мин/сутки)", 100, 700, 480, 5,
                           help="Норма: 420–540 мин · Критично: <180")
    hr = st.slider("💓 ЧСС (уд/мин)", 40, 130, 68, 1,
                   help="Норма: 60–80 уд/мин · Критично: <50 или >115")

st.divider()

# ====== Кнопка предсказания ======
predict_btn = st.button("🔍 Анализировать состояние", type="primary", use_container_width=True)

if predict_btn:
    # Формируем вектор признаков в правильном порядке
    inputs = {
        "cbt_mean": cbt, "thi_mean": thi, "milk_yield_kg": milk,
        "activity": activity, "rumination": rumination, "hr": hr,
    }
    X_input = np.array([[inputs[f] for f in feature_names]])
    X_scaled = scaler.transform(X_input)

    prediction_idx = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    predicted_class = classes[prediction_idx]

    alarms = clinical_alarms(inputs)
    ood = ood_features(inputs)

    # ====== Финальный вердикт: правила перекрывают ML, если что-то критично ======
    st.subheader("📊 Результат анализа")

    if alarms:
        signs_md = "\n".join(
            f"- **{a['feature']}** = {a['value']} {a['unit']} — {a['sign']} (порог {a['threshold']})"
            for a in alarms
        )
        st.error(
            f"### 🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ\n"
            f"Сработали клинические правила — обнаружены опасные физиологические отклонения:\n\n"
            f"{signs_md}\n\n"
            f"**Требуется НЕМЕДЛЕННОЕ вмешательство ветеринара.** "
            f"ML-предсказание ниже показано справочно — входные данные находятся "
            f"за пределами обучающего распределения и модель в этой области ненадёжна."
        )
        verdict_for_metric = "🚨 КРИТИЧЕСКОЕ"
        ml_advisory = True
    elif ood:
        st.warning(
            "⚠️ **Часть показателей выходит за пределы обучающих данных** — ML-модель "
            "вынуждена экстраполировать, и результат может быть нестабильным:\n\n- "
            + "\n- ".join(ood)
        )
        verdict_for_metric = None
        ml_advisory = True
    else:
        verdict_for_metric = None
        ml_advisory = False

    # ====== Блок ML-предсказания (справочно при alarms/ood, иначе основной) ======
    if ml_advisory and alarms:
        st.markdown("##### 🤖 ML-предсказание (справочно, может быть недостоверным)")
    elif ml_advisory and ood:
        st.markdown("##### 🤖 ML-предсказание (с пометкой о ненадёжности)")
    else:
        st.markdown("##### 🤖 ML-предсказание")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if predicted_class == "healthy":
            st.success("### ✅ ЗДОРОВА")
            if not ml_advisory:
                st.markdown("Все показатели в норме. Корова не требует немедленного внимания.")
        elif predicted_class == "at_risk":
            st.warning("### ⚠️ В ЗОНЕ РИСКА")
            if not ml_advisory:
                st.markdown(
                    "Отмечены отклонения в показателях. **Рекомендуется усиленное наблюдение** "
                    "и проверка условий содержания."
                )
        else:
            st.error("### 🚨 БОЛЬНА")
            if not ml_advisory:
                st.markdown(
                    "Серьёзные отклонения от нормы. **Требуется немедленный осмотр ветеринаром.**"
                )

        st.metric("Уверенность модели", f"{proba[prediction_idx]*100:.1f}%")
        if verdict_for_metric:
            st.caption(f"Итоговый вердикт системы: **{verdict_for_metric}** (по правилам)")

    with res_col2:
        proba_df = pd.DataFrame({
            "Класс": classes,
            "Вероятность": proba * 100,
        }).sort_values("Вероятность", ascending=True)

        colors_map = {"healthy": "#4CAF50", "at_risk": "#FFB300", "sick": "#E53935"}
        bar_colors = [colors_map[c] for c in proba_df["Класс"]]

        fig = go.Figure(go.Bar(
            x=proba_df["Вероятность"],
            y=proba_df["Класс"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in proba_df["Вероятность"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Распределение вероятностей по классам",
            xaxis_title="Вероятность (%)",
            xaxis_range=[0, 105],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ====== Объяснение признаков ======
    st.divider()
    st.subheader("🔬 Анализ введённых показателей")

    norms = {
        "Температура тела": (cbt, "°C", 38.0, 39.5),
        "THI": (thi, "", 0, 72),
        "Удой": (milk, "кг", 25, 50),
        "Активность": (activity, "шагов", 5000, 12000),
        "Руминация": (rumination, "мин", 420, 540),
        "ЧСС": (hr, "уд/мин", 60, 80),
    }

    cols = st.columns(3)
    for i, (name, (val, unit, lo, hi)) in enumerate(norms.items()):
        with cols[i % 3]:
            if lo <= val <= hi:
                st.markdown(f"✅ **{name}**: {val} {unit} — *норма*")
            elif val < lo:
                st.markdown(f"⬇️ **{name}**: {val} {unit} — *ниже нормы* ({lo}–{hi})")
            else:
                st.markdown(f"⬆️ **{name}**: {val} {unit} — *выше нормы* ({lo}–{hi})")

else:
    st.info("👆 Установите значения показателей и нажмите кнопку для анализа.")

# ====== Раздел "Сценарии" ======
st.divider()
with st.expander("📚 Готовые сценарии для тестирования"):
    st.markdown(
        """
**Здоровая корова в комфортных условиях:**
- Температура: 38.7 °C, THI: 65, Удой: 33 кг, Активность: 8000 шагов, Руминация: 490 мин, ЧСС: 68

**Тепловой стресс (зона риска):**
- Температура: 39.3 °C, THI: 76, Удой: 26 кг, Активность: 5500 шагов, Руминация: 390 мин, ЧСС: 78

**Острое заболевание:**
- Температура: 40.2 °C, THI: 75, Удой: 15 кг, Активность: 3000 шагов, Руминация: 270 мин, ЧСС: 95

**Критическое состояние (срабатывают клинические правила):**
- Температура: 36.5 °C, ЧСС: 40, Удой: 0, Активность: 0, Руминация: 100, THI: 55
  *(гипотермия + брадикардия — ML выдаст «healthy 99.9%», правила перекроют вердикт)*
"""
    )

# ====== Объяснение архитектуры ======
with st.expander("ℹ️ Почему система комбинирует ML и правила?"):
    st.markdown(
        """
Обученная **Logistic Regression** — линейная модель. На тренировочных данных
она показывает отличные метрики (F1-macro ≈ 0.95), потому что в датасете
больные коровы имеют **высокую** температуру (~40 °C) и **высокий** пульс (~92 уд/мин),
а здоровые — нормальные. Модель учит: «выше → хуже».

Но синтетический датасет не содержит примеров **гипотермии** (T < 37 °C)
или **брадикардии** (ЧСС < 50) — клинически это тоже опасные состояния. Линейная модель
по-прежнему экстраполирует: «температура ещё ниже → ещё «здоровее»» —
это **out-of-distribution failure**, классическая проблема линейных классификаторов.

Решение: поверх ML работает слой **клинических правил** на основе ветеринарных
нормативов. Если хоть один показатель пересекает критический порог (см. подсказки
у слайдеров) — система выдаёт вердикт «🚨 КРИТИЧЕСКОЕ» независимо от ML,
а вывод модели помечается как справочный. Такая гибридная схема (ML + expert rules)
широко применяется в production-системах медицинской и ветеринарной диагностики.
"""
    )
