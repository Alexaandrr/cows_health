"""
Сравнение моделей ML: метрики, confusion matrix, ROC-кривые, важность признаков.
Показываем готовые графики, обученные на этапе подготовки модели.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_loaded, FIGURES_DIR

df, bundle = ensure_loaded()

st.title("🔬 Сравнение моделей классификации")
st.markdown(
    "В рамках работы обучены и сравнены три алгоритма машинного обучения "
    "для классификации состояния здоровья КРС по биометрическим показателям."
)

# ====== Описание моделей ======
st.subheader("🧠 Используемые модели")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        """
**Logistic Regression**

Линейный классификатор. Простая интерпретируемая baseline-модель.
Применяется балансировка классов через `class_weight='balanced'`.
        """
    )
with col2:
    st.info(
        """
**Random Forest**

Ансамбль решающих деревьев (200 деревьев, max_depth=10).
Устойчив к шуму, работает с нелинейными зависимостями.
        """
    )
with col3:
    st.info(
        """
**Gradient Boosting**

Последовательное построение слабых деревьев с минимизацией ошибки.
150 деревьев, скорость обучения 0.1.
        """
    )

st.divider()

# ====== Методология ======
with st.expander("⚙️ Методология обучения и валидации"):
    st.markdown(
        """
**Подготовка данных:**
- Стратифицированное разделение train/test = 75/25
- Стандартизация признаков (`StandardScaler`)
- Балансировка классов через `class_weight='balanced'` (LogReg, RF)

**Кросс-валидация:** Stratified K-Fold, 5 фолдов, метрика — F1-macro

**Метрики качества:**
- **Accuracy** — общая доля правильных ответов
- **Precision (macro)** — точность, усреднённая по классам
- **Recall (macro)** — полнота, усреднённая по классам
- **F1-macro** — гармоническое среднее P и R (основная метрика выбора модели)
- **ROC-AUC (OVR)** — площадь под ROC-кривой, One-vs-Rest
        """
    )

st.divider()

# ====== Сравнительная таблица метрик ======
st.subheader("📊 Сравнение моделей по метрикам качества")

comparison_csv = FIGURES_DIR / "models_comparison.csv"
if comparison_csv.exists():
    table_df = pd.read_csv(comparison_csv, index_col=0)
    st.dataframe(
        table_df.style.background_gradient(cmap="YlGnBu", axis=None).format("{:.3f}"),
        use_container_width=True,
    )

st.image(str(FIGURES_DIR / "05_models_comparison.png"), use_container_width=True)

# Подсветка лучшей
best_name = bundle["name"]
st.success(
    f"🏆 **Лучшая модель: {best_name}** "
    f"(F1-macro = {bundle['metrics']['f1']:.3f}, ROC-AUC = {bundle['metrics']['roc_auc']:.3f})"
)

st.divider()

# ====== Матрицы ошибок ======
st.subheader("🎯 Матрицы ошибок")
st.markdown(
    "Confusion matrix показывает, какие классы модель путает между собой. "
    "По диагонали — правильные предсказания, вне диагонали — ошибки."
)
st.image(str(FIGURES_DIR / "06_confusion_matrices.png"), use_container_width=True)

st.divider()

# ====== ROC-кривые ======
st.subheader("📈 ROC-анализ")
st.markdown(
    "ROC-кривая показывает баланс между чувствительностью и специфичностью модели. "
    "Чем ближе кривая к левому верхнему углу — тем лучше. AUC = 1.0 — идеальный классификатор."
)
st.image(str(FIGURES_DIR / "07_roc_curves.png"), use_container_width=True)

st.divider()

# ====== Важность признаков ======
st.subheader("⭐ Важность признаков")
st.markdown(
    "Feature importance показывает, какие биометрические показатели вносят "
    "наибольший вклад в предсказание модели."
)
st.image(str(FIGURES_DIR / "08_feature_importance.png"), use_container_width=True)

st.info(
    "💡 **Вывод:** Наиболее значимыми признаками для определения состояния здоровья "
    "оказались **температура тела (CBT)** и **время руминации (rumination)**, "
    "что согласуется с практикой ветеринарной медицины: повышение температуры — "
    "классический индикатор воспалительных процессов, а снижение руминации часто "
    "предшествует клиническим проявлениям заболеваний."
)

st.divider()

# ====== Classification report ======
report_path = FIGURES_DIR / "classification_report.txt"
if report_path.exists():
    with st.expander("📝 Подробный classification report для лучшей модели"):
        st.code(report_path.read_text(encoding="utf-8"), language="text")
