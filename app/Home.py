"""
Точка входа Streamlit-приложения «Мониторинг здоровья КРС».
Запуск: streamlit run app/Home.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Мониторинг здоровья КРС",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "Курсовая работа: Система мониторинга здоровья КРС "
            "на основе биометрических данных и ML.\n\n"
            "Пилтакян А.А., ИСП-23В, Физтех-колледж."
        ),
    },
)

# ====== Глобальные стили ======
st.markdown(
    """
    <style>
    /* Скрываем встроенные английские элементы Streamlit */
    [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }
    .stDeployButton { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }

    /* Сайдбар: тёмно-зелёный градиент */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1b3a2e 0%, #0e1f17 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e8f5e9;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1);
    }

    /* Логотип в шапке сайдбара */
    .sidebar-brand {
        text-align: center;
        padding: 0.6rem 0 0.9rem 0;
        margin: -0.5rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-brand .logo {
        font-size: 2.2rem;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .sidebar-brand h2 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .sidebar-brand p {
        color: #a5d6a7 !important;
        margin: 0.15rem 0 0 0;
        font-size: 0.78rem;
        opacity: 0.85;
    }

    /* Навигация: ссылки на страницы */
    [data-testid="stSidebarNav"] {
        padding-top: 0.4rem;
    }
    [data-testid="stSidebarNav"] ul {
        padding: 0 0.4rem;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        padding: 0.45rem 0.75rem !important;
        margin: 2px 0;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: rgba(76, 175, 80, 0.18);
        transform: translateX(2px);
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(76, 175, 80, 0.35) !important;
        font-weight: 600;
        box-shadow: inset 3px 0 0 #66bb6a;
    }
    [data-testid="stSidebarNav"] span {
        font-size: 0.95rem;
    }

    /* Заголовки секций в навигации (Основное, Мониторинг, Анализ…) */
    [data-testid="stSidebarNav"] h2,
    [data-testid="stSidebarNavSeparator"] {
        color: #81c784 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1rem 0 0.25rem 0.75rem !important;
        padding: 0 !important;
        opacity: 0.9;
    }

    /* Подвал сайдбара */
    .sidebar-footer {
        margin-top: 1rem;
        padding: 0.6rem 0.75rem;
        font-size: 0.72rem;
        color: rgba(232, 245, 233, 0.55) !important;
        line-height: 1.4;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-footer b {
        color: #c8e6c9 !important;
    }

    /* Чуть приятнее метрики на главной */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ====== Главная страница (как функция, чтобы зарегистрировать через st.Page) ======
def render_home():
    from utils import ensure_loaded
    df, bundle = ensure_loaded()

    st.title("🐄 Система мониторинга здоровья скота")
    st.markdown(
        "Прототип системы автоматического выявления отклонений в состоянии здоровья КРС "
        "на основе мультимодальных сенсорных данных и методов машинного обучения."
    )
    st.divider()

    # ====== KPI ======
    total_cows = df["cow_id"].nunique()
    total_obs = len(df)

    latest = df.sort_values("date").groupby("cow_id").tail(1)
    healthy_n = (latest["health_status"] == "healthy").sum()
    at_risk_n = (latest["health_status"] == "at_risk").sum()
    sick_n = (latest["health_status"] == "sick").sum()
    avg_milk = latest["milk_yield_kg"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🐄 Всего коров", total_cows)
    c2.metric("✅ Здоровых", healthy_n, delta=f"{healthy_n/total_cows:.0%}")
    c3.metric("⚠️ В зоне риска", at_risk_n, delta=f"{at_risk_n/total_cows:.0%}", delta_color="off")
    c4.metric("🚨 Больных", sick_n, delta=f"{sick_n/total_cows:.0%}", delta_color="inverse")
    c5.metric("🥛 Средний удой", f"{avg_milk:.1f} кг")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("📋 О системе")
        st.markdown(
            """
**Возможности системы:**

- 📊 **Дашборд стада** — мониторинг общего состояния поголовья в реальном времени
- 🐄 **Карточка животного** — детальная информация по каждой корове с историей показателей
- 🤖 **Предсказание здоровья** — оценка состояния по биометрическим данным с помощью ML
- 🔬 **Сравнение моделей** — анализ качества обученных алгоритмов классификации
- 📈 **Аналитика** — разведочный анализ данных (EDA) и визуализация закономерностей

**Используемые биометрические показатели:**

| Показатель | Обозначение | Норма |
|---|---|---|
| Температура тела | CBT | 38.0–39.5 °C |
| Индекс температуры-влажности | THI | < 72 (комфорт) |
| Удой | milk_yield_kg | 25–40 кг/сутки |
| Активность | activity | 5000–10000 шагов |
| Время руминации | rumination | 420–540 мин/сутки |
| Частота сердечных сокращений | hr | 60–80 уд/мин |
"""
        )

    with right:
        st.subheader("📂 О датасете")
        st.info(
            f"""
**Источник:** синтетический датасет,
сгенерированный по структуре MmCows
и ветеринарным нормативам КРС

**Объём:**
{total_obs:,} наблюдений
{total_cows} животных × 30 дней

**Классы:**
- ✅ Healthy ({(df['health_status']=='healthy').sum():,})
- ⚠️ At-risk ({(df['health_status']=='at_risk').sum():,})
- 🚨 Sick ({(df['health_status']=='sick').sum():,})
            """
        )

        st.subheader("🤖 Лучшая модель")
        st.success(
            f"""
**{bundle['name']}**

- Accuracy: **{bundle['metrics']['accuracy']:.3f}**
- F1-macro: **{bundle['metrics']['f1']:.3f}**
- ROC-AUC: **{bundle['metrics']['roc_auc']:.3f}**
            """
        )

    st.divider()
    st.markdown(
        """
### 🧭 Навигация
Используйте боковое меню слева для переключения между разделами системы.
"""
    )


# ====== Шапка сайдбара (брендинг) ======
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">🐄</div>
            <h2>CowHealth</h2>
            <p>Мониторинг здоровья КРС</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====== Регистрация страниц ======
HERE = Path(__file__).parent
PAGES_DIR = HERE / "pages"

home_pg = st.Page(
    render_home,
    title="Главная",
    icon="🏠",
    default=True,
    url_path="home",
)
dashboard_pg = st.Page(
    str(PAGES_DIR / "3_herd_dashboard.py"),
    title="Дашборд стада",
    icon="📊",
    url_path="dashboard",
)
card_pg = st.Page(
    str(PAGES_DIR / "1_cow_card.py"),
    title="Карточка животного",
    icon="🐄",
    url_path="cow-card",
)
predict_pg = st.Page(
    str(PAGES_DIR / "2_predict.py"),
    title="Предсказание здоровья",
    icon="🤖",
    url_path="predict",
)
analytics_pg = st.Page(
    str(PAGES_DIR / "5_analytics.py"),
    title="Аналитика данных",
    icon="📈",
    url_path="analytics",
)
models_pg = st.Page(
    str(PAGES_DIR / "4_models_comparison.py"),
    title="Сравнение моделей",
    icon="🔬",
    url_path="models",
)

nav = st.navigation(
    {
        "Основное": [home_pg],
        "Мониторинг": [dashboard_pg, card_pg, predict_pg],
        "Анализ и модели": [analytics_pg, models_pg],
    }
)

# ====== Подвал сайдбара ======
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-footer">
            <b>Курсовая работа</b><br>
            УП.03 · ИСП-23В<br>
            Пилтакян А.А., 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

nav.run()
