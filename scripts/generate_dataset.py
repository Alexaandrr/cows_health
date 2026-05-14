"""
Генерация расширенного синтетического датасета по структуре MmCows.
100 коров × 30 дней = 3000 наблюдений, 3 класса здоровья.

Опирается на ветеринарные нормативы:
- Температура тела (CBT): норма 38.0-39.5°C, лихорадка >39.5
- THI (Temperature-Humidity Index): комфорт <72, стресс 72-78, тяжёлый >78
- Удой: норма 25-40 кг/день для дойных коров
- Активность (шаги/сутки): норма 5000-10000
- Время руминации: норма 420-540 мин/сутки
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_COWS = 100
DAYS = 30
START_DATE = datetime(2024, 7, 1)

# Распределение классов в популяции
CLASS_DISTRIBUTION = {
    "healthy": 0.72,
    "at_risk": 0.20,
    "sick": 0.08,
}

# Реалистичные диапазоны по классам (mean, std)
PROFILES = {
    "healthy": {
        "cbt_mean":      (38.7, 0.25),   # норма
        "thi_mean":      (68.0, 4.0),    # комфорт/лёгкий стресс
        "milk_yield_kg": (32.0, 4.5),    # хороший удой
        "activity":      (7800, 1300),   # норма активности
        "rumination":    (480, 35),      # норма жвачки
        "hr":            (68, 6),        # норма пульса
    },
    "at_risk": {
        "cbt_mean":      (39.3, 0.30),   # повышена
        "thi_mean":      (74.0, 3.5),    # тепловой стресс
        "milk_yield_kg": (26.0, 5.5),    # снижение удоя
        "activity":      (5500, 1500),   # снижена
        "rumination":    (390, 50),      # снижена жвачка
        "hr":            (78, 8),        # повышен пульс
    },
    "sick": {
        "cbt_mean":      (40.0, 0.45),   # лихорадка
        "thi_mean":      (76.0, 4.0),    # стресс
        "milk_yield_kg": (18.0, 6.0),    # сильный спад
        "activity":      (3200, 1200),   # резкое снижение
        "rumination":    (280, 60),      # резкое снижение
        "hr":            (92, 10),       # тахикардия
    },
}

BEHAVIOR_MODES = ["standing", "lying", "walking", "grazing"]


def assign_class(cow_idx: int) -> str:
    """Распределяем классы детерминированно по индексу коровы для стабильности."""
    n_healthy = int(N_COWS * CLASS_DISTRIBUTION["healthy"])
    n_at_risk = int(N_COWS * CLASS_DISTRIBUTION["at_risk"])
    if cow_idx < n_healthy:
        return "healthy"
    elif cow_idx < n_healthy + n_at_risk:
        return "at_risk"
    else:
        return "sick"


def sample_day(profile: dict) -> dict:
    """Генерация одной дневной записи из профиля класса."""
    row = {}
    for feature, (mean, std) in profile.items():
        val = np.random.normal(mean, std)
        # клиппинг по физиологически разумным границам
        if feature == "cbt_mean":
            val = np.clip(val, 36.5, 42.0)
        elif feature == "thi_mean":
            val = np.clip(val, 55, 85)
        elif feature == "milk_yield_kg":
            val = max(0, val)
        elif feature == "activity":
            val = max(0, int(val))
        elif feature == "rumination":
            val = np.clip(val, 0, 700)
        elif feature == "hr":
            val = np.clip(val, 40, 130)
        row[feature] = round(float(val), 2) if isinstance(val, float) else val
    return row


def generate_behavior(profile_name: str) -> str:
    """Поведение зависит от состояния."""
    if profile_name == "healthy":
        return np.random.choice(BEHAVIOR_MODES, p=[0.30, 0.25, 0.15, 0.30])
    elif profile_name == "at_risk":
        return np.random.choice(BEHAVIOR_MODES, p=[0.35, 0.40, 0.10, 0.15])
    else:  # sick
        return np.random.choice(BEHAVIOR_MODES, p=[0.15, 0.70, 0.05, 0.10])


def generate_dataset() -> pd.DataFrame:
    rows = []
    for cow_idx in range(N_COWS):
        cow_id = f"T{cow_idx+1:03d}"
        cls = assign_class(cow_idx)
        profile = PROFILES[cls]

        # лёгкие "персональные" сдвиги для каждой коровы (индивидуальная норма)
        personal_shift = {
            "cbt_mean":      np.random.normal(0, 0.10),
            "milk_yield_kg": np.random.normal(0, 1.5),
            "activity":      np.random.normal(0, 400),
            "rumination":    np.random.normal(0, 15),
            "hr":            np.random.normal(0, 2),
        }

        for day in range(DAYS):
            date = START_DATE + timedelta(days=day)
            day_row = sample_day(profile)

            # применяем персональные сдвиги
            for k, shift in personal_shift.items():
                day_row[k] += shift

            # клиппинг ещё раз после сдвигов
            day_row["cbt_mean"] = round(np.clip(day_row["cbt_mean"], 36.5, 42.0), 2)
            day_row["activity"] = max(0, int(day_row["activity"]))
            day_row["rumination"] = round(np.clip(day_row["rumination"], 0, 700), 1)
            day_row["hr"] = round(np.clip(day_row["hr"], 40, 130), 1)
            day_row["milk_yield_kg"] = round(max(0, day_row["milk_yield_kg"]), 2)
            day_row["thi_mean"] = round(np.clip(day_row["thi_mean"], 55, 85), 1)

            rows.append({
                "cow_id":         cow_id,
                "date":           date.strftime("%Y-%m-%d"),
                "cbt_mean":       day_row["cbt_mean"],
                "thi_mean":       day_row["thi_mean"],
                "milk_yield_kg":  day_row["milk_yield_kg"],
                "activity":       day_row["activity"],
                "rumination":     day_row["rumination"],
                "hr":             day_row["hr"],
                "behavior_mode":  generate_behavior(cls),
                "health_status":  cls,
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    out_path = "data/cow_health_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Сгенерирован датасет: {out_path}")
    print(f"     Записей: {len(df)}")
    print(f"     Коров: {df['cow_id'].nunique()}")
    print(f"     Дней: {df['date'].nunique()}")
    print()
    print("Распределение классов:")
    print(df["health_status"].value_counts())
    print()
    print("Распределение поведения:")
    print(df["behavior_mode"].value_counts())
    print()
    print("Статистика признаков:")
    print(df[["cbt_mean","thi_mean","milk_yield_kg","activity","rumination","hr"]].describe().round(2))
