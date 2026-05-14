"""
EDA, обучение и сравнение моделей классификации здоровья КРС.

Модели: Logistic Regression, Random Forest, Gradient Boosting.
Балансировка: class_weight='balanced'.
Валидация: StratifiedKFold (5 фолдов) + отложенная тестовая выборка.
Метрики: accuracy, precision (macro), recall (macro), F1 (macro), ROC-AUC (OVR).
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/cow_health_dataset.csv"
OUT_DIR = "data/figures"
MODEL_DIR = "data/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["cbt_mean", "thi_mean", "milk_yield_kg", "activity", "rumination", "hr"]
TARGET = "health_status"
CLASS_ORDER = ["healthy", "at_risk", "sick"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["font.size"] = 10


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Загружено: {len(df)} строк, {df['cow_id'].nunique()} коров")
    return df


# ==================== EDA ====================

def plot_class_balance(df):
    plt.figure(figsize=(7, 4.5))
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    colors = ["#4CAF50", "#FFB300", "#E53935"]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
    plt.title("Распределение классов здоровья в датасете")
    plt.xlabel("Состояние здоровья")
    plt.ylabel("Количество наблюдений")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_class_balance.png")
    plt.close()


def plot_feature_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = {
        "cbt_mean": "Температура тела, °C",
        "thi_mean": "Индекс температуры-влажности (THI)",
        "milk_yield_kg": "Удой, кг/сутки",
        "activity": "Активность, шагов/сутки",
        "rumination": "Время руминации, мин/сутки",
        "hr": "ЧСС, уд/мин",
    }
    for ax, feat in zip(axes.flatten(), FEATURES):
        for cls, color in zip(CLASS_ORDER, ["#4CAF50", "#FFB300", "#E53935"]):
            subset = df[df[TARGET] == cls][feat]
            sns.kdeplot(subset, ax=ax, label=cls, fill=True, alpha=0.3, color=color)
        ax.set_title(titles[feat])
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    plt.suptitle("Распределение признаков по классам здоровья", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_feature_distributions.png")
    plt.close()


def plot_correlation_matrix(df):
    plt.figure(figsize=(8, 6.5))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Корреляционная матрица признаков")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_correlation_matrix.png")
    plt.close()


def plot_boxplots(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flatten(), FEATURES):
        sns.boxplot(data=df, x=TARGET, y=feat, order=CLASS_ORDER,
                    palette=["#4CAF50", "#FFB300", "#E53935"], ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("")
    plt.suptitle("Boxplots признаков по классам", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_boxplots.png")
    plt.close()


# ==================== Обучение моделей ====================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
        ),
    }


def evaluate_models(X_train, X_test, y_train, y_test, classes):
    results = {}
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n[{name}]")

        # Кросс-валидация на трейне
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="f1_macro", n_jobs=-1)
        print(f"  CV F1-macro (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Обучение на трейне и оценка на тесте
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # ROC-AUC One-vs-Rest для мультикласса
        y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
        try:
            roc_auc = auc(*roc_curve(y_test_bin.ravel(), y_proba.ravel())[:2])
        except Exception:
            roc_auc = float("nan")

        print(f"  Accuracy:       {acc:.3f}")
        print(f"  Precision (M):  {prec:.3f}")
        print(f"  Recall (M):     {rec:.3f}")
        print(f"  F1-macro:       {f1:.3f}")
        print(f"  ROC-AUC (OVR):  {roc_auc:.3f}")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
        }
    return results


def plot_comparison_table(results):
    """Сравнительная таблица в виде heatmap."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "ROC-AUC\n(OVR)"]
    df_table = pd.DataFrame({
        name: [r[m] for m in metrics] for name, r in results.items()
    }, index=metric_labels)

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_table, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={"label": "Значение метрики"}, vmin=0.5, vmax=1.0)
    plt.title("Сравнение моделей по метрикам качества", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_models_comparison.png")
    plt.close()

    # Также сохраним как CSV для отчёта
    df_table.T.to_csv(f"{OUT_DIR}/models_comparison.csv")
    print("\nТаблица сравнения сохранена.")
    return df_table


def plot_confusion_matrices(results, y_test, classes):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    label_indices = list(range(len(classes)))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"], labels=label_indices)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
        ax.set_title(f"{name}\nF1-macro = {r['f1']:.3f}")
        ax.set_xlabel("Предсказание")
        ax.set_ylabel("Истинный класс")
    plt.suptitle("Матрицы ошибок", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_confusion_matrices.png")
    plt.close()


def plot_roc_curves(results, y_test, classes):
    """ROC-кривые One-vs-Rest для лучшей модели + сравнение макро."""
    y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Слева — детальная для лучшей модели по F1
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    colors = ["#4CAF50", "#FFB300", "#E53935"]
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], best["y_proba"][:, i])
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC-кривые по классам ({best_name})")
    axes[0].legend(loc="lower right")

    # Справа — макро-средняя для всех моделей
    for (name, r), c in zip(results.items(), ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), r["y_proba"].ravel())
        axes[1].plot(fpr, tpr, color=c, lw=2,
                     label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Сравнение моделей (micro-average)")
    axes[1].legend(loc="lower right")

    plt.suptitle("ROC-анализ", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_roc_curves.png")
    plt.close()


def plot_feature_importance(results):
    """Важность признаков для древесных моделей."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
        model = results[name]["model"]
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        sns.barplot(x=[importances[i] for i in order],
                    y=[FEATURES[i] for i in order],
                    palette="viridis", ax=ax)
        ax.set_title(f"{name}: важность признаков")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_feature_importance.png")
    plt.close()


# ==================== MAIN ====================

def main():
    df = load_data()

    # EDA
    print("\n=== EDA ===")
    plot_class_balance(df)
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_boxplots(df)
    print("EDA-графики сохранены.")

    # Подготовка данных
    X = df[FEATURES].values
    y_raw = df[TARGET].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # Важно: используем порядок, который установил энкодер (алфавитный)
    classes = list(le.classes_)
    print(f"Порядок классов в энкодере: {classes}")

    # Скейлинг (для логрега принципиально)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Обучение и сравнение
    print("\n=== Обучение и сравнение моделей ===")
    results = evaluate_models(X_train, X_test, y_train, y_test, classes)

    # Сравнительные графики
    print("\n=== Графики результатов ===")
    plot_comparison_table(results)
    plot_confusion_matrices(results, y_test, classes)
    plot_roc_curves(results, y_test, classes)
    plot_feature_importance(results)

    # Выбор лучшей и сохранение
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n[BEST] Лучшая модель: {best_name} (F1-macro = {results[best_name]['f1']:.3f})")

    # Сохраняем модель, скейлер и кодировщик
    joblib.dump({
        "model":   best_model,
        "scaler":  scaler,
        "encoder": le,
        "features": FEATURES,
        "classes":  classes,
        "name":     best_name,
        "metrics":  {k: float(v) for k, v in results[best_name].items()
                     if k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    }, f"{MODEL_DIR}/health_model.pkl")
    print(f"Модель сохранена: {MODEL_DIR}/health_model.pkl")

    # Classification report для лучшей в текстовом виде
    y_pred_best = results[best_name]["y_pred"]
    report = classification_report(y_test, y_pred_best, target_names=classes, digits=3)
    with open(f"{OUT_DIR}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Лучшая модель: {best_name}\n\n")
        f.write(report)
    print("Отчёт классификации сохранён.")


if __name__ == "__main__":
    main()
