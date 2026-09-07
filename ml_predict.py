"""
ml_predict — ML-прогноз для Henry Hub Natural Gas
Модель: RandomForestClassifier, обучается на лету.
Фичи: лаг-доходности, RSI, momentum, волатильность, MA-спред, объём.
Цель: направление цены через 1 день (up/down).
Возвращает: ml_score от -3 до +3 и текстовое сообщение.
"""

import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт матрицу признаков из исходного OHLCV-датафрейма."""
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df.get("Volume", pd.Series(0, index=df.index))

    # Доходности
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_3d"] = close.pct_change(3)
    feat["ret_5d"] = close.pct_change(5)

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    feat["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # Momentum
    feat["mom_5"] = close / close.shift(5) - 1
    feat["mom_10"] = close / close.shift(10) - 1

    # Волатильность
    feat["vol_10"] = close.pct_change().rolling(10).std()
    feat["vol_20"] = close.pct_change().rolling(20).std()

    # MA-спреды
    feat["ma_spread_50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1
    feat["price_vs_ma20"] = close / close.rolling(20).mean() - 1

    # ATR-отношение (нормализованное)
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1)),
    )
    feat["atr_pct"] = (tr.rolling(14).mean() / close).fillna(0)

    # Объём (нормализованный)
    if vol.sum() > 0:
        feat["vol_ratio"] = vol / vol.rolling(20).mean()
    else:
        feat["vol_ratio"] = 1.0

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(0)
    return feat


def get_ml_prediction(df: pd.DataFrame) -> dict | None:
    """
    Обучает RandomForest на исторических данных и прогнозирует
    направление цены на следующий день.

    Возвращает dict:
        {
            "ml_score": int от -3 до +3,
            "message": str — текст для Telegram,
        }
    """
    try:
        if len(df) < 220:
            logging.warning("[ML] Недостаточно данных для обучения (< 220 свечей)")
            return {
                "ml_score": 0,
                "message": "🤖 ML: недостаточно данных (< 220 свечей)",
            }

        feat = _build_features(df)

        # Целевая переменная: цена выросла на следующий день (1) или нет (0)
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # Обучающая выборка: всё, кроме последних 5 строк
        split = len(feat) - 5
        X_train = feat.iloc[:split]
        y_train = y.iloc[:split]

        # Убираем NaN из целевой переменной (последние строки)
        mask = y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(X_train) < 100:
            logging.warning("[ML] Слишком мало обучающих примеров после фильтрации")
            return {
                "ml_score": 0,
                "message": "🤖 ML: мало обучающих данных",
            }

        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Обучение
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Прогноз на последней доступной свече
        X_latest = feat.iloc[-1:].values.reshape(1, -1)
        X_latest_scaled = scaler.transform(X_latest)

        proba = model.predict_proba(X_latest_scaled)[0]
        # proba[1] — вероятность роста
        p_up = proba[1] if len(proba) > 1 else 0.5

        # Оценка на тестовом окне (последние 5 свечей до прогноза)
        test_pred = model.predict(scaler.transform(feat.iloc[split - 1 : split + 4]))
        test_true = y.iloc[split - 1 : split + 4].dropna().values
        test_pred = test_pred[: len(test_true)]
        accuracy = (test_pred == test_true).mean() if len(test_true) > 0 else 0

        # Скоринг: чем выше вероятность роста, тем позитивнее ml_score
        if p_up >= 0.70:
            ml_score = 3
        elif p_up >= 0.60:
            ml_score = 2
        elif p_up >= 0.55:
            ml_score = 1
        elif p_up <= 0.30:
            ml_score = -3
        elif p_up <= 0.40:
            ml_score = -2
        elif p_up <= 0.45:
            ml_score = -1
        else:
            ml_score = 0

        direction = "📈 рост" if p_up > 0.5 else "📉 падение"
        confidence = "высокая" if abs(p_up - 0.5) > 0.15 else "средняя" if abs(p_up - 0.5) > 0.08 else "низкая"

        msg = (
            f"🤖 ML (RandomForest, {len(X_train)} свечей)\n"
            f"Прогноз: {direction} (P={p_up:.1%})\n"
            f"Уверенность: {confidence}\n"
            f"Точность на тесте: {accuracy:.0%}\n"
            f"ML-скор: {ml_score:+d}"
        )

        logging.info(f"[ML] P(up)={p_up:.3f}, score={ml_score:+d}, accuracy={accuracy:.2f}")
        return {"ml_score": ml_score, "message": msg}

    except Exception as e:
        logging.error(f"[ML] Ошибка: {e}")
        return {
            "ml_score": 0,
            "message": f"🤖 ML: ошибка прогноза ({e})",
        }
