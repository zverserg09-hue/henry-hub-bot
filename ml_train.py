# ml_train.py — обучение ML-модели для Henry Hub
import os
import sys
import time
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "NG=F"
HORIZON = 5
THRESHOLD_ATR = 0.5
MIN_TRAIN = 500
TEST_SIZE = 100
STEP = 50
MODEL_FILE = "ml_model_henry_hub.joblib"
CONFIG_FILE = "ml_config.json"

FEATURE_COLS = [
    "rsi", "dist_ma50", "dist_ma200", "bb_pos", "bb_width",
    "macd_hist", "atr_norm", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "rel_volume",
    "roc_1", "roc_3", "roc_5", "roc_10",
    "ret_1", "ret_2", "ret_3", "ret_5",
    "poc_dist", "rsi_slope",
]

def fetch_history(years=5):
    import requests
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
        f"?period1={int(time.time()) - 365*86400*years}"
        f"&period2={int(time.time())}"
        f"&interval=1d"
    )
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    data = r.json()
    timestamps = data["chart"]["result"][0]["timestamp"]
    quotes = data["chart"]["result"][0]["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Date": [datetime.fromtimestamp(t) for t in timestamps],
        "Open": quotes["open"],
        "High": quotes["high"],
        "Low": quotes["low"],
        "Close": quotes["close"],
        "Volume": quotes["volume"],
    })
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)
    df.index = df.index.normalize()
    return df

def compute_features(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    features = pd.DataFrame(index=df.index)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], np.nan)

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    features["dist_ma50"] = close / ma50 - 1
    features["dist_ma200"] = close / ma200 - 1

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    features["bb_pos"] = ((close - bb_lower) / bb_width).replace([np.inf, -np.inf], np.nan)
    features["bb_width"] = bb_width / close

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    features["macd_hist"] = (macd - signal_line) / close

    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1)),
    )
    features["atr_norm"] = tr.rolling(14).mean() / close

    month = df.index.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    dow = df.index.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    features["rel_volume"] = vol / vol.rolling(20).mean()

    for p in [1, 3, 5, 10]:
        features[f"roc_{p}"] = close.pct_change(p)

    for lag in [1, 2, 3, 5]:
        features[f"ret_{lag}"] = close.pct_change(lag).shift(lag)

    close_vals = close.values
    poc_dist = np.full(len(close_vals), np.nan)
    for i in range(60, len(close_vals)):
        median = np.median(close_vals[i - 60:i])
        poc_dist[i] = (close_vals[i] - median) / median
    features["poc_dist"] = poc_dist

    features["rsi_slope"] = features["rsi"].diff(3)

    return features, tr.rolling(14).mean()

def make_labels(close, atr, horizon, threshold_atr):
    fwd_ret = close.shift(-horizon) / close - 1
    fwd_atr = fwd_ret / atr
    labels = np.where(fwd_atr > threshold_atr, 2,
             np.where(fwd_atr < -threshold_atr, 0, 1))
    return pd.Series(labels, index=close.index), fwd_atr

def walk_forward_backtest(X, y, dates):
    results = []
    n = len(X)
    start = MIN_TRAIN
    while start + TEST_SIZE <= n:
        train_end = start
        test_end = start + TEST_SIZE
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=2.0, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "train_size": train_end,
            "test_start": str(dates[train_end].date()),
            "test_end": str(dates[test_end - 1].date()),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })
        start += STEP
    return pd.DataFrame(results)

def train():
    logger.info("Загрузка истории NG=F (5 лет)...")
    df = fetch_history(years=5)
    logger.info(f"Получено {len(df)} свечей: {df.index[0].date()} — {df.index[-1].date()}")

    logger.info("Вычисление признаков...")
    features, atr = compute_features(df)
    labels, fwd_atr = make_labels(df["Close"], atr, HORIZON, THRESHOLD_ATR)

    features["label"] = labels.values
    features_clean = features.dropna()
    mask = ~fwd_atr.loc[features_clean.index].isna()
    features_clean = features_clean[mask]

    X = features_clean[FEATURE_COLS].values
    y = features_clean["label"].values
    dates = features_clean.index

    logger.info(f"Датасет: {len(X)} строк, {len(FEATURE_COLS)} признаков")
    logger.info(f"Классы: падение={np.sum(y==0)}, нейтрально={np.sum(y==1)}, рост={np.sum(y==2)}")

    logger.info("Walk-forward backtest...")
    results = walk_forward_backtest(X, y, dates)
    logger.info(f"Точность: {results['accuracy'].mean():.1%} ± {results['accuracy'].std():.1%}")
    logger.info(f"F1:       {results['f1'].mean():.1%}")
    logger.info(f"Лучший:   {results['accuracy'].max():.1%}")
    logger.info(f"Худший:   {results['accuracy'].min():.1%}")

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=20, l2_regularization=2.0, random_state=42,
    )
    model.fit(X, y)

    last = features_clean[FEATURE_COLS].iloc[-1:].values
    pred = model.predict(last)[0]
    proba = model.predict_proba(last)[0]
    classes = model.classes_

    logger.info(f"Прогноз на {dates[-1].date()}:")
    for i, cls in enumerate(classes):
        label = {0: "📉 Падение", 1: "➡️ Нейтрально", 2: "📈 Рост"}.get(cls, str(cls))
        logger.info(f"  {label}: {proba[i]:.1%}")

    joblib.dump(model, MODEL_FILE)
    config = {
        "feature_cols": FEATURE_COLS,
        "horizon": HORIZON,
        "threshold_atr": THRESHOLD_ATR,
        "classes": {str(c): {0: "down", 1: "neutral", 2: "up"}.get(c, str(c))
                     for c in model.classes_},
        "backtest_accuracy": float(results['accuracy'].mean()),
        "backtest_f1": float(results['f1'].mean()),
        "train_date": datetime.now().isoformat(),
        "train_samples": len(X),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Модель сохранена: {MODEL_FILE}")
    logger.info(f"✅ Конфиг сохранён: {CONFIG_FILE}")
    logger.info(f"✅ Точность backtest: {results['accuracy'].mean():.1%}")

if __name__ == "__main__":
    train()

