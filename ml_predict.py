# ml_predict.py — ML-прогноз для основного скрипта
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

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

_model = None
_config = None

def _load_model():
    global _model, _config
    if _model is not None:
        return
    if not os.path.exists(MODEL_FILE):
        logger.warning(f"ML-модель не найдена: {MODEL_FILE}. Запустите ml_train.py")
        return
    _model = joblib.load(MODEL_FILE)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            _config = json.load(f)
    logger.info(f"ML-модель загружена: {MODEL_FILE}")

def _compute_features_for_latest(df):
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
        features[f"ret_{lag}"] = close.pct_change(lag).shift

