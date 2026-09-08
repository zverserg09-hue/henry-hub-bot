"""
Henry Hub Natural Gas — автоматическая система сигналов
Анализ: техника + уровни + Volume Profile + запасы EIA + новости + Powerburn + ML
Источники цен: Yahoo Finance (NG=F фьючерс) + EIA API v2 (Henry Hub спот, RNGWHHD)
Режимы: --once (один прогон) | --loop (цикл) | --test (без Telegram)

Версия: 2026-09-08 (Strategy v3)
Все изменения:

  СКОРИНГ И СИГНАЛЫ:
  1. Расширена нейтральная зона: score -1..+1 → ВНЕ ПОЗИЦИИ (было только 0)
  2. Новая карта порогов: СИЛЬНЫЙ ±6, средний ±4, слабый ±2
  3. RSI+MA200: в сильном тренде (ADX≥25) RSI>70 или <30 не штрафуется
  4. ADX-фильтр: BB работают как контртренд в боковике, нейтральны в тренде
  5. Сезонность снижена с ±3 до ±1 — статистическая склонность, не сигнал
  6. ML: не блокирующий — без ML торгуются только сильные сигналы (|score|≥4)
  7. News score=0: различие «нет заголовков» (фиоды упали) vs «нейтральные» (даунгрейд)

  ФУНДАМЕНТ:
  8. Storage: сезонная норма по месяцам вместо фиксированных 3000 Bcf
  9. Volume Profile: детектор нулевого объёма — отключение VP-скоринга при vol=0

  НОВОСТИ v2 (накопительная модель + качество):
  10. Word-boundary regex: «heat» не совпадает с «wheat», «storm» не с «stormy»
  11. Масштабирование по подтверждениям: 1 заголовок ×0.4, 2 ×0.7, 3+ ×1.0
  12. Мульти-категорийный бонус: 2+ однонаправленные категории → ±1
  (вес ±5 сохранён — для газа новости критичны)

  РИСК-МЕНЕДЖМЕНТ:
  13. TP1=3×ATR, TP2=5×ATR (R/R=2:1, было 1.33:1)
"""

import os
import re
import sys
import time
import json
import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ML-модуль
try:
    from ml_predict import get_ml_prediction
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ml_predict не найден — ML-прогноз отключён")

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
EIA_API_KEY        = os.environ.get("EIA_API_KEY", "")

STORAGE_CURRENT_BCF = float(os.environ.get("STORAGE_CURRENT_BCF", "3153"))
LAST_STORAGE_BUILD  = float(os.environ.get("LAST_STORAGE_BUILD", "15"))
STORAGE_FORECAST    = float(os.environ.get("STORAGE_FORECAST", "19"))

SYMBOL = "NG=F"
LOG_FILE = "henry_hub_signals.log"
LAST_SIGNAL_FILE = "last_signal.json"
NEWS_HISTORY_FILE = "news_history.json"

MSK = ZoneInfo("Europe/Moscow")

CME_START_HOUR_MSK = 9
CME_END_HOUR_MSK   = 23

REGULAR_INTERVAL_HOURS = 4
SCORE_CHANGE_THRESHOLD  = 3

# ── Настройки новостного модуля v2 ──
NEWS_DECAY_HOURS         = 24    # период полураспада (часы)
NEWS_MAX_AGE_HOURS       = 48    # удалять новости старше этого срока
NEWS_BURST_WINDOW_HOURS  = 2     # окно для детекции всплеска (часы)
NEWS_BURST_THRESHOLD     = 3     # сколько однонаправленных новостей = всплеск
NEWS_MOMENTUM_WEIGHT     = 0.5   # вес моментума в composite-скоре

# ── Сезонная норма запасов (5-летнее среднее по месяцам, Bcf) ──
STORAGE_SEASONAL_NORM = {
    1: 2700, 2: 2300, 3: 1900, 4: 1700, 5: 1900, 6: 2200,
    7: 2500, 8: 2800, 9: 3100, 10: 3400, 11: 3600, 12: 3300,
}

# ── ADX пороги ──
ADX_TREND_THRESHOLD  = 25   # ADX ≥ 25 → тренд
ADX_RANGE_THRESHOLD  = 20   # ADX < 20 → боковик

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# ── Сессия с retry ──
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

session = get_session()

# ── Отправка запроса к EIA с сохранением [] в URL ──
def eia_get(url, timeout=15):
    req = requests.Request("GET", url, headers={"User-Agent": "Mozilla/5.0"})
    prepared = req.prepare()
    prepared.url = url
    return session.send(prepared, timeout=timeout)

# ============================================================
# МОДУЛЬ 1: ЦЕНОВЫЕ ДАННЫЕ — Yahoo + EIA
# ============================================================

def fetch_prices_yahoo():
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
            f"?period1={int(time.time()) - 365*86400*2}"
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
        logging.info(f"Yahoo: получено {len(df)} свечей")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки цен Yahoo: {e}")
        return None

def fetch_prices_eia():
    if not EIA_API_KEY:
        logging.info("EIA API key не задан — цены EIA пропускаются")
        return None

    key_preview = EIA_API_KEY[:4] + "..." + EIA_API_KEY[-4:] if len(EIA_API_KEY) > 8 else "***"
    logging.info(f"[EIA Prices] Ключ установлен: {key_preview}")

    url = (
        f"https://api.eia.gov/v2/natural-gas/pri/fut/data/"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=daily"
        f"&data[0]=value"
        f"&facets[series][]=RNGWHHD"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&length=730"
    )

    logging.info(f"[EIA Prices] Запрос: {url[:100]}...")

    try:
        r = eia_get(url, timeout=15)
        logging.info(f"[EIA Prices] HTTP {r.status_code}, Content-Type: {r.headers.get('content-type', '')}")

        if r.status_code != 200:
            logging.error(f"[EIA Prices] Ответ: {r.text[:500]}")
            return None

        data = r.json()
        records = data.get("response", {}).get("data", [])
        logging.info(f"[EIA Prices] Получено записей: {len(records)}")

        if not records:
            logging.warning("[EIA Prices] Пустой массив data — проверьте серию RNGWHHD")
            return None

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["period"])
        df["Close"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        df = df.sort_values("Date").set_index("Date")
        df.index = df.index.normalize()

        df["Open"]  = df["Close"].shift(1)
        df["High"]  = df[["Open", "Close"]].max(axis=1)
        df["Low"]   = df[["Open", "Close"]].min(axis=1)
        df["Volume"] = 0
        df.dropna(inplace=True)

        logging.info(f"✅ EIA: получено {len(df)} дневных цен Henry Hub")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки цен EIA: {e}")
        return None

def fetch_prices():
    df_yahoo = fetch_prices_yahoo()
    df_eia   = fetch_prices_eia()

    source_label = ""
    primary = None

    if df_yahoo is not None and len(df_yahoo) >= 50:
        primary = df_yahoo.copy()
        source_label = "Yahoo Finance (NG=F фьючерс)"
        logging.info(f"[ИСТОЧНИКИ] Загружено {len(df_yahoo)} свечей из Yahoo Finance")
    elif df_eia is not None and len(df_eia) >= 50:
        primary = df_eia.copy()
        source_label = "EIA (Henry Hub спот)"
        logging.info(f"[ИСТОЧНИКИ] Загружено {len(df_eia)} цен из EIA API")
    else:
        logging.error("[ИСТОЧНИКИ] Не удалось получить данные ни из Yahoo, ни из EIA")
        raise RuntimeError("Не удалось получить достаточно данных ни из Yahoo, ни из EIA")

    if df_eia is not None:
        eia_close = df_eia["Close"].rename("EIA_Spot")
        primary = primary.join(eia_close, how="left")
        matched = primary["EIA_Spot"].notna().sum()
        total = len(primary)
        logging.info(f"[ИСТОЧНИКИ] EIA_Spot: {matched}/{total} дат совпали при join")
    else:
        primary["EIA_Spot"] = np.nan
        logging.warning("[ИСТОЧНИКИ] Данные EIA недоступны — колонка EIA_Spot заполнена NaN")

    logging.info(f"[ИСТОЧНИКИ] Финальный источник цен: {source_label}")
    return primary, source_label

# ============================================================
# МОДУЛЬ 2: ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (с ADX)
# ============================================================

def calc_adx(df, period=14):
    """Расчёт ADX (Average Directional Index) по Wilder."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()  # prevLow - Low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index, dtype=float
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index, dtype=float
    )

    atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * plus_dm_s / atr_w
    minus_di = 100 * minus_dm_s / atr_w

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    dx = dx.replace([np.inf, -np.inf], np.nan)

    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    result = adx.iloc[-1]
    return result if not np.isnan(result) else 25.0


def calc_indicators(df):
    close = df["Close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    if np.isnan(rsi):
        rsi = 50.0

    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd - signal_line).iloc[-1]

    high, low = df["High"], df["Low"]
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1)),
    )
    atr = tr.rolling(14).mean().iloc[-1]

    adx = calc_adx(df)

    return {
        "price": close.iloc[-1], "rsi": rsi, "ma50": ma50, "ma200": ma200,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
        "macd_hist": macd_hist, "atr": atr, "adx": adx,
    }

def seasonality_score(month):
    # Снижено с ±3 до ±1 — статистическая склонность, не торговый сигнал
    scores = {
        1: 1, 2: 1, 3: 0, 4: -1, 5: -1, 6: 0,
        7: 0, 8: 0, 9: -1, 10: 0, 11: 1, 12: 1,
    }
    return scores.get(month, 0)

# ============================================================
# МОДУЛЬ 3: УРОВНИ
# ============================================================

def find_swing_levels(df, swing_bars=6):
    data = df.tail(120)
    highs = data["High"].values
    lows = data["Low"].values
    resistance, support = [], []
    for i in range(swing_bars, len(highs) - swing_bars):
        if highs[i] == max(highs[i - swing_bars : i + swing_bars + 1]):
            resistance.append(highs[i])
        if lows[i] == min(lows[i - swing_bars : i + swing_bars + 1]):
            support.append(lows[i])
    return sorted(set(support)), sorted(set(resistance), reverse=True)

def calc_pivots(df):
    last = df.iloc[-1]
    h, l, c = last["High"], last["Low"], last["Close"]
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h,
            "R2": p + (h - l), "S2": p - (h - l)}

# ============================================================
# МОДУЛЬ 4: VOLUME PROFILE (с детектором нулевого объёма)
# ============================================================

def volume_profile(df, lookback=60, num_bins=40):
    data = df.tail(lookback)
    if len(data) == 0:
        return {"poc": 0, "val": 0, "vah": 0, "hvn": [], "zero_volume": True}

    # Проверка: есть ли реальный объём
    avg_vol = data["Volume"].mean() if "Volume" in data.columns else 0
    zero_volume = (avg_vol is None or np.isnan(avg_vol) or avg_vol < 100)

    min_p = data["Low"].min()
    max_p = data["High"].max()
    if min_p == max_p:
        return {"poc": min_p, "val": min_p, "vah": min_p, "hvn": [min_p], "zero_volume": zero_volume}

    bins = np.linspace(min_p, max_p, num_bins + 1)
    vol_by_bin = np.zeros(num_bins)
    for _, row in data.iterrows():
        price = row["Close"]
        vol = row.get("Volume", 1)
        if vol is None or (isinstance(vol, float) and np.isnan(vol)):
            vol = 1
        if zero_volume:
            vol = 1  # Равномерное распределение при отсутствии объёма
        for b in range(num_bins):
            if bins[b] <= price < bins[b + 1]:
                vol_by_bin[b] += vol
                break

    poc_idx = int(np.argmax(vol_by_bin))
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
    total_vol = vol_by_bin.sum()
    if total_vol == 0:
        return {"poc": poc, "val": min_p, "vah": max_p, "hvn": [poc], "zero_volume": zero_volume}

    target = 0.70 * total_vol
    accumulated = vol_by_bin[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while accumulated < target and (lo_idx > 0 or hi_idx < num_bins - 1):
        left_vol = vol_by_bin[lo_idx - 1] if lo_idx > 0 else 0
        right_vol = vol_by_bin[hi_idx + 1] if hi_idx < num_bins - 1 else 0
        if right_vol >= left_vol and hi_idx < num_bins - 1:
            hi_idx += 1
            accumulated += vol_by_bin[hi_idx]
        elif lo_idx > 0:
            lo_idx -= 1
            accumulated += vol_by_bin[lo_idx]
        else:
            break

    val = (bins[lo_idx] + bins[lo_idx + 1]) / 2
    vah = (bins[hi_idx] + bins[hi_idx + 1]) / 2
    top_indices = np.argsort(vol_by_bin)[-3:][::-1]
    hvn = [(bins[i] + bins[i + 1]) / 2 for i in top_indices if vol_by_bin[i] > 0]
    return {"poc": poc, "val": val, "vah": vah, "hvn": hvn, "zero_volume": zero_volume}

def format_levels_message(price, vp, support_lvls, resistance_lvls, pivots):
    level_score = 0
    zero_vol = vp.get("zero_volume", False)

    msg = f"POC: ${vp['poc']:.3f}"
    if zero_vol:
        msg += " ⚠️ (нет данных объёма — равномерное распределение)"
    msg += "\n"
    msg += f"Value Area: ${vp['val']:.3f} — ${vp['vah']:.3f}\n"

    nearest_sup = None
    nearest_res = None

    # ── Swing levels: всегда работают ──
    sups_below = [s for s in support_lvls if s < price]
    if sups_below:
        nearest_sup = max(sups_below)
        dist = abs(price - nearest_sup) / price
        msg += f"🟢 Поддержка: ${nearest_sup:.3f} ({dist*100:.1f}%)\n"
        if dist < 0.015:
            level_score += 1
    else:
        msg += "🟢 Поддержка: нет в окне\n"

    ress_above = [r for r in resistance_lvls if r > price]
    if ress_above:
        nearest_res = min(ress_above)
        dist = abs(nearest_res - price) / price
        msg += f"🔴 Сопротивление: ${nearest_res:.3f} ({dist*100:.1f}%)\n"
        if dist < 0.015:
            level_score -= 1
    else:
        msg += "🔴 Сопротивление: нет в окне\n"

    msg += f"📐 Pivot: P=${pivots['P']:.3f} R1=${pivots['R1']:.3f} S1=${pivots['S1']:.3f}\n"

    # ── VP-скоринг: только при реальном объёме ──
    if not zero_vol:
        if price > vp["poc"]:
            level_score += 1
        if vp["val"] <= price <= vp["vah"]:
            pass
        elif price > vp["vah"]:
            level_score += 1
        elif price < vp["val"]:
            level_score -= 1

    if vp["hvn"]:
        msg += f"📊 HVN: {', '.join([f'${h:.3f}' for h in vp['hvn']])}\n"

    return msg, level_score, nearest_sup, nearest_res

# ============================================================
# МОДУЛЬ 5: ЗАПАСЫ EIA (с сезонной нормой)
# ============================================================

def get_eia_storage():
    if not EIA_API_KEY:
        logging.info("EIA API key не задан — fallback-значения запасов")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

    attempts = [
        {"facets": "facets[duoarea][]=NUS&facets[process][]=SAV", "label": "duoarea=NUS+process=SAV"},
        {"facets": "facets[process][]=SAV", "label": "process=SAV"},
        {"facets": "facets[duoarea][]=NUS", "label": "duoarea=NUS"},
        {"facets": "", "label": "без facets"},
    ]

    base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for attempt in attempts:
        try:
            url = (
                f"{base_url}"
                f"?api_key={EIA_API_KEY}"
                f"&frequency=weekly"
                f"&data[0]=value"
                f"&start={start_date}"
            )
            if attempt["facets"]:
                url += f"&{attempt['facets']}"
            url += (
                f"&sort[0][column]=period"
                f"&sort[0][direction]=desc"
                f"&length=50"
            )

            logging.info(f"[EIA Storage] Попытка ({attempt['label']}): {url[:120]}...")

            r = eia_get(url, timeout=15)
            logging.info(f"[EIA Storage] {attempt['label']} → HTTP {r.status_code}")

            if r.status_code != 200:
                logging.warning(f"[EIA Storage] {attempt['label']} ответ: {r.text[:300]}")
                continue

            records = r.json().get("response", {}).get("data", [])
            logging.info(f"[EIA Storage] {attempt['label']} → записей: {len(records)}")

            if not records:
                continue

            us_records = []
            for rec in records:
                area = str(rec.get("area", "")) + str(rec.get("area-name", "")) + \
                       str(rec.get("duoarea", "")) + str(rec.get("region", ""))
                if "US" in area or "U.S." in area or "United States" in area or "NUS" in area:
                    us_records.append(rec)

            if not us_records and records:
                us_records = [max(records, key=lambda r: float(r.get("value", 0) or 0))]

            if len(us_records) >= 2:
                us_records.sort(key=lambda r: r.get("period", ""), reverse=True)
                current = float(us_records[0]["value"])
                prev = float(us_records[1]["value"])
                build = current - prev
                logging.info(f"✅ [EIA Storage] {attempt['label']}: текущие={current:.0f} Bcf, закачка={build:.0f} Bcf")
                return current, build, STORAGE_FORECAST
            elif len(us_records) == 1:
                current = float(us_records[0]["value"])
                logging.warning(f"[EIA Storage] {attempt['label']}: 1 запись ({current:.0f} Bcf)")
                return current, LAST_STORAGE_BUILD, STORAGE_FORECAST

        except Exception as e:
            logging.error(f"[EIA Storage] {attempt['label']} exception: {e}")
            continue

    logging.warning("❌ [EIA Storage] Все попытки неудачны — fallback")
    return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

def score_storage(storage_bcf, build, forecast, month=None):
    """Скоринг запасов с сезонной нормой вместо фиксированных 3000 Bcf."""
    if month is None:
        month = datetime.now().month
    norm = STORAGE_SEASONAL_NORM.get(month, 3000)

    score = 0
    pct = (storage_bcf - norm) / norm * 100
    msg = f"Текущие: {storage_bcf:.0f} Bcf ({pct:+.1f}% к сезонной норме {norm:.0f})\n"
    if pct > 5:
        score -= 2
        msg += "📈 Запасы выше нормы → давление на цену\n"
    elif pct < -5:
        score += 2
        msg += "📉 Запасы ниже нормы → поддержка цены\n"
    if build > 0 and forecast > 0:
        if build > forecast * 1.3:
            score -= 1
            msg += f"⚠️ Закачка {build:.0f} > прогноз {forecast:.0f} (медвежий сюрприз)\n"
        elif build < forecast * 0.7:
            score += 1
            msg += f"✅ Закачка {build:.0f} < прогноз {forecast:.0f} (бычий сюрприз)\n"
    return score, msg

# ============================================================
# МОДУЛЬ 6: НОВОСТИ v2 — накопительная модель + качество
# ============================================================

NEWS_KEYWORDS = {
    "weather": [("heatwave",3,"bull"),("unseasonable heat",3,"bull"),("hot weather",2,"bull"),("heat",1,"bull"),("cold snap",2,"bull"),("polar vortex",3,"bull"),("freeze",3,"bull"),("blizzard",2,"bull"),("storm",1,"bull"),("hurricane",2,"bull"),("mild weather",-2,"bear"),("warm winter",-2,"bear"),("above normal temperatures",-2,"bear")],
    "lng": [("lng exports",3,"bull"),("export surge",3,"bull"),("lng capacity",2,"bull"),("higher capacity",2,"bull"),("lng lend support",2,"bull"),("export rose",2,"bull"),("lng outage",-3,"bear"),("lng maintenance",-2,"bear"),("export down",-2,"bear"),("export delay",-2,"bear"),("freeport outage",-3,"bear"),("lng shutdown",-3,"bear"),("pipeline restrictions",1,"bull")],
    "production": [("record production",-3,"bear"),("rig activity",-2,"bear"),("production rise",-2,"bear"),("output increase",-2,"bear"),("production cut",2,"bull"),("rig count decline",2,"bull"),("supply drop",2,"bull"),("reduced output",1,"bull")],
    "demand": [("record demand",3,"bull"),("demand surge",2,"bull"),("data center",2,"bull"),("ai demand",2,"bull"),("power generation",1,"bull"),("gas generation",1,"bull"),("demand fall",-2,"bear"),("weak demand",-2,"bear")],
    "geopolitics": [("sanctions",2,"bull"),("ukraine",2,"bull"),("hormuz",2,"bull"),("middle east tension",2,"bull"),("russia gas",2,"bull"),("trade war",-1,"bear")],
}

JUNK_STOPWORDS = [
    "human verification", "captcha", "verify your identity",
    "error 404", "not found", "page not found", "access denied",
    "internal server error", "500 error", "maintenance",
    "site is under maintenance", "coming soon", "under construction",
    "xml parsing", "rss error", "forbidden", "403 error",
    "temporarily unavailable", "service unavailable", "rate limit exceeded",
]

def is_valid_title(title: str) -> bool:
    if not title or len(title.strip()) < 5:
        return False
    t_lower = title.lower()
    for word in JUNK_STOPWORDS:
        if word in t_lower:
            return False
    return True

def parse_news():
    primary_feeds = [
        "https://oilprice.com/rss/main",
        "https://invezz.com/news/commodities/feed/",
        "https://www.naturalgasworld.com/rss",
        "https://worldoil.com/rss?feed=news",
    ]
    backup_feeds = [
        "https://finance.yahoo.com/rss/sector/energy",
    ]

    titles = []
    cutoff = datetime.now() - timedelta(hours=NEWS_MAX_AGE_HOURS)

    def try_feed(feed_url, feed_type="primary"):
        feed_count = 0
        try:
            r = requests.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            content_type = r.headers.get("Content-Type", "")
            if "xml" not in content_type.lower() and "rss" not in content_type.lower() and "text" not in content_type.lower():
                logging.warning(f"[News] {feed_type}: {feed_url} → Content-Type={content_type} — пропускаем")
                return 0
            if r.status_code != 200:
                logging.warning(f"[News] {feed_type}: {feed_url} HTTP {r.status_code} — пропускаем")
                return 0
            content = r.content
            content = content.replace(b"\x00", b"").replace(b"\x0b", b"")
            try:
                root = ET.fromstring(content)
                for item in root.iter():
                    tag_name = item.tag.split("}")[-1] if "}" in item.tag else item.tag
                    if tag_name == "item":
                        title = item.findtext("title", "")
                        pub_str = item.findtext("pubDate", "")
                        if not is_valid_title(title):
                            continue
                        pub_date = None
                        if pub_str:
                            try:
                                pub_date = parsedate_to_datetime(pub_str)
                                if pub_date.tzinfo is not None:
                                    pub_date = pub_date.replace(tzinfo=None)
                            except Exception:
                                pub_date = None
                        if title and (pub_date is None or pub_date >= cutoff):
                            titles.append(title)
                            feed_count += 1
            except ET.ParseError as e:
                logging.error(f"[News] XML Parse Error for {feed_url}: {e}")
                titles_text = re.findall(
                    r"<title>(.*?)</title>",
                    content.decode("utf-8", errors="ignore"),
                    re.DOTALL,
                )
                for t in titles_text[:5]:
                    clean_t = t.strip()
                    if is_valid_title(clean_t):
                        titles.append(clean_t)
                        feed_count += 1
        except Exception as e:
            logging.warning(f"News feed error: {feed_url} — {e}")
        logging.info(f"[News] {feed_type}: {feed_url.split('/')[2]} → {feed_count} валидных заголовков")
        return feed_count

    for feed in primary_feeds:
        try_feed(feed, "primary")
    if len(titles) == 0:
        logging.info("[News] Приоритетные фиды пусты — подключаем резервные...")
        for feed in backup_feeds:
            try_feed(feed, "backup")
    else:
        logging.info(f"[News] Уже есть {len(titles)} новостей из приоритетных фидов — резервные пропускаем.")

    logging.info(f"[News] Всего собрано ВАЛИДНЫХ заголовков: {len(titles)}")
    if len(titles) == 0:
        logging.error("[News] КРИТИЧНО: Новости не получены ни из одного источника!")
    else:
        for t in titles[:3]:
            logging.info(f"[News Sample] {t[:80]}")
    return titles


# ── Хранилище истории новостей ──

def _news_hash(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _score_single_title(title: str):
    """
    Скорит один заголовок по ключевым словам с word-boundary regex.
    «heat» не совпадает с «wheat», «storm» не с «stormy».
    """
    t = title.lower()
    best = (0, None, None)
    for category, keywords in NEWS_KEYWORDS.items():
        for word, pts, direction in keywords:
            # Word-boundary regex вместо простого in
            if re.search(r'\b' + re.escape(word) + r'\b', t):
                sign = 1 if direction == "bull" else -1
                contribution = pts * sign
                # Берём максимальный по модулю результат
                if abs(contribution) > abs(best[0]):
                    best = (contribution, category, direction)
    return best


def _decay_weight(age_hours: float, half_life: float = NEWS_DECAY_HOURS) -> float:
    if age_hours < 0:
        age_hours = 0
    return 0.5 ** (age_hours / half_life)


def load_news_history() -> dict:
    try:
        with open(NEWS_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_news_history(history: dict):
    try:
        with open(NEWS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"[News History] Ошибка сохранения: {e}")


def score_news_v2(titles, history):
    """
    Накопительный скоринг с временным затуханием, моментумом, burst-детекцией,
    word-boundary regex, масштабированием по подтверждениям и мульти-категорийным бонусом.
    """
    now = datetime.now()

    # ── Шаг 1: Добавить новые заголовки ──
    new_count = 0
    for title in titles:
        h = _news_hash(title)
        if h not in history:
            raw_score, category, direction = _score_single_title(title)
            history[h] = {
                "title": title,
                "score": raw_score,
                "category": category,
                "direction": direction,
                "timestamp": now.isoformat(),
            }
            new_count += 1
        else:
            entry = history[h]
            if entry.get("category") is None:
                raw_score, category, direction = _score_single_title(title)
                entry["score"] = raw_score
                entry["category"] = category
                entry["direction"] = direction

    logging.info(f"[News v2] Новых заголовков: {new_count}")

    # ── Шаг 2: Удалить устаревшие ──
    cutoff = now - timedelta(hours=NEWS_MAX_AGE_HOURS)
    expired = [h for h, e in history.items()
               if h != "__meta__" and (
                   not e.get("timestamp") or
                   _safe_parse_dt(e["timestamp"]) < cutoff
               )]
    for h in expired:
        del history[h]
    if expired:
        logging.info(f"[News v2] Удалено устаревших: {len(expired)}")

    # ── Шаг 3: Взвешенный скор с затуханием ──
    category_scores = {}
    category_titles = {}
    category_item_counts = {}
    scored_count = 0

    for h, entry in history.items():
        if h == "__meta__":
            continue
        if entry.get("score") is None or entry.get("score") == 0:
            continue
        entry_dt = _safe_parse_dt(entry.get("timestamp", ""))
        if entry_dt is None:
            continue

        age_hours = max(0, (now - entry_dt).total_seconds() / 3600)
        weight = _decay_weight(age_hours, NEWS_DECAY_HOURS)
        weighted = entry["score"] * weight

        cat = entry.get("category", "other")
        if cat not in category_scores:
            category_scores[cat] = 0.0
            category_titles[cat] = []
            category_item_counts[cat] = 0

        category_scores[cat] += weighted
        category_item_counts[cat] += 1
        scored_count += 1

        if len(category_titles[cat]) < 2:
            emoji = "📈" if entry["score"] > 0 else "📉"
            category_titles[cat].append(
                f'  {emoji} "{entry["title"][:80]}" → {entry["score"]:+d} (w={weight:.2f})'
            )

    # ── Шаг 3b: Масштабирование по подтверждениям ──
    # 1 заголовок → ×0.4, 2 → ×0.7, 3+ → ×1.0
    for cat in category_scores:
        count = category_item_counts.get(cat, 0)
        if count == 1:
            scale = 0.4
        elif count == 2:
            scale = 0.7
        else:
            scale = 1.0
        category_scores[cat] *= scale
        logging.info(f"[News v2] Категория '{cat}': {count} заголовков, масштаб ×{scale}, score={category_scores[cat]:+.2f}")

    # Клампим каждую категорию
    for cat in category_scores:
        category_scores[cat] = max(-5, min(5, category_scores[cat]))

    weighted_total = sum(category_scores.values())
    weighted_total = max(-5, min(5, weighted_total))

    # ── Шаг 4: Моментум ──
    prev_weighted = history.get("__meta__", {}).get("prev_weighted_score", 0)
    momentum = max(-5, min(5, weighted_total - prev_weighted))
    logging.info(f"[News v2] weighted={weighted_total:+.2f}, prev={prev_weighted:+.2f}, momentum={momentum:+.2f}")

    if "__meta__" not in history:
        history["__meta__"] = {}
    history["__meta__"]["prev_weighted_score"] = weighted_total
    history["__meta__"]["last_run"] = now.isoformat()

    # ── Шаг 5: Burst-детекция ──
    burst_window_start = now - timedelta(hours=NEWS_BURST_WINDOW_HOURS)
    bull_burst = 0
    bear_burst = 0

    for h, entry in history.items():
        if h == "__meta__":
            continue
        if entry.get("score") is None or entry.get("score") == 0:
            continue
        entry_dt = _safe_parse_dt(entry.get("timestamp", ""))
        if entry_dt is None:
            continue
        if entry_dt >= burst_window_start:
            if entry["score"] > 0:
                bull_burst += 1
            elif entry["score"] < 0:
                bear_burst += 1

    burst = 0
    burst_msg = ""
    if bull_burst >= NEWS_BURST_THRESHOLD:
        burst = min(2, bull_burst - NEWS_BURST_THRESHOLD + 1)
        burst_msg = f"🔥 Бычий всплеск: {bull_burst} позитивных за {NEWS_BURST_WINDOW_HOURS}ч → +{burst}"
    elif bear_burst >= NEWS_BURST_THRESHOLD:
        burst = -min(2, bear_burst - NEWS_BURST_THRESHOLD + 1)
        burst_msg = f"🔥 Медвежий всплеск: {bear_burst} негативных за {NEWS_BURST_WINDOW_HOURS}ч → {burst}"

    # ── Шаг 5b: Мульти-категорийный бонус ──
    bull_cats = sum(1 for s in category_scores.values() if s > 0.5)
    bear_cats = sum(1 for s in category_scores.values() if s < -0.5)

    multi_cat_bonus = 0
    multi_cat_msg = ""
    if bull_cats >= 2:
        multi_cat_bonus = 1
        multi_cat_msg = f"🔗 Мульти-категорийный бонус: {bull_cats} бычьих категорий → +1"
    elif bear_cats >= 2:
        multi_cat_bonus = -1
        multi_cat_msg = f"🔗 Мульти-категорийный бонус: {bear_cats} медвежьих категорий → -1"

    # ── Шаг 6: Composite-скор ──
    composite = weighted_total + momentum * NEWS_MOMENTUM_WEIGHT + burst + multi_cat_bonus
    composite = max(-5, min(5, composite))
    composite_int = int(round(composite))

    logging.info(
        f"[News v2] composite={composite_int:+d} "
        f"(weighted={weighted_total:+.2f} + momentum={momentum:+.2f}×{NEWS_MOMENTUM_WEIGHT} "
        f"+ burst={burst:+d} + multi_cat={multi_cat_bonus:+d})"
    )

    save_news_history(history)

    # ── Формирование сообщения ──
    cat_names = {
        "weather": "🌤️ Погода",
        "lng": "🚢 LNG/Экспорт",
        "production": "⛏️ Добыча",
        "demand": "⚡ Спрос",
        "geopolitics": "🌍 Геополитика",
    }

    total_in_history = len(history) - (1 if "__meta__" in history else 0)

    msg = f"В истории: {total_in_history} новостей (окно {NEWS_MAX_AGE_HOURS}ч)\n"
    msg += f"Новых за прогон: {new_count}\n"
    msg += f"Composite: {composite_int:+d} ("
    msg += f"{'📈 бычий' if composite_int > 0 else '📉 медвежий' if composite_int < 0 else '➡️ нейтральный'})\n"

    if abs(weighted_total) > 0.1:
        msg += f"  Взвешенный: {weighted_total:+.2f}\n"
    if abs(momentum) > 0.1:
        arrow = "⬆️" if momentum > 0 else "⬇️"
        msg += f"  Моментум: {momentum:+.2f} {arrow}\n"
    if burst != 0:
        msg += f"  {burst_msg}\n"
    if multi_cat_bonus != 0:
        msg += f"  {multi_cat_msg}\n"

    msg += "\n"
    for cat, score in sorted(category_scores.items(), key=lambda x: abs(x[1]), reverse=True):
        cat_label = cat_names.get(cat, cat)
        count = category_item_counts.get(cat, 0)
        arrow = "📈" if score > 0 else "📉" if score < 0 else "➡️"
        msg += f"{cat_label}: {score:+.2f} {arrow} ({count} заголовков)\n"
        for t_line in category_titles.get(cat, []):
            msg += f"{t_line}\n"

    unscored_new = []
    for title in titles:
        h = _news_hash(title)
        entry = history.get(h)
        if entry and (entry.get("score") is None or entry.get("score") == 0):
            unscored_new.append(title)

    if unscored_new:
        msg += f"📄 Без скоринга ({len(unscored_new)} новых):\n"
        for t in unscored_new[:3]:
            msg += f"  • {t[:80]}\n"

    stats = {
        "total": total_in_history,
        "new": new_count,
        "scored": scored_count,
        "unscored": len(unscored_new),
        "composite": composite_int,
        "weighted": round(weighted_total, 2),
        "momentum": round(momentum, 2),
        "burst": burst,
        "multi_cat": multi_cat_bonus,
    }

    return composite_int, msg, stats


def _safe_parse_dt(dt_str):
    """Безопасный парсинг ISO datetime из JSON."""
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


# ============================================================
# МОДУЛЬ 7: POWERBURN
# ============================================================

def fetch_powerburn():
    if EIA_API_KEY:
        try:
            url = (
                f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
                f"?api_key={EIA_API_KEY}"
                f"&frequency=hourly"
                f"&data[0]=value"
                f"&sort[0][column]=period"
                f"&sort[0][direction]=desc"
                f"&length=500"
            )
            r = eia_get(url, timeout=15)

            if r.status_code == 200:
                records = r.json().get("response", {}).get("data", [])
                if records:
                    latest_period = records[0].get("period", "")
                    hour_records = [
                        rec for rec in records
                        if rec.get("period") == latest_period
                    ]

                    fuel_totals = {}
                    for rec in hour_records:
                        fuel_type = rec.get("fueltype", rec.get("type", ""))
                        val = float(rec.get("value", 0) or 0)
                        if fuel_type in fuel_totals:
                            fuel_totals[fuel_type] += val
                        else:
                            fuel_totals[fuel_type] = val

                    total_gen = sum(fuel_totals.values())
                    ng_gen = fuel_totals.get("NG", 0)
                    coal_gen = fuel_totals.get("COL", 0)

                    ng_pct = (ng_gen / total_gen * 100) if total_gen > 0 else 0
                    coal_pct = (coal_gen / total_gen * 100) if total_gen > 0 else 0
                    daily_bcf = (ng_gen * 7.5 / 1_000_000) * 24 if ng_gen > 0 else 0

                    logging.info(
                        f"[Powerburn] EIA API: газ={ng_pct:.1f}%, уголь={coal_pct:.1f}%, "
                        f"powerburn≈{daily_bcf:.1f} BCF/d (период: {latest_period})"
                    )
                    return {
                        "realtime_bcf": daily_bcf, "vs_yesterday": 0,
                        "daily_bcf": daily_bcf, "vs_yoy": 0,
                        "ng_pct": ng_pct, "ng_yoy": 0, "coal_pct": coal_pct,
                    }
            else:
                logging.warning(f"[Powerburn] EIA API HTTP {r.status_code}")
        except Exception as e:
            logging.error(f"[Powerburn] EIA API error: {e}")

    try:
        url = "https://www.celsiusenergy.net/p/powerburn.html"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=15)
        html = r.text
        pb_match = re.search(r"(\d{1,2}\.\d)\s*BCF", html, re.I)
        current_pb = float(pb_match.group(1)) if pb_match else 0
        daily_match = re.search(r"(\d{1,2}\.\d)\s*Bc[fF]/d", html, re.I)
        daily_bcf = float(daily_match.group(1)) if daily_match else 0
        ng_match = re.search(r"Natural\s*Gas\s*</th>\s*<td[^>]*>\s*(\d{1,2}\.?\d*)\s*%", html, re.I)
        ng_pct = float(ng_match.group(1)) if ng_match else 0
        coal_match = re.search(r"Coal\s*</th>\s*<td[^>]*>\s*(\d{1,2}\.?\d*)\s*%", html, re.I)
        coal_pct = float(coal_match.group(1)) if coal_match else 0

        if ng_pct > 0 or daily_bcf > 0:
            logging.info(f"[Powerburn] CelsiusEnergy: газ={ng_pct}%, уголь={coal_pct}%, BCF/d={daily_bcf}")
            return {
                "realtime_bcf": current_pb, "vs_yesterday": 0,
                "daily_bcf": daily_bcf, "vs_yoy": 0,
                "ng_pct": ng_pct, "ng_yoy": 0, "coal_pct": coal_pct,
            }
    except Exception as e:
        logging.error(f"[Powerburn] CelsiusEnergy error: {e}")

    logging.warning("[Powerburn] Все источники недоступны — fallback")
    return {
        "realtime_bcf": 28.0, "vs_yesterday": 0,
        "daily_bcf": 28.0, "vs_yoy": 0,
        "ng_pct": 42.0, "ng_yoy": 0, "coal_pct": 15.0,
    }

def score_powerburn(pb):
    score = 0
    msg = ""
    if pb["realtime_bcf"] == 0 and pb["daily_bcf"] == 0:
        msg += "⚠️ Данные недоступны (парсинг не удался)\n"
        return 0, msg
    msg += f"Realtime: {pb['realtime_bcf']:.1f} BCF"
    if pb["vs_yesterday"] != 0:
        msg += f" ({'+' if pb['vs_yesterday'] > 0 else ''}{pb['vs_yesterday']:.1f} к вчера)"
    msg += "\n"
    msg += f"Daily: {pb['daily_bcf']:.1f} BCF/d"
    if pb["vs_yoy"] != 0:
        msg += f" ({'+' if pb['vs_yoy'] > 0 else ''}{pb['vs_yoy']:.1f} к пр.году)"
        if pb["vs_yoy"] > 2: score += 1
        elif pb["vs_yoy"] < -2: score -= 1
    msg += "\n"
    msg += f"Доля газа в генерации: {pb['ng_pct']:.1f}%"
    if pb["ng_yoy"] != 0:
        msg += f" ({'+' if pb['ng_yoy'] > 0 else ''}{pb['ng_yoy']:.1f}% к пр.году)"
        if pb["ng_yoy"] > 2: score += 1
        elif pb["ng_yoy"] < -2: score -= 1
    msg += "\n"
    if pb["coal_pct"] > 0:
        msg += f"Доля угля: {pb['coal_pct']:.1f}%\n"
        if pb["coal_pct"] > 25:
            score -= 1
            msg += "⚠️ Уголь замещает газ (fuel switching)\n"
    return max(-3, min(3, score)), msg

# ============================================================
# МОДУЛЬ 8: СКОРИНГ И СИГНАЛ (v3)
# ============================================================

def calculate_score(ind, level_score, storage_score, season_score,
                    news_score, pb_score, ml_score=0):
    score = 0
    price, rsi, ma200 = ind["price"], ind["rsi"], ind["ma200"]
    adx = ind.get("adx", 25.0)

    is_trend = adx >= ADX_TREND_THRESHOLD
    is_range = adx < ADX_RANGE_THRESHOLD
    above_ma200 = (not np.isnan(ma200)) and (price > ma200)
    below_ma200 = (not np.isnan(ma200)) and (price < ma200)

    # ── RSI с разрешением конфликта vs MA200 ──
    if rsi > 70:
        if not (above_ma200 and is_trend):
            score -= 2  # Перекупленность вне сильного uptrend → штраф
    elif rsi < 30:
        if not (below_ma200 and is_trend):
            score += 2  # Перепроданность вне сильного downtrend → бонус
    elif rsi > 60:
        if not (above_ma200 and is_trend):
            score -= 1
    elif rsi < 40:
        if not (below_ma200 and is_trend):
            score += 1

    # ── MA200 — трендовый фильтр ──
    if not np.isnan(ma200):
        if price > ma200: score += 1
        elif price < ma200: score -= 1

    # ── Bollinger Bands — контекстные ──
    if is_range:
        # Боковик: BB = контртренд
        if abs(price - ind["bb_upper"]) < 0.02 * price: score -= 1
        if abs(price - ind["bb_lower"]) < 0.02 * price: score += 1
    else:
        # Тренд: BB касание по трегу = нейтрально, против = разворот
        if abs(price - ind["bb_upper"]) < 0.02 * price:
            if not above_ma200: score -= 1  # Перекуплен против тренда
        if abs(price - ind["bb_lower"]) < 0.02 * price:
            if not below_ma200: score += 1  # Перепродан против тренда

    # ── MACD ──
    if ind["macd_hist"] < 0: score -= 1
    elif ind["macd_hist"] > 0: score += 1

    # ── Сезонность: ±1 (статистическая склонность) ──
    if season_score > 0: score += 1
    elif season_score < 0: score -= 1

    # ── Фундамент ──
    score += storage_score + level_score + news_score + pb_score + ml_score

    return max(-15, min(15, score))


def determine_signal(score, ml_available, news_score, price,
                     support=None, resistance=None, news_headlines_count=0):
    """
    Новая карта порогов с расширенной нейтральной зоной:
      ≥ +6: 🟢 СИЛЬНЫЙ ЛОНГ
      +4..+5: 🟡 ЛОНГ
      +2..+3: ⚪ СЛАБЫЙ ЛОНГ
      -1..+1: ⬜ ВНЕ ПОЗИЦИИ
      -2..-3: 🔵 СЛАБЫЙ ШОРТ
      -4..-5: 🟠 ШОРТ
      ≤ -6: 🔴 СИЛЬНЫЙ ШОРТ
    """
    # ML — не блокирующий, а ограничивающий
    if not ml_available and abs(score) < 4:
        return "⬜ ВНЕ ПОЗИЦИИ (нет ML — торгуем только при |score| ≥ 4)"

    # News score=0: различаем «нет заголовков» vs «нейтральные»
    if news_score == 0:
        if news_headlines_count > 0:
            # Заголовки есть, но все нейтральные — ограничиваем силу сигнала
            score = max(-3, min(3, score))
        # Если заголовков 0 (фиоды упали) — не ограничиваем, доверяем технике

    # Защита от входа вплотную к уровням
    if support and price > 0 and ((price - support) / support) < 0.005:
        return "⬜ ВНЕ ПОЗИЦИИ (цена вплотную к поддержке — риск ложного пробоя)"
    if resistance and price > 0 and ((resistance - price) / price) < 0.005:
        return "⬜ ВНЕ ПОЗИЦИИ (цена вплотную к сопротивлению — риск выноса)"

    if score >= 6:
        return "🟢 СИЛЬНЫЙ ЛОНГ"
    elif score >= 4:
        return "🟡 ЛОНГ"
    elif score >= 2:
        return "⚪ СЛАБЫЙ ЛОНГ"
    elif score <= -6:
        return "🔴 СИЛЬНЫЙ ШОРТ"
    elif score <= -4:
        return "🟠 ШОРТ"
    elif score <= -2:
        return "🔵 СЛАБЫЙ ШОРТ"
    else:
        return "⬜ ВНЕ ПОЗИЦИИ"

# ============================================================
# МОДУЛЬ 9: TELEGRAM
# ============================================================

def send_telegram(text, is_change_alert=False):
    now = datetime.now(MSK)
    is_cme_hours = CME_START_HOUR_MSK <= now.hour < CME_END_HOUR_MSK
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[DEBUG] Telegram отключён — нет токена/chat_id")
        return False
    if not is_cme_hours and not is_change_alert:
        logging.info(f"Вне CME-часов — Telegram не отправляется: {text[:100]}")
        print("[Вне CME-часов] Сигнал залогирован, но не отправлен в Telegram")
        return False

    if len(text) > 4000:
        text = text[:4000] + "\n…(обрезано)"

    full_text = ("🚨 *СМЕНА СИГНАЛА* 🚨\n\n" + text) if is_change_alert else text
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": full_text},
            timeout=10,
        )
        if r.status_code == 200: return True
        logging.error(f"Telegram error: {r.status_code} {r.text}")
        return False
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False

def load_last_signal():
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            data = json.load(f)
            return data.get("signal", ""), data.get("score", 0), data.get("price", 0), data.get("timestamp", "")
    except Exception:
        return "", 0, 0, ""

def save_last_signal(signal, score, price):
    try:
        with open(LAST_SIGNAL_FILE, "w") as f:
            json.dump({"signal": signal, "score": score, "price": price,
                        "timestamp": datetime.now().isoformat()}, f)
    except Exception:
        pass

def should_send_signal(signal, score, price, prev_signal, prev_score, prev_timestamp, is_cme):
    now = datetime.now()
    if prev_signal and prev_signal != signal:
        return True, "change"
    if not prev_signal:
        return True, "first_run_force"
    if abs(score - prev_score) >= SCORE_CHANGE_THRESHOLD:
        return True, f"score_change ({prev_score:+d} → {score:+d})"
    if is_cme and prev_timestamp:
        try:
            last_dt = datetime.fromisoformat(prev_timestamp)
            if (now - last_dt).total_seconds() >= REGULAR_INTERVAL_HOURS * 3600:
                return True, "regular_interval"
        except Exception:
            return True, "regular_interval_no_timestamp"
    elif is_cme and not prev_timestamp:
        return True, "regular_no_timestamp"
    return False, "no_change"

# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

def main():
    now = datetime.now(MSK)
    logging.info(f"=== Запуск цикла {now.strftime('%Y-%m-%d %H:%M:%S')} МСК ===")

    try:
        df, source_label = fetch_prices()
        if len(df) < 50:
            logging.error("Недостаточно данных для анализа")
            return
    except Exception as e:
        logging.error(f"Критическая ошибка загрузки цен: {e}")
        return

    ind = calc_indicators(df)
    support_lvls, resistance_lvls = find_swing_levels(df)
    pivots = calc_pivots(df)
    vp = volume_profile(df, lookback=60, num_bins=40)
    level_msg, level_score, nearest_sup, nearest_res = format_levels_message(
        ind["price"], vp, support_lvls, resistance_lvls, pivots)

    storage_bcf, build, forecast = get_eia_storage()
    storage_score, storage_msg = score_storage(storage_bcf, build, forecast, month=now.month)

    season_score = seasonality_score(now.month)

    # ── Новости v2 ──
    news_titles = parse_news()
    news_history = load_news_history()
    news_score, news_msg, news_stats = score_news_v2(news_titles, news_history)

    pb_data = fetch_powerburn()
    pb_score, pb_msg = score_powerburn(pb_data)

    # ML-прогноз
    if ML_AVAILABLE:
        ml_result = get_ml_prediction(df)
        ml_score = ml_result["ml_score"] if ml_result else 0
        ml_msg = ml_result["message"] if ml_result else "🤖 ML: модель недоступна"
    else:
        ml_score = 0
        ml_msg = "🤖 ML: модуль не загружен"

    total_score = calculate_score(
        ind, level_score, storage_score, season_score,
        news_score, pb_score, ml_score
    )

    signal = determine_signal(
        score=total_score,
        ml_available=ML_AVAILABLE,
        news_score=news_score,
        price=ind["price"],
        support=nearest_sup,
        resistance=nearest_res,
        news_headlines_count=len(news_titles),
    )

    price, atr = ind["price"], ind["atr"]

    is_long = "ЛОНГ" in signal
    is_short = "ШОРТ" in signal

    # SL/TP: TP1=3×ATR (R/R=2:1), TP2=5×ATR
    if is_long:
        sl, tp1, tp2 = price - 1.5 * atr, price + 3 * atr, price + 5 * atr
        if nearest_sup and sl > nearest_sup: sl = nearest_sup - 0.02
    elif is_short:
        sl, tp1, tp2 = price + 1.5 * atr, price - 3 * atr, price - 5 * atr
        if nearest_res and sl < nearest_res: sl = nearest_res + 0.02
    else:
        sl, tp1, tp2 = price - 1.5 * atr, price + 3 * atr, price + 5 * atr

    risk = abs(price - sl)
    rr1 = abs(tp1 - price) / risk if risk > 0 else 0

    prev_signal, prev_score, prev_price, prev_timestamp = load_last_signal()
    is_cme = CME_START_HOUR_MSK <= now.hour < CME_END_HOUR_MSK
    should_send, send_reason = should_send_signal(
        signal, total_score, price, prev_signal, prev_score, prev_timestamp, is_cme)

    # Режим рынка для лога
    adx = ind.get("adx", 25)
    if adx >= ADX_TREND_THRESHOLD:
        market_mode = "ТРЕНД"
    elif adx < ADX_RANGE_THRESHOLD:
        market_mode = "Боковик"
    else:
        market_mode = "Переходный"

    log_line = (f"{signal} | score={total_score} | price=${price:.3f} | "
                f"ADX={adx:.0f} ({market_mode}) | "
                f"prev={prev_signal} score={prev_score} | "
                f"send={should_send} ({send_reason})")
    logging.info(log_line)
    print(log_line)

    save_last_signal(signal, total_score, price)

    # ── Формирование сообщения ──
    msg = f"{signal}\n"
    msg += f"Score: {total_score}/15\n"
    msg += f"Цена: ${price:.3f}\n"
    eia_spot = df["EIA_Spot"].iloc[-1] if "EIA_Spot" in df.columns else np.nan
    if not np.isnan(eia_spot):
        msg += f"EIA спот: ${eia_spot:.3f}\n"
    if is_long or is_short:
        msg += f"SL: ${sl:.3f}\nTP1: ${tp1:.3f} | TP2: ${tp2:.3f}\nR/R: {rr1:.2f}\n"

    msg += "━━━━ ИСТОЧНИКИ ━━━━\n"
    msg += f"Основной: {source_label}\n"
    if "Open" in df.columns and not df["Open"].isna().all():
        msg += "✅ Yahoo Finance (NG=F): данные загружены\n"
    else:
        msg += "❌ Yahoo Finance: данные недоступны\n"
    if "EIA_Spot" in df.columns and not df["EIA_Spot"].isna().all():
        msg += "✅ EIA API (Henry Hub спот): данные загружены\n"
    else:
        msg += "❌ EIA API: данные недоступны\n"

    msg += "━━━━ ИНДИКАТОРЫ ━━━━\n"
    msg += f"RSI: {ind['rsi']:.1f} | MA50: ${ind['ma50']:.3f} | MA200: ${ind['ma200']:.3f}\n"
    msg += f"ADX: {adx:.0f} ({market_mode})\n"
    msg += f"ATR: ${ind['atr']:.3f}\n"
    msg += "━━━━ ЗАПАСЫ EIA ━━━━\n" + storage_msg
    msg += "━━━━ УРОВНИ ━━━━\n" + level_msg
    msg += "━━━━ POWERBURN ━━━━\n" + pb_msg

    # ── Блок новостей v2 ──
    msg += "━━━━ НОВОСТИ (v2) ━━━━\n"
    if news_stats['total'] == 0:
        msg += "📰 Новостей в истории нет.\n"
        msg += "(Проверьте доступность RSS-фидов или попробуйте позже)\n"
    else:
        msg += f"📰 В истории: {news_stats['total']} | Новых: {news_stats['new']} | Скоринг: {news_stats['scored']}\n"
        msg += news_msg

    msg += "━━━━ ML-ПРОГНОЗ ━━━━\n" + ml_msg + "\n"

    if should_send:
        is_change_alert = (send_reason == "change")
        success = send_telegram(msg, is_change_alert=is_change_alert)
        print(f"✅ Отправлено в Telegram ({send_reason})" if success else f"❌ Не отправлено ({send_reason})")
    else:
        print(f"⏸ Не отправлено — нет изменений ({send_reason})")
    print(f"\n--- Полное сообщение ---\n{msg}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--once"
    if mode == "--once":
        main()
    elif mode == "--test":
        TELEGRAM_BOT_TOKEN = ""
        main()
    elif mode == "--loop":
        while True:
            try:
                main()
            except Exception as e:
                logging.error(f"Цикл error: {e}")
            time.sleep(3600)
    else:
        print(f"Неизвестный режим: {mode}")
        print("Использование: python henry_hub_signals.py [--once|--test|--loop]")
