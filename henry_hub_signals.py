"""
Henry Hub Natural Gas — автоматическая система сигналов
Анализ: техника + уровни + Volume Profile + запасы EIA + новости + Powerburn
Источники цен: Yahoo Finance (NG=F фьючерс) + EIA API v2 (Henry Hub спот, RNGWHHD)
Режимы: --once (один прогон) | --loop (цикл) | --test (без Telegram)
"""

import os
import re
import time
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import requests
import pandas as pd
import numpy as np

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
EIA_API_KEY        = os.environ.get("EIA_API_KEY", "")

STORAGE_CURRENT_BCF = float(os.environ.get("STORAGE_CURRENT_BCF", "3153"))
LAST_STORAGE_BUILD  = float(os.environ.get("LAST_STORAGE_BUILD", "15"))
STORAGE_FORECAST    = float(os.environ.get("STORAGE_FORECAST", "19"))

SYMBOL = "NG=F"
LOG_FILE = "henry_hub_signals.log"

CME_START_HOUR_MSK = 9
CME_END_HOUR_MSK   = 23

REGULAR_INTERVAL_HOURS = 4
SCORE_CHANGE_THRESHOLD  = 3   # если score изменился на эту величину — отправить даже без смены сигнала

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

LAST_SIGNAL_FILE = "last_signal.json"

# ============================================================
# МОДУЛЬ 1: ЦЕНОВЫЕ ДАННЫЕ — Yahoo + EIA
# ============================================================

def fetch_prices_yahoo():
    """Получаем дневные свечи NG=F (фьючерс) через Yahoo Finance."""
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
        logging.info(f"Yahoo: получено {len(df)} свечей")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки цен Yahoo: {e}")
        return None


def fetch_prices_eia():
    """Получаем дневные спот-цены Henry Hub через EIA API v2 (серия RNGWHHD)."""
    if not EIA_API_KEY:
        logging.info("EIA API key не задан — цены EIA пропускаются")
        return None

    try:
        url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": "RNGWHHD",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 730,  # ~2 года
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        records = data["response"]["data"]
        if not records:
            logging.warning("EIA: пустой ответ")
            return None

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["period"])
        df["Close"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        df = df.sort_values("Date").set_index("Date")
        # У EIA спот-цены только Close — синтетически достраиваем OHLC
        df["Open"]  = df["Close"].shift(1)
        df["High"]  = df[["Open", "Close"]].max(axis=1)
        df["Low"]   = df[["Open", "Close"]].min(axis=1)
        df["Volume"] = 0
        df.dropna(inplace=True)
        logging.info(f"EIA: получено {len(df)} дневных цен спот Henry Hub")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки цен EIA: {e}")
        return None


def fetch_prices():
    """Объединяет данные Yahoo (фьючерс) и EIA (спот). Возвращает основной DataFrame."""
    df_yahoo = fetch_prices_yahoo()
    df_eia   = fetch_prices_eia()

    if df_yahoo is not None and len(df_yahoo) >= 50:
        primary = df_yahoo.copy()
        source_label = "Yahoo (NG=F фьючерс)"
    elif df_eia is not None and len(df_eia) >= 50:
        primary = df_eia.copy()
        source_label = "EIA (Henry Hub спот)"
    else:
        raise RuntimeError("Не удалось получить достаточно данных ни из Yahoo, ни из EIA")

    # Если есть оба источника — добавляем колонку со спот-ценой EIA для справки
    if df_yahoo is not None and df_eia is not None:
        eia_close = df_eia["Close"].rename("EIA_Spot")
        primary = primary.join(eia_close, how="left")
    elif df_eia is not None:
        primary["EIA_Spot"] = primary["Close"]
    else:
        primary["EIA_Spot"] = np.nan

    logging.info(f"Источник цен: {source_label}")
    return primary


# ============================================================
# МОДУЛЬ 2: ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================

def calc_indicators(df):
    close = df["Close"]

    # RSI-14
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

    return {
        "price": close.iloc[-1],
        "rsi": rsi,
        "ma50": ma50,
        "ma200": ma200,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "macd_hist": macd_hist,
        "atr": atr,
    }

def seasonality_score(month):
    scores = {
        1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -1,
        7: 0, 8: -1, 9: -2, 10: -1, 11: 2, 12: 3,
    }
    return scores.get(month, 0)

# ============================================================
# МОДУЛЬ 3: УРОВНИ ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ
# ============================================================

def find_swing_levels(df, swing_bars=6):
    data = df.tail(120)
    highs = data["High"].values
    lows = data["Low"].values

    resistance = []
    support = []

    for i in range(swing_bars, len(highs) - swing_bars):
        if highs[i] == max(highs[i - swing_bars : i + swing_bars + 1]):
            resistance.append(highs[i])
        if lows[i] == min(lows[i - swing_bars : i + swing_bars + 1]):
            support.append(lows[i])

    res_vals = sorted(set(resistance), reverse=True)
    sup_vals = sorted(set(support))
    return sup_vals, res_vals

def calc_pivots(df):
    last = df.iloc[-1]
    h, l, c = last["High"], last["Low"], last["Close"]
    p = (h + l + c) / 3
    return {
        "P": p,
        "R1": 2 * p - l,
        "S1": 2 * p - h,
        "R2": p + (h - l),
        "S2": p - (h - l),
    }

# ============================================================
# МОДУЛЬ 4: VOLUME PROFILE (исправленный Value Area)
# ============================================================

def volume_profile(df, lookback=60, num_bins=40):
    data = df.tail(lookback)
    if len(data) == 0:
        return {"poc": 0, "val": 0, "vah": 0, "hvn": []}

    min_p = data["Low"].min()
    max_p = data["High"].max()
    if min_p == max_p:
        return {"poc": min_p, "val": min_p, "vah": min_p, "hvn": [min_p]}

    bins = np.linspace(min_p, max_p, num_bins + 1)
    vol_by_bin = np.zeros(num_bins)

    for _, row in data.iterrows():
        price = row["Close"]
        vol = row.get("Volume", 1)
        if vol is None or (isinstance(vol, float) and np.isnan(vol)):
            vol = 1
        for b in range(num_bins):
            if bins[b] <= price < bins[b + 1]:
                vol_by_bin[b] += vol
                break

    poc_idx = int(np.argmax(vol_by_bin))
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    total_vol = vol_by_bin.sum()
    if total_vol == 0:
        return {"poc": poc, "val": min_p, "vah": max_p, "hvn": [poc]}

    # Value Area: расширяемся от POC вверх и вниз, пока не наберём 70% объёма
    target = 0.70 * total_vol
    accumulated = vol_by_bin[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target and (lo_idx > 0 or hi_idx < num_bins - 1):
        left_vol  = vol_by_bin[lo_idx - 1] if lo_idx > 0 else 0
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

    return {"poc": poc, "val": val, "vah": vah, "hvn": hvn}

def format_levels_message(price, vp, support_lvls, resistance_lvls, pivots):
    level_score = 0
    msg = ""

    msg += f"POC: ${vp['poc']:.3f}\n"
    msg += f"Value Area: ${vp['val']:.3f} — ${vp['vah']:.3f}\n"

    nearest_sup = None
    nearest_res = None

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

    if price > vp["poc"]:
        level_score += 1
    if vp["val"] <= price <= vp["vah"]:
        pass
    elif price > vp["vah"]:
        level_score += 1
    elif price < vp["val"]:
        level_score -= 1

    if vp["hvn"]:
        hvn_str = ", ".join([f"${h:.3f}" for h in vp["hvn"]])
        msg += f"📊 HVN: {hvn_str}\n"

    return msg, level_score, nearest_sup, nearest_res

# ============================================================
# МОДУЛЬ 5: ЗАПАСЫ EIA (API v2)
# ============================================================

def get_eia_storage():
    if not EIA_API_KEY:
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

    try:
        url = "https://api.eia.gov/v2/natural-gas/stor/sum/data/"
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[process][]": "SAV",
            "facets[region][]": "US",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 2,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        records = data["response"]["data"]
        current = float(records[0]["value"])
        prev = float(records[1]["value"])
        build = current - prev
        return current, build, STORAGE_FORECAST
    except Exception as e:
        logging.error(f"EIA storage API v2 error: {e}")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

def score_storage(storage_bcf, build, forecast):
    score = 0
    msg = ""

    avg_5yr = 3000
    pct = ((storage_bcf - avg_5yr) / avg_5yr) * 100
    msg += f"Текущие: {storage_bcf:.0f} Bcf ({pct:+.1f}% к 5л ср.)\n"

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
# МОДУЛЬ 6: НОВОСТНОЙ ФОН (с фильтрацией по дате)
# ============================================================

NEWS_KEYWORDS = {
    "weather": [
        ("heatwave", 3, "bull"), ("unseasonable heat", 3, "bull"),
        ("hot weather", 2, "bull"), ("heat", 1, "bull"),
        ("cold snap", 2, "bull"), ("polar vortex", 3, "bull"),
        ("freeze", 3, "bull"), ("blizzard", 2, "bull"),
        ("storm", 1, "bull"), ("hurricane", 2, "bull"),
        ("mild weather", -2, "bear"), ("warm winter", -2, "bear"),
        ("above normal temperatures", -2, "bear"),
    ],
    "lng": [
        ("lng exports", 3, "bull"), ("export surge", 3, "bull"),
        ("lng capacity", 2, "bull"), ("higher capacity", 2, "bull"),
        ("lng lend support", 2, "bull"), ("export rose", 2, "bull"),
        ("lng outage", -3, "bear"), ("lng maintenance", -2, "bear"),
        ("export down", -2, "bear"), ("export delay", -2, "bear"),
        ("freeport outage", -3, "bear"), ("lng shutdown", -3, "bear"),
        ("pipeline restrictions", 1, "bull"),
    ],
    "production": [
        ("record production", -3, "bear"), ("rig activity", -2, "bear"),
        ("production rise", -2, "bear"), ("output increase", -2, "bear"),
        ("production cut", 2, "bull"), ("rig count decline", 2, "bull"),
        ("supply drop", 2, "bull"), ("reduced output", 1, "bull"),
    ],
    "demand": [
        ("record demand", 3, "bull"), ("demand surge", 2, "bull"),
        ("data center", 2, "bull"), ("ai demand", 2, "bull"),
        ("power generation", 1, "bull"), ("gas generation", 1, "bull"),
        ("demand fall", -2, "bear"), ("weak demand", -2, "bear"),
    ],
    "geopolitics": [
        ("sanctions", 2, "bull"), ("ukraine", 2, "bull"),
        ("hormuz", 2, "bull"), ("middle east tension", 2, "bull"),
        ("russia gas", 2, "bull"), ("trade war", -1, "bear"),
    ],
}

def parse_news():
    feeds = [
        "https://www.naturalgasintelligence.com/feed/",
        "https://www.eia.gov/todayinenergy/rss.xml",
        "https://oilprice.com/rss/home.rss",
    ]
    titles = []
    cutoff = datetime.now() - timedelta(hours=24)

    for feed in feeds:
        try:
            r = requests.get(feed, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                pub_str = item.findtext("pubDate", "")
                pub_date = None
                if pub_str:
                    try:
                        pub_date = parsedate_to_datetime(pub_str).replace(tzinfo=None)
                    except Exception:
                        pub_date = None

                if title and (pub_date is None or pub_date >= cutoff):
                    titles.append(title)
        except Exception as e:
            logging.warning(f"News feed error: {feed} — {e}")

    return titles

def score_news(titles):
    category_scores = {}
    category_msgs = {}
    top_headlines = []

    for title in titles:
        t = title.lower()
        for category, keywords in NEWS_KEYWORDS.items():
            for word, pts, direction in keywords:
                if word in t:
                    if category not in category_scores:
                        category_scores[category] = 0
                        category_msgs[category] = []
                    sign = 1 if direction == "bull" else -1
                    contribution = pts * sign
                    category_scores[category] += contribution
                    if len(category_msgs[category]) < 2:
                        emoji = "📈" if contribution > 0 else "📉"
                        category_msgs[category].append(
                            f"{emoji} \"{title[:80]}\" → {contribution:+d}"
                        )
                    if len(top_headlines) < 5:
                        top_headlines.append((title, contribution))

    for cat in category_scores:
        category_scores[cat] = max(-5, min(5, category_scores[cat]))

    total = sum(category_scores.values())
    total = max(-5, min(5, total))

    cat_names = {
        "weather": "🌤️ Погода", "lng": "🚢 LNG/Экспорт",
        "production": "⛏️ Добыча", "demand": "⚡ Спрос",
        "geopolitics": "🌍 Геополитика",
    }

    msg = f"Скоринг: {total:+d} ({'📈 бычий' if total > 0 else '📉 медвежий' if total < 0 else '➡️ нейтральный'})\n"

    for cat, score in sorted(category_scores.items(), key=lambda x: abs(x[1]), reverse=True):
        name = cat_names.get(cat, cat)
        emoji = "📈" if score > 0 else "📉"
        msg += f"{name}: {score:+d} {emoji}\n"
        for m in category_msgs[cat]:
            msg += f"  {m}\n"

    return total, msg

# ============================================================
# МОДУЛЬ 7: POWERBURN (с проверкой данных)
# ============================================================

def fetch_powerburn():
    try:
        url = "https://www.celsiusenergy.net/p/powerburn.html"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=15)
        html = r.text

        pb_match = re.search(r"(\d{1,2}\.\d)\s*BCF", html, re.I)
        current_pb = float(pb_match.group(1)) if pb_match else 0

        vs_yesterday_match = re.search(r"vs\s*yes.*?([+-]?\d{1,2}\.\d)", html, re.I)
        vs_yesterday = float(vs_yesterday_match.group(1)) if vs_yesterday_match else 0

        daily_match = re.search(r"(\d{1,2}\.\d)\s*Bc[fF]/d", html, re.I)
        daily_bcf = float(daily_match.group(1)) if daily_match else 0

        vs_yoy_match = re.search(r"vs\s*202[0-9].*?([+-]?\d{1,2}\.\d)", html, re.I)
        vs_yoy = float(vs_yoy_match.group(1)) if vs_yoy_match else 0

        ng_match = re.search(r"Natural\s*Gas\s*</th>\s*<td[^>]*>\s*(\d{1,2}\.?\d*)\s*%", html, re.I)
        if not ng_match:
            ng_match = re.search(r"natural\s*gas.*?(\d{1,2}\.?\d*)\s*%", html, re.I)
        ng_pct = float(ng_match.group(1)) if ng_match else 0

        ng_yoy_match = re.search(r"Natural\s*Gas.*?([+-]?\d{1,2}\.?\d*)\s*%",
                                 html[ng_match.end():] if ng_match else "", re.I)
        ng_yoy = float(ng_yoy_match.group(1)) if ng_yoy_match else 0

        coal_match = re.search(r"Coal\s*</th>\s*<td[^>]*>\s*(\d{1,2}\.?\d*)\s*%", html, re.I)
        coal_pct = float(coal_match.group(1)) if coal_match else 0

        return {
            "realtime_bcf": current_pb,
            "vs_yesterday": vs_yesterday,
            "daily_bcf": daily_bcf,
            "vs_yoy": vs_yoy,
            "ng_pct": ng_pct,
            "ng_yoy": ng_yoy,
            "coal_pct": coal_pct,
        }
    except Exception as e:
        logging.error(f"Powerburn fetch error: {e}")
        return {
            "realtime_bcf": 0, "vs_yesterday": 0, "daily_bcf": 0,
            "vs_yoy": 0, "ng_pct": 0, "ng_yoy": 0, "coal_pct": 0,
        }

def score_powerburn(pb):
    score = 0
    msg = ""

    if pb["realtime_bcf"] == 0 and pb["daily_bcf"] == 0:
        msg += "⚠️ Данные недоступны (парсинг не удался)\n"
        return 0, msg

    msg += f"Realtime: {pb['realtime_bcf']:.1f} BCF"
    if pb["vs_yesterday"] != 0:
        sign = "+" if pb["vs_yesterday"] > 0 else ""
        msg += f" ({sign}{pb['vs_yesterday']:.1f} к вчера)"
    msg += "\n"

    msg += f"Daily: {pb['daily_bcf']:.1f} BCF/d"
    if pb["vs_yoy"] != 0:
        sign = "+" if pb["vs_yoy"] > 0 else ""
        msg += f" ({sign}{pb['vs_yoy']:.1f} к пр.году)"
        if pb["vs_yoy"] > 2:
            score += 1
        elif pb["vs_yoy"] < -2:
            score -= 1
    msg += "\n"

    msg += f"Доля газа в генерации: {pb['ng_pct']:.1f}%"
    if pb["ng_yoy"] != 0:
        sign = "+" if pb["ng_yoy"] > 0 else ""
        msg += f" ({sign}{pb['ng_yoy']:.1f}% к пр.году)"
        if pb["ng_yoy"] > 2:
            score += 1
        elif pb["ng_yoy"] < -2:
            score -= 1
    msg += "\n"

    if pb["coal_pct"] > 0:
        msg += f"Доля угля: {pb['coal_pct']:.1f}%\n"
        if pb["coal_pct"] > 25:
            score -= 1
            msg += "⚠️ Уголь замещает газ (fuel switching)\n"

    score = max(-3, min(3, score))
    return score, msg

# ============================================================
# МОДУЛЬ 8: СКОРИНГ И СИГНАЛЫ
# ============================================================

def calculate_score(ind, level_score, storage_score, season_score,
                   news_score, pb_score):
    score = 0
    price = ind["price"]
    rsi = ind["rsi"]
    ma200 = ind["ma200"]
    bb_upper = ind["bb_upper"]
    macd_hist = ind["macd_hist"]

    if rsi > 70:
        score -= 2
    elif rsi < 30:
        score += 2
    elif rsi > 60:
        score -= 1
    elif rsi < 40:
        score += 1

    if price < ma200:
        score -= 1
    elif price > ma200:
        score += 1

    if abs(price - bb_upper) < 0.02 * price:
        score -= 1
    if abs(price - ind["bb_lower"]) < 0.02 * price:
        score += 1

    if macd_hist < 0:
        score -= 1
    elif macd_hist > 0:
        score += 1

    score += season_score
    score += storage_score
    score += level_score
    score += news_score
    score += pb_score

    return max(-15, min(15, score))

def determine_signal(score):
    if score >= 4:
        return "🟢 СИЛЬНЫЙ ЛОНГ"
    elif score >= 2:
        return "🟡 ЛОНГ"
    elif score >= 1:
        return "⚪ СЛАБЫЙ ЛОНГ"
    elif score <= -4:
        return "🔴 СИЛЬНЫЙ ШОРТ"
    elif score <= -2:
        return "🟠 ШОРТ"
    elif score <= -1:
        return "🔵 СЛАБЫЙ ШОРТ"
    else:
        return "⬜ ВНЕ ПОЗИЦИИ"

# ============================================================
# МОДУЛЬ 9: TELEGRAM (анти-спам)
# ============================================================

def send_telegram(text, is_change_alert=False):
    now = datetime.now()
    is_cme_hours = CME_START_HOUR_MSK <= now.hour < CME_END_HOUR_MSK

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[DEBUG] Telegram отключён — нет токена/chat_id")
        return False

    if not is_cme_hours and not is_change_alert:
        logging.info(f"Вне CME-часов — Telegram не отправляется: {text[:100]}")
        print("[Вне CME-часов] Сигнал залогирован, но не отправлен в Telegram")
        return False

    full_text = text
    if is_change_alert:
        full_text = "🚨 *СМЕНА СИГНАЛА* 🚨\n\n" + text

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": full_text,
            "parse_mode": "Markdown",
        }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            logging.error(f"Telegram error: {r.status_code} {r.text}")
            return False
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        return False


def load_last_signal():
    """Читает предыдущий сигнал, score, цену и время отправки."""
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            data = json.load(f)
            return (
                data.get("signal", ""),
                data.get("score", 0),
                data.get("price", 0),
                data.get("timestamp", ""),
            )
    except Exception:
        return "", 0, 0, ""


def save_last_signal(signal, score, price):
    """Сохраняет текущий сигнал, score, цену и время."""
    try:
        with open(LAST_SIGNAL_FILE, "w") as f:
            json.dump({
                "signal": signal,
                "score": score,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }, f)
    except Exception:
        pass


def should_send_signal(signal, score, price, prev_signal, prev_score,
                       prev_timestamp, is_cme):
    """
    Решает, отправлять ли сообщение в Telegram.
    Возвращает (should_send, reason).
    """
    now = datetime.now()

    # 1. Сигнал изменился → всегда отправляем (даже вне CME)
    if prev_signal and prev_signal != signal:
        return True, "change"

    # 2. Нет предыдущего сигнала (первый запуск) → отправляем
    if not prev_signal:
        return True, "first_run"

    # 3. Сигнал тот же, но score значительно изменился
    if abs(score - prev_score) >= SCORE_CHANGE_THRESHOLD:
        return True, f"score_change ({prev_score:+d} → {score:+d})"

    # 4. Сигнал и score те же — регулярный отчёт раз в REGULAR_INTERVAL_HOURS
    if is_cme and prev_timestamp:
        try:
            last_dt = datetime.fromisoformat(prev_timestamp)
            elapsed = (now - last_dt).total_seconds()
            if elapsed >= REGULAR_INTERVAL_HOURS * 3600:
                return True, "regular_interval"
        except Exception:
            return True, "regular_interval_no_timestamp"
    elif is_cme and not prev_timestamp:
        return True, "regular_no_timestamp"

    # 5. Всё то же самое, интервал не прошёл → не отправляем
    return False, "no_change"

# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

def main():
    now = datetime.now()
    logging.info(f"=== Запуск цикла {now.strftime('%Y-%m-%d %H:%M:%S')} ===")

    try:
        df = fetch_prices()
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
        ind["price"], vp, support_lvls, resistance_lvls, pivots
    )

    storage_bcf, build, forecast = get_eia_storage()
    storage_score, storage_msg = score_storage(storage_bcf, build, forecast)

    season_score = seasonality_score(now.month)

    news_titles = parse_news()
    news_score, news_msg = score_news(news_titles)

    pb_data = fetch_powerburn()
    pb_score, pb_msg = score_powerburn(pb_data)

    total_score = calculate_score(
        ind, level_score, storage_score, season_score, news_score, pb_score
    )

    signal = determine_signal(total_score)
    price = ind["price"]
    atr = ind["atr"]

    is_long  = "ЛОНГ" in signal
    is_short = "ШОРТ" in signal

    if is_long:
        sl  = price - 1.5 * atr
        tp1 = price + 2 * atr
        tp2 = price + 4 * atr
        if nearest_sup and sl > nearest_sup:
            sl = nearest_sup - 0.02
    elif is_short:
        sl  = price + 1.5 * atr
        tp1 = price - 2 * atr
        tp2 = price - 4 * atr
        if nearest_res and sl < nearest_res:
            sl = nearest_res + 0.02
    else:
        sl  = price - 1.5 * atr
        tp1 = price + 2 * atr
        tp2 = price + 4 * atr

    risk = abs(price - sl)
    reward1 = abs(tp1 - price)
    rr1 = reward1 / risk if risk > 0 else 0

    # ── Анти-спам: проверка, нужно ли отправлять ──
    prev_signal, prev_score, prev_price, prev_timestamp = load_last_signal()
    is_cme = CME_START_HOUR_MSK <= now.hour < CME_END_HOUR_MSK
    should_send, send_reason = should_send_signal(
        signal, total_score, price,
        prev_signal, prev_score, prev_timestamp, is_cme
    )

    # Логирование
    log_line = (f"{signal} | score={total_score} | price=${price:.3f} | "
                f"prev={prev_signal} score={prev_score} | "
                f"send={should_send} ({send_reason})")
    logging.info(log_line)
    print(log_line)

    # Сохраняем актуальный сигнал ДО отправки (чтобы даже при ошибке TG не зацикливаться)
    save_last_signal(signal, total_score, price)

    # Формирование сообщения
    msg = f"{signal}\n"
    msg += f"Score: {total_score}/15\n"
    msg += f"Цена: ${price:.3f}\n"

    # EIA спот для справки
    eia_spot = df["EIA_Spot"].iloc[-1] if "EIA_Spot" in df.columns else np.nan
    if not np.isnan(eia_spot):
        msg += f"EIA спот: ${eia_spot:.3f}\n"

    if is_long or is_short:
        msg += f"SL: ${sl:.3f}\n"
        msg += f"TP1: ${tp1:.3f} | TP2: ${tp2:.3f}\n"
        msg += f"R/R: {rr1:.2f}\n"

    msg += "━━━━ ИНДИКАТОРЫ ━━━━\n"
    msg += f"RSI: {ind['rsi']:.1f} | MA50: ${ind['ma50']:.3f} | MA200: ${ind['ma200']:.3f}\n"
    msg += f"ATR: ${ind['atr']:.3f}\n"

    msg += "━━━━ ЗАПАСЫ EIA ━━━━\n"
    msg += storage_msg

    msg += "━━━━ УРОВНИ ━━━━\n"
    msg += level_msg

    msg += "━━━━ POWERBURN ━━━━\n"
    msg += pb_msg

    msg += "━━━━ НОВОСТИ ━━━━\n"
    msg += news_msg

    # Отправка
    if should_send:
        is_change_alert = (send_reason == "change")
        success = send_telegram(msg, is_change_alert=is_change_alert)
        if success:
            print(f"✅ Отправлено в Telegram ({send_reason})")
        else:
            print(f"❌ Не отправлено ({send_reason})")
    else:
        print(f"⏸ Не отправлено — нет изменений ({send_reason})")

    print(f"\n--- Полное сообщение ---\n{msg}")


if __name__ == "__main__":
    import sys
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

