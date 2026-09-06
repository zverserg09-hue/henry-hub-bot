"""
Henry Hub Natural Gas — автоматическая система сигналов
Анализ: техника + уровни + Volume Profile + запасы EIA + новости + Powerburn
Источники цен: Yahoo Finance (NG=F фьючерс) + EIA API v2 (Henry Hub спот, RNGWHHD)
Режимы: --once (один прогон) | --loop (цикл) | --test (без Telegram)

ИСПРАВЛЕНО:
1. requests кодирует [] в %5B/%5D — теперь используется PreparedRequest с ручным URL
2. Date mismatch между Yahoo (21:00) и EIA (00:00) — теперь .normalize() перед join
"""

import os
import re
import sys
import time
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

MSK = ZoneInfo("Europe/Moscow")

CME_START_HOUR_MSK = 9
CME_END_HOUR_MSK   = 23

REGULAR_INTERVAL_HOURS = 4
SCORE_CHANGE_THRESHOLD  = 3

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

# ── КЛЮЧЕВОЕ: отправка запроса к EIA с сохранением [] в URL ──
def eia_get(url, timeout=15):
    """
    requests кодирует [ ] в %5B %5D даже в готовой строке URL.
    EIA API не декодирует их обратно → facets не работают.
    Решение: PreparedRequest.prepare() → подменяем url на исходный.
    """
    req = requests.Request("GET", url, headers={"User-Agent": "Mozilla/5.0"})
    prepared = req.prepare()
    prepared.url = url  # ← подмена закодированного URL на исходный со скобками
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
        df.index = df.index.normalize()  # ← нормализация дат (убираем время)
        logging.info(f"Yahoo: получено {len(df)} свечей")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки цен Yahoo: {e}")
        return None

def fetch_prices_eia():
    """
    EIA API v2 — Henry Hub спот (RNGWHHD).
    Маршрут: natural-gas/pri/fut/data/
    Facet: facets[series][]=RNGWHHD
    URL собирается вручную, отправляется через eia_get() для сохранения скобок.
    """
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
        df.index = df.index.normalize()  # ← нормализация дат

        # У EIA спот-цены только Close — синтетически достраиваем OHLC
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

    # Добавляем справочную колонку со спот-ценой EIA
    if df_eia is not None:
        eia_close = df_eia["Close"].rename("EIA_Spot")
        # Оба индекса уже нормализованы (без времени) → join сработает
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
# МОДУЛЬ 2: ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================

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

    return {
        "price": close.iloc[-1], "rsi": rsi, "ma50": ma50, "ma200": ma200,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
        "macd_hist": macd_hist, "atr": atr,
    }

def seasonality_score(month):
    scores = {
        1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -1,
        7: 0, 8: -1, 9: -2, 10: -1, 11: 2, 12: 3,
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
# МОДУЛЬ 4: VOLUME PROFILE
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
    return {"poc": poc, "val": val, "vah": vah, "hvn": hvn}

def format_levels_message(price, vp, support_lvls, resistance_lvls, pivots):
    level_score = 0
    msg = f"POC: ${vp['poc']:.3f}\n"
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
        msg += f"📊 HVN: {', '.join([f'${h:.3f}' for h in vp['hvn']])}\n"

    return msg, level_score, nearest_sup, nearest_res

# ============================================================
# МОДУЛЬ 5: ЗАПАСЫ EIA (stor/wkly, eia_get для сохранения скобок)
# ============================================================

def get_eia_storage():
    """
    EIA API v2 — недельные запасы природного газа.
    Маршрут: natural-gas/stor/wkly/data/
    """
    if not EIA_API_KEY:
        logging.info("EIA API key не задан — fallback-значения запасов")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

    # Каскад: пробуем разные комбинации facets
    attempts = [
        {"facets": "facets[duoarea][]=NUS&facets[process][]=SAV", "label": "duoarea=NUS+process=SAV"},
        {"facets": "facets[process][]=SAV", "label": "process=SAV"},
        {"facets": "facets[duoarea][]=NUS", "label": "duoarea=NUS"},
        {"facets": "facets[region][]=US", "label": "region=US"},
        {"facets": "", "label": "без facets"},
    ]

    base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    for attempt in attempts:
        try:
            url = (
                f"{base_url}"
                f"?api_key={EIA_API_KEY}"
                f"&frequency=weekly"
                f"&data[0]=value"
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

            # Ищем US total
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

def score_storage(storage_bcf, build, forecast):
    score = 0
    pct = (storage_bcf - 3000) / 3000 * 100
    msg = f"Текущие: {storage_bcf:.0f} Bcf ({pct:+.1f}% к 5л ср.)\n"
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
# МОДУЛЬ 6: НОВОСТИ
# ============================================================

NEWS_KEYWORDS = {
    "weather": [("heatwave",3,"bull"),("unseasonable heat",3,"bull"),("hot weather",2,"bull"),("heat",1,"bull"),("cold snap",2,"bull"),("polar vortex",3,"bull"),("freeze",3,"bull"),("blizzard",2,"bull"),("storm",1,"bull"),("hurricane",2,"bull"),("mild weather",-2,"bear"),("warm winter",-2,"bear"),("above normal temperatures",-2,"bear")],
    "lng": [("lng exports",3,"bull"),("export surge",3,"bull"),("lng capacity",2,"bull"),("higher capacity",2,"bull"),("lng lend support",2,"bull"),("export rose",2,"bull"),("lng outage",-3,"bear"),("lng maintenance",-2,"bear"),("export down",-2,"bear"),("export delay",-2,"bear"),("freeport outage",-3,"bear"),("lng shutdown",-3,"bear"),("pipeline restrictions",1,"bull")],
    "production": [("record production",-3,"bear"),("rig activity",-2,"bear"),("production rise",-2,"bear"),("output increase",-2,"bear"),("production cut",2,"bull"),("rig count decline",2,"bull"),("supply drop",2,"bull"),("reduced output",1,"bull")],
    "demand": [("record demand",3,"bull"),("demand surge",2,"bull"),("data center",2,"bull"),("ai demand",2,"bull"),("power generation",1,"bull"),("gas generation",1,"bull"),("demand fall",-2,"bear"),("weak demand",-2,"bear")],
    "geopolitics": [("sanctions",2,"bull"),("ukraine",2,"bull"),("hormuz",2,"bull"),("middle east tension",2,"bull"),("russia gas",2,"bull"),("trade war",-1,"bear")],
}

def parse_news():
    feeds = ["https://www.naturalgasintelligence.com/feed/", "https://www.eia.gov/todayinenergy/rss.xml", "https://oilprice.com/rss/home.rss"]
    titles = []
    cutoff = datetime.now() - timedelta(hours=24)
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
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
                        category_msgs[category].append(f'{emoji} "{title[:80]}" → {contribution:+d}')
    for cat in category_scores:
        category_scores[cat] = max(-5, min(5, category_scores[cat]))
    total = max(-5, min(5, sum(category_scores.values())))
    cat_names = {"weather": "🌤️ Погода", "lng": "🚢 LNG/Экспорт", "production": "⛏️ Добыча", "demand": "⚡ Спрос", "geopolitics": "🌍 Геополитика"}
    msg = f"Скоринг: {total:+d} ({'📈 бычий' if total > 0 else '📉 медвежий' if total < 0 else '➡️ нейтральный'})\n"
    for cat, score in sorted(category_scores.items(), key=lambda x: abs(x[1]), reverse=True):
        msg += f"{cat_names.get(cat, cat)}: {score:+d} {'📈' if score > 0 else '📉'}\n"
        for m in category_msgs[cat]:
            msg += f"  {m}\n"
    return total, msg

# ============================================================
# МОДУЛЬ 7: POWERBURN
# ============================================================

def fetch_powerburn():
    try:
        url = "https://www.celsiusenergy.net/p/powerburn.html"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=15)
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
        ng_yoy_match = re.search(r"Natural\s*Gas.*?([+-]?\d{1,2}\.?\d*)\s*%", html[ng_match.end():] if ng_match else "", re.I)
        ng_yoy = float(ng_yoy_match.group(1)) if ng_yoy_match else 0
        coal_match = re.search(r"Coal\s*</th>\s*<td[^>]*>\s*(\d{1,2}\.?\d*)\s*%", html, re.I)
        coal_pct = float(coal_match.group(1)) if coal_match else 0
        return {"realtime_bcf": current_pb, "vs_yesterday": vs_yesterday, "daily_bcf": daily_bcf, "vs_yoy": vs_yoy, "ng_pct": ng_pct, "ng_yoy": ng_yoy, "coal_pct": coal_pct}
    except Exception as e:
        logging.error(f"Powerburn fetch error: {e}")
        return {"realtime_bcf": 0, "vs_yesterday": 0, "daily_bcf": 0, "vs_yoy": 0, "ng_pct": 0, "ng_yoy": 0, "coal_pct": 0}

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
# МОДУЛЬ 8: СКОРИНГ
# ============================================================

def calculate_score(ind, level_score, storage_score, season_score, news_score, pb_score):
    score = 0
    price, rsi, ma200 = ind["price"], ind["rsi"], ind["ma200"]
    if rsi > 70: score -= 2
    elif rsi < 30: score += 2
    elif rsi > 60: score -= 1
    elif rsi < 40: score += 1
    if price < ma200: score -= 1
    elif price > ma200: score += 1
    if abs(price - ind["bb_upper"]) < 0.02 * price: score -= 1
    if abs(price - ind["bb_lower"]) < 0.02 * price: score += 1
    if ind["macd_hist"] < 0: score -= 1
    elif ind["macd_hist"] > 0: score += 1
    score += season_score + storage_score + level_score + news_score + pb_score
    return max(-15, min(15, score))

def determine_signal(score):
    if score >= 4: return "🟢 СИЛЬНЫЙ ЛОНГ"
    elif score >= 2: return "🟡 ЛОНГ"
    elif score >= 1: return "⚪ СЛАБЫЙ ЛОНГ"
    elif score <= -4: return "🔴 СИЛЬНЫЙ ШОРТ"
    elif score <= -2: return "🟠 ШОРТ"
    elif score <= -1: return "🔵 СЛАБЫЙ ШОРТ"
    else: return "⬜ ВНЕ ПОЗИЦИИ"

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
    full_text = ("🚨 *СМЕНА СИГНАЛА* 🚨\n\n" + text) if is_change_alert else text
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": full_text, "parse_mode": "Markdown"}, timeout=10)
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
            json.dump({"signal": signal, "score": score, "price": price, "timestamp": datetime.now().isoformat()}, f)
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
    storage_score, storage_msg = score_storage(storage_bcf, build, forecast)

    season_score = seasonality_score(now.month)
    news_titles = parse_news()
    news_score, news_msg = score_news(news_titles)
    pb_data = fetch_powerburn()
    pb_score, pb_msg = score_powerburn(pb_data)

    total_score = calculate_score(ind, level_score, storage_score, season_score, news_score, pb_score)
    signal = determine_signal(total_score)
    price, atr = ind["price"], ind["atr"]

    is_long = "ЛОНГ" in signal
    is_short = "ШОРТ" in signal

    if is_long:
        sl, tp1, tp2 = price - 1.5 * atr, price + 2 * atr, price + 4 * atr
        if nearest_sup and sl > nearest_sup: sl = nearest_sup - 0.02
    elif is_short:
        sl, tp1, tp2 = price + 1.5 * atr, price - 2 * atr, price - 4 * atr
        if nearest_res and sl < nearest_res: sl = nearest_res + 0.02
    else:
        sl, tp1, tp2 = price - 1.5 * atr, price + 2 * atr, price + 4 * atr

    risk = abs(price - sl)
    rr1 = abs(tp1 - price) / risk if risk > 0 else 0

    prev_signal, prev_score, prev_price, prev_timestamp = load_last_signal()
    is_cme = CME_START_HOUR_MSK <= now.hour < CME_END_HOUR_MSK
    should_send, send_reason = should_send_signal(signal, total_score, price, prev_signal, prev_score, prev_timestamp, is_cme)

    log_line = f"{signal} | score={total_score} | price=${price:.3f} | prev={prev_signal} score={prev_score} | send={should_send} ({send_reason})"
    logging.info(log_line)
    print(log_line)

    save_last_signal(signal, total_score, price)

    msg = f"{signal}\nScore: {total_score}/15\nЦена: ${price:.3f}\n"
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
    msg += f"ATR: ${ind['atr']:.3f}\n"
    msg += "━━━━ ЗАПАСЫ EIA ━━━━\n" + storage_msg
    msg += "━━━━ УРОВНИ ━━━━\n" + level_msg
    msg += "━━━━ POWERBURN ━━━━\n" + pb_msg
    msg += "━━━━ НОВОСТИ ━━━━\n" + news_msg

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

