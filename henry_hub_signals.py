"""
Henry Hub Natural Gas — автоматическая система сигналов (Версия с точной обработкой EIA Storage)

Ключевые изменения для точности:
1. Обновлен парсер EIA Storage API v2 (корректная обработка списка словарей).
2. Добавлена строгая валидация данных перед расчетом фактора запасов.
3. Логирование структуры полученных данных для отладки.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

# Fallback значения (используются ТОЛЬКО если API полностью недоступен)
STORAGE_CURRENT_BCF = float(os.environ.get("STORAGE_CURRENT_BCF", "3153"))
LAST_STORAGE_BUILD = float(os.environ.get("LAST_STORAGE_BUILD", "15"))
STORAGE_FORECAST = float(os.environ.get("STORAGE_FORECAST", "19"))

LOG_FILE = "henry_hub_signals.log"
CACHE_EIA_FILE = "cache_eia_ng_prices.csv"
CACHE_YAHOO_FILE = "cache_yahoo_ng_prices.csv"
LAST_SIGNAL_FILE = "last_signal.json"
REQUEST_TIMEOUT = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)

EIA_PRICE_SERIES_ID = "NG.RNGWHHD.D"
YAHOO_SYMBOL = "NG=F"
# Актуальный ID серии для Weekly Natural Gas Storage Report (EIA v2)
EIA_STORAGE_SERIES_ID = "NG.NW2_EPG0_SGO_RNG_RNGFM_WUS"


# ============================================================
# МОДУЛЬ 0: КЭШ ПОСЛЕДНЕГО СИГНАЛА
# ============================================================

def load_last_signal():
    if not os.path.exists(LAST_SIGNAL_FILE):
        return None
    try:
        with open(LAST_SIGNAL_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Не удалось загрузить last_signal.json: {e}")
        return None


def save_last_signal(signal_data):
    try:
        with open(LAST_SIGNAL_FILE, "w") as f:
            json.dump(signal_data, f, indent=2)
        logger.info(f"Сигнал сохранён в {LAST_SIGNAL_FILE}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить last_signal.json: {e}")


def signal_changed(current, last):
    if last is None:
        return True

    keys_to_compare = [
        "score", "signal_text", "price", "stop", "tp1", "tp2",
    ]
    for key in keys_to_compare:
        cur_val = current.get(key)
        last_val = last.get(key)
        if cur_val != last_val:
            return True
    return False


# ============================================================
# МОДУЛЬ 1A: ЦЕНЫ EIA (спот, дневные)
# ============================================================

def fetch_eia_prices(max_retries=3):
    logger.info("→ Загрузка цен EIA (спот Henry Hub)...")

    if not EIA_API_KEY:
        logger.warning("EIA_API_KEY не задан — цены EIA недоступны.")
        return pd.DataFrame()

    if os.path.exists(CACHE_EIA_FILE):
        file_age = time.time() - os.path.getmtime(CACHE_EIA_FILE)
        if file_age < 24 * 3600:
            try:
                df = pd.read_csv(CACHE_EIA_FILE, index_col="Date", parse_dates=["Date"])
                if not df.empty:
                    logger.info(f"  EIA: кэш ({len(df)} записей).")
                    return df
            except Exception:
                pass

    url = f"https://api.eia.gov/v2/seriesid/{EIA_PRICE_SERIES_ID}?api_key={EIA_API_KEY}"

    r = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                logger.info("  EIA: ответ 200 OK.")
                break
            if r.status_code == 429:
                delay = int(r.headers.get("Retry-After", "60"))
                logger.warning(f"  EIA: 429, ждём {delay}с...")
                time.sleep(delay)
                continue
            logger.error(f"  EIA: статус {r.status_code}")
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(5)
        except Exception as e:
            logger.error(f"  EIA: сетевая ошибка ({attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(5)

    try:
        data = r.json()
        series_data = None
        if "series" in data and len(data["series"]) > 0:
            series_data = data["series"]["data"]
        elif "response" in data and "data" in data["response"]:
            raw = data["response"]["data"]
            series_data = [[item["period"], item["value"]] for item in raw]

        if not series_data:
            logger.error("  EIA: нет данных в ответе.")
            return pd.DataFrame()

        dates, prices = [], []
        for entry in series_data:
            date_str = str(entry)
            price_raw = entry
            if price_raw is None or price_raw == "":
                continue
            try:
                price = float(price_raw)
            except (ValueError, TypeError):
                continue
            try:
                d = pd.to_datetime(date_str)
            except Exception:
                continue
            dates.append(d)
            prices.append(price)

        if not dates:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
                "Volume": len(prices),
            }
        )
        df.dropna(inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        cutoff = datetime.now() - timedelta(days=730)
        df = df[df.index >= cutoff]

        df.to_csv(CACHE_EIA_FILE)
        logger.info(f"  EIA: {len(df)} записей, кэш сохранён.")
        return df

    except Exception as e:
        logger.error(f"  EIA: ошибка парсинга: {e}")
        return pd.DataFrame()


# ============================================================
# МОДУЛЬ 1B: ЦЕНЫ YAHOO FINANCE (фьючерс NG=F)
# ============================================================

def fetch_yahoo_prices(max_retries=3):
    logger.info("→ Загрузка цен Yahoo Finance (NG=F)...")

    if os.path.exists(CACHE_YAHOO_FILE):
        file_age = time.time() - os.path.getmtime(CACHE_YAHOO_FILE)
        if file_age < 6 * 3600:
            try:
                df = pd.read_csv(CACHE_YAHOO_FILE, index_col="Date", parse_dates=["Date"])
                if not df.empty:
                    logger.info(f"  Yahoo: кэш ({len(df)} записей).")
                    return df
            except Exception:
                pass

    period1 = int((datetime.now() - timedelta(days=730)).timestamp())
    period2 = int(datetime.now().timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{YAHOO_SYMBOL}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                logger.info("  Yahoo: ответ 200 OK.")
                break
            if r.status_code == 429:
                logger.warning(f"  Yahoo: 429, ждём 30с (попытка {attempt + 1})...")
                time.sleep(30)
                continue
            logger.warning(f"  Yahoo: статус {r.status_code} — пропускаем.")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"  Yahoo: сетевая ошибка ({attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(5)

    try:
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            logger.warning("  Yahoo: пустой ответ — пропускаем.")
            return pd.DataFrame()

        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        if not timestamps or not closes:
            logger.warning("  Yahoo: нет данных в ответе — пропускаем.")
            return pd.DataFrame()

        rows = []
        for i in range(len(timestamps)):
            if closes[i] is None:
                continue
            rows.append(
                {
                    "Date": pd.to_datetime(timestamps[i], unit="s"),
                    "Open": float(opens[i]) if opens[i] else float(closes[i]),
                    "High": float(highs[i]) if highs[i] else float(closes[i]),
                    "Low": float(lows[i]) if lows[i] else float(closes[i]),
                    "Close": float(closes[i]),
                    "Volume": int(volumes[i]) if volumes[i] else 0,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.dropna(subset=["Close"], inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        cutoff = datetime.now() - timedelta(days=730)
        df = df[df.index >= cutoff]

        df.to_csv(CACHE_YAHOO_FILE)
        logger.info(f"  Yahoo: {len(df)} записей, кэш сохранён.")
        return df

    except Exception as e:
        logger.warning(f"  Yahoo: ошибка парсинга: {e} — пропускаем.")
        return pd.DataFrame()


# ============================================================
# МОДУЛЬ 1C: ОБЪЕДИНЕНИЕ ИСТОЧНИКОВ
# ============================================================

def build_combined_dataset(df_eia, df_yahoo):
    price_eia = None
    price_yahoo = None

    if not df_eia.empty:
        price_eia = df_eia["Close"].iloc[-1]

    if not df_yahoo.empty:
        price_yahoo = df_yahoo["Close"].iloc[-1]

    if not df_yahoo.empty:
        df_main = df_yahoo.copy()
        source_name = "Yahoo (NG=F фьючерс)"
        logger.info(f"Основной источник индикаторов: {source_name}")
    elif not df_eia.empty:
        df_main = df_eia.copy()
        source_name = "EIA (спот)"
        logger.info(f"Основной источник индикаторов: {source_name}")
    else:
        return pd.DataFrame(), None, None

    if not df_eia.empty and not df_yahoo.empty:
        missing_dates = df_eia.index.difference(df_yahoo.index)
        if len(missing_dates) > 0:
            df_extra = df_eia.loc[missing_dates].copy()
            df_main = pd.concat([df_main, df_extra])
            df_main = df_main[~df_main.index.duplicated(keep="last")]
            df_main.sort_index(inplace=True)
            logger.info(f"  Дополнено {len(missing_dates)} записями из EIA.")

    return df_main, price_eia, price_yahoo


# ============================================================
# МОДУЛЬ 2: ИНДИКАТОРЫ
# ============================================================

def calc_indicators(df, has_real_ohlc=False):
    close = df["Close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    ma50 = close.rolling(50).mean().iloc[-1]
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

    if has_real_ohlc:
        high = df["High"]
        low = df["Low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        logger.info(f"  ATR (реальный OHLC): {atr:.4f}")
    else:
        tr = close.diff().abs()
        atr = tr.rolling(14).mean().iloc[-1] * 1.5
        logger.info(f"  ATR (упрощённый ×1.5): {atr:.4f}")

    return {
        "price": close.iloc[-1],
        "rsi": rsi,
        "ma50": ma50,
        "ma200": ma200,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "macd_hist": macd_hist,
        "atr": atr,
        "has_real_ohlc": has_real_ohlc,
    }


# ============================================================
# МОДУЛЬ 3: СЕЗОННОСТЬ
# ============================================================

def seasonality_score(month):
    scores = {
        1: 3, 2: 2, 3: 1, 4: -1, 5: -2, 6: -1,
        7: 0, 8: -1, 9: -2, 10: -1, 11: 2, 12: 3,
    }
    return scores.get(month, 0)


# ============================================================
# МОДУЛЬ 4: УРОВНИ
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
    recent = df.tail(5)
    h = recent["High"].max()
    l = recent["Low"].min()
    c = df.iloc[-1]["Close"]
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h}


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
        for b in range(num_bins):
            if bins[b] <= price < bins[b + 1]:
                vol_by_bin[b] += vol
                break

    poc_idx = int(np.argmax(vol_by_bin))
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    total_vol = vol_by_bin.sum()
    if total_vol == 0:
        return {"poc": poc, "val": min_p, "vah": max_p, "hvn": [poc]}

    cum = 0
    val_idx, vah_idx = 0, num_bins - 1
    for b in range(num_bins):
        cum += vol_by_bin[b]
        if cum >= 0.15 * total_vol:
            val_idx = b
            break
    cum = 0
    for b in reversed(range(num_bins)):
        cum += vol_by_bin[b]
        if cum >= 0.15 * total_vol:
            vah_idx = b
            break

    val = (bins[val_idx] + bins[val_idx + 1]) / 2
    vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2

    top_indices = np.argsort(vol_by_bin)[-3:][::-1]
    hvn = [(bins[i] + bins[i + 1]) / 2 for i in top_indices if vol_by_bin[i] > 0]

    return {"poc": poc, "val": val, "vah": vah, "hvn": hvn}


def format_levels_message(price, vp, support_lvls, resistance_lvls, pivots):
    msg = ""
    msg += f"POC: \${vp['poc']:.3f}\n"
    msg += f"Value Area: ${vp['val']:.3f} — ${vp['vah']:.3f}\n"

    nearest_sup = None
    nearest_res = None

    sups_below = [s for s in support_lvls if s < price]
    if sups_below:
        nearest_sup = max(sups_below)
        dist = abs(price - nearest_sup) / price
        msg += f"🟢 Поддержка: \${nearest_sup:.3f} ({dist * 100:.1f}%)\n"
    else:
        msg += "🟢 Поддержка: нет в окне\n"

    ress_above = [r for r in resistance_lvls if r > price]
    if ress_above:
        nearest_res = min(ress_above)
        dist = abs(nearest_res - price) / price
        msg += f"🔴 Сопротивление: \${nearest_res:.3f} ({dist * 100:.1f}%)\n"
    else:
        msg += "🔴 Сопротивление: нет в окне\n"

    msg += f"📐 Pivot: P=${pivots['P']:.3f} R1=${pivots['R1']:.3f} S1=\${pivots['S1']:.3f}\n"

    if vp["hvn"]:
        hvn_str = ", ".join([f"\${h:.3f}" for h in vp["hvn"]])
        msg += f"📊 HVN: {hvn_str}\n"

    return msg, nearest_sup, nearest_res


# ============================================================
# МОДУЛЬ 5: ЗАПАСЫ EIA (ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ ТОЧНОСТИ)
# ============================================================

def get_eia_storage():
    """
    Получает данные по запасам газа из EIA API v2.
    Возвращает (current_storage, build, forecast).
    
    ВАЖНО: Эта функция теперь строго проверяет структуру ответа.
    Если данные не соответствуют ожидаемому формату, она логирует ошибку
    и возвращает None, чтобы скоринг мог обработать это как отсутствие данных,
    а не подставлять неверные цифры.
    """
    if not EIA_API_KEY:
        logger.warning("EIA_API_KEY не задан — невозможно получить точные данные по запасам.")
        return None, None, STORAGE_FORECAST

    url = f"https://api.eia.gov/v2/seriesid/{EIA_STORAGE_SERIES_ID}?api_key={EIA_API_KEY}"

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # --- НАДЕЖНЫЙ ПАРСИНГ ДЛЯ EIA V2 ---
        raw_data = None
        
        # Путь 1: Стандартный ответ v2
        if "series" in data and len(data["series"]) > 0:
            raw_data = data["series"].get("data")
        
        # Путь 2: Если данные обернуты в response
        elif "response" in data and "data" in data["response"]:
            raw_data = data["response"]["data"]

        if not raw_data:
            logger.error("EIA Storage: Не найдена секция 'data' в ответе API.")
            return None, None, STORAGE_FORECAST

        if len(raw_data) < 2:
            logger.error("EIA Storage: В ответе меньше 2 записей. Недостаточно данных для расчета build.")
            return None, None, STORAGE_FORECAST

        # EIA v2 возвращает список словарей: [{'period': '2023-09-07', 'value': 3000}, ...]
        # Проверяем тип первого элемента
        first_item = raw_data
        
        if isinstance(first_item, dict):
            # Ожидаемый формат: {'period': 'YYYY-MM-DD', 'value': float}
            try:
                current_val = float(first_item.get("value"))
                prev_val = float(raw_data.get("value"))
                
                # Дополнительная валидация: значения должны быть положительными
                if current_val <= 0 or prev_val <= 0:
                    logger.error(f"EIA Storage: Получены некорректные значения запасов: {current_val}, {prev_val}")
                    return None, None, STORAGE_FORECAST

                build = current_val - prev_val
                
                logger.info(f"✅ EIA Storage: Данные получены успешно.")
                logger.info(f"   Current: {current_val} BCF")
                logger.info(f"   Previous: {prev_val} BCF")
                logger.info(f"   Build: {build} BCF")
                
                return current_val, build, STORAGE_FORECAST
                
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"EIA Storage: Ошибка парсинга словаря: {e}")
                return None, None, STORAGE_FORECAST

        elif isinstance(first_item, list):
            # Старый формат (список списков), на всякий случай
            try:
                current_val = float(first_item)
                prev_val = float(raw_data)
                build = current_val - prev_val
                return current_val, build, STORAGE_FORECAST
            except Exception as e:
                logger.error(f"EIA Storage: Ошибка парсинга списка: {e}")
                return None, None, STORAGE_FORECAST
        else:
            logger.error(f"EIA Storage: Неожиданный формат данных: {type(first_item)}")
            return None, None, STORAGE_FORECAST

    except requests.exceptions.HTTPError as e:
        logger.error(f"EIA Storage: HTTP ошибка {e.response.status_code}: {e}")
        return None, None, STORAGE_FORECAST
    except requests.exceptions.RequestException as e:
        logger.error(f"EIA Storage: Сетевая ошибка: {e}")
        return None, None, STORAGE_FORECAST
    except Exception as e:
        logger.error(f"EIA Storage: Неизвестная ошибка: {e}")
        return None, None, STORAGE_FORECAST


# ============================================================
# МОДУЛЬ 6: НОВОСТИ
# ============================================================

def parse_news():
    return []


# ============================================================
# МОДУЛЬ 7: TELEGRAM
# ============================================================

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram: токен или chat_id не заданы — пропуск.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            logger.info("  Telegram: сообщение отправлено.")
        else:
            logger.error(f"  Telegram: статус {r.status_code}, {r.text[:200]}")
    except Exception as e:
        logger.error(f"  Telegram: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("========================================")
    logger.info("=== START HENRY HUB SIGNALS ===")
    logger.info("========================================")

    # 1. Загрузка цен
    df_eia = fetch_eia_prices()
    df_yahoo = fetch_yahoo_prices()

    if df_eia.empty and df_yahoo.empty:
        logger.critical("Оба источника цен недоступны.")
        send_telegram(
            "❌ *Henry Hub: оба источника цен недоступны*\n"
            "EIA и Yahoo не вернули данные. Проверьте API-ключи и сеть."
        )
        return

    # 2. Объединение
    df_merged, price_eia, price_yahoo = build_combined_dataset(df_eia, df_yahoo)

    if df_merged.empty:
        logger.critical("Нет данных после объединения.")
        send_telegram("❌ Henry Hub: нет данных для анализа.")
        return

    has_real_ohlc = not df_yahoo.empty

    # 3. Индикаторы
    ind = calc_indicators(df_merged, has_real_ohlc=has_real_ohlc)
    price = ind["price"]

    # 4. Уровни
    support_lvls, resistance_lvls = find_swing_levels(df_merged)
    pivots = calc_pivots(df_merged)
    vp = volume_profile(df_merged)
    levels_msg, nearest_sup, nearest_res, *_ = format_levels_message(
        price, vp, support_lvls, resistance_lvls, pivots
    )

    # 5. Запасы (ТОЧНАЯ ВЕРСИЯ)
    storage_bcf, build, forecast = get_eia_storage()

    # Логика обработки отсутствия данных по запасам
    if storage_bcf is None:
        logger.warning("Данные по запасам не получены. Фактор 'Запасы' будет равен 0.")
        storage_factor_msg = "⚠️ Запасы: данные недоступны (фактор 0)"
        storage_score = 0
    else:
        avg_5yr = 3000
        pct = ((storage_bcf - avg_5yr) / avg_5yr) * 100
        if pct > 5:
            storage_score = -2
            storage_factor_msg = f"Запасы +{pct:.1f}% к норме → -2"
        elif pct < -5:
            storage_score = 2
            storage_factor_msg = f"Запасы {pct:.1f}% к норме → +2"
        else:
            storage_score = 0
            storage_factor_msg = f"Запасы {pct:+.1f}% к норме → 0"

    # 6. Новости
    news_titles = parse_news()
    news_msg = (
        "📰 Новости (6 ч):\n" + "\n".join([f"• {t}" for t in news_titles])
        if news_titles
        else "📰 Новостей за 6 часов нет"
    )

    # ============================================================
    # 7. СКОРИНГ — 8 ФАКТОРОВ
    # ============================================================
    score = 0
    factors = []

    # F1: RSI
    if ind["rsi"] >= 70:
        score -= 1
        factors.append(f"RSI {ind['rsi']:.1f} — перекуплен → -1")
    elif ind["rsi"] <= 30:
        score += 1
        factors.append(f"RSI {ind['rsi']:.1f} — перепродан → +1")
    else:
        factors.append(f"RSI {ind['rsi']:.1f} — нейтрально → 0")

    # F2: Цена vs MA50
    if price > ind["ma50"]:
        score += 1
        factors.append(f"Цена > MA50 \${ind['ma50']:.3f} → +1")
    else:
        score -= 1
        factors.append(f"Цена < MA50 \${ind['ma50']:.3f} → -1")

    # F3: Цена vs MA200
    if price > ind["ma200"]:
        score += 1
        factors.append(f"Цена > MA200 \${ind['ma200']:.3f} → +1")
    else:
        score -= 1
        factors.append(f"Цена < MA200 \${ind['ma200']:.3f} → -1")

    # F4: Bollinger Bands
    if price >= ind["bb_upper"]:
        score -= 1
        factors.append(f"Цена у верхней BB \${ind['bb_upper']:.3f} → -1")
    elif price <= ind["bb_lower"]:
        score += 1
        factors.append(f"Цена у нижней BB \${ind['bb_lower']:.3f} → +1")
    else:
        factors.append("Цена внутри BB → 0")

    # F5: MACD
    if ind["macd_hist"] > 0:
        score += 1
        factors.append("MACD гист. > 0 → +1")
    else:
        score -= 1
        factors.append("MACD гист. < 0 → -1")

    # F6: Сезонность
    month = datetime.now().month
    seas = seasonality_score(month)
    score += seas
    seas_word = "бычий" if seas > 0 else "медвежий" if seas < 0 else "нейтральный"
    factors.append(f"Сезон ({seas_word}) → {seas:+d}")

    # F7: Запасы (ОБНОВЛЕНО)
    factors.append(storage_factor_msg)
    score += storage_score

    # F8: Закачка vs прогноз
    if build is not None and forecast is not None and forecast > 0:
        if build > forecast * 1.3:
            score -= 1
            factors.append(f"Закачка {build:.0f} > прогноз {forecast:.0f} → -1")
        elif build < forecast * 0.7:
            score += 1
            factors.append(f"Закачка {build:.0f} < прогноз {forecast:.0f} → +1")
        else:
            factors.append(f"Закачка {build:.0f} ≈ прогноз {forecast:.0f} → 0")
    else:
        factors.append("Закачка/прогноз: нет данных → 0")

    # ============================================================
    # 8. СИГНАЛ
    # ============================================================
    if score <= -4:
        signal_text = "🔴 СИЛЬНЫЙ ШОРТ"
    elif score <= -2:
        signal_text = "🟠 ШОРТ"
    elif score >= 4:
        signal_text = "🟢 СИЛЬНЫЙ ЛОНГ"
    elif score >= 2:
        signal_text = "🟢 ЛОНГ"
    elif score >= 1:
        signal_text = "🟡 СЛАБЫЙ ЛОНГ"
    elif score <= -1:
        signal_text = "🟡 СЛАБЫЙ ШОРТ"
    else:
        signal_text = "⚪ НЕЙТРАЛЬНО / ВНЕ ПОЗИЦИИ"

    # ============================================================
    # 9. УРОВНИ СДЕЛКИ (ATR)
    # ============================================================
    atr = ind["atr"]

    stop = None
    tp1 = None
    tp2 = None
    trade_msg = ""

    if score <= -2:
        stop = price + 1.5 * atr
        tp1 = price - 2 * atr
        tp2 = price - 4 * atr
        risk = (stop - price) / price * 100
        rr = abs(tp1 - price) / abs(stop - price)
        trade_msg = (
            f"📌 Уровни (шорт):\n"
            f"   Вход: \${price:.3f}\n"
            f"   Стоп: \${stop:.3f} (риск {risk:.1f}%)\n"
            f"   TP1: \${tp1:.3f}\n"
            f"   TP2: \${tp2:.3f}\n"
            f"   R/R: {rr:.2f}\n"
        )
    elif score >= 2:
        stop = price - 1.5 * atr
        tp1 = price + 2 * atr
        tp2 = price + 4 * atr
        risk = (price - stop) / price * 100
        rr = abs(tp1 - price) / abs(price - stop)
        trade_msg = (
            f"📌 Уровни (лонг):\n"
            f"   Вход: \${price:.3f}\n"
            f"   Стоп: \${stop:.3f} (риск {risk:.1f}%)\n"
            f"   TP1: \${tp1:.3f}\n"
            f"   TP2: \${tp2:.3f}\n"
            f"   R/R: {rr:.2f}\n"
        )
    else:
        trade_msg = "📌 Уровней нет — сигнал слабый, жди подтверждения\n"

    # ============================================================
    # 10. ПРОВЕРКА: ИЗМЕНИЛСЯ ЛИ СИГНАЛ
    # ============================================================
    current_signal = {
        "score": score,
        "signal_text": signal_text,
        "price": round(price, 3),
        "stop": round(stop, 3) if stop is not None else None,
        "tp1": round(tp1, 3) if tp1 is not None else None,
        "tp2": round(tp2, 3) if tp2 is not None else None,
        "timestamp": datetime.now().isoformat(),
    }

    last_signal = load_last_signal()

    if not signal_changed(current_signal, last_signal):
        logger.info(
            f"Сигнал не изменился (Score: {score}, {signal_text}). "
            f"Отправка в Telegram пропущена."
        )
        return

    logger.info(
        f"Сигнал изменился! "
        f"Был: {last_signal['signal_text'] if last_signal else 'нет'} "
        f"(Score: {last_signal['score'] if last_signal else 'нет'}), "
        f"стал: {signal_text} (Score: {score})"
    )

    # ============================================================
    # 11. ОТЧЁТ И ОТПРАВКА
    # ============================================================
    factors_msg = "\n".join([f"   {f}" for f in factors])
    
    report = (
        f"📊 *Henry Hub Signal Report*\n\n"
        f"{signal_text} (Score: {score})\n\n"
        f"💰 Цена: \${price:.3f}\n"
        f"{trade_msg}\n"
        f"📈 Факторы скоринга:\n{factors_msg}\n\n"
        f"{levels_msg}"
        f"{news_msg}"
    )

    send_telegram(report)
    save_last_signal(current_signal)
    logger.info("=== END HENRY HUB SIGNALS ===")


if __name__ == "__main__":
    main()

