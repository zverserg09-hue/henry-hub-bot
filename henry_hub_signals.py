"""
Henry Hub Natural Gas — автоматическая система сигналов
Версия с жёсткими таймаутами, кэшированием и обработкой 429
"""

import os
import time
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

STORAGE_CURRENT_BCF = float(os.environ.get("STORAGE_CURRENT_BCF", "3153"))
LAST_STORAGE_BUILD = float(os.environ.get("LAST_STORAGE_BUILD", "15"))
STORAGE_FORECAST = float(os.environ.get("STORAGE_FORECAST", "19"))

SYMBOL = "NG=F"
LOG_FILE = "henry_hub_signals.log"

CME_START_HOUR_MSK = 16
CME_END_HOUR_MSK = 23
REGULAR_INTERVAL_HOURS = 4

LAST_SIGNAL_FILE = "last_signal.json"
REQUEST_TIMEOUT_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# МОДУЛЬ 1: ЦЕНОВЫЕ ДАННЫЕ (с кэшем и обработкой 429)
# ============================================================

CACHE_FILE = "cache_prices_ng.csv"


def fetch_prices(symbol=SYMBOL, days=365 * 2, max_retries=3):
    """
    Загружает цены через JSON-эндпоинт Yahoo с обработкой 429 и кэшированием.
    Возвращает DataFrame с индексом Date.
    """
    logger.info(f"Начинаем загрузку цен для {symbol}")

    # 1. Проверяем кэш
    if os.path.exists(CACHE_FILE):
        file_age = time.time() - os.path.getmtime(CACHE_FILE)
        if file_age < 24 * 3600:
            try:
                df = pd.read_csv(CACHE_FILE, index_col="Date", parse_dates=["Date"])
                logger.info("Данные загружены из кэша (свежие).")
                return df
            except Exception as e:
                logger.warning(f"Не удалось прочитать кэш: {e}. Запрашиваем заново.")

    # 2. Запрос к Yahoo с повторами при 429
    now = int(time.time())
    period1 = now - days * 86400
    period2 = now

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )

    r = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            if r.status_code == 200:
                logger.info("Цены получены через основной эндпоинт.")
                break

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                delay = 60
                if retry_after:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        delay = 60
                logger.warning(
                    f"429 от Yahoo. Ждём {delay} сек (попытка {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue

            logger.error(
                f"Ошибка Yahoo API: статус {r.status_code}, ответ: {r.text[:200]}"
            )
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(5)

        except Exception as e:
            logger.error(f"Сетевая ошибка (попытка {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(5)

    # 3. Парсим JSON
    try:
        data = r.json()
        chart = data.get("chart", {})
        result = chart.get("result", [])

        if not result:
            logger.error("В ответе Yahoo нет данных (result пуст).")
            return pd.DataFrame()

        res = result[0]
        timestamps = res.get("timestamp", [])
        quotes = res.get("indicators", {}).get("quote", [{}])[0]

        if len(timestamps) == 0:
            logger.error("Нет временных меток в ответе Yahoo.")
            return pd.DataFrame()

        def safe_get(key):
            arr = quotes.get(key, [])
            return arr[: len(timestamps)]

        df = pd.DataFrame(
            {
                "Date": [datetime.fromtimestamp(t) for t in timestamps],
                "Open": safe_get("open"),
                "High": safe_get("high"),
                "Low": safe_get("low"),
                "Close": safe_get("close"),
                "Volume": safe_get("volume"),
            }
        )
        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)

        df.to_csv(CACHE_FILE, index=True)
        logger.info("Данные получены от Yahoo и сохранены в кэш.")
        return df

    except Exception as e:
        logger.error(f"Ошибка парсинга JSON от Yahoo: {e}")
        return pd.DataFrame()


# ============================================================
# МОДУЛЬ 2: ИНДИКАТОРЫ
# ============================================================


def calc_indicators(df):
    logger.info("Расчёт технических индикаторов")
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
        msg += f"🟢 Поддержка: ${nearest_sup:.3f} ({dist * 100:.1f}%)\n"
        if dist < 0.015:
            level_score += 1
    else:
        msg += "🟢 Поддержка: нет в окне\n"

    ress_above = [r for r in resistance_lvls if r > price]
    if ress_above:
        nearest_res = min(ress_above)
        dist = abs(nearest_res - price) / price
        msg += f"🔴 Сопротивление: ${nearest_res:.3f} ({dist * 100:.1f}%)\n"
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
# МОДУЛЬ 3: ДАННЫЕ EIA
# ============================================================


def get_eia_storage():
    if not EIA_API_KEY:
        logger.warning("EIA API KEY не задан, используем fallback-значения")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

    try:
        url = (
            f"https://api.eia.gov/v2/seriesid/NG.NW2_EPG0_SGO_RNG_RNGFM_WUS"
            f"?api_key={EIA_API_KEY}"
        )
        r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        values = data["series"][0]["data"]
        current = float(values[0][1])
        prev = float(values[1][1])
        build = current - prev
        logger.info(f"EIA Storage: current={current}, build={build}")
        return current, build, STORAGE_FORECAST
    except Exception as e:
        logger.error(f"Ошибка EIA API: {e}")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST


def score_storage(storage_bcf, build, forecast):
    """
    Считает скоринг по запасам газа (от -3 до +3) и формирует пояснительное сообщение.
    Логика: отклонение от 5-летней средней + сюрприз по закачке относительно прогноза.
    """
    score = 0
    avg_5yr = 3000  # упрощённая 5-летняя средняя (Bcf)
    pct = ((storage_bcf - avg_5yr) / avg_5yr) * 100
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
            msg += f"⚠️ Закачка {build:.0f} Bcf > прогноз {forecast:.0f} Bcf (+30%)\n"
        elif build < forecast * 0.7:
            score += 1
            msg += f"✅ Закачка {build:.0f} Bcf < прогноз {forecast:.0f} Bcf (-30%)\n"
        else:
            msg += f"⚖️ Закачка {build:.0f} Bcf ≈ прогноз {forecast:.0f} Bcf\n"

    return score, msg


# ============================================================
# МОДУЛЬ 4: НОВОСТИ
# ============================================================


def parse_news():
    """
    Парсит RSS-ленты на свежие новости по газу/энергии за последние 6 часов.
    Возвращает список заголовков (не более 5 штук).
    """
    feeds = [
        "https://www.naturalgasintelligence.com/feed/",
        "https://www.eia.gov/todayinenergy/rss.xml",
        "https://oilprice.com/rss/home.rss",
    ]
    titles = []
    cutoff = datetime.now() - timedelta(hours=6)

    for feed in feeds:
        try:
            r = requests.get(
                feed,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0 (compatible; HenryHubBot/1.0)"},
            )
            r.raise_for_status()
            root = ET.fromstring(r.content)

            items = root.findall(".//item")
            if not items:
                items = root.findall(".//channel/item")

            for item in items:
                title = item.findtext("title", "")
                pub_str = item.findtext("pubDate", "")

                if not title:
                    continue

                pub_dt = None
                if pub_str:
                    for fmt in [
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%d %H:%M:%S",
                    ]:
                        try:
                            pub_dt = datetime.strptime(pub_str, fmt)
                            break
                        except ValueError:
                            continue

                if pub_dt is None or pub_dt >= cutoff:
                    titles.append(title)
                    if len(titles) >= 5:
                        return titles

        except Exception as e:
            logger.warning(f"Ошибка парсинга ленты {feed}: {e}")
            continue

    return titles


# ============================================================
# МОДУЛЬ 5: TELEGRAM
# ============================================================


def send_telegram(message, max_retries=3):
    """Отправляет сообщение в Telegram с обработкой 429."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram credentials missing. Skipping send.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)

            if r.status_code == 200:
                logger.info(f"Telegram message sent (status: {r.status_code})")
                return

            if r.status_code == 429:
                data = r.json()
                retry_after = data.get("parameters", {}).get("retry_after", 60)
                logger.warning(
                    f"Telegram 429. Ждём {retry_after} сек (попытка {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_after)
                continue

            logger.error(
                f"Telegram API error: статус {r.status_code}, ответ: {r.text[:200]}"
            )
            return

        except Exception as e:
            logger.error(f"Failed to send Telegram message (попытка {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return
            time.sleep(5)


# ============================================================
# МОДУЛЬ 6: MAIN
# ============================================================


def main():
    logger.info("=== START HENRY HUB SIGNALS ===")

    # 1. Цены
    try:
        df = fetch_prices()
        if df.empty:
            logger.critical("Не удалось загрузить цены (пустой DataFrame).")
            send_telegram(
                "❌ Henry Hub: ошибка загрузки цен\nYahoo вернул пустые данные или 429."
            )
            return
        logger.info(f"Loaded {len(df)} price bars")
    except Exception as e:
        logger.critical(f"Cannot load prices: {e}")
        send_telegram(f"❌ Henry Hub: ошибка загрузки цен\n{e}")
        return

    # 2. Индикаторы
    ind = calc_indicators(df)
    logger.info(
        f"Indicators calculated: RSI={ind['rsi']:.2f}, Price=${ind['price']:.3f}"
    )

    # 3. Уровни
    support_lvls, resistance_lvls = find_swing_levels(df)
    pivots = calc_pivots(df)
    vp = volume_profile(df)

    levels_msg, level_score, nearest_sup, nearest_res = format_levels_message(
        ind["price"], vp, support_lvls, resistance_lvls, pivots
    )

    # 4. Запасы EIA
    storage_bcf, build, forecast = get_eia_storage()
    storage_score, storage_msg = score_storage(storage_bcf, build, forecast)

    # 5. Новости
    news_titles = parse_news()
    news_msg = (
        "📰 Новости (последние 6 ч):\n" + "\n".join([f"• {t}" for t in news_titles])
        if news_titles
        else "📰 Новостей за 6 часов нет"
    )

    # 6. Итоговый скоринг и сообщение
    total_score = level_score + storage_score
    score_emoji = "🟢" if total_score > 0 else "🔴" if total_score < 0 else "⚪"

    report = (
        f"{score_emoji} **Henry Hub Signals** (Score: {total_score})\n\n"
        f"💵 Цена: ${ind['price']:.3f}\n"
        f"📈 RSI-14: {ind['rsi']:.2f} | MA50: ${ind['ma50']:.3f} | MA200: ${ind['ma200']:.3f}\n\n"
        f"{levels_msg}\n"
        f"{storage_msg}\n"
        f"{news_msg}"
    )

    logger.info("Report generated")
    send_telegram(report)
    logger.info("=== END HENRY HUB SIGNALS ===")


if __name__ == "__main__":
    main()

