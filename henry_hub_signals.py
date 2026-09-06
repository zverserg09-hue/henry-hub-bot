"""
Henry Hub Natural Gas — автоматическая система сигналов
Источник цен: EIA API v2 (дневные спотовые цены Henry Hub)
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

LOG_FILE = "henry_hub_signals.log"
CACHE_FILE = "cache_eia_ng_prices.csv"

LAST_SIGNAL_FILE = "last_signal.json"
REQUEST_TIMEOUT_SECONDS = 10

# EIA Series ID: Henry Hub Natural Gas Spot Price, Daily
EIA_PRICE_SERIES_ID = "NG.RNGWHHD.D"
# EIA Series ID: Weekly Natural Gas Storage Report
EIA_STORAGE_SERIES_ID = "NG.NW2_EPG0_SGO_RNG_RNGFM_WUS"

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
# МОДУЛЬ 1: ЦЕНОВЫЕ ДАННЫЕ (EIA API v2)
# ============================================================

def fetch_prices(max_retries=3):
    """
    Загружает дневные спотовые цены Henry Hub из EIA API v2.
    Поскольку EIA отдаёт только одну цену в день (spot),
    Open=High=Low=Close=price, Volume=1.
    Возвращает DataFrame с индексом Date.
    """
    logger.info("Начинаем загрузку цен Henry Hub из EIA API")

    if not EIA_API_KEY:
        logger.error("EIA_API_KEY не задан. Невозможно загрузить цены.")
        return pd.DataFrame()

    # 1. Проверяем кэш (свежее ли он — моложе 24 часов)
    if os.path.exists(CACHE_FILE):
        file_age = time.time() - os.path.getmtime(CACHE_FILE)
        if file_age < 24 * 3600:
            try:
                df = pd.read_csv(CACHE_FILE, index_col="Date", parse_dates=["Date"])
                if not df.empty:
                    logger.info(f"Данные загружены из кэша ({len(df)} записей).")
                    return df
            except Exception as e:
                logger.warning(f"Не удалось прочитать кэш: {e}. Запрашиваем заново.")

    # 2. Запрос к EIA через /seriesid/ (совместимость с v1 Series ID)
    url = (
        f"https://api.eia.gov/v2/seriesid/{EIA_PRICE_SERIES_ID}"
        f"?api_key={EIA_API_KEY}"
    )

    r = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            if r.status_code == 200:
                logger.info("Цены получены от EIA API.")
                break

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "60")
                delay = int(retry_after) if str(retry_after).isdigit() else 60
                logger.warning(
                    f"429 от EIA. Ждём {delay} сек (попытка {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue

            logger.error(
                f"Ошибка EIA API: статус {r.status_code}, ответ: {r.text[:200]}"
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

        # Пробуем v1-style формат (series[0].data — массив [date, value])
        series_data = None
        if "series" in data and len(data["series"]) > 0:
            series_data = data["series"][0]["data"]
        # Пробуем v2 native формат (response.data — массив объектов)
        elif "response" in data and "data" in data["response"]:
            raw = data["response"]["data"]
            series_data = [[item["period"], item["value"]] for item in raw]

        if not series_data:
            logger.error("В ответе EIA нет данных о ценах.")
            return pd.DataFrame()

        # Парсим даты и значения
        dates = []
        prices = []
        for entry in series_data:
            date_str = str(entry[0])
            price_raw = entry[1]

            if price_raw is None or price_raw == "":
                continue

            try:
                price = float(price_raw)
            except (ValueError, TypeError):
                continue

            # EIA возвращает даты в разных форматах: "20240101" или "2024-01-01"
            try:
                if "-" in date_str:
                    d = pd.to_datetime(date_str)
                else:
                    d = pd.to_datetime(date_str, format="%Y%m%d")
            except Exception:
                try:
                    d = pd.to_datetime(date_str)
                except Exception:
                    continue

            dates.append(d)
            prices.append(price)

        if not dates:
            logger.error("Нет валидных цен в ответе EIA.")
            return pd.DataFrame()

        # EIA отдаёт данные от новых к старым — сортируем по возрастанию
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
                "Volume": [1] * len(prices),  # нет реального объёма — ставим 1
            }
        )
        df.dropna(inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Оставляем последние 2 года
        cutoff = datetime.now() - timedelta(days=730)
        df = df[df.index >= cutoff]

        # Сохраняем в кэш
        df.to_csv(CACHE_FILE, index=True)
        logger.info(f"Данные получены от EIA и сохранены в кэш ({len(df)} записей).")
        return df

    except Exception as e:
        logger.error(f"Ошибка парсинга JSON от EIA: {e}")
        return pd.DataFrame()


# ============================================================
# МОДУЛЬ 2: ИНДИКАТОРЫ
# ============================================================


def calc_indicators(df):
    logger.info("Расчёт технических индикаторов")
    close = df["Close"]

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Moving Averages
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd - signal_line).iloc[-1]

    # ATR: поскольку H=L=C (spot price), True Range = |Close - Close_prev|
    # Это упрощённый ATR — среднее абсолютное изменение цены за 14 дней
    tr = np.abs(close - close.shift(1))
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
    """
    Ищет уровни поддержки и сопротивления по экстремумам.
    Поскольку H=L=C (spot price), используются Close как High и Low.
    """
    data = df.tail(120)
    highs = data["Close"].values  # используем Close вместо High
    lows = data["Close"].values   # используем Close вместо Low

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
    """
    Считает пивот-уровни. Поскольку H=L=C (spot price),
    используем диапазон за последние 5 дней как proxy для H/L.
    """
    recent = df.tail(5)
    h = recent["Close"].max()
    l = recent["Close"].min()
    c = df.iloc[-1]["Close"]
    p = (h + l + c) / 3
    return {
        "P": p,
        "R1": 2 * p - l,
        "S1": 2 * p - h,
        "R2": p + (h - l),
        "S2": p - (h - l),
    }


def volume_profile(df, lookback=60, num_bins=40):
    """
    Профиль объёма. Поскольку реального объёма нет (Volume=1),
    это фактически профиль частоты цен — где цена проводила больше всего дней.
    """
    data = df.tail(lookback)
    if len(data) == 0:
        return {"poc": 0, "val": 0, "vah": 0, "hvn": []}

    min_p = data["Close"].min()
    max_p = data["Close"].max()
    if min_p == max_p:
        return {"poc": min_p, "val": min_p, "vah": min_p, "hvn": [min_p]}

    bins = np.linspace(min_p, max_p, num_bins + 1)
    vol_by_bin = np.zeros(num_bins)

    for _, row in data.iterrows():
        price = row["Close"]
        vol = row.get("Volume", 1)
        if vol == 0:
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

    # Value Area: 70% от общего объёма (как в TradingView)
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
# МОДУЛЬ 3: ДАННЫЕ EIA (запасы)
# ============================================================


def get_eia_storage():
    if not EIA_API_KEY:
        logger.warning("EIA API KEY не задан, используем fallback-значения")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

    try:
        url = (
            f"https://api.eia.gov/v2/seriesid/{EIA_STORAGE_SERIES_ID}"
            f"?api_key={EIA_API_KEY}"
        )
        r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()

        # Пробуем v1-style и v2-style форматы
        values = None
        if "series" in data and len(data["series"]) > 0:
            values = data["series"][0]["data"]
        elif "response" in data and "data" in data["response"]:
            raw = data["response"]["data"]
            values = [[item["period"], item["value"]] for item in raw]

        if not values:
            logger.warning("EIA Storage: нет данных в ответе.")
            return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST

        current = float(values[0][1])
        prev = float(values[1][1])
        build = current - prev
        logger.info(f"EIA Storage: current={current}, build={build}")
        return current, build, STORAGE_FORECAST
    except Exception as e:
        logger.error(f"Ошибка EIA API (storage): {e}")
        return STORAGE_CURRENT_BCF, LAST_STORAGE_BUILD, STORAGE_FORECAST


def score_storage(storage_bcf, build, forecast):
    """
    Скоринг по запасам газа (от -3 до +3).
    Логика: отклонение от 5-летней средней + сюрприз по закачке.
    """
    score = 0
    avg_5yr = 3000
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
    logger.info("=== START HENRY HUB SIGNALS (EIA) ===")

    # 1. Цены
    df = fetch_prices()
    if df.empty:
        logger.critical("Не удалось загрузить цены из EIA.")
        send_telegram(
            "❌ Henry Hub: ошибка загрузки цен\n"
            "EIA API вернул пустые данные. Проверьте EIA_API_KEY."
        )
        return
    logger.info(f"Loaded {len(df)} price bars (EIA daily spot)")

    # 2. Индикаторы
    ind = calc_indicators(df)

    # 3. Уровни
    support_lvls, resistance_lvls = find_swing_levels(df)
    pivots = calc_pivots(df)
    vp = volume_profile(df)
    levels_msg, nearest_sup, nearest_res = format_levels_message(
        ind["price"], vp, support_lvls, resistance_lvls, pivots
    )

    # 4. Запасы EIA
    storage_bcf, build, forecast = get_eia_storage()

    # 5. Новости
    news_titles = parse_news()
    news_msg = (
        "📰 Новости (последние 6 ч):\n" + "\n".join([f"• {t}" for t in news_titles])
        if news_titles
        else "📰 Новостей за 6 часов нет"
    )

    # ============================================================
    # 6. СКОРИНГ — 8 ФАКТОРОВ (от -8 до +8)
    # ============================================================
    score = 0
    factors = []

    # Фактор 1: RSI дневной
    if ind["rsi"] >= 70:
        score -= 1
        factors.append(f"RSI {ind['rsi']:.1f} — перекупленность → -1")
    elif ind["rsi"] <= 30:
        score += 1
        factors.append(f"RSI {ind['rsi']:.1f} — перепроданность → +1")
    else:
        factors.append(f"RSI {ind['rsi']:.1f} — нейтрально → 0")

    # Фактор 2: Цена vs MA50
    if ind["price"] > ind["ma50"]:
        score += 1
        factors.append(f"Цена > MA50 ${ind['ma50']:.3f} → +1")
    else:
        score -= 1
        factors.append(f"Цена < MA50 ${ind['ma50']:.3f} → -1")

    # Фактор 3: Цена vs MA200
    if ind["price"] > ind["ma200"]:
        score += 1
        factors.append(f"Цена > MA200 ${ind['ma200']:.3f} → +1")
    else:
        score -= 1
        factors.append(f"Цена < MA200 ${ind['ma200']:.3f} → -1")

    # Фактор 4: Bollinger Bands
    if ind["price"] >= ind["bb_upper"]:
        score -= 1
        factors.append(f"Цена у верхней BB ${ind['bb_upper']:.3f} → -1")
    elif ind["price"] <= ind["bb_lower"]:
        score += 1
        factors.append(f"Цена у нижней BB ${ind['bb_lower']:.3f} → +1")
    else:
        factors.append("Цена внутри BB → 0")

    # Фактор 5: MACD гистограмма
    if ind["macd_hist"] > 0:
        score += 1
        factors.append("MACD гист. > 0 → +1")
    else:
        score -= 1
        factors.append("MACD гист. < 0 → -1")

    # Фактор 6: Сезонность
    month = datetime.now().month
    seas = seasonality_score(month)
    score += seas
    seas_word = "бычий" if seas > 0 else "медвежий" if seas < 0 else "нейтральный"
    factors.append(f"Сезон ({seas_word}) → {seas:+d}")

    # Фактор 7: Запасы (отклонение от 5-летней средней)
    avg_5yr = 3000
    pct = ((storage_bcf - avg_5yr) / avg_5yr) * 100
    if pct > 5:
        score -= 2
        factors.append(f"Запасы +{pct:.1f}% к норме → -2")
    elif pct < -5:
        score += 2
        factors.append(f"Запасы {pct:.1f}% к норме → +2")
    else:
        factors.append(f"Запасы {pct:+.1f}% к норме → 0")

    # Фактор 8: Закачка vs прогноз
    if build > 0 and forecast > 0:
        if build > forecast * 1.3:
            score -= 1
            factors.append(f"Закачка {build:.0f} > прогноз {forecast:.0f} → -1")
        elif build < forecast * 0.7:
            score += 1
            factors.append(f"Закачка {build:.0f} < прогноз {forecast:.0f} → +1")
        else:
            factors.append("Закачка ≈ прогноз → 0")
    else:
        factors.append("Закачка/прогноз: нет данных → 0")

    # ============================================================
    # 7. СИГНАЛ
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
    # 8. УРОВНИ СДЕЛКИ (ATR × 1.5 — компенсация заниженного ATR)
    # ============================================================
    atr_adj = ind["atr"] * 1.5
    price = ind["price"]

    trade_msg = ""
    if score <= -2:
        # Шорт: стоп выше, тейки ниже
        stop = price + atr_adj
        tp1 = price - 2 * atr_adj
        tp2 = price - 4 * atr_adj
        risk = (stop - price) / price * 100
        rr = abs(tp1 - price) / abs(stop - price)
        trade_msg = (
            f"📌 Уровни (шорт):\n"
            f"   Вход: ${price:.3f}\n"
            f"   Стоп: ${stop:.3f} (риск {risk:.1f}%)\n"
            f"   TP1: ${tp1:.3f}\n"
            f"   TP2: ${tp2:.3f}\n"
            f"   R/R: {rr:.2f}\n"
        )
    elif score >= 2:
        # Лонг: стоп ниже, тейки выше
        stop = price - atr_adj
        tp1 = price + 2 * atr_adj
        tp2 = price + 4 * atr_adj
        risk = (price - stop) / price * 100
        rr = abs(tp1 - price) / abs(price - stop)
        trade_msg = (
            f"📌 Уровни (лонг):\n"
            f"   Вход: ${price:.3f}\n"
            f"   Стоп: ${stop:.3f} (риск {risk:.1f}%)\n"
            f"   TP1: ${tp1:.3f}\n"
            f"   TP2: ${tp2:.3f}\n"
            f"   R/R: {rr:.2f}\n"
        )
    else:
        trade_msg = "📌 Уровней нет — сигнал слабый, жди подтверждения\n"

    # ============================================================
    # 9. ОТЧЁТ
    # ============================================================
    factors_msg = "\n".join([f"   {f}" for f in factors])
    last_date = df.index[-1].strftime("%Y-%m-%d")

    report = (
        f"{signal_text} (Score: {score:+d}/8)\n\n"
        f"💵 Цена (EIA spot): ${price:.3f}\n"
        f"📅 Последняя дата данных: {last_date}\n"
        f"📈 RSI: {ind['rsi']:.1f} | MA50: ${ind['ma50']:.3f} | MA200: ${ind['ma200']:.3f}\n\n"
        f"{trade_msg}\n"
        f"{levels_msg}\n"
        f"📊 Факторы:\n{factors_msg}\n\n"
        f"Текущие: {storage_bcf:.0f} Bcf ({pct:+.1f}% к 5л ср.)\n"
        f"⚖️ Закачка {build:.0f} Bcf, прогноз {forecast:.0f} Bcf\n\n"
        f"{news_msg}"
    )

    logger.info(f"Signal: {signal_text}, Score: {score}")
    send_telegram(report)
    logger.info("=== END HENRY HUB SIGNALS ===")


if __name__ == "__main__":
    main()




if __name__ == "__main__":
    main()

