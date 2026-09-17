"""update_eia.py — запрос данных EIA и сохранение в data/eia_cache.json.

Запускается из GitHub Actions (американский IP, EIA доступен).
Локально из РФ — 403, поэтому только через Actions.

Патч от 2026-09-18:
- facets[series][]=NW2_EPG0_SWO_R48_BCF (Total Working Gas, US);
- start_date = 730 дней;
- debug-логирование сырого ответа EIA;
- fallback на v1 API (старый endpoint).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
CACHE_PATH = Path("data/eia_cache.json")

STORAGE_SEASONAL_NORM = {
    1: 2700, 2: 2300, 3: 1900, 4: 1700, 5: 1900, 6: 2200,
    7: 2500, 8: 2800, 9: 3100, 10: 3400, 11: 3600, 12: 3300,
}


def _get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _eia_get(url: str, timeout: int = 20):
    """Запрос к EIA с сохранением [] в URL (иначе EIA возвращает ошибку)."""
    s = _get_session()
    req = requests.Request("GET", url, headers={"User-Agent": "Mozilla/5.0"})
    prepared = req.prepare()
    prepared.url = url
    return s.send(prepared, timeout=timeout)


def _compute_storage_result(records: list, latest_period: str) -> dict:
    """Из записей EIA делает итоговый dict."""
    us_records = []
    for rec in records:
        area = str(rec.get("area", "")) + str(rec.get("area-name", "")) + \
               str(rec.get("duoarea", "")) + str(rec.get("region", ""))
        if "US" in area or "U.S." in area or "United States" in area or "NUS" in area:
            us_records.append(rec)

    if not us_records and records:
        us_records = [max(records, key=lambda x: float(x.get("value", 0) or 0))]

    if len(us_records) >= 2:
        us_records.sort(key=lambda x: x.get("period", ""), reverse=True)
        current = float(us_records[0]["value"])
        prev = float(us_records[1]["value"])
        build = current - prev
        period = us_records[0].get("period", "")
        month = datetime.now().month
        norm = STORAGE_SEASONAL_NORM.get(month, 3000)
        deviation_pct = ((current - norm) / norm * 100) if norm > 0 else 0

        print(f"[EIA Storage] OK: {current:.0f} Bcf, закачка {build:+.0f} Bcf, "
              f"период {period}, отклонение {deviation_pct:+.1f}%")

        return {
            "status": "ok",
            "storage": current,
            "change": build,
            "five_year_avg": norm,
            "deviation_pct": deviation_pct,
            "latest_period": period,
            "source": "eia_api",
        }
    elif len(us_records) == 1:
        current = float(us_records[0]["value"])
        period = us_records[0].get("period", "")
        month = datetime.now().month
        norm = STORAGE_SEASONAL_NORM.get(month, 3000)
        deviation_pct = ((current - norm) / norm * 100) if norm > 0 else 0

        print(f"[EIA Storage] Только 1 запись: {current:.0f} Bcf ({period})")

        return {
            "status": "ok",
            "storage": current,
            "change": 0,
            "five_year_avg": norm,
            "deviation_pct": deviation_pct,
            "latest_period": period,
            "source": "eia_api_partial",
        }

    return None


def fetch_storage(api_key: str) -> dict:
    """Запасы газа (weekly). Пробует несколько endpoints и фильтров."""

    # ── Попытка 1: EIA v2 с series NW2 (правильная серия) ──
    base_url_v2 = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    attempts = [
        {
            "facets": "facets[series][]=NW2_EPG0_SWO_R48_BCF",
            "label": "v2 series=NW2_EPG0_SWO_R48_BCF",
        },
        {
            "facets": "facets[duoarea][]=NUS&facets[process][]=SAV",
            "label": "v2 NUS+SAV",
        },
        {
            "facets": "facets[process][]=SAV",
            "label": "v2 process=SAV",
        },
        {
            "facets": "",
            "label": "v2 no facets",
        },
    ]

    for attempt in attempts:
        try:
            url = (
                f"{base_url_v2}"
                f"?api_key={api_key}"
                f"&frequency=weekly"
                f"&data[0]=value"
                f"&start={start_date}"
            )
            if attempt["facets"]:
                url += f"&{attempt['facets']}"
            url += (
                f"&sort[0][column]=period"
                f"&sort[0][direction]=desc"
                f"&length=20"
            )

            print(f"[EIA Storage v2] Попытка: {attempt['label']}")
            r = _eia_get(url, timeout=20)
            print(f"[EIA Storage v2] HTTP {r.status_code}")

            if r.status_code != 200:
                print(f"[EIA Storage v2] Ответ: {r.text[:300]}")
                continue

            body = r.json()
            response = body.get("response", {})
            records = response.get("data", [])

            print(f"[EIA Storage v2] Записей: {len(records)}")

            if not records:
                # Debug: показываем структуру ответа
                print(f"[EIA Storage v2] RAW (первые 500): "
                      f"{json.dumps(body, ensure_ascii=False)[:500]}")
                continue

            result = _compute_storage_result(records, "")
            if result:
                return result

        except Exception as e:
            print(f"[EIA Storage v2] Exception: {e}")
            continue

    # ── Попытка 2: EIA v1 (старый endpoint, часто надёжнее) ──
    try:
        print("[EIA Storage v1] Fallback на старый API...")
        v1_url = (
            f"https://api.eia.gov/series/"
            f"?api_key={api_key}"
            f"&series_id=NW2_EPG0_SWO_R48_BCF"
        )
        r = _eia_get(v1_url, timeout=20)
        print(f"[EIA Storage v1] HTTP {r.status_code}")

        if r.status_code == 200:
            body = r.json()
            series = body.get("series", [])
            if series:
                data_points = series[0].get("data", [])
                print(f"[EIA Storage v1] Точек данных: {len(data_points)}")

                if len(data_points) >= 2:
                    data_points_sorted = sorted(data_points, key=lambda x: x[0], reverse=True)
                    latest_period, latest_val = data_points_sorted[0]
                    prev_period, prev_val = data_points_sorted[1]
                    current = float(latest_val)
                    prev = float(prev_val)
                    build = current - prev

                    month = datetime.now().month
                    norm = STORAGE_SEASONAL_NORM.get(month, 3000)
                    deviation_pct = ((current - norm) / norm * 100) if norm > 0 else 0

                    print(f"[EIA Storage v1] OK: {current:.0f} Bcf, "
                          f"закачка {build:+.0f} Bcf, период {latest_period}")

                    return {
                        "status": "ok",
                        "storage": current,
                        "change": build,
                        "five_year_avg": norm,
                        "deviation_pct": deviation_pct,
                        "latest_period": latest_period,
                        "source": "eia_v1_api",
                    }
    except Exception as e:
        print(f"[EIA Storage v1] Exception: {e}")

    print("[EIA Storage] ВСЕ попытки неудачны")
    return {
        "status": "error",
        "storage": 0, "change": 0, "five_year_avg": 0,
        "deviation_pct": 0, "latest_period": "",
        "source": "error",
    }


def fetch_powerburn(api_key: str) -> dict:
    """Powerburn (hourly)."""
    try:
        url = (
            f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
            f"?api_key={api_key}"
            f"&frequency=hourly"
            f"&data[0]=value"
            f"&sort[0][column]=period"
            f"&sort[0][direction]=desc"
            f"&length=500"
        )
        print("[Powerburn] Запрос...")
        r = _eia_get(url, timeout=20)
        print(f"[Powerburn] HTTP {r.status_code}")

        if r.status_code != 200:
            print(f"[Powerburn] Ответ: {r.text[:200]}")
            return {"status": "error", "source": "error"}

        records = r.json().get("response", {}).get("data", [])
        if not records:
            return {"status": "error", "source": "error"}

        latest_period = records[0].get("period", "")
        hour_records = [rec for rec in records if rec.get("period") == latest_period]

        fuel_totals = {}
        for rec in hour_records:
            ft = rec.get("fueltype", rec.get("type", ""))
            val = float(rec.get("value", 0) or 0)
            fuel_totals[ft] = fuel_totals.get(ft, 0) + val

        total_gen = sum(fuel_totals.values())
        ng_gen = fuel_totals.get("NG", 0)
        coal_gen = fuel_totals.get("COL", 0)

        ng_pct = (ng_gen / total_gen * 100) if total_gen > 0 else 0
        coal_pct = (coal_gen / total_gen * 100) if total_gen > 0 else 0
        daily_bcf = (ng_gen * 7.5 / 1_000_000) * 24 if ng_gen > 0 else 0

        print(f"[Powerburn] OK: газ {ng_pct:.1f}%, уголь {coal_pct:.1f}%, "
              f"{daily_bcf:.1f} BCF/d, период {latest_period}")

        return {
            "status": "ok",
            "daily_bcf": daily_bcf,
            "ng_pct": ng_pct,
            "coal_pct": coal_pct,
            "latest_period": latest_period,
            "source": "eia_api",
        }
    except Exception as e:
        print(f"[Powerburn] Exception: {e}")
        return {"status": "error", "source": "error"}


def main():
    if not EIA_API_KEY:
        print("[ERROR] EIA_API_KEY не задан. Установи секрет в GitHub Actions.")
        sys.exit(1)

    print(f"=== EIA Update {datetime.now(timezone.utc).isoformat()} UTC ===")

    storage = fetch_storage(EIA_API_KEY)
    time.sleep(1)  # rate limit EIA
    powerburn = fetch_powerburn(EIA_API_KEY)

    cache = {
        "ts": time.time(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "storage": storage,
        "powerburn": powerburn,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ Сохранено в {CACHE_PATH}")
    print(json.dumps(cache, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
