from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# KurgiMootor LAB-135 HYBRID
# ÜKS fikseeritud testmudel: mitu viimast sama põllu korjet + kogu põlluploki
# värske režiim + kasvuaeg + kasvuperioodi ilm + ilmamuutus.
#
# READ ONLY:
# - ei impordi production app.py-d ega WeatherService'it
# - ei käivita Jäljeotsijat / challenger-search'i
# - ei uuenda ilma
# - ei tee ühtegi db.save_* / db.set_* / db.delete_* kutset
# - ei muuda production prognoosi
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-135-HYBRID-V1"
SEASON_START = date(TODAY.year, 6, 15)

# Fikseeritud enne testi. Neid EI valita viimase 14 päeva tulemuse järgi.
RIDGE_ALPHA = 22.0
RECENCY_HALFLIFE_DAYS = 18.0
MIN_TRAIN_ROWS = 18
TARGET_EPS = 0.25

WEATHER_COLS = (
    "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
    "wind_avg_ms", "radiation_mj_m2", "humidity_avg_pct",
    "precipitation_mm", "et0_mm",
)

# Üks ja ainus mudelipere. Eelmiste korjete kaalud õpib ridge ise, sest lag1/2/3
# on eraldi tunnused; me ei käsitsi sega neid üheks keskmiseks.
FEATURES = [
    # sama põllu seisund ja trend
    "log_lag1", "log_lag2", "log_lag3", "has_lag2", "has_lag3",
    "same_trend1", "same_trend2",
    # kasvuaeg
    "growth", "prev_growth", "growth_delta", "log_growth_ratio",
    # praeguse kasvuperioodi ilm (päevakeskmine / päevane tase)
    "gw_temp", "gw_rad", "gw_gdd10", "gw_et0", "gw_rh", "gw_wind", "gw_rain",
    # muutus võrreldes eelmise sama põllu kasvuperioodiga
    "dw_temp", "dw_rad", "dw_gdd10", "dw_et0", "dw_rh", "dw_wind", "dw_rain",
    # 7 päeva taustailm ja muutus eelneva 7 päeva vastu
    "d7_temp", "d7_rad", "d7_gdd10", "d7_et0", "d7_rh", "d7_wind",
    # värske kogu põlluploki sensor: viimased 1 ja 3 täielikku korjepäeva
    "regime1", "regime3", "regime_growth1", "regime_growth3",
    # hooaja aeg ainult pehme trendina; ridge regulariseerib
    "season_day",
]


def _d(v) -> Optional[date]:
    if isinstance(v, date):
        return v
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except Exception:
        return None


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc(row: dict) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    return None if any(v is None for v in vals) else float(sum(vals))


def _quality_ok(row: dict) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _weather_rank(row: dict, day_value: date, today: date) -> int:
    kind = str(row.get("data_kind") or "").lower()
    checked = bool(row.get("checked"))
    # Minevik: ainult mõõdetud read on ausad. Tänane/tulevik: prognoos lubatud.
    if day_value < today:
        if kind == "measured" and checked:
            return 5
        if kind == "measured":
            return 4
        return 0
    if kind == "measured" and checked:
        return 5
    if kind == "forecast":
        return 4
    if kind == "measured":
        return 3
    return 1


def _weather_complete(row: Optional[dict]) -> bool:
    return row is not None and all(_f(row.get(c)) is not None for c in WEATHER_COLS)


def _build_weather_map(rows: List[dict], today: date) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    ranks: Dict[date, int] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None or not _weather_complete(r):
            continue
        rank = _weather_rank(r, dd, today)
        if rank > ranks.get(dd, -1):
            out[dd] = r
            ranks[dd] = rank
    return out


def _range_weather(wmap: Dict[date, dict], start_day: date, end_day: date) -> Optional[List[dict]]:
    if start_day is None or end_day is None or start_day > end_day:
        return []
    out = []
    dd = start_day
    while dd <= end_day:
        r = wmap.get(dd)
        if r is None:
            return None
        out.append(r)
        dd += timedelta(days=1)
    return out


def _agg_weather(rows: Optional[List[dict]]) -> Optional[dict]:
    if not rows:
        return None
    night = np.asarray([float(r["temp_night_avg_c"]) for r in rows], dtype=float)
    dayt = np.asarray([float(r["temp_day_avg_c"]) for r in rows], dtype=float)
    rad = np.asarray([float(r["radiation_mj_m2"]) for r in rows], dtype=float)
    et0 = np.asarray([float(r["et0_mm"]) for r in rows], dtype=float)
    rh = np.asarray([float(r["humidity_avg_pct"]) for r in rows], dtype=float)
    wind = np.asarray([float(r["wind_avg_ms"]) for r in rows], dtype=float)
    rain = np.asarray([float(r["precipitation_mm"]) for r in rows], dtype=float)
    temp = 0.5 * (night + dayt)
    gdd10 = np.maximum(0.0, temp - 10.0)
    return {
        "temp": float(temp.mean()),
        "rad": float(rad.mean()),
        "gdd10": float(gdd10.mean()),
        "et0": float(et0.mean()),
        "rh": float(rh.mean()),
        "wind": float(wind.mean()),
        "rain": float(rain.mean()),
        "n": int(len(rows)),
    }


def _growth_days(prev_row: dict, cur_row: dict, prev_day: date, cur_day: date, cur_order: Optional[int] = None) -> float:
    base = float((cur_day - prev_day).days)
    po = int(prev_row.get("harvest_order") or 1)
    co = int(cur_order if cur_order is not None else (cur_row.get("harvest_order") or 1))
    return base + (co - po) * (3.0 / 24.0)


def _complete_days(by_day: Dict[date, List[dict]], plans) -> List[date]:
    out = []
    for dd, rows in by_day.items():
        fields = {int(r["field_no"]) for r in rows}
        plan = plans.get(dd.isoformat()) if isinstance(plans, dict) else None
        if plan is not None:
            try:
                expected = {int(x) for x in plan}
            except Exception:
                expected = set()
            ok = bool(expected) and fields == expected and len(rows) == len(expected)
        else:
            ok = len(rows) == 3 and len(fields) == 3
        if ok:
            out.append(dd)
    return sorted(set(out))


def _history_before(items: List[Tuple[date, dict]], target_day: date) -> List[Tuple[date, dict]]:
    return [x for x in items if x[0] < target_day]


def _growth_for_history_item(items: List[Tuple[date, dict]], idx: int) -> Optional[float]:
    if idx <= 0 or idx >= len(items):
        return None
    d0, r0 = items[idx - 1]
    d1, r1 = items[idx]
    return _growth_days(r0, r1, d0, d1)


def _day_regime_stats(
    cutoff_day: date,
    by_day: Dict[date, List[dict]],
    by_field: Dict[int, List[Tuple[date, dict]]],
    complete_days: List[date],
) -> dict:
    """Režiimisensor ainult cutoff_day'ile eelnenud täielikest päevadest.

    Päeva signaal = mediaan üle põldude log(ABC_now / ABC_prev_same_field).
    Lisame eraldi kasvuaegade suhte mediaani, et 4p/5p erinevus ei maskeeruks
    režiimiks; mudel õpib ise, kui palju seda arvestada.
    """
    prior_days = [d for d in complete_days if d < cutoff_day]
    day_signals = []
    for dd in prior_days:
        ratios = []
        growth_ratios = []
        for row in by_day.get(dd, []):
            f = int(row["field_no"])
            cur_abc = _abc(row)
            items = by_field.get(f, [])
            pos = next((i for i, (xday, xrow) in enumerate(items) if xday == dd and xrow is row), None)
            if pos is None:
                # Fallback: kuupäev + sama field; duplikaate tavaliselt pole.
                pos = next((i for i, (xday, _xrow) in enumerate(items) if xday == dd), None)
            if pos is None or pos <= 0 or cur_abc is None or cur_abc <= 0:
                continue
            pd, pr = items[pos - 1]
            prev_abc = _abc(pr)
            if prev_abc is None or prev_abc <= 0:
                continue
            ratios.append(math.log((cur_abc + TARGET_EPS) / (prev_abc + TARGET_EPS)))
            cur_g = _growth_days(pr, row, pd, dd)
            prev_g = _growth_for_history_item(items, pos - 1)
            if prev_g is not None and prev_g > 0 and cur_g > 0:
                growth_ratios.append(math.log(cur_g / prev_g))
        if ratios:
            day_signals.append((dd, float(np.median(ratios)), float(np.median(growth_ratios)) if growth_ratios else 0.0))

    if not day_signals:
        return {"regime1": 0.0, "regime3": 0.0, "regime_growth1": 0.0, "regime_growth3": 0.0}

    last = day_signals[-1]
    last3 = day_signals[-3:]
    # Viimase 3 päeva sees värskematel on suurem kaal; see ei ole mudeli saagikaal,
    # vaid ainult sensori ajaliseks silumiseks 3 päeva aknas.
    base_w = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
    base_w = base_w / base_w.sum()
    return {
        "regime1": float(last[1]),
        "regime3": float(sum(w * x[1] for w, x in zip(base_w, last3))),
        "regime_growth1": float(last[2]),
        "regime_growth3": float(sum(w * x[2] for w, x in zip(base_w, last3))),
    }


def _build_record(
    *,
    field: int,
    target_day: date,
    cur_row: dict,
    cur_order: int,
    by_field: Dict[int, List[Tuple[date, dict]]],
    by_day: Dict[date, List[dict]],
    complete_days: List[date],
    wmap: Dict[date, dict],
    historical_actual: bool,
) -> Optional[dict]:
    items_all = by_field.get(field, [])
    if historical_actual:
        # Leia täpselt target_day rida; tema ees olevad read on lagid.
        idx = next((i for i, (dd, _r) in enumerate(items_all) if dd == target_day), None)
        if idx is None or idx < 1:
            return None
        hist = items_all[:idx]
    else:
        hist = _history_before(items_all, target_day)
        if len(hist) < 1:
            return None

    lag1_day, lag1 = hist[-1]
    lag2_day, lag2 = hist[-2] if len(hist) >= 2 else (None, None)
    lag3_day, lag3 = hist[-3] if len(hist) >= 3 else (None, None)

    a1 = _abc(lag1)
    a2 = _abc(lag2) if lag2 else None
    a3 = _abc(lag3) if lag3 else None
    if a1 is None or a1 <= 0:
        return None

    # Puuduv lag täidetakse lag1 tasemega + eraldi has_lag tunnus hoiab selle ausana.
    has2 = 1.0 if a2 is not None and a2 > 0 else 0.0
    has3 = 1.0 if a3 is not None and a3 > 0 else 0.0
    a2f = a2 if has2 else a1
    a3f = a3 if has3 else a2f

    growth = _growth_days(lag1, cur_row, lag1_day, target_day, cur_order=cur_order)
    prev_growth = _growth_days(lag2, lag1, lag2_day, lag1_day) if lag2 is not None else growth
    if growth <= 0 or prev_growth <= 0:
        return None

    # Aus ilm: sihtpäeva mõõdetud ilma ei kasutata. Kasvuperioodi täielikud päevad
    # on eelmise korje järgmisest päevast kuni sihtpäeva eelse päevani.
    gw_rows = _range_weather(wmap, lag1_day + timedelta(days=1), target_day - timedelta(days=1))
    if gw_rows is None or not gw_rows:
        return None
    gw = _agg_weather(gw_rows)
    if gw is None:
        return None

    # Eelmise sama põllu kasvuperioodi ilm võrdluseks.
    if lag2_day is not None:
        pgw_rows = _range_weather(wmap, lag2_day + timedelta(days=1), lag1_day - timedelta(days=1))
        pgw = _agg_weather(pgw_rows) if pgw_rows is not None and pgw_rows else None
    else:
        pgw = None
    if pgw is None:
        pgw = dict(gw)

    # 7p vs eelnev 7p taust. Kõik lõpeb sihtpäeva eel.
    r7_rows = _range_weather(wmap, target_day - timedelta(days=7), target_day - timedelta(days=1))
    p7_rows = _range_weather(wmap, target_day - timedelta(days=14), target_day - timedelta(days=8))
    r7 = _agg_weather(r7_rows) if r7_rows is not None else None
    p7 = _agg_weather(p7_rows) if p7_rows is not None else None
    if r7 is None or p7 is None:
        return None

    regime = _day_regime_stats(target_day, by_day, by_field, complete_days)

    rec = {
        "target_day": target_day,
        "field_no": int(field),
        "log_lag1": math.log(a1 + TARGET_EPS),
        "log_lag2": math.log(float(a2f) + TARGET_EPS),
        "log_lag3": math.log(float(a3f) + TARGET_EPS),
        "has_lag2": has2,
        "has_lag3": has3,
        "same_trend1": math.log((a1 + TARGET_EPS) / (float(a2f) + TARGET_EPS)) if has2 else 0.0,
        "same_trend2": math.log((float(a2f) + TARGET_EPS) / (float(a3f) + TARGET_EPS)) if has3 else 0.0,
        "growth": float(growth),
        "prev_growth": float(prev_growth),
        "growth_delta": float(growth - prev_growth),
        "log_growth_ratio": math.log(growth / prev_growth),
        "gw_temp": gw["temp"], "gw_rad": gw["rad"], "gw_gdd10": gw["gdd10"],
        "gw_et0": gw["et0"], "gw_rh": gw["rh"], "gw_wind": gw["wind"], "gw_rain": gw["rain"],
        "dw_temp": gw["temp"] - pgw["temp"], "dw_rad": gw["rad"] - pgw["rad"],
        "dw_gdd10": gw["gdd10"] - pgw["gdd10"], "dw_et0": gw["et0"] - pgw["et0"],
        "dw_rh": gw["rh"] - pgw["rh"], "dw_wind": gw["wind"] - pgw["wind"],
        "dw_rain": gw["rain"] - pgw["rain"],
        "d7_temp": r7["temp"] - p7["temp"], "d7_rad": r7["rad"] - p7["rad"],
        "d7_gdd10": r7["gdd10"] - p7["gdd10"], "d7_et0": r7["et0"] - p7["et0"],
        "d7_rh": r7["rh"] - p7["rh"], "d7_wind": r7["wind"] - p7["wind"],
        **regime,
        "season_day": float((target_day - SEASON_START).days),
        # diagnostika
        "lag1_abc": float(a1), "lag2_abc": float(a2) if a2 is not None else None,
        "lag3_abc": float(a3) if a3 is not None else None,
    }

    if historical_actual:
        cur_abc = _abc(cur_row)
        if cur_abc is None or cur_abc <= 0:
            return None
        rec["actual_abc"] = float(cur_abc)
        # Target on MUUTUS viimase sama põllu tasemest, mitte absoluutne saak.
        rec["y"] = math.log((cur_abc + TARGET_EPS) / (a1 + TARGET_EPS))
    else:
        rec["actual_abc"] = None
        rec["y"] = None
    return rec


def _ridge_fit_predict(train_records: List[dict], pred_record: dict, cutoff_day: date) -> Optional[float]:
    clean = []
    for r in train_records:
        if r.get("y") is None:
            continue
        vals = [_f(r.get(c)) for c in FEATURES]
        if any(v is None for v in vals):
            continue
        age = max(0.0, float((cutoff_day - r["target_day"]).days))
        # Värske info kaalub rohkem, kuid vana info ei kao. 18p poolestusaeg on
        # fikseeritud enne replay'd, mitte viimase 14 päeva järgi optimeeritud.
        weight = 0.5 ** (age / RECENCY_HALFLIFE_DAYS)
        clean.append((vals, float(r["y"]), weight))

    xnew = [_f(pred_record.get(c)) for c in FEATURES]
    if len(clean) < MIN_TRAIN_ROWS or any(v is None for v in xnew):
        return None

    X = np.asarray([x for x, _, _ in clean], dtype=float)
    y = np.asarray([yy for _, yy, _ in clean], dtype=float)
    w = np.asarray([ww for _, _, ww in clean], dtype=float)
    x = np.asarray(xnew, dtype=float)

    # Kaalutud standardiseerimine ainult treeningandmete pealt.
    sw = float(w.sum())
    mu = (X * w[:, None]).sum(axis=0) / sw
    var = ((X - mu) ** 2 * w[:, None]).sum(axis=0) / sw
    sd = np.sqrt(np.maximum(var, 1e-8))
    Z = (X - mu) / sd
    z = (x - mu) / sd
    # Väldi ühe tuleviku ekstreemse ilma tunnuse domineerimist.
    z = np.clip(z, -3.0, 3.0)

    D = np.column_stack([np.ones(len(Z)), Z])
    Wsqrt = np.sqrt(w)[:, None]
    Dw = D * Wsqrt
    yw = y * Wsqrt[:, 0]
    P = np.eye(D.shape[1], dtype=float)
    P[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Dw.T @ Dw + RIDGE_ALPHA * P, Dw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Dw.T @ Dw + RIDGE_ALPHA * P) @ (Dw.T @ yw)

    yhat = float(np.r_[1.0, z] @ beta)
    # Turvapiir õpitakse ainult varem nähtud muutustest; ei ole saagitaseme hard-code.
    if len(y) >= 12:
        lo, hi = np.quantile(y, [0.02, 0.98])
        pad = max(0.05, 0.12 * float(hi - lo))
        yhat = float(np.clip(yhat, lo - pad, hi + pad))
    return yhat


def _predict_abc(train_records: List[dict], rec: dict, cutoff_day: date) -> Optional[float]:
    yhat = _ridge_fit_predict(train_records, rec, cutoff_day)
    if yhat is None:
        return None
    lag1 = float(rec["lag1_abc"])
    return max(0.0, (lag1 + TARGET_EPS) * math.exp(yhat) - TARGET_EPS)


def _prepare_harvests(harvest_rows_raw: List[dict]):
    harvest_rows = []
    for r in harvest_rows_raw:
        dd = _d(r.get("harvest_date"))
        try:
            f = int(r.get("field_no"))
        except Exception:
            continue
        if dd is None or not (1 <= f <= 14) or not _quality_ok(r) or _abc(r) is None:
            continue
        rr = dict(r)
        rr["field_no"] = f
        rr["_day"] = dd
        harvest_rows.append(rr)

    by_field = {f: [] for f in range(1, 15)}
    by_day: Dict[date, List[dict]] = {}
    for r in harvest_rows:
        by_field[r["field_no"]].append((r["_day"], r))
        by_day.setdefault(r["_day"], []).append(r)
    for f in by_field:
        by_field[f].sort(key=lambda x: (x[0], int(x[1].get("harvest_order") or 99)))
    return harvest_rows, by_field, by_day


def _make_historical_records(by_field, by_day, complete_days, wmap) -> List[dict]:
    records = []
    for f, items in by_field.items():
        for i in range(1, len(items)):
            dd, row = items[i]
            rec = _build_record(
                field=f, target_day=dd, cur_row=row,
                cur_order=int(row.get("harvest_order") or 1),
                by_field=by_field, by_day=by_day, complete_days=complete_days,
                wmap=wmap, historical_actual=True,
            )
            if rec is not None:
                records.append(rec)
    records.sort(key=lambda r: (r["target_day"], r["field_no"]))
    return records


def _official_1d_lookup(forecasts: List[dict], actual_fields_by_day: Dict[date, set]) -> Dict[Tuple[date, int], dict]:
    candidates = {}
    for fr in forecasts:
        td = _d(fr.get("target_date"))
        fd = _d(fr.get("forecast_date"))
        try:
            field = int(fr.get("field_no"))
            lead = int(fr.get("lead_days") or 0)
        except Exception:
            continue
        if td is None or field not in actual_fields_by_day.get(td, set()):
            continue
        if lead != 1 and not (fd is not None and fd == td - timedelta(days=1)):
            continue
        gen = str(fr.get("generated_at") or fr.get("created_at") or "")
        key = (td, field)
        if key not in candidates or gen >= candidates[key][0]:
            candidates[key] = (gen, fr)
    return {k: v[1] for k, v in candidates.items()}


def _latest_official_for_target(forecast_rows: List[dict], target_day: date, fields: List[int]) -> Dict[int, dict]:
    field_set = set(fields)
    rows = []
    for fr in forecast_rows:
        td = _d(fr.get("target_date"))
        try:
            ff = int(fr.get("field_no"))
        except Exception:
            continue
        if td == target_day and ff in field_set:
            rows.append(fr)
    if not rows:
        return {}
    max_fd = max((_d(r.get("forecast_date")) or date.min) for r in rows)
    rows = [r for r in rows if (_d(r.get("forecast_date")) or date.min) == max_fd]
    out = {}
    for r in rows:
        f = int(r.get("field_no"))
        gen = str(r.get("generated_at") or r.get("created_at") or "")
        if f not in out or gen >= out[f][0]:
            out[f] = (gen, r)
    return {f: x[1] for f, x in out.items()}


def _run_replay(records, replay_days, by_day):
    field_rows = []
    for dd in replay_days:
        day_rows = sorted(by_day.get(dd, []), key=lambda r: int(r.get("harvest_order") or 99))
        for row in day_rows:
            f = int(row["field_no"])
            rec = next((r for r in records if r["target_day"] == dd and r["field_no"] == f), None)
            if rec is None:
                continue
            train = [r for r in records if r["target_day"] < dd]
            pred = _predict_abc(train, rec, dd)
            if pred is None:
                continue
            field_rows.append({
                "day": dd, "field": f,
                "actual_abc": float(rec["actual_abc"]), "pred_abc": float(pred),
                "growth": float(rec["growth"]), "lag1_abc": float(rec["lag1_abc"]),
                "regime1": float(rec["regime1"]), "regime3": float(rec["regime3"]),
                "dw_rad": float(rec["dw_rad"]), "dw_gdd10": float(rec["dw_gdd10"]),
                "dw_et0": float(rec["dw_et0"]),
            })
    fdf = pd.DataFrame(field_rows)
    if fdf.empty:
        return fdf, pd.DataFrame()
    daily = fdf.groupby("day", as_index=False).agg(
        actual_abc=("actual_abc", "sum"), pred_abc=("pred_abc", "sum"), n=("field", "count")
    )
    daily["expected_n"] = daily["day"].map(lambda d: len(by_day.get(d, [])))
    daily = daily[daily["n"] == daily["expected_n"]].copy()
    daily["err"] = daily["pred_abc"] - daily["actual_abc"]
    daily["abs_err"] = daily["err"].abs()
    daily["ape"] = daily["abs_err"] / daily["actual_abc"].clip(lower=0.1)
    return fdf, daily


def _find_future_plan(plans, today: date):
    if not isinstance(plans, dict):
        return None
    for key in sorted(plans.keys()):
        dd = _d(key)
        if dd is None or dd < today:
            continue
        try:
            fields = [int(x) for x in plans[key]]
        except Exception:
            continue
        if fields:
            return dd, fields
    return None


def _self_test() -> None:
    """Väike lokaalne test ilma Streamliti/Supabase'ita."""
    # 28 päeva täielikku sünteetilist ilma.
    today = date(2026, 8, 18)
    weather = []
    for i in range(40):
        dd = today - timedelta(days=39-i)
        weather.append({
            "weather_date": dd.isoformat(), "data_kind": "measured", "checked": True,
            "temp_night_avg_c": 15 + 0.03*i, "temp_day_avg_c": 22 + 0.04*i,
            "temp_min_c": 12, "temp_max_c": 25, "wind_avg_ms": 2.0,
            "radiation_mj_m2": 18 + 0.1*i, "humidity_avg_pct": 75,
            "precipitation_mm": 0.5, "et0_mm": 3.0,
        })
    wmap = _build_weather_map(weather, today)
    assert len(wmap) == 40

    # Üks põld, 4 korjet; helperite järjepidevus.
    rows = []
    for j, dd in enumerate([date(2026,7,30), date(2026,8,3), date(2026,8,8), date(2026,8,13)]):
        rows.append({"harvest_date": dd.isoformat(), "field_no": 1, "harvest_order": 1,
                     "a": 0.2, "b": 3+j, "c": 4+j, "xl": 1.0, "data_quality": "täpne"})
    _, by_field, by_day = _prepare_harvests(rows)
    plans = {}
    cdays = _complete_days(by_day, plans)
    # Pole 3-põllu päevi, seega režiim null, kuid record peab tekkima.
    rec = _build_record(field=1, target_day=date(2026,8,13), cur_row=by_day[date(2026,8,13)][0],
                        cur_order=1, by_field=by_field, by_day=by_day, complete_days=cdays,
                        wmap=wmap, historical_actual=True)
    assert rec is not None and rec["lag1_abc"] > 0 and math.isfinite(rec["y"])
    assert all(k in rec for k in FEATURES)

    # Ridge numbriline test 24 sünteetilise kirje peal.
    fake_train = []
    for k in range(24):
        rr = dict(rec)
        rr["target_day"] = date(2026,7,20) + timedelta(days=k)
        rr["y"] = 0.01 * math.sin(k)
        for c in FEATURES:
            rr[c] = float(rr[c]) + 0.001*k
        fake_train.append(rr)
    pred = _predict_abc(fake_train, rec, date(2026,8,18))
    assert pred is not None and math.isfinite(pred) and pred >= 0
    print("LAB-135 SELF-TEST OK")


def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-135", layout="wide")
    st.error("🧪 LAB-135 AKTIIVNE · READ-ONLY · üks HYBRID testmudel")
    st.title("KurgiMootor · LAB-135 HYBRID")
    st.caption(
        "Üks fikseeritud mudel: viimased 1–3 sama põllu A+B+C korjet + viimaste täielike korjepäevade režiim + "
        "täpne kasvuaeg + kasvuperioodi ilm + ilmamuutus. Mudel ei vali endale viimase 14 päeva järgi sobivat varianti."
    )

    @st.cache_data(ttl=60, show_spinner=False)
    def _load_data():
        harvests = db.get_harvest_history(limit=4000)
        plans = db.get_harvest_plans()
        weather = db.get_weather_rows(SEASON_START - timedelta(days=21), TODAY + timedelta(days=9))
        try:
            forecasts = db.get_yield_forecasts(limit=12000) if db.yield_forecasts_available() else []
        except Exception:
            forecasts = []
        return harvests, plans, weather, forecasts

    if st.button("Värskenda andmed", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Loen ainult saagi-, ilma-, plaani- ja prognoosiajaloo…"):
        harvest_raw, plans, weather_raw, forecast_rows = _load_data()

    _, by_field, by_day = _prepare_harvests(harvest_raw)
    complete_days = _complete_days(by_day, plans)
    wmap = _build_weather_map(weather_raw, TODAY)
    records = _make_historical_records(by_field, by_day, complete_days, wmap)

    if len(records) < MIN_TRAIN_ROWS + 6:
        st.error(f"Mudeli jaoks on liiga vähe täielikke õppimisridu: {len(records)}. Vaja vähemalt {MIN_TRAIN_ROWS + 6}.")
        st.stop()

    replay_days = complete_days[-14:]
    if len(replay_days) < 7:
        st.error("Viimase ausa replay jaoks on vaja vähemalt 7 täielikku korjepäeva.")
        st.stop()

    fdf, daily = _run_replay(records, replay_days, by_day)
    if daily.empty:
        st.error("Replay ei saanud ühtegi täielikku päeva prognoosida. Tõenäoline põhjus on puuduv ilmaaken.")
        st.stop()

    actual_fields_by_day = {d: {int(r["field_no"]) for r in rows} for d, rows in by_day.items()}
    official_lookup = _official_1d_lookup(forecast_rows, actual_fields_by_day)

    off_rows = []
    for dd in daily["day"]:
        abc_sum = 0.0
        have_all = True
        for hr in by_day.get(dd, []):
            fr = official_lookup.get((dd, int(hr["field_no"])))
            if fr is None or _f(fr.get("abc_forecast")) is None:
                have_all = False
                break
            abc_sum += float(fr["abc_forecast"])
        off_rows.append({"day": dd, "official_abc": abc_sum if have_all else np.nan})
    daily = daily.merge(pd.DataFrame(off_rows), on="day", how="left")
    daily["official_ape"] = (daily["official_abc"] - daily["actual_abc"]).abs() / daily["actual_abc"].clip(lower=0.1)

    last7 = daily.tail(min(7, len(daily))).copy()
    last3 = daily.tail(min(3, len(daily))).copy()
    off7 = last7[pd.notna(last7["official_ape"])].copy()

    st.markdown("### 1. Aus walk-forward · viimased korjepäevad")
    st.caption(
        "Iga päeva prognoos treenitakse ainult sellele päevale EELNENUD korjetel ja ilmal. "
        "Sihtpäeva tegelik saak ega sihtpäeva mõõdetud ilm ei lähe sisendisse."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HYBRID · 14p MAPE", f"{100*daily['ape'].mean():.1f}%")
    c2.metric("HYBRID · viimased 7p", f"{100*last7['ape'].mean():.1f}%")
    c3.metric("Viimased 7p · ±20%", f"{100*(last7['ape'] <= 0.20).mean():.0f}%")
    c4.metric("Ametlik 1p · samad 7p", "—" if off7.empty else f"{100*off7['official_ape'].mean():.1f}%")

    if float(last7["ape"].mean()) <= 0.15:
        st.success("✅ Viimase 7 päeva keskmine päevaviga ≤15% — väga tugev varimootori kandidaat.")
    elif float(last7["ape"].mean()) <= 0.20:
        st.success("✅ Viimase 7 päeva keskmine päevaviga ≤20% — praktiliselt hea kandidaat tulevikutestiks.")
    elif float(last7["ape"].mean()) <= 0.25:
        st.warning("🟡 Viimase 7 päeva viga 20–25% — lubav, kuid mitte veel piisavalt hea.")
    else:
        st.error("🔴 Viimase 7 päeva viga >25% — seda versiooni ei tasu productionisse viia.")

    show = daily.copy()
    show["Kuupäev"] = show["day"].map(lambda d: d.strftime("%d.%m"))
    show["Tegelik ABC"] = show["actual_abc"]
    show["HYBRID ABC"] = show["pred_abc"]
    show["HYBRID viga %"] = 100 * show["ape"]
    show["Ametlik 1p ABC"] = show["official_abc"]
    show["Ametlik viga %"] = 100 * show["official_ape"]
    st.dataframe(
        show[["Kuupäev", "Tegelik ABC", "HYBRID ABC", "HYBRID viga %", "Ametlik 1p ABC", "Ametlik viga %"]].style.format({
            "Tegelik ABC": "{:.1f}", "HYBRID ABC": "{:.1f}", "HYBRID viga %": "{:.1f}%",
            "Ametlik 1p ABC": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            "Ametlik viga %": lambda x: "—" if pd.isna(x) else f"{x:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Vaata põldude kaupa, mida mudel nägi"):
        ff = fdf.copy()
        ff["Kuupäev"] = ff["day"].map(lambda d: d.strftime("%d.%m"))
        ff["Tegelik ABC"] = ff["actual_abc"]
        ff["HYBRID ABC"] = ff["pred_abc"]
        ff["Eelmine ABC"] = ff["lag1_abc"]
        ff["Kasvuaeg p"] = ff["growth"]
        ff["Režiim 1p %"] = 100 * (np.exp(ff["regime1"]) - 1.0)
        ff["Režiim 3p %"] = 100 * (np.exp(ff["regime3"]) - 1.0)
        ff["Δ radiatsioon"] = ff["dw_rad"]
        ff["Δ GDD10"] = ff["dw_gdd10"]
        ff["Δ ET0"] = ff["dw_et0"]
        st.dataframe(
            ff[["Kuupäev", "field", "Tegelik ABC", "HYBRID ABC", "Eelmine ABC", "Kasvuaeg p",
                "Režiim 1p %", "Režiim 3p %", "Δ radiatsioon", "Δ GDD10", "Δ ET0"]].rename(columns={"field": "Põld"}).style.format({
                    "Tegelik ABC": "{:.1f}", "HYBRID ABC": "{:.1f}", "Eelmine ABC": "{:.1f}", "Kasvuaeg p": "{:.2f}",
                    "Režiim 1p %": "{:+.1f}%", "Režiim 3p %": "{:+.1f}%",
                    "Δ radiatsioon": "{:+.2f}", "Δ GDD10": "{:+.2f}", "Δ ET0": "{:+.2f}",
                }), use_container_width=True, hide_index=True,
        )

    st.markdown("### 2. Tuleviku variprognoos")
    st.caption(
        "Sama fikseeritud HYBRID-mudel. ABC tuleb testmudelist. XL-i LAB ei õpi ümber; kui ametlik XL on olemas, "
        "liidame selle ainult kogusaagi orientiiriks. Midagi ei salvestata productionisse."
    )

    future_plan = _find_future_plan(plans, TODAY)
    if future_plan is None:
        st.info("Tänasele või tulevasele päevale pole korjeplaani. Sisesta plaan production-appis ja värskenda siin.")
    else:
        target_day, fields = future_plan
        train = [r for r in records if r["target_day"] < target_day]
        off_future = _latest_official_for_target(forecast_rows, target_day, fields)
        out = []
        for order, field in enumerate(fields, start=1):
            fake = {"field_no": field, "harvest_order": order}
            rec = _build_record(
                field=field, target_day=target_day, cur_row=fake, cur_order=order,
                by_field=by_field, by_day=by_day, complete_days=complete_days,
                wmap=wmap, historical_actual=False,
            )
            if rec is None:
                out.append({"Põld": field, "Viga": "puuduv ajalugu või ilmaaken"})
                continue
            pred = _predict_abc(train, rec, target_day)
            fr = off_future.get(field)
            off_abc = _f(fr.get("abc_forecast")) if fr else None
            off_xl = _f(fr.get("xl_forecast")) if fr else None
            out.append({
                "Põld": field,
                "Kasvuaeg p": rec["growth"],
                "Viimane ABC": rec["lag1_abc"],
                "2. eelmine ABC": rec["lag2_abc"],
                "3. eelmine ABC": rec["lag3_abc"],
                "HYBRID ABC": pred,
                "Ametlik ABC": off_abc,
                "Ametlik XL": off_xl,
                "HYBRID kokku + ametlik XL": (pred + off_xl) if pred is not None and off_xl is not None else None,
                "Režiim 1p %": 100 * (math.exp(rec["regime1"]) - 1.0),
                "Režiim 3p %": 100 * (math.exp(rec["regime3"]) - 1.0),
                "Δ rad": rec["dw_rad"], "Δ GDD10": rec["dw_gdd10"], "Δ ET0": rec["dw_et0"],
                "Viga": "",
            })
        fut = pd.DataFrame(out)
        st.markdown(f"#### {target_day.strftime('%d.%m.%Y')} · põllud {', '.join(map(str, fields))}")
        valid = fut[pd.notna(fut.get("HYBRID ABC"))] if "HYBRID ABC" in fut.columns else pd.DataFrame()
        m1, m2, m3 = st.columns(3)
        m1.metric("HYBRID A+B+C", "—" if valid.empty else f"{valid['HYBRID ABC'].sum():.1f} kasti")
        if "Ametlik ABC" in fut.columns and fut["Ametlik ABC"].notna().all():
            m2.metric("Ametlik A+B+C", f"{fut['Ametlik ABC'].sum():.1f} kasti")
        else:
            m2.metric("Ametlik A+B+C", "—")
        if "HYBRID kokku + ametlik XL" in fut.columns and fut["HYBRID kokku + ametlik XL"].notna().all():
            m3.metric("HYBRID kokku + ametlik XL", f"{fut['HYBRID kokku + ametlik XL'].sum():.1f} kasti")
        else:
            m3.metric("HYBRID kokku + ametlik XL", "—")

        formatters = {
            "Kasvuaeg p": "{:.2f}", "Viimane ABC": "{:.1f}", "2. eelmine ABC": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            "3. eelmine ABC": lambda x: "—" if pd.isna(x) else f"{x:.1f}", "HYBRID ABC": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            "Ametlik ABC": lambda x: "—" if pd.isna(x) else f"{x:.1f}", "Ametlik XL": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            "HYBRID kokku + ametlik XL": lambda x: "—" if pd.isna(x) else f"{x:.1f}",
            "Režiim 1p %": "{:+.1f}%", "Režiim 3p %": "{:+.1f}%", "Δ rad": "{:+.2f}", "Δ GDD10": "{:+.2f}", "Δ ET0": "{:+.2f}",
        }
        cols = [c for c in ["Põld", "Kasvuaeg p", "Viimane ABC", "2. eelmine ABC", "3. eelmine ABC", "HYBRID ABC",
                               "Ametlik ABC", "Ametlik XL", "HYBRID kokku + ametlik XL", "Režiim 1p %", "Režiim 3p %",
                               "Δ rad", "Δ GDD10", "Δ ET0", "Viga"] if c in fut.columns]
        st.dataframe(fut[cols].style.format({k: v for k, v in formatters.items() if k in fut.columns}), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"{LAB_VERSION} · READ ONLY · üks fikseeritud mudel · ridge α={RIDGE_ALPHA:.0f} · recency half-life={RECENCY_HALFLIFE_DAYS:.0f}p. "
        "Siht = log(ABC_nüüd / ABC_viimane_sama_põld). Eelmised 1–3 sama põllu korjet on eraldi tunnused; "
        "kogu põlluploki viimased täielikud korjepäevad annavad värske režiimisignaali; muutuse õpib kasvuaeg + ilm."
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
