from __future__ import annotations

"""
KurgiMootor LAB-157 · THERMAL COHORT GATE
==========================================

Deployment note
---------------
The filename is intentionally kept as KurgiMootor_LAB156_order13_edge_weather.py so the
existing temporary Streamlit app can be updated without changing its Main file path.
The CONTENT is a clean LAB-157; none of the old LAB-156 sections are executed here.

Question
--------
Does cucumber yield become more predictable when source/stress weather is aligned to a
fruit's THERMAL AGE rather than to ordinary calendar days before harvest?

Why this test exists
--------------------
Cucumber fruit development is closely related to temperature sum, and individual fruit
sink strength depends on thermal age after anthesis. Commercial cucumbers are commonly
harvested during the rapid growth phase roughly 7–14 days after anthesis. We do NOT know
anthesis dates for individual fruits, so this LAB does not invent them. Instead it asks a
smaller falsifiable question: if we align the same radiation and WIND×DRY exposure to
fixed backward GDD10 bands, is strict walk-forward prediction better than aligning the
same exposures to fixed calendar-day bands?

Locked design (chosen before seeing results)
---------------------------------------------
Target:
    A+B+C growth rate for each field = ABC / exact harvest-to-harvest growth days.
    XL is excluded because it also reflects harvest timing / overgrowth.

No previous-yield anchor:
    Previous ABC, previous rate, plant index, residual carry and future actual yield are
    NOT model inputs. Historical yield is used only as training target.

Controls in every model:
    field identity + harvest order + exact growth days + smooth season age (linear + sq).

Models:
    BASE      : controls only.
    CALENDAR  : BASE + radiation sum and WIND×DRY load in fixed lags 1–3, 4–6, 7–10 d.
    THERMAL   : BASE + the same two quantities aligned to backward GDD10 bands
                0–25, 25–50, 50–75 °C·d.

Thermal bins:
    GDD10 = max(0, daily mean temperature - 10°C).
    We walk backward from target-1 day. A weather day that crosses a GDD boundary is
    split fractionally between adjacent thermal bands. A <=10°C day does not advance
    thermal age but its source/stress exposure remains in the current band.

Evaluation:
    - strict date-wise walk-forward; target day is trained only on earlier target days;
    - all three models are evaluated on the SAME feature-complete rows;
    - no archived ECMWF replay and no hourly API: measured daily DB only;
    - primary business metric = 3-field full-day ABC MAE;
    - recent-5 full-day MAE and wave direction are shown as diagnostics.

Gate:
    THERMAL is allowed to continue only if it beats CALENDAR in field MAE, full-day MAE,
    recent-5 full-day MAE, AND wins >50% of full days; it must also beat BASE on full-day
    MAE. Otherwise we stop the cohort branch rather than tune GDD bands after the fact.

References behind the mechanism (not parameter fitting):
    Marcelis & Hofman-Eijer 1993, Physiologia Plantarum 87:321–328,
      doi:10.1111/j.1399-3054.1993.tb01737.x
    Marcelis 1994, Annals of Botany 74:43–52, doi:10.1093/aob/74.1.43
    Cucumber fruit developmental reviews: commercial harvest typically ~7–14 DAA.

READ ONLY. Does not write to Supabase and does not change production forecasts.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import math
import sys

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# LOCKED CONFIG
# -----------------------------------------------------------------------------

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-157-THERMAL-COHORT-GATE-V1"

WEATHER_START = date(2026, 7, 1)
SEASON_START = date(2026, 6, 15)
HOURS_PER_FIELD = 3.5
TARGET_EPS = 0.20
RIDGE_ALPHA = 24.0
MIN_TRAIN_ROWS = 24
MAX_THERMAL_LOOKBACK_DAYS = 18
THERMAL_TOTAL_GDD = 75.0

CALENDAR_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("1_3", 1, 3),
    ("4_6", 4, 6),
    ("7_10", 7, 10),
)
THERMAL_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_75", 50.0, 75.0),
)

BASE_CONT = ["season_day", "season_day_sq", "growth_days"]
BASE_BINARY = ["order2", "order3"] + [f"field_{i}" for i in range(2, 15)]
CAL_EXTRA = [f"cal_rad_{name}" for name, _, _ in CALENDAR_BANDS] + [
    f"cal_wd_{name}" for name, _, _ in CALENDAR_BANDS
]
THERM_EXTRA = [f"th_rad_{name}" for name, _, _ in THERMAL_BANDS] + [
    f"th_wd_{name}" for name, _, _ in THERMAL_BANDS
]
MODEL_FEATURES: Dict[str, Tuple[List[str], List[str]]] = {
    "BASE": (BASE_CONT, BASE_BINARY),
    "CALENDAR": (BASE_CONT + CAL_EXTRA, BASE_BINARY),
    "THERMAL": (BASE_CONT + THERM_EXTRA, BASE_BINARY),
}
ALL_READY_COLS = sorted(set(BASE_CONT + BASE_BINARY + CAL_EXTRA + THERM_EXTRA + ["y", "actual_abc", "growth_days"]))


# -----------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    day: date
    field: int
    order: int
    abc: float


def _d(v) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
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


def _confirmed(v) -> bool:
    return str(v or "").strip().lower() == "kinnitatud"


def _abc(row: dict) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        return None
    out = float(sum(vals))
    return out if out >= 0 else None


def _prepare_events(rows: Iterable[dict]) -> List[Event]:
    out: List[Event] = []
    for r in rows:
        if not _confirmed(r.get("data_quality")):
            continue
        dd = _d(r.get("harvest_date"))
        ff = _f(r.get("field_no"))
        oo = _f(r.get("harvest_order"))
        aa = _abc(r)
        if dd is None or ff is None or oo is None or aa is None:
            continue
        field = int(ff)
        order = int(oo)
        if not (1 <= field <= 14 and 1 <= order <= 3):
            continue
        out.append(Event(dd, field, order, aa))
    # de-duplicate exact day+field by keeping the last sorted record
    dedup: Dict[Tuple[date, int], Event] = {}
    for e in sorted(out, key=lambda x: (x.day, x.order, x.field)):
        dedup[(e.day, e.field)] = e
    return sorted(dedup.values(), key=lambda x: (x.day, x.order, x.field))


def _weather_map(rows: Iterable[dict]) -> Dict[date, dict]:
    out: Dict[date, dict] = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None or dd < WEATHER_START:
            continue
        if str(r.get("data_kind") or "").lower() != "measured" or not bool(r.get("checked")):
            continue
        night = _f(r.get("temp_night_avg_c"))
        dayt = _f(r.get("temp_day_avg_c"))
        tmin = _f(r.get("temp_min_c"))
        tmax = _f(r.get("temp_max_c"))
        rad = _f(r.get("radiation_mj_m2"))
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))
        if rad is None or wind is None or rh is None:
            continue
        if night is not None and dayt is not None:
            temp = 0.5 * (night + dayt)
        elif tmin is not None and tmax is not None:
            temp = 0.5 * (tmin + tmax)
        else:
            continue
        gdd10 = max(0.0, temp - 10.0)
        wd = max(0.0, wind) * max(0.0, 100.0 - rh)
        out[dd] = {
            "temp": float(temp),
            "gdd10": float(gdd10),
            "rad": float(max(0.0, rad)),
            "wd": float(wd),
        }
    return out


def _growth_days(prev: Event, cur: Event) -> float:
    days = float((cur.day - prev.day).days) + (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, days)


def _calendar_features(target: date, weather: Dict[date, dict]) -> Optional[dict]:
    rec: dict = {}
    for name, lo, hi in CALENDAR_BANDS:
        rad_sum = 0.0
        wd_sum = 0.0
        for lag in range(lo, hi + 1):
            rr = weather.get(target - timedelta(days=lag))
            if rr is None:
                return None
            rad_sum += float(rr["rad"])
            wd_sum += float(rr["wd"])
        rec[f"cal_rad_{name}"] = rad_sum
        rec[f"cal_wd_{name}"] = wd_sum
    return rec


def _band_name_at(age: float) -> Optional[str]:
    # age is cumulative backward GDD before consuming the current day.
    # Use the upper band for an exact interior boundary to keep bins non-overlapping.
    for name, lo, hi in THERMAL_BANDS:
        if lo <= age < hi or (age == 0 and lo == 0):
            return name
    return None


def _thermal_features(target: date, weather: Dict[date, dict]) -> Optional[dict]:
    rad = {name: 0.0 for name, _, _ in THERMAL_BANDS}
    wd = {name: 0.0 for name, _, _ in THERMAL_BANDS}
    exposure_days = {name: 0.0 for name, _, _ in THERMAL_BANDS}

    thermal_age = 0.0
    for lag in range(1, MAX_THERMAL_LOOKBACK_DAYS + 1):
        rr = weather.get(target - timedelta(days=lag))
        if rr is None:
            return None
        g = max(0.0, float(rr["gdd10"]))

        if thermal_age >= THERMAL_TOTAL_GDD:
            break

        if g <= 1e-12:
            name = _band_name_at(thermal_age)
            if name is not None:
                rad[name] += float(rr["rad"])
                wd[name] += float(rr["wd"])
                exposure_days[name] += 1.0
            continue

        day_start = thermal_age
        day_end = thermal_age + g
        for name, lo, hi in THERMAL_BANDS:
            overlap = max(0.0, min(day_end, hi) - max(day_start, lo))
            if overlap <= 0:
                continue
            frac = overlap / g
            rad[name] += float(rr["rad"]) * frac
            wd[name] += float(rr["wd"]) * frac
            exposure_days[name] += frac
        thermal_age = day_end

    if thermal_age + 1e-9 < THERMAL_TOTAL_GDD:
        return None
    if any(exposure_days[name] <= 0 for name, _, _ in THERMAL_BANDS):
        return None

    rec: dict = {}
    for name, _, _ in THERMAL_BANDS:
        rec[f"th_rad_{name}"] = float(rad[name])
        rec[f"th_wd_{name}"] = float(wd[name])
        # Duration is intentionally NOT a model feature in V1. It is only retained for audit.
        rec[f"th_days_{name}"] = float(exposure_days[name])
    rec["th_calendar_days"] = float(sum(exposure_days.values()))
    return rec


def _build_records(events: List[Event], weather: Dict[date, dict]) -> pd.DataFrame:
    by_field: Dict[int, List[Event]] = {i: [] for i in range(1, 15)}
    for e in events:
        by_field[e.field].append(e)
    for f in by_field:
        by_field[f].sort(key=lambda e: (e.day, e.order))

    rows: List[dict] = []
    for field, hist in by_field.items():
        for i in range(1, len(hist)):
            prev, cur = hist[i - 1], hist[i]
            growth = _growth_days(prev, cur)
            rate = cur.abc / growth
            if rate <= 0:
                continue
            cf = _calendar_features(cur.day, weather)
            tf = _thermal_features(cur.day, weather)
            if cf is None or tf is None:
                continue

            season_day = float((cur.day - SEASON_START).days)
            rec = {
                "date": cur.day,
                "field": field,
                "order": cur.order,
                "actual_abc": float(cur.abc),
                "growth_days": float(growth),
                "rate": float(rate),
                "y": float(math.log(rate + TARGET_EPS)),
                "season_day": season_day,
                "season_day_sq": (season_day / 60.0) ** 2,
                "order2": 1.0 if cur.order == 2 else 0.0,
                "order3": 1.0 if cur.order == 3 else 0.0,
            }
            for ff in range(2, 15):
                rec[f"field_{ff}"] = 1.0 if field == ff else 0.0
            rec.update(cf)
            rec.update(tf)
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["date", "order", "field"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# FIXED RIDGE + STRICT WALK-FORWARD
# -----------------------------------------------------------------------------

def _ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cont_cols: Sequence[str],
    binary_cols: Sequence[str],
) -> np.ndarray:
    # Standardize ONLY continuous columns from the current training fold.
    xtr_c = train[list(cont_cols)].to_numpy(dtype=float) if cont_cols else np.empty((len(train), 0))
    xte_c = test[list(cont_cols)].to_numpy(dtype=float) if cont_cols else np.empty((len(test), 0))
    if cont_cols:
        mu = np.mean(xtr_c, axis=0)
        sd = np.std(xtr_c, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        xtr_c = (xtr_c - mu) / sd
        xte_c = (xte_c - mu) / sd

    xtr_b = train[list(binary_cols)].to_numpy(dtype=float) if binary_cols else np.empty((len(train), 0))
    xte_b = test[list(binary_cols)].to_numpy(dtype=float) if binary_cols else np.empty((len(test), 0))
    xtr = np.column_stack([np.ones(len(train)), xtr_c, xtr_b])
    xte = np.column_stack([np.ones(len(test)), xte_c, xte_b])
    y = train["y"].to_numpy(dtype=float)

    reg = np.eye(xtr.shape[1], dtype=float) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtr.T @ xtr + reg) @ (xtr.T @ y)
    return xte @ beta


def _walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in MODEL_FEATURES:
        out[f"pred_{name}"] = np.nan

    for dd in sorted(out["date"].unique()):
        train = out[out["date"] < dd].copy()
        test_idx = out.index[out["date"] == dd]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue
        test = out.loc[test_idx].copy()

        # Every model uses the exact same feature-complete training rows.
        common_cols = sorted(set(ALL_READY_COLS))
        ok_train = np.all(np.isfinite(train[common_cols].to_numpy(dtype=float)), axis=1)
        train = train.loc[ok_train].copy()
        if len(train) < MIN_TRAIN_ROWS:
            continue
        ok_test = np.all(np.isfinite(test[common_cols].to_numpy(dtype=float)), axis=1)
        if not np.all(ok_test):
            continue

        for name, (cont, binary) in MODEL_FEATURES.items():
            pred_y = _ridge_fit_predict(train, test, cont, binary)
            pred_rate = np.maximum(0.0, np.exp(pred_y) - TARGET_EPS)
            pred_abc = pred_rate * test["growth_days"].to_numpy(dtype=float)
            out.loc[test_idx, f"pred_{name}"] = pred_abc

    pred_cols = [f"pred_{n}" for n in MODEL_FEATURES]
    mask = np.all(np.isfinite(out[pred_cols].to_numpy(dtype=float)), axis=1)
    return out.loc[mask].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def _full_days(field_df: pd.DataFrame) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for dd, g in field_df.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3:
            continue
        row = {"date": dd, "actual": float(g["actual_abc"].sum())}
        for name in MODEL_FEATURES:
            row[name] = float(g[f"pred_{name}"].sum())
            row[f"{name}_abs_err"] = abs(row[name] - row["actual"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _trend_hit(daily: pd.DataFrame, model: str) -> float:
    if len(daily) < 2:
        return float("nan")
    actual_d = np.diff(daily["actual"].to_numpy(dtype=float))
    pred_d = np.diff(daily[model].to_numpy(dtype=float))
    valid = np.abs(actual_d) > 1e-12
    if not np.any(valid):
        return float("nan")
    return 100.0 * float(np.mean(np.sign(actual_d[valid]) == np.sign(pred_d[valid])))


def _metrics(field_df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    last5_dates = list(daily["date"].tail(5)) if not daily.empty else []
    for name in MODEL_FEATURES:
        field_mae = float(np.mean(np.abs(field_df[f"pred_{name}"] - field_df["actual_abc"]))) if not field_df.empty else np.nan
        day_mae = float(daily[f"{name}_abs_err"].mean()) if not daily.empty else np.nan
        recent5 = float(daily.loc[daily["date"].isin(last5_dates), f"{name}_abs_err"].mean()) if last5_dates else np.nan
        within20 = 100.0 * float(np.mean(daily[f"{name}_abs_err"] <= 0.20 * daily["actual"].clip(lower=0.1))) if not daily.empty else np.nan
        rows.append({
            "Mudel": name,
            "Põllu MAE": field_mae,
            "3-põllu päeva MAE": day_mae,
            "Viimase 5 päeva MAE": recent5,
            "±20% sees %": within20,
            "Lainete suund %": _trend_hit(daily, name),
            "Põlluridu": int(len(field_df)),
            "Täispäevi": int(len(daily)),
        })
    return pd.DataFrame(rows)


def _gate(metrics: pd.DataFrame, daily: pd.DataFrame) -> Tuple[bool, List[str]]:
    mm = metrics.set_index("Mudel")
    reasons: List[str] = []
    if not {"BASE", "CALENDAR", "THERMAL"}.issubset(mm.index):
        return False, ["Mudelite mõõdikud puudulikud."]

    def better(metric: str, a: str, b: str) -> bool:
        av = float(mm.loc[a, metric])
        bv = float(mm.loc[b, metric])
        return math.isfinite(av) and math.isfinite(bv) and av < bv

    c1 = better("Põllu MAE", "THERMAL", "CALENDAR")
    c2 = better("3-põllu päeva MAE", "THERMAL", "CALENDAR")
    c3 = better("Viimase 5 päeva MAE", "THERMAL", "CALENDAR")
    c4 = better("3-põllu päeva MAE", "THERMAL", "BASE")

    if daily.empty:
        win_share = np.nan
        c5 = False
    else:
        t = daily["THERMAL_abs_err"].to_numpy(dtype=float)
        c = daily["CALENDAR_abs_err"].to_numpy(dtype=float)
        win_share = 100.0 * float(np.mean(t < c))
        c5 = win_share > 50.0

    reasons.append(f"Põllu MAE THERMAL < CALENDAR: {'JAH' if c1 else 'EI'}")
    reasons.append(f"Päeva MAE THERMAL < CALENDAR: {'JAH' if c2 else 'EI'}")
    reasons.append(f"Viimase 5 päeva MAE THERMAL < CALENDAR: {'JAH' if c3 else 'EI'}")
    reasons.append(f"Päeva MAE THERMAL < BASE: {'JAH' if c4 else 'EI'}")
    reasons.append(f"THERMAL võidab >50% täispäevi vs CALENDAR: {'JAH' if c5 else 'EI'} ({win_share:.0f}% kui hinnatav)")
    return bool(c1 and c2 and c3 and c4 and c5), reasons


# -----------------------------------------------------------------------------
# SYNTHETIC SELF TEST
# -----------------------------------------------------------------------------

def _self_test() -> None:
    # Thermal bin accounting: warm days should reach 75 GDD and all bins be populated.
    weather: Dict[date, dict] = {}
    start = date(2026, 7, 1)
    for i in range(90):
        dd = start + timedelta(days=i)
        temp = 17.0 + 3.0 * math.sin(i / 5.0)
        gdd = max(0.0, temp - 10.0)
        rad = 15.0 + 7.0 * max(0.0, math.sin((i + 1) / 3.0))
        rh = 78.0 + 10.0 * math.sin(i / 7.0)
        wind = 2.0 + 0.8 * math.cos(i / 4.0)
        weather[dd] = {
            "temp": temp,
            "gdd10": gdd,
            "rad": rad,
            "wd": wind * (100.0 - rh),
        }
    tf = _thermal_features(date(2026, 8, 10), weather)
    assert tf is not None
    assert all(math.isfinite(tf[f"th_rad_{n}"]) for n, _, _ in THERMAL_BANDS)
    assert all(tf[f"th_days_{n}"] > 0 for n, _, _ in THERMAL_BANDS)

    # Build a rotating 3-fields/day synthetic harvest history with a thermal weather signal.
    events: List[Event] = []
    field_last: Dict[int, date] = {}
    d0 = date(2026, 7, 15)
    for day_i in range(48):
        dd = d0 + timedelta(days=day_i)
        fields = [((3 * day_i + j) % 14) + 1 for j in range(3)]
        for order, field in enumerate(fields, start=1):
            prev_day = field_last.get(field)
            if prev_day is None:
                abc = 7.0 + 0.15 * field
            else:
                tf2 = _thermal_features(dd, weather)
                # Synthetic target depends on thermal source and stress. This is ONLY a code smoke-test.
                sig = 0.0
                if tf2 is not None:
                    sig = 0.010 * tf2["th_rad_25_50"] - 0.0012 * tf2["th_wd_25_50"]
                abc = max(1.0, 6.5 + 0.08 * field + sig + 0.4 * math.sin(day_i / 3.0))
            events.append(Event(dd, field, order, abc))
            field_last[field] = dd

    df = _build_records(events, weather)
    assert not df.empty and len(df) > MIN_TRAIN_ROWS
    wf = _walk_forward(df)
    assert not wf.empty
    assert all(np.isfinite(wf[f"pred_{m}"]).all() for m in MODEL_FEATURES)
    daily = _full_days(wf)
    assert not daily.empty
    mt = _metrics(wf, daily)
    assert set(mt["Mudel"]) == set(MODEL_FEATURES)
    print(f"{LAB_VERSION} SELF-TEST OK · records={len(df)} · oos={len(wf)} · days={len(daily)}")


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-157", layout="wide")
    st.error("🧪 LAB-157 · THERMAL COHORT GATE · READ ONLY")
    st.title("Kas me oleme vilja arengut vaadanud vale kellaga?")
    st.caption(
        "Üks puhas test. Ei tunniilma, ei ECMWF replay'd, ei 10 vana LAB-156 haru. "
        "Võrdleme sama radiatsiooni ja WIND×DRY infot kalendripäevade vs termilise vanuse järgi."
    )

    with st.expander("Katse lukud", expanded=False):
        st.markdown(
            f"""
- **Target:** A+B+C / täpne kasvuaeg; XL ei osale.
- **Eelmise saagi ankrut ei ole.** Previous ABC/rate, taimeindeks ja residual carry puuduvad sisenditest.
- **BASE:** põld + order + kasvuaeg + hooaja lineaarne/ruuttrend.
- **CALENDAR:** sama + radiatsioon ja WIND×DRY akendes **1–3, 4–6, 7–10 päeva** enne korjet.
- **THERMAL:** sama info, aga tagasi mõõdetud GDD10 vanuses **0–25, 25–50, 50–75 °C·d**.
- Termilise akna maksimaalne tagasivaade on {MAX_THERMAL_LOOKBACK_DAYS} päeva. Lävendeid pärast tulemust ei muudeta.
- **Strict walk-forward:** target-päev näeb treeningus ainult varasemaid target-päevi; min train {MIN_TRAIN_ROWS} rida.
- Kõik mudelid hinnatakse **samadel ridadel**. Ridge α={RIDGE_ALPHA:g} on fikseeritud.
- Ilm = ainult `weather_daily` kontrollitud mõõdetud read. Target-päeva ilma ei kasutata; viimane ilm on target−1.
- See LAB **ei kirjuta DB-sse** ega muuda productionit.
            """
        )

    st.info(
        "CPU-hoid: siin puudub hourly API ja arhiveeritud prognooside allalaadimine. "
        "Üks nupp teeb ühe väikese strict walk-forward arvutuse."
    )

    if st.button("▶ Jooksuta THERMAL COHORT värav", type="primary"):
        st.session_state.pop("lab157_result", None)
        try:
            harvest_rows = db.get_harvest_history(limit=5000)
            weather_rows = db.get_weather_rows(WEATHER_START, TODAY)
            events = _prepare_events(harvest_rows)
            weather = _weather_map(weather_rows)
            records = _build_records(events, weather)
            if records.empty:
                st.error("Feature-complete ridu ei tekkinud. Kontrolli mõõdetud ilma ajalugu.")
                st.stop()
            wf = _walk_forward(records)
            daily = _full_days(wf)
            if wf.empty or daily.empty:
                st.error(
                    f"Strict OOS jaoks pole veel piisavalt ridu. Feature-ready={len(records)}, "
                    f"min train={MIN_TRAIN_ROWS}."
                )
                st.stop()
            metrics = _metrics(wf, daily)
            passed, reasons = _gate(metrics, daily)
            st.session_state["lab157_result"] = {
                "records": records,
                "wf": wf,
                "daily": daily,
                "metrics": metrics,
                "passed": passed,
                "reasons": reasons,
                "latest_weather": max(weather) if weather else None,
            }
        except Exception as exc:
            st.exception(exc)
            st.stop()

    result = st.session_state.get("lab157_result")
    if not result:
        st.stop()

    records: pd.DataFrame = result["records"]
    wf: pd.DataFrame = result["wf"]
    daily: pd.DataFrame = result["daily"]
    metrics: pd.DataFrame = result["metrics"]
    passed: bool = result["passed"]
    reasons: List[str] = result["reasons"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feature-ready ridu", len(records))
    m2.metric("Strict OOS ridu", len(wf))
    m3.metric("Täispäevi", len(daily))
    lw = result.get("latest_weather")
    m4.metric("Mõõdetud ilm kuni", lw.strftime("%d.%m") if lw else "—")

    st.markdown("### 1. Põhitulemus")
    st.dataframe(
        metrics.style.format({
            "Põllu MAE": "{:.3f}",
            "3-põllu päeva MAE": "{:.3f}",
            "Viimase 5 päeva MAE": "{:.3f}",
            "±20% sees %": "{:.0f}%",
            "Lainete suund %": lambda x: "—" if pd.isna(x) else f"{x:.0f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if passed:
        st.success(
            "✅ THERMAL COHORT VÄRAV PASS. Sama ilmainfo töötab termilise vanuse järgi paremini kui "
            "kalendripäevade järgi ning tulemus peab ka BASE vastu. Järgmine samm võib olla üks väike latentne cohort-mudel."
        )
    else:
        st.warning(
            "⛔ THERMAL COHORT VÄRAV EI LÄINUD LÄBI. Selle V1 GDD-joonduse järgi ei ole põhjust uut cohort-mootorit ehitada. "
            "GDD lävendeid ei hakata selle hooaja tulemuse järgi ümber häälestama."
        )
    for r in reasons:
        st.write("• " + r)

    st.markdown("### 2. Päev-päevalt · viimased kuni 12 täispäeva")
    show = daily.tail(12).copy()
    show["Kuupäev"] = show["date"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    show["Tegelik ABC"] = show["actual"]
    for name in MODEL_FEATURES:
        show[name] = show[name]
        show[f"{name} viga"] = show[name] - show["actual"]
    cols = ["Kuupäev", "Tegelik ABC", "BASE", "CALENDAR", "THERMAL", "BASE viga", "CALENDAR viga", "THERMAL viga"]
    st.dataframe(
        show[cols].style.format({c: "{:.1f}" for c in cols if c not in ("Kuupäev",)}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Kas termiline kell muutis akna tegelikku pikkust?")
    audit = records.tail(18).copy()
    audit["Kuupäev"] = audit["date"].map(lambda d: pd.Timestamp(d).strftime("%d.%m"))
    audit["Põld"] = audit["field"]
    audit["0–25 GDD p"] = audit["th_days_0_25"]
    audit["25–50 GDD p"] = audit["th_days_25_50"]
    audit["50–75 GDD p"] = audit["th_days_50_75"]
    audit["75 GDD kokku p"] = audit["th_calendar_days"]
    st.dataframe(
        audit[["Kuupäev", "Põld", "0–25 GDD p", "25–50 GDD p", "50–75 GDD p", "75 GDD kokku p"]]
        .style.format({
            "0–25 GDD p": "{:.2f}", "25–50 GDD p": "{:.2f}", "50–75 GDD p": "{:.2f}", "75 GDD kokku p": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "See tabel on ainult geomeetria kontroll. Kui 75 GDD võtab eri perioodidel selgelt erineva arvu kalendripäevi, "
        "siis THERMAL ja CALENDAR mudelid vaatavad päriselt erinevalt joondatud viljafaase."
    )

    st.markdown("### 4. Lekkeaudit")
    st.success("✅ Previous yield / previous rate EI ole sisend")
    st.success("✅ Target-päeva ilm EI ole sisend; viimane ilm = target−1")
    st.success("✅ Strict date-wise walk-forward; sama päeva 3 põldu ei õpeta üksteist")
    st.success("✅ BASE, CALENDAR ja THERMAL hinnatakse samadel feature-ready ridadel")
    st.success("✅ Ainult mõõdetud daily DB; hourly/API/ECMWF replay puudub")
    st.info(f"Versioon {LAB_VERSION} · failinimi hoitud vana Streamliti Main path'i pärast")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
