# LAB-156 RH/VPD V5 — 2026-08-23 — threshold-duration + clock-window audit
from __future__ import annotations

"""
KurgiMootor LAB-156 — order 1 vs order 3 edge-weather natural experiment

READ ONLY. Does not write to Supabase and does not change production forecasts.

Question
--------
Each day we harvest three fields, about 3.5 h/field starting at 09:00.
Because there are 14 fields, current order-1 and order-3 fields have heavily
overlapping growth intervals, but different "edge" hours. Can variation in the
weather during only those non-overlapping hours explain variation in the A+B+C
yield difference between order 1 and order 3?

Primary design
--------------
1. Use only confirmed order 1 and order 3 harvest rows. Order 2 is excluded.
2. Use A+B+C only; XL is excluded from the target.
3. Reconstruct harvest midpoint times from the fixed work schedule:
       order1 10:45, order2 14:15, order3 17:45.
4. For each day, calculate the overlap of the two fields' previous->current
   growth intervals and isolate the field-1-only and field-3-only edge windows.
5. Build edge-weather contrasts from official Pärnu hourly data:
   TA, RH, WS10M, PR1H, SDUR1H.
6. Primary inference is SAME-FIELD-PAIR difference-in-differences (DID):
       D(day) = ABC(order1) - ABC(order3)
       ΔD = D(later repeat) - D(earlier repeat)
   and compare it with the corresponding change in edge-weather contrast.
   This removes a large part of the stable field-pair baseline.
7. Secondary target is symmetric relative difference, which is less sensitive
   to whole-season yield-level changes.

This is a discovery audit, not a production model.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
import math
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import db
import core

ESTONIA = ZoneInfo("Europe/Tallinn")
SEASON_START = date(2026, 7, 1)
ORDER_MIDPOINT = {
    1: time(10, 45),
    2: time(14, 15),
    3: time(17, 45),
}
MIN_COMMON_HOURS = 60.0
MAX_REPEAT_GAP_DAYS = 18
MIN_REPEAT_GAP_DAYS = 10

ELEMENTS = ("TA", "RH", "WS10M", "PR1H", "SDUR1H")


@dataclass(frozen=True)
class Segment:
    start: datetime
    end: datetime

    @property
    def hours(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds() / 3600.0)


def _local_dt(day: date, order: int) -> datetime:
    return datetime.combine(day, ORDER_MIDPOINT[int(order)], tzinfo=ESTONIA)


def _confirmed(v: Any) -> bool:
    return str(v or "").strip().lower() == "kinnitatud"


def _num(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def prepare_harvests(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    need = {"harvest_date", "field_no", "harvest_order", "a", "b", "c", "data_quality"}
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f"Korjeandmetest puuduvad veerud: {sorted(missing)}")

    d["harvest_date"] = pd.to_datetime(d["harvest_date"], errors="coerce").dt.date
    for c in ("field_no", "harvest_order", "a", "b", "c", "xl", "total"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["data_quality"].map(_confirmed)].copy()
    d = d[d[["harvest_date", "field_no", "harvest_order", "a", "b", "c"]].notna().all(axis=1)].copy()
    d["field_no"] = d["field_no"].astype(int)
    d["harvest_order"] = d["harvest_order"].astype(int)
    d["ABC"] = d["a"] + d["b"] + d["c"]
    d = d.sort_values(["field_no", "harvest_date", "harvest_order"]).reset_index(drop=True)

    # Previous confirmed harvest of the same field. No previous yield enters any feature.
    d["prev_date"] = d.groupby("field_no")["harvest_date"].shift(1)
    d["prev_order"] = d.groupby("field_no")["harvest_order"].shift(1)
    return d


def build_order13_days(d: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    by_day = d.sort_values(["harvest_date", "harvest_order"]).groupby("harvest_date", sort=True)

    for day, g in by_day:
        g1 = g[g["harvest_order"] == 1]
        g3 = g[g["harvest_order"] == 3]
        if len(g1) != 1 or len(g3) != 1:
            continue
        r1 = g1.iloc[0]
        r3 = g3.iloc[0]
        if pd.isna(r1["prev_date"]) or pd.isna(r3["prev_date"]) or pd.isna(r1["prev_order"]) or pd.isna(r3["prev_order"]):
            continue

        p1o, p3o = int(r1["prev_order"]), int(r3["prev_order"])
        # Normal 14-field rotation: current order1 came from previous order2;
        # current order3 came from previous order1. Keep only this clean geometry.
        if not (p1o == 2 and p3o == 1):
            continue

        s1 = _local_dt(r1["prev_date"], p1o)
        e1 = _local_dt(day, 1)
        s3 = _local_dt(r3["prev_date"], p3o)
        e3 = _local_dt(day, 3)
        if not (s1 < e1 and s3 < e3):
            continue

        common_start = max(s1, s3)
        common_end = min(e1, e3)
        common_h = max(0.0, (common_end - common_start).total_seconds() / 3600.0)
        if common_h < MIN_COMMON_HOURS:
            continue

        # General interval difference. In normal geometry these are one segment each.
        f1_only: List[Segment] = []
        f3_only: List[Segment] = []
        if s1 < common_start:
            f1_only.append(Segment(s1, common_start))
        if common_end < e1:
            f1_only.append(Segment(common_end, e1))
        if s3 < common_start:
            f3_only.append(Segment(s3, common_start))
        if common_end < e3:
            f3_only.append(Segment(common_end, e3))

        abc1, abc3 = float(r1["ABC"]), float(r3["ABC"])
        avg = (abc1 + abc3) / 2.0
        sym = np.nan if avg <= 0 else (abc1 - abc3) / avg
        recs.append({
            "day": day,
            "field1": int(r1["field_no"]),
            "field3": int(r3["field_no"]),
            "pair_key": f"{int(r1['field_no'])}-{int(r3['field_no'])}",
            "ABC1": abc1,
            "ABC3": abc3,
            "D_boxes": abc1 - abc3,
            "D_rel": sym,
            "start1": s1,
            "end1": e1,
            "start3": s3,
            "end3": e3,
            "common_h": common_h,
            "f1_only_h": sum(x.hours for x in f1_only),
            "f3_only_h": sum(x.hours for x in f3_only),
            "net_h": sum(x.hours for x in f1_only) - sum(x.hours for x in f3_only),
            "f1_segments": f1_only,
            "f3_segments": f3_only,
        })

    return pd.DataFrame(recs).sort_values("day").reset_index(drop=True) if recs else pd.DataFrame()


def _row_local_dt(row: Dict[str, Any]) -> datetime | None:
    try:
        utc_dt = datetime(
            int(row["aasta"]), int(row["kuu"]), int(row["paev"]), int(row.get("tund") or 0),
            tzinfo=timezone.utc,
        )
        return utc_dt.astimezone(ESTONIA)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hourly(start_iso: str, end_iso: str) -> pd.DataFrame:
    start_day = date.fromisoformat(start_iso)
    end_day = date.fromisoformat(end_iso)
    svc = core.WeatherService()
    all_rows: List[Dict[str, Any]] = []
    # One extra UTC calendar day at both ends protects local-time conversion.
    qstart = start_day - timedelta(days=1)
    qend = end_day + timedelta(days=1)
    for code in ELEMENTS:
        rows = svc._official_rows(core.OFFICIAL_HOURLY, "Pärnu", code, qstart, qend)
        for r in rows:
            dt = _row_local_dt(r)
            val = _num(r.get("vaartus"))
            if dt is not None and np.isfinite(val):
                all_rows.append({"dt": dt, "code": code, "value": val})
    if not all_rows:
        return pd.DataFrame(columns=["dt", *ELEMENTS])
    x = pd.DataFrame(all_rows)
    p = x.pivot_table(index="dt", columns="code", values="value", aggfunc="mean").sort_index()
    p = p.reindex(columns=list(ELEMENTS))
    p = p.reset_index()
    return p


def _state_series(hourly: pd.DataFrame, code: str) -> pd.Series:
    if code not in hourly.columns:
        return pd.Series(dtype=float)
    s = hourly.set_index("dt")[code].dropna().sort_index()
    if s.empty:
        return s
    # Interpolate instantaneous/state variables to 15-minute grid.
    idx = pd.date_range(s.index.min(), s.index.max(), freq="15min", tz=ESTONIA)
    s2 = s.reindex(s.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)
    return s2


def _integral_state(series: pd.Series, segments: List[Segment], transform=lambda x: x) -> float:
    if series.empty:
        return np.nan
    total = 0.0
    found = False
    for seg in segments:
        # quarter-hour midpoint-ish samples; duration weight 0.25h
        mask = (series.index >= seg.start) & (series.index < seg.end)
        vals = series.loc[mask].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            total += float(np.sum(transform(vals)) * 0.25)
            found = True
    return total if found else np.nan


def _mean_state(series: pd.Series, segments: List[Segment], transform=lambda x: x) -> float:
    vals: List[float] = []
    for seg in segments:
        mask = (series.index >= seg.start) & (series.index < seg.end)
        arr = series.loc[mask].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        vals.extend(transform(arr).tolist())
    return float(np.mean(vals)) if vals else np.nan


def _clock_is_night_morning(idx: pd.DatetimeIndex) -> np.ndarray:
    """Pre-specified clock split, independent of yield: 20:00–10:00 vs 10:00–20:00."""
    mins = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    return (mins >= 20 * 60) | (mins < 10 * 60)


def _integral_state_clock(
    series: pd.Series,
    segments: List[Segment],
    transform,
    night_morning: bool,
) -> float:
    """15-min state integral restricted to a fixed clock window.

    Night/morning = 20:00–10:00. Day/evening = 10:00–20:00.
    Returns 0 when the edge is covered by weather data but contains no time in
    the selected clock window; returns NaN only when the edge itself lacks data.
    """
    if series.empty:
        return np.nan
    total = 0.0
    covered = False
    for seg in segments:
        base = (series.index >= seg.start) & (series.index < seg.end)
        if not bool(np.any(base)):
            continue
        vals_all = series.loc[base]
        vals_all = vals_all[np.isfinite(vals_all.to_numpy(dtype=float))]
        if vals_all.empty:
            continue
        covered = True
        nm = _clock_is_night_morning(vals_all.index)
        keep = nm if night_morning else ~nm
        vals = vals_all.to_numpy(dtype=float)[keep]
        if len(vals):
            total += float(np.sum(transform(vals)) * 0.25)
    return total if covered else np.nan


def _mean_state_clock(
    series: pd.Series,
    segments: List[Segment],
    night_morning: bool,
) -> float:
    """Mean state value inside fixed 20:00–10:00 or 10:00–20:00 clock window.

    Unlike duration features, this does not reward the longer F1 edge merely for
    containing more clock-hours. It asks whether the selected edge-hours were
    warmer/cooler on average.
    """
    if series.empty:
        return np.nan
    vals_keep: List[float] = []
    covered = False
    for seg in segments:
        base = (series.index >= seg.start) & (series.index < seg.end)
        if not bool(np.any(base)):
            continue
        vals_all = series.loc[base]
        vals_all = vals_all[np.isfinite(vals_all.to_numpy(dtype=float))]
        if vals_all.empty:
            continue
        covered = True
        nm = _clock_is_night_morning(vals_all.index)
        keep = nm if night_morning else ~nm
        vals = vals_all.to_numpy(dtype=float)[keep]
        if len(vals):
            vals_keep.extend(vals.tolist())
    if vals_keep:
        return float(np.mean(vals_keep))
    return np.nan if not covered else np.nan


def _integral_pair_product(a: pd.Series, b: pd.Series, segments: List[Segment]) -> float:
    if a.empty or b.empty:
        return np.nan
    z = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    total = 0.0
    found = False
    for seg in segments:
        m = (z.index >= seg.start) & (z.index < seg.end)
        q = z.loc[m]
        if len(q):
            total += float(np.sum(q["a"].to_numpy() * q["b"].to_numpy()) * 0.25)
            found = True
    return total if found else np.nan


def _hourly_accum(hourly: pd.DataFrame, code: str, segments: List[Segment], sunshine=False) -> float:
    """Integrate hourly accumulations over arbitrary edge windows.

    PR1H/SDUR1H are treated as accumulation for the hour ending at the timestamp.
    Partial boundary hours are allocated in proportion to overlap.
    SDUR1H is minutes -> returned as sunshine hours.
    """
    if code not in hourly.columns:
        return np.nan
    x = hourly[["dt", code]].dropna().copy()
    if x.empty:
        return np.nan
    total = 0.0
    found = False
    for _, row in x.iterrows():
        end = row["dt"]
        start = end - timedelta(hours=1)
        for seg in segments:
            lo, hi = max(start, seg.start), min(end, seg.end)
            overlap = max(0.0, (hi - lo).total_seconds() / 3600.0)
            if overlap > 0:
                total += float(row[code]) * overlap
                found = True
    if not found:
        return np.nan
    return total / 60.0 if sunshine else total


def edge_features(dayrow: pd.Series, hourly: pd.DataFrame, wd_q75: float) -> Dict[str, float]:
    sT = _state_series(hourly, "TA")
    sRH = _state_series(hourly, "RH")
    sW = _state_series(hourly, "WS10M")
    # DRY as a state variable.
    sDry = (100.0 - sRH).clip(lower=0.0, upper=100.0)
    sWD = pd.concat([sW.rename("w"), sDry.rename("d")], axis=1).dropna()
    sWDv = (sWD["w"] * sWD["d"]).rename("wd")

    # Vapour pressure deficit (kPa): a more physical form of the RH signal.
    trh = pd.concat([sT.rename("t"), sRH.rename("rh")], axis=1).dropna()
    es = 0.6108 * np.exp((17.27 * trh["t"]) / (trh["t"] + 237.3))
    sVPD = (es * (1.0 - trh["rh"].clip(0.0, 100.0) / 100.0)).clip(lower=0.0).rename("vpd")
    # Locked V6 diagnostic: very humid AND warm at the same quarter-hour.
    # 15 °C is pre-specified before looking at the score; no threshold search.
    sWetWarm15 = ((trh["t"] >= 15.0) & (trh["rh"] >= 90.0)).astype(float).rename("wetwarm15")

    def one(seg_key: str, segs: List[Segment]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out["deg10_h"] = _integral_state(sT, segs, lambda x: np.maximum(x - 10.0, 0.0))
        out["deg12_h"] = _integral_state(sT, segs, lambda x: np.maximum(x - 12.0, 0.0))
        out["deg15_h"] = _integral_state(sT, segs, lambda x: np.maximum(x - 15.0, 0.0))
        out["warm18_h"] = _integral_state(sT, segs, lambda x: (x >= 18.0).astype(float))
        out["cold12_h"] = _integral_state(sT, segs, lambda x: (x <= 12.0).astype(float))
        out["temp_mean"] = _mean_state(sT, segs)
        out["rh_mean"] = _mean_state(sRH, segs)
        out["rh90_h"] = _integral_state(sRH, segs, lambda x: (x >= 90.0).astype(float))
        out["vpd_h"] = _integral_state(sVPD, segs)
        out["vpd_mean"] = _mean_state(sVPD, segs)
        # HYDRATION EDGE: one pre-specified, smooth physical candidate.
        # It is the integrated "room below 0.8 kPa VPD"; higher = longer/stronger
        # low-drying-demand exposure. No yield-fitted weights are used.
        out["hydration08_h"] = _integral_state(sVPD, segs, lambda x: np.maximum(0.80 - x, 0.0))
        out["lowvpd03_h"] = _integral_state(sVPD, segs, lambda x: (x <= 0.30).astype(float))
        out["highvpd08_h"] = _integral_state(sVPD, segs, lambda x: (x >= 0.80).astype(float))
        # V5 locked follow-up: duration threshold + fixed clock split.
        # Thresholds and clock boundaries are chosen before looking at this test's yield score.
        out["nm_lowvpd03_h"] = _integral_state_clock(
            sVPD, segs, lambda x: (x <= 0.30).astype(float), night_morning=True
        )
        out["de_lowvpd03_h"] = _integral_state_clock(
            sVPD, segs, lambda x: (x <= 0.30).astype(float), night_morning=False
        )
        out["nm_rh90_h"] = _integral_state_clock(
            sRH, segs, lambda x: (x >= 90.0).astype(float), night_morning=True
        )
        out["de_rh90_h"] = _integral_state_clock(
            sRH, segs, lambda x: (x >= 90.0).astype(float), night_morning=False
        )
        # V6 locked night-temperature competitors, all in the already frozen 20:00–10:00 window.
        out["nm_temp_mean"] = _mean_state_clock(sT, segs, night_morning=True)
        out["nm_warm15_h"] = _integral_state_clock(
            sT, segs, lambda x: (x >= 15.0).astype(float), night_morning=True
        )
        out["nm_warm18_h"] = _integral_state_clock(
            sT, segs, lambda x: (x >= 18.0).astype(float), night_morning=True
        )
        out["nm_wetwarm15_h"] = _integral_state_clock(
            sWetWarm15, segs, lambda x: x, night_morning=True
        )
        out["wind_h"] = _integral_state(sW, segs)
        out["wind_mean"] = _mean_state(sW, segs)
        out["dry_h"] = _integral_state(sDry, segs)
        out["wd_h"] = _integral_state(sWDv, segs)
        out["wd_mean"] = _mean_state(sWDv, segs)
        out["wd_high_h"] = _integral_state(sWDv, segs, lambda x: (x >= wd_q75).astype(float)) if np.isfinite(wd_q75) else np.nan
        out["rain_mm"] = _hourly_accum(hourly, "PR1H", segs, sunshine=False)
        out["sun_h"] = _hourly_accum(hourly, "SDUR1H", segs, sunshine=True)
        h = sum(s.hours for s in segs)
        out["sun_frac"] = out["sun_h"] / h if h > 0 and np.isfinite(out["sun_h"]) else np.nan
        return {f"{seg_key}_{k}": v for k, v in out.items()}

    f1 = one("f1", dayrow["f1_segments"])
    f3 = one("f3", dayrow["f3_segments"])
    out = {**f1, **f3}
    base_names = [k[3:] for k in f1 if k.startswith("f1_")]
    for name in base_names:
        a, b = f1.get(f"f1_{name}"), f3.get(f"f3_{name}")
        out[f"edge_{name}"] = (a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
    return out


def build_did(daydf: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pair, g in daydf.groupby("pair_key", sort=False):
        g = g.sort_values("day")
        vals = list(g.to_dict("records"))
        for prev, cur in zip(vals[:-1], vals[1:]):
            gap = (cur["day"] - prev["day"]).days
            if not (MIN_REPEAT_GAP_DAYS <= gap <= MAX_REPEAT_GAP_DAYS):
                continue
            rec = {
                "pair_key": pair,
                "earlier": prev["day"],
                "later": cur["day"],
                "gap_days": gap,
                "D_earlier": prev["D_boxes"],
                "D_later": cur["D_boxes"],
                "delta_D": cur["D_boxes"] - prev["D_boxes"],
                "rel_earlier": prev["D_rel"],
                "rel_later": cur["D_rel"],
                "delta_rel": cur["D_rel"] - prev["D_rel"],
            }
            for c in daydf.columns:
                if c.startswith("edge_") or c.startswith("f1_") or c.startswith("f3_"):
                    if c in {"f1_segments", "f3_segments"}:
                        continue
                    a, b = prev.get(c), cur.get(c)
                    if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)) and np.isfinite(a) and np.isfinite(b):
                        rec[f"delta_{c}"] = float(b - a)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values("later").reset_index(drop=True) if rows else pd.DataFrame()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def score_features(did: pd.DataFrame, target: str = "delta_D", prefix: str = "delta_edge_") -> pd.DataFrame:
    """Rank discovery features with TRUE Spearman leave-one-out robustness.

    LAB-156 v1 displayed Spearman rho but its LOO columns were accidentally
    based on Pearson r.  This version keeps both metrics but all LOO robustness
    is calculated from Spearman, matching the displayed discovery statistic.
    """
    candidates = [c for c in did.columns if c.startswith(prefix)]
    out: List[Dict[str, Any]] = []
    for c in candidates:
        z = did[[c, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 5:
            continue
        x = z[c].to_numpy(dtype=float)
        y = z[target].to_numpy(dtype=float)
        rp = _pearson(x, y)
        rs = _spearman(x, y)
        full_sign = np.sign(rs) if np.isfinite(rs) and abs(rs) > 1e-12 else 0
        loo: List[float] = []
        for i in range(len(z)):
            xx = np.delete(x, i); yy = np.delete(y, i)
            rr = _spearman(xx, yy)
            if np.isfinite(rr):
                loo.append(rr)
        same = 100.0 * np.mean([np.sign(r) == full_sign for r in loo]) if loo and full_sign != 0 else np.nan
        min_abs = float(np.min(np.abs(loo))) if loo else np.nan

        # Also remove the single largest target movement.  With N~7 this is an
        # important guard against one dramatic field-pair change driving a clue.
        rs_no_big = np.nan
        if len(z) >= 6:
            j = int(np.nanargmax(np.abs(y)))
            rs_no_big = _spearman(np.delete(x, j), np.delete(y, j))

        robust = (abs(rs) if np.isfinite(rs) else 0.0)
        robust *= (same / 100.0 if np.isfinite(same) else 0.0)
        robust *= (min_abs if np.isfinite(min_abs) else 0.0)
        out.append({
            "feature": c.replace(prefix, ""),
            "N": len(z),
            "Pearson r": rp,
            "Spearman ρ": rs,
            "LOO sama suund %": same,
            "LOO min |ρ|": min_abs,
            "ρ ilma suurima Δ-ta": rs_no_big,
            "robust_score": robust,
        })
    return pd.DataFrame(out).sort_values(["robust_score", "Spearman ρ"], ascending=[False, False]).reset_index(drop=True) if out else pd.DataFrame()


def one_feature_score(did: pd.DataFrame, feature: str, target: str = "delta_D") -> Dict[str, float]:
    z = did[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3:
        return {"N": len(z), "rho": np.nan}
    return {"N": len(z), "rho": _spearman(z[feature].to_numpy(float), z[target].to_numpy(float))}

def permutation_spearman_p(x: np.ndarray, y: np.ndarray, max_exact_n: int = 8, monte_carlo: int = 50000) -> Dict[str, float | str]:
    """Two-sided permutation p-value for Spearman rho.

    Exhaustive for N<=8. For larger N, deterministic Monte Carlo is used so a
    future N=9 refresh does not make the app unnecessarily heavy.
    """
    import itertools

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return {"N": n, "rho": np.nan, "p": np.nan, "mode": "too-small"}
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    obs = _pearson(rx, ry)
    if not np.isfinite(obs):
        return {"N": n, "rho": np.nan, "p": np.nan, "mode": "constant"}

    threshold = abs(obs) - 1e-12
    extreme = 0
    total = 0
    if n <= max_exact_n:
        for perm in itertools.permutations(range(n)):
            r = _pearson(rx, ry[list(perm)])
            if np.isfinite(r) and abs(r) >= threshold:
                extreme += 1
            total += 1
        p = extreme / total if total else np.nan
        mode = f"exact {total:,}"
    else:
        rng = np.random.default_rng(156)
        for _ in range(monte_carlo):
            r = _pearson(rx, ry[rng.permutation(n)])
            if np.isfinite(r) and abs(r) >= threshold:
                extreme += 1
            total += 1
        p = (extreme + 1) / (total + 1)
        mode = f"MC {total:,}"
    return {"N": n, "rho": obs, "p": float(p), "mode": mode}


def loo_linear_test(did: pd.DataFrame, feature_col: str, target: str = "delta_D") -> Dict[str, float]:
    """Strict leave-one-pair-out 1-feature linear test against intercept-only baseline."""
    z = did[[feature_col, target]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(z)
    if n < 5:
        return {"N": n, "base_mae": np.nan, "feature_mae": np.nan, "improvement": np.nan, "wins_pct": np.nan, "slope_same_sign_pct": np.nan}
    x = z[feature_col].to_numpy(float)
    y = z[target].to_numpy(float)
    eb, ef, slopes = [], [], []
    expected_sign = 1.0  # more HYDRATION edge should increase D = order1 - order3
    for i in range(n):
        mask = np.arange(n) != i
        xt, yt = x[mask], y[mask]
        base = float(np.mean(yt))
        if np.nanstd(xt) <= 1e-12:
            pred = base
            slope = 0.0
        else:
            slope = float(np.sum((xt - xt.mean()) * (yt - yt.mean())) / np.sum((xt - xt.mean()) ** 2))
            pred = float(yt.mean() + slope * (x[i] - xt.mean()))
        eb.append(abs(y[i] - base))
        ef.append(abs(y[i] - pred))
        slopes.append(slope)
    eb = np.asarray(eb); ef = np.asarray(ef); slopes = np.asarray(slopes)
    return {
        "N": n,
        "base_mae": float(np.mean(eb)),
        "feature_mae": float(np.mean(ef)),
        "improvement": float(np.mean(eb) - np.mean(ef)),
        "wins_pct": float(100.0 * np.mean(ef < eb)),
        "slope_same_sign_pct": float(100.0 * np.mean(np.sign(slopes) == expected_sign)),
    }


def locked_candidate_row(
    did: pd.DataFrame,
    label: str,
    feature_col: str,
    target: str = "delta_D",
) -> Dict[str, Any]:
    z = did[[feature_col, target]].replace([np.inf, -np.inf], np.nan).dropna() if feature_col in did.columns else pd.DataFrame()
    if len(z) < 5:
        return {
            "Kandidaat": label, "N": len(z), "ρ": np.nan, "perm p": np.nan,
            "Baseline MAE": np.nan, "Tunnuse MAE": np.nan, "Paranemine": np.nan,
            "Võite %": np.nan, "Tõusu + %": np.nan, "Edasi?": "andmeid vähe",
        }
    perm = permutation_spearman_p(z[feature_col].to_numpy(float), z[target].to_numpy(float))
    cv = loo_linear_test(did, feature_col, target)
    improvement = cv.get("improvement", np.nan)
    return {
        "Kandidaat": label,
        "N": int(cv.get("N", len(z))),
        "ρ": perm.get("rho", np.nan),
        "perm p": perm.get("p", np.nan),
        "Baseline MAE": cv.get("base_mae", np.nan),
        "Tunnuse MAE": cv.get("feature_mae", np.nan),
        "Paranemine": improvement,
        "Võite %": cv.get("wins_pct", np.nan),
        "Tõusu + %": cv.get("slope_same_sign_pct", np.nan),
        "Edasi?": "JAH" if np.isfinite(improvement) and improvement > 0 else "EI",
    }


def loo_multi_feature_test(
    did: pd.DataFrame,
    feature_cols: List[str],
    target: str = "delta_D",
) -> Dict[str, float]:
    """Strict leave-one-pair-out OLS diagnostic with several frozen features.

    Used only as a diagnostic because N is tiny. It answers whether night
    temperature adds out-of-pair information on top of RH>=90% duration.
    """
    cols = list(feature_cols) + [target]
    z = did[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(z)
    if n < max(6, len(feature_cols) + 3):
        return {"N": n, "base_mae": np.nan, "model_mae": np.nan, "improvement": np.nan, "wins_pct": np.nan}
    X = z[feature_cols].to_numpy(float)
    y = z[target].to_numpy(float)
    eb, em = [], []
    for i in range(n):
        mask = np.arange(n) != i
        Xt, yt = X[mask], y[mask]
        base = float(np.mean(yt))
        # Standardise from training fold only, then fit intercept + frozen features.
        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0)
        sd = np.where(sd <= 1e-12, 1.0, sd)
        Zt = (Xt - mu) / sd
        zi = (X[i] - mu) / sd
        A = np.column_stack([np.ones(len(Zt)), Zt])
        coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
        pred = float(np.r_[1.0, zi] @ coef)
        eb.append(abs(y[i] - base))
        em.append(abs(y[i] - pred))
    eb = np.asarray(eb); em = np.asarray(em)
    return {
        "N": n,
        "base_mae": float(np.mean(eb)),
        "model_mae": float(np.mean(em)),
        "improvement": float(np.mean(eb) - np.mean(em)),
        "wins_pct": float(100.0 * np.mean(em < eb)),
    }



# -----------------------------------------------------------------------------
# LAB-157 GATE inside the same read-only app: one-harvest-cycle SOURCE–SINK test
# -----------------------------------------------------------------------------
# This is deliberately NOT a production feature. The current harvest is used only
# as a diagnostic sink sensor to ask whether it contains one-cycle-ahead information.
SS_MIN_TRAIN = 12
SS_RIDGE_ALPHA = 8.0
SS_EPS = 0.05
SS_HOURS_PER_ORDER = 3.5
SS_NUMERIC_CLIP = 1.5


def _ss_growth_days(prev: pd.Series, cur: pd.Series) -> float:
    d0 = prev.get("harvest_date")
    d1 = cur.get("harvest_date")
    if not isinstance(d0, date) or not isinstance(d1, date):
        return float("nan")
    o0 = int(prev.get("harvest_order") or 1)
    o1 = int(cur.get("harvest_order") or 1)
    g = float((d1 - d0).days) + float(o1 - o0) * (SS_HOURS_PER_ORDER / 24.0)
    return max(0.5, g)


def _ss_weather_map(rows: List[Dict[str, Any]]) -> Dict[date, Dict[str, float]]:
    out: Dict[date, Dict[str, float]] = {}
    for r in rows:
        try:
            dd = date.fromisoformat(str(r.get("weather_date"))[:10])
        except Exception:
            continue
        if str(r.get("data_kind") or "").strip().lower() != "measured" or not bool(r.get("checked")):
            continue
        rad = _num(r.get("radiation_mj_m2"))
        tn = _num(r.get("temp_night_avg_c"))
        td = _num(r.get("temp_day_avg_c"))
        if not np.isfinite(rad):
            continue
        tmean = np.nanmean([tn, td]) if (np.isfinite(tn) or np.isfinite(td)) else np.nan
        out[dd] = {
            "rad": float(rad),
            "tmean": float(tmean) if np.isfinite(tmean) else np.nan,
            "gdd10": max(0.0, float(tmean) - 10.0) if np.isfinite(tmean) else np.nan,
        }
    return out


def _ss_cycle_source(wmap: Dict[date, Dict[str, float]], prev_day: date, cur_day: date) -> Dict[str, float] | None:
    # Only completed calendar days between harvests. The current harvest-day weather
    # after the harvest moment must never leak into an issue-time state sensor.
    vals: List[Dict[str, float]] = []
    dd = prev_day + timedelta(days=1)
    while dd <= cur_day - timedelta(days=1):
        w = wmap.get(dd)
        if w is None or not np.isfinite(w.get("rad", np.nan)):
            return None
        vals.append(w)
        dd += timedelta(days=1)
    if len(vals) < 2:
        return None
    rad = np.asarray([v["rad"] for v in vals], dtype=float)
    gdd = np.asarray([v.get("gdd10", np.nan) for v in vals], dtype=float)
    return {
        "source_rad_mean": float(np.mean(rad)),
        "source_rad_sum": float(np.sum(rad)),
        "source_gdd10_mean": float(np.nanmean(gdd)) if np.isfinite(gdd).any() else np.nan,
        "source_days": float(len(vals)),
    }


def _ss_build_records(h: pd.DataFrame, wmap: Dict[date, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if h.empty:
        return pd.DataFrame()
    d = h.sort_values(["field_no", "harvest_date", "harvest_order"]).reset_index(drop=True)
    for field, g in d.groupby("field_no", sort=True):
        g = g.sort_values(["harvest_date", "harvest_order"]).reset_index(drop=True)
        # Precompute actual rate for each harvest from the previous harvest clock.
        rates: List[float] = [np.nan] * len(g)
        growths: List[float] = [np.nan] * len(g)
        for i in range(1, len(g)):
            gr = _ss_growth_days(g.iloc[i-1], g.iloc[i])
            abc = _num(g.iloc[i].get("ABC"))
            growths[i] = gr
            if np.isfinite(gr) and gr > 0 and np.isfinite(abc) and abc >= 0:
                rates[i] = float(abc / gr)

        # Issue is current harvest i; target is the next same-field harvest i+1.
        # Field baseline deliberately excludes current harvest, so LOAD is a sensor,
        # not a tautological denominator in the target.
        for i in range(2, len(g) - 1):
            cur_rate = rates[i]
            next_rate = rates[i+1]
            if not (np.isfinite(cur_rate) and cur_rate > 0 and np.isfinite(next_rate) and next_rate >= 0):
                continue
            prior_rates = [float(rates[j]) for j in range(1, i) if np.isfinite(rates[j]) and rates[j] > 0]
            if len(prior_rates) < 2:
                continue
            field_base = float(np.median(prior_rates))
            if not np.isfinite(field_base) or field_base <= 0:
                continue

            prev_row = g.iloc[i-1]
            cur_row = g.iloc[i]
            next_row = g.iloc[i+1]
            src = _ss_cycle_source(wmap, prev_row["harvest_date"], cur_row["harvest_date"])
            if src is None:
                continue

            next_growth = growths[i+1]
            next_abc = _num(next_row.get("ABC"))
            if not (np.isfinite(next_growth) and next_growth > 0 and np.isfinite(next_abc)):
                continue

            load_dev = math.log((cur_rate + SS_EPS) / (field_base + SS_EPS))
            target_dev = math.log((next_rate + SS_EPS) / (field_base + SS_EPS))
            rows.append({
                "field": int(field),
                "issue_day": cur_row["harvest_date"],
                "target_day": next_row["harvest_date"],
                "issue_order": int(cur_row["harvest_order"]),
                "target_order": int(next_row["harvest_order"]),
                "field_base_rate": field_base,
                "current_rate": float(cur_rate),
                "next_rate": float(next_rate),
                "next_growth": float(next_growth),
                "actual_next_abc": float(next_abc),
                "load_dev": float(load_dev),
                "target_dev": float(target_dev),
                "season_day": float((next_row["harvest_date"] - SEASON_START).days),
                **src,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target_day", "target_order", "field"]).reset_index(drop=True)
    return out


def _ss_ridge_one(X: np.ndarray, y: np.ndarray, x: np.ndarray, alpha: float = SS_RIDGE_ALPHA) -> Tuple[float, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X - mu) / sd
    z = (x - mu) / sd
    Xd = np.column_stack([np.ones(len(Z)), Z])
    reg = np.eye(Xd.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Xd.T @ Xd + reg) @ (Xd.T @ y)
    pred = float(np.r_[1.0, z] @ beta)
    return pred, beta


def _ss_walk_forward(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    pred = np.full(len(df), np.nan, dtype=float)
    coef_hist: Dict[str, List[float]] = {c: [] for c in feature_cols}
    vals = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["target_dev"], errors="coerce").to_numpy(dtype=float)
    dates = np.asarray(df["target_day"].tolist(), dtype=object)
    for dd in sorted(set(dates)):
        test_idx = np.where(dates == dd)[0]
        train_idx = np.where(dates < dd)[0]
        if len(train_idx) < SS_MIN_TRAIN:
            continue
        ok = np.isfinite(y[train_idx]) & np.all(np.isfinite(vals[train_idx]), axis=1)
        tr = train_idx[ok]
        if len(tr) < SS_MIN_TRAIN:
            continue
        for j in test_idx:
            if not (np.isfinite(y[j]) and np.all(np.isfinite(vals[j]))):
                continue
            pp, beta = _ss_ridge_one(vals[tr], y[tr], vals[j])
            pred[j] = float(np.clip(pp, -SS_NUMERIC_CLIP, SS_NUMERIC_CLIP))
            for k, c in enumerate(feature_cols):
                coef_hist[c].append(float(beta[k+1]))
    return pred, coef_hist


def _ss_metrics(df: pd.DataFrame, pred_col: str, base_col: str = "pred_BASE") -> Dict[str, float]:
    g = df[np.isfinite(df[pred_col])].copy()
    if g.empty:
        return {"N": 0, "field_mae": np.nan, "day_mae": np.nan, "wins": np.nan, "recent_day_mae": np.nan, "days": 0}
    g["pred_abc"] = g["field_base_rate"] * np.exp(g[pred_col]) * g["next_growth"]
    g["err"] = g["pred_abc"] - g["actual_next_abc"]
    if base_col in g.columns:
        g["base_abc"] = g["field_base_rate"] * np.exp(g[base_col]) * g["next_growth"]
        g["base_err"] = g["base_abc"] - g["actual_next_abc"]
        wins = 100.0 * float(np.mean(np.abs(g["err"]) < np.abs(g["base_err"])))
    else:
        wins = np.nan
    daily_rows = []
    for dd, q in g.groupby("target_day", sort=True):
        if len(q) != 3:
            continue
        daily_rows.append({
            "day": dd,
            "actual": float(q["actual_next_abc"].sum()),
            "pred": float(q["pred_abc"].sum()),
        })
    daily = pd.DataFrame(daily_rows)
    day_mae = float(np.mean(np.abs(daily["pred"] - daily["actual"]))) if not daily.empty else np.nan
    recent = daily.sort_values("day").tail(5) if not daily.empty else daily
    recent_mae = float(np.mean(np.abs(recent["pred"] - recent["actual"]))) if not recent.empty else np.nan
    return {
        "N": int(len(g)),
        "field_mae": float(np.mean(np.abs(g["err"]))),
        "day_mae": day_mae,
        "wins": wins,
        "recent_day_mae": recent_mae,
        "days": int(len(daily)),
    }


def _ss_run(h: pd.DataFrame, weather_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    wmap = _ss_weather_map(weather_rows)
    rec = _ss_build_records(h, wmap)
    if rec.empty:
        return rec, pd.DataFrame(), {}
    specs = {
        "BASE · hooaeg": ["season_day"],
        "LOAD · hooaeg + praegune koormus": ["season_day", "load_dev"],
        "SOURCE · hooaeg + eelmise tsükli kiirgus": ["season_day", "source_rad_mean"],
        "SOURCE+SINK · kiirgus + koormus": ["season_day", "source_rad_mean", "load_dev"],
    }
    coef_all: Dict[str, Dict[str, List[float]]] = {}
    for name, cols in specs.items():
        pp, cc = _ss_walk_forward(rec, cols)
        key = "pred_" + name.split(" · ")[0].replace("+", "PLUS")
        rec[key] = pp
        coef_all[name] = cc
    # Friendly aliases used by metrics.
    rec["pred_BASE"] = rec["pred_BASE"] if "pred_BASE" in rec else np.nan
    summary = []
    for name in specs:
        key = "pred_" + name.split(" · ")[0].replace("+", "PLUS")
        m = _ss_metrics(rec, key, "pred_BASE")
        cc = coef_all.get(name, {})
        rad_hist = cc.get("source_rad_mean", [])
        load_hist = cc.get("load_dev", [])
        summary.append({
            "Mudel": name,
            "N": m["N"],
            "Põllu MAE": m["field_mae"],
            "3-põllu päeva MAE": m["day_mae"],
            "Viimase 5 päeva MAE": m["recent_day_mae"],
            "Võite vs BASE %": m["wins"],
            "Täispäevi": m["days"],
            "Kiirguse + märk %": (100.0 * float(np.mean(np.asarray(rad_hist) > 0))) if rad_hist else np.nan,
            "Koormuse − märk %": (100.0 * float(np.mean(np.asarray(load_hist) < 0))) if load_hist else np.nan,
        })
    return rec, pd.DataFrame(summary), coef_all

def main() -> None:
    st.set_page_config(page_title="KurgiMootor LAB-156", layout="wide")
    st.error("🧪 LAB-156 · ORDER 1 vs 3 SERVAILM · READ-ONLY")
    st.title("Kasvu servatundide audit · RH/VPD V6 · ÖÖTEMP")
    st.caption(
        "Päeva 1. ja 3. korje kui looduslik paariskatse. Ühine kasvuilm taandub; uurime ainult "
        "mittekattuvate tundide ilma. Eelmist saaki ei kasutata ennustajana. Order 2 on teadlikult väljas."
    )

    with st.expander("Katse täpne loogika", expanded=False):
        st.markdown(
            """
- tööpäev: **09:00**, ~**3,5 h/põld** → keskhetked 10:45 / 14:15 / 17:45;
- siht: **A+B+C**, mitte XL;
- kasutame ainult päevi, kus current order1 eelmine order oli 2 ja current order3 eelmine order oli 1;
- ilm: Pärnu ametlikud **TA, RH, WS10M, PR1H, SDUR1H** tunniandmed;
- põhikatse: sama väljade paari kordus ~14 päeva hiljem → **difference-in-differences**;
- tunnuste otsing on discovery. Productionit ei muudeta ja DB-sse ei kirjutata.
            """
        )

    try:
        harvest_rows = db.get_harvest_history(limit=5000)
        h = prepare_harvests(harvest_rows)
        days = build_order13_days(h)
    except Exception as exc:
        st.error(f"Korjeandmete lugemine ebaõnnestus: {exc}")
        st.stop()

    if days.empty:
        st.warning("Struktuurselt puhtaid order1/order3 paarpäevi ei leitud.")
        st.stop()

    min_wx = min(min(s.start.date() for s in segs) for segs in days["f1_segments"].tolist() + days["f3_segments"].tolist() if segs)
    max_wx = max(max(s.end.date() for s in segs) for segs in days["f1_segments"].tolist() + days["f3_segments"].tolist() if segs)

    st.subheader("1. Katse geomeetria")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puhtaid paarpäevi", len(days))
    c2.metric("Ühine kasvuaeg · mediaan", f"{days['common_h'].median():.1f} h")
    c3.metric("Order1-only · mediaan", f"{days['f1_only_h'].median():.1f} h")
    c4.metric("Order3-only · mediaan", f"{days['f3_only_h'].median():.1f} h")
    st.caption(f"Ilma servaaknad: {min_wx.isoformat()}…{max_wx.isoformat()}.")

    preview = days[["day", "field1", "field3", "ABC1", "ABC3", "D_boxes", "common_h", "f1_only_h", "f3_only_h", "net_h"]].copy()
    preview.columns = ["Päev", "Põld 1", "Põld 3", "ABC 1", "ABC 3", "Vahe 1−3", "Ühine h", "1-only h", "3-only h", "Net h"]
    st.dataframe(preview, use_container_width=True, hide_index=True)

    if st.button("🔎 Lae ametlik tunniilm ja otsi seos", type="primary"):
        st.session_state["lab156_run"] = True

    if not st.session_state.get("lab156_run"):
        st.info("Vajuta üks kord. See teeb ainult ametliku tunniilma lugemise ja väikese statistilise auditi; production-mootorit ei treenita.")
        st.stop()

    with st.spinner("Laen Pärnu ametlikud tunniandmed ja arvutan edge-window tunnused…"):
        try:
            hourly = fetch_hourly(min_wx.isoformat(), max_wx.isoformat())
        except Exception as exc:
            st.error(f"Tunniilma laadimine ebaõnnestus: {exc}")
            st.stop()

        if hourly.empty:
            st.error("Ametlikust allikast ei tulnud tunniandmeid.")
            st.stop()

        # Hourly WD threshold based only on weather, never target yield.
        hh = hourly[["dt", "WS10M", "RH"]].dropna().copy()
        wd_vals = hh["WS10M"].to_numpy(dtype=float) * (100.0 - hh["RH"].to_numpy(dtype=float))
        wd_q75 = float(np.quantile(wd_vals[np.isfinite(wd_vals)], 0.75)) if np.isfinite(wd_vals).sum() >= 20 else np.nan

        feat_rows: List[Dict[str, Any]] = []
        for _, r in days.iterrows():
            rec = {**r.to_dict(), **edge_features(r, hourly, wd_q75)}
            feat_rows.append(rec)
        fd = pd.DataFrame(feat_rows)
        did = build_did(fd)

    st.success("Tunniilma audit valmis.")
    st.subheader("2. Sama põllupaari kordus · põhikatse")
    if did.empty:
        st.warning("Sama order1/order3 põllupaari kordusi pole veel piisavalt.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("DID paare", len(did))
    c2.metric("Mediaan kordusvahe", f"{did['gap_days'].median():.0f} p")
    c3.metric("WD HIGH tunni-lävend", "—" if not np.isfinite(wd_q75) else f"{wd_q75:.0f}")

    did_show = did[["pair_key", "earlier", "later", "D_earlier", "D_later", "delta_D", "delta_rel"]].copy()
    did_show["delta_rel"] *= 100.0
    did_show.columns = ["Põllupaar", "Varem", "Hiljem", "Vahe varem", "Vahe hiljem", "Δ vahe kastides", "Δ suhteline pp"]
    st.dataframe(did_show, use_container_width=True, hide_index=True)

    # A structural fact worth surfacing before weather correlations.
    pos_share = 100.0 * float((did["delta_D"] > 0).mean())
    st.caption(
        f"Struktuurikontroll: Δ vahe on positiivne {pos_share:.0f}% korduspaaridest "
        f"({int((did['delta_D'] > 0).sum())}/{len(did)}); keskmine nihe {did['delta_D'].mean():+.2f} kasti. "
        "See võib olla ühine hooaja-/korjekorra nihe, seega ilma-signaali ei tohi sellest automaatselt järeldada."
    )

    st.subheader("3. Mis servailm liigub koos saagivahe muutusega?")
    scores = score_features(did, "delta_D", "delta_edge_")
    scores_rel = score_features(did, "delta_rel", "delta_edge_")
    if scores.empty:
        st.warning("Tunnuste skoorimiseks jäi liiga vähe täielikke tunniilma ridu.")
        st.stop()

    weather_n = int(scores["N"].max())
    if weather_n < len(did):
        st.info(
            f"DID põllupaare on {len(did)}, kuid täieliku tunniilmaga on praegu {weather_n}. "
            "Seetõttu allolev ilmaseos ei kasuta veel kõiki põllupaare."
        )

    # Is the target itself just a chronological trend?
    zz = did[["later", "delta_D"]].dropna().sort_values("later")
    time_rho = _spearman(np.arange(len(zz), dtype=float), zz["delta_D"].to_numpy(float)) if len(zz) >= 3 else np.nan
    st.caption(f"Δ-vahe enda kronoloogiline Spearman ρ: {time_rho:+.2f}. Nullilähedane väärtus vähendab lihtsa ajatrendi kahtlust.")

    nice = scores.copy()
    for c in ("Pearson r", "Spearman ρ", "LOO min |ρ|", "ρ ilma suurima Δ-ta", "robust_score"):
        nice[c] = nice[c].round(3)
    nice["LOO sama suund %"] = nice["LOO sama suund %"].round(0)
    st.dataframe(
        nice[["feature", "N", "Pearson r", "Spearman ρ", "LOO sama suund %", "LOO min |ρ|", "ρ ilma suurima Δ-ta"]],
        use_container_width=True, hide_index=True
    )

    top = scores.iloc[0]
    st.markdown("#### Robustseim vihje (mitte lihtsalt suurim ρ)")
    st.metric(
        top["feature"],
        f"Spearman ρ {top['Spearman ρ']:+.2f}",
        f"LOO min |ρ| {top['LOO min |ρ|']:.2f} · sama suund {top['LOO sama suund %']:.0f}%"
    )

    # Cross-check against relative target, to avoid a result caused only by whole-season scale.
    if not scores_rel.empty:
        m = scores_rel[scores_rel["feature"] == top["feature"]]
        if len(m):
            rr = m.iloc[0]
            st.caption(
                f"Sama tunnus suhtelise 1-vs-3 vahe muutusega: Spearman ρ {rr['Spearman ρ']:+.2f}, "
                f"LOO sama suund {rr['LOO sama suund %']:.0f}%."
            )

    st.subheader("4. Niiskusklaster: RH või tegelikult VPD / vihm / päike?")
    cluster = ["rh_mean", "rh90_h", "vpd_mean", "hydration08_h", "lowvpd03_h", "highvpd08_h", "rain_mm", "sun_h", "dry_h", "wd_h"]
    cl = scores[scores["feature"].isin(cluster)].copy()
    if not cl.empty:
        for c in ("Spearman ρ", "LOO min |ρ|", "ρ ilma suurima Δ-ta"):
            cl[c] = cl[c].round(3)
        cl["LOO sama suund %"] = cl["LOO sama suund %"].round(0)
        st.dataframe(cl[["feature", "N", "Spearman ρ", "LOO sama suund %", "LOO min |ρ|", "ρ ilma suurima Δ-ta"]], use_container_width=True, hide_index=True)

    st.subheader("5. Kummast servast signaal tuleb?")
    st.caption(
        "F1-only on päeva 1. põllu varasem ~20,5 h serv; F3-only on päeva 3. põllu hilisem ~7 h serv. "
        "Kui signaal on päriselt kasvukeskkond, tahame näha, kumb serv seda kannab."
    )
    side1 = score_features(did, "delta_D", "delta_f1_")
    side3 = score_features(did, "delta_D", "delta_f3_")
    rows = []
    for feat in cluster:
        r1 = side1[side1["feature"] == feat]
        r3 = side3[side3["feature"] == feat]
        rows.append({
            "feature": feat,
            "F1-only Δ vs ΔD ρ": (float(r1.iloc[0]["Spearman ρ"]) if len(r1) else np.nan),
            "F3-only Δ vs ΔD ρ": (float(r3.iloc[0]["Spearman ρ"]) if len(r3) else np.nan),
        })
    sides = pd.DataFrame(rows)
    sides["F1-only Δ vs ΔD ρ"] = sides["F1-only Δ vs ΔD ρ"].round(3)
    sides["F3-only Δ vs ΔD ρ"] = sides["F3-only Δ vs ΔD ρ"].round(3)
    st.dataframe(sides, use_container_width=True, hide_index=True)

    st.subheader("6. Üks füüsikaline kandidaat · HYDRATION EDGE")
    st.caption(
        "Et mitte 7 rea peal kümnete tunnuste seast parimat valida, lukustame ühe sileda kandidaadi: "
        "HYDRATION08 = ∫max(0, 0.8−VPD)dt. Suurem väärtus tähendab rohkem madala atmosfäärse kuivatusnõudlusega aega. "
        "Kaalusid saagi järgi ei sobitata."
    )
    hcol = "delta_edge_hydration08_h"
    if hcol in did.columns:
        hz = did[[hcol, "delta_D"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(hz) >= 4:
            perm = permutation_spearman_p(hz[hcol].to_numpy(float), hz["delta_D"].to_numpy(float))
            cv = loo_linear_test(did, hcol, "delta_D")
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("HYDRATION ρ", "—" if not np.isfinite(perm["rho"]) else f"{perm['rho']:+.2f}")
            hc2.metric("Permutatsioon p", "—" if not np.isfinite(perm["p"]) else f"{perm['p']:.3f}", str(perm["mode"]))
            hc3.metric("LOO MAE · ilma ilmata", "—" if not np.isfinite(cv["base_mae"]) else f"{cv['base_mae']:.2f}")
            hc4.metric("LOO MAE · HYDRATION", "—" if not np.isfinite(cv["feature_mae"]) else f"{cv['feature_mae']:.2f}",
                       "" if not np.isfinite(cv["improvement"]) else f"{cv['improvement']:+.2f} kasti")
            st.caption(
                f"Strict leave-one-pair-out: HYDRATION võidab baseline'i "
                f"{cv['wins_pct']:.0f}% hoitud paaridest; sobitatud tõusu märk on oodatud (+) "
                f"{cv['slope_same_sign_pct']:.0f}% foldidest. "
                "See on esimene test, mis küsib mitte ainult 'kas korreleerub?', vaid 'kas servailm aitab uut põllupaari ennustada?'."
            )

    st.subheader("7. Millised korduspaarid on tunniilmaga kaetud?")
    cov_cols = [c for c in ["delta_edge_rh_mean", "delta_edge_vpd_mean", "delta_edge_hydration08_h", "delta_edge_rain_mm", "delta_edge_sun_h"] if c in did.columns]
    cov = did[["pair_key", "earlier", "later"] + cov_cols].copy()
    if cov_cols:
        cov["Täielik põhiilm"] = cov[cov_cols].notna().all(axis=1)
        cov_show = cov[["pair_key", "earlier", "later", "Täielik põhiilm"]].copy()
        cov_show.columns = ["Põllupaar", "Varem", "Hiljem", "Täielik põhiilm"]
        st.dataframe(cov_show, use_container_width=True, hide_index=True)
        miss = cov_show[~cov_show["Täielik põhiilm"]]
        if len(miss):
            st.info(
                f"Praegu puudub täielik tunniilm {len(miss)} korduspaaril. Tunniilma cache on V4-s 30 min; "
                "kui ametlik API lisab värsked tunnid, tulevad need ilma koodi muutmata järgmise värskendusega sisse."
            )

    st.subheader("8. Lukustatud järeltest · kestuslävend + kellaaeg")
    st.caption(
        "Siin ei otsita enam kümnete valemite seast parimat. H1 küsib, kas loeb väga niiskete / väga madala VPD-ga "
        "tundide KESTUS kogu servas. H2 kasutab sama VPD≤0,30 kPa lävendit, kuid jagab aja ette lukustatud akendeks: "
        "öö/hommik 20:00–10:00 ja päev/õhtu 10:00–20:00. Edasi pääseb ainult tunnus, mille strict LOO MAE on baseline'ist väiksem."
    )

    locked = [
        ("H1 · VPD≤0,30 h · kogu serv", "delta_edge_lowvpd03_h"),
        ("H1 · RH≥90% h · kogu serv", "delta_edge_rh90_h"),
        ("H2 · VPD≤0,30 h · öö/hommik 20–10", "delta_edge_nm_lowvpd03_h"),
        ("H2 · VPD≤0,30 h · päev/õhtu 10–20", "delta_edge_de_lowvpd03_h"),
    ]
    locked_rows = [locked_candidate_row(did, label, col) for label, col in locked]
    locked_df = pd.DataFrame(locked_rows)
    for c in ["ρ", "perm p", "Baseline MAE", "Tunnuse MAE", "Paranemine", "Võite %", "Tõusu + %"]:
        if c in locked_df.columns:
            locked_df[c] = pd.to_numeric(locked_df[c], errors="coerce").round(3 if c not in ["Võite %", "Tõusu + %"] else 0)
    st.dataframe(locked_df, use_container_width=True, hide_index=True)

    passes = locked_df[locked_df["Edasi?"] == "JAH"] if not locked_df.empty else pd.DataFrame()
    if passes.empty:
        st.error(
            "Tulemus: kumbki lukustatud hüpotees ei paranda praeguse andmestikuga strict LOO-d. "
            "Siis jätame RH/VPD mehhanismi vihjeks, mitte ennustustunnuseks."
        )
    else:
        best = passes.sort_values("Paranemine", ascending=False).iloc[0]
        st.success(
            f"Strict LOO-st pääseb edasi: {best['Kandidaat']} · "
            f"MAE paranemine {best['Paranemine']:+.2f} kasti. "
            "See ei muuda productionit; järgmine kontroll on sama tunnus N=9 värske tunniilmaga."
        )

    # Secondary clock sanity check with the independent RH>=90% proxy. It is shown,
    # but it is deliberately NOT used to choose the VPD timing winner above.
    rh_clock = [
        locked_candidate_row(did, "RH≥90% · öö/hommik 20–10", "delta_edge_nm_rh90_h"),
        locked_candidate_row(did, "RH≥90% · päev/õhtu 10–20", "delta_edge_de_rh90_h"),
    ]
    with st.expander("RH≥90% kellajaotuse kontroll (sekundaarne, mitte valikukriteerium)", expanded=False):
        rhdf = pd.DataFrame(rh_clock)
        for c in ["ρ", "perm p", "Baseline MAE", "Tunnuse MAE", "Paranemine", "Võite %", "Tõusu + %"]:
            if c in rhdf.columns:
                rhdf[c] = pd.to_numeric(rhdf[c], errors="coerce").round(3 if c not in ["Võite %", "Tõusu + %"] else 0)
        st.dataframe(rhdf, use_container_width=True, hide_index=True)

    st.subheader("9. Lukustatud öötemperatuuri kontroll · kas RH90 on lihtsalt temperatuuri vari?")
    st.caption(
        "RH≥90% 20–10 kandidaat jääb muutmata. Tema kõrvale pannakse neli ette lukustatud öötemperatuuri kontrolli: "
        "F1-only unikaalsete öö/hommiku tundide keskmine T, T≥15 °C tunnid, T≥18 °C tunnid ning RH≥90% & T≥15 °C samaaegsed tunnid. "
        "Lävendeid ei otsita saagi järgi. F3-only hiline ~7 h serv on päevasel ajal, seega öötemperatuuri test on teadlikult F1-only unikaalsel öö/hommiku serval. Eesmärk on teada, kas temperatuur seletab RH90 signaali ära või lisab sellele midagi."
    )

    night_locked = [
        ("LUKUS · RH≥90% · 20–10", "delta_edge_nm_rh90_h"),
        ("T · keskmine F1-only · 20–10", "delta_f1_nm_temp_mean"),
        ("T≥15 °C h · F1-only 20–10", "delta_f1_nm_warm15_h"),
        ("T≥18 °C h · F1-only 20–10", "delta_f1_nm_warm18_h"),
        ("RH≥90% & T≥15 °C h · F1-only 20–10", "delta_f1_nm_wetwarm15_h"),
    ]
    nt_rows = [locked_candidate_row(did, label, col) for label, col in night_locked]
    ntdf = pd.DataFrame(nt_rows)
    for c in ["ρ", "perm p", "Baseline MAE", "Tunnuse MAE", "Paranemine", "Võite %", "Tõusu + %"]:
        if c in ntdf.columns:
            ntdf[c] = pd.to_numeric(ntdf[c], errors="coerce").round(3 if c not in ["Võite %", "Tõusu + %"] else 0)
    st.dataframe(ntdf, use_container_width=True, hide_index=True)

    # Are RH90 and night temperature themselves moving together?
    pair_cols = ["delta_edge_nm_rh90_h", "delta_f1_nm_temp_mean", "delta_D"]
    q = did[pair_cols].replace([np.inf, -np.inf], np.nan).dropna() if all(c in did.columns for c in pair_cols) else pd.DataFrame()
    if len(q) >= 5:
        rho_rh_t = _spearman(q["delta_edge_nm_rh90_h"].to_numpy(float), q["delta_f1_nm_temp_mean"].to_numpy(float))
        st.metric("RH90 Δ ↔ öö-T Δ Spearman ρ", f"{rho_rh_t:+.2f}")

    # Diagnostic only: does adding mean night T to frozen RH90 improve out-of-pair prediction?
    rh_only = loo_linear_test(did, "delta_edge_nm_rh90_h", "delta_D")
    rh_t = loo_multi_feature_test(did, ["delta_edge_nm_rh90_h", "delta_f1_nm_temp_mean"], "delta_D")
    dcols = st.columns(3)
    dcols[0].metric("Baseline LOO MAE", "—" if not np.isfinite(rh_t.get("base_mae", np.nan)) else f"{rh_t['base_mae']:.3f}")
    dcols[1].metric("RH90 üksi LOO MAE", "—" if not np.isfinite(rh_only.get("feature_mae", np.nan)) else f"{rh_only['feature_mae']:.3f}")
    dcols[2].metric("RH90 + öö-T LOO MAE", "—" if not np.isfinite(rh_t.get("model_mae", np.nan)) else f"{rh_t['model_mae']:.3f}")
    if np.isfinite(rh_t.get("model_mae", np.nan)) and np.isfinite(rh_only.get("feature_mae", np.nan)):
        delta = rh_only["feature_mae"] - rh_t["model_mae"]
        if delta > 0:
            st.success(f"Diagnostika: öö-T lisamine parandab RH90 mudelit {delta:+.3f} kasti. N on väike — see ei ole veel uus kandidaat.")
        else:
            st.info(f"Diagnostika: öö-T ei lisa RH90-le väljajäetud paarides infot ({delta:+.3f} kasti). Siis on RH90 hetkel puhtam signaal.")


    # ------------------------------------------------------------------
    # 10. SOURCE–SINK one-cycle gate. This is a mechanism diagnostic only.
    # ------------------------------------------------------------------
    st.subheader("10. SOURCE–SINK tsükli värav · kas praegune viljakoormus + assimilaatide pakkumine ennustab järgmist korjet?")
    st.caption(
        "Marcelise kurgitööde järgi mõjutab source strength uute mitteabortuvate viljade arvu viitega, "
        "samal ajal kui olemasolevad viljad moodustavad tugeva sink-koormuse. Siin EI otsita uut päevaakent: "
        "ühik on üks päris sama põllu korjetsükkel. Praegust saaki kasutatakse ainult diagnostilise sink-sensorina; "
        "see EI lähe productionisse ega muuda lukustatud reeglit, et eelmine saak ei ole järgmise prognoosi ankur."
    )
    with st.expander("SOURCE–SINK katse lukud", expanded=False):
        st.markdown(
            """
- **Issue:** sama põllu tänane korje; **target:** sama põllu järgmine korje (~4–5 p hiljem).
- **Sink-sensor:** tänase `ABC / kasvupäev` kõrvalekalle sama põllu varasemast expanding-mediaanist.
- **Source-sensor:** ainult lõpetatud kalendripäevade keskmine Pärnu globaalradiatsioon eelmise→tänase korje vahel; tänase korjepäeva hilisem ilm on väljas.
- **Target:** järgmise korje kasvukiirus võrreldes sama issue-hetke vana põllubaasiga. Targeti nimetajas EI ole tänast saaki, et vältida kunstlikku regressioon-korrelatsiooni.
- **Strict walk-forward:** target-päev näeb treeningus ainult varasemaid target-päevi; sama päeva 3 põldu treeningusse ei leki.
- Enne jooksu on lukus täpselt 3 mehhanismikandidaati: **LOAD**, **SOURCE**, **SOURCE+SINK**. Ei mingit akna- ega lävendiotsingut.
- Füsioloogiline ootus: SOURCE koefitsient **+**, LOAD koefitsient **−**. Kui SOURCE+SINK ei paranda väljajäetud päevi, täiemahulist cohort-mudelit ei ehitata.
            """
        )

    if st.button("▶ Jooksuta SOURCE–SINK ühe-tsükli kontroll", key="lab157_ss_button"):
        st.session_state["lab157_ss_run"] = True
    if st.session_state.get("lab157_ss_run"):
        try:
            ss_weather_rows = db.get_weather_rows(SEASON_START, datetime.now(ESTONIA).date())
            ss_rec, ss_summary, ss_coef = _ss_run(h, ss_weather_rows)
        except Exception as exc:
            st.error(f"SOURCE–SINK kontroll ebaõnnestus: {exc}")
            ss_rec, ss_summary = pd.DataFrame(), pd.DataFrame()

        if ss_summary.empty:
            st.warning("SOURCE–SINK jaoks ei tekkinud veel piisavalt täielikke ühe-tsükli ridu.")
        else:
            show = ss_summary.copy()
            for c in ["Põllu MAE", "3-põllu päeva MAE", "Viimase 5 päeva MAE"]:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
            for c in ["Võite vs BASE %", "Kiirguse + märk %", "Koormuse − märk %"]:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(0)
            st.dataframe(show, use_container_width=True, hide_index=True)

            base_row = ss_summary[ss_summary["Mudel"].str.startswith("BASE")]
            ss_row = ss_summary[ss_summary["Mudel"].str.startswith("SOURCE+SINK")]
            if not base_row.empty and not ss_row.empty:
                b = base_row.iloc[0]
                q = ss_row.iloc[0]
                day_imp = float(b["3-põllu päeva MAE"] - q["3-põllu päeva MAE"]) if np.isfinite(b["3-põllu päeva MAE"]) and np.isfinite(q["3-põllu päeva MAE"]) else np.nan
                field_imp = float(b["Põllu MAE"] - q["Põllu MAE"]) if np.isfinite(b["Põllu MAE"]) and np.isfinite(q["Põllu MAE"]) else np.nan
                rad_ok = np.isfinite(q["Kiirguse + märk %"]) and float(q["Kiirguse + märk %"]) >= 70.0
                load_ok = np.isfinite(q["Koormuse − märk %"]) and float(q["Koormuse − märk %"]) >= 70.0
                pred_ok = np.isfinite(day_imp) and day_imp > 0 and np.isfinite(field_imp) and field_imp > 0 and float(q["Võite vs BASE %"]) > 50.0
                if pred_ok and rad_ok and load_ok:
                    st.success(
                        f"SOURCE–SINK VÄRAV LÄBITUD · põllu MAE paraneb {field_imp:+.2f} ja 3-põllu päeva MAE {day_imp:+.2f} kasti; "
                        "SOURCE on valdavalt + ning LOAD valdavalt −. Siis on põhjendatud järgmine, päris latentse kohordi katse."
                    )
                elif pred_ok:
                    st.warning(
                        f"Prognoos paraneb (põld {field_imp:+.2f}, päev {day_imp:+.2f}), kuid koefitsientide füsioloogiline märk ei ole stabiilne. "
                        "See ei ole veel source–sink tõend."
                    )
                else:
                    st.info(
                        "SOURCE–SINK ühe-tsükli kandidaat ei löö strict walk-forward BASE'i korraga nii põllu- kui päevatasemel. "
                        "Siis ei ehita me cohort-mudelit ainult sellepärast, et kirjandus kõlab hästi."
                    )

            with st.expander("SOURCE–SINK rea-detail · ainult audit", expanded=False):
                if not ss_rec.empty:
                    det = ss_rec[[
                        "issue_day", "target_day", "field", "field_base_rate", "current_rate", "next_rate",
                        "load_dev", "source_rad_mean", "source_gdd10_mean", "actual_next_abc",
                        "pred_BASE", "pred_LOAD", "pred_SOURCE", "pred_SOURCEPLUSSINK",
                    ]].copy()
                    st.dataframe(det, use_container_width=True, hide_index=True)

    st.warning(
        "Tõlgendus: N on endiselt väike ja korraga vaadatakse mitut tunnust. See on mehhanismi-otsing, mitte tõestus. "
        "Kõige väärtuslikum leid on klaster, mis püsib TRUE Spearman leave-one-out'is, ilma suurima Δ-ta, "
        "suhtelise sihiga ning mille füüsiline vastaspaar (RH ↔ VPD, päike ↔ märg/pilvine) räägib sama lugu. "
        "HYDRATION EDGE on eraldi lukustatud 1-tunnuse kontroll, mitte uus production-mootor."
    )

    with st.expander("Näita edge-ilma detaile", expanded=False):
        cols = ["day", "pair_key", "D_boxes", "f1_only_h", "f3_only_h"] + [c for c in fd.columns if c.startswith("edge_")]
        det = fd[cols].copy()
        st.dataframe(det, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
