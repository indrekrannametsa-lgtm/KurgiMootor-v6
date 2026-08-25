from __future__ import annotations

"""
KurgiMootor LAB-157 — locked RH>=90% 20:00–10:00 N=11 validation

READ ONLY. Does not write to Supabase and does not change production forecasts.

Purpose
-------
Carry forward exactly one candidate from LAB-156:
    delta_edge_nm_rh90_h
    = change in order1-vs-order3 edge duration with RH>=90%
      restricted to the pre-locked 20:00–10:00 clock window.

No threshold search, no alternative weather candidates, no temperature audit.
The only question is whether the locked RH90 signal survives when the next
complete repeated field-pair becomes available (expected N=11).

Frozen LAB-156 reference at N=10:
    Spearman rho      +0.596
    permutation p      0.073
    baseline LOO MAE   1.302
    RH90 LOO MAE       1.193
    improvement       +0.110 boxes
    wins               70%
    slope + sign      100%
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import db
import core

ESTONIA = ZoneInfo("Europe/Tallinn")
ORDER_MIDPOINT = {1: time(10, 45), 2: time(14, 15), 3: time(17, 45)}
MIN_COMMON_HOURS = 60.0
MIN_REPEAT_GAP_DAYS = 10
MAX_REPEAT_GAP_DAYS = 18

REF_N = 10
REF_RHO = 0.596
REF_P = 0.073
REF_BASE_MAE = 1.302
REF_FEATURE_MAE = 1.193
REF_IMPROVEMENT = 0.110
REF_WINS = 70.0
REF_SLOPE_POS = 100.0


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
    for c in ("field_no", "harvest_order", "a", "b", "c"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["data_quality"].map(_confirmed)].copy()
    d = d[d[["harvest_date", "field_no", "harvest_order", "a", "b", "c"]].notna().all(axis=1)].copy()
    d["field_no"] = d["field_no"].astype(int)
    d["harvest_order"] = d["harvest_order"].astype(int)
    d["ABC"] = d["a"] + d["b"] + d["c"]
    d = d.sort_values(["field_no", "harvest_date", "harvest_order"]).reset_index(drop=True)
    d["prev_date"] = d.groupby("field_no")["harvest_date"].shift(1)
    d["prev_order"] = d.groupby("field_no")["harvest_order"].shift(1)
    return d


def build_order13_days(d: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for day, g in d.sort_values(["harvest_date", "harvest_order"]).groupby("harvest_date", sort=True):
        g1 = g[g["harvest_order"] == 1]
        g3 = g[g["harvest_order"] == 3]
        if len(g1) != 1 or len(g3) != 1:
            continue
        r1, r3 = g1.iloc[0], g3.iloc[0]
        if pd.isna(r1["prev_date"]) or pd.isna(r3["prev_date"]) or pd.isna(r1["prev_order"]) or pd.isna(r3["prev_order"]):
            continue
        if not (int(r1["prev_order"]) == 2 and int(r3["prev_order"]) == 1):
            continue

        s1, e1 = _local_dt(r1["prev_date"], 2), _local_dt(day, 1)
        s3, e3 = _local_dt(r3["prev_date"], 1), _local_dt(day, 3)
        if not (s1 < e1 and s3 < e3):
            continue
        common_start, common_end = max(s1, s3), min(e1, e3)
        common_h = max(0.0, (common_end - common_start).total_seconds() / 3600.0)
        if common_h < MIN_COMMON_HOURS:
            continue

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
        recs.append({
            "day": day,
            "field1": int(r1["field_no"]),
            "field3": int(r3["field_no"]),
            "pair_key": f"{int(r1['field_no'])}-{int(r3['field_no'])}",
            "D_boxes": abc1 - abc3,
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
def fetch_hourly_rh(start_iso: str, end_iso: str) -> pd.DataFrame:
    start_day, end_day = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    svc = core.WeatherService()
    rows = svc._official_rows(
        core.OFFICIAL_HOURLY, "Pärnu", "RH",
        start_day - timedelta(days=1), end_day + timedelta(days=1),
    )
    out = []
    for r in rows:
        dt, val = _row_local_dt(r), _num(r.get("vaartus"))
        if dt is not None and np.isfinite(val):
            out.append({"dt": dt, "RH": val})
    return pd.DataFrame(out).sort_values("dt").reset_index(drop=True) if out else pd.DataFrame(columns=["dt", "RH"])


def _rh_series(hourly: pd.DataFrame) -> pd.Series:
    if hourly.empty:
        return pd.Series(dtype=float)
    s = hourly.set_index("dt")["RH"].dropna().sort_index()
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), s.index.max(), freq="15min", tz=ESTONIA)
    return s.reindex(s.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)


def _clock_nm(idx: pd.DatetimeIndex) -> np.ndarray:
    mins = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    return (mins >= 20 * 60) | (mins < 10 * 60)


def _rh90_nm_hours(series: pd.Series, segments: List[Segment]) -> float:
    if series.empty:
        return np.nan
    total = 0.0
    covered = False
    for seg in segments:
        base = (series.index >= seg.start) & (series.index < seg.end)
        if not bool(np.any(base)):
            continue
        vals = series.loc[base]
        vals = vals[np.isfinite(vals.to_numpy(float))]
        if vals.empty:
            continue
        covered = True
        keep = _clock_nm(vals.index)
        arr = vals.to_numpy(float)[keep]
        if len(arr):
            total += float(np.sum(arr >= 90.0) * 0.25)
    return total if covered else np.nan


def add_locked_feature(days: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    rh = _rh_series(hourly)
    rows = []
    for _, r in days.iterrows():
        a = _rh90_nm_hours(rh, r["f1_segments"])
        b = _rh90_nm_hours(rh, r["f3_segments"])
        rec = r.to_dict()
        rec["edge_nm_rh90_h"] = (a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def build_did(daydf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in daydf.groupby("pair_key", sort=False):
        vals = list(g.sort_values("day").to_dict("records"))
        for prev, cur in zip(vals[:-1], vals[1:]):
            gap = (cur["day"] - prev["day"]).days
            if not (MIN_REPEAT_GAP_DAYS <= gap <= MAX_REPEAT_GAP_DAYS):
                continue
            a, b = prev.get("edge_nm_rh90_h"), cur.get("edge_nm_rh90_h")
            rows.append({
                "pair_key": pair,
                "earlier": prev["day"],
                "later": cur["day"],
                "delta_D": cur["D_boxes"] - prev["D_boxes"],
                "delta_edge_nm_rh90_h": float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan,
            })
    return pd.DataFrame(rows).sort_values("later").reset_index(drop=True) if rows else pd.DataFrame()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def permutation_spearman_p(x: np.ndarray, y: np.ndarray, monte_carlo: int = 50000) -> Dict[str, float | str]:
    x, y = np.asarray(x, float), np.asarray(y, float)
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
    rng = np.random.default_rng(157)
    threshold = abs(obs) - 1e-12
    extreme = 0
    for _ in range(monte_carlo):
        r = _pearson(rx, ry[rng.permutation(n)])
        if np.isfinite(r) and abs(r) >= threshold:
            extreme += 1
    p = (extreme + 1) / (monte_carlo + 1)
    return {"N": n, "rho": obs, "p": float(p), "mode": f"MC {monte_carlo:,}"}


def loo_linear_test(did: pd.DataFrame) -> Dict[str, float]:
    feature, target = "delta_edge_nm_rh90_h", "delta_D"
    z = did[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(z)
    if n < 5:
        return {"N": n, "base_mae": np.nan, "feature_mae": np.nan, "improvement": np.nan, "wins_pct": np.nan, "slope_pos_pct": np.nan}
    x, y = z[feature].to_numpy(float), z[target].to_numpy(float)
    eb, ef, slopes = [], [], []
    for i in range(n):
        mask = np.arange(n) != i
        xt, yt = x[mask], y[mask]
        base = float(np.mean(yt))
        if np.nanstd(xt) <= 1e-12:
            slope, pred = 0.0, base
        else:
            slope = float(np.sum((xt - xt.mean()) * (yt - yt.mean())) / np.sum((xt - xt.mean()) ** 2))
            pred = float(yt.mean() + slope * (x[i] - xt.mean()))
        eb.append(abs(y[i] - base))
        ef.append(abs(y[i] - pred))
        slopes.append(slope)
    eb, ef, slopes = np.asarray(eb), np.asarray(ef), np.asarray(slopes)
    return {
        "N": n,
        "base_mae": float(np.mean(eb)),
        "feature_mae": float(np.mean(ef)),
        "improvement": float(np.mean(eb) - np.mean(ef)),
        "wins_pct": float(100.0 * np.mean(ef < eb)),
        "slope_pos_pct": float(100.0 * np.mean(np.asarray(slopes) > 0)),
    }


def main() -> None:
    st.set_page_config(page_title="KurgiMootor LAB-157", layout="wide")
    st.error("🧪 LAB-157 · LUKUS RH≥90% 20–10 · READ-ONLY")
    st.title("Üks kandidaat. Üks järgmine kontroll.")
    st.caption(
        "LAB-156-st on edasi kantud ainult üks enne uute andmete nägemist lukustatud tunnus: "
        "RH≥90% kestus servas kell 20:00–10:00. Midagi uut ei otsita ega sobitata."
    )

    with st.expander("Lukustatud N=10 võrdluspunkt", expanded=False):
        ref = pd.DataFrame([{
            "N": REF_N, "ρ": REF_RHO, "perm p": REF_P,
            "Baseline MAE": REF_BASE_MAE, "RH90 MAE": REF_FEATURE_MAE,
            "Paranemine": REF_IMPROVEMENT, "Võite %": REF_WINS,
            "Tõusu + %": REF_SLOPE_POS,
        }])
        st.dataframe(ref, use_container_width=True, hide_index=True)

    try:
        h = prepare_harvests(db.get_harvest_history(limit=5000))
        days = build_order13_days(h)
    except Exception as exc:
        st.error(f"Korjeandmete lugemine ebaõnnestus: {exc}")
        st.stop()

    if days.empty:
        st.warning("Puhtaid order1/order3 paarpäevi ei leitud.")
        st.stop()

    min_wx = min(min(s.start.date() for s in segs) for segs in days["f1_segments"] if segs)
    max_wx = max(max(s.end.date() for s in segs) for segs in days["f3_segments"] if segs)

    if st.button("🔎 Käivita lukustatud RH90 kontroll", type="primary"):
        st.session_state["lab157_run"] = True
    if not st.session_state.get("lab157_run"):
        st.info("Vajuta üks kord. Loetakse ainult RH tunniandmed ja arvutatakse lukustatud kandidaat.")
        st.stop()

    with st.spinner("Laen Pärnu ametliku RH tunniilma ja arvutan lukustatud testi…"):
        try:
            hourly = fetch_hourly_rh(min_wx.isoformat(), max_wx.isoformat())
        except Exception as exc:
            st.error(f"Tunniilma laadimine ebaõnnestus: {exc}")
            st.stop()
        if hourly.empty:
            st.error("Ametlikust allikast ei tulnud RH tunniandmeid.")
            st.stop()
        fd = add_locked_feature(days, hourly)
        did = build_did(fd)

    if did.empty:
        st.warning("DID korduspaare pole piisavalt.")
        st.stop()

    feature = "delta_edge_nm_rh90_h"
    complete = did[[feature, "delta_D"]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(complete)
    perm = permutation_spearman_p(complete[feature].to_numpy(float), complete["delta_D"].to_numpy(float)) if n >= 4 else {"rho": np.nan, "p": np.nan}
    cv = loo_linear_test(did)

    st.subheader("Tulemus")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Täielik RH90 N", n, f"{n - REF_N:+d} vs lukustatud N=10")
    c2.metric("Spearman ρ", "—" if not np.isfinite(perm.get("rho", np.nan)) else f"{perm['rho']:+.3f}",
              "—" if not np.isfinite(perm.get("rho", np.nan)) else f"{perm['rho'] - REF_RHO:+.3f} vs N=10")
    c3.metric("LOO MAE · baseline", "—" if not np.isfinite(cv["base_mae"]) else f"{cv['base_mae']:.3f}")
    c4.metric("LOO MAE · RH90", "—" if not np.isfinite(cv["feature_mae"]) else f"{cv['feature_mae']:.3f}",
              "" if not np.isfinite(cv["improvement"]) else f"{cv['improvement']:+.3f} kasti")

    res = pd.DataFrame([{
        "Kandidaat": "LUKUS · RH≥90% · 20–10",
        "N": n,
        "ρ": perm.get("rho", np.nan),
        "perm p": perm.get("p", np.nan),
        "Baseline MAE": cv.get("base_mae", np.nan),
        "Tunnuse MAE": cv.get("feature_mae", np.nan),
        "Paranemine": cv.get("improvement", np.nan),
        "Võite %": cv.get("wins_pct", np.nan),
        "Tõusu + %": cv.get("slope_pos_pct", np.nan),
    }])
    for c in ["ρ", "perm p", "Baseline MAE", "Tunnuse MAE", "Paranemine"]:
        res[c] = pd.to_numeric(res[c], errors="coerce").round(3)
    for c in ["Võite %", "Tõusu + %"]:
        res[c] = pd.to_numeric(res[c], errors="coerce").round(0)
    st.dataframe(res, use_container_width=True, hide_index=True)

    st.subheader("Katvus")
    cov = did[["pair_key", "earlier", "later", feature]].copy()
    cov["RH90 täielik"] = cov[feature].notna()
    cov_show = cov[["pair_key", "earlier", "later", "RH90 täielik"]].copy()
    cov_show.columns = ["Põllupaar", "Varem", "Hiljem", "RH90 täielik"]
    st.dataframe(cov_show, use_container_width=True, hide_index=True)
    missing = cov_show[~cov_show["RH90 täielik"]]

    if n < 11:
        st.info(f"Lukustatud test ei ole veel N=11: praegu N={n}. Puuduva RH90 katvusega korduspaare: {len(missing)}.")
    else:
        if np.isfinite(cv.get("improvement", np.nan)) and cv["improvement"] > 0 and np.isfinite(perm.get("rho", np.nan)) and perm["rho"] > 0:
            st.success(
                f"N={n}: lukustatud RH90 kandidaat püsib samas (+) suunas ja parandab strict LOO MAE-d "
                f"{cv['improvement']:+.3f} kasti. See on valideerimise tulemus, mitte productioni muudatus."
            )
        else:
            st.error(
                f"N={n}: lukustatud kandidaat ei läbinud eelnevalt määratud põhikriteeriumi "
                "(positiivne suund + strict LOO paranemine). Ära asenda seda uue kandidaadiga selles LAB-is."
            )

    st.caption("LAB-157 ei otsi alternatiive. Kui see kontroll on dokumenteeritud, võib LAB-156 eemaldada.")


if __name__ == "__main__":
    main()
