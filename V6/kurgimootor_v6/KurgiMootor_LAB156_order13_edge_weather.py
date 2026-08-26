from __future__ import annotations

"""
KurgiMootor · edge_weather-17
=============================

LATENT CROP MOMENTUM AUDIT · READ ONLY

Core question
-------------
Do the three fields harvested today reveal a common crop state that helps predict
other fields harvested tomorrow?

This deliberately uses NO WEATHER.

1) Build a strict walk-forward BASE:
   log(A+B+C) ~ growth time + growth-time change + season curve + field + harvest order
   Previous yield is NOT an input.

2) For each complete harvest day, compute the BASE's strict-OOS mean log residual
   across the three harvested fields. This is the observed "system state":
       state_t = mean(log(actual+eps) - log(base_pred+eps))

3) Tomorrow's prediction is corrected with ONLY past observed system state:
       MOM1 = BASE tomorrow * exp(state yesterday)
       MOM2 = BASE tomorrow * exp(mean(state last 2 complete days))
       MOM3 = BASE tomorrow * exp(mean(state last 3 complete days))

No coefficient is fitted to make this work. The correction coefficient is fixed at 1
because the state itself is defined on the log multiplicative scale.

Decision
--------
If the common-state idea is real, MOM should improve BASE not just overall, but
also in both chronological halves. If it does not, this direction is rejected.

Important
---------
- Strict date cutoff in BASE: train dates < test date.
- State for target date uses only earlier complete harvest days.
- No target/future yield enters its own state.
- No weather.
- No DB writes.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import db

SEASON_START = date(2026, 6, 15)
ABC_EPS = 0.20
HOURS_PER_FIELD = 3.0
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 24

BASE_NUM_COLS = ["growth", "growth_delta", "season_day", "season_day2"]

@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None


def _d(v):
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


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc(r):
    vals = [_f(r.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        x = _f(r.get("_abc"))
        return x if x is not None and x >= 0 else None
    return float(sum(vals))


def _reliable(r):
    q = str(r.get("data_quality") or r.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _events(rows):
    out = []
    for r in rows:
        dd = _d(r.get("harvest_date"))
        if dd is None or not _reliable(r):
            continue
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        if not 1 <= field <= 14:
            continue
        abc = _abc(r)
        if abc is None or abc < 0:
            continue
        try:
            order = int(r.get("harvest_order") or 1)
        except Exception:
            order = 1
        out.append(Event(dd, field, order, float(abc), _f(r.get("interval_days"))))
    return sorted(out, key=lambda e: (e.day, e.order, e.field))


def _field_hist(events: Sequence[Event], field: int):
    return sorted([e for e in events if e.field == field], key=lambda e: (e.day, e.order, e.field))


def _growth(prev: Event, cur: Event):
    g = float((cur.day - prev.day).days)
    g += (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _samples(events: List[Event]):
    day_fields: Dict[date, set] = {}
    for e in events:
        day_fields.setdefault(e.day, set()).add(e.field)

    rows = []
    for field in range(1, 15):
        hist = _field_hist(events, field)
        for i in range(1, len(hist)):
            cur = hist[i]
            prev = hist[i - 1]
            prev2 = hist[i - 2] if i >= 2 else None

            growth = _growth(prev, cur)
            if prev2 is not None:
                prev_growth = _growth(prev2, prev)
            elif prev.interval_days is not None and prev.interval_days > 0:
                prev_growth = float(prev.interval_days)
            else:
                prev_growth = growth

            season_day = float((cur.day - SEASON_START).days)
            rows.append({
                "date": cur.day,
                "field": int(field),
                "order": int(cur.order),
                "actual_abc": float(cur.abc),
                "y": math.log(float(cur.abc) + ABC_EPS),
                "growth": float(growth),
                "growth_delta": float(growth - prev_growth),
                "season_day": season_day,
                "season_day2": season_day * season_day,
                "complete_day": len(day_fields.get(cur.day, set())) == 3,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "order", "field"]).reset_index(drop=True)
    return df


def _fit(train: pd.DataFrame):
    x = train[BASE_NUM_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float)
    fields = pd.to_numeric(train["field"], errors="coerce").to_numpy(dtype=float)
    orders = pd.to_numeric(train["order"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & np.isfinite(fields) & np.isfinite(orders)
    x, y = x[ok], y[ok]
    fields, orders = fields[ok].astype(int), orders[ok].astype(int)

    if len(y) < MIN_TRAIN_ROWS:
        return None

    mu, sd = np.mean(x, axis=0), np.std(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd

    fd = np.column_stack([(fields == f).astype(float) for f in range(2, 15)])
    od = np.column_stack([(orders == o).astype(float) for o in (2, 3)])
    X = np.column_stack([np.ones(len(z)), z, fd, od])

    reg = np.eye(X.shape[1]) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ X + reg) @ (X.T @ y)

    return {"mu": mu, "sd": sd, "beta": beta, "n": len(y)}


def _predict_log(model, test: pd.DataFrame):
    x = test[BASE_NUM_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fields = pd.to_numeric(test["field"], errors="coerce").to_numpy(dtype=float)
    orders = pd.to_numeric(test["order"], errors="coerce").to_numpy(dtype=float)

    out = np.full(len(test), np.nan, dtype=float)
    ok = np.all(np.isfinite(x), axis=1) & np.isfinite(fields) & np.isfinite(orders)
    if not np.any(ok):
        return out

    z = (x[ok] - model["mu"]) / model["sd"]
    fi, oi = fields[ok].astype(int), orders[ok].astype(int)
    fd = np.column_stack([(fi == f).astype(float) for f in range(2, 15)])
    od = np.column_stack([(oi == o).astype(float) for o in (2, 3)])
    X = np.column_stack([np.ones(len(z)), z, fd, od])
    out[ok] = X @ model["beta"]
    return out


def _strict_base(df: pd.DataFrame):
    out = df[["date", "field", "order", "actual_abc", "y", "complete_day"]].copy()
    out["base_log"] = np.nan
    out["base_abc"] = np.nan
    out["train_n"] = np.nan

    for dd in sorted(df["date"].unique()):
        train = df[df["date"] < dd]
        idx = df.index[df["date"] == dd].tolist()
        if not idx:
            continue
        model = _fit(train)
        if model is None:
            continue
        pred_log = _predict_log(model, df.loc[idx])
        pred_abc = np.maximum(0.0, np.exp(np.clip(pred_log, -6.0, 8.0)) - ABC_EPS)
        out.loc[idx, "base_log"] = pred_log
        out.loc[idx, "base_abc"] = pred_abc
        out.loc[idx, "train_n"] = int(model["n"])

    out["log_resid"] = out["y"] - out["base_log"]
    return out


def _complete_daily(base_rows: pd.DataFrame):
    use = base_rows[
        base_rows["complete_day"].astype(bool)
        & base_rows["base_log"].notna()
        & base_rows["base_abc"].notna()
        & base_rows["actual_abc"].notna()
    ].copy()

    rows = []
    for dd, g in use.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3:
            continue
        rows.append({
            "date": dd,
            "fields": ",".join(str(int(x)) for x in g.sort_values("order")["field"].tolist()),
            "actual": float(g["actual_abc"].sum()),
            "base": float(g["base_abc"].sum()),
            "state": float(g["log_resid"].mean()),
            "state_sd": float(g["log_resid"].std(ddof=0)),
            "train_n": int(g["train_n"].min()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _apply_momentum(base_rows: pd.DataFrame, daily: pd.DataFrame):
    if daily.empty:
        return pd.DataFrame()

    state_by_date = {row["date"]: float(row["state"]) for _, row in daily.iterrows()}
    complete_dates = sorted(state_by_date)
    date_pos = {dd: i for i, dd in enumerate(complete_dates)}

    use = base_rows[base_rows["complete_day"].astype(bool) & base_rows["base_log"].notna()].copy()
    rows = []

    for dd, g in use.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3 or dd not in date_pos:
            continue

        pos = date_pos[dd]
        previous_dates = complete_dates[:pos]
        rec = {
            "date": dd,
            "fields": ",".join(str(int(x)) for x in g.sort_values("order")["field"].tolist()),
            "actual": float(g["actual_abc"].sum()),
            "base": float(g["base_abc"].sum()),
        }

        for n in (1, 2, 3):
            if len(previous_dates) < n:
                rec[f"mom{n}"] = np.nan
                rec[f"state{n}"] = np.nan
                continue

            chosen = previous_dates[-n:]
            state = float(np.mean([state_by_date[x] for x in chosen]))
            pred_log = g["base_log"].to_numpy(dtype=float) + state
            pred_abc = np.maximum(0.0, np.exp(np.clip(pred_log, -6.0, 8.0)) - ABC_EPS)
            rec[f"mom{n}"] = float(np.sum(pred_abc))
            rec[f"state{n}"] = state

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _mae(df: pd.DataFrame, col: str):
    x = pd.to_numeric(df[col], errors="coerce")
    a = pd.to_numeric(df["actual"], errors="coerce")
    ok = x.notna() & a.notna()
    return float(np.mean(np.abs(x[ok] - a[ok]))) if ok.any() else np.nan


def _mape(df: pd.DataFrame, col: str):
    x = pd.to_numeric(df[col], errors="coerce")
    a = pd.to_numeric(df["actual"], errors="coerce")
    ok = x.notna() & a.notna() & (a.abs() > 1e-9)
    return float(np.mean(np.abs(x[ok] - a[ok]) / np.abs(a[ok])) * 100.0) if ok.any() else np.nan


def _wins(df: pd.DataFrame, col: str):
    x = pd.to_numeric(df[col], errors="coerce")
    b = pd.to_numeric(df["base"], errors="coerce")
    a = pd.to_numeric(df["actual"], errors="coerce")
    ok = x.notna() & b.notna() & a.notna()
    if not ok.any():
        return 0, 0
    ce = np.abs(x[ok] - a[ok])
    be = np.abs(b[ok] - a[ok])
    return int((ce < be).sum()), int(ok.sum())


def _metrics_row(df: pd.DataFrame, label: str, col: str):
    valid = df[df[col].notna()].copy()
    b = _mae(valid, "base")
    c = _mae(valid, col)
    wins, n = _wins(valid, col)
    imp = 100.0 * (b - c) / b if np.isfinite(b) and b > 1e-9 and np.isfinite(c) else np.nan
    return {
        "Variant": label,
        "N päeva": n,
        "BASE MAE": b,
        "MOM MAE": c,
        "Paranemine %": imp,
        "Võite": wins,
        "MAPE %": _mape(valid, col),
    }


def _split_dates(df):
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return set(dates), set()
    cut = len(dates) // 2
    return set(dates[:cut]), set(dates[cut:])


def _diagnostics(daily):
    if len(daily) < 3:
        return np.nan, np.nan, np.nan
    s = daily["state"].to_numpy(dtype=float)
    corr = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if np.std(s[:-1]) > 1e-9 and np.std(s[1:]) > 1e-9 else np.nan
    same = float(np.mean(np.sign(s[:-1]) == np.sign(s[1:])) * 100.0)
    spread = float(np.nanmedian(daily["state_sd"].to_numpy(dtype=float)))
    return corr, same, spread


def main():
    st.set_page_config(page_title="KurgiMootor · crop momentum", layout="wide")
    st.title("Kas 14 põllul on ühine crop momentum?")
    st.caption("Tänased 3 põldu → homsed teised põllud · ilma ilmata · strict OOS · READ ONLY")
    st.info(
        "BASE ei kasuta eelmist saaki. Momentum tuleb ainult varem korjatud põldude BASE-veast "
        "ja korrigeerib järgmise täieliku korjepäeva teisi põlde."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        samples = _samples(events)
        if samples.empty:
            st.error("Õppimisridu ei tekkinud.")
            st.stop()
        base_rows = _strict_base(samples)
        daily = _complete_daily(base_rows)
        test = _apply_momentum(base_rows, daily)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if daily.empty or test.empty:
        st.error("Täielikke strict-OOS korjepäevi ei tekkinud piisavalt.")
        st.stop()

    corr, same, spread = _diagnostics(daily)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Täielikke OOS päevi", len(daily))
    c2.metric("State autokorrelatsioon", "—" if not np.isfinite(corr) else f"{corr:+.2f}")
    c3.metric("Sama märk järgmisel päeval", "—" if not np.isfinite(same) else f"{same:.0f}%")
    c4.metric("3 põllu sisemine hajuvus", "—" if not np.isfinite(spread) else f"{spread:.2f} log")

    st.markdown("### 1. Kas eilne ühine seisund aitab homset?")
    rows = [_metrics_row(test, f"MOM{n} · viimase {n} päeva state", f"mom{n}") for n in (1, 2, 3)]
    metrics = pd.DataFrame(rows)
    st.dataframe(
        metrics.style.format({
            "BASE MAE": lambda x: "—" if pd.isna(x) else f"{float(x):.2f}",
            "MOM MAE": lambda x: "—" if pd.isna(x) else f"{float(x):.2f}",
            "Paranemine %": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%",
            "MAPE %": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%",
        }), use_container_width=True, hide_index=True
    )

    usable = test[test["mom1"].notna()].copy()
    half1, half2 = _split_dates(usable)
    early = _metrics_row(usable[usable["date"].isin(half1)], "MOM1 · I pool", "mom1")
    late = _metrics_row(usable[usable["date"].isin(half2)], "MOM1 · II pool", "mom1")

    st.markdown("### 2. Kõige puhtam test: MOM1 kahes ajapooles")
    halves = pd.DataFrame([early, late])
    st.dataframe(
        halves.style.format({
            "BASE MAE": lambda x: "—" if pd.isna(x) else f"{float(x):.2f}",
            "MOM MAE": lambda x: "—" if pd.isna(x) else f"{float(x):.2f}",
            "Paranemine %": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%",
            "MAPE %": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%",
        }), use_container_width=True, hide_index=True
    )

    overall = rows[0]
    good_overall = (
        np.isfinite(overall["Paranemine %"]) and overall["Paranemine %"] >= 10.0
        and overall["Võite"] >= math.ceil(0.55 * max(1, overall["N päeva"]))
    )
    good_halves = (
        np.isfinite(early["Paranemine %"]) and np.isfinite(late["Paranemine %"])
        and early["Paranemine %"] > 0.0 and late["Paranemine %"] > 0.0
    )

    if good_overall and good_halves:
        st.success("✅ ÜHINE MOMENTUM ON TOETATUD: eelmise päeva teiste põldude OOS-viga parandab homsete põldude prognoosi ja eelis püsib mõlemas ajapooles.")
    elif np.isfinite(overall["Paranemine %"]) and overall["Paranemine %"] > 0:
        st.warning("🟡 Ühine state annab üldiselt eelise, kuid stabiilsus pole piisav. Seda ei tohiks veel uueks baasiks kuulutada.")
    else:
        st.error("❌ ÜHISE MOMENTUMI HÜPOTEES EI PEA: eilsete põldude ühine viga ei paranda järgmise päeva teiste põldude prognoosi.")

    st.markdown("### 3. Päev-päevalt")
    show = test[["date", "fields", "actual", "base", "state1", "mom1", "state2", "mom2", "state3", "mom3"]].copy()
    show["BASE viga"] = show["base"] - show["actual"]
    show["MOM1 viga"] = show["mom1"] - show["actual"]
    st.dataframe(
        show.style.format({
            "date": lambda x: x.strftime("%d.%m") if hasattr(x, "strftime") else str(x),
            "actual": "{:.1f}", "base": "{:.1f}",
            "state1": lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}",
            "mom1": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "state2": lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}",
            "mom2": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "state3": lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}",
            "mom3": lambda x: "—" if pd.isna(x) else f"{float(x):.1f}",
            "BASE viga": "{:+.1f}",
            "MOM1 viga": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}",
        }), use_container_width=True, hide_index=True
    )

    st.caption("state1 = eelmise täieliku korjepäeva kolme põllu keskmine strict-OOS log-residuaal. state −0.20 tähendab ligikaudu 18% alla BASE ootuse.")

    with st.expander("Kontrolliks · päevane latentne state", expanded=False):
        state_show = daily.copy()
        state_show["state multiplier"] = np.exp(state_show["state"])
        st.dataframe(
            state_show.style.format({
                "date": lambda x: x.strftime("%d.%m"),
                "actual": "{:.1f}", "base": "{:.1f}",
                "state": "{:+.3f}", "state_sd": "{:.3f}", "state multiplier": "{:.3f}",
            }), use_container_width=True, hide_index=True
        )

    st.divider()
    st.caption("AUDIT LOCK: BASE treenib alati date < test date. Target-päeva momentum kasutab ainult varasemate täielike korjepäevade OOS-residuaale. Eelmise sama põllu saak ei ole BASE sisend.")
    st.caption("READ ONLY: ainult db.get_harvest_history. Ilmaandmeid ega production snapshotte ei kasutata.")


if __name__ == "__main__":
    main()
