from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import db

# KurgiMootor edge_weather-16
# Biological timing audit L1..L14. READ ONLY.
SEASON_START = date(2026, 6, 15)
WEATHER_START = date(2026, 7, 1)
DISCOVERY_END = date(2026, 8, 20)
HOLDOUT_START = date(2026, 8, 21)
MAX_LAG = 14
RIDGE_ALPHA = 10.0
MIN_TRAIN_ROWS = 30
ABC_EPS = 0.20
HOURS_PER_FIELD = 3.0

BASE_COLS = ["growth", "growth_delta", "season_day", "season_day2"]
CHANNELS = ["ÖöT kõver", "Radiatsioon", "WIND×DRY"]

@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: float
    interval_days: Optional[float] = None

def _d(v):
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date): return v
    try: return date.fromisoformat(str(v)[:10])
    except Exception: return None

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
        if dd is None or not _reliable(r): continue
        try: field = int(r.get("field_no"))
        except Exception: continue
        if not 1 <= field <= 14: continue
        abc = _abc(r)
        if abc is None: continue
        try: order = int(r.get("harvest_order") or 1)
        except Exception: order = 1
        out.append(Event(dd, field, order, float(abc), _f(r.get("interval_days"))))
    return sorted(out, key=lambda e: (e.day, e.order, e.field))

def _field_hist(events, field):
    return sorted([e for e in events if e.field == field], key=lambda e: (e.day, e.order, e.field))

def _growth(prev, cur):
    g = (cur.day - prev.day).days + (cur.order - prev.order) * HOURS_PER_FIELD / 24.0
    return max(0.5, float(g))

def _measured(rows):
    out = {}
    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None: continue
        if str(r.get("data_kind") or "").strip().lower() != "measured": continue
        if not bool(r.get("checked")): continue
        night = _f(r.get("temp_night_avg_c"))
        rad = _f(r.get("radiation_mj_m2"))
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))
        if None in (night, rad, wind, rh): continue
        out[dd] = {
            "night": night,
            "rad": rad,
            "wd": wind * (100.0 - rh),
        }
    return out

def _night_curve(v):
    cool = max(0.0, 16.0 - v)
    warm = min(max(v - 16.0, 0.0), 4.0)
    heat = max(0.0, v - 20.0)
    return cool, cool * cool, warm, heat

def _lag_values(target, weather):
    rec = {}
    for lag in range(1, MAX_LAG + 1):
        w = weather.get(target - timedelta(days=lag))
        if w is None: return None
        c, c2, warm, heat = _night_curve(float(w["night"]))
        rec[f"night_cool_L{lag}"] = c
        rec[f"night_cool2_L{lag}"] = c2
        rec[f"night_warm_L{lag}"] = warm
        rec[f"night_heat_L{lag}"] = heat
        rec[f"rad_L{lag}"] = float(w["rad"])
        rec[f"wd_L{lag}"] = float(w["wd"])
    return rec

def _samples(events, weather):
    day_fields = {}
    for e in events:
        day_fields.setdefault(e.day, set()).add(e.field)

    rows = []
    for field in range(1, 15):
        hist = _field_hist(events, field)
        for i in range(1, len(hist)):
            cur, prev = hist[i], hist[i - 1]
            prev2 = hist[i - 2] if i >= 2 else None
            g = _growth(prev, cur)
            if prev2 is not None:
                prev_g = _growth(prev2, prev)
            elif prev.interval_days is not None and prev.interval_days > 0:
                prev_g = float(prev.interval_days)
            else:
                prev_g = g
            lv = _lag_values(cur.day, weather)
            if lv is None: continue
            sd = float((cur.day - SEASON_START).days)
            rows.append({
                "date": cur.day,
                "field": field,
                "order": cur.order,
                "actual_abc": cur.abc,
                "y": math.log(cur.abc + ABC_EPS),
                "growth": g,
                "growth_delta": g - prev_g,
                "season_day": sd,
                "season_day2": sd * sd,
                "complete_day": len(day_fields.get(cur.day, set())) == 3,
                **lv,
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "order", "field"]).reset_index(drop=True) if not df.empty else df

def _fit(train, num_cols):
    x = train[list(num_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(train["y"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(train["field"], errors="coerce").to_numpy(float)
    o = pd.to_numeric(train["order"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & np.isfinite(f) & np.isfinite(o)
    x, y, f, o = x[ok], y[ok], f[ok].astype(int), o[ok].astype(int)
    if len(y) < MIN_TRAIN_ROWS: return None

    mu, sd = np.mean(x, axis=0), np.std(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    fd = np.column_stack([(f == k).astype(float) for k in range(2, 15)])
    od = np.column_stack([(o == k).astype(float) for k in (2, 3)])
    X = np.column_stack([np.ones(len(z)), z, fd, od])

    reg = np.eye(X.shape[1]) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ X + reg) @ (X.T @ y)
    return {"cols": list(num_cols), "mu": mu, "sd": sd, "beta": beta, "n": len(y)}

def _predict(model, test):
    x = test[model["cols"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    f = pd.to_numeric(test["field"], errors="coerce").to_numpy(float)
    o = pd.to_numeric(test["order"], errors="coerce").to_numpy(float)
    out = np.full(len(test), np.nan)
    ok = np.all(np.isfinite(x), axis=1) & np.isfinite(f) & np.isfinite(o)
    if not np.any(ok): return out
    z = (x[ok] - model["mu"]) / model["sd"]
    fi, oi = f[ok].astype(int), o[ok].astype(int)
    fd = np.column_stack([(fi == k).astype(float) for k in range(2, 15)])
    od = np.column_stack([(oi == k).astype(float) for k in (2, 3)])
    X = np.column_stack([np.ones(len(z)), z, fd, od])
    out[ok] = X @ model["beta"]
    return out

def _walk(df, cols):
    out = df[["date", "field", "actual_abc", "complete_day"]].copy()
    out["pred_abc"] = np.nan
    out["train_n"] = np.nan
    for dd in sorted(df["date"].unique()):
        tr = df[df["date"] < dd]
        idx = df.index[df["date"] == dd].tolist()
        model = _fit(tr, cols)
        if model is None or not idx: continue
        py = _predict(model, df.loc[idx])
        pa = np.maximum(0.0, np.exp(np.clip(py, -6.0, 8.0)) - ABC_EPS)
        out.loc[idx, "pred_abc"] = pa
        out.loc[idx, "train_n"] = model["n"]
    return out

def _daily(pred):
    use = pred[pred["complete_day"].astype(bool) & pred["pred_abc"].notna()].copy()
    rows = []
    for dd, g in use.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3: continue
        rows.append({
            "date": dd,
            "actual": float(g["actual_abc"].sum()),
            "pred": float(g["pred_abc"].sum()),
            "train_n": float(g["train_n"].min()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

def _compare(base, cand):
    m = base.merge(cand, on="date", suffixes=("_base", "_cand"))
    if m.empty:
        return {"n": 0, "b": np.nan, "c": np.nan, "imp": np.nan, "wins": 0}
    eb = np.abs(m["pred_base"] - m["actual_base"])
    ec = np.abs(m["pred_cand"] - m["actual_cand"])
    b, c = float(eb.mean()), float(ec.mean())
    return {
        "n": len(m),
        "b": b,
        "c": c,
        "imp": 100.0 * (b - c) / b if b > 1e-9 else np.nan,
        "wins": int((ec < eb).sum()),
    }

def _lag_cols(channel, lag):
    if channel == "ÖöT kõver":
        return [
            f"night_cool_L{lag}", f"night_cool2_L{lag}",
            f"night_warm_L{lag}", f"night_heat_L{lag}",
        ]
    if channel == "Radiatsioon": return [f"rad_L{lag}"]
    if channel == "WIND×DRY": return [f"wd_L{lag}"]
    raise KeyError(channel)

def _screen(df):
    disc = df[df["date"] <= DISCOVERY_END].copy()
    base_daily = _daily(_walk(disc, BASE_COLS))
    if base_daily.empty: raise RuntimeError("Discovery baseline OOS puudub.")
    days = sorted(base_daily["date"].tolist())
    cut = len(days) // 2
    h1, h2 = set(days[:cut]), set(days[cut:])

    rows = []
    for ch in CHANNELS:
        for lag in range(1, MAX_LAG + 1):
            cand = _daily(_walk(disc, BASE_COLS + _lag_cols(ch, lag)))
            all_s = _compare(base_daily, cand)
            s1 = _compare(base_daily[base_daily["date"].isin(h1)], cand[cand["date"].isin(h1)])
            s2 = _compare(base_daily[base_daily["date"].isin(h2)], cand[cand["date"].isin(h2)])
            rows.append({
                "Kanal": ch, "Lag": lag, "N": all_s["n"],
                "BASE MAE": all_s["b"], "Lag MAE": all_s["c"],
                "Paranemine %": all_s["imp"],
                "I pool %": s1["imp"], "II pool %": s2["imp"],
                "Võite": all_s["wins"],
                "Stabiilne +": bool(
                    np.isfinite(all_s["imp"]) and np.isfinite(s1["imp"]) and np.isfinite(s2["imp"])
                    and all_s["imp"] > 0 and s1["imp"] > 0 and s2["imp"] > 0
                ),
            })
    return pd.DataFrame(rows), base_daily

def _runs(vals):
    vals = sorted(set(int(x) for x in vals))
    if not vals: return []
    out = [[vals[0]]]
    for v in vals[1:]:
        if v == out[-1][-1] + 1: out[-1].append(v)
        else: out.append([v])
    return out

def _choose_windows(screen):
    chosen = {}
    for ch in CHANNELS:
        g = screen[(screen["Kanal"] == ch) & screen["Stabiilne +"].astype(bool)]
        runs = [r for r in _runs(g["Lag"].tolist()) if len(r) >= 2]
        if not runs: continue

        gg = screen[screen["Kanal"] == ch].set_index("Lag")
        candidates = []
        for run in runs:
            subs = [run] if len(run) <= 6 else [run[i:i+6] for i in range(len(run)-5)]
            for sr in subs:
                worst = [min(float(gg.loc[l, "I pool %"]), float(gg.loc[l, "II pool %"])) for l in sr]
                overall = [float(gg.loc[l, "Paranemine %"]) for l in sr]
                candidates.append((float(np.mean(worst)), float(np.mean(overall)), len(sr), sr))
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        chosen[ch] = list(candidates[0][3])
    return chosen

def _add_window(df, ch, lags, prefix):
    out = df.copy()
    lags = sorted(lags)
    if ch == "ÖöT kõver":
        cols = []
        for part in ("cool", "cool2", "warm", "heat"):
            name = f"{prefix}_{part}"
            out[name] = out[[f"night_{part}_L{l}" for l in lags]].mean(axis=1)
            cols.append(name)
        return out, cols
    if ch == "Radiatsioon":
        name = f"{prefix}_rad"
        out[name] = out[[f"rad_L{l}" for l in lags]].mean(axis=1)
        return out, [name]
    if ch == "WIND×DRY":
        name = f"{prefix}_wd"
        out[name] = out[[f"wd_L{l}" for l in lags]].mean(axis=1)
        return out, [name]
    raise KeyError(ch)

def _holdout(df, chosen):
    if not chosen: return pd.DataFrame(), pd.DataFrame()
    base_all = _daily(_walk(df, BASE_COLS))
    base = base_all[base_all["date"] >= HOLDOUT_START].copy()
    summary, detail = [], base[["date", "actual", "pred"]].rename(
        columns={"actual": "Tegelik ABC", "pred": "BASE ABC"}
    )
    for i, (ch, lags) in enumerate(chosen.items(), 1):
        aug, extra = _add_window(df, ch, lags, f"w{i}")
        cd = _daily(_walk(aug, BASE_COLS + extra))
        chold = cd[cd["date"] >= HOLDOUT_START].copy()
        s = _compare(base, chold)
        label = f"{ch} L{min(lags)}–L{max(lags)}"
        summary.append({
            "Kanal": ch, "Discovery aken": f"L{min(lags)}–L{max(lags)}",
            "Holdout N": s["n"], "BASE MAE": s["b"], "Akna MAE": s["c"],
            "Paranemine %": s["imp"], "Võite": s["wins"],
            "Holdout +": bool(s["n"] >= 3 and np.isfinite(s["imp"]) and s["imp"] > 0 and s["wins"] >= math.ceil(s["n"]/2)),
        })
        detail = detail.merge(chold[["date", "pred"]].rename(columns={"pred": label}), on="date", how="left")
    return pd.DataFrame(summary), detail

def main():
    st.set_page_config(page_title="KurgiMootor · L1–L14 timing", layout="wide")
    st.title("Ilma bioloogiline ajatelg · L1…L14")
    st.caption("ÖöT kõver · radiatsioon · WIND×DRY · discovery ≤20.08 · holdout ≥21.08 · READ ONLY")
    st.info(
        "Küsimus ei ole enam „milline uus valem?“, vaid „mitu päeva enne korjet ilm infot kannab?“. "
        "BASE ei kasuta eelmist saaki ankruna."
    )

    try:
        hr = db.get_harvest_history(limit=5000)
        ev = _events(hr)
        if not ev:
            st.error("Korjeajalugu puudub."); st.stop()
        latest = max(e.day for e in ev)
        wr = db.get_weather_rows(WEATHER_START, latest)
        wm = _measured(wr)
        df = _samples(ev, wm)
        if df.empty:
            st.error("L1–L14 täieliku mõõdetud ilma ridu ei tekkinud."); st.stop()
        screen, base_disc = _screen(df)
        chosen = _choose_windows(screen)
        holdout, holdout_detail = _holdout(df, chosen)
    except Exception as exc:
        st.exception(exc); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Õppimisridu", len(df))
    c2.metric("Discovery ridu", len(df[df["date"] <= DISCOVERY_END]))
    c3.metric("Holdout ridu", len(df[df["date"] >= HOLDOUT_START]))
    c4.metric("Mõõdetud ilmapäevi", len(wm))

    st.markdown("### 1. Kus ajateljel signaal elab?")
    st.caption("Arv = strict-OOS päevase MAE paranemine võrreldes weatherless BASE-ga. Positiivne on parem.")
    piv = screen.pivot(index="Lag", columns="Kanal", values="Paranemine %").reset_index()
    for ch in CHANNELS:
        if ch not in piv.columns: piv[ch] = np.nan
    st.dataframe(
        piv[["Lag"] + CHANNELS].style.format({
            "Lag": lambda x: f"L{int(x)}",
            "ÖöT kõver": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
            "Radiatsioon": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
            "WIND×DRY": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 2. Discovery pealt valitud stabiilne aken")
    sel = []
    for ch in CHANNELS:
        lags = chosen.get(ch)
        if not lags:
            sel.append({"Kanal": ch, "Valitud aken": "—", "Põhjus": "pole 2 kõrvuti lagi, mis parandaksid mõlemat discovery-poolt"})
        else:
            g = screen[(screen["Kanal"] == ch) & screen["Lag"].isin(lags)]
            sel.append({
                "Kanal": ch, "Valitud aken": f"L{min(lags)}–L{max(lags)}",
                "Põhjus": f"kõik lagid + mõlemas pooles; keskmine single-lag paranemine {g['Paranemine %'].mean():+.1f}%",
            })
    st.dataframe(pd.DataFrame(sel), use_container_width=True, hide_index=True)
    st.caption("Aken kasutab päevade keskmist. Fikseeritud pikkuse korral kannab summa sama infot, ainult teisel skaalal.")

    st.markdown("### 3. 21.08+ holdout · aken muutmata")
    if holdout.empty:
        st.warning("Discovery ei valinud ühtegi stabiilset vähemalt 2-päevast akent.")
    else:
        st.dataframe(
            holdout.style.format({
                "BASE MAE": "{:.2f}", "Akna MAE": "{:.2f}",
                "Paranemine %": lambda x: "—" if pd.isna(x) else f"{x:+.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )
        passed = holdout[holdout["Holdout +"].astype(bool)]
        if len(passed):
            names = ", ".join(f"{r['Kanal']} {r['Discovery aken']}" for _, r in passed.iterrows())
            st.success("✅ Vähemalt üks ENNE holdout'i valitud ajavöönd säilitas eelise: " + names)
        else:
            st.error("❌ Ükski discovery pealt valitud ajavöönd ei kinnitunud 21.08+ holdout'is.")

    if not holdout_detail.empty:
        st.markdown("### 4. Holdout päev-päevalt")
        fmt = {"date": lambda x: x.strftime("%d.%m"), "Tegelik ABC": "{:.1f}", "BASE ABC": "{:.1f}"}
        for c in holdout_detail.columns:
            if c not in fmt and c != "date": fmt[c] = "{:.1f}"
        st.dataframe(holdout_detail.style.format(fmt), use_container_width=True, hide_index=True)

    with st.expander("Kontrolliks · kõik 42 single-lag tulemust", expanded=False):
        det = screen.copy()
        det["Lag"] = det["Lag"].map(lambda x: f"L{int(x)}")
        st.dataframe(
            det.style.format({
                "BASE MAE": "{:.2f}", "Lag MAE": "{:.2f}",
                "Paranemine %": "{:+.1f}%", "I pool %": "{:+.1f}%", "II pool %": "{:+.1f}%",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.caption(
        "BASE = kasvuaeg + kasvuaeg muutus + hooajapäev + hooajapäev² + põld + korjejärjekord. "
        "Eelmise korje saaki ei kasutata sisendina."
    )
    st.caption(
        "AUDIT LOCK: lag/aken valitakse ainult ≤20.08. 21.08+ ei mõjuta valikut. "
        "Strict WF treenib alati date < test date."
    )
    st.caption(
        "READ ONLY: ainult db.get_harvest_history ja db.get_weather_rows. "
        "Ei ole DB kirjutamisi ega ilma värskendust."
    )

if __name__ == "__main__":
    main()
