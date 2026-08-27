from __future__ import annotations

"""KurgiMootor edge_weather-18 · latent crop-state state-space audit · READ ONLY."""

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
MIN_BASE_TRAIN_ROWS = 24
MIN_STATE_DAYS = 6
MIN_TRANSITIONS = 5
BASE_COLS = ["growth", "growth_delta", "season_day", "season_day2"]

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
    if v is None: return None
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
        if abc is None or abc < 0: continue
        try: order = int(r.get("harvest_order") or 1)
        except Exception: order = 1
        out.append(Event(dd, field, order, float(abc), _f(r.get("interval_days"))))
    return sorted(out, key=lambda e: (e.day, e.order, e.field))

def _field_hist(events: Sequence[Event], field: int):
    return sorted([e for e in events if e.field == field], key=lambda e: (e.day, e.order, e.field))

def _growth(prev: Event, cur: Event):
    g = float((cur.day - prev.day).days) + (cur.order - prev.order) * HOURS_PER_FIELD / 24.0
    return max(0.5, g)

def _samples(events: List[Event]):
    day_fields: Dict[date, set] = {}
    for e in events:
        day_fields.setdefault(e.day, set()).add(e.field)
    rows = []
    for field in range(1, 15):
        hist = _field_hist(events, field)
        for i in range(1, len(hist)):
            cur, prev = hist[i], hist[i-1]
            prev2 = hist[i-2] if i >= 2 else None
            growth = _growth(prev, cur)
            if prev2 is not None:
                prev_growth = _growth(prev2, prev)
            elif prev.interval_days is not None and prev.interval_days > 0:
                prev_growth = float(prev.interval_days)
            else:
                prev_growth = growth
            sd = float((cur.day - SEASON_START).days)
            rows.append({
                "date": cur.day, "field": field, "order": cur.order,
                "actual_abc": cur.abc, "y": math.log(cur.abc + ABC_EPS),
                "growth": growth, "growth_delta": growth - prev_growth,
                "season_day": sd, "season_day2": sd*sd,
                "complete_day": len(day_fields.get(cur.day, set())) == 3,
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "order", "field"]).reset_index(drop=True) if not df.empty else df

def _fit_base(train: pd.DataFrame):
    x = train[BASE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(train["y"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(train["field"], errors="coerce").to_numpy(float)
    o = pd.to_numeric(train["order"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & np.isfinite(f) & np.isfinite(o)
    x, y, f, o = x[ok], y[ok], f[ok].astype(int), o[ok].astype(int)
    if len(y) < MIN_BASE_TRAIN_ROWS: return None
    mu, sd = np.mean(x, axis=0), np.std(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    fd = np.column_stack([(f == k).astype(float) for k in range(2, 15)])
    od = np.column_stack([(o == k).astype(float) for k in (2, 3)])
    X = np.column_stack([np.ones(len(z)), z, fd, od])
    reg = np.eye(X.shape[1]) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    try: beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    except np.linalg.LinAlgError: beta = np.linalg.pinv(X.T @ X + reg) @ (X.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta, "n": len(y)}

def _predict_base_log(model, test: pd.DataFrame):
    x = test[BASE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
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

def _strict_base(df: pd.DataFrame):
    out = df[["date", "field", "order", "actual_abc", "y", "complete_day"]].copy()
    out["base_log"] = np.nan; out["base_abc"] = np.nan; out["train_n"] = np.nan
    for dd in sorted(df["date"].unique()):
        tr = df[df["date"] < dd]
        idx = df.index[df["date"] == dd].tolist()
        model = _fit_base(tr)
        if model is None or not idx: continue
        plog = _predict_base_log(model, df.loc[idx])
        pabc = np.maximum(0.0, np.exp(np.clip(plog, -6.0, 8.0)) - ABC_EPS)
        out.loc[idx, "base_log"] = plog
        out.loc[idx, "base_abc"] = pabc
        out.loc[idx, "train_n"] = model["n"]
    out["log_resid"] = out["y"] - out["base_log"]
    return out

def _daily_obs(base_rows: pd.DataFrame):
    use = base_rows[base_rows["complete_day"].astype(bool) & base_rows["base_log"].notna() & base_rows["log_resid"].notna()].copy()
    rows = []
    for dd, g in use.groupby("date", sort=True):
        if len(g) != 3 or g["field"].nunique() != 3: continue
        resid = g["log_resid"].to_numpy(float)
        within_var = float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0
        rows.append({
            "date": dd,
            "fields": ",".join(str(int(x)) for x in g.sort_values("order")["field"]),
            "actual": float(g["actual_abc"].sum()), "base": float(g["base_abc"].sum()),
            "z_state": float(np.mean(resid)), "state_sd": float(np.std(resid, ddof=0)),
            "obs_var": max(within_var / len(resid), 1e-6),
            "train_n": int(g["train_n"].min()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

def _estimate_dynamics(hist: pd.DataFrame):
    if len(hist) < MIN_STATE_DAYS: return None
    z = hist["z_state"].to_numpy(float)
    r = hist["obs_var"].to_numpy(float)
    xp, xn = z[:-1], z[1:]
    pv = r[:-1] + r[1:]
    ok = np.isfinite(xp) & np.isfinite(xn) & np.isfinite(pv)
    xp, xn, pv = xp[ok], xn[ok], pv[ok]
    if len(xp) < MIN_TRANSITIONS: return None
    w = 1.0 / np.maximum(pv, 1e-6)
    den = float(np.sum(w*xp*xp))
    phi = 0.0 if den <= 1e-12 else float(np.sum(w*xp*xn) / den)
    phi = float(np.clip(phi, -0.98, 0.98))
    innov = xn - phi*xp
    innov_var = float(np.sum(w*innov*innov) / np.sum(w))
    meas = r[1:][ok] + (phi*phi)*r[:-1][ok]
    meas_var = float(np.sum(w*meas) / np.sum(w))
    q = max(innov_var - meas_var, 1e-6)
    init_var = max(float(np.var(z, ddof=1)) if len(z) > 1 else q, q, 1e-6)
    return {"phi": phi, "q": q, "init_var": init_var, "n": len(hist)}

def _kalman_one_step(hist: pd.DataFrame):
    dyn = _estimate_dynamics(hist)
    if dyn is None: return None
    phi, q = dyn["phi"], dyn["q"]
    m, P = 0.0, dyn["init_var"]
    last_gain = np.nan
    for obs, obs_var in zip(hist["z_state"].to_numpy(float), hist["obs_var"].to_numpy(float)):
        mp = phi*m
        Pp = phi*phi*P + q
        S = Pp + max(float(obs_var), 1e-6)
        K = Pp / S
        m = mp + K*(float(obs) - mp)
        P = max((1.0-K)*Pp, 1e-9)
        last_gain = K
    return {
        "state_pred": float(phi*m),
        "state_pred_sd": float(math.sqrt(max(phi*phi*P + q, 0.0))),
        "phi": float(phi), "q": float(q),
        "last_filtered_state": float(m),
        "last_filtered_sd": float(math.sqrt(max(P, 0.0))),
        "last_gain": float(last_gain) if np.isfinite(last_gain) else np.nan,
        "n_state_days": int(dyn["n"]),
    }

def _strict_state_preds(base_rows: pd.DataFrame, daily: pd.DataFrame):
    rows = []
    for target in daily["date"].tolist():
        tg = base_rows[(base_rows["date"] == target) & base_rows["complete_day"].astype(bool) & base_rows["base_log"].notna()].copy()
        if len(tg) != 3 or tg["field"].nunique() != 3: continue
        hist = daily[daily["date"] < target].copy()
        kf = _kalman_one_step(hist)
        if kf is None: continue
        actual = float(tg["actual_abc"].sum())
        base = float(tg["base_abc"].sum())
        klog = tg["base_log"].to_numpy(float) + kf["state_pred"]
        kalman = float(np.sum(np.maximum(0.0, np.exp(np.clip(klog, -6.0, 8.0)) - ABC_EPS)))
        obs = daily[daily["date"] == target].iloc[0]
        rows.append({
            "date": target,
            "fields": ",".join(str(int(x)) for x in tg.sort_values("order")["field"]),
            "actual": actual, "base": base, "kalman": kalman,
            "base_error": base-actual, "kalman_error": kalman-actual,
            "state_pred": kf["state_pred"], "state_pred_sd": kf["state_pred_sd"],
            "phi": kf["phi"], "q": kf["q"],
            "last_filtered_state": kf["last_filtered_state"],
            "last_filtered_sd": kf["last_filtered_sd"],
            "last_gain": kf["last_gain"], "n_state_days": kf["n_state_days"],
            "observed_target_state": float(obs["z_state"]),
            "observed_target_sd": float(obs["state_sd"]),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

def _metrics(df: pd.DataFrame):
    if df.empty: return {"n":0,"base_mae":np.nan,"kalman_mae":np.nan,"imp":np.nan,"wins":0,"base_mape":np.nan,"kalman_mape":np.nan}
    a,b,k = df["actual"].to_numpy(float), df["base"].to_numpy(float), df["kalman"].to_numpy(float)
    eb, ek = np.abs(b-a), np.abs(k-a)
    bm, km = float(np.mean(eb)), float(np.mean(ek))
    ok = np.abs(a) > 1e-9
    return {
        "n": len(df), "base_mae": bm, "kalman_mae": km,
        "imp": 100.0*(bm-km)/bm if bm > 1e-9 else np.nan,
        "wins": int(np.sum(ek < eb)),
        "base_mape": float(np.mean(eb[ok]/np.abs(a[ok]))*100.0),
        "kalman_mape": float(np.mean(ek[ok]/np.abs(a[ok]))*100.0),
    }

def _halves(df: pd.DataFrame):
    days = sorted(df["date"].tolist())
    cut = len(days)//2
    a,b = set(days[:cut]), set(days[cut:])
    return df[df["date"].isin(a)].copy(), df[df["date"].isin(b)].copy()

def main():
    st.set_page_config(page_title="KurgiMootor · latent state-space", layout="wide")
    st.title("Latentne crop-state · state-space audit")
    st.caption("BASE → õppiv AR(1) latentne state → Kalman → järgmine päev · ilma ilmata · strict OOS · READ ONLY")
    st.info("Eilset state'i ei kanta automaatselt homsesse. φ, Q ja mõõtmisusaldus hinnatakse igal testpäeval ainult varasemast ajaloost.")

    try:
        harvest = db.get_harvest_history(limit=5000)
        samples = _samples(_events(harvest))
        if samples.empty:
            st.error("Õppimisridu ei tekkinud."); st.stop()
        base_rows = _strict_base(samples)
        daily = _daily_obs(base_rows)
        preds = _strict_state_preds(base_rows, daily)
    except Exception as exc:
        st.exception(exc); st.stop()

    if preds.empty:
        st.error("State-space strict-OOS prognoose ei tekkinud piisavalt."); st.stop()

    allm = _metrics(preds)
    p1,p2 = _halves(preds)
    m1,m2 = _metrics(p1), _metrics(p2)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("OOS päevi", allm["n"])
    c2.metric("BASE MAE", f"{allm['base_mae']:.2f}")
    c3.metric("KALMAN MAE", f"{allm['kalman_mae']:.2f}", delta=f"{allm['base_mae']-allm['kalman_mae']:+.2f} kasti parem")
    c4.metric("MAPE", f"{allm['base_mape']:.1f}% → {allm['kalman_mape']:.1f}%")
    c5.metric("KALMAN võidab", f"{allm['wins']}/{allm['n']} päeva")

    good_all = np.isfinite(allm["imp"]) and allm["imp"] >= 10.0 and allm["wins"] >= math.ceil(0.55*allm["n"])
    good_halves = np.isfinite(m1["imp"]) and np.isfinite(m2["imp"]) and m1["imp"] > 0 and m2["imp"] > 0
    if good_all and good_halves:
        st.success(f"✅ STATE-SPACE TOETATUD: MAE paraneb {allm['imp']:.1f}% ({allm['base_mae']:.2f} → {allm['kalman_mae']:.2f}) ja eelis püsib mõlemas ajapooles.")
    elif np.isfinite(allm["imp"]) and allm["imp"] > 0:
        st.warning(f"🟡 Üldine eelis {allm['imp']:+.1f}%, aga mõlema ajapoole stabiilsus pole piisav.")
    else:
        st.error(f"❌ STATE-SPACE EI PARANDA BASE'i: MAE {allm['base_mae']:.2f} → {allm['kalman_mae']:.2f}.")

    st.markdown("### 1. Kõige tähtsam kontroll · kaks ajapoolt")
    halves = pd.DataFrame([
        {"Periood":"I pool","N":m1["n"],"BASE MAE":m1["base_mae"],"KALMAN MAE":m1["kalman_mae"],"Paranemine %":m1["imp"],"Võite":m1["wins"],"BASE MAPE %":m1["base_mape"],"KALMAN MAPE %":m1["kalman_mape"]},
        {"Periood":"II pool","N":m2["n"],"BASE MAE":m2["base_mae"],"KALMAN MAE":m2["kalman_mae"],"Paranemine %":m2["imp"],"Võite":m2["wins"],"BASE MAPE %":m2["base_mape"],"KALMAN MAPE %":m2["kalman_mape"]},
    ])
    st.dataframe(halves.style.format({"BASE MAE":"{:.2f}","KALMAN MAE":"{:.2f}","Paranemine %":lambda x:"—" if pd.isna(x) else f"{x:+.1f}%","BASE MAPE %":"{:.1f}","KALMAN MAPE %":"{:.1f}"}), use_container_width=True, hide_index=True)

    st.markdown("### 2. Päev-päevalt")
    show = preds[["date","fields","actual","base","kalman","base_error","kalman_error","state_pred","state_pred_sd","phi","last_filtered_state","last_gain","observed_target_state","observed_target_sd","n_state_days"]].copy()
    show["Parandus"] = show["base_error"].abs() - show["kalman_error"].abs()
    st.dataframe(show.style.format({
        "date":lambda x:x.strftime("%d.%m"),"actual":"{:.1f}","base":"{:.1f}","kalman":"{:.1f}","base_error":"{:+.1f}","kalman_error":"{:+.1f}","Parandus":"{:+.1f}",
        "state_pred":"{:+.3f}","state_pred_sd":"{:.3f}","phi":"{:+.3f}","last_filtered_state":"{:+.3f}","last_gain":"{:.3f}","observed_target_state":"{:+.3f}","observed_target_sd":"{:.3f}",
    }), use_container_width=True, hide_index=True)
    st.caption("state_pred = ühe päeva ette latentse crop-state prognoos enne target-päeva saagi nägemist; observed_target_state on ainult järelkontroll.")

    focus = preds[(preds["date"] >= date(2026,8,14)) & (preds["date"] <= date(2026,8,24))].copy()
    if not focus.empty:
        st.markdown("### 3. 14.–24.08 · probleemne pööre")
        fs = focus[["date","actual","base","kalman","state_pred","phi","last_filtered_state","observed_target_state","observed_target_sd"]]
        st.dataframe(fs.style.format({"date":lambda x:x.strftime("%d.%m"),"actual":"{:.1f}","base":"{:.1f}","kalman":"{:.1f}","state_pred":"{:+.3f}","phi":"{:+.3f}","last_filtered_state":"{:+.3f}","observed_target_state":"{:+.3f}","observed_target_sd":"{:.3f}"}), use_container_width=True, hide_index=True)

    with st.expander("Kontrolliks · daily-state vaatlusmüra", expanded=False):
        diag = daily.copy()
        diag["obs_sd_mean"] = np.sqrt(diag["obs_var"])
        diag["state_multiplier"] = np.exp(diag["z_state"])
        st.dataframe(diag.style.format({"date":lambda x:x.strftime("%d.%m"),"actual":"{:.1f}","base":"{:.1f}","z_state":"{:+.3f}","state_sd":"{:.3f}","obs_var":"{:.4f}","obs_sd_mean":"{:.3f}","state_multiplier":"{:.3f}"}), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("AUDIT LOCK: BASE treenib alati date < target. φ ja Q hinnatakse igal target-päeval ainult varasematest daily-state vaatlustest. Target actual ei sisene prognoosi.")
    st.caption("BASE ei kasuta eelmise sama põllu saaki. Weather puudub täielikult. Measurement-noise tuleb kolme põllu residualide omavahelisest hajuvusest.")
    st.caption("READ ONLY: ainult db.get_harvest_history. DB kirjutamisi ei ole.")

if __name__ == "__main__":
    main()
