from __future__ import annotations

"""
KurgiMootor · edge_weather-29
=============================

PRODUCTION TAIMEINDEKS · 24.08 FREEZE AUDIT · READ ONLY

Purpose
-------
Answer one narrow question only:

    What did app-128's already-existing production plant index know by the
    evening of 24.08, BEFORE the 26.–27.08 harvest outcomes were known?

This LAB does NOT fit a new plant index, does NOT use weather, and does NOT
change any parameter. It reconstructs app-128's exact plant-index update rule:

    start 15.08 at 1.00 for every field
    signal = clip(actual_total / locked_raw_total, 0.50, 1.00)
    new_index = 0.30 * old_index + 0.70 * signal
    index = clip(new_index, 0.50, 1.00)

Only harvest events dated 15.08..24.08 are allowed to update the frozen state.
26.–27.08 actuals are shown only afterwards as holdout diagnostics.

Primary raw-anchor source is the same app_setting used by app-128:
    plant_index_raw_forecasts_2026

If that setting is unavailable/incomplete, the LAB may show a clearly labelled
snapshot fallback parsed from yield_forecasts. It never silently mixes sources.

READ ONLY:
- db.get_harvest_history
- db.get_app_setting
- db.get_yield_forecasts (fallback/diagnostic only)
- no DB writes
- no weather calls
- no model fitting
- no scipy
"""

from datetime import date, datetime
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db

YEAR = 2026
PLANT_INDEX_START = date(YEAR, 8, 15)
FREEZE_DAY = date(YEAR, 8, 24)
HOLDOUT_DAYS = [date(YEAR, 8, 26), date(YEAR, 8, 27)]
PLANT_INDEX_ALPHA = 0.70
PLANT_INDEX_MIN = 0.50
RAW_SETTING_KEY = f"plant_index_raw_forecasts_{YEAR}"


def _d(v: Any) -> Optional[date]:
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


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _i(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _load_raw_setting() -> Tuple[Dict[str, dict], str]:
    """Read the exact app-128 raw anchor map without writing anything."""
    if not hasattr(db, "get_app_setting"):
        return {}, "app_setting API puudub"
    try:
        raw = db.get_app_setting(RAW_SETTING_KEY, "")
    except Exception as exc:
        return {}, f"app_setting lugemine ebaõnnestus: {exc}"
    if not raw:
        return {}, "app_setting on tühi"
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {}, f"app_setting JSON vigane: {exc}"
    if not isinstance(payload, dict):
        return {}, "app_setting ei ole dict"
    cleaned = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    return cleaned, "app-128 plant_index_raw_forecasts_2026"


def _parse_basis_number(text: str, key: str) -> Optional[float]:
    m = re.search(rf"(?:^|[;\s]){re.escape(key)}=([-+0-9.eE]+)", str(text or ""))
    return _f(m.group(1)) if m else None


def _snapshot_fallback_raw() -> Dict[str, dict]:
    """
    Conservative fallback only. Prefer same-day/nearest-to-target production
    snapshots that explicitly contain raw_total in basis. This is labelled as
    fallback and is NOT claimed to be identical to the app_setting lock map.
    """
    if not hasattr(db, "get_yield_forecasts"):
        return {}
    try:
        rows = db.get_yield_forecasts(limit=5000)
    except Exception:
        return {}

    candidates: Dict[str, List[dict]] = {}
    for r in rows or []:
        td = _d(r.get("target_date"))
        fd = _d(r.get("forecast_date"))
        f = _i(r.get("field_no"))
        if td is None or f is None or not (1 <= f <= 14):
            continue
        if td < PLANT_INDEX_START or td > FREEZE_DAY:
            continue
        raw_total = _parse_basis_number(str(r.get("basis") or ""), "raw_total")
        if raw_total is None or raw_total <= 0:
            continue
        lead = _i(r.get("lead_days"))
        if lead is None and fd is not None:
            lead = (td - fd).days
        key = f"{td.isoformat()}|{f}"
        candidates.setdefault(key, []).append({
            "target_date": td.isoformat(),
            "field_no": f,
            "raw_total": raw_total,
            "captured_at": str(r.get("generated_at") or ""),
            "source": "yield_forecasts basis fallback",
            "forecast_date": fd,
            "lead_days": lead if lead is not None else 999,
        })

    out: Dict[str, dict] = {}
    for key, items in candidates.items():
        # Best operational approximation: smallest non-negative lead; then latest forecast date.
        valid = [x for x in items if int(x.get("lead_days", 999)) >= 0]
        if not valid:
            continue
        valid.sort(
            key=lambda x: (
                int(x.get("lead_days", 999)),
                -(x.get("forecast_date").toordinal() if isinstance(x.get("forecast_date"), date) else 0),
            )
        )
        pick = dict(valid[0])
        pick.pop("forecast_date", None)
        out[key] = pick
    return out


def _normalise_harvest(rows: List[dict]) -> pd.DataFrame:
    data = []
    for r in rows or []:
        d = _d(r.get("harvest_date"))
        f = _i(r.get("field_no"))
        order = _i(r.get("harvest_order")) or 99
        total = _f(r.get("total"))
        a, b, c = _f(r.get("a")), _f(r.get("b")), _f(r.get("c"))
        if d is None or f is None or total is None or not (1 <= f <= 14):
            continue
        abc = (a + b + c) if None not in (a, b, c) else None
        data.append({
            "date": d,
            "field": f,
            "order": order,
            "total": total,
            "abc": abc,
            "quality": str(r.get("data_quality") or "").strip().lower(),
        })
    if not data:
        return pd.DataFrame(columns=["date", "field", "order", "total", "abc", "quality"])
    return pd.DataFrame(data).sort_values(["date", "order", "field"]).reset_index(drop=True)


def _reconstruct_index(hdf: pd.DataFrame, raw_map: Dict[str, dict], source_label: str):
    idx = {f: 1.0 for f in range(1, 15)}
    last_event: Dict[int, dict] = {}
    trace = []
    eligible = 0
    used = 0
    missing = []

    hist = hdf[(hdf["date"] >= PLANT_INDEX_START) & (hdf["date"] <= FREEZE_DAY)].copy()
    for _, r in hist.iterrows():
        if str(r["quality"]) in {"hinnanguline", "ligikaudne"}:
            continue
        eligible += 1
        d = r["date"]
        f = int(r["field"])
        actual_total = float(r["total"])
        key = f"{d.isoformat()}|{f}"
        rec = raw_map.get(key)
        if not isinstance(rec, dict):
            missing.append(key)
            continue
        raw_total = _f(rec.get("raw_total"))
        if raw_total is None or raw_total <= 0:
            missing.append(key)
            continue

        ratio = actual_total / raw_total
        signal = max(PLANT_INDEX_MIN, min(1.0, ratio))
        old = float(idx[f])
        new = (1.0 - PLANT_INDEX_ALPHA) * old + PLANT_INDEX_ALPHA * signal
        new = max(PLANT_INDEX_MIN, min(1.0, new))
        idx[f] = new
        used += 1
        ev = {
            "date": d,
            "field": f,
            "order": int(r["order"]),
            "actual_total": actual_total,
            "raw_total": raw_total,
            "actual/raw": ratio,
            "signal": signal,
            "index_before": old,
            "index_after": new,
            "raw_source": str(rec.get("source") or source_label),
            "captured_at": str(rec.get("captured_at") or ""),
        }
        trace.append(ev)
        last_event[f] = ev

    return idx, last_event, pd.DataFrame(trace), eligible, used, missing


def _target_fields(hdf: pd.DataFrame, d: date) -> List[int]:
    x = hdf[hdf["date"] == d].sort_values(["order", "field"])
    return [int(v) for v in x["field"].tolist()]


def _actual_day_abc(hdf: pd.DataFrame, d: date) -> Optional[float]:
    x = hdf[hdf["date"] == d]
    vals = pd.to_numeric(x["abc"], errors="coerce")
    if len(x) == 0 or vals.isna().any():
        return None
    return float(vals.sum())


def _target_audit(hdf: pd.DataFrame, idx: Dict[int, float], last_event: Dict[int, dict]):
    field_rows = []
    day_rows = []
    for d in HOLDOUT_DAYS:
        fields = _target_fields(hdf, d)
        values = []
        for f in fields:
            pi = float(idx.get(f, 1.0))
            values.append(pi)
            ev = last_event.get(f) or {}
            field_rows.append({
                "Target": d,
                "Põld": f,
                "Taimeindeks 24.08": pi,
                "Viimane indeksit uuendanud korje": ev.get("date"),
                "Viimane raw total": ev.get("raw_total"),
                "Viimane tegelik total": ev.get("actual_total"),
                "Viimane actual/raw": ev.get("actual/raw"),
                "Indeks enne": ev.get("index_before"),
                "Indeks pärast": ev.get("index_after"),
            })
        day_rows.append({
            "Target": d,
            "Põllud": ",".join(map(str, fields)) if fields else "—",
            "N põldu": len(fields),
            "Keskmine taimeindeks": float(np.mean(values)) if values else np.nan,
            "Min taimeindeks": float(np.min(values)) if values else np.nan,
            "Max taimeindeks": float(np.max(values)) if values else np.nan,
            "Tegelik ABC (ainult kontrolliks)": _actual_day_abc(hdf, d),
        })
    return pd.DataFrame(field_rows), pd.DataFrame(day_rows)


def main():
    st.set_page_config(page_title="KurgiMootor · production taimeindeks 24.08", layout="wide")
    st.title("Production taimeindeks · mida me 24.08 juba teadsime?")
    st.caption("app-128 exact update rule · cutoff 24.08 · 26.–27.08 actuals only as holdout diagnostic · READ ONLY")

    st.info(
        "See LAB ei loo uut taimeindeksit. Ta taastab app-128 olemasoleva indeksi täpselt 24.08 õhtu seisuga. "
        "26. ja 27.08 tulemused EI TOHI indeksit muuta; neid näidatakse alles pärast freeze'i."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        hdf = _normalise_harvest(harvest)
        if hdf.empty:
            st.error("Korjeajalugu on tühi.")
            st.stop()

        raw_map, primary_status = _load_raw_setting()
        primary_count = len(raw_map)
        source_label = primary_status
        used_fallback = False

        # Reconstruct once from the exact app_setting first.
        idx, last_event, trace, eligible, used, missing = _reconstruct_index(hdf, raw_map, source_label)

        # Only if exact raw anchors are missing, build a separate labelled fallback reconstruction.
        fallback_result = None
        if used < eligible:
            fb = _snapshot_fallback_raw()
            if fb:
                fb_idx, fb_last, fb_trace, fb_eligible, fb_used, fb_missing = _reconstruct_index(
                    hdf, fb, "yield_forecasts fallback"
                )
                fallback_result = (fb_idx, fb_last, fb_trace, fb_eligible, fb_used, fb_missing, fb)

        target_fields, target_days = _target_audit(hdf, idx, last_event)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Freeze", "24.08")
    c2.metric("Indeksi sündmusi", f"{used}/{eligible}")
    c3.metric("Raw ankrud settingus", primary_count)
    c4.metric("Alpha / min", f"{PLANT_INDEX_ALPHA:.2f} / {PLANT_INDEX_MIN:.2f}")

    if used == eligible and eligible > 0:
        st.success(
            "✅ TÄPNE REKONSTRUKTSIOON: kõik 15.–24.08 kvalifitseeruvad korjed leidsid app-128 raw-lock ankru. "
            "Allolev 24.08 taimeindeks on production-loogika järgi taastatav ilma 26.–27.08 teadmiseta."
        )
    else:
        st.warning(
            f"⚠️ Täpsest app_setting raw-mapist leiti {used}/{eligible} vajalikku sündmust. "
            "Puuduvate ankrutega põllud jäävad exact-vaates varasemasse indeksisse; allpool on eraldi fallback, kui see oli võimalik."
        )

    st.markdown("### 1. Kõige tähtsam · 26. ja 27.08 pärispõldude indeks OLI 24.08 selline")
    st.dataframe(
        target_days.style.format({
            "Target": lambda x: x.strftime("%d.%m"),
            "Keskmine taimeindeks": "{:.3f}",
            "Min taimeindeks": "{:.3f}",
            "Max taimeindeks": "{:.3f}",
            "Tegelik ABC (ainult kontrolliks)": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Tegelik ABC 26.–27.08 on ainult kõrvalkontroll. Taimeindeksi veerg on juba enne nende tulemuste vaatamist 24.08 peal külmutatud."
    )

    vals = pd.to_numeric(target_days["Keskmine taimeindeks"], errors="coerce").dropna()
    if len(vals) == 2 and used == eligible:
        mean_holdout_pi = float(vals.mean())
        if 0.68 <= mean_holdout_pi <= 0.82:
            st.success(
                f"✅ Production taimeindeks oli juba 24.08 sihtpõldudel keskmiselt {mean_holdout_pi:.3f}. "
                "See on just selles suurusjärgus, mida 26.–27.08 hilisem BASE→actual langus kaudselt vihjas."
            )
        elif mean_holdout_pi < 0.90:
            st.info(
                f"Production indeks nägi enne holdouti selget hilishooaja langust: sihtpõldude keskmine {mean_holdout_pi:.3f}. "
                "See pole automaatselt tõend, et indeks oli täpselt õige, kuid ta ei olnud 1.00 lähedal."
            )
        else:
            st.warning(
                f"Production indeks oli 24.08 sihtpõldudel endiselt kõrge: keskmine {mean_holdout_pi:.3f}. "
                "Siis ei olnud 26.–27.08 nähtud ~0.75 tasemelangus olemasoleva indeksi poolt ette nähtud."
            )

    st.markdown("### 2. Põllu kaupa · mis info oli enne holdouti olemas")
    st.dataframe(
        target_fields.style.format({
            "Target": lambda x: x.strftime("%d.%m"),
            "Taimeindeks 24.08": "{:.3f}",
            "Viimane indeksit uuendanud korje": lambda v: "—" if pd.isna(v) else v.strftime("%d.%m"),
            "Viimane raw total": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
            "Viimane tegelik total": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
            "Viimane actual/raw": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
            "Indeks enne": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
            "Indeks pärast": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Kõik 14 põldu · freeze 24.08")
    all_rows = []
    for f in range(1, 15):
        ev = last_event.get(f) or {}
        all_rows.append({
            "Põld": f,
            "Taimeindeks 24.08": float(idx[f]),
            "Viimane sündmus": ev.get("date"),
            "actual/raw": ev.get("actual/raw"),
            "raw allikas": ev.get("raw_source"),
        })
    all_df = pd.DataFrame(all_rows)
    st.dataframe(
        all_df.style.format({
            "Taimeindeks 24.08": "{:.3f}",
            "Viimane sündmus": lambda v: "—" if pd.isna(v) else v.strftime("%d.%m"),
            "actual/raw": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 4. Rekonstruktsiooni jälg · ainult 15.–24.08")
    if trace.empty:
        st.error("Exact app_setting raw-mapist ei saanud ühtegi indeksiuuendust taastada.")
    else:
        st.dataframe(
            trace.style.format({
                "date": lambda x: x.strftime("%d.%m"),
                "actual_total": "{:.2f}",
                "raw_total": "{:.2f}",
                "actual/raw": "{:.3f}",
                "signal": "{:.3f}",
                "index_before": "{:.3f}",
                "index_after": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    if missing:
        with st.expander(f"Puuduvad exact raw-ankrud ({len(missing)})"):
            st.write(missing)

    if fallback_result is not None:
        fb_idx, fb_last, fb_trace, fb_eligible, fb_used, fb_missing, fb_map = fallback_result
        st.markdown("### 5. Ainult kontrolliks · yield_forecasts fallback")
        st.warning(
            "See osa EI OLE exact app_setting rekonstruktsioon. Seda näidatakse ainult juhul, kui raw-lock mapis oli auke. "
            "Ära kasuta fallbacki otsuseks, kui exact-vaade on täielik."
        )
        fb_fields, fb_days = _target_audit(hdf, fb_idx, fb_last)
        st.dataframe(
            fb_days.style.format({
                "Target": lambda x: x.strftime("%d.%m"),
                "Keskmine taimeindeks": "{:.3f}",
                "Min taimeindeks": "{:.3f}",
                "Max taimeindeks": "{:.3f}",
                "Tegelik ABC (ainult kontrolliks)": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Fallback coverage: {fb_used}/{fb_eligible}; missing {len(fb_missing)}.")

    # Hard leakage assertions shown to the user.
    leak_dates = []
    if not trace.empty:
        leak_dates = [d for d in trace["date"].tolist() if isinstance(d, date) and d > FREEZE_DAY]
    if leak_dates:
        st.error(f"❌ LEAKAGE: indeksijäljes on pärast 24.08 kuupäevi: {leak_dates}")
    else:
        st.success("🔒 LEAKAGE LOCK OK: ükski indeksiuuendus ei kasuta 25.–27.08 korjet.")

    st.caption(
        "App-128 reegel: kuni 14.08 indeks 1.00; 15.08 alates field-specific actual_total / enne korjet lukustatud raw_total, "
        "signal cap 0.50..1.00, alpha 0.70, indeks 0.50..1.00. See LAB kasutab sama reeglit muutmata."
    )
    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_app_setting; yield_forecasts ainult selgelt märgitud fallbackiks. "
        "Ei ilma, ei fit'i, ei DB kirjutusi."
    )


if __name__ == "__main__":
    main()
