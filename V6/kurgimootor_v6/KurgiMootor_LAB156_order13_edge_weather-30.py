from __future__ import annotations

"""
KurgiMootor · edge_weather-30
=============================

FROZEN PRODUCTION TAIMEINDEKS × INTERVAL-BASE × CONSERVATIVE WEATHER
26.–27.08 TRUE FORWARD HOLDOUT · READ ONLY

One narrow question
-------------------
Did the already-existing production plant index, frozen on 24.08, improve the
interval-aware BASE on the real 26.–27.08 holdout — and does the SAME locked
conservative weather-delta add anything on top?

No new parameter is fitted to 26.–27.08.

Variants
--------
1) BASE
   edge_weather-28 interval-aware weatherless BASE, field by field.

2) BASE × PI24
   Each target field's BASE is multiplied by its exact app-128 production
   plant index reconstructed and frozen at the evening of 24.08.

3) BASE × PI24 × WX-CAP
   Same frozen field-specific plant index, then ONLY the short-term weather
   transition delta from edge_weather-28 is applied as exp(wx_delta).
   The weather delta remains capped at ±0.15 log units exactly as in -28.

Important architectural point
-----------------------------
STATE3 is NOT added to the PI variants. The purpose is to test whether the
production plant index can serve as the slow crop-level state, while weather
contributes only a short-term change. Adding STATE3 as well would double-count
crop state.

For reference only, the old -28 STATE3+WX-CAP score is displayed beside the
new variants, but it is not used in their calculation.

Locks
-----
- PI reconstruction uses only 15.–24.08 harvests and exact app-128 raw locks.
- 25.08 is not treated as zero yield.
- The -28 weather-transition mechanism and cap are unchanged.
- 26.–27.08 actual A+B+C are used only after predictions for scoring.
- No DB writes.

This file intentionally reuses the already-audited machinery in:
  KurgiMootor_LAB156_order13_edge_weather-28.py
  KurgiMootor_LAB156_order13_edge_weather-29.py
so this LAB stays small and does not duplicate ~2000 lines of locked code.
"""

from datetime import timedelta
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

import db

HERE = Path(__file__).resolve().parent
LAB28_NAME = "KurgiMootor_LAB156_order13_edge_weather-28.py"
LAB29_NAME = "KurgiMootor_LAB156_order13_edge_weather-29.py"


def _load_lab(filename: str, module_name: str):
    path = HERE / filename
    if not path.exists():
        raise RuntimeError(
            f"Vajalik lukustatud alusfail puudub: {filename}. "
            "-30 ei kopeeri vana mudelikoodi, vaid kasutab auditeeritud -28/-29 faile otse."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Ei suutnud moodulit laadida: {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _score(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(pred - actual))),
        "MAPE %": float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 0.5)) * 100.0),
    }


def _fmt_day(v):
    try:
        return v.strftime("%d.%m")
    except Exception:
        return str(v)


def main():
    st.set_page_config(
        page_title="KurgiMootor · PI24 × interval-base × weather",
        layout="wide",
    )
    st.title("24.08 frozen taimeindeks × intervall × ilm")
    st.caption("26.–27.08 true forward holdout · no new tuning · READ ONLY")

    st.info(
        "Üks test, kolm varianti. BASE on sama interval-aware BASE nagu -28. "
        "PI24 on app-128 päris production taimeindeks, külmutatud 24.08. "
        "WX-CAP on sama -28 konservatiivne weather-delta ±0.15. "
        "STATE3 ei lähe PI-varianti juurde — muidu loeksime crop-state'i kaks korda."
    )

    try:
        m28 = _load_lab(LAB28_NAME, "kurgimootor_edge28_locked")
        m29 = _load_lab(LAB29_NAME, "kurgimootor_edge29_locked")

        # -------------------------------------------------------------
        # 1) Exact frozen production plant index through 24.08 only.
        # -------------------------------------------------------------
        harvest = db.get_harvest_history(limit=5000)
        hdf = m29._normalise_harvest(harvest)
        raw_map, raw_status = m29._load_raw_setting()
        idx, last_event, trace, eligible, used, missing = m29._reconstruct_index(
            hdf, raw_map, raw_status
        )

        if eligible <= 0:
            raise RuntimeError("24.08 taimeindeksi rekonstruktsioonis pole ühtegi sündmust.")
        if used != eligible or missing:
            raise RuntimeError(
                f"Exact PI reconstruction incomplete: {used}/{eligible}; missing={len(missing)}. "
                "Fallbacki selles otsustestis ei kasutata."
            )
        if not trace.empty:
            bad = [d for d in trace["date"].tolist() if d > m29.FREEZE_DAY]
            if bad:
                raise RuntimeError(f"PI leakage lock failed: {bad}")

        # -------------------------------------------------------------
        # 2) Re-run the unchanged -28 forward machinery.
        #    This gives field-specific interval BASE and the locked
        #    weather transition deltas. No new model choice here.
        # -------------------------------------------------------------
        events = m28._events(harvest)
        intervals = m28._build_intervals(events)
        if intervals.empty:
            raise RuntimeError("Korjeintervalle ei tekkinud.")

        event25 = [e for e in events if e.day == m28.HOLDOUT_GAP_DAY]
        target_counts = {
            dd: len([e for e in events if e.day == dd])
            for dd in m28.HOLDOUT_DAYS
        }

        earliest = min(intervals["target_date"])
        latest_weather_needed = m28.HOLDOUT_DAYS[-1] - timedelta(days=1)
        weather_from = max(
            m28.WEATHER_START,
            earliest - timedelta(days=2 * m28.WEATHER_BLOCK_DAYS),
        )
        weather = m28._measured_weather(
            db.get_weather_rows(weather_from, latest_weather_needed)
        )

        old_summary, old_days, field_old, bridge, diag = m28._build_holdout(
            events, intervals, weather
        )

        if event25:
            raise RuntimeError(
                f"25.08 holdout assumption failed: DB has {len(event25)} harvest rows."
            )
        if target_counts.get(m28.HOLDOUT_DAYS[0]) != 2 or target_counts.get(m28.HOLDOUT_DAYS[1]) != 2:
            raise RuntimeError(
                "Holdout field count changed: "
                f"26.08={target_counts.get(m28.HOLDOUT_DAYS[0])}, "
                f"27.08={target_counts.get(m28.HOLDOUT_DAYS[1])}."
            )

        # -------------------------------------------------------------
        # 3) New composition only: field BASE × frozen PI24, then
        #    optional short-term weather delta. No STATE3.
        # -------------------------------------------------------------
        wx_by_day = {
            m28.HOLDOUT_DAYS[0]: float(diag["wx26"]["cap_delta"]),
            m28.HOLDOUT_DAYS[1]: float(diag["wx27"]["cap_delta"]),
        }

        work = field_old.copy()
        work["date"] = work["Päev"].map({
            "26.08": m28.HOLDOUT_DAYS[0],
            "27.08 SEQ": m28.HOLDOUT_DAYS[1],
        })
        if work["date"].isna().any():
            raise RuntimeError("-28 field table has an unexpected branch label.")

        work["PI24"] = work["Põld"].map(lambda f: float(idx[int(f)]))
        work["PI last event"] = work["Põld"].map(
            lambda f: (last_event.get(int(f)) or {}).get("date")
        )
        work["PI last actual/raw"] = work["Põld"].map(
            lambda f: (last_event.get(int(f)) or {}).get("actual/raw")
        )
        work["PI pred"] = work["BASE"].astype(float) * work["PI24"].astype(float)
        work["WX cap delta"] = work["date"].map(wx_by_day).astype(float)
        work["WX factor"] = np.exp(work["WX cap delta"].astype(float))
        work["PI+WX pred"] = work["PI pred"] * work["WX factor"]

        work["BASE viga"] = work["BASE"].astype(float) - work["Tegelik"].astype(float)
        work["PI viga"] = work["PI pred"] - work["Tegelik"].astype(float)
        work["PI+WX viga"] = work["PI+WX pred"] - work["Tegelik"].astype(float)

        day = (
            work.groupby("date", as_index=False)
            .agg(
                fields=("Põld", lambda s: ",".join(str(int(x)) for x in s)),
                actual=("Tegelik", "sum"),
                BASE=("BASE", "sum"),
                PI24=("PI pred", "sum"),
                PI24_WX=("PI+WX pred", "sum"),
                wx_cap_delta=("WX cap delta", "first"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        day["BASE viga"] = day["BASE"] - day["actual"]
        day["PI24 viga"] = day["PI24"] - day["actual"]
        day["PI24+WX viga"] = day["PI24_WX"] - day["actual"]
        day["BASE APE %"] = np.abs(day["BASE viga"]) / np.maximum(day["actual"], 0.5) * 100.0
        day["PI24 APE %"] = np.abs(day["PI24 viga"]) / np.maximum(day["actual"], 0.5) * 100.0
        day["PI24+WX APE %"] = np.abs(day["PI24+WX viga"]) / np.maximum(day["actual"], 0.5) * 100.0

        actual = day["actual"].to_numpy(dtype=float)
        s_base = _score(actual, day["BASE"].to_numpy(dtype=float))
        s_pi = _score(actual, day["PI24"].to_numpy(dtype=float))
        s_piwx = _score(actual, day["PI24_WX"].to_numpy(dtype=float))

        summary = pd.DataFrame([
            {"Variant": "BASE", **s_base},
            {"Variant": "BASE × frozen PI24", **s_pi},
            {"Variant": "BASE × frozen PI24 × WX-CAP", **s_piwx},
            {
                "Variant": "REFERENCE: -28 STATE3+WX-CAP",
                "MAE": float(old_summary["cap_mae"]),
                "MAPE %": float(old_summary["cap_mape"]),
            },
        ])
        summary["Paranemine BASE suhtes %"] = (
            (s_base["MAE"] - summary["MAE"]) / s_base["MAE"] * 100.0
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    # -----------------------------------------------------------------
    # UI: deliberately short. One decision table, one day table, one
    # field audit table. No discovery kitchen sink.
    # -----------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PI freeze", "24.08")
    c2.metric("PI reconstruction", f"{used}/{eligible}")
    c3.metric("Holdout", "26–27.08")
    c4.metric("25.08 harvest", "0 rida")

    st.markdown("### 1. Otsustabel · kas frozen taimeindeks töötab päris holdout'is?")
    st.dataframe(
        summary.style.format({
            "MAE": "{:.2f}",
            "MAPE %": "{:.1f}",
            "Paranemine BASE suhtes %": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    pi_beats = s_pi["MAE"] < s_base["MAE"]
    wx_adds = s_piwx["MAE"] < s_pi["MAE"]

    if pi_beats and wx_adds:
        st.success(
            f"✅ FROZEN PI24 LÖÖB BASE'i JA ILM LISAB VEEL: "
            f"MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f} → {s_piwx['MAE']:.2f}. "
            "See toetab arhitektuuri: aeglane field-state = production PI, lühike muutus = weather-delta."
        )
    elif pi_beats:
        st.success(
            f"✅ FROZEN PI24 LÖÖB BASE'i: MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f}. "
            f"Weather-delta {'ei lisa' if not wx_adds else 'lisab'} selles 2-päevases kontrollis."
        )
    else:
        st.error(
            f"❌ FROZEN PI24 EI LÖÖ BASE'i: MAE {s_base['MAE']:.2f} → {s_pi['MAE']:.2f}. "
            "Siis ei tohi taimeindeksit uue arhitektuuri ankruks lihtsalt eeldada."
        )

    st.markdown("### 2. Päev-päevalt")
    show_day = day.rename(columns={
        "date": "Päev",
        "fields": "Põllud",
        "actual": "Tegelik ABC",
        "PI24": "BASE×PI24",
        "PI24_WX": "BASE×PI24×WX",
        "wx_cap_delta": "WX cap delta",
    })
    st.dataframe(
        show_day.style.format({
            "Päev": _fmt_day,
            "Tegelik ABC": "{:.1f}",
            "BASE": "{:.1f}",
            "BASE×PI24": "{:.1f}",
            "BASE×PI24×WX": "{:.1f}",
            "WX cap delta": "{:+.3f}",
            "BASE viga": "{:+.1f}",
            "PI24 viga": "{:+.1f}",
            "PI24+WX viga": "{:+.1f}",
            "BASE APE %": "{:.1f}",
            "PI24 APE %": "{:.1f}",
            "PI24+WX APE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Põllu kaupa · siin on intervall ja päris 24.08 indeks")
    field_show = work[[
        "date", "Põld", "Jrk", "Eelmine korje", "Kalendriintervall p",
        "Order-adjusted growth p", "BASE", "PI24", "PI last event",
        "PI last actual/raw", "PI pred", "WX cap delta", "PI+WX pred", "Tegelik",
        "BASE viga", "PI viga", "PI+WX viga",
    ]].copy()
    field_show = field_show.rename(columns={
        "date": "Päev",
        "PI pred": "BASE×PI24",
        "PI+WX pred": "BASE×PI24×WX",
    })
    st.dataframe(
        field_show.style.format({
            "Päev": _fmt_day,
            "Eelmine korje": _fmt_day,
            "Order-adjusted growth p": "{:.2f}",
            "BASE": "{:.2f}",
            "PI24": "{:.3f}",
            "PI last event": _fmt_day,
            "PI last actual/raw": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
            "BASE×PI24": "{:.2f}",
            "WX cap delta": "{:+.3f}",
            "BASE×PI24×WX": "{:.2f}",
            "Tegelik": "{:.2f}",
            "BASE viga": "{:+.2f}",
            "PI viga": "{:+.2f}",
            "PI+WX viga": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "🔒 LEAKAGE LOCK: PI kasutab ainult 15.–24.08 sündmusi; 25.08 ei ole nullsaak; "
        "weather-fit/cap on -28 järgi muutmata; 26.–27.08 actual kasutatakse ainult skooriks."
    )
    st.caption(
        "27.08 BASE järgib sama operatiivset reeglit nagu -28: 26.08 on selleks hetkeks juba minevik ja võib BASE fit'i sisse minna. "
        "PI24 ise jääb mõlema päeva jaoks 24.08 peale külmutatuks."
    )
    st.caption(
        "READ ONLY. See LAB ei otsi uut akent, ridge'i, cap'i, taimeindeksi alpha't ega ühtegi muud parameetrit."
    )


if __name__ == "__main__":
    main()
