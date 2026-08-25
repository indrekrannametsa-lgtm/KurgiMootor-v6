from __future__ import annotations

"""
KurgiMootor LAB-159 · PRODUCTION SNAPSHOT REPLAY AUDIT
======================================================

Purpose
-------
Explain why the saved production forecast on 21–23 Aug was much lower than the
strict OOS WD result seen in LAB-158.

This LAB does NOT fit a model and does NOT fetch weather. It only reads already
saved production forecast snapshots + harvest history and decomposes the saved
A+B+C forecast into:

    raw A+B+C before plant index  ->  saved A+B+C after plant index

The production app stores `taimeindeks=<value>` in each snapshot basis string.
Because production applies the same plant index multiplicatively to A+B+C and XL,
we can reconstruct raw A+B+C exactly as abc_forecast / taimeindeks.

READ ONLY. No DB writes. No API calls. No CPU-heavy walk-forward.
Filename intentionally stays the old LAB filename so the existing Streamlit LAB
app can be overwritten without changing Main file path.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-159-PRODUCTION-SNAPSHOT-REPLAY-V1"
WD_VERSION_HINT = "winddry"
FOCUS_DAYS = [date(2026, 8, 21), date(2026, 8, 22), date(2026, 8, 23)]


def _d(v) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
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


def _basis_number(basis, key: str) -> Optional[float]:
    try:
        m = re.search(rf"(?:^|;\s*){re.escape(key)}=([-+0-9.eE]+)", str(basis or ""))
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _is_prod_wd(row: dict) -> bool:
    mv = str(row.get("model_version") or "").strip().lower()
    return WD_VERSION_HINT in mv and "lab" not in mv and "test" not in mv


def _snapshot_key(row: dict) -> Tuple[str, str, str]:
    return (
        str(row.get("forecast_date") or ""),
        str(row.get("generated_at") or ""),
        str(row.get("model_version") or ""),
    )


def _batches_for_target(rows: Iterable[dict], target: date) -> List[Tuple[Tuple[str, str, str], List[dict]]]:
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in rows:
        if _d(r.get("target_date")) != target or not _is_prod_wd(r):
            continue
        groups.setdefault(_snapshot_key(r), []).append(r)
    out = []
    for key, batch in groups.items():
        # Full production day = exactly three unique fields.
        fields = {int(r.get("field_no")) for r in batch if _f(r.get("field_no")) is not None}
        if len(fields) == 3:
            out.append((key, batch))
    out.sort(key=lambda kv: kv[0])
    return out


def _actual_abc_by_day(harvest_rows: Iterable[dict]) -> Dict[date, float]:
    by_day: Dict[date, Dict[int, float]] = {}
    for r in harvest_rows:
        dd = _d(r.get("harvest_date"))
        ff = _f(r.get("field_no"))
        aa, bb, cc = (_f(r.get(k)) for k in ("a", "b", "c"))
        if dd is None or ff is None or None in (aa, bb, cc):
            continue
        by_day.setdefault(dd, {})[int(ff)] = float(aa + bb + cc)
    return {dd: float(sum(vals.values())) for dd, vals in by_day.items() if len(vals) == 3}


def _summarize_batch(target: date, key: Tuple[str, str, str], batch: List[dict], actual_abc: Optional[float]) -> dict:
    forecast_date_s, generated_at, model_version = key
    fd = _d(forecast_date_s)
    lead = (target - fd).days if fd else None

    saved_abc = 0.0
    raw_abc = 0.0
    idx_values = []
    field_rows = []
    exact = True

    for r in sorted(batch, key=lambda x: int(x.get("field_no") or 999)):
        abc = _f(r.get("abc_forecast"))
        idx = _basis_number(r.get("basis"), "taimeindeks")
        raw_total = _basis_number(r.get("basis"), "raw_total")
        if abc is None:
            exact = False
            continue
        saved_abc += abc
        if idx is None or idx <= 0:
            exact = False
            raw = None
        else:
            raw = abc / idx
            raw_abc += raw
            idx_values.append(idx)
        field_rows.append({
            "Põld": int(r.get("field_no")),
            "Issue": forecast_date_s,
            "Lead": lead,
            "Ametlik ABC": abc,
            "Taimeindeks": idx,
            "Raw ABC": raw,
            "Maha võetud": (raw - abc) if raw is not None else None,
            "Raw total basis": raw_total,
            "Model version": model_version,
            "Generated": generated_at,
        })

    effective_idx = (saved_abc / raw_abc) if exact and raw_abc > 0 else None
    actual = actual_abc
    official_abs_err = abs(saved_abc - actual) if actual is not None else None
    raw_abs_err = abs(raw_abc - actual) if actual is not None and exact else None

    return {
        "Kuupäev": target,
        "Issue": fd,
        "Lead p": lead,
        "Tegelik ABC": actual,
        "Ametlik ABC": saved_abc,
        "Raw ABC enne indeksit": raw_abc if exact else None,
        "Efektiivne taimeindeks": effective_idx,
        "Indeks vähendas": (raw_abc - saved_abc) if exact else None,
        "Ametlik abs viga": official_abs_err,
        "Raw abs viga": raw_abs_err,
        "Raw parem?": (raw_abs_err < official_abs_err) if raw_abs_err is not None and official_abs_err is not None else None,
        "Model version": model_version,
        "Generated": generated_at,
        "_fields": field_rows,
        "_exact": exact,
    }


def main() -> None:
    import db

    st.set_page_config(page_title="KurgiMootor LAB-159", layout="wide")
    st.title("LAB-159 · Production replay audit")
    st.caption("Miks oli päris äpi WD prognoos madalam kui LAB-158 WD? · ainult salvestatud snapshotid · READ ONLY")
    st.info("See LAB ei treeni mudelit ega lae ilma. Ta loeb ainult juba salvestatud production-snapshotte ja korjeid.")

    try:
        saved = db.get_yield_forecasts(limit=5000) if db.yield_forecasts_available() else []
        harvest = db.get_harvest_history(limit=5000)
    except Exception as exc:
        st.error(f"DB lugemine ebaõnnestus: {exc}")
        st.stop()

    prod_rows = [r for r in saved if _is_prod_wd(r)]
    st.caption(f"Leitud {len(prod_rows)} WIND×DRY production snapshot-rida · versioon {LAB_VERSION}")
    if not prod_rows:
        st.error("WIND×DRY production snapshotte ei leitud.")
        st.stop()

    actual_map = _actual_abc_by_day(harvest)

    summaries = []
    all_by_day: Dict[date, List[dict]] = {}
    for td in FOCUS_DAYS:
        day_summaries = []
        for key, batch in _batches_for_target(prod_rows, td):
            s = _summarize_batch(td, key, batch, actual_map.get(td))
            day_summaries.append(s)
            summaries.append(s)
        all_by_day[td] = day_summaries

    if not summaries:
        st.error("21.–23.08 jaoks ei leitud ühtegi täielikku 3-põllu WIND×DRY snapshot-batch'i.")
        st.stop()

    st.header("1. Kõik päriselt salvestatud issue-hetked")
    show_cols = [
        "Kuupäev", "Issue", "Lead p", "Tegelik ABC", "Ametlik ABC",
        "Raw ABC enne indeksit", "Efektiivne taimeindeks", "Indeks vähendas",
        "Ametlik abs viga", "Raw abs viga", "Raw parem?", "Model version",
    ]
    sdf = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries])
    st.dataframe(
        sdf[show_cols].style.format({
            "Tegelik ABC": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "Ametlik ABC": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "Raw ABC enne indeksit": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "Efektiivne taimeindeks": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
            "Indeks vähendas": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "Ametlik abs viga": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            "Raw abs viga": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Closest pre-harvest snapshot for each target (latest forecast_date <= target).
    closest = []
    for td in FOCUS_DAYS:
        candidates = [s for s in all_by_day.get(td, []) if s.get("Issue") is not None and s["Issue"] <= td]
        if candidates:
            candidates.sort(key=lambda s: (s["Issue"], str(s.get("Generated") or "")))
            closest.append(candidates[-1])

    st.header("2. Korjele kõige lähem production snapshot")
    if closest:
        cdf = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in closest])
        st.dataframe(
            cdf[[
                "Kuupäev", "Issue", "Lead p", "Tegelik ABC", "Ametlik ABC",
                "Raw ABC enne indeksit", "Efektiivne taimeindeks", "Indeks vähendas",
                "Ametlik abs viga", "Raw abs viga", "Raw parem?",
            ]].style.format({
                "Tegelik ABC": "{:.1f}",
                "Ametlik ABC": "{:.1f}",
                "Raw ABC enne indeksit": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Efektiivne taimeindeks": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
                "Indeks vähendas": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Ametlik abs viga": "{:.1f}",
                "Raw abs viga": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        valid = [s for s in closest if s.get("Raw abs viga") is not None]
        if valid:
            official_mae = float(np.mean([s["Ametlik abs viga"] for s in valid]))
            raw_mae = float(np.mean([s["Raw abs viga"] for s in valid]))
            raw_wins = sum(bool(s.get("Raw parem?")) for s in valid)
            idx_mean = float(np.mean([s["Efektiivne taimeindeks"] for s in valid if s.get("Efektiivne taimeindeks") is not None]))
            a, b, c, d = st.columns(4)
            a.metric("Ametlik MAE", f"{official_mae:.2f}")
            b.metric("Raw MAE", f"{raw_mae:.2f}", delta=f"{official_mae-raw_mae:+.2f} parem")
            c.metric("Raw võidab", f"{raw_wins}/{len(valid)} päeva")
            d.metric("Keskmine taimeindeks", f"{idx_mean:.3f}")

            if raw_mae + 0.20 < official_mae and raw_wins >= 2:
                st.error(
                    "🔎 PEAMINE LAHKNEMINE ON TAIMEINDEKSIS. "
                    "Salvestatud puhas/raw production-potentsiaal oli tegelikule lähem kui pärast taimeindeksit näidatud ametlik ABC."
                )
            elif raw_mae < official_mae:
                st.warning(
                    "Taimeindeks halvendas neid päevi, kuid mõju ei seleta veel üksi kogu lahknevust. "
                    "Vaata all per-field ridu ja issue-hetki."
                )
            else:
                st.success(
                    "Taimeindeks ei paista olevat peamine süüdlane: raw production ei olnud ametlikust parem. "
                    "Siis tuleb järgmises sammus auditeerida issue-hetke ilma / treeningu cutoff'i."
                )

    st.header("3. Põllu kaupa")
    for s in closest:
        td = s["Kuupäev"]
        with st.expander(f"{td.strftime('%d.%m')} · issue {s['Issue']} · lead {s['Lead p']} p", expanded=(td == date(2026, 8, 22))):
            fdf = pd.DataFrame(s.get("_fields") or [])
            if fdf.empty:
                st.caption("Põlluridu pole.")
                continue
            st.dataframe(
                fdf[["Põld", "Ametlik ABC", "Taimeindeks", "Raw ABC", "Maha võetud", "Raw total basis"]].style.format({
                    "Ametlik ABC": "{:.2f}",
                    "Taimeindeks": lambda v: "—" if pd.isna(v) else f"{float(v):.3f}",
                    "Raw ABC": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                    "Maha võetud": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                    "Raw total basis": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.header("4. Mida see test tõestab / ei tõesta")
    st.markdown(
        "- Kui `Raw ABC enne indeksit` on ligikaudu sama suur kui LAB-158 WD ja ametlik number on palju väiksem, "
        "siis lahknemine tekib pärast WD-mootorit — taimeindeksis.\n"
        "- Kui raw production on samuti madal, ei ole taimeindeks põhjus; siis erinevad issue-hetke sisendid või treeningu cutoff.\n"
        "- See audit ei muuda productionit ega arvuta uut prognoosi."
    )


if __name__ == "__main__":
    main()
