from __future__ import annotations

"""
KurgiMootor LAB-157 — PRODUCTION SNAPSHOT / REPLAY AUDIT
========================================================

READ ONLY.
- Does not write to Supabase.
- Does not refresh weather.
- Does not change production forecasts or app settings.

Purpose
-------
Audit what KurgiMootor production actually forecast historically, using the
saved yield_forecasts snapshots and comparing them with later actual harvests.

This is deliberately a single-purpose LAB:
1) operational daily forecast error by lead,
2) exact-plan subset (forecast fields == actually harvested fields),
3) field-level error where the same field exists in forecast and actual,
4) naive previous-same-field-harvest benchmark,
5) champion / plant-index diagnostics from snapshot basis text.

Important limitation
--------------------
The production MODEL_VERSION was intentionally kept stable across multiple app
patches. Therefore old rows with the same model_version are genuine historical
production snapshots, but the DB does not always identify which app patch
created each row. This LAB audits the real operational production history.
"""

from datetime import date, datetime
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import db


TZ_NAME = "Europe/Tallinn"
DEFAULT_MODEL_VERSION = "v6.5-v18-complete-daily-research-observation-snapshot"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _d(v: Any) -> date | None:
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


def _f(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _confirmed(row: dict) -> bool:
    q = str(row.get("data_quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


def _abc(row: dict) -> float | None:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        return None
    return float(sum(vals))


def _total(row: dict) -> float | None:
    direct = _f(row.get("total"))
    if direct is not None:
        return direct
    abc = _abc(row)
    xl = _f(row.get("xl"))
    if abc is None or xl is None:
        return None
    return abc + xl


def _basis_text(row: dict) -> str:
    return str(row.get("basis") or "")


def _basis_number(basis: str, key: str) -> float | None:
    try:
        m = re.search(rf"(?:^|;\s*){re.escape(key)}=([-+0-9.eE]+)", basis)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _basis_text_value(basis: str, key: str) -> str | None:
    try:
        m = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", basis)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def _fmt_pct(v: Any) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.1f}%"


def _safe_mape(pred: pd.Series, actual: pd.Series) -> float:
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    m = np.isfinite(p) & np.isfinite(a) & (np.abs(a) > 1e-9)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(p[m] - a[m]) / np.abs(a[m])) * 100.0)


def _metric_block(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    if df.empty:
        return {
            "N": 0, "MAE": np.nan, "MAPE %": np.nan, "Bias %": np.nan,
            "±20% sees %": np.nan,
        }
    p = pd.to_numeric(df[pred_col], errors="coerce")
    a = pd.to_numeric(df[actual_col], errors="coerce")
    m = p.notna() & a.notna()
    p = p[m]
    a = a[m]
    if len(a) == 0:
        return {
            "N": 0, "MAE": np.nan, "MAPE %": np.nan, "Bias %": np.nan,
            "±20% sees %": np.nan,
        }
    err = p - a
    ape = (err.abs() / a.abs().replace(0, np.nan))
    return {
        "N": int(len(a)),
        "MAE": float(err.abs().mean()),
        "MAPE %": float(100.0 * ape.mean()),
        "Bias %": float(100.0 * (err.sum() / a.abs().sum())) if float(a.abs().sum()) > 0 else np.nan,
        "±20% sees %": float(100.0 * (ape <= 0.20).mean()),
    }


def _weekday_letter(d: date) -> str:
    return ["E", "T", "K", "N", "R", "L", "P"][d.weekday()]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def _load_data():
    harvests = db.get_harvest_history(limit=5000)
    forecasts = db.get_yield_forecasts(limit=20000) if db.yield_forecasts_available() else []
    return harvests, forecasts


def _actual_tables(harvest_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    field_rows = []
    history_by_field: dict[int, list[tuple[date, float]]] = {}

    parsed = []
    for r in harvest_rows:
        day = _d(r.get("harvest_date"))
        field = r.get("field_no")
        if day is None or field is None or not _confirmed(r):
            continue
        try:
            field = int(field)
        except Exception:
            continue
        abc = _abc(r)
        total = _total(r)
        if abc is None or total is None:
            continue
        parsed.append((day, field, r, abc, total))

    parsed.sort(key=lambda x: (x[0], int(x[2].get("harvest_order") or 99), x[1]))

    prev_lookup = {}
    for day, field, r, abc, total in parsed:
        hist = history_by_field.setdefault(field, [])
        prev_abc = hist[-1][1] if hist else None
        prev_day = hist[-1][0] if hist else None
        prev_lookup[(day, field)] = {
            "prev_abc": prev_abc,
            "prev_day": prev_day,
        }
        field_rows.append({
            "target_date": day,
            "field_no": field,
            "actual_abc": abc,
            "actual_total": total,
            "actual_xl": total - abc,
            "harvest_order": int(r.get("harvest_order") or 99),
            "prev_actual_abc": prev_abc,
            "prev_actual_day": prev_day,
        })
        hist.append((day, abc))

    field_df = pd.DataFrame(field_rows)
    if field_df.empty:
        return field_df, pd.DataFrame(), prev_lookup

    daily_rows = []
    for day, g in field_df.groupby("target_date", sort=True):
        fields = tuple(sorted(int(x) for x in g["field_no"].tolist()))
        naive_ok = g["prev_actual_abc"].notna().all()
        daily_rows.append({
            "target_date": day,
            "actual_abc": float(g["actual_abc"].sum()),
            "actual_total": float(g["actual_total"].sum()),
            "actual_fields": fields,
            "actual_n_fields": len(fields),
            "naive_prev_abc": float(g["prev_actual_abc"].sum()) if naive_ok else np.nan,
        })
    daily_df = pd.DataFrame(daily_rows).sort_values("target_date").reset_index(drop=True)
    return field_df, daily_df, prev_lookup


def _forecast_table(rows: list[dict], model_version: str) -> pd.DataFrame:
    out = []
    for r in rows:
        if str(r.get("model_version") or "") != model_version:
            continue
        fd = _d(r.get("forecast_date"))
        td = _d(r.get("target_date"))
        try:
            field = int(r.get("field_no"))
        except Exception:
            continue
        abc = _f(r.get("abc_forecast"))
        total = _f(r.get("total_forecast"))
        xl = _f(r.get("xl_forecast"))
        lead = r.get("lead_days")
        try:
            lead = int(lead) if lead is not None else (td - fd).days
        except Exception:
            lead = None
        if fd is None or td is None or abc is None or total is None or lead is None:
            continue

        basis = _basis_text(r)
        plant_index = _basis_number(basis, "taimeindeks")
        raw_total = _basis_number(basis, "raw_total")
        champion = _basis_text_value(basis, "champion")
        if champion is None:
            # Older basis forms may begin with a free-text model name.
            champion = "—"

        out.append({
            "forecast_date": fd,
            "target_date": td,
            "field_no": field,
            "lead": lead,
            "pred_abc": abc,
            "pred_xl": xl,
            "pred_total": total,
            "interval_days": _f(r.get("interval_days")),
            "basis": basis,
            "champion": champion,
            "plant_index": plant_index if plant_index is not None else 1.0,
            "raw_total": raw_total,
            "generated_at": str(r.get("generated_at") or ""),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["target_date", "forecast_date", "field_no"]).reset_index(drop=True)
    return df


def _join_field(forecast_df: pd.DataFrame, actual_field_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty or actual_field_df.empty:
        return pd.DataFrame()
    return forecast_df.merge(
        actual_field_df,
        on=["target_date", "field_no"],
        how="inner",
        validate="many_to_one",
    )


def _daily_snapshots(forecast_df: pd.DataFrame, actual_daily_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty or actual_daily_df.empty:
        return pd.DataFrame()

    rows = []
    for (fd, td, lead), g in forecast_df.groupby(["forecast_date", "target_date", "lead"], sort=True):
        ffields = tuple(sorted(int(x) for x in g["field_no"].tolist()))
        raw_ok = g["raw_total"].notna().all()
        rows.append({
            "forecast_date": fd,
            "target_date": td,
            "lead": int(lead),
            "pred_abc": float(g["pred_abc"].sum()),
            "pred_total": float(g["pred_total"].sum()),
            "raw_total": float(g["raw_total"].sum()) if raw_ok else np.nan,
            "forecast_fields": ffields,
            "forecast_n_fields": len(ffields),
            "mean_plant_index": float(pd.to_numeric(g["plant_index"], errors="coerce").mean()),
            "champions": " | ".join(sorted(set(str(x) for x in g["champion"] if str(x).strip()))),
        })

    daily = pd.DataFrame(rows).merge(actual_daily_df, on="target_date", how="inner", validate="many_to_one")
    if daily.empty:
        return daily

    daily["plan_exact"] = daily.apply(
        lambda r: tuple(r["forecast_fields"]) == tuple(r["actual_fields"]), axis=1
    )
    daily["plan_overlap"] = daily.apply(
        lambda r: len(set(r["forecast_fields"]) & set(r["actual_fields"])), axis=1
    )
    daily["plan_coverage_pct"] = 100.0 * daily["plan_overlap"] / daily["actual_n_fields"].clip(lower=1)

    daily["abc_error"] = daily["pred_abc"] - daily["actual_abc"]
    daily["abc_ape"] = daily["abc_error"].abs() / daily["actual_abc"].abs().replace(0, np.nan)
    daily["total_error"] = daily["pred_total"] - daily["actual_total"]
    daily["total_ape"] = daily["total_error"].abs() / daily["actual_total"].abs().replace(0, np.nan)

    daily["naive_error"] = daily["naive_prev_abc"] - daily["actual_abc"]
    daily["naive_ape"] = daily["naive_error"].abs() / daily["actual_abc"].abs().replace(0, np.nan)
    daily["production_beats_naive"] = daily["abc_error"].abs() < daily["naive_error"].abs()

    return daily.sort_values(["target_date", "lead", "forecast_date"]).reset_index(drop=True)


def _lead_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    rows = []
    for lead, g in daily.groupby("lead", sort=True):
        prod = _metric_block(g, "pred_abc", "actual_abc")
        exact = g[g["plan_exact"]].copy()
        exact_prod = _metric_block(exact, "pred_abc", "actual_abc")
        naive = _metric_block(exact.dropna(subset=["naive_prev_abc"]), "naive_prev_abc", "actual_abc")
        comparable = exact.dropna(subset=["naive_prev_abc"]).copy()
        win_rate = (
            float(100.0 * comparable["production_beats_naive"].mean())
            if len(comparable) else np.nan
        )
        rows.append({
            "Lead": int(lead),
            "Päevi": prod["N"],
            "Production MAE ABC": prod["MAE"],
            "Production MAPE %": prod["MAPE %"],
            "Bias %": prod["Bias %"],
            "±20% sees %": prod["±20% sees %"],
            "Plaan täpne %": float(100.0 * g["plan_exact"].mean()) if len(g) else np.nan,
            "Exact-plan N": exact_prod["N"],
            "Exact-plan MAE": exact_prod["MAE"],
            "Naive MAE": naive["MAE"],
            "Production võidab naive %": win_rate,
        })
    return pd.DataFrame(rows).sort_values("Lead").reset_index(drop=True)


def _field_lead_table(field_join: pd.DataFrame) -> pd.DataFrame:
    if field_join.empty:
        return pd.DataFrame()
    rows = []
    for lead, g in field_join.groupby("lead", sort=True):
        m = _metric_block(g, "pred_abc", "actual_abc")
        rows.append({
            "Lead": int(lead),
            "Põlluridu": m["N"],
            "MAE ABC": m["MAE"],
            "MAPE %": m["MAPE %"],
            "Bias %": m["Bias %"],
            "±20% sees %": m["±20% sees %"],
        })
    return pd.DataFrame(rows)


def _direction_score(daily: pd.DataFrame, lead: int) -> tuple[int, float]:
    """
    Same-lead day-to-day direction hit. Only truly consecutive target dates count.
    Tiny moves (<2% of previous actual) are treated as flat.
    """
    g = daily[daily["lead"] == lead].sort_values("target_date").copy()
    if len(g) < 2:
        return 0, np.nan

    hits = []
    prev = None
    for _, r in g.iterrows():
        if prev is None:
            prev = r
            continue
        if (r["target_date"] - prev["target_date"]).days != 1:
            prev = r
            continue

        actual_delta = float(r["actual_abc"] - prev["actual_abc"])
        pred_delta = float(r["pred_abc"] - prev["pred_abc"])
        deadband = 0.02 * max(abs(float(prev["actual_abc"])), 1.0)

        def sign(x):
            if abs(x) <= deadband:
                return 0
            return 1 if x > 0 else -1

        hits.append(sign(actual_delta) == sign(pred_delta))
        prev = r

    return len(hits), (100.0 * float(np.mean(hits)) if hits else np.nan)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="KurgiMootor LAB-157", layout="wide")
    st.error("🧪 LAB-157 · PRODUCTION SNAPSHOT / REPLAY AUDIT · READ ONLY")
    st.title("KurgiMootor · production replay audit")
    st.caption(
        "See LAB ei ehita uut saagimudelit. Ta kontrollib productioni päriselt salvestatud "
        "prognoosisnapshotte hiljem saabunud tegelike korjete vastu."
    )

    with st.expander("Mida see audit tõestab — ja mida mitte", expanded=False):
        st.markdown(
            """
- **Tugev külg:** `yield_forecasts` snapshot oli salvestatud enne tegeliku tulemuse teadmist. Siin ei saa tänane saak eilset prognoosi tagantjärele parandada.
- **Lead 0:** production lukustab tänase snapshoti enne esimese tegeliku korje sisestamist.
- **Lead +1…+9:** iga forecast-date → target-date kombinatsioon hinnatakse eraldi.
- **Plaaniviga:** päevatulemust näitame ka siis, kui prognoositud põllud ei langenud tegelikega kokku; lisaks on eraldi **exact-plan** alamhulk.
- **Naive benchmark:** sama target-päeva tegelike põldude eelmiste korjete A+B+C summa. See ei ole uus mootor, vaid kontroll, kas production lisab lihtsale mälule väärtust.
- **Piirang:** productioni `model_version` oli teadlikult mitme app-patchi jooksul sama. Seega DB tõestab päris operational productioni ajaloo, kuid vanade ridade juures ei pruugi olla võimalik öelda täpset `app-12x` patchi.
            """
        )

    if st.button("Värskenda DB andmed", type="secondary"):
        _load_data.clear()
        st.rerun()

    try:
        harvest_rows, forecast_rows = _load_data()
    except Exception as exc:
        st.error(f"DB lugemine ebaõnnestus: {exc}")
        st.stop()

    if not forecast_rows:
        st.error("yield_forecasts on tühi või tabel pole saadaval.")
        st.stop()

    versions = sorted(set(str(r.get("model_version") or "") for r in forecast_rows if r.get("model_version")))
    default_idx = versions.index(DEFAULT_MODEL_VERSION) if DEFAULT_MODEL_VERSION in versions else max(0, len(versions) - 1)
    selected_version = st.selectbox("Production model_version", versions, index=default_idx)

    actual_field, actual_daily, _prev = _actual_tables(harvest_rows)
    fc = _forecast_table(forecast_rows, selected_version)

    if actual_field.empty:
        st.error("Kinnitatud numbrilisi korjeridu ei leitud.")
        st.stop()
    if fc.empty:
        st.error("Valitud model_version jaoks prognoosisnapshotte ei leitud.")
        st.stop()

    overlap_start = max(fc["target_date"].min(), actual_daily["target_date"].min())
    overlap_end = min(fc["target_date"].max(), actual_daily["target_date"].max())
    if overlap_start > overlap_end:
        st.error("Prognooside ja tegelike korjete kuupäevad ei kattu.")
        st.stop()

    c1, c2 = st.columns(2)
    start_day = c1.date_input("Algus", value=overlap_start, min_value=overlap_start, max_value=overlap_end)
    end_day = c2.date_input("Lõpp", value=overlap_end, min_value=overlap_start, max_value=overlap_end)
    if start_day > end_day:
        st.error("Algus ei saa olla lõpust hilisem.")
        st.stop()

    fc = fc[(fc["target_date"] >= start_day) & (fc["target_date"] <= end_day)].copy()
    af = actual_field[(actual_field["target_date"] >= start_day) & (actual_field["target_date"] <= end_day)].copy()
    ad = actual_daily[(actual_daily["target_date"] >= start_day) & (actual_daily["target_date"] <= end_day)].copy()

    field_join = _join_field(fc, af)
    daily = _daily_snapshots(fc, ad)
    lead_table = _lead_table(daily)
    field_leads = _field_lead_table(field_join)

    st.markdown("### 1. Kas production oli päriselt täpne?")
    a, b, c, d = st.columns(4)
    a.metric("Snapshot-ridu", f"{len(fc)}")
    b.metric("Skooritavaid põlluridu", f"{len(field_join)}")
    c.metric("Skooritavaid päev-snapshotte", f"{len(daily)}")
    d.metric("Tegelike korjepäevi", f"{ad['target_date'].nunique()}")

    if daily.empty:
        st.warning("Valitud perioodil ei ole veel ühtegi snapshot → actual paari.")
        st.stop()

    lead0 = daily[daily["lead"] == 0].copy()
    if not lead0.empty:
        m0 = _metric_block(lead0, "pred_abc", "actual_abc")
        exact0 = lead0[lead0["plan_exact"]]
        m0e = _metric_block(exact0, "pred_abc", "actual_abc")
        comp0 = exact0.dropna(subset=["naive_prev_abc"]).copy()
        naive0 = _metric_block(comp0, "naive_prev_abc", "actual_abc")
        n_dir, dir_hit = _direction_score(daily, 0)

        x1, x2, x3, x4, x5 = st.columns(5)
        x1.metric("Lead 0 MAE", "—" if pd.isna(m0["MAE"]) else f"{m0['MAE']:.1f} kasti")
        x2.metric("Lead 0 MAPE", _fmt_pct(m0["MAPE %"]))
        x3.metric("±20% sees", _fmt_pct(m0["±20% sees %"]))
        x4.metric("Plaan täpne", _fmt_pct(100.0 * lead0["plan_exact"].mean()))
        x5.metric(
            "Pöörde/tabamuse suund",
            "—" if pd.isna(dir_hit) else f"{dir_hit:.0f}%",
            help=f"Consecutive target-päevade same-lead suund; N={n_dir}.",
        )

        if len(comp0):
            p1, p2, p3 = st.columns(3)
            p1.metric("Exact-plan production MAE", f"{m0e['MAE']:.1f}")
            p2.metric("Naive eelmise korje MAE", f"{naive0['MAE']:.1f}")
            win = 100.0 * comp0["production_beats_naive"].mean()
            p3.metric("Production võidab naive'i", f"{win:.0f}% päevadest")

            if m0e["MAE"] < naive0["MAE"]:
                st.success(
                    f"Lead 0 exact-plan päevadel lisab production lihtsale eelmise-korje benchmarkile väärtust: "
                    f"MAE {m0e['MAE']:.2f} vs {naive0['MAE']:.2f}."
                )
            elif m0e["MAE"] > naive0["MAE"]:
                st.warning(
                    f"Lead 0 exact-plan päevadel ei löö production praegu lihtsat eelmise-korje benchmarki: "
                    f"MAE {m0e['MAE']:.2f} vs {naive0['MAE']:.2f}."
                )

    st.markdown("### 2. Täpsus lead'i kaupa")
    show_leads = lead_table.copy()
    if not show_leads.empty:
        st.dataframe(
            show_leads.style.format({
                "Production MAE ABC": "{:.2f}",
                "Production MAPE %": "{:.1f}%",
                "Bias %": "{:+.1f}%",
                "±20% sees %": "{:.0f}%",
                "Plaan täpne %": "{:.0f}%",
                "Exact-plan MAE": "{:.2f}",
                "Naive MAE": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                "Production võidab naive %": lambda v: "—" if pd.isna(v) else f"{v:.0f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Production MAE/MAPE = operatiivne päevanumber isegi siis, kui põlluplaan eksis. "
        "Exact-plan MAE = ainult päevad, mil prognoositud põllukomplekt kattus tegelikuga."
    )

    st.markdown("### 3. Päev-päevalt — kus mootor mööda pani?")
    available_leads = sorted(int(x) for x in daily["lead"].unique())
    default_lead_idx = available_leads.index(0) if 0 in available_leads else 0
    selected_lead = st.selectbox("Näita lead'i", available_leads, index=default_lead_idx)

    dd = daily[daily["lead"] == selected_lead].copy().sort_values("target_date")
    dd["Päev"] = dd["target_date"].map(lambda x: f"{_weekday_letter(x)} {x.strftime('%d.%m')}")
    dd["Prognoos ABC"] = dd["pred_abc"]
    dd["Tegelik ABC"] = dd["actual_abc"]
    dd["Viga kasti"] = dd["abc_error"]
    dd["Viga %"] = 100.0 * dd["abc_error"] / dd["actual_abc"].replace(0, np.nan)
    dd["Plaan"] = dd.apply(
        lambda r: "✓" if r["plan_exact"] else f"{','.join(map(str,r['forecast_fields']))} → {','.join(map(str,r['actual_fields']))}",
        axis=1,
    )
    dd["Taimeindeks"] = dd["mean_plant_index"]
    dd["Champion"] = dd["champions"]

    st.dataframe(
        dd[[
            "Päev", "Prognoos ABC", "Tegelik ABC", "Viga kasti", "Viga %",
            "Plaan", "Taimeindeks", "Champion"
        ]].style.format({
            "Prognoos ABC": "{:.1f}",
            "Tegelik ABC": "{:.1f}",
            "Viga kasti": "{:+.1f}",
            "Viga %": "{:+.1f}%",
            "Taimeindeks": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if len(dd):
        worst = dd.loc[dd["abc_error"].abs().idxmax()]
        st.info(
            f"Selle lead'i suurim päevaviga: {worst['target_date'].strftime('%d.%m')} — "
            f"prognoos {worst['pred_abc']:.1f}, tegelik {worst['actual_abc']:.1f}, "
            f"viga {worst['abc_error']:+.1f} kasti."
        )

    st.markdown("### 4. Põllutaseme audit")
    if field_leads.empty:
        st.info("Põllutasemel kattuvaid forecast → actual ridu pole.")
    else:
        st.dataframe(
            field_leads.style.format({
                "MAE ABC": "{:.2f}",
                "MAPE %": "{:.1f}%",
                "Bias %": "{:+.1f}%",
                "±20% sees %": "{:.0f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fj = field_join[field_join["lead"] == selected_lead].copy()
        if not fj.empty:
            fj["abs_err"] = (fj["pred_abc"] - fj["actual_abc"]).abs()
            field_stats = []
            for field, g in fj.groupby("field_no"):
                m = _metric_block(g, "pred_abc", "actual_abc")
                field_stats.append({
                    "Põld": int(field),
                    "N": m["N"],
                    "MAE": m["MAE"],
                    "MAPE %": m["MAPE %"],
                    "Bias %": m["Bias %"],
                    "±20% sees %": m["±20% sees %"],
                })
            fstats = pd.DataFrame(field_stats).sort_values(["MAE", "Põld"], ascending=[False, True])
            st.dataframe(
                fstats.style.format({
                    "MAE": "{:.2f}",
                    "MAPE %": "{:.1f}%",
                    "Bias %": "{:+.1f}%",
                    "±20% sees %": "{:.0f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 5. Taimeindeks — parandas või halvendas?")
    idx = daily[
        daily["raw_total"].notna()
        & (daily["mean_plant_index"] < 0.999)
    ].copy()

    if idx.empty:
        st.info(
            "Valitud perioodil ei ole piisavalt snapshotte, kus basis sisaldaks korraga "
            "`raw_total` ja taimeindeksi mõju. See osa ei tee oletusi."
        )
    else:
        idx["raw_err"] = idx["raw_total"] - idx["actual_total"]
        idx["adj_err"] = idx["pred_total"] - idx["actual_total"]
        idx["index_helped"] = idx["adj_err"].abs() < idx["raw_err"].abs()

        q1, q2, q3 = st.columns(3)
        q1.metric("Indeksiga snapshotte", str(len(idx)))
        q2.metric("Indeks parandas", f"{100.0 * idx['index_helped'].mean():.0f}%")
        q3.metric(
            "MAE muutus",
            f"{idx['raw_err'].abs().mean():.2f} → {idx['adj_err'].abs().mean():.2f}",
            help="Raw total MAE → taimeindeksiga production total MAE",
        )

        iv = idx.copy()
        iv["Päev"] = iv["target_date"].map(lambda x: x.strftime("%d.%m"))
        iv["Raw kokku"] = iv["raw_total"]
        iv["Production kokku"] = iv["pred_total"]
        iv["Tegelik kokku"] = iv["actual_total"]
        iv["Indeks"] = iv["mean_plant_index"]
        iv["Aitas"] = iv["index_helped"].map({True: "✓", False: "✗"})
        st.dataframe(
            iv[["Päev", "lead", "Raw kokku", "Production kokku", "Tegelik kokku", "Indeks", "Aitas"]]
            .style.format({
                "Raw kokku": "{:.1f}",
                "Production kokku": "{:.1f}",
                "Tegelik kokku": "{:.1f}",
                "Indeks": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 6. Audit-lukk")
    checks = []

    # Same target/field/model should have unique forecast_date rows after DB upsert.
    dup_n = int(fc.duplicated(["forecast_date", "target_date", "field_no"]).sum())
    checks.append(("Duplicate snapshot key", dup_n == 0, f"{dup_n} dubletti"))

    future_leak = int((fc["forecast_date"] > fc["target_date"]).sum())
    checks.append(("Forecast date ≤ target date", future_leak == 0, f"{future_leak} rikkumist"))

    lead_mismatch = int(
        sum(
            1 for _, r in fc.iterrows()
            if int(r["lead"]) != (r["target_date"] - r["forecast_date"]).days
        )
    )
    checks.append(("Lead vastab kuupäevade vahele", lead_mismatch == 0, f"{lead_mismatch} rikkumist"))

    audit_df = pd.DataFrame([
        {"Kontroll": name, "Seis": "✓" if ok else "✗", "Detail": detail}
        for name, ok, detail in checks
    ])
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    if all(ok for _, ok, _ in checks):
        st.success("Snapshot-võtmete ja lead-kuupäevade struktuur on korras.")
    else:
        st.error("Audit leidis snapshot-struktuuris vea. Enne täpsusjäreldusi tuleb see lahendada.")

    st.caption(
        "LAB-157 on READ ONLY. Järgmine aste, kui seda on vaja, on current app-128 mootori "
        "ajaline rekonstruktsioon cutoff-haaval; seda ei segata siia snapshot-auditi sisse."
    )


if __name__ == "__main__":
    main()
