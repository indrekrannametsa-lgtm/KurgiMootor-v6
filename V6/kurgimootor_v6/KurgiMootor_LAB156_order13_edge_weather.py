from __future__ import annotations

"""
KurgiMootor · edge_weather-20
=============================

ROBUST TIMING AUDIT · DECONVOLUTION SMOOTHNESS ROBUSTNESS · READ ONLY

Purpose
-------
LAB-19 supported the interval-sum idea:
harvest is treated as accumulated production since the previous same-field harvest.

This LAB asks only:
    Does the timing of the August production collapse stay in the same place
    when the latent daily curve is reconstructed with four different,
    pre-fixed smoothness levels?

Fixed smoothness levels:
    0.03, 0.10, 0.30, 1.00

No one curve is selected after seeing August.
All four are shown side-by-side.

For each curve we report:
- held-out target-date CV MAE
- date of strongest 3-day production decline during 08–24 Aug
- magnitude of that strongest 3-day decline
- date of minimum 3-day mean production during 08–24 Aug
- minimum 3-day mean index
- field-factor range

Interpretation:
If the strongest decline and trough remain within a narrow date band across
all four smoothness levels, the reconstructed timing is robust enough to use
as a biological time axis in the next weather audit.
If dates move widely, only a broader multi-day window is identifiable.

READ ONLY:
- db.get_harvest_history only
- no weather
- no production snapshots
- no DB writes
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

import db


HOURS_PER_FIELD = 3.0
ALS_ITERS = 20
MIN_EVENTS = 35
CV_FOLDS = 4

SMOOTHS = [0.03, 0.10, 0.30, 1.00]
FOCUS_START = date(2026, 8, 8)
FOCUS_END = date(2026, 8, 24)


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

        out.append(Event(
            day=dd,
            field=field,
            order=order,
            abc=float(abc),
            interval_days=_f(r.get("interval_days")),
        ))

    return sorted(
        out,
        key=lambda e: (e.day, e.order, e.field),
    )


def _field_hist(events: Sequence[Event], field: int):
    return sorted(
        [e for e in events if e.field == field],
        key=lambda e: (e.day, e.order, e.field),
    )


def _growth(prev: Event, cur: Event):
    g = float((cur.day - prev.day).days)
    g += (cur.order - prev.order) * (HOURS_PER_FIELD / 24.0)
    return max(0.5, g)


def _build_intervals(events: List[Event]):
    rows = []

    for field in range(1, 15):
        hist = _field_hist(events, field)

        for i in range(1, len(hist)):
            prev = hist[i - 1]
            cur = hist[i]

            calendar_gap = int((cur.day - prev.day).days)
            if calendar_gap <= 0:
                continue

            growth = _growth(prev, cur)

            days = [
                prev.day + timedelta(days=k)
                for k in range(1, calendar_gap + 1)
            ]

            per_day_weight = growth / len(days)

            rows.append({
                "target_date": cur.day,
                "field": int(field),
                "order": int(cur.order),
                "actual": float(cur.abc),
                "growth": float(growth),
                "start_date": prev.day,
                "days": days,
                "per_day_weight": float(per_day_weight),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["target_date", "order", "field"]
        ).reset_index(drop=True)

    return df


def _day_axis(intervals: pd.DataFrame):
    all_days = sorted({
        dd
        for days in intervals["days"].tolist()
        for dd in days
    })

    return all_days, {
        dd: i
        for i, dd in enumerate(all_days)
    }


def _matrix(intervals, all_days, day_idx):
    A = np.zeros(
        (len(intervals), len(all_days)),
        dtype=float,
    )

    for i, row in intervals.iterrows():
        w = float(row["per_day_weight"])

        for dd in row["days"]:
            A[i, day_idx[dd]] += w

    y = intervals["actual"].to_numpy(dtype=float)
    fields = intervals["field"].to_numpy(dtype=int)

    return A, y, fields


def _second_diff_penalty(n_days):
    if n_days < 3:
        return np.zeros((n_days, n_days), dtype=float)

    D = np.zeros(
        (n_days - 2, n_days),
        dtype=float,
    )

    for i in range(n_days - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    return D.T @ D


def _fit_latent(A, y, fields, smooth_rel):
    n_events, n_days = A.shape

    factors = np.ones(15, dtype=float)

    daily = np.full(
        n_days,
        max(
            float(np.mean(
                y / np.maximum(A.sum(axis=1), 0.5)
            )),
            0.05,
        ),
        dtype=float,
    )

    penalty = _second_diff_penalty(n_days)

    for _ in range(ALS_ITERS):
        row_f = np.asarray(
            [factors[int(f)] for f in fields],
            dtype=float,
        )

        B = A * row_f[:, None]

        diag = np.diag(B.T @ B)
        positive = diag[diag > 1e-12]
        scale = (
            float(np.median(positive))
            if len(positive)
            else 1.0
        )

        lam = float(smooth_rel) * scale

        lhs = (
            B.T @ B
            + lam * penalty
            + np.eye(n_days) * 1e-8
        )
        rhs = B.T @ y

        try:
            new_daily = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            new_daily = np.linalg.pinv(lhs) @ rhs

        new_daily = np.maximum(
            new_daily,
            0.001,
        )

        sums = A @ new_daily

        new_factors = factors.copy()

        for field in range(1, 15):
            mask = fields == field

            if int(mask.sum()) < 2:
                continue

            s = sums[mask]
            yy = y[mask]

            denom = float(np.sum(s * s))
            if denom <= 1e-12:
                continue

            new_factors[field] = max(
                float(np.sum(s * yy) / denom),
                0.05,
            )

        mean_f = float(np.mean(
            new_factors[1:15]
        ))

        if mean_f <= 1e-12:
            mean_f = 1.0

        new_factors[1:15] /= mean_f
        new_daily *= mean_f

        delta = max(
            float(np.max(np.abs(
                new_daily - daily
            ))),
            float(np.max(np.abs(
                new_factors - factors
            ))),
        )

        daily = new_daily
        factors = new_factors

        if delta < 1e-6:
            break

    pred = np.asarray(
        [factors[int(f)] for f in fields],
        dtype=float,
    ) * (A @ daily)

    return {
        "daily": daily,
        "factors": factors,
        "pred": pred,
    }


def _date_folds(intervals, n_folds):
    dates = sorted(
        intervals["target_date"].unique()
    )

    fold_map = {
        dd: i % n_folds
        for i, dd in enumerate(dates)
    }

    return np.asarray([
        fold_map[dd]
        for dd in intervals["target_date"].tolist()
    ])


def _cv_mae(intervals, A, y, fields, smooth_rel):
    folds = _date_folds(
        intervals,
        CV_FOLDS,
    )

    actual_all = []
    pred_all = []

    for fold in range(CV_FOLDS):
        tr = folds != fold
        te = folds == fold

        if int(tr.sum()) < MIN_EVENTS:
            continue

        if int(te.sum()) == 0:
            continue

        fit = _fit_latent(
            A[tr],
            y[tr],
            fields[tr],
            smooth_rel,
        )

        daily = fit["daily"]
        factors = fit["factors"]

        pred = np.asarray(
            [
                factors[int(f)]
                for f in fields[te]
            ],
            dtype=float,
        ) * (A[te] @ daily)

        actual_all.extend(
            y[te].tolist()
        )
        pred_all.extend(
            pred.tolist()
        )

    if not actual_all:
        return np.nan, np.nan, 0

    a = np.asarray(
        actual_all,
        dtype=float,
    )
    p = np.asarray(
        pred_all,
        dtype=float,
    )

    mae = float(
        np.mean(np.abs(p - a))
    )

    mape = float(
        np.mean(
            np.abs(p - a)
            / np.maximum(np.abs(a), 0.5)
        ) * 100.0
    )

    return mae, mape, len(a)


def _curve(all_days, daily, smooth):
    med = float(np.median(daily))

    if med <= 1e-9:
        med = 1.0

    df = pd.DataFrame({
        "date": all_days,
        "latent_prod": daily,
    })

    df["index_100"] = (
        100.0 * df["latent_prod"] / med
    )

    df["mean_3d"] = (
        df["latent_prod"]
        .rolling(3, min_periods=3)
        .mean()
    )

    df["mean_3d_index"] = (
        100.0
        * df["mean_3d"]
        / med
    )

    df["change_3d_pct"] = (
        100.0
        * (
            df["mean_3d"]
            / df["mean_3d"].shift(3)
            - 1.0
        )
    )

    df["smooth"] = smooth

    return df


def _turning_stats(curve: pd.DataFrame):
    focus = curve[
        (curve["date"] >= FOCUS_START)
        & (curve["date"] <= FOCUS_END)
    ].copy()

    valid_drop = focus[
        focus["change_3d_pct"].notna()
    ].copy()

    valid_mean = focus[
        focus["mean_3d"].notna()
    ].copy()

    if valid_drop.empty or valid_mean.empty:
        return None

    drop_row = (
        valid_drop
        .sort_values(
            ["change_3d_pct", "date"]
        )
        .iloc[0]
    )

    trough_row = (
        valid_mean
        .sort_values(
            ["mean_3d", "date"]
        )
        .iloc[0]
    )

    return {
        "drop_date": drop_row["date"],
        "drop_pct": float(
            drop_row["change_3d_pct"]
        ),
        "trough_date": trough_row["date"],
        "trough_index": float(
            trough_row["mean_3d_index"]
        ),
    }


def _date_spread_days(values):
    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return np.nan

    return int(
        (max(values) - min(values)).days
    )


def main():
    st.set_page_config(
        page_title="KurgiMootor · robust timing",
        layout="wide",
    )

    st.title(
        "Kas taastatud pöörde ajastus on päris?"
    )

    st.caption(
        "Intervallisumma dekonvolutsioon · "
        "siledus 0.03 / 0.1 / 0.3 / 1.0 · "
        "RETROSPEKTIIVNE AUDIT · READ ONLY"
    )

    st.info(
        "Ühtegi siledust ei valita augusti järgi. "
        "Kõik neli eelnevalt fikseeritud varianti peavad näitama, "
        "kas languse ajastus püsib või ujub."
    )

    try:
        harvest = db.get_harvest_history(
            limit=5000
        )

        events = _events(harvest)
        intervals = _build_intervals(
            events
        )

        if len(intervals) < MIN_EVENTS:
            st.error(
                f"Korjeintervalle on ainult "
                f"{len(intervals)}."
            )
            st.stop()

        all_days, day_idx = _day_axis(
            intervals
        )

        A, y, fields = _matrix(
            intervals,
            all_days,
            day_idx,
        )

        summaries = []
        curves = []

        for smooth in SMOOTHS:
            fit = _fit_latent(
                A,
                y,
                fields,
                smooth,
            )

            cv_mae, cv_mape, cv_n = _cv_mae(
                intervals,
                A,
                y,
                fields,
                smooth,
            )

            cur = _curve(
                all_days,
                fit["daily"],
                smooth,
            )

            stat = _turning_stats(
                cur
            )

            if stat is None:
                continue

            factor_values = np.asarray(
                fit["factors"][1:15],
                dtype=float,
            )

            summaries.append({
                "Siledus": smooth,
                "CV N": cv_n,
                "CV MAE": cv_mae,
                "CV MAPE %": cv_mape,
                "Tugevaim langus": stat["drop_date"],
                "3p langus %": stat["drop_pct"],
                "3p põhi": stat["trough_date"],
                "Põhja indeks": stat["trough_index"],
                "Põllufaktor min": float(
                    np.min(factor_values)
                ),
                "Põllufaktor max": float(
                    np.max(factor_values)
                ),
            })

            curves.append(cur)

        summary = pd.DataFrame(
            summaries
        )

        if summary.empty:
            st.error(
                "Robustsuse tabelit ei tekkinud."
            )
            st.stop()

        all_curves = pd.concat(
            curves,
            ignore_index=True,
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    drop_spread = _date_spread_days(
        summary["Tugevaim langus"].tolist()
    )

    trough_spread = _date_spread_days(
        summary["3p põhi"].tolist()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Korjeintervalle",
        len(intervals),
    )

    c2.metric(
        "Languse kuupäeva hajuvus",
        f"{int(drop_spread)} päeva",
    )

    c3.metric(
        "Põhja kuupäeva hajuvus",
        f"{int(trough_spread)} päeva",
    )

    cv_range = (
        float(summary["CV MAE"].max())
        - float(summary["CV MAE"].min())
    )

    c4.metric(
        "CV MAE vahemik",
        f"{summary['CV MAE'].min():.2f}"
        f"–{summary['CV MAE'].max():.2f}",
        delta=f"{cv_range:.2f} kasti",
    )

    if (
        drop_spread <= 2
        and trough_spread <= 2
    ):
        st.success(
            "✅ AJASTUS ON ROBUSTNE: kõik neli siledust "
            "paigutavad nii tugevaima languse kui 3-päevase põhja "
            "maksimaalselt 2 päeva sisse."
        )

    elif (
        drop_spread <= 4
        and trough_spread <= 4
    ):
        st.warning(
            "🟡 AJASTUS ON PLOKITASEMEL ROBUSTNE: päevatäpsus ujub, "
            "kuid kõik neli siledust paigutavad pöörde sama "
            "ligikaudu 3–5 päeva akna sisse."
        )

    else:
        st.error(
            "❌ PÄEVATÄPNE AJATELG EI OLE IDENTIFITSEERITAV: "
            "pöörde kuupäev liigub siledusest sõltuvalt liiga palju."
        )

    st.markdown(
        "### 1. Neli kõverat · üks otsus"
    )

    display_summary = summary.copy()

    st.dataframe(
        display_summary.style.format({
            "Siledus": lambda x: f"{float(x):g}",
            "CV MAE": "{:.2f}",
            "CV MAPE %": "{:.1f}",
            "Tugevaim langus": lambda x: (
                x.strftime("%d.%m")
                if hasattr(x, "strftime")
                else str(x)
            ),
            "3p langus %": "{:+.1f}%",
            "3p põhi": lambda x: (
                x.strftime("%d.%m")
                if hasattr(x, "strftime")
                else str(x)
            ),
            "Põhja indeks": "{:.1f}",
            "Põllufaktor min": "{:.3f}",
            "Põllufaktor max": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Tugevaim langus = kõige negatiivsem muutus "
        "3-päevase keskmise vahel võrreldes eelmise 3-päevase plokiga. "
        "3p põhi = madalaim 3-päevane keskmine 08.–24.08."
    )

    st.markdown(
        "### 2. Kõverad samal teljel"
    )

    focus = all_curves[
        (all_curves["date"] >= FOCUS_START)
        & (all_curves["date"] <= FOCUS_END)
    ].copy()

    wide = focus.pivot(
        index="date",
        columns="smooth",
        values="mean_3d_index",
    )

    wide.columns = [
        f"s={float(c):g}"
        for c in wide.columns
    ]

    st.line_chart(
        wide
    )

    st.caption(
        "Siin on teadlikult 3-päevane keskmine, mitte toores päevakõver. "
        "Eesmärk on vaadata bioloogilise pöörde akent, mitte ühe päeva võnkumist."
    )

    st.markdown(
        "### 3. Päev-päevalt 3-päevane indeks"
    )

    table = wide.reset_index()

    st.dataframe(
        table.style.format({
            "date": lambda x: x.strftime("%d.%m"),
            **{
                c: "{:.1f}"
                for c in table.columns
                if c != "date"
            },
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.caption(
        "SEE EI OLE PROGNOOS: kõik kõverad on full-data retrospektiivsed "
        "dekonvolutsioonid. Held-out CV MAE on lisatud ainult selleks, "
        "et kontrollida, et mõni siledus ei sobitu korjeintervallidele selgelt halvemini."
    )

    st.caption(
        "READ ONLY: ainult db.get_harvest_history. "
        "Ilma, production snapshotte ja DB kirjutamisi ei kasutata."
    )


if __name__ == "__main__":
    main()
