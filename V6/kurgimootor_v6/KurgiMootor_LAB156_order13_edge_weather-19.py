from __future__ import annotations

"""
KurgiMootor · edge_weather-19
=============================

DAILY PRODUCTION DECONVOLUTION AUDIT · READ ONLY

Purpose
-------
A harvest observed on date T is NOT treated as production "of date T".
Instead each same-field harvest is an interval observation:

    harvest(field, T)
        ≈ field_factor[field]
          × sum(daily_common_production[d] × interval_weight[d])

The interval runs from the previous harvest of that same field to the current
harvest. Because the 14 fields are harvested on different days, these intervals
overlap. The overlap can be used to reconstruct a latent daily common production
curve.

This is RETROSPECTIVE reconstruction, not a forecast.
No weather is used.

Key safeguards
--------------
- Field multipliers and daily curve are estimated jointly by alternating least squares.
- Daily curve smoothness is NOT hand-picked to make August look good.
  A small pre-fixed smoothness grid is selected by held-out TARGET-DATE folds.
- Held-out dates are excluded as observation equations when scoring CV.
- Final curve is then fitted on all intervals with the CV-selected smoothness.

Interval timing
---------------
We only know harvest date and within-day harvest order, not exact clock time.
For each interval, calendar days (previous harvest + 1 ... current harvest)
share the exact order-adjusted growth duration evenly.
Thus:
    sum(interval weights) = exact growth duration
This avoids inventing an absolute harvest clock.

READ ONLY
---------
- db.get_harvest_history only
- no weather
- no production snapshots
- no DB writes
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db


HOURS_PER_FIELD = 3.0
MIN_EVENTS = 35
ALS_ITERS = 20

# Dimensionless multiplier of a data-scale reference.
SMOOTH_GRID = [0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0]
CV_FOLDS = 4


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

    return sorted(out, key=lambda e: (e.day, e.order, e.field))


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

            calendar_gap = (cur.day - prev.day).days
            if calendar_gap <= 0:
                continue

            growth = _growth(prev, cur)

            days = [
                prev.day + timedelta(days=k)
                for k in range(1, calendar_gap + 1)
            ]

            # Each calendar day shares the exact order-adjusted duration equally.
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

    return pd.DataFrame(rows).sort_values(
        ["target_date", "order", "field"]
    ).reset_index(drop=True)


def _day_axis(intervals: pd.DataFrame):
    all_days = sorted({
        dd
        for days in intervals["days"].tolist()
        for dd in days
    })
    return all_days, {dd: i for i, dd in enumerate(all_days)}


def _interval_matrix(intervals: pd.DataFrame, all_days, day_idx):
    A = np.zeros((len(intervals), len(all_days)), dtype=float)

    for i, row in intervals.iterrows():
        w = float(row["per_day_weight"])
        for dd in row["days"]:
            A[i, day_idx[dd]] += w

    y = intervals["actual"].to_numpy(dtype=float)
    fields = intervals["field"].to_numpy(dtype=int)
    return A, y, fields


def _second_diff_penalty(n_days: int):
    if n_days < 3:
        return np.zeros((n_days, n_days), dtype=float)

    D = np.zeros((n_days - 2, n_days), dtype=float)
    for i in range(n_days - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    return D.T @ D


def _fit_latent(
    A: np.ndarray,
    y: np.ndarray,
    fields: np.ndarray,
    smooth_rel: float,
    init_field_factors: Optional[np.ndarray] = None,
):
    n_events, n_days = A.shape

    if init_field_factors is None:
        factors = np.ones(15, dtype=float)
    else:
        factors = np.asarray(init_field_factors, dtype=float).copy()

    factors[0] = 1.0

    penalty = _second_diff_penalty(n_days)

    daily = np.full(
        n_days,
        max(float(np.mean(y / np.maximum(A.sum(axis=1), 0.5))), 0.05),
        dtype=float,
    )

    for _ in range(ALS_ITERS):
        # Solve daily common production with current field multipliers.
        row_f = np.asarray([factors[int(f)] for f in fields], dtype=float)
        B = A * row_f[:, None]

        diag = np.diag(B.T @ B)
        positive_diag = diag[diag > 1e-12]
        scale = float(np.median(positive_diag)) if len(positive_diag) else 1.0

        lam = float(smooth_rel) * scale

        lhs = B.T @ B + lam * penalty + np.eye(n_days) * 1e-8
        rhs = B.T @ y

        try:
            new_daily = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            new_daily = np.linalg.pinv(lhs) @ rhs

        # Biological production cannot be negative.
        new_daily = np.maximum(new_daily, 0.001)

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

            f = float(np.sum(s * yy) / denom)
            new_factors[field] = max(f, 0.05)

        # Identify scale: mean field factor = 1, preserving fitted values.
        valid_f = new_factors[1:15]
        mean_f = float(np.mean(valid_f))
        if mean_f <= 1e-12:
            mean_f = 1.0

        new_factors[1:15] = valid_f / mean_f
        new_daily = new_daily * mean_f

        change_daily = float(np.max(np.abs(new_daily - daily)))
        change_f = float(np.max(np.abs(new_factors - factors)))

        daily = new_daily
        factors = new_factors

        if max(change_daily, change_f) < 1e-6:
            break

    pred = np.asarray([
        factors[int(f)] for f in fields
    ]) * (A @ daily)

    return {
        "daily": daily,
        "factors": factors,
        "pred": pred,
    }


def _date_folds(intervals: pd.DataFrame, n_folds: int):
    dates = sorted(intervals["target_date"].unique())
    fold_map = {
        dd: i % n_folds
        for i, dd in enumerate(dates)
    }
    return np.asarray([
        fold_map[dd]
        for dd in intervals["target_date"].tolist()
    ], dtype=int)


def _cv_select(
    intervals: pd.DataFrame,
    A: np.ndarray,
    y: np.ndarray,
    fields: np.ndarray,
):
    folds = _date_folds(intervals, CV_FOLDS)
    rows = []

    for smooth_rel in SMOOTH_GRID:
        held_actual = []
        held_pred = []

        for fold in range(CV_FOLDS):
            train_mask = folds != fold
            test_mask = folds == fold

            if int(train_mask.sum()) < MIN_EVENTS or int(test_mask.sum()) == 0:
                continue

            fit = _fit_latent(
                A[train_mask],
                y[train_mask],
                fields[train_mask],
                smooth_rel,
            )

            daily = fit["daily"]
            factors = fit["factors"]

            pred_test = np.asarray([
                factors[int(f)]
                for f in fields[test_mask]
            ]) * (A[test_mask] @ daily)

            held_actual.extend(y[test_mask].tolist())
            held_pred.extend(pred_test.tolist())

        if not held_actual:
            continue

        aa = np.asarray(held_actual, dtype=float)
        pp = np.asarray(held_pred, dtype=float)

        mae = float(np.mean(np.abs(pp - aa)))
        mape = float(np.mean(
            np.abs(pp - aa) / np.maximum(np.abs(aa), 0.5)
        ) * 100.0)

        rows.append({
            "Smooth rel": float(smooth_rel),
            "CV N": len(aa),
            "CV MAE": mae,
            "CV MAPE %": mape,
        })

    cv = pd.DataFrame(rows).sort_values(
        ["CV MAE", "Smooth rel"]
    ).reset_index(drop=True)

    if cv.empty:
        raise RuntimeError("Smoothness CV ei tekkinud.")

    best = float(cv.iloc[0]["Smooth rel"])
    return best, cv


# ---------------------------------------------------------------------
# Weatherless event baseline for CV context
# ---------------------------------------------------------------------

def _fit_event_baseline(train: pd.DataFrame):
    """
    Simple event-level comparator:
      log(actual+eps) ~ log(growth) + season day + season day^2 + field + order
    """
    dd = pd.to_datetime(train["target_date"])
    season0 = dd.min()
    season_day = (dd - season0).dt.days.to_numpy(dtype=float)

    xnum = np.column_stack([
        np.log(np.maximum(train["growth"].to_numpy(dtype=float), 0.5)),
        season_day,
        season_day ** 2,
    ])

    ylog = np.log(np.maximum(train["actual"].to_numpy(dtype=float), 0.0) + 0.20)
    fields = train["field"].to_numpy(dtype=int)
    orders = train["order"].to_numpy(dtype=int)

    mu = np.mean(xnum, axis=0)
    sd = np.std(xnum, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (xnum - mu) / sd

    fd = np.column_stack([
        (fields == f).astype(float)
        for f in range(2, 15)
    ])

    od = np.column_stack([
        (orders == o).astype(float)
        for o in (2, 3)
    ])

    X = np.column_stack([
        np.ones(len(z)),
        z,
        fd,
        od,
    ])

    reg = np.eye(X.shape[1]) * 10.0
    reg[0, 0] = 0.0

    try:
        beta = np.linalg.solve(X.T @ X + reg, X.T @ ylog)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ X + reg) @ (X.T @ ylog)

    return {
        "season0": season0,
        "mu": mu,
        "sd": sd,
        "beta": beta,
    }


def _predict_event_baseline(model, test: pd.DataFrame):
    dd = pd.to_datetime(test["target_date"])
    season_day = (dd - model["season0"]).dt.days.to_numpy(dtype=float)

    xnum = np.column_stack([
        np.log(np.maximum(test["growth"].to_numpy(dtype=float), 0.5)),
        season_day,
        season_day ** 2,
    ])

    z = (xnum - model["mu"]) / model["sd"]

    fields = test["field"].to_numpy(dtype=int)
    orders = test["order"].to_numpy(dtype=int)

    fd = np.column_stack([
        (fields == f).astype(float)
        for f in range(2, 15)
    ])

    od = np.column_stack([
        (orders == o).astype(float)
        for o in (2, 3)
    ])

    X = np.column_stack([
        np.ones(len(z)),
        z,
        fd,
        od,
    ])

    pred_log = X @ model["beta"]

    return np.maximum(
        0.0,
        np.exp(np.clip(pred_log, -6.0, 8.0)) - 0.20,
    )


def _baseline_cv(intervals: pd.DataFrame):
    folds = _date_folds(intervals, CV_FOLDS)

    aa = []
    pp = []

    for fold in range(CV_FOLDS):
        train = intervals[folds != fold].copy()
        test = intervals[folds == fold].copy()

        if len(train) < MIN_EVENTS or test.empty:
            continue

        model = _fit_event_baseline(train)
        pred = _predict_event_baseline(model, test)

        aa.extend(test["actual"].to_numpy(dtype=float).tolist())
        pp.extend(pred.tolist())

    if not aa:
        return np.nan, np.nan, 0

    a = np.asarray(aa, dtype=float)
    p = np.asarray(pp, dtype=float)

    mae = float(np.mean(np.abs(p - a)))
    mape = float(np.mean(
        np.abs(p - a) / np.maximum(np.abs(a), 0.5)
    ) * 100.0)

    return mae, mape, len(a)


# ---------------------------------------------------------------------
# Final reconstruction diagnostics
# ---------------------------------------------------------------------

def _curve_table(all_days, daily):
    df = pd.DataFrame({
        "date": all_days,
        "latent_prod": daily,
    })

    med = float(np.median(daily))
    if med <= 1e-9:
        med = 1.0

    df["index_100"] = 100.0 * df["latent_prod"] / med
    df["change_1d_pct"] = 100.0 * (
        df["latent_prod"] / df["latent_prod"].shift(1) - 1.0
    )
    df["mean_3d"] = df["latent_prod"].rolling(3, min_periods=1).mean()
    df["change_3d_pct"] = 100.0 * (
        df["mean_3d"] / df["mean_3d"].shift(3) - 1.0
    )

    return df


def _largest_drop(curve: pd.DataFrame):
    focus = curve[
        (curve["date"] >= date(2026, 8, 8))
        & (curve["date"] <= date(2026, 8, 24))
    ].copy()

    focus = focus[focus["change_3d_pct"].notna()]

    if focus.empty:
        return None

    row = focus.sort_values("change_3d_pct").iloc[0]

    return {
        "date": row["date"],
        "drop_pct": float(row["change_3d_pct"]),
        "latent": float(row["latent_prod"]),
    }


def _interval_fit_table(
    intervals: pd.DataFrame,
    pred: np.ndarray,
):
    out = intervals[[
        "target_date",
        "field",
        "order",
        "start_date",
        "growth",
        "actual",
    ]].copy()

    out["reconstructed"] = pred
    out["error"] = out["reconstructed"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out


def main():
    st.set_page_config(
        page_title="KurgiMootor · daily deconvolution",
        layout="wide",
    )

    st.title("Korje ei ole päev · latentne päevane produktsioon")
    st.caption(
        "14 põllu kattuvad korjeintervallid → ühine päevane produktsioonikõver · "
        "RETROSPEKTIIVNE AUDIT · READ ONLY"
    )

    st.info(
        "See LAB ei ennusta veel homset. Ta kontrollib kõigepealt meie uut põhihüpoteesi: "
        "kas korjeid saab seletada kui eelmise sama põllu korje järel kogunenud päevase "
        "produktsiooni summat. Ilma ei kasutata."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)

        if len(intervals) < MIN_EVENTS:
            st.error(
                f"Korjeintervalle on ainult {len(intervals)}; vaja vähemalt {MIN_EVENTS}."
            )
            st.stop()

        all_days, day_idx = _day_axis(intervals)
        A, y, fields = _interval_matrix(
            intervals,
            all_days,
            day_idx,
        )

        best_smooth, cv = _cv_select(
            intervals,
            A,
            y,
            fields,
        )

        baseline_mae, baseline_mape, baseline_n = _baseline_cv(intervals)

        fit = _fit_latent(
            A,
            y,
            fields,
            best_smooth,
        )

        curve = _curve_table(
            all_days,
            fit["daily"],
        )

        fit_table = _interval_fit_table(
            intervals,
            fit["pred"],
        )

        drop = _largest_drop(curve)

    except Exception as exc:
        st.exception(exc)
        st.stop()

    best_cv_mae = float(cv.iloc[0]["CV MAE"])
    best_cv_mape = float(cv.iloc[0]["CV MAPE %"])

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Korjeintervalle",
        len(intervals),
    )
    c2.metric(
        "Latentseid päevi",
        len(all_days),
    )
    c3.metric(
        "Valitud siledus",
        f"{best_smooth:g}",
    )
    c4.metric(
        "Baseline CV MAE",
        "—" if not np.isfinite(baseline_mae) else f"{baseline_mae:.2f}",
    )
    c5.metric(
        "Latent CV MAE",
        f"{best_cv_mae:.2f}",
        delta=(
            f"{baseline_mae - best_cv_mae:+.2f} kasti parem"
            if np.isfinite(baseline_mae) else None
        ),
    )

    if np.isfinite(baseline_mae) and baseline_mae > 1e-9:
        imp = 100.0 * (baseline_mae - best_cv_mae) / baseline_mae
    else:
        imp = np.nan

    if np.isfinite(imp) and imp >= 10.0:
        st.success(
            f"✅ INTERVALLISUMMA HÜPOTEES ON TOETATUD: held-out korjeintervallide CV MAE "
            f"paraneb {imp:.1f}% ({baseline_mae:.2f} → {best_cv_mae:.2f})."
        )
    elif np.isfinite(imp) and imp > 0.0:
        st.warning(
            f"🟡 Intervallisumma annab väikese eelise: CV MAE "
            f"{baseline_mae:.2f} → {best_cv_mae:.2f} ({imp:+.1f}%)."
        )
    else:
        st.error(
            f"❌ Päevase latentse produktsiooni rekonstruktsioon ei löö lihtsat weatherless baseline'i "
            f"held-out korjeintervallidel: {baseline_mae:.2f} → {best_cv_mae:.2f}."
        )

    if drop is not None:
        st.markdown("### Kõigepealt see")
        st.metric(
            "Suurim 3-päevase produktsiooni langus 08.–24.08",
            f"{drop['date'].strftime('%d.%m')}",
            delta=f"{drop['drop_pct']:+.1f}%",
        )
        st.caption(
            "See on latentse päevase produktsioonikõvera pöördekoht, mitte korjepäeva residual."
        )

    # -------------------------------------------------------------
    # Main curve
    # -------------------------------------------------------------

    st.markdown("### 1. Taastatud päevane produktsioon")

    chart = curve.set_index("date")[["index_100"]].copy()
    st.line_chart(chart)

    focus = curve[
        (curve["date"] >= date(2026, 8, 8))
        & (curve["date"] <= date(2026, 8, 26))
    ].copy()

    st.dataframe(
        focus[[
            "date",
            "latent_prod",
            "index_100",
            "change_1d_pct",
            "change_3d_pct",
        ]].style.format({
            "date": lambda x: x.strftime("%d.%m"),
            "latent_prod": "{:.3f}",
            "index_100": "{:.1f}",
            "change_1d_pct": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%",
            "change_3d_pct": lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Index 100 = kogu taastatud perioodi mediaanpäev. latent_prod on normaliseeritud "
        "keskmise põllu ABC-kastid ühe kasvupäeva kohta."
    )

    # -------------------------------------------------------------
    # CV smoothness
    # -------------------------------------------------------------

    st.markdown("### 2. Kas pöördekoht sõltub käsitsi valitud siledusest?")

    st.dataframe(
        cv.style.format({
            "Smooth rel": "{:g}",
            "CV MAE": "{:.2f}",
            "CV MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Peamine kõver kasutab ainult selle tabeli väikseima held-out CV MAE-ga siledust. "
        "Augusti pöörde järgi midagi ei valitud."
    )

    # -------------------------------------------------------------
    # Field multipliers
    # -------------------------------------------------------------

    st.markdown("### 3. Põldude püsitasemed")

    factors = pd.DataFrame({
        "Põld": list(range(1, 15)),
        "Püsitase": [float(fit["factors"][f]) for f in range(1, 15)],
    })

    st.dataframe(
        factors.style.format({
            "Püsitase": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Keskmine põllutase on identifitseerimiseks 1.00. See ei ole hetkeseis, vaid kogu perioodi "
        "püsiv põlluefekt selles esimeses dekonvolutsiooni auditis."
    )

    # -------------------------------------------------------------
    # Interval reconstruction
    # -------------------------------------------------------------

    st.markdown("### 4. 14.–24.08 korjeintervallide järelkontroll")

    ffocus = fit_table[
        (fit_table["target_date"] >= date(2026, 8, 14))
        & (fit_table["target_date"] <= date(2026, 8, 24))
    ].copy()

    st.dataframe(
        ffocus[[
            "target_date",
            "field",
            "start_date",
            "growth",
            "actual",
            "reconstructed",
            "error",
        ]].style.format({
            "target_date": lambda x: x.strftime("%d.%m"),
            "start_date": lambda x: x.strftime("%d.%m"),
            "growth": "{:.2f}",
            "actual": "{:.2f}",
            "reconstructed": "{:.2f}",
            "error": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "See tabel on final full-data rekonstruktsiooni järelkontroll, mitte OOS skoor. "
        "OOS kontroll on ülal CV MAE."
    )

    st.divider()

    st.caption(
        "INTERVALLI REEGEL: sama põllu eelmise korje järgse päeva kuni sihtkorje päevani "
        "jaotatakse exact order-adjusted growth duration ühtlaselt kalendripäevadele."
    )

    st.caption(
        "READ ONLY: ainult db.get_harvest_history. Ilma, production snapshotte ja DB kirjutamisi ei kasutata."
    )


if __name__ == "__main__":
    main()
