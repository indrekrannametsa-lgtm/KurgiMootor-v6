from __future__ import annotations

"""
KurgiMootor · edge_weather-21
=============================

WEATHER ON THE CORRECT BASE · INTERVAL-INTEGRATED DAILY PRODUCTION · READ ONLY

Core idea
---------
Do NOT attach weather to the harvest date.

For each calendar day d:
    common daily production(d) = exp( daily model(d) )

For each same-field harvest interval i:
    harvest_i
      ≈ field_factor[field_i]
        × sum_d( interval_weight_i,d × daily_production(d) )

This directly encodes the LAB-19 insight that a harvest is accumulated production
since the previous same-field harvest.

STRICT WALK-FORWARD
-------------------
For each target harvest date T:
- fit model using only intervals with target_date < T
- predict the 3 fields harvested on T
- target-day actual is never used in its own fit
- measured weather is used only as a mechanism audit; this is NOT archived
  operational forecast-weather replay yet.

PRE-FIXED MODELS
----------------
M0 · SEASON
    daily production = season curve only

M1 · SOURCE
    M0 + daily radiation + nonlinear night temperature curve

M2 · SOURCE+WD
    M1 + same-day WIND×DRY = wind × (100 - RH)

NO lag search.
NO 17.08 tuning.
NO previous-yield anchor.
NO latent full-data daily target is used.

Why these channels?
- radiation represents assimilate/source supply
- cucumber fruit development/growth is temperature dependent
- water/atmospheric stress can suppress fruit growth; WIND×DRY is retained only
  as the already-discovered project stress proxy

Decision
--------
Weather is supported on the new interval-sum base only if a weather model beats
M0 overall AND in both chronological halves.

READ ONLY
---------
- db.get_harvest_history
- db.get_weather_rows
- no writes
- no production snapshots
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
WEATHER_START = date(2026, 7, 1)

MIN_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2

# Regularization is fixed before seeing OOS results.
FIELD_RIDGE = 1.5
WEATHER_RIDGE = 1.0
SEASON_RIDGE = 0.10

MODEL_SPECS = {
    "M0 · SEASON": ["intercept", "season", "season2"],
    "M1 · +RAD+ööT": [
        "intercept", "season", "season2",
        "rad",
        "night_cool", "night_cool2", "night_warm", "night_heat",
    ],
    "M2 · +RAD+ööT+WD": [
        "intercept", "season", "season2",
        "rad",
        "night_cool", "night_cool2", "night_warm", "night_heat",
        "winddry",
    ],
}


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
            dd,
            field,
            order,
            float(abc),
            _f(r.get("interval_days")),
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


def _build_intervals(events):
    rows = []

    for field in range(1, 15):
        hist = _field_hist(events, field)

        for i in range(1, len(hist)):
            prev = hist[i - 1]
            cur = hist[i]

            gap = int((cur.day - prev.day).days)
            if gap <= 0:
                continue

            growth = _growth(prev, cur)

            days = [
                prev.day + timedelta(days=k)
                for k in range(1, gap + 1)
            ]

            rows.append({
                "target_date": cur.day,
                "start_date": prev.day,
                "field": field,
                "order": cur.order,
                "actual": cur.abc,
                "growth": growth,
                "days": days,
                "per_day_weight": growth / len(days),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["target_date", "order", "field"]
        ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------

def _measured_weather(rows):
    out = {}

    for r in rows:
        dd = _d(r.get("weather_date"))
        if dd is None:
            continue

        if str(r.get("data_kind") or "").strip().lower() != "measured":
            continue

        if not bool(r.get("checked")):
            continue

        night = _f(r.get("temp_night_avg_c"))
        rad = _f(r.get("radiation_mj_m2"))
        wind = _f(r.get("wind_avg_ms"))
        rh = _f(r.get("humidity_avg_pct"))

        if None in (night, rad, wind, rh):
            continue

        out[dd] = {
            "night": float(night),
            "rad": float(rad),
            "winddry": float(wind) * (100.0 - float(rh)),
        }

    return out


def _night_curve(v):
    # Same project thresholds used previously, but now attached to DAILY production.
    cool = max(0.0, 16.0 - v)
    warm = min(max(v - 16.0, 0.0), 4.0)
    heat = max(0.0, v - 20.0)

    return {
        "night_cool": cool / 5.0,
        "night_cool2": (cool * cool) / 25.0,
        "night_warm": warm / 4.0,
        "night_heat": heat / 5.0,
    }


def _daily_features(dd, weather):
    w = weather.get(dd)
    if w is None:
        return None

    # Fixed scales keep coefficients numerically comparable without using
    # test-date outcomes or test-date statistics.
    season = float((dd - WEATHER_START).days) / 30.0

    out = {
        "intercept": 1.0,
        "season": season,
        "season2": season * season,
        "rad": float(w["rad"]) / 20.0,
        "winddry": float(w["winddry"]) / 100.0,
    }

    out.update(
        _night_curve(
            float(w["night"])
        )
    )

    return out


def _keep_weather_complete(intervals, weather):
    mask = []

    for _, row in intervals.iterrows():
        ok = all(
            _daily_features(dd, weather) is not None
            for dd in row["days"]
        )
        mask.append(ok)

    return intervals[
        np.asarray(mask, dtype=bool)
    ].reset_index(drop=True)


# ---------------------------------------------------------------------
# Direct interval-integrated model
# ---------------------------------------------------------------------

def _build_day_feature_cache(intervals, weather, feature_names):
    cache = {}

    all_days = sorted({
        dd
        for days in intervals["days"].tolist()
        for dd in days
    })

    for dd in all_days:
        rec = _daily_features(dd, weather)
        if rec is None:
            raise RuntimeError(
                f"Puuduv mõõdetud ilm {dd}"
            )

        cache[dd] = np.asarray(
            [float(rec[name]) for name in feature_names],
            dtype=float,
        )

    return cache


def _predict_intervals(
    intervals,
    beta,
    gammas,
    feature_names,
    feature_cache,
):
    preds = []

    for _, row in intervals.iterrows():
        daily = np.vstack([
            feature_cache[dd]
            for dd in row["days"]
        ])

        eta = daily @ beta
        prod = np.exp(
            np.clip(eta, -6.0, 6.0)
        )

        total_common = float(
            row["per_day_weight"]
            * np.sum(prod)
        )

        field = int(row["field"])
        field_factor = (
            1.0
            if field == 1
            else math.exp(
                float(gammas[field - 2])
            )
        )

        preds.append(
            field_factor * total_common
        )

    return np.asarray(
        preds,
        dtype=float,
    )


def _fit_model(
    train,
    weather,
    feature_names,
):
    """
    Pure NumPy optimizer.

    Objective:
      log(predicted interval ABC + 0.20)
        - log(actual interval ABC + 0.20)

    plus fixed ridge penalties.

    No scipy dependency. No target-day actual enters its own fit because
    the surrounding walk-forward calls this only with target_date < target.
    """
    feature_cache = _build_day_feature_cache(
        train,
        weather,
        feature_names,
    )

    n_beta = len(feature_names)
    n_gamma = 13

    y = train["actual"].to_numpy(
        dtype=float
    )
    growth = train["growth"].to_numpy(
        dtype=float
    )

    mean_daily = float(
        np.mean(
            y / np.maximum(growth, 0.5)
        )
    )

    beta = np.zeros(
        n_beta,
        dtype=float,
    )
    beta[0] = math.log(
        max(mean_daily, 0.05)
    )

    gammas = np.zeros(
        n_gamma,
        dtype=float,
    )

    # Pre-build day-feature matrices for each harvest interval.
    interval_X = []
    interval_w = []
    interval_field = []

    for _, row in train.iterrows():
        X = np.vstack([
            feature_cache[dd]
            for dd in row["days"]
        ])

        interval_X.append(X)
        interval_w.append(
            float(row["per_day_weight"])
        )
        interval_field.append(
            int(row["field"])
        )

    # Fixed optimizer settings; not tuned to August or OOS results.
    learning_rate = 0.035
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    max_iter = 350

    m_beta = np.zeros_like(beta)
    v_beta = np.zeros_like(beta)
    m_gamma = np.zeros_like(gammas)
    v_gamma = np.zeros_like(gammas)

    previous_obj = None

    for step in range(1, max_iter + 1):
        grad_beta = np.zeros_like(beta)
        grad_gamma = np.zeros_like(gammas)
        data_obj = 0.0

        for i, (X, w, field) in enumerate(
            zip(
                interval_X,
                interval_w,
                interval_field,
            )
        ):
            eta = X @ beta
            daily_prod = np.exp(
                np.clip(eta, -6.0, 6.0)
            )

            common_sum = float(
                w * np.sum(daily_prod)
            )

            if field == 1:
                field_factor = 1.0
                gamma_idx = None
            else:
                gamma_idx = field - 2
                field_factor = math.exp(
                    float(gammas[gamma_idx])
                )

            pred = max(
                field_factor * common_sum,
                1e-8,
            )

            resid = (
                math.log(pred + 0.20)
                - math.log(float(y[i]) + 0.20)
            )
            data_obj += resid * resid

            # d log(pred+0.20) / d log(pred)
            shrink = pred / (pred + 0.20)

            denom = max(
                float(np.sum(daily_prod)),
                1e-12,
            )

            # Derivative of log(sum(exp(X beta))) is the
            # production-weighted mean feature vector.
            x_bar = (
                (
                    daily_prod[:, None]
                    * X
                ).sum(axis=0)
                / denom
            )

            grad_beta += (
                2.0
                * resid
                * shrink
                * x_bar
            )

            if gamma_idx is not None:
                grad_gamma[gamma_idx] += (
                    2.0
                    * resid
                    * shrink
                )

        # Fixed ridge penalties.
        reg_obj = 0.0

        for j, name in enumerate(
            feature_names
        ):
            if name == "intercept":
                continue

            lam = (
                SEASON_RIDGE
                if name in {"season", "season2"}
                else WEATHER_RIDGE
            )

            reg_obj += (
                lam
                * beta[j]
                * beta[j]
            )
            grad_beta[j] += (
                2.0
                * lam
                * beta[j]
            )

        reg_obj += (
            FIELD_RIDGE
            * float(
                np.sum(
                    gammas * gammas
                )
            )
        )
        grad_gamma += (
            2.0
            * FIELD_RIDGE
            * gammas
        )

        obj = data_obj + reg_obj

        # Keep optimizer step scale broadly independent of train size.
        n_scale = max(
            len(train),
            1,
        )
        grad_beta /= n_scale
        grad_gamma /= n_scale

        # Numerical safety clipping.
        beta_norm = float(
            np.linalg.norm(
                grad_beta
            )
        )
        gamma_norm = float(
            np.linalg.norm(
                grad_gamma
            )
        )

        if beta_norm > 10.0:
            grad_beta *= (
                10.0 / beta_norm
            )

        if gamma_norm > 10.0:
            grad_gamma *= (
                10.0 / gamma_norm
            )

        # Adam update.
        m_beta = (
            adam_beta1 * m_beta
            + (1.0 - adam_beta1)
            * grad_beta
        )
        v_beta = (
            adam_beta2 * v_beta
            + (1.0 - adam_beta2)
            * (grad_beta * grad_beta)
        )

        m_gamma = (
            adam_beta1 * m_gamma
            + (1.0 - adam_beta1)
            * grad_gamma
        )
        v_gamma = (
            adam_beta2 * v_gamma
            + (1.0 - adam_beta2)
            * (grad_gamma * grad_gamma)
        )

        m_beta_hat = (
            m_beta
            / (
                1.0
                - adam_beta1 ** step
            )
        )
        v_beta_hat = (
            v_beta
            / (
                1.0
                - adam_beta2 ** step
            )
        )

        m_gamma_hat = (
            m_gamma
            / (
                1.0
                - adam_beta1 ** step
            )
        )
        v_gamma_hat = (
            v_gamma
            / (
                1.0
                - adam_beta2 ** step
            )
        )

        beta -= (
            learning_rate
            * m_beta_hat
            / (
                np.sqrt(
                    v_beta_hat
                )
                + adam_eps
            )
        )

        gammas -= (
            learning_rate
            * m_gamma_hat
            / (
                np.sqrt(
                    v_gamma_hat
                )
                + adam_eps
            )
        )

        # Numerical safety only; ±50% is far wider than the field factors
        # seen in LAB-19 and is not selected from target outcomes.
        gammas = np.clip(
            gammas,
            math.log(0.5),
            math.log(1.5),
        )

        if (
            previous_obj is not None
            and step > 80
            and abs(
                previous_obj - obj
            )
            < (
                1e-7
                * max(
                    1.0,
                    abs(previous_obj),
                )
            )
        ):
            break

        previous_obj = obj

    return {
        "beta": beta,
        "gammas": gammas,
        "feature_names": list(
            feature_names
        ),
        "success": True,
        "cost": float(
            previous_obj
            if previous_obj is not None
            else np.nan
        ),
        "iterations": int(step),
    }


def _predict_with_fit(
    fit,
    intervals,
    weather,
):
    cache = _build_day_feature_cache(
        intervals,
        weather,
        fit["feature_names"],
    )

    return _predict_intervals(
        intervals,
        fit["beta"],
        fit["gammas"],
        fit["feature_names"],
        cache,
    )


# ---------------------------------------------------------------------
# Strict walk-forward
# ---------------------------------------------------------------------

def _complete_day_map(intervals):
    out = {}

    for dd, g in intervals.groupby(
        "target_date",
        sort=True,
    ):
        if (
            len(g) == 3
            and g["field"].nunique() == 3
        ):
            out[dd] = g.index.tolist()

    return out


def _walk_forward(
    intervals,
    weather,
):
    complete = _complete_day_map(
        intervals
    )

    rows = []

    for target in sorted(complete):
        train = intervals[
            intervals["target_date"] < target
        ].copy()

        test = intervals.loc[
            complete[target]
        ].copy()

        if len(train) < MIN_TRAIN_INTERVALS:
            continue

        # Require some history for every test field; no unlearned field extrapolation.
        counts = train.groupby(
            "field"
        ).size().to_dict()

        if any(
            counts.get(int(f), 0) < MIN_FIELD_OBS
            for f in test["field"].tolist()
        ):
            continue

        rec = {
            "date": target,
            "fields": ",".join(
                str(int(x))
                for x in test.sort_values(
                    "order"
                )["field"].tolist()
            ),
            "actual": float(
                test["actual"].sum()
            ),
            "train_n": len(train),
        }

        for model_name, feature_names in MODEL_SPECS.items():
            fit = _fit_model(
                train,
                weather,
                feature_names,
            )

            pred = _predict_with_fit(
                fit,
                test,
                weather,
            )

            rec[model_name] = float(
                np.sum(pred)
            )

        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _metrics(df, model_col, base_col="M0 · SEASON"):
    use = df[
        df[model_col].notna()
        & df[base_col].notna()
        & df["actual"].notna()
    ].copy()

    if use.empty:
        return {
            "n": 0,
            "base_mae": np.nan,
            "model_mae": np.nan,
            "improvement": np.nan,
            "wins": 0,
            "base_mape": np.nan,
            "model_mape": np.nan,
        }

    a = use["actual"].to_numpy(dtype=float)
    b = use[base_col].to_numpy(dtype=float)
    p = use[model_col].to_numpy(dtype=float)

    eb = np.abs(b - a)
    ep = np.abs(p - a)

    bmae = float(np.mean(eb))
    pmae = float(np.mean(ep))

    return {
        "n": len(use),
        "base_mae": bmae,
        "model_mae": pmae,
        "improvement": (
            100.0 * (bmae - pmae) / bmae
            if bmae > 1e-9 else np.nan
        ),
        "wins": int(np.sum(ep < eb)),
        "base_mape": float(
            np.mean(
                eb / np.maximum(np.abs(a), 0.5)
            ) * 100.0
        ),
        "model_mape": float(
            np.mean(
                ep / np.maximum(np.abs(a), 0.5)
            ) * 100.0
        ),
    }


def _halves(df):
    days = sorted(
        df["date"].tolist()
    )

    cut = len(days) // 2

    return (
        df[df["date"].isin(
            set(days[:cut])
        )].copy(),
        df[df["date"].isin(
            set(days[cut:])
        )].copy(),
    )


def _metric_table(df):
    rows = []

    for model_name in [
        "M1 · +RAD+ööT",
        "M2 · +RAD+ööT+WD",
    ]:
        m = _metrics(
            df,
            model_name,
        )

        rows.append({
            "Mudel": model_name,
            "N päeva": m["n"],
            "SEASON MAE": m["base_mae"],
            "Mudeli MAE": m["model_mae"],
            "Paranemine %": m["improvement"],
            "Võite": m["wins"],
            "SEASON MAPE %": m["base_mape"],
            "Mudeli MAPE %": m["model_mape"],
        })

    return pd.DataFrame(rows)


def main():
    st.set_page_config(
        page_title="KurgiMootor · weather on interval base",
        layout="wide",
    )

    st.title(
        "Ilm õigel baasil · päevane produktsioon → korjeintervalli summa"
    )

    st.caption(
        "Strict walk-forward · measured-weather mechanism audit · "
        "ei lag search'i · READ ONLY"
    )

    st.info(
        "See test ei küsi enam, milline ilm oli enne KORJEPÄEVA. "
        "Ilm mõjutab iga kasvupäeva produktsiooni ning konkreetne korje on nende päevade summa. "
        "Kõik mudelivariandid olid enne tulemuse nägemist fikseeritud."
    )

    try:
        harvest = db.get_harvest_history(
            limit=5000
        )

        events = _events(harvest)
        intervals = _build_intervals(
            events
        )

        if intervals.empty:
            st.error(
                "Korjeintervalle ei tekkinud."
            )
            st.stop()

        weather_end = max(
            dd
            for days in intervals["days"]
            for dd in days
        )

        weather_rows = db.get_weather_rows(
            WEATHER_START,
            weather_end,
        )

        weather = _measured_weather(
            weather_rows
        )

        intervals = _keep_weather_complete(
            intervals,
            weather,
        )

        oos = _walk_forward(
            intervals,
            weather,
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()

    if oos.empty:
        st.error(
            "Strict walk-forward OOS päevi ei tekkinud."
        )
        st.stop()

    full_metrics = _metric_table(
        oos
    )

    first, second = _halves(
        oos
    )

    first_metrics = _metric_table(
        first
    )

    second_metrics = _metric_table(
        second
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "OOS päevi",
        len(oos),
    )

    c2.metric(
        "Korjeintervalle ilmaga",
        len(intervals),
    )

    c3.metric(
        "Esimene OOS päev",
        min(oos["date"]).strftime("%d.%m"),
    )

    c4.metric(
        "Viimane OOS päev",
        max(oos["date"]).strftime("%d.%m"),
    )

    st.markdown(
        "### 1. Kas ilm lisab intervallisumma baasile infot?"
    )

    st.dataframe(
        full_metrics.style.format({
            "SEASON MAE": "{:.2f}",
            "Mudeli MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "SEASON MAPE %": "{:.1f}",
            "Mudeli MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Conservative decision on M2, because it contains the full fixed weather set.
    m2 = _metrics(
        oos,
        "M2 · +RAD+ööT+WD",
    )

    m2a = _metrics(
        first,
        "M2 · +RAD+ööT+WD",
    )

    m2b = _metrics(
        second,
        "M2 · +RAD+ööT+WD",
    )

    stable = (
        np.isfinite(m2["improvement"])
        and np.isfinite(m2a["improvement"])
        and np.isfinite(m2b["improvement"])
        and m2["improvement"] > 0.0
        and m2a["improvement"] > 0.0
        and m2b["improvement"] > 0.0
    )

    if stable:
        st.success(
            f"✅ ILM ANNAB UUEL BAASIL STABIILSE EELISE: "
            f"M2 MAE {m2['base_mae']:.2f} → {m2['model_mae']:.2f} "
            f"({m2['improvement']:+.1f}%) ja eelis on mõlemas ajapooles."
        )
    elif (
        np.isfinite(m2["improvement"])
        and m2["improvement"] > 0.0
    ):
        st.warning(
            f"🟡 Ilm annab üldiselt eelise ({m2['improvement']:+.1f}%), "
            "kuid see ei püsi mõlemas ajapooles. Veel ei saa uut weather-base'i lukustada."
        )
    else:
        st.error(
            f"❌ ILM EI PARANDA UUT INTERVALLISUMMA BAASI: "
            f"M2 MAE {m2['base_mae']:.2f} → {m2['model_mae']:.2f}."
        )

    st.markdown(
        "### 2. Kõige tähtsam kontroll · kaks ajapoolt"
    )

    half_rows = []

    for label, mt in [
        ("I pool", first_metrics),
        ("II pool", second_metrics),
    ]:
        for _, r in mt.iterrows():
            half_rows.append({
                "Periood": label,
                **r.to_dict(),
            })

    halves_df = pd.DataFrame(
        half_rows
    )

    st.dataframe(
        halves_df.style.format({
            "SEASON MAE": "{:.2f}",
            "Mudeli MAE": "{:.2f}",
            "Paranemine %": lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x):+.1f}%"
            ),
            "SEASON MAPE %": "{:.1f}",
            "Mudeli MAPE %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 3. Päev-päevalt"
    )

    show = oos.copy()

    for model_name in MODEL_SPECS:
        show[
            model_name + " viga"
        ] = (
            show[model_name]
            - show["actual"]
        )

    cols = [
        "date",
        "fields",
        "actual",
        "M0 · SEASON",
        "M1 · +RAD+ööT",
        "M2 · +RAD+ööT+WD",
        "M0 · SEASON viga",
        "M1 · +RAD+ööT viga",
        "M2 · +RAD+ööT+WD viga",
        "train_n",
    ]

    st.dataframe(
        show[cols].style.format({
            "date": lambda x: x.strftime("%d.%m"),
            "actual": "{:.1f}",
            "M0 · SEASON": "{:.1f}",
            "M1 · +RAD+ööT": "{:.1f}",
            "M2 · +RAD+ööT+WD": "{:.1f}",
            "M0 · SEASON viga": "{:+.1f}",
            "M1 · +RAD+ööT viga": "{:+.1f}",
            "M2 · +RAD+ööT+WD viga": "{:+.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.caption(
        "MEHHANISMI AUDIT: mõõdetud ilm annab meile küsimusele „kas ilm kannab infot õigel "
        "intervallisumma baasil?“ puhtama vastuse. See ei ole veel +1…+9 operatiivprognoosi replay."
    )

    st.caption(
        "AUDIT LOCK: target-päeva mudel treenib ainult intervalle target_date < target. "
        "Eelmise sama põllu saaki ei kasutata mudeli sisendina."
    )

    st.caption(
        "READ ONLY: db.get_harvest_history + db.get_weather_rows. DB kirjutamisi ei ole. Optimeerimine on pure NumPy; SciPy pole vajalik."
    )


if __name__ == "__main__":
    main()
