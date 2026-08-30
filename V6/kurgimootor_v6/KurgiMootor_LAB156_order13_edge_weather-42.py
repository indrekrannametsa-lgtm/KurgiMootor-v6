from __future__ import annotations

"""
KurgiMootor · edge_weather-43b
=============================

FIRST-DERIVATIVE WEATHER AUDIT · STRICT WALK-FORWARD · READ ONLY

One question only
-----------------
Have we been asking weather to predict the wrong target?

Earlier LABs tried to predict the LEVEL of the common BASE residual.  The clean
healthy-August wave suggests a different mechanism: pre-target weather may tell us
whether the common production state is ACCELERATING or DECELERATING, i.e. the
first difference of the common residual rather than its absolute level.

Locked hypothesis
-----------------
Primary signal (fixed from LAB-37, no new window search):
    Δ residual_t  <-  Δ radiation 4d
where
    Δ residual_t = common_residual_t - common_residual_previous_harvest_day
and Δ radiation 4d is exactly LAB-37's pre-target 4d-vs-4d transition:
    mean(T-4..T-1) - mean(T-8..T-5).

Negative controls, using the exact same 4d-vs-4d construction:
    Δ night-stress 4d
    Δ WIND×DRY 4d

Two deliberately separate checks are reported:
1) RAW SENSOR: without fitting a coefficient, does the sign/rank of Δ radiation
   track the sign/rank of Δ common residual both before 19.08 and during the clean
   19-24.08 wave?
2) STRICT ONE-STEP MODEL: one fixed one-feature OLS relation is fitted only on
   earlier completed derivative rows.  At target t it predicts Δ residual_t, then
   reconstructs residual_t from the previous completed harvest-day residual.
   There is no coefficient search, ridge search, cap, lag search or window search.

Decision periods
----------------
- backward validation: every eligible derivative day before 19.08
- clean healthy-August wave: 19.08-24.08
- 26.08+ ageing tail is not used for the verdict

Important exclusions
--------------------
- no PI, FAST, slow-state, TOMO or previous-yield anchor
- no WIND×DRY L3-7 HIGH feature family; only the fixed LAB-37 4v4 control
- no 3/5/6-day weather windows
- no cap and no post-result coefficient tuning
- target-day weather is never used; all weather ends at T-1
- target actual becomes available only after its forecast/derivative prediction
- READ ONLY: db.get_harvest_history + db.get_weather_rows only
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import db

HOURS_PER_FIELD = 3.0
WEATHER_START = date(2026, 7, 1)  # BASE season origin and measured-weather origin
ABC_EPS = 0.20
MIN_BASE_TRAIN_INTERVALS = 35
MIN_FIELD_OBS = 2
BASE_FIELD_RIDGE = 1.5
BASE_SEASON_RIDGE = 0.10
BASE_MAX_ITER = 300

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
        return date.fromisoformat(
            str(v)[:10]
        )
    except Exception:
        return None

def _f(v):
    try:
        x = float(v)

        return (
            x
            if math.isfinite(x)
            else None
        )

    except Exception:
        return None

def _abc(row):
    vals = [
        _f(
            row.get(k)
        )
        for k in (
            "a",
            "b",
            "c",
        )
    ]

    if any(
        v is None
        for v in vals
    ):
        x = _f(
            row.get("_abc")
        )

        return (
            x
            if x is not None
            and x >= 0
            else None
        )

    return float(
        sum(vals)
    )

def _reliable(row):
    q = str(
        row.get("data_quality")
        or row.get("quality")
        or ""
    ).strip().lower()

    return q not in {
        "hinnanguline",
        "ligikaudne",
        "estimated",
        "approximate",
    }

def _events(rows):
    out = []

    for row in rows:
        dd = _d(
            row.get("harvest_date")
        )

        if (
            dd is None
            or not _reliable(row)
        ):
            continue

        try:
            field = int(
                row.get("field_no")
            )
        except Exception:
            continue

        if not 1 <= field <= 14:
            continue

        abc = _abc(row)

        if (
            abc is None
            or abc < 0
        ):
            continue

        try:
            order = int(
                row.get(
                    "harvest_order"
                )
                or 1
            )
        except Exception:
            order = 1

        out.append(
            Event(
                day=dd,
                field=field,
                order=order,
                abc=float(abc),
                interval_days=_f(
                    row.get(
                        "interval_days"
                    )
                ),
            )
        )

    return sorted(
        out,
        key=lambda e: (
            e.day,
            e.order,
            e.field,
        ),
    )

def _field_hist(
    events: Sequence[Event],
    field: int,
):
    return sorted(
        [
            e
            for e in events
            if e.field == field
        ],
        key=lambda e: (
            e.day,
            e.order,
            e.field,
        ),
    )

def _growth(
    prev: Event,
    cur: Event,
):
    g = float(
        (
            cur.day
            - prev.day
        ).days
    )

    g += (
        cur.order
        - prev.order
    ) * (
        HOURS_PER_FIELD
        / 24.0
    )

    return max(
        0.5,
        g,
    )

def _build_intervals(
    events: List[Event],
):
    rows = []

    for field in range(
        1,
        15,
    ):
        hist = _field_hist(
            events,
            field,
        )

        for i in range(
            1,
            len(hist),
        ):
            prev = hist[
                i - 1
            ]
            cur = hist[i]

            gap = int(
                (
                    cur.day
                    - prev.day
                ).days
            )

            if gap <= 0:
                continue

            growth = _growth(
                prev,
                cur,
            )

            days = [
                prev.day
                + timedelta(
                    days=k
                )
                for k in range(
                    1,
                    gap + 1,
                )
            ]

            rows.append({
                "target_date": cur.day,
                "start_date": prev.day,
                "field": int(field),
                "order": int(
                    cur.order
                ),
                "actual": float(
                    cur.abc
                ),
                "growth": float(
                    growth
                ),
                "days": days,
                "per_day_weight": (
                    float(growth)
                    / len(days)
                ),
            })

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        df = df.sort_values(
            [
                "target_date",
                "order",
                "field",
            ]
        ).reset_index(
            drop=True
        )

    return df


def _season_vec(
    dd: date,
):
    season = float(
        (
            dd
            - WEATHER_START
        ).days
    ) / 30.0

    return np.asarray(
        [
            1.0,
            season,
            season * season,
        ],
        dtype=float,
    )

def _base_cache(
    intervals,
):
    days = sorted({
        dd
        for interval_days
        in intervals[
            "days"
        ].tolist()
        for dd in interval_days
    })

    return {
        dd: _season_vec(
            dd
        )
        for dd in days
    }

def _predict_base_intervals(
    intervals,
    beta,
    gammas,
    cache,
):
    preds = []

    for _, row in (
        intervals.iterrows()
    ):
        X = np.vstack([
            cache[dd]
            for dd in row["days"]
        ])

        eta = (
            X @ beta
        )

        daily_prod = np.exp(
            np.clip(
                eta,
                -6.0,
                6.0,
            )
        )

        common = float(
            row[
                "per_day_weight"
            ]
            * np.sum(
                daily_prod
            )
        )

        field = int(
            row["field"]
        )

        if field == 1:
            ff = 1.0
        else:
            ff = math.exp(
                float(
                    gammas[
                        field - 2
                    ]
                )
            )

        preds.append(
            ff * common
        )

    return np.asarray(
        preds,
        dtype=float,
    )

def _fit_base(
    train,
):
    cache = _base_cache(
        train
    )

    y = train[
        "actual"
    ].to_numpy(
        dtype=float
    )

    growth = train[
        "growth"
    ].to_numpy(
        dtype=float
    )

    mean_daily = float(
        np.mean(
            y
            / np.maximum(
                growth,
                0.5,
            )
        )
    )

    beta = np.zeros(
        3,
        dtype=float,
    )

    beta[0] = math.log(
        max(
            mean_daily,
            0.05,
        )
    )

    gammas = np.zeros(
        13,
        dtype=float,
    )

    X_list = []
    w_list = []
    field_list = []

    for _, row in (
        train.iterrows()
    ):
        X_list.append(
            np.vstack([
                cache[dd]
                for dd in row[
                    "days"
                ]
            ])
        )

        w_list.append(
            float(
                row[
                    "per_day_weight"
                ]
            )
        )

        field_list.append(
            int(
                row["field"]
            )
        )

    lr = 0.035
    b1 = 0.9
    b2 = 0.999
    adam_eps = 1e-8

    mb = np.zeros_like(
        beta
    )
    vb = np.zeros_like(
        beta
    )
    mg = np.zeros_like(
        gammas
    )
    vg = np.zeros_like(
        gammas
    )

    prev_obj = None

    for step in range(
        1,
        BASE_MAX_ITER + 1,
    ):
        gb = np.zeros_like(
            beta
        )
        gg = np.zeros_like(
            gammas
        )

        obj_data = 0.0

        for i, (
            X,
            weight,
            field,
        ) in enumerate(
            zip(
                X_list,
                w_list,
                field_list,
            )
        ):
            eta = X @ beta

            prod = np.exp(
                np.clip(
                    eta,
                    -6.0,
                    6.0,
                )
            )

            common = float(
                weight
                * np.sum(
                    prod
                )
            )

            if field == 1:
                ff = 1.0
                gi = None
            else:
                gi = (
                    field - 2
                )
                ff = math.exp(
                    float(
                        gammas[gi]
                    )
                )

            pred = max(
                ff * common,
                1e-8,
            )

            resid = (
                math.log(
                    pred
                    + ABC_EPS
                )
                - math.log(
                    float(y[i])
                    + ABC_EPS
                )
            )

            obj_data += (
                resid
                * resid
            )

            shrink = (
                pred
                / (
                    pred
                    + ABC_EPS
                )
            )

            denom = max(
                float(
                    np.sum(
                        prod
                    )
                ),
                1e-12,
            )

            x_bar = (
                (
                    prod[:, None]
                    * X
                ).sum(axis=0)
                / denom
            )

            gb += (
                2.0
                * resid
                * shrink
                * x_bar
            )

            if gi is not None:
                gg[gi] += (
                    2.0
                    * resid
                    * shrink
                )

        reg = (
            BASE_SEASON_RIDGE
            * float(
                np.sum(
                    beta[1:]
                    * beta[1:]
                )
            )
            + BASE_FIELD_RIDGE
            * float(
                np.sum(
                    gammas
                    * gammas
                )
            )
        )

        gb[1:] += (
            2.0
            * BASE_SEASON_RIDGE
            * beta[1:]
        )

        gg += (
            2.0
            * BASE_FIELD_RIDGE
            * gammas
        )

        obj = (
            obj_data
            + reg
        )

        scale = max(
            len(train),
            1,
        )

        gb /= scale
        gg /= scale

        nb = float(
            np.linalg.norm(
                gb
            )
        )

        ng = float(
            np.linalg.norm(
                gg
            )
        )

        if nb > 10.0:
            gb *= (
                10.0
                / nb
            )

        if ng > 10.0:
            gg *= (
                10.0
                / ng
            )

        mb = (
            b1 * mb
            + (
                1.0
                - b1
            ) * gb
        )

        vb = (
            b2 * vb
            + (
                1.0
                - b2
            )
            * (
                gb * gb
            )
        )

        mg = (
            b1 * mg
            + (
                1.0
                - b1
            ) * gg
        )

        vg = (
            b2 * vg
            + (
                1.0
                - b2
            )
            * (
                gg * gg
            )
        )

        mbh = (
            mb
            / (
                1.0
                - b1 ** step
            )
        )

        vbh = (
            vb
            / (
                1.0
                - b2 ** step
            )
        )

        mgh = (
            mg
            / (
                1.0
                - b1 ** step
            )
        )

        vgh = (
            vg
            / (
                1.0
                - b2 ** step
            )
        )

        beta -= (
            lr
            * mbh
            / (
                np.sqrt(
                    vbh
                )
                + adam_eps
            )
        )

        gammas -= (
            lr
            * mgh
            / (
                np.sqrt(
                    vgh
                )
                + adam_eps
            )
        )

        gammas = np.clip(
            gammas,
            math.log(0.5),
            math.log(1.5),
        )

        if (
            prev_obj is not None
            and step > 80
            and abs(
                prev_obj
                - obj
            )
            < (
                1e-7
                * max(
                    1.0,
                    abs(
                        prev_obj
                    ),
                )
            )
        ):
            break

        prev_obj = obj

    return {
        "beta": beta,
        "gammas": gammas,
    }

def _predict_base(
    fit,
    test,
):
    cache = _base_cache(
        test
    )

    return _predict_base_intervals(
        test,
        fit["beta"],
        fit["gammas"],
        cache,
    )

def _strict_base_rows(
    intervals,
):
    rows = []

    for target in sorted(
        intervals[
            "target_date"
        ].unique()
    ):
        train = intervals[
            intervals[
                "target_date"
            ] < target
        ].copy()

        test = intervals[
            intervals[
                "target_date"
            ] == target
        ].copy()

        if len(train) < (
            MIN_BASE_TRAIN_INTERVALS
        ):
            continue

        counts = (
            train.groupby(
                "field"
            ).size().to_dict()
        )

        valid_idx = []

        for idx, row in (
            test.iterrows()
        ):
            if (
                counts.get(
                    int(
                        row["field"]
                    ),
                    0,
                )
                >= MIN_FIELD_OBS
            ):
                valid_idx.append(
                    idx
                )

        if not valid_idx:
            continue

        test = intervals.loc[
            valid_idx
        ].copy()

        fit = _fit_base(
            train
        )

        pred = _predict_base(
            fit,
            test,
        )

        for j, (
            idx,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            rows.append({
                "target_date": target,
                "start_date": row["start_date"],
                "field": int(
                    row["field"]
                ),
                "order": int(
                    row["order"]
                ),
                "growth": float(
                    row["growth"]
                ),
                "actual": float(
                    row["actual"]
                ),
                "base": float(
                    pred[j]
                ),
                "train_n": int(
                    len(train)
                ),
            })

    return pd.DataFrame(
        rows
    )

# =====================================================================
# FIRST-DERIVATIVE WEATHER AUDIT
# =====================================================================

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 24)
EARLIER_END = date(2026, 8, 18)
LATE_INFO_START = date(2026, 8, 26)

WX_BLOCK_DAYS = 4              # EXACT LAB-37 definition; locked
DERIV_MIN_TRAIN_DAYS = 5       # fixed expanding OLS minimum; no search
DERIV_EPS = ABC_EPS

# Pre-declared evidence gates. These do NOT choose a model/window after results.
EARLIER_MIN_N = 4
EARLIER_SIGN_GATE = 65.0
EARLIER_RHO_GATE = 0.30
FOCUS_MIN_N = 4
FOCUS_SIGN_GATE = 80.0
FOCUS_RHO_GATE = 0.50
MODEL_FOCUS_SIGN_GATE = 67.0
MODEL_WAVE_DIR_GATE = 80.0

FEATURES = {
    'rad': ('wx4_d_rad', 'Δ radiatsioon 4d'),
    'night': ('wx4_d_nightstress', 'Δ ööstress 4d · kontroll'),
    'wd': ('wx4_d_winddry', 'Δ WIND×DRY 4d · kontroll'),
}


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _weather_map(rows):
    """Exact measured/checked weather parsing used by LAB-37."""
    out = {}
    for row in rows:
        dd = _d(row.get('weather_date'))
        if dd is None:
            continue
        if str(row.get('data_kind') or '').strip().lower() != 'measured':
            continue
        if not bool(row.get('checked')):
            continue
        night = _f(row.get('temp_night_avg_c'))
        rad = _f(row.get('radiation_mj_m2'))
        wind = _f(row.get('wind_avg_ms'))
        rh = _f(row.get('humidity_avg_pct'))
        if None in (night, rad, wind, rh):
            continue
        out[dd] = {
            'night': float(night),
            'rad': float(rad),
            'wind': float(wind),
            'rh': float(rh),
            'winddry': float(wind) * (100.0 - float(rh)),
        }
    return out


def _night_stress(night_c):
    """Exact LAB-37 dimensionless night-stress hinge."""
    cold = max(0.0, 16.0 - float(night_c)) / 5.0
    heat = max(0.0, float(night_c) - 20.0) / 5.0
    return cold * cold + heat * heat


def _weather_block(end_day, weather):
    days = [end_day - timedelta(days=k) for k in reversed(range(WX_BLOCK_DAYS))]
    if any(dd not in weather for dd in days):
        return None
    arr = np.asarray([
        [
            float(weather[dd]['rad']),
            float(_night_stress(weather[dd]['night'])),
            float(weather[dd]['winddry']),
        ]
        for dd in days
    ], dtype=float)
    return np.mean(arr, axis=0)


def _weather_delta4(target, weather):
    """Exact LAB-37 4v4: current=T-4..T-1, previous=T-8..T-5."""
    cur = _weather_block(target - timedelta(days=1), weather)
    prev = _weather_block(target - timedelta(days=WX_BLOCK_DAYS + 1), weather)
    if cur is None or prev is None:
        return None
    return cur - prev


def _daily_derivative_rows(strict, weather):
    """Build one causal daily common residual row, then first-difference it by harvest day."""
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue
        resid_fields = (
            np.log(g['actual'].to_numpy(dtype=float) + DERIV_EPS)
            - np.log(g['base'].to_numpy(dtype=float) + DERIV_EPS)
        )
        wx4 = _weather_delta4(target, weather)
        row = {
            'date': target,
            'fields': ','.join(str(int(x)) for x in g.sort_values(['order','field'])['field'].tolist()),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            # Positive residual = actual above BASE; negative = BASE too high.
            'common_resid': float(np.median(resid_fields)),
            'base_error': float(g['base'].sum() - g['actual'].sum()),
            'base_train_n': int(g['train_n'].min()),
        }
        if wx4 is None:
            row['wx4_d_rad'] = np.nan
            row['wx4_d_nightstress'] = np.nan
            row['wx4_d_winddry'] = np.nan
        else:
            row['wx4_d_rad'] = float(wx4[0])
            row['wx4_d_nightstress'] = float(wx4[1])
            row['wx4_d_winddry'] = float(wx4[2])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if out.empty:
        return out
    out['prev_date'] = out['date'].shift(1)
    out['prev_common_resid'] = out['common_resid'].shift(1)
    out['delta_common_resid'] = out['common_resid'] - out['prev_common_resid']
    out['actual_per_field'] = out['actual'] / np.maximum(out['n_fields'], 1)
    out['base_per_field'] = out['base'] / np.maximum(out['n_fields'], 1)
    return out


def _spearman_no_scipy(a, b):
    aa = pd.Series(np.asarray(a, dtype=float)).rank(method='average').to_numpy(dtype=float)
    bb = pd.Series(np.asarray(b, dtype=float)).rank(method='average').to_numpy(dtype=float)
    if len(aa) < 2 or np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _raw_sensor_score(df, feature):
    x = df[['delta_common_resid', feature]].dropna().copy()
    if x.empty:
        return {'N':0, 'Scored N':0, 'Suunahitt %':np.nan, 'rho':np.nan, 'Pearson':np.nan}
    y = x['delta_common_resid'].to_numpy(dtype=float)
    z = x[feature].to_numpy(dtype=float)
    mask = (np.abs(y) > 1e-12) & (np.abs(z) > 1e-12)
    scored = int(mask.sum())
    sign = float(np.mean(np.sign(y[mask]) == np.sign(z[mask])) * 100.0) if scored else np.nan
    rho = _spearman_no_scipy(y, z)
    pear = float(np.corrcoef(y, z)[0, 1]) if len(x) >= 2 and np.std(y)>1e-12 and np.std(z)>1e-12 else np.nan
    return {'N':int(len(x)), 'Scored N':scored, 'Suunahitt %':sign, 'rho':rho, 'Pearson':pear}


def _fit_one_feature_delta(train, feature):
    """Fixed one-feature OLS with intercept. Coefficients are estimated, never searched."""
    x = train[feature].to_numpy(dtype=float)
    y = train['delta_common_resid'].to_numpy(dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        sd = 1.0
    z = (x - mu) / sd
    y0 = float(np.mean(y))
    denom = float(np.dot(z, z))
    beta = float(np.dot(z, y - y0) / denom) if denom > 1e-12 else 0.0
    return {'feature':feature, 'mu':mu, 'sd':sd, 'intercept':y0, 'beta':beta}


def _predict_one_feature_delta(row, fit):
    z = (float(row[fit['feature']]) - fit['mu']) / fit['sd']
    return float(fit['intercept'] + fit['beta'] * z)


def _walk_derivative_models(daily):
    """Strict expanding one-step derivative models for primary + negative controls."""
    out = daily.copy()
    for slug, (feature, _) in FEATURES.items():
        out[f'{slug}_delta_pred'] = np.nan
        out[f'{slug}_resid_pred'] = np.nan
        out[f'{slug}_forecast'] = np.nan
        out[f'{slug}_train_n'] = 0
        out[f'{slug}_beta_std'] = np.nan

        for i, row in out.iterrows():
            if pd.isna(row.get(feature)) or pd.isna(row.get('prev_common_resid')):
                continue
            prior = out.iloc[:i].dropna(subset=[feature, 'delta_common_resid']).copy()
            if len(prior) < DERIV_MIN_TRAIN_DAYS:
                continue
            fit = _fit_one_feature_delta(prior, feature)
            d_hat = _predict_one_feature_delta(row, fit)
            # Crucial distinction from FAST: previous residual is not copied forward.
            # Weather predicts its CHANGE first; only then do we form today's level.
            r_hat = float(row['prev_common_resid']) + float(d_hat)
            try:
                forecast = float(row['base']) * math.exp(r_hat)
            except OverflowError:
                forecast = np.nan
            out.at[i, f'{slug}_delta_pred'] = d_hat
            out.at[i, f'{slug}_resid_pred'] = r_hat
            out.at[i, f'{slug}_forecast'] = forecast
            out.at[i, f'{slug}_train_n'] = int(len(prior))
            out.at[i, f'{slug}_beta_std'] = float(fit['beta'])
    return out


def _strict_delta_score(df, slug):
    col = f'{slug}_delta_pred'
    x = df[['delta_common_resid', col]].dropna().copy()
    if x.empty:
        return {'N':0, 'Suunahitt %':np.nan, 'rho':np.nan, 'Delta MAE':np.nan}
    y = x['delta_common_resid'].to_numpy(dtype=float)
    p = x[col].to_numpy(dtype=float)
    mask = (np.abs(y) > 1e-12) & (np.abs(p) > 1e-12)
    sign = float(np.mean(np.sign(y[mask]) == np.sign(p[mask])) * 100.0) if int(mask.sum()) else np.nan
    return {
        'N':int(len(x)),
        'Suunahitt %':sign,
        'rho':_spearman_no_scipy(y, p),
        'Delta MAE':float(np.mean(np.abs(y - p))),
    }


def _forecast_metrics(df, pred_col):
    x = df[['actual','base',pred_col,'n_fields']].dropna().copy()
    if x.empty:
        return {'N':0,'MAE':np.nan,'MAPE %':np.nan,'Bias':np.nan,'±20%':np.nan,'Worst AE':np.nan}
    a = x['actual'].to_numpy(dtype=float)
    p = x[pred_col].to_numpy(dtype=float)
    ae = np.abs(p-a)
    ape = ae / np.maximum(np.abs(a), 0.5)
    return {
        'N':int(len(x)),
        'MAE':float(np.mean(ae)),
        'MAPE %':100.0*float(np.mean(ape)),
        'Bias':float(np.mean(p-a)),
        '±20%':100.0*float(np.mean(ape <= 0.20)),
        'Worst AE':float(np.max(ae)),
    }


def _matched_base_vs_rad(df):
    """Compare BASE and RAD derivative on exactly the same strict RAD-ready days."""
    x = df[['date','fields','actual','base','n_fields','rad_forecast']].dropna().copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame()

    base_m = _forecast_metrics(x.assign(_base_same=x['base']), '_base_same')
    rad_m = _forecast_metrics(x, 'rad_forecast')
    improvement_pct = (
        100.0 * (base_m['MAE'] - rad_m['MAE']) / base_m['MAE']
        if pd.notna(base_m['MAE']) and base_m['MAE'] > 1e-12 and pd.notna(rad_m['MAE'])
        else np.nan
    )
    summary = pd.DataFrame([
        {
            'Variant':'A · LONG BASE · same strict days',
            'N':base_m['N'], 'MAE':base_m['MAE'], 'MAPE %':base_m['MAPE %'],
            'Bias':base_m['Bias'], '±20%':base_m['±20%'], 'Worst AE':base_m['Worst AE'],
            'Parandus vs BASE %':0.0,
        },
        {
            'Variant':'B · RAD derivative · same strict days',
            'N':rad_m['N'], 'MAE':rad_m['MAE'], 'MAPE %':rad_m['MAPE %'],
            'Bias':rad_m['Bias'], '±20%':rad_m['±20%'], 'Worst AE':rad_m['Worst AE'],
            'Parandus vs BASE %':improvement_pct,
        },
    ])

    day = x.sort_values('date').copy()
    day['BASE viga'] = day['base'] - day['actual']
    day['RAD viga'] = day['rad_forecast'] - day['actual']
    day['BASE AE'] = np.abs(day['BASE viga'])
    day['RAD AE'] = np.abs(day['RAD viga'])
    day['RAD võit kasti'] = day['BASE AE'] - day['RAD AE']
    return summary, day


def _wave_direction_stats(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if len(x) < 2:
        return np.nan, 0
    a = x['actual'].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float),1.0)
    p = x[pred_col].to_numpy(dtype=float) / np.maximum(x['n_fields'].to_numpy(dtype=float),1.0)
    hits = []
    for i in range(1,len(x)):
        da = a[i]-a[i-1]
        dp = p[i]-p[i-1]
        if abs(da) <= 1e-12:
            continue
        hits.append(int(np.sign(da)==np.sign(dp)))
    return (100.0*float(np.mean(hits)), len(hits)) if hits else (np.nan,0)


def _peak_info(df, pred_col):
    x = df[['date','actual','n_fields',pred_col]].dropna().sort_values('date').copy()
    if x.empty:
        return None, None, None
    x['a_pf'] = x['actual'] / np.maximum(x['n_fields'],1)
    x['p_pf'] = x[pred_col] / np.maximum(x['n_fields'],1)
    ad = x.loc[x['a_pf'].idxmax(),'date']
    pd_ = x.loc[x['p_pf'].idxmax(),'date']
    dates = list(x['date'])
    shift = dates.index(pd_) - dates.index(ad)
    return ad, pd_, int(shift)


def _period_row(df, slug, label):
    feature = FEATURES[slug][0]
    raw = _raw_sensor_score(df, feature)
    strict = _strict_delta_score(df, slug)
    fm = _forecast_metrics(df, f'{slug}_forecast')
    wave_hit, wave_n = _wave_direction_stats(df, f'{slug}_forecast')
    return {
        'Variant':label,
        'Raw N':raw['N'],
        'Raw suunahitt %':raw['Suunahitt %'],
        'Raw rho':raw['rho'],
        'Strict N':strict['N'],
        'Strict Δ suunahitt %':strict['Suunahitt %'],
        'Strict Δ rho':strict['rho'],
        'Strict Δ MAE':strict['Delta MAE'],
        'Forecast MAE':fm['MAE'],
        'Wave suunahitt %':wave_hit,
        'Wave N':wave_n,
    }


def _verdict(earlier, focus, healthy):
    er = _raw_sensor_score(earlier, FEATURES['rad'][0])
    fr = _raw_sensor_score(focus, FEATURES['rad'][0])
    fs = _strict_delta_score(focus, 'rad')
    fm = _forecast_metrics(focus, 'rad_forecast')
    bm = _forecast_metrics(focus.assign(base_identity=focus['base']), 'base_identity')
    wave_hit, wave_n = _wave_direction_stats(focus, 'rad_forecast')

    if er['N'] < EARLIER_MIN_N or fr['N'] < FOCUS_MIN_N:
        return 'INSUFFICIENT DERIVATIVE HISTORY', (
            f'Only {er["N"]} earlier and {fr["N"]} focus raw derivative pairs are available. '
            'Report the tables, but do not change the architecture or weather window.'
        )

    raw_supported = (
        pd.notna(er['Suunahitt %']) and er['Suunahitt %'] >= EARLIER_SIGN_GATE
        and pd.notna(er['rho']) and er['rho'] >= EARLIER_RHO_GATE
        and pd.notna(fr['Suunahitt %']) and fr['Suunahitt %'] >= FOCUS_SIGN_GATE
        and pd.notna(fr['rho']) and fr['rho'] >= FOCUS_RHO_GATE
    )

    # Specificity is diagnostic: if a negative control is at least as strong on both
    # sign and rho over the whole healthy set, we should not call the effect radiation-specific.
    hr = _raw_sensor_score(healthy, FEATURES['rad'][0])
    controls = [_raw_sensor_score(healthy, FEATURES[s][0]) for s in ('night','wd')]
    non_specific = any(
        pd.notna(c['Suunahitt %']) and pd.notna(c['rho'])
        and pd.notna(hr['Suunahitt %']) and pd.notna(hr['rho'])
        and c['Suunahitt %'] >= hr['Suunahitt %']
        and c['rho'] >= hr['rho']
        for c in controls
    )

    model_ready = (
        raw_supported
        and fs['N'] >= 3
        and pd.notna(fs['Suunahitt %']) and fs['Suunahitt %'] >= MODEL_FOCUS_SIGN_GATE
        and pd.notna(wave_hit) and wave_hit >= MODEL_WAVE_DIR_GATE
        and fm['N'] >= 3 and bm['N'] >= 3
        and pd.notna(fm['MAE']) and pd.notna(bm['MAE']) and fm['MAE'] <= bm['MAE']
    )

    if model_ready and not non_specific:
        return 'FIRST-DERIVATIVE MODEL CANDIDATE', (
            f'Radiation 4v4 tracks Δ common residual both earlier ({er["Suunahitt %"]:.0f}%, rho {er["rho"]:+.2f}) '
            f'and in 19-24.08 ({fr["Suunahitt %"]:.0f}%, rho {fr["rho"]:+.2f}); the strict one-step model also '
            f'reconstructs the focus wave with {wave_hit:.0f}% direction hit and does not worsen focus MAE. '
            'Freeze this exact derivative architecture; do not tune it yet.'
        )

    if raw_supported:
        if non_specific:
            return 'WEATHER DERIVATIVE SIGNAL · NOT RADIATION-SPECIFIC', (
                f'The first-derivative idea survives backward validation and the clean wave, but a negative control is at least as strong '
                f'over the healthy period. The target change appears real; attribution to radiation alone is not yet justified.'
            )
        return 'DERIVATIVE SIGNAL YES · LEVEL MODEL NOT READY', (
            f'Raw Δ radiation 4v4 tracks Δ common residual earlier ({er["Suunahitt %"]:.0f}%, rho {er["rho"]:+.2f}) '
            f'and in 19-24.08 ({fr["Suunahitt %"]:.0f}%, rho {fr["rho"]:+.2f}), but the fixed strict one-step reconstruction '
            'does not pass the practical wave/MAE gates. Keep the derivative insight; do not turn it into a correction yet.'
        )

    er_sign = '—' if pd.isna(er['Suunahitt %']) else f"{float(er['Suunahitt %']):.0f}%"
    er_rho = '—' if pd.isna(er['rho']) else f"{float(er['rho']):+.2f}"
    fr_sign = '—' if pd.isna(fr['Suunahitt %']) else f"{float(fr['Suunahitt %']):.0f}%"
    fr_rho = '—' if pd.isna(fr['rho']) else f"{float(fr['rho']):+.2f}"
    return 'FIRST-DERIVATIVE HYPOTHESIS FAILS BACKWARD CHECK', (
        f'Radiation 4v4 does not meet the pre-declared raw-sensor gates on both periods: earlier '
        f'{er_sign}, rho {er_rho}; focus {fr_sign}, rho {fr_rho}. '
        'Do not rescue it with a different weather window or lag.'
    )


def _fmt_pct(v):
    return '—' if pd.isna(v) else f'{float(v):.0f}%'


def _fmt_rho(v):
    return '—' if pd.isna(v) else f'{float(v):+.2f}'


def main():
    st.set_page_config(page_title='KurgiMootor · first derivative', layout='wide')
    st.title('KurgiMootor · first-derivative weather audit')
    st.caption('LAB-43b · same BASE · exact LAB-37 4v4 weather · new target = Δ common residual · strict walk-forward · matched BASE control added · READ ONLY')
    st.info(
        'This LAB changes the TARGET, not the weather window. Earlier weather tests asked for the absolute BASE correction. '
        'Here weather only asks whether the common BASE residual is moving up or down. Primary = Δ radiation 4d; '
        'night-stress and WIND×DRY 4d are fixed negative controls.'
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        strict = _strict_base_rows(intervals)
        if strict.empty:
            raise RuntimeError('Strict BASE OOS rows did not form.')
        earliest = min(strict['target_date'])
        latest = max(strict['target_date'])
        weather_rows = db.get_weather_rows(max(WEATHER_START, earliest - timedelta(days=20)), latest)
        weather = _weather_map(weather_rows)
        daily0 = _daily_derivative_rows(strict, weather)
        if daily0.empty:
            raise RuntimeError('Daily common-residual rows did not form.')
        daily = _walk_derivative_models(daily0)
        earlier = daily[daily['date'] <= EARLIER_END].copy()
        focus = daily[(daily['date'] >= FOCUS_START) & (daily['date'] <= FOCUS_END)].copy()
        healthy = daily[daily['date'] <= FOCUS_END].copy()
        late = daily[daily['date'] >= LATE_INFO_START].copy()
        verdict, verdict_text = _verdict(earlier, focus, healthy)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.markdown('### 1. Otsus · kas puuduv info oli taseme asemel muutuses?')
    if verdict.startswith('FIRST-DERIVATIVE MODEL CANDIDATE'):
        st.success('✅ ' + verdict + ': ' + verdict_text)
    elif verdict.startswith('DERIVATIVE SIGNAL') or verdict.startswith('WEATHER DERIVATIVE') or verdict.startswith('INSUFFICIENT'):
        st.warning('🟡 ' + verdict + ': ' + verdict_text)
    else:
        st.error('⛔ ' + verdict + ': ' + verdict_text)

    # RAW SENSOR is the key anti-circle table: no fitted coefficient at all.
    sensor_rows = []
    for slug, (_, label) in FEATURES.items():
        e = _raw_sensor_score(earlier, FEATURES[slug][0])
        f = _raw_sensor_score(focus, FEATURES[slug][0])
        h = _raw_sensor_score(healthy, FEATURES[slug][0])
        sensor_rows.append({
            'Sensor':label,
            'Varasem N':e['N'],'Varasem suunahitt %':e['Suunahitt %'],'Varasem rho':e['rho'],
            '19-24 N':f['N'],'19-24 suunahitt %':f['Suunahitt %'],'19-24 rho':f['rho'],
            'Terve periood N':h['N'],'Terve periood suunahitt %':h['Suunahitt %'],'Terve periood rho':h['rho'],
        })
    sensor = pd.DataFrame(sensor_rows)
    st.dataframe(sensor.style.format({
        'Varasem suunahitt %':_fmt_pct,'Varasem rho':_fmt_rho,
        '19-24 suunahitt %':_fmt_pct,'19-24 rho':_fmt_rho,
        'Terve periood suunahitt %':_fmt_pct,'Terve periood rho':_fmt_rho,
    }), use_container_width=True, hide_index=True)
    st.caption(
        f'Pre-declared primary raw gates: earlier N≥{EARLIER_MIN_N}, sign≥{EARLIER_SIGN_GATE:.0f}%, rho≥{EARLIER_RHO_GATE:.2f}; '
        f'19-24 N≥{FOCUS_MIN_N}, sign≥{FOCUS_SIGN_GATE:.0f}%, rho≥{FOCUS_RHO_GATE:.2f}. '
        'No coefficient is fitted in this table.'
    )

    st.markdown('### 2. Puhas terve taime laine · 19.-24.08')
    comp = []
    base_m = _forecast_metrics(focus.assign(base_identity=focus['base']), 'base_identity')
    base_wave, base_wave_n = _wave_direction_stats(focus, 'base')
    comp.append({'Variant':'A · LONG BASE','N':base_m['N'],'MAE':base_m['MAE'],'MAPE %':base_m['MAPE %'],'±20%':base_m['±20%'],'Worst AE':base_m['Worst AE'],'Wave suunahitt %':base_wave,'Wave N':base_wave_n})
    for slug, (_, label) in FEATURES.items():
        m = _forecast_metrics(focus, f'{slug}_forecast')
        wh, wn = _wave_direction_stats(focus, f'{slug}_forecast')
        comp.append({'Variant':('B · RAD derivative' if slug=='rad' else label),'N':m['N'],'MAE':m['MAE'],'MAPE %':m['MAPE %'],'±20%':m['±20%'],'Worst AE':m['Worst AE'],'Wave suunahitt %':wh,'Wave N':wn})
    st.dataframe(pd.DataFrame(comp).style.format({
        'MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
        '±20%':_fmt_pct,'Worst AE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Wave suunahitt %':_fmt_pct,
    }), use_container_width=True, hide_index=True)

    ad, pd_, shift = _peak_info(focus, 'rad_forecast')
    if ad is not None:
        st.caption(f'RAD derivative focus peak: actual {_fmt_day(ad)} · predicted {_fmt_day(pd_)} · shift {shift:+d} harvest day(s).')

    focus_show = focus.copy()
    focus_show['actual_minus_base'] = focus_show['actual'] - focus_show['base']
    focus_show['rad_error'] = focus_show['rad_forecast'] - focus_show['actual']
    st.dataframe(
        focus_show[[
            'date','fields','actual','base','actual_minus_base','common_resid','delta_common_resid',
            'wx4_d_rad','wx4_d_nightstress','wx4_d_winddry',
            'rad_delta_pred','rad_resid_pred','rad_forecast','rad_error','rad_train_n','rad_beta_std',
        ]].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'LONG BASE','actual_minus_base':'Tegelik−BASE',
            'common_resid':'Ühine log-viga','delta_common_resid':'Δ ühine log-viga',
            'wx4_d_rad':'Δ rad 4d','wx4_d_nightstress':'Δ ööstress 4d','wx4_d_winddry':'Δ WD 4d',
            'rad_delta_pred':'RAD Δ ennustus','rad_resid_pred':'RAD taseme ennustus','rad_forecast':'RAD forecast',
            'rad_error':'RAD viga','rad_train_n':'Train N','rad_beta_std':'β std',
        }).style.format({
            'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}','Tegelik−BASE':'{:+.1f}',
            'Ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ rad 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
            'Δ ööstress 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ WD 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'RAD Δ ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'RAD taseme ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'RAD forecast':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
            'RAD viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'β std':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
        }), use_container_width=True, hide_index=True
    )

    st.markdown('### 3. Varasem kontroll · sama sihtmuutuja enne 19.08')
    st.caption('Kõigepealt aus tasemevõrdlus: LONG BASE ja RAD derivative ainult neil samadel varasematel päevadel, mil strict RAD forecast oli juba võimalik. Nii ei võrrelda 4-päevast kandidaati pikema BASE perioodiga.')

    matched_summary, matched_days = _matched_base_vs_rad(earlier)
    if matched_summary.empty:
        st.warning('Varasemas perioodis ei ole veel ühtegi ühist strict RAD-ready päeva BASE-võrdluseks.')
    else:
        st.dataframe(matched_summary.style.format({
            'MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
            'MAPE %':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
            'Bias':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
            '±20%':_fmt_pct,
            'Worst AE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
            'Parandus vs BASE %':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}%',
        }), use_container_width=True, hide_index=True)

        st.dataframe(
            matched_days[[
                'date','fields','actual','base','rad_forecast','BASE viga','RAD viga','BASE AE','RAD AE','RAD võit kasti'
            ]].rename(columns={
                'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'LONG BASE','rad_forecast':'RAD derivative'
            }).style.format({
                'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','LONG BASE':'{:.1f}','RAD derivative':'{:.1f}',
                'BASE viga':'{:+.1f}','RAD viga':'{:+.1f}','BASE AE':'{:.1f}','RAD AE':'{:.1f}',
                'RAD võit kasti':'{:+.1f}',
            }), use_container_width=True, hide_index=True
        )

    st.markdown('#### 3b. Derivative diagnostika · samad varasemad andmed')
    earlier_scores = []
    for slug, (_, label) in FEATURES.items():
        earlier_scores.append(_period_row(earlier, slug, label))
    st.dataframe(pd.DataFrame(earlier_scores).style.format({
        'Raw suunahitt %':_fmt_pct,'Raw rho':_fmt_rho,
        'Strict Δ suunahitt %':_fmt_pct,'Strict Δ rho':_fmt_rho,
        'Strict Δ MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.3f}',
        'Forecast MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'Wave suunahitt %':_fmt_pct,
    }), use_container_width=True, hide_index=True)

    earlier_show = earlier.dropna(subset=['delta_common_resid']).copy()
    st.dataframe(
        earlier_show[[
            'date','fields','common_resid','delta_common_resid','wx4_d_rad','wx4_d_nightstress','wx4_d_winddry',
            'rad_delta_pred','rad_train_n'
        ]].rename(columns={
            'date':'Päev','fields':'Põllud','common_resid':'Ühine log-viga','delta_common_resid':'Δ ühine log-viga',
            'wx4_d_rad':'Δ rad 4d','wx4_d_nightstress':'Δ ööstress 4d','wx4_d_winddry':'Δ WD 4d',
            'rad_delta_pred':'RAD Δ strict ennustus','rad_train_n':'Train N',
        }).style.format({
            'Päev':_fmt_day,
            'Ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ rad 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
            'Δ ööstress 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ WD 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'RAD Δ strict ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
        }), use_container_width=True, hide_index=True
    )

    if not late.empty:
        with st.expander('26.08+ vananeva taime saba · ainult info, mitte verdict'):
            st.dataframe(late[['date','fields','actual','base','common_resid','delta_common_resid','wx4_d_rad','rad_delta_pred']].rename(columns={
                'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'BASE','common_resid':'Ühine log-viga',
                'delta_common_resid':'Δ ühine log-viga','wx4_d_rad':'Δ rad 4d','rad_delta_pred':'RAD Δ ennustus'
            }).style.format({
                'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','BASE':'{:.1f}',
                'Ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
                'Δ ühine log-viga':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
                'Δ rad 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
                'RAD Δ ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            }), use_container_width=True, hide_index=True)

    st.caption(
        f'Locks: weather blocks exactly {WX_BLOCK_DAYS}d vs {WX_BLOCK_DAYS}d; strict OLS starts after '
        f'{DERIV_MIN_TRAIN_DAYS} earlier derivative rows; no ridge, cap, lag or window search. '
        'RAW SENSOR table is coefficient-free and is the primary anti-overfit check. The earlier BASE-vs-RAD table is matched only on days where the strict RAD forecast already exists; it adds no tuning.'
    )
    st.success(
        '🔒 LEAKAGE LOCK: every BASE target is trained only on earlier intervals. Δ residual at target is unknown until harvest. '
        'Every strict derivative coefficient uses only earlier completed derivative rows. The reconstructed target residual starts from '
        'the previous completed harvest-day residual, which is already known, plus a weather-predicted CHANGE. Target-day weather is excluded. READ ONLY.'
    )


if __name__ == '__main__':
    main()
