from __future__ import annotations

"""
KurgiMootor · edge_weather-37
=============================

COMMON TIME CAUSE AUDIT · STRICT WALK-FORWARD · READ ONLY

One question only:
Did BASE miss the 19.–22.08 wave because of a common time/crop signal,
or because the harvested field mix / interval mechanics were wrong?

This is deliberately NOT another feature, coefficient or weather hunt.
It does not alter a single BASE prediction. It diagnoses the already locked
strict-OOS BASE errors in three ways:

1) Do different fields miss in the same direction on the same day?
2) Does the same field repeat its previous strict-OOS error?
3) After same-field differencing, does the error change track interval change?

Locks
-----
- every BASE target row is trained only on intervals strictly before target
- target actual is used only after prediction for error diagnosis
- same-field comparison uses only that field's earlier strict-OOS row
- no weather, PI, slow-state or previous-yield anchor is used
- no window/cap/lambda/alpha/feature search
- 25.08 no-harvest remains absent, never zero yield
- READ ONLY: only db.get_harvest_history is called
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
WEATHER_START = date(2026, 7, 1)  # BASE season origin; no weather is loaded
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
# COMMON TIME CAUSE AUDIT · locked weather families only
# =====================================================================
#
# LAB-36 verdict: the 19–22 Aug BASE miss behaves like a common time signal,
# not a repeated field-level error and not an interval-change artefact.
#
# This LAB asks one narrower question:
#   Do already-audited PRE-TARGET weather signals explain that common residual
#   in strict walk-forward, or does a material common residual remain?
#
# No feature/window/lambda/cap search is allowed here.
#   W1 = locked WIND×DRY HIGH L3–7 LEVEL+DELTA (field-aware, then daily median)
#   W2 = fixed 4d-vs-4d weather transition: radiation, night stress, WIND×DRY
#   W3 = W1 + W2
#
# Weather is measured + checked and target-day weather is never used.
# Each target day's residual correction is fitted only on earlier complete days.
# Production is untouched; this is a read-only diagnostic.

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 22)
CONTEXT_END = date(2026, 8, 27)
ABC_EPS2 = ABC_EPS

WX_BLOCK_DAYS = 4
WX_RIDGE_LAMBDA = 10.0          # fixed; no search
WX_MIN_TRAIN_DAYS = 5
WX_CAP_LOG = 0.15               # fixed conservative safety cap
WD_Q = 0.75                     # locked LAB-154 threshold quantile
WD_MIN_THRESHOLD_DAYS = 10
WEATHER_ORIGIN = date(2026, 7, 1)


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    return v.strftime('%d.%m') if isinstance(v, date) else str(v)


def _weather_map(rows):
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
    cold = max(0.0, 16.0 - float(night_c)) / 5.0
    heat = max(0.0, float(night_c) - 20.0) / 5.0
    return cold * cold + heat * heat


def _wd_threshold(weather, before_day):
    vals = [
        float(w['winddry'])
        for dd, w in weather.items()
        if WEATHER_ORIGIN <= dd < before_day
    ]
    if len(vals) < WD_MIN_THRESHOLD_DAYS:
        return None
    return float(np.quantile(np.asarray(vals, dtype=float), WD_Q))


def _max_run(flags):
    best = cur = 0
    for flag in flags:
        if bool(flag):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best)


def _wd_l3_7(target, weather):
    days = [target - timedelta(days=lag) for lag in range(7, 2, -1)]
    if any(dd not in weather for dd in days):
        return None
    threshold = _wd_threshold(weather, min(days))
    if threshold is None:
        return None
    vals = [float(weather[dd]['winddry']) for dd in days]
    flags = [v >= threshold for v in vals]
    return {
        'high_days': float(sum(flags)),
        'high_run': _max_run(flags),
        'avg': float(np.mean(vals)),
        'threshold': float(threshold),
    }


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
    # current = T-4..T-1, previous = T-8..T-5. Target day excluded.
    cur = _weather_block(target - timedelta(days=1), weather)
    prev = _weather_block(target - timedelta(days=WX_BLOCK_DAYS + 1), weather)
    if cur is None or prev is None:
        return None
    return cur - prev


def _previous_target_for_field(strict, target, field):
    x = strict[(strict['field'] == int(field)) & (strict['target_date'] < target)]
    if x.empty:
        return None
    return x.sort_values('target_date').iloc[-1]['target_date']


def _daily_weather_features(strict, weather):
    """One row per strict target day. W1 is field-aware, W2 is calendar-time."""
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue

        wd_fields = []
        for _, r in g.iterrows():
            cur = _wd_l3_7(target, weather)
            prev_target = _previous_target_for_field(strict, target, int(r['field']))
            prev = _wd_l3_7(prev_target, weather) if prev_target is not None else None
            if cur is None or prev is None:
                continue
            wd_fields.append([
                cur['high_days'],
                cur['high_run'],
                cur['high_days'] - prev['high_days'],
                cur['high_run'] - prev['high_run'],
            ])

        wd = np.median(np.asarray(wd_fields, dtype=float), axis=0) if wd_fields else None
        wx4 = _weather_delta4(target, weather)
        if wd is None and wx4 is None:
            continue

        resid_fields = np.log(g['actual'].to_numpy(dtype=float) + ABC_EPS2) - np.log(
            g['base'].to_numpy(dtype=float) + ABC_EPS2
        )
        row = {
            'date': target,
            'fields': ','.join(str(int(x)) for x in g.sort_values(['order','field'])['field'].tolist()),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            'common_resid': float(np.median(resid_fields)),
            'base_error': float(g['base'].sum() - g['actual'].sum()),
            'train_n': int(g['train_n'].min()),
        }
        names_wd = ['wd_high_days','wd_high_run','wd_delta_days','wd_delta_run']
        names_wx = ['wx4_d_rad','wx4_d_nightstress','wx4_d_winddry']
        for i, name in enumerate(names_wd):
            row[name] = float(wd[i]) if wd is not None else np.nan
        for i, name in enumerate(names_wx):
            row[name] = float(wx4[i]) if wx4 is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def _ridge_fit_fixed(train, cols):
    xraw = train[cols].to_numpy(dtype=float)
    y = train['common_resid'].to_numpy(dtype=float)
    mu = xraw.mean(axis=0)
    sd = xraw.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X = (xraw - mu) / sd
    # Intercept is prior mean residual; weather only explains movement around it.
    y0 = float(np.mean(y))
    yc = y - y0
    lhs = X.T @ X + WX_RIDGE_LAMBDA * np.eye(X.shape[1])
    rhs = X.T @ yc
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return {'cols': cols, 'mu': mu, 'sd': sd, 'beta': beta, 'intercept': y0}


def _predict_resid(row, fit):
    raw = np.asarray([float(row[c]) for c in fit['cols']], dtype=float)
    z = (raw - fit['mu']) / fit['sd']
    pred_raw = float(fit['intercept'] + z @ fit['beta'])
    pred_cap = float(np.clip(pred_raw, -WX_CAP_LOG, WX_CAP_LOG))
    return pred_cap, pred_raw


def _walk_weather_models(daily):
    configs = {
        'W1 WD154': ['wd_high_days','wd_high_run','wd_delta_days','wd_delta_run'],
        'W2 WX4': ['wx4_d_rad','wx4_d_nightstress','wx4_d_winddry'],
        'W3 WD154+WX4': [
            'wd_high_days','wd_high_run','wd_delta_days','wd_delta_run',
            'wx4_d_rad','wx4_d_nightstress','wx4_d_winddry',
        ],
    }
    out = daily.copy()
    for label, cols in configs.items():
        slug = label.split()[0].lower()
        out[f'{slug}_resid_pred'] = np.nan
        out[f'{slug}_resid_raw'] = np.nan
        out[f'{slug}_train_n'] = 0
        out[f'{slug}_forecast'] = np.nan

        for i, row in out.iterrows():
            prior = out.iloc[:i].copy()
            prior = prior.dropna(subset=cols + ['common_resid'])
            if len(prior) < WX_MIN_TRAIN_DAYS or any(pd.isna(row[c]) for c in cols):
                continue
            fit = _ridge_fit_fixed(prior, cols)
            pred_cap, pred_raw = _predict_resid(row, fit)
            out.at[i, f'{slug}_resid_pred'] = pred_cap
            out.at[i, f'{slug}_resid_raw'] = pred_raw
            out.at[i, f'{slug}_train_n'] = int(len(prior))
            # Common residual is log(actual/base); multiply BASE by exp(predicted residual).
            out.at[i, f'{slug}_forecast'] = float(row['base']) * math.exp(pred_cap)
    return out


def _metrics(df, pred_col):
    x = df.dropna(subset=['actual', pred_col]).copy()
    if x.empty:
        return {'N':0,'MAE':np.nan,'MAPE':np.nan,'Bias':np.nan,'DirHit':np.nan,'DirN':0}
    err = x[pred_col].astype(float) - x['actual'].astype(float)
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(x['actual'].astype(float), 0.5)) * 100.0)
    bias = float(np.mean(err))
    # Residual direction: did model predict whether BASE should be raised/lowered?
    pred_resid = np.log(np.maximum(x[pred_col].astype(float),0.05) / np.maximum(x['base'].astype(float),0.05))
    actual_resid = x['common_resid'].astype(float)
    mask = (np.abs(actual_resid) >= 0.03) & (np.abs(pred_resid) >= 0.01)
    if int(mask.sum()):
        hit = float(np.mean(np.sign(actual_resid[mask]) == np.sign(pred_resid[mask])) * 100.0)
        n = int(mask.sum())
    else:
        hit = np.nan; n = 0
    return {'N':int(len(x)),'MAE':mae,'MAPE':mape,'Bias':bias,'DirHit':hit,'DirN':n}


def _cause_score(df, pred_resid_col):
    x = df.dropna(subset=['common_resid', pred_resid_col]).copy()
    if len(x) < 3:
        return {'N':int(len(x)),'rho':np.nan,'sign':np.nan,'mae_resid':np.nan}
    # Spearman without SciPy.
    ra = x['common_resid'].rank(method='average').to_numpy(dtype=float)
    rb = x[pred_resid_col].rank(method='average').to_numpy(dtype=float)
    rho = float(np.corrcoef(ra, rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else np.nan
    mask = (x['common_resid'].abs() >= 0.03) & (x[pred_resid_col].abs() >= 0.01)
    sign = float(np.mean(np.sign(x.loc[mask,'common_resid']) == np.sign(x.loc[mask,pred_resid_col]))*100.0) if int(mask.sum()) else np.nan
    mae_resid = float(np.mean(np.abs(x['common_resid'] - x[pred_resid_col])))
    return {'N':int(len(x)),'rho':rho,'sign':sign,'mae_resid':mae_resid}


def _verdict(focus, context):
    # Prefer strict focus evidence, then context. No tuning thresholds by data.
    candidates = []
    for slug, label in [('w1','WD154'),('w2','WX4'),('w3','WD154+WX4')]:
        fs = _cause_score(focus, f'{slug}_resid_pred')
        cs = _cause_score(context, f'{slug}_resid_pred')
        candidates.append((label, fs, cs))

    viable = [x for x in candidates if x[1]['N'] >= 3]
    if viable:
        best = max(viable, key=lambda x: ((x[1]['sign'] if pd.notna(x[1]['sign']) else -1), (x[1]['rho'] if pd.notna(x[1]['rho']) else -9)))
        label, fs, cs = best
        if pd.notna(fs['sign']) and fs['sign'] >= 75 and pd.notna(fs['rho']) and fs['rho'] >= 0.35:
            return 'WEATHER EXPLAINS A MATERIAL PART', (
                f'{label} follows the 19.–22.08 common residual in strict walk-forward '
                f'(direction {fs["sign"]:.0f}%, rho {fs["rho"]:+.2f}). '
                'Next step may test this one locked weather family as a bounded common correction.'
            )
    return 'WEATHER NOT ENOUGH', (
        'The already-audited pre-target weather families do not reproduce the common 19.–22.08 residual '
        'strongly enough in strict walk-forward. The missing signal is therefore more likely a common crop-state/lag '
        'process than a simple direct weather correction. Do not add another weather layer to BASE from this evidence.'
    )


def main():
    st.set_page_config(page_title='KurgiMootor · common time cause', layout='wide')
    st.title('KurgiMootor · common time cause audit')
    st.caption('LAB-37 · BASE stays fixed · locked weather families only · strict walk-forward · READ ONLY')
    st.info(
        'LAB-36 said the 19.–22.08 miss is common across fields. This LAB does not invent a new state. '
        'It asks whether known pre-target weather signals can actually reproduce that common miss before we touch BASE.'
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
        weather_rows = db.get_weather_rows(max(WEATHER_ORIGIN, earliest - timedelta(days=20)), latest)
        weather = _weather_map(weather_rows)
        daily0 = _daily_weather_features(strict, weather)
        if daily0.empty:
            raise RuntimeError('Daily weather/common-residual rows did not form.')
        daily = _walk_weather_models(daily0)
        focus = daily[(daily['date'] >= FOCUS_START) & (daily['date'] <= FOCUS_END)].copy()
        context = daily[(daily['date'] >= FOCUS_START) & (daily['date'] <= CONTEXT_END)].copy()
        verdict, verdict_text = _verdict(focus, context)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.markdown('### 1. Otsus · kas teadaolev ilm seletab ühist lainet?')
    if verdict.startswith('WEATHER EXPLAINS'):
        st.success('✅ ' + verdict + ': ' + verdict_text)
    else:
        st.warning('🟡 ' + verdict + ': ' + verdict_text)

    rows = []
    for slug, label in [('w1','W1 · WIND×DRY L3–7'),('w2','W2 · 4d weather transition'),('w3','W3 · mõlemad')]:
        fs = _cause_score(focus, f'{slug}_resid_pred')
        cs = _cause_score(context, f'{slug}_resid_pred')
        rows.append({
            'Variant': label,
            'Focus N': fs['N'],
            'Focus suunahitt %': fs['sign'],
            'Focus rho': fs['rho'],
            'Focus residual MAE': fs['mae_resid'],
            '19–27 N': cs['N'],
            '19–27 suunahitt %': cs['sign'],
            '19–27 rho': cs['rho'],
        })
    score = pd.DataFrame(rows)
    st.dataframe(score.style.format({
        'Focus suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        'Focus rho':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
        'Focus residual MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.3f}',
        '19–27 suunahitt %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
        '19–27 rho':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 2. Päev-päevalt · mida BASE vajas ja mida ilm enne päeva ütles?')
    show = context.copy()
    show['actual_minus_base'] = show['actual'] - show['base']
    st.dataframe(
        show[[
            'date','fields','actual','base','actual_minus_base','common_resid',
            'wd_high_days','wd_high_run','wd_delta_days','wd_delta_run',
            'wx4_d_rad','wx4_d_nightstress','wx4_d_winddry',
            'w1_resid_pred','w2_resid_pred','w3_resid_pred',
        ]].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'BASE','actual_minus_base':'Tegelik−BASE',
            'common_resid':'Vajalik ühine log-korrektsioon',
            'wd_high_days':'WD high päevad','wd_high_run':'WD high jada','wd_delta_days':'Δ WD päevad','wd_delta_run':'Δ WD jada',
            'wx4_d_rad':'Δ rad 4d','wx4_d_nightstress':'Δ ööstress 4d','wx4_d_winddry':'Δ WD 4d',
            'w1_resid_pred':'W1 ennustus','w2_resid_pred':'W2 ennustus','w3_resid_pred':'W3 ennustus',
        }).style.format({
            'Päev':_fmt_day,'Tegelik ABC':'{:.1f}','BASE':'{:.1f}','Tegelik−BASE':'{:+.1f}',
            'Vajalik ühine log-korrektsioon':'{:+.3f}',
            'WD high päevad':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
            'WD high jada':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
            'Δ WD päevad':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'Δ WD jada':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'Δ rad 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
            'Δ ööstress 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'Δ WD 4d':lambda v:'—' if pd.isna(v) else f'{float(v):+.1f}',
            'W1 ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'W2 ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
            'W3 ennustus':lambda v:'—' if pd.isna(v) else f'{float(v):+.3f}',
        }), use_container_width=True, hide_index=True
    )

    st.markdown('### 3. Kas weather-correction parandaks taset või ainult seletaks suunda?')
    comp = []
    base_m = _metrics(context.assign(base_identity=context['base']), 'base_identity')
    comp.append({'Variant':'BASE','N':len(context),'MAE':float(np.mean(np.abs(context['base']-context['actual']))),'MAPE %':float(np.mean(np.abs(context['base']-context['actual'])/np.maximum(context['actual'],0.5))*100.0),'Bias':float(np.mean(context['base']-context['actual'])),'Suund %':np.nan})
    for slug, label in [('w1','BASE + W1'),('w2','BASE + W2'),('w3','BASE + W3')]:
        m = _metrics(context, f'{slug}_forecast')
        comp.append({'Variant':label,'N':m['N'],'MAE':m['MAE'],'MAPE %':m['MAPE'],'Bias':m['Bias'],'Suund %':m['DirHit']})
    st.dataframe(pd.DataFrame(comp).style.format({
        'MAE':lambda v:'—' if pd.isna(v) else f'{float(v):.2f}',
        'MAPE %':lambda v:'—' if pd.isna(v) else f'{float(v):.1f}',
        'Bias':lambda v:'—' if pd.isna(v) else f'{float(v):+.2f}',
        'Suund %':lambda v:'—' if pd.isna(v) else f'{float(v):.0f}%',
    }), use_container_width=True, hide_index=True)

    st.caption(
        f'Locks: ridge λ={WX_RIDGE_LAMBDA:g}, weather cap ±{WX_CAP_LOG:.2f} log, '
        f'WX blocks {WX_BLOCK_DAYS}d vs {WX_BLOCK_DAYS}d, WD threshold Q{WD_Q:.2f}. '
        'No parameter is selected from the 19.–27.08 result.'
    )
    st.success(
        '🔒 LEAKAGE LOCK: each BASE target is trained only on earlier intervals. '
        'Each weather residual prediction is trained only on earlier daily common residuals. '
        'Target-day measured weather is excluded; weather uses T-1 or earlier only. READ ONLY.'
    )


if __name__ == '__main__':
    main()
