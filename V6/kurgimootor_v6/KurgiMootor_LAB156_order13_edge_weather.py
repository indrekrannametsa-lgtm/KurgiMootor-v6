from __future__ import annotations

"""
KurgiMootor · edge_weather-38
=============================

WIND×DRY DIRECTION SENSOR AUDIT · STRICT CAUSAL TIMING · READ ONLY

One question only:
Does the already-locked change in WIND×DRY HIGH L3–7 regime predict the
DIRECTION of BASE's common daily residual across all available strict-OOS days?

This LAB deliberately does NOT fit a weather regression and does NOT alter BASE.
It tests the raw locked signal itself:

PRIMARY sensor
- Δ WD HIGH L3–7 days versus the same field's previous harvest-cycle L3–7 regime
- strengthening (>0) predicts BASE too low / positive common residual
- weakening (<0) predicts BASE too high / negative common residual
- zero change abstains

SECONDARY diagnostics
- Δ WD HIGH L3–7 run length, same sign rule
- WD HIGH L3–7 level association with the common residual (rho only)

Locks
-----
- every BASE target row is trained only on intervals strictly before target
- target actual is used only after prediction for scoring
- WIND×DRY threshold for each L3–7 window uses only checked measured weather
  strictly before that window
- previous-cycle weather window is the target field's actual previous harvest day
  (the strict interval start_date), never a predicted/recursive anchor
- target-day weather is never used
- no regression, coefficient, cap, lag search, threshold search or correction size
- primary direction rule is fixed before looking at the score
- 25.08 no-harvest remains absent, never zero yield
- READ ONLY: only db.get_harvest_history and db.get_weather_rows are called
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
# WIND×DRY DIRECTION SENSOR AUDIT · raw locked signal only
# =====================================================================

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 27)
ABC_EPS2 = ABC_EPS

WD_Q = 0.75
WD_MIN_THRESHOLD_DAYS = 10
WEATHER_ORIGIN = date(2026, 7, 1)
RESID_DEADBAND = 0.03
MIN_STRONG_N = 8
MIN_PROMISING_N = 6


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
        wind = _f(row.get('wind_avg_ms'))
        rh = _f(row.get('humidity_avg_pct'))
        if wind is None or rh is None:
            continue
        out[dd] = {
            'wind': float(wind),
            'rh': float(rh),
            'winddry': float(wind) * (100.0 - float(rh)),
        }
    return out


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
    if target is None:
        return None
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
        'window_start': days[0],
        'window_end': days[-1],
    }


def _daily_sensor_rows(strict, weather):
    rows = []
    for target, g in strict.groupby('target_date', sort=True):
        if len(g) < 2:
            continue

        field_features = []
        for _, r in g.iterrows():
            cur = _wd_l3_7(target, weather)
            # start_date is the actual previous same-field harvest day for this interval.
            prev_target = _d(r.get('start_date'))
            prev = _wd_l3_7(prev_target, weather)
            if cur is None or prev is None:
                continue
            field_features.append({
                'field': int(r['field']),
                'high_days': float(cur['high_days']),
                'high_run': float(cur['high_run']),
                'delta_days': float(cur['high_days'] - prev['high_days']),
                'delta_run': float(cur['high_run'] - prev['high_run']),
            })

        # Keep only complete daily sensor rows: every scored field must have the
        # current and previous-cycle weather regime available.
        if len(field_features) != len(g):
            continue

        ff = pd.DataFrame(field_features)
        resid_fields = np.log(g['actual'].to_numpy(dtype=float) + ABC_EPS2) - np.log(
            g['base'].to_numpy(dtype=float) + ABC_EPS2
        )
        common_resid = float(np.median(resid_fields))
        delta_days = float(np.median(ff['delta_days'].to_numpy(dtype=float)))
        delta_run = float(np.median(ff['delta_run'].to_numpy(dtype=float)))
        high_days = float(np.median(ff['high_days'].to_numpy(dtype=float)))
        high_run = float(np.median(ff['high_run'].to_numpy(dtype=float)))

        if common_resid > RESID_DEADBAND:
            truth_dir = 1
        elif common_resid < -RESID_DEADBAND:
            truth_dir = -1
        else:
            truth_dir = 0

        def sensor_dir(v):
            if v > 0:
                return 1
            if v < 0:
                return -1
            return 0

        rows.append({
            'date': target,
            'fields': ','.join(str(int(x)) for x in g.sort_values(['order','field'])['field'].tolist()),
            'n_fields': int(len(g)),
            'actual': float(g['actual'].sum()),
            'base': float(g['base'].sum()),
            'actual_minus_base': float(g['actual'].sum() - g['base'].sum()),
            'common_resid': common_resid,
            'truth_dir': int(truth_dir),
            'wd_high_days': high_days,
            'wd_high_run': high_run,
            'wd_delta_days': delta_days,
            'wd_delta_run': delta_run,
            'sensor_days_dir': int(sensor_dir(delta_days)),
            'sensor_run_dir': int(sensor_dir(delta_run)),
            'train_n': int(g['train_n'].min()),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    out['days_hit'] = np.where(
        (out['truth_dir'] != 0) & (out['sensor_days_dir'] != 0),
        out['truth_dir'] == out['sensor_days_dir'],
        np.nan,
    )
    out['run_hit'] = np.where(
        (out['truth_dir'] != 0) & (out['sensor_run_dir'] != 0),
        out['truth_dir'] == out['sensor_run_dir'],
        np.nan,
    )
    return out


def _spearman_no_scipy(a, b):
    x = pd.DataFrame({'a': a, 'b': b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return np.nan
    ra = x['a'].rank(method='average').to_numpy(dtype=float)
    rb = x['b'].rank(method='average').to_numpy(dtype=float)
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _binom_tail_half(k, n):
    if n <= 0:
        return np.nan
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n))


def _direction_metrics(df, sensor_col):
    x = df.copy()
    truth_eligible = x[x['truth_dir'] != 0].copy()
    scored = truth_eligible[truth_eligible[sensor_col] != 0].copy()
    hits = int((scored['truth_dir'] == scored[sensor_col]).sum()) if len(scored) else 0
    n = int(len(scored))
    return {
        'days': int(len(x)),
        'truth_n': int(len(truth_eligible)),
        'n': n,
        'coverage': float(100.0 * n / len(truth_eligible)) if len(truth_eligible) else np.nan,
        'hit': float(100.0 * hits / n) if n else np.nan,
        'hits': hits,
        'p_tail': _binom_tail_half(hits, n),
    }


def _association_metrics(df, value_col):
    x = df.dropna(subset=[value_col, 'common_resid']).copy()
    return {
        'N': int(len(x)),
        'rho': _spearman_no_scipy(x[value_col], x['common_resid']),
    }


def _split_halves(df):
    x = df.sort_values('date').reset_index(drop=True)
    if len(x) < 4:
        return x.iloc[:0].copy(), x.iloc[:0].copy()
    cut = len(x) // 2
    return x.iloc[:cut].copy(), x.iloc[cut:].copy()


def _regime_table(df):
    x = df.copy()
    x['regime'] = np.where(
        x['wd_delta_days'] > 0,
        'Tugevneb (+)',
        np.where(x['wd_delta_days'] < 0, 'Nõrgeneb (−)', 'Muutuseta (0)')
    )
    rows = []
    for label in ['Nõrgeneb (−)', 'Muutuseta (0)', 'Tugevneb (+)']:
        g = x[x['regime'] == label]
        if g.empty:
            rows.append({'WD režiim': label, 'N': 0, 'Mediaan log-viga': np.nan, 'Keskmine log-viga': np.nan,
                         'Tegelik−BASE / päev': np.nan, 'BASE liiga madal %': np.nan})
            continue
        rows.append({
            'WD režiim': label,
            'N': int(len(g)),
            'Mediaan log-viga': float(np.median(g['common_resid'])),
            'Keskmine log-viga': float(np.mean(g['common_resid'])),
            'Tegelik−BASE / päev': float(np.mean(g['actual_minus_base'])),
            'BASE liiga madal %': float(np.mean(g['common_resid'] > RESID_DEADBAND) * 100.0),
        })
    return pd.DataFrame(rows)


def _verdict(daily):
    primary = _direction_metrics(daily, 'sensor_days_dir')
    assoc = _association_metrics(daily, 'wd_delta_days')
    first, second = _split_halves(daily)
    m1 = _direction_metrics(first, 'sensor_days_dir') if not first.empty else None
    m2 = _direction_metrics(second, 'sensor_days_dir') if not second.empty else None
    regimes = _regime_table(daily)
    med = dict(zip(regimes['WD režiim'], regimes['Mediaan log-viga']))
    weak = med.get('Nõrgeneb (−)', np.nan)
    strong = med.get('Tugevneb (+)', np.nan)
    ordered = pd.notna(weak) and pd.notna(strong) and strong > weak
    halves_ok = True
    for m in (m1, m2):
        if m is not None and m['n'] >= 2 and pd.notna(m['hit']):
            halves_ok = halves_ok and (m['hit'] >= 60.0)

    rho = assoc['rho']
    if (
        primary['n'] >= MIN_STRONG_N
        and pd.notna(primary['hit']) and primary['hit'] >= 70.0
        and pd.notna(primary['coverage']) and primary['coverage'] >= 50.0
        and pd.notna(rho) and rho >= 0.45
        and ordered and halves_ok
    ):
        return 'STRONG DIRECTION SENSOR', (
            'Δ WD HIGH L3–7 days carries a stable sign relationship with the common BASE residual. '
            'This is evidence for a direction sensor, not yet evidence for a box-level correction.'
        )
    if (
        primary['n'] >= MIN_PROMISING_N
        and pd.notna(primary['hit']) and primary['hit'] >= 65.0
        and pd.notna(rho) and rho >= 0.30
        and ordered
    ):
        return 'PROMISING, NOT PROVEN', (
            'The raw Δ WD HIGH L3–7 sign behaves in the expected direction often enough to keep, '
            'but the sample/stability is not strong enough for production use. Preserve a future holdout.'
        )
    return 'NOT STABLE ENOUGH', (
        'The raw locked WIND×DRY regime change does not predict the common BASE residual direction '
        'consistently enough across the available strict-OOS period. Do not turn it into a production correction.'
    )


def _dir_label(v):
    if pd.isna(v):
        return '—'
    v = int(v)
    return '↑ BASE liiga madal' if v > 0 else ('↓ BASE liiga kõrge' if v < 0 else '0 / abstain')


def main():
    st.set_page_config(page_title='KurgiMootor · WD direction sensor', layout='wide')
    st.title('KurgiMootor · WIND×DRY direction sensor audit')
    st.caption('LAB-38 · BASE fixed · raw Δ WD HIGH L3–7 only · no regression · strict causal timing · READ ONLY')
    st.info(
        'LAB-36 found a common time signal. LAB-37 showed the fitted weather correction was not reliable, '
        'but the raw Δ WD HIGH L3–7 sequence appeared to turn with the BASE residual. '
        'This LAB tests that raw sign across every available strict-OOS day without fitting a coefficient.'
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
        weather_rows = db.get_weather_rows(max(WEATHER_ORIGIN, earliest - timedelta(days=40)), latest)
        weather = _weather_map(weather_rows)
        daily = _daily_sensor_rows(strict, weather)
        if daily.empty:
            raise RuntimeError('No complete strict-OOS WD direction rows formed.')
        verdict, verdict_text = _verdict(daily)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.markdown('### 1. Otsus · kas Δ WIND×DRY on päris pöördeandur?')
    if verdict.startswith('STRONG'):
        st.success('✅ ' + verdict + ': ' + verdict_text)
    elif verdict.startswith('PROMISING'):
        st.warning('🟡 ' + verdict + ': ' + verdict_text)
    else:
        st.error('⛔ ' + verdict + ': ' + verdict_text)

    primary = _direction_metrics(daily, 'sensor_days_dir')
    secondary = _direction_metrics(daily, 'sensor_run_dir')
    assoc_days = _association_metrics(daily, 'wd_delta_days')
    assoc_run = _association_metrics(daily, 'wd_delta_run')
    assoc_level = _association_metrics(daily, 'wd_high_days')
    score = pd.DataFrame([
        {
            'Kontroll': 'PRIMARY · Δ WD high päevad sign',
            'Päevi': primary['days'], 'Scored N': primary['n'], 'Coverage %': primary['coverage'],
            'Suunahitt %': primary['hit'], 'rho vs ühine viga': assoc_days['rho'], 'Binom p≥hits': primary['p_tail'],
        },
        {
            'Kontroll': 'SECONDARY · Δ WD high jada sign',
            'Päevi': secondary['days'], 'Scored N': secondary['n'], 'Coverage %': secondary['coverage'],
            'Suunahitt %': secondary['hit'], 'rho vs ühine viga': assoc_run['rho'], 'Binom p≥hits': secondary['p_tail'],
        },
        {
            'Kontroll': 'LEVEL only · WD high päevad',
            'Päevi': int(len(daily)), 'Scored N': assoc_level['N'], 'Coverage %': np.nan,
            'Suunahitt %': np.nan, 'rho vs ühine viga': assoc_level['rho'], 'Binom p≥hits': np.nan,
        },
    ])
    st.dataframe(score.style.format({
        'Coverage %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Suunahitt %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'rho vs ühine viga': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
        'Binom p≥hits': lambda v: '—' if pd.isna(v) else f'{float(v):.3f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 2. Stabiilsus · esimene pool vs teine pool')
    first, second = _split_halves(daily)
    half_rows = []
    for label, part in [('I pool', first), ('II pool', second)]:
        if part.empty:
            continue
        m = _direction_metrics(part, 'sensor_days_dir')
        a = _association_metrics(part, 'wd_delta_days')
        half_rows.append({
            'Periood': label,
            'Kuupäevad': f"{_fmt_day(part['date'].min())}–{_fmt_day(part['date'].max())}",
            'Päevi': len(part),
            'Scored N': m['n'],
            'Coverage %': m['coverage'],
            'Suunahitt %': m['hit'],
            'rho': a['rho'],
        })
    st.dataframe(pd.DataFrame(half_rows).style.format({
        'Coverage %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'Suunahitt %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
        'rho': lambda v: '—' if pd.isna(v) else f'{float(v):+.2f}',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 3. Režiimiefekt · kas nõrgenemine ja tugevnemine annavad eri märgiga vea?')
    regimes = _regime_table(daily)
    st.dataframe(regimes.style.format({
        'Mediaan log-viga': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Keskmine log-viga': lambda v: '—' if pd.isna(v) else f'{float(v):+.3f}',
        'Tegelik−BASE / päev': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}',
        'BASE liiga madal %': lambda v: '—' if pd.isna(v) else f'{float(v):.0f}%',
    }), use_container_width=True, hide_index=True)

    st.markdown('### 4. 19.–27.08 · visuaalne kontroll')
    focus = daily[(daily['date'] >= FOCUS_START) & (daily['date'] <= FOCUS_END)].copy()
    show = focus if not focus.empty else daily.tail(10).copy()
    show['Tegelik suund'] = show['truth_dir'].map(_dir_label)
    show['WD sensori suund'] = show['sensor_days_dir'].map(_dir_label)
    show['Hitt'] = show['days_hit'].map(lambda v: '—' if pd.isna(v) else ('✅' if bool(v) else '❌'))
    st.dataframe(
        show[[
            'date','fields','actual','base','actual_minus_base','common_resid',
            'wd_high_days','wd_high_run','wd_delta_days','wd_delta_run',
            'Tegelik suund','WD sensori suund','Hitt',
        ]].rename(columns={
            'date':'Päev','fields':'Põllud','actual':'Tegelik ABC','base':'BASE','actual_minus_base':'Tegelik−BASE',
            'common_resid':'Ühine log-viga','wd_high_days':'WD high päevad','wd_high_run':'WD high jada',
            'wd_delta_days':'Δ WD päevad','wd_delta_run':'Δ WD jada',
        }).style.format({
            'Päev': _fmt_day,
            'Tegelik ABC': '{:.1f}', 'BASE': '{:.1f}', 'Tegelik−BASE': '{:+.1f}',
            'Ühine log-viga': '{:+.3f}',
            'WD high päevad': lambda v: '—' if pd.isna(v) else f'{float(v):.1f}',
            'WD high jada': lambda v: '—' if pd.isna(v) else f'{float(v):.1f}',
            'Δ WD päevad': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}',
            'Δ WD jada': lambda v: '—' if pd.isna(v) else f'{float(v):+.1f}',
        }), use_container_width=True, hide_index=True,
    )

    with st.expander('Kõik strict-OOS päevad'):
        full = daily.copy()
        full['Tegelik suund'] = full['truth_dir'].map(_dir_label)
        full['WD sensori suund'] = full['sensor_days_dir'].map(_dir_label)
        full['Hitt'] = full['days_hit'].map(lambda v: '—' if pd.isna(v) else ('✅' if bool(v) else '❌'))
        st.dataframe(full[[
            'date','fields','n_fields','actual','base','actual_minus_base','common_resid',
            'wd_high_days','wd_delta_days','wd_high_run','wd_delta_run','Tegelik suund','WD sensori suund','Hitt','train_n'
        ]].rename(columns={
            'date':'Päev','fields':'Põllud','n_fields':'N põldu','actual':'Tegelik ABC','base':'BASE',
            'actual_minus_base':'Tegelik−BASE','common_resid':'Ühine log-viga','wd_high_days':'WD high päevad',
            'wd_delta_days':'Δ WD päevad','wd_high_run':'WD high jada','wd_delta_run':'Δ WD jada','train_n':'BASE train N'
        }), use_container_width=True, hide_index=True)

    st.caption(
        f'PRIMARY rule is fixed: sign(Δ WD high days). Residual deadband ±{RESID_DEADBAND:.2f} log; '
        f'WD threshold Q{WD_Q:.2f}; no coefficient, no cap, no lag/window search. '
        'A zero ΔWD day abstains instead of being forced into a direction.'
    )
    st.success(
        '🔒 LEAKAGE LOCK: BASE uses only earlier intervals. Current and previous-cycle WIND×DRY windows use '
        'checked measured pre-target weather only. Actual target yield is used only to score the direction after prediction. '
        'READ ONLY. No production forecast is changed.'
    )


if __name__ == '__main__':
    main()
