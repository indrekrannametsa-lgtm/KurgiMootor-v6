from __future__ import annotations

"""
KurgiMootor · edge_weather-36
=============================

BASE ERROR ANATOMY · STRICT WALK-FORWARD · READ ONLY

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
# BASE error anatomy · no new model, no new coefficient
# =====================================================================

FOCUS_START = date(2026, 8, 19)
FOCUS_END = date(2026, 8, 22)
CONTEXT_END = date(2026, 8, 27)
RESID_DEADBAND_LOG = 0.03   # ~3%; only for direction/sign diagnostics


def _fmt_day(v):
    if isinstance(v, pd.Timestamp):
        v = v.date()
    if isinstance(v, date):
        return v.strftime("%d.%m")
    return str(v)


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without SciPy dependency.

    pandas delegates method="spearman" to scipy.stats, but production/LAB
    requirements intentionally do not include SciPy. Spearman is simply the
    Pearson correlation of average ranks, so compute that directly.
    """
    x = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(x) < 3 or x["a"].nunique() < 2 or x["b"].nunique() < 2:
        return np.nan
    ra = x["a"].rank(method="average").to_numpy(dtype=float)
    rb = x["b"].rank(method="average").to_numpy(dtype=float)
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _sign_hit(a: pd.Series, b: pd.Series, deadband: float = RESID_DEADBAND_LOG) -> Tuple[int, float]:
    x = pd.DataFrame({"a": a, "b": b}).dropna()
    x = x[(x["a"].abs() >= deadband) & (x["b"].abs() >= deadband)]
    if x.empty:
        return 0, np.nan
    hit = np.sign(x["a"].to_numpy(float)) == np.sign(x["b"].to_numpy(float))
    return int(len(x)), float(np.mean(hit) * 100.0)


def _add_same_field_history(strict: pd.DataFrame) -> pd.DataFrame:
    """
    Add only PRIOR strict-OOS information from the same field.

    Positive residual means actual > BASE (BASE was too low).
    This is diagnostic only; none of these columns changes a forecast.
    """
    df = strict.sort_values(["target_date", "order", "field"]).copy().reset_index(drop=True)
    df["resid_log"] = np.log(df["actual"].astype(float) + ABC_EPS) - np.log(df["base"].astype(float) + ABC_EPS)
    df["base_error"] = df["base"].astype(float) - df["actual"].astype(float)
    df["ape_pct"] = df["base_error"].abs() / np.maximum(df["actual"].astype(float), 0.5) * 100.0

    prev_resid = []
    prev_growth = []
    prev_date = []
    prev_n = []
    prior_median = []

    for _, row in df.iterrows():
        prior = df[(df["field"] == int(row["field"])) & (df["target_date"] < row["target_date"])].copy()
        prev_n.append(int(len(prior)))
        if prior.empty:
            prev_resid.append(np.nan)
            prev_growth.append(np.nan)
            prev_date.append(None)
            prior_median.append(np.nan)
        else:
            last = prior.sort_values("target_date").iloc[-1]
            prev_resid.append(float(last["resid_log"]))
            prev_growth.append(float(last["growth"]))
            prev_date.append(last["target_date"])
            prior_median.append(float(prior["resid_log"].median()))

    df["field_prev_resid"] = prev_resid
    df["field_prev_growth"] = prev_growth
    df["field_prev_date"] = prev_date
    df["field_prior_n"] = prev_n
    df["field_prior_median"] = prior_median
    df["resid_delta_vs_prev"] = df["resid_log"] - df["field_prev_resid"]
    df["growth_delta_vs_prev"] = df["growth"] - df["field_prev_growth"]
    return df


def _sign_agreement(vals: Sequence[float], deadband: float = RESID_DEADBAND_LOG) -> Tuple[int, float, str]:
    x = np.asarray([float(v) for v in vals if pd.notna(v)], dtype=float)
    x = x[np.abs(x) >= deadband]
    if len(x) == 0:
        return 0, np.nan, "—"
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    dominant = max(pos, neg)
    label = "actual > BASE" if pos > neg else "actual < BASE" if neg > pos else "mixed"
    return int(len(x)), float(dominant / len(x) * 100.0), label


def _day_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dd, g in df.groupby("target_date", sort=True):
        n_sign, agree, label = _sign_agreement(g["resid_log"].tolist())
        n_delta, delta_agree, delta_label = _sign_agreement(g["resid_delta_vs_prev"].tolist())
        med = float(g["resid_log"].median())
        mad = float(np.median(np.abs(g["resid_log"].to_numpy(float) - med)))
        prev_med = float(g["field_prev_resid"].median()) if g["field_prev_resid"].notna().any() else np.nan
        delta_med = float(g["resid_delta_vs_prev"].median()) if g["resid_delta_vs_prev"].notna().any() else np.nan
        growth_delta_med = float(g["growth_delta_vs_prev"].median()) if g["growth_delta_vs_prev"].notna().any() else np.nan
        rows.append({
            "date": dd,
            "fields": ",".join(str(int(x)) for x in g.sort_values(["order", "field"])["field"].tolist()),
            "n_fields": int(len(g)),
            "actual": float(g["actual"].sum()),
            "base": float(g["base"].sum()),
            "base_error": float(g["base"].sum() - g["actual"].sum()),
            "median_resid_log": med,
            "within_day_mad": mad,
            "sign_n": n_sign,
            "sign_agreement_pct": agree,
            "dominant_side": label,
            "median_growth": float(g["growth"].median()),
            "field_prev_median": prev_med,
            "median_resid_delta_vs_prev": delta_med,
            "delta_sign_n": n_delta,
            "delta_sign_agreement_pct": delta_agree,
            "delta_dominant_side": delta_label,
            "median_growth_delta_vs_prev": growth_delta_med,
            "train_n": int(g["train_n"].min()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _diagnostic_scorecards(strict: pd.DataFrame, focus: pd.DataFrame, day_focus: pd.DataFrame) -> Dict[str, Any]:
    # 1) Does a field's previous strict-OOS error persist into its next cycle?
    field_n, field_sign = _sign_hit(strict["resid_log"], strict["field_prev_resid"])
    field_rho = _safe_spearman(strict["resid_log"], strict["field_prev_resid"])

    focus_field_n, focus_field_sign = _sign_hit(focus["resid_log"], focus["field_prev_resid"])
    focus_field_rho = _safe_spearman(focus["resid_log"], focus["field_prev_resid"])

    # 2) After removing same-field last-cycle error, does changing interval explain changing error?
    delta = strict[["resid_delta_vs_prev", "growth_delta_vs_prev"]].dropna()
    interval_rho = _safe_spearman(delta["resid_delta_vs_prev"], delta["growth_delta_vs_prev"])
    interval_n, interval_sign = _sign_hit(delta["resid_delta_vs_prev"], delta["growth_delta_vs_prev"])

    fdelta = focus[["resid_delta_vs_prev", "growth_delta_vs_prev"]].dropna()
    focus_interval_rho = _safe_spearman(fdelta["resid_delta_vs_prev"], fdelta["growth_delta_vs_prev"])
    focus_interval_n, focus_interval_sign = _sign_hit(fdelta["resid_delta_vs_prev"], fdelta["growth_delta_vs_prev"])

    # 3) Are field errors coherent within each focus day?
    coherence = day_focus["sign_agreement_pct"].dropna()
    delta_coherence = day_focus["delta_sign_agreement_pct"].dropna()
    mean_coherence = float(coherence.mean()) if len(coherence) else np.nan
    mean_delta_coherence = float(delta_coherence.mean()) if len(delta_coherence) else np.nan

    # 4) Does prior field mix even get the DAILY residual direction right?
    day_field_n, day_field_sign = _sign_hit(day_focus["median_resid_log"], day_focus["field_prev_median"])
    day_interval_n, day_interval_sign = _sign_hit(day_focus["median_resid_delta_vs_prev"], day_focus["median_growth_delta_vs_prev"])

    return {
        "field_n": field_n,
        "field_sign": field_sign,
        "field_rho": field_rho,
        "focus_field_n": focus_field_n,
        "focus_field_sign": focus_field_sign,
        "focus_field_rho": focus_field_rho,
        "interval_n": interval_n,
        "interval_sign": interval_sign,
        "interval_rho": interval_rho,
        "focus_interval_n": focus_interval_n,
        "focus_interval_sign": focus_interval_sign,
        "focus_interval_rho": focus_interval_rho,
        "mean_coherence": mean_coherence,
        "mean_delta_coherence": mean_delta_coherence,
        "day_field_n": day_field_n,
        "day_field_sign": day_field_sign,
        "day_interval_n": day_interval_n,
        "day_interval_sign": day_interval_sign,
    }


def _verdict(s: Dict[str, Any]) -> Tuple[str, str]:
    """
    Fixed descriptive thresholds, written before seeing this run.
    They are not a model-selection search.
    """
    coh = s.get("mean_coherence")
    dcoh = s.get("mean_delta_coherence")
    field_sign = s.get("focus_field_sign")
    field_rho = s.get("focus_field_rho")
    int_rho = s.get("focus_interval_rho")

    common_strong = pd.notna(coh) and coh >= 75.0 and pd.notna(dcoh) and dcoh >= 67.0
    field_strong = (
        (pd.notna(field_sign) and field_sign >= 70.0)
        or (pd.notna(field_rho) and field_rho >= 0.50)
    )
    interval_strong = pd.notna(int_rho) and abs(float(int_rho)) >= 0.50

    if common_strong and not field_strong and not interval_strong:
        return (
            "COMMON TIME SIGNAL",
            "19.–22.08 viga liigub põldude vahel samas suunas isegi pärast sama põllu eelmise tsükli vea eemaldamist; "
            "sama põllu varasem viga ja intervallimuutus ei seleta seda piisavalt. Järgmine samm peab otsima ajas liikuvat ühist põhjust, mitte põlluankrut."
        )
    if field_strong and not interval_strong:
        return (
            "FIELD / FIELD-MIX SIGNAL",
            "Sama põllu varasem BASE-viga kandub piisavalt järjekindlalt edasi. Enne uue ajasignaali otsimist tuleb BASE põllutasemed üle auditeerida."
        )
    if interval_strong and not field_strong:
        return (
            "INTERVAL RESPONSE SIGNAL",
            "Sama põllu tsüklite vahel muutuv BASE-viga liigub tugevalt koos intervallimuutusega. Järgmine samm on BASE intervallivastuse audit, mitte uus state."
        )
    if common_strong:
        return (
            "MIXED, BUT COMMON-TIME DOMINANT",
            "Ühine ajakomponent on tugev, kuid ka põllu või intervalli jälg pole piisavalt nõrk, et neid veel välistada. Uut parandust ei lisata; järgmine audit peab ühise ajasignaali põhjust eristama."
        )
    return (
        "NO CLEAN SEPARATION",
        "See test ei erista ühist ajasignaali, põllu-mixi ja intervalli piisavalt puhtalt. Sellisel juhul ei ole aus BASE'ile uut kihti lisada."
    )


def main():
    st.set_page_config(page_title="KurgiMootor · BASE error anatomy", layout="wide")
    st.title("KurgiMootor · BASE error anatomy")
    st.caption("edge_weather-36 · üks küsimus: miks BASE 19.–22.08 laine maha magas? · READ ONLY")
    st.info(
        "See LAB EI ehita uut mudelit. BASE jääb täpselt samaks nagu -35. "
        "Vaatame BASE strict-OOS viga põllu kaupa ja võrdleme kolme seletust: "
        "(1) ühine päev/ajasignaal, (2) sama põllu korduv tasemeviga, (3) muutunud korjeintervall."
    )

    try:
        harvest = db.get_harvest_history(limit=5000)
        events = _events(harvest)
        intervals = _build_intervals(events)
        if intervals.empty:
            raise RuntimeError("Korjeintervalle ei tekkinud.")
        strict0 = _strict_base_rows(intervals)
        if strict0.empty:
            raise RuntimeError("Strict BASE OOS ridu ei tekkinud.")
        strict = _add_same_field_history(strict0)
        focus = strict[(strict["target_date"] >= FOCUS_START) & (strict["target_date"] <= FOCUS_END)].copy()
        if focus.empty:
            raise RuntimeError("19.–22.08 focus-ridu ei leitud.")
        context = strict[(strict["target_date"] >= FOCUS_START) & (strict["target_date"] <= CONTEXT_END)].copy()
        days = _day_anatomy(strict)
        day_focus = days[(days["date"] >= FOCUS_START) & (days["date"] <= FOCUS_END)].copy()
        day_context = days[(days["date"] >= FOCUS_START) & (days["date"] <= CONTEXT_END)].copy()
        s = _diagnostic_scorecards(strict, focus, day_focus)
        verdict, verdict_text = _verdict(s)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strict field-OOS ridu", len(strict))
    c2.metric("Focus 19.–22.08 ridu", len(focus))
    c3.metric("Focus päevi", day_focus["date"].nunique())
    c4.metric("BASE train N 22.08", int(day_focus.iloc[-1]["train_n"]) if len(day_focus) else 0)

    st.markdown("### 1. Otsus · kas laine on ühine aeg, põld või intervall?")
    cards = pd.DataFrame([
        {
            "Kontroll": "Sama päeva põldude BASE-viga sama suunda",
            "N": int(day_focus["sign_n"].sum()),
            "Tulemus": s["mean_coherence"],
            "Ühik": "% keskmine kooskõla",
        },
        {
            "Kontroll": "Pärast eelmise sama põllu vea eemaldamist sama suunda",
            "N": int(day_focus["delta_sign_n"].sum()),
            "Tulemus": s["mean_delta_coherence"],
            "Ühik": "% keskmine kooskõla",
        },
        {
            "Kontroll": "Eelmise sama põllu vea suunahitt focuses",
            "N": s["focus_field_n"],
            "Tulemus": s["focus_field_sign"],
            "Ühik": "%",
        },
        {
            "Kontroll": "Δviga vs Δintervall Spearman focuses",
            "N": int(focus[["resid_delta_vs_prev", "growth_delta_vs_prev"]].dropna().shape[0]),
            "Tulemus": s["focus_interval_rho"],
            "Ühik": "rho",
        },
    ])
    def _fmt_result(row):
        v = row["Tulemus"]
        if pd.isna(v):
            return "—"
        if row["Ühik"].startswith("%") or row["Ühik"] == "%":
            return f"{float(v):.0f}%"
        return f"{float(v):+.2f}"
    cards["Tulemus"] = cards.apply(_fmt_result, axis=1)
    st.dataframe(cards, use_container_width=True, hide_index=True)

    if verdict == "COMMON TIME SIGNAL":
        st.success(f"✅ {verdict}: {verdict_text}")
    elif verdict.startswith("MIXED"):
        st.warning(f"🟡 {verdict}: {verdict_text}")
    else:
        st.error(f"❌ {verdict}: {verdict_text}")

    st.markdown("### 2. Laine anatoomia · 19.–27.08 päev-päevalt")
    wave = day_context.copy()
    wave["actual_minus_base"] = wave["actual"] - wave["base"]
    st.dataframe(
        wave[[
            "date", "fields", "n_fields", "actual", "base", "actual_minus_base",
            "median_resid_log", "sign_agreement_pct", "median_growth",
            "field_prev_median", "median_resid_delta_vs_prev",
            "delta_sign_agreement_pct", "median_growth_delta_vs_prev"
        ]].rename(columns={
            "date":"Päev", "fields":"Põllud", "n_fields":"N põldu", "actual":"Tegelik ABC", "base":"BASE",
            "actual_minus_base":"Tegelik−BASE", "median_resid_log":"Päeva mediaan log-viga",
            "sign_agreement_pct":"Põldude sama suund %", "median_growth":"Mediaan intervall p",
            "field_prev_median":"Eelmise tsükli põlluvea mediaan", "median_resid_delta_vs_prev":"Δviga vs sama põllu eelmine tsükkel",
            "delta_sign_agreement_pct":"Δvea sama suund %", "median_growth_delta_vs_prev":"Δintervall vs eelmine tsükkel",
        }).style.format({
            "Päev": _fmt_day, "Tegelik ABC":"{:.1f}", "BASE":"{:.1f}", "Tegelik−BASE":"{:+.1f}",
            "Päeva mediaan log-viga":"{:+.3f}", "Põldude sama suund %":lambda v:"—" if pd.isna(v) else f"{float(v):.0f}%",
            "Mediaan intervall p":"{:.2f}", "Eelmise tsükli põlluvea mediaan":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
            "Δviga vs sama põllu eelmine tsükkel":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
            "Δvea sama suund %":lambda v:"—" if pd.isna(v) else f"{float(v):.0f}%",
            "Δintervall vs eelmine tsükkel":lambda v:"—" if pd.isna(v) else f"{float(v):+.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Kõige olulisemad veerud on Tegelik−BASE ja Δviga vs sama põllu eelmine tsükkel. "
        "Kui erinevad põllud pööravad koos sama suunda ka pärast oma eelmise tsükli vea eemaldamist, ei saa seda seletada lihtsalt põllutasemega."
    )

    st.markdown("### 3. 19.–22.08 põld-põllult · kas field-mix seletab pöörde?")
    ff = focus.copy()
    ff["actual_minus_base"] = ff["actual"] - ff["base"]
    st.dataframe(
        ff[[
            "target_date", "field", "order", "start_date", "growth", "actual", "base", "actual_minus_base",
            "resid_log", "field_prev_date", "field_prev_resid", "resid_delta_vs_prev",
            "field_prev_growth", "growth_delta_vs_prev", "field_prior_n"
        ]].rename(columns={
            "target_date":"Päev", "field":"Põld", "order":"Jrk", "start_date":"Eelmine korje", "growth":"Intervall p",
            "actual":"Tegelik", "base":"BASE", "actual_minus_base":"Tegelik−BASE", "resid_log":"Praegune log-viga",
            "field_prev_date":"Eelmise strict vea päev", "field_prev_resid":"Eelmine sama põllu log-viga",
            "resid_delta_vs_prev":"Δviga", "field_prev_growth":"Eelmine intervall p", "growth_delta_vs_prev":"Δintervall",
            "field_prior_n":"Varasemaid strict ridu",
        }).style.format({
            "Päev":_fmt_day, "Eelmine korje":_fmt_day, "Eelmise strict vea päev":lambda v:"—" if v is None or pd.isna(v) else _fmt_day(v),
            "Intervall p":"{:.2f}", "Tegelik":"{:.2f}", "BASE":"{:.2f}", "Tegelik−BASE":"{:+.2f}",
            "Praegune log-viga":"{:+.3f}", "Eelmine sama põllu log-viga":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
            "Δviga":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}", "Eelmine intervall p":lambda v:"—" if pd.isna(v) else f"{float(v):.2f}",
            "Δintervall":lambda v:"—" if pd.isna(v) else f"{float(v):+.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 4. Kontroll kogu strict ajaloos · kas põlluviga või intervall kordub?")
    history = pd.DataFrame([
        {
            "Test": "Praegune log-viga vs eelmise sama põllu log-viga",
            "N": int(strict[["resid_log", "field_prev_resid"]].dropna().shape[0]),
            "Spearman rho": s["field_rho"],
            "Suunahitt %": s["field_sign"],
        },
        {
            "Test": "Sama põllu Δviga vs Δintervall",
            "N": int(strict[["resid_delta_vs_prev", "growth_delta_vs_prev"]].dropna().shape[0]),
            "Spearman rho": s["interval_rho"],
            "Suunahitt %": s["interval_sign"],
        },
        {
            "Test": "FOCUS: praegune viga vs eelmise sama põllu viga",
            "N": int(focus[["resid_log", "field_prev_resid"]].dropna().shape[0]),
            "Spearman rho": s["focus_field_rho"],
            "Suunahitt %": s["focus_field_sign"],
        },
        {
            "Test": "FOCUS: sama põllu Δviga vs Δintervall",
            "N": int(focus[["resid_delta_vs_prev", "growth_delta_vs_prev"]].dropna().shape[0]),
            "Spearman rho": s["focus_interval_rho"],
            "Suunahitt %": s["focus_interval_sign"],
        },
    ])
    st.dataframe(
        history.style.format({
            "Spearman rho":lambda v:"—" if pd.isna(v) else f"{float(v):+.2f}",
            "Suunahitt %":lambda v:"—" if pd.isna(v) else f"{float(v):.0f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Kontroll · kõik strict field-OOS read"):
        allshow = strict.copy()
        allshow["actual_minus_base"] = allshow["actual"] - allshow["base"]
        st.dataframe(
            allshow[[
                "target_date", "field", "order", "growth", "actual", "base", "actual_minus_base", "resid_log",
                "field_prev_resid", "resid_delta_vs_prev", "growth_delta_vs_prev", "train_n"
            ]].style.format({
                "target_date":_fmt_day, "growth":"{:.2f}", "actual":"{:.2f}", "base":"{:.2f}",
                "actual_minus_base":"{:+.2f}", "resid_log":"{:+.3f}",
                "field_prev_resid":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
                "resid_delta_vs_prev":lambda v:"—" if pd.isna(v) else f"{float(v):+.3f}",
                "growth_delta_vs_prev":lambda v:"—" if pd.isna(v) else f"{float(v):+.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.success(
        "🔒 LEAKAGE LOCK: iga BASE target on treenitud ainult varasematel intervalle. "
        "'Eelmine sama põllu viga' kasutab ainult varasemat strict-OOS tulemust. Target actual'i kasutatakse siin ainult pärast prognoosi vea diagnoosimiseks."
    )
    st.caption(
        "See LAB ei kasuta ilma, PI-d, slow-state'i ega previous-yield ankrut prognoosi muutmiseks. "
        "Ta ei otsi ühtegi uut akent, cap'i, lambda't ega koefitsienti. Järgmine samm valitakse alles selle vea-anatoomia järgi."
    )


if __name__ == "__main__":
    main()
