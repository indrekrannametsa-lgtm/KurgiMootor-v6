from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Iterable

import numpy as np

DAYLENGTH_LAT = 58.38


def temperature_curve_features(night_avgs: Iterable[float], day_avgs: Iterable[float]) -> dict[str, float]:
    """Bioloogiliselt informeeritud mittelineaarne kurgi temperatuuribaas."""
    night_avgs = np.asarray(list(night_avgs), dtype=float)
    day_avgs = np.asarray(list(day_avgs), dtype=float)
    if len(night_avgs) == 0 or len(day_avgs) == 0:
        return {}

    night_cold = np.maximum(0.0, 16.0 - night_avgs)
    night_warm = np.clip(night_avgs - 16.0, 0.0, 4.0)
    night_hot = np.maximum(0.0, night_avgs - 20.0)
    day_cool = np.maximum(0.0, 20.0 - day_avgs)
    day_warm = np.clip(day_avgs - 20.0, 0.0, 8.0)
    day_hot = np.maximum(0.0, day_avgs - 30.0)

    return {
        "Öö jahedus <16": float(np.mean(night_cold)),
        "Öö jahedus² <16": float(np.mean(night_cold ** 2)),
        "Öö soojus 16-20": float(np.mean(night_warm)),
        "Öö kuumus >20": float(np.mean(night_hot)),
        "Päeva jahedus <20": float(np.mean(day_cool)),
        "Päeva jahedus² <20": float(np.mean(day_cool ** 2)),
        "Päeva soojus 20-28": float(np.mean(day_warm)),
        "Päeva kuumus >30": float(np.mean(day_hot)),
        "Päeva kuumus² >30": float(np.mean(day_hot ** 2)),
    }


def daylength_hours(day_value: date) -> float:
    n = int(day_value.timetuple().tm_yday)
    lat = math.radians(DAYLENGTH_LAT)
    decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + n) / 365.0)
    cos_omega = -math.tan(lat) * math.tan(decl)
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = math.acos(cos_omega)
    return 24.0 * omega / math.pi


def daylength_change_7d(day_value: date) -> float:
    return daylength_hours(day_value) - daylength_hours(day_value - timedelta(days=7))


def season_curve_features(day_value: date, season_start: date) -> dict[str, float]:
    """Paindlik hooaja/taime kulumise alus ilma languse märki ette kirjutamata.

    Hooajapäev on baasmudelis juba lineaarne. Need lisatunnused on kandidaadid,
    mille väärtuse peab ajaline avastus+kinnitus test tõestama.
    Hinge-punktid annavad kõverale võimaluse muuta kallet hilishooajal.
    """
    d = float((day_value - season_start).days)
    d_nonneg = max(0.0, d)
    h35 = max(0.0, d_nonneg - 35.0)
    h50 = max(0.0, d_nonneg - 50.0)
    h65 = max(0.0, d_nonneg - 65.0)
    return {
        "Hooajapäev²": d_nonneg ** 2,
        "Hooaeg 35+": h35,
        "Hooaeg 50+": h50,
        "Hooaeg 65+": h65,
        "Hooaeg 50+²": h50 ** 2,
    }


def chronological_discovery_confirmation_days(
    dates: np.ndarray,
    base_pred: np.ndarray,
    target: np.ndarray,
    discovery_fraction: float = 0.60,
    min_days_each: int = 3,
) -> tuple[set[date], set[date]]:
    """Jaga ausad walk-forward testipäevad kronoloogiliselt avastuseks ja kinnituseks."""
    valid = np.isfinite(base_pred) & np.isfinite(target)
    unique_days = sorted(set(dates[np.where(valid)[0]]))
    if len(unique_days) < min_days_each * 2:
        return set(), set()
    discovery_count = int(math.ceil(len(unique_days) * discovery_fraction))
    discovery_count = max(min_days_each, min(discovery_count, len(unique_days) - min_days_each))
    return set(unique_days[:discovery_count]), set(unique_days[discovery_count:])


def build_ridge_design(X_base, fields, train_idx, test_idx, extra_arrays, n_fields: int = 14):
    tr_parts = [X_base[train_idx]]
    te_parts = [X_base[test_idx]]
    tr_missing_parts, te_missing_parts = [], []
    fills = []
    for arr in extra_arrays:
        tr = arr[train_idx]
        te = arr[test_idx]
        finite = tr[np.isfinite(tr)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        fills.append(fill)
        tr_parts.append(np.where(np.isfinite(tr), tr, fill).reshape(-1, 1))
        te_parts.append(np.where(np.isfinite(te), te, fill).reshape(-1, 1))
        tr_missing_parts.append((~np.isfinite(tr)).astype(float).reshape(-1, 1))
        te_missing_parts.append((~np.isfinite(te)).astype(float).reshape(-1, 1))
    xtr = np.column_stack(tr_parts)
    xte = np.column_stack(te_parts)
    means = xtr.mean(axis=0)
    scales = xtr.std(axis=0)
    scales[scales < 1e-9] = 1.0
    ztr = (xtr - means) / scales
    zte = (xte - means) / scales
    ftr = np.zeros((len(train_idx), n_fields), dtype=float)
    fte = np.zeros((len(test_idx), n_fields), dtype=float)
    for ri, f in enumerate(fields[train_idx]):
        if 1 <= int(f) <= n_fields:
            ftr[ri, int(f) - 1] = 1.0
    for ri, f in enumerate(fields[test_idx]):
        if 1 <= int(f) <= n_fields:
            fte[ri, int(f) - 1] = 1.0
    tr_missing = np.column_stack(tr_missing_parts) if tr_missing_parts else np.empty((len(train_idx), 0))
    te_missing = np.column_stack(te_missing_parts) if te_missing_parts else np.empty((len(test_idx), 0))
    Xtr = np.column_stack([np.ones(len(train_idx)), ztr, tr_missing, ftr])
    Xte = np.column_stack([np.ones(len(test_idx)), zte, te_missing, fte])
    return Xtr, Xte, means, scales, fills


def ridge_walk_predict(X_base, fields, target, extra_arrays, train_idx, test_idx, *, alpha=10.0, floor_zero=True, field_alpha=80.0):
    Xtr, Xte, _, _, _ = build_ridge_design(X_base, fields, train_idx, test_idx, extra_arrays)
    penalty = np.eye(Xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    penalty[-14:, -14:] = np.eye(14) * field_alpha
    beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ target[train_idx]
    values = Xte @ beta
    return np.maximum(values, 0.0) if floor_zero else values


def abc_growth_walk_predict(X_base, fields, log_target, extra_arrays, train_idx, test_idx, *, min_train_rows=10, alpha=10.0, field_alpha=80.0, z_clip=2.5, log_eps=0.05):
    valid_train = train_idx[np.isfinite(log_target[train_idx])]
    if len(valid_train) < min_train_rows:
        return np.full(len(test_idx), np.nan, dtype=float)
    Xtr, Xte, _, _, _ = build_ridge_design(X_base, fields, valid_train, test_idx, extra_arrays)
    n_numeric = X_base.shape[1] + len(extra_arrays)
    Xtr[:, 1:1+n_numeric] = np.clip(Xtr[:, 1:1+n_numeric], -z_clip, z_clip)
    Xte[:, 1:1+n_numeric] = np.clip(Xte[:, 1:1+n_numeric], -z_clip, z_clip)
    penalty = np.eye(Xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    penalty[-14:, -14:] = np.eye(14) * field_alpha
    beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ log_target[valid_train]
    latent = Xte @ beta
    return np.exp(np.clip(latent, np.log(log_eps), 6.0))


def fit_full_generic(X_base, fields, target, extra_arrays, *, alpha=10.0, field_alpha=80.0):
    idx = np.where(np.isfinite(target))[0]
    Xtr, _, means, scales, fills = build_ridge_design(X_base, fields, idx, idx, extra_arrays)
    penalty = np.eye(Xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    penalty[-14:, -14:] = np.eye(14) * field_alpha
    beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ target[idx]
    return {"beta": beta, "means": means, "scales": scales, "fills": fills, "n_extra": len(extra_arrays)}


def predict_full_generic(model, field_no, base_values, extra_values, *, floor_zero=True):
    x = list(base_values)
    miss = []
    for i, value in enumerate(extra_values):
        try:
            finite_value = value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            finite_value = False
        if not finite_value:
            x.append(model["fills"][i])
            miss.append(1.0)
        else:
            x.append(float(value))
            miss.append(0.0)
    x = np.array([x], dtype=float)
    z = (x - model["means"]) / model["scales"]
    onehot = np.zeros((1, 14), dtype=float)
    if 1 <= int(field_no) <= 14:
        onehot[0, int(field_no) - 1] = 1.0
    Xp = np.column_stack([np.ones(1), z, np.array([miss], dtype=float), onehot])
    value = float((Xp @ model["beta"])[0])
    return max(0.0, value) if floor_zero else value


def fit_full_abc_growth(X_base, fields, log_target, extra_arrays, *, alpha=10.0, field_alpha=80.0, z_clip=2.5):
    idx = np.where(np.isfinite(log_target))[0]
    Xtr, _, means, scales, fills = build_ridge_design(X_base, fields, idx, idx, extra_arrays)
    n_numeric = X_base.shape[1] + len(extra_arrays)
    Xtr[:, 1:1+n_numeric] = np.clip(Xtr[:, 1:1+n_numeric], -z_clip, z_clip)
    penalty = np.eye(Xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    penalty[-14:, -14:] = np.eye(14) * field_alpha
    beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ log_target[idx]
    return {"beta": beta, "means": means, "scales": scales, "fills": fills, "n_extra": len(extra_arrays), "z_clip": z_clip}


def predict_full_abc_growth(model, field_no, base_values, extra_values, *, log_eps=0.05):
    x = list(base_values)
    miss = []
    for i, value in enumerate(extra_values):
        try:
            finite_value = value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            finite_value = False
        if not finite_value:
            x.append(model["fills"][i])
            miss.append(1.0)
        else:
            x.append(float(value))
            miss.append(0.0)
    x = np.array([x], dtype=float)
    z = (x - model["means"]) / model["scales"]
    z = np.clip(z, -model.get("z_clip", 2.5), model.get("z_clip", 2.5))
    onehot = np.zeros((1, 14), dtype=float)
    if 1 <= int(field_no) <= 14:
        onehot[0, int(field_no) - 1] = 1.0
    Xp = np.column_stack([np.ones(1), z, np.array([miss], dtype=float), onehot])
    latent = float((Xp @ model["beta"])[0])
    return float(np.exp(np.clip(latent, np.log(log_eps), 6.0)))
