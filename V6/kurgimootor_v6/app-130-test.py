from __future__ import annotations

"""
KurgiMootor LAB-138 STATE BENCHMARK
===================================

Eesmärk
-------
Enne uue saagimootori ehitamist mõõdame ajamasinas, kui palju infot annab ainult
korje-state. Ilma ei kasutata selles katses üldse. See on teadlikult "nullkiht":
kui lihtne state ei tööta, pole mõtet ilmaefekte selle peale kuhjata.

Replay
------
- issue-päevad algavad 22.07.2026;
- iga issue-päeva järel prognoositakse teadaoleva ajaloolise korjeplaani järgi +1...+9;
- tuleviku tegelikust reast kasutatakse AINULT kuupäeva, põldu ja järjekorda;
- A/B/C/XL/total ei lähe prognoosi sisendisse;
- sama issue +1...+9 sees prognoositud saaki state'ina tagasi ei söödeta;
- päevatäpsust skooritakse ainult päevadel, kus on 3 usaldusväärset ABC-rida.

Viis fikseeritud benchmarki
---------------------------
OWN1
    Sama põllu viimane usaldusväärne ABC-kasvukiirus (ABC / täpne kasvupäev).
OWN2
    Sama põllu kahe viimase kasvukiiruse mediaan.
BLOCK1
    Kõige värskema täieliku korjepäeva põldude kasvukiiruste mediaan.
BLOCK3
    Kolme viimase täieliku korjepäeva mediaanide kaalutud keskmine (1,2,4; värskeim 4).
OWN2+BLOCK3
    OWN2 ja BLOCK3 50/50 log-skaala segu (geomeetriline keskmine).

Cold-start
----------
Kui OWN1/OWN2 pole põllul veel arvutatav, kasutatakse selle variandi jaoks BLOCK3
fallback'i. Fallback'i osakaal kuvatakse eraldi, et me ei peidaks vähese ajaloo mõju.

READ ONLY: ainult db.get_harvest_history(). DB kirjutamisi ega production-importi pole.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Tuple
import math
import sys

import numpy as np
import pandas as pd

TZ = ZoneInfo("Europe/Tallinn")
TODAY = datetime.now(TZ).date()
LAB_VERSION = "LAB-138-STATE-BENCHMARK-V1"
REPLAY_START = date(2026, 7, 22)
WARM_START = date(2026, 8, 1)
DEFAULT_GROWTH_DAYS = 14.0 / 3.0
EXPECTED_FIELDS_PER_DAY = 3
BLOCK_LOOKBACK_DAYS = 12

MODEL_NAMES = ["OWN1", "OWN2", "BLOCK1", "BLOCK3", "OWN2+BLOCK3"]


def _d(value) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _f(value) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _abc_from_row(row: dict) -> Optional[float]:
    vals = [_f(row.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in vals):
        return None
    abc = float(sum(vals))
    return abc if abc > 0 else None


def _quality_reliable(row: dict) -> bool:
    q = str(row.get("data_quality") or row.get("quality") or "").strip().lower()
    return q not in {"hinnanguline", "ligikaudne", "estimated", "approximate"}


@dataclass
class Event:
    day: date
    field: int
    order: int
    abc: Optional[float]
    reliable: bool
    source: str


def _event_sort_key(e: Event) -> Tuple[date, int, int]:
    return (e.day, e.order, e.field)


def _copy_events(events: Iterable[Event]) -> List[Event]:
    return [Event(e.day, e.field, e.order, e.abc, e.reliable, e.source) for e in events]


def _prepare_events(harvest_rows_raw: List[dict]) -> Tuple[List[Event], Dict[Tuple[date, int], dict]]:
    events: List[Event] = []
    actual_lookup: Dict[Tuple[date, int], dict] = {}
    for raw in harvest_rows_raw:
        dd = _d(raw.get("harvest_date"))
        try:
            field = int(raw.get("field_no"))
        except Exception:
            continue
        if dd is None or not (1 <= field <= 14):
            continue
        try:
            order = int(raw.get("harvest_order") or 1)
        except Exception:
            order = 1
        reliable = _quality_reliable(raw)
        abc = _abc_from_row(raw) if reliable else None
        source = "actual" if abc is not None else "estimated_event"
        events.append(Event(dd, field, order, abc, abc is not None, source))
        if abc is not None:
            rr = dict(raw)
            rr["_abc"] = float(abc)
            rr["_day"] = dd
            rr["_field"] = field
            rr["_order"] = order
            actual_lookup[(dd, field)] = rr
    events.sort(key=_event_sort_key)
    return events, actual_lookup


def _field_events_before(events: List[Event], field: int, cutoff_exclusive: date) -> List[Event]:
    return sorted(
        [e for e in events if e.field == field and e.day < cutoff_exclusive],
        key=_event_sort_key,
    )


def _growth_between(prev: Event, cur_day: date, cur_order: int) -> float:
    g = float((cur_day - prev.day).days) + (cur_order - prev.order) * (3.0 / 24.0)
    return max(0.5, g)


def _growth_for_target(schedule_events: List[Event], field: int, target_day: date, target_order: int) -> float:
    hist = _field_events_before(schedule_events, field, target_day)
    if hist:
        return _growth_between(hist[-1], target_day, target_order)
    return DEFAULT_GROWTH_DAYS


def _rate_history(events: List[Event], field: int, cutoff_exclusive: date) -> List[Tuple[Event, float]]:
    """Rate = selle korje ABC / kasvuaeg eelmisest korjesündmusest."""
    hist = _field_events_before(events, field, cutoff_exclusive)
    out: List[Tuple[Event, float]] = []
    for i in range(1, len(hist)):
        prev, cur = hist[i - 1], hist[i]
        if not (cur.reliable and cur.abc is not None and cur.abc > 0):
            continue
        growth = _growth_between(prev, cur.day, cur.order)
        rate = float(cur.abc) / growth
        if math.isfinite(rate) and rate > 0:
            out.append((cur, rate))
    return out


def _own_rates_snapshot(known_events: List[Event], field: int, issue_day: date) -> List[float]:
    """Ainult issue-päevaks teada olevad rate'id. Viimane rate peab vastama viimasele korjesündmusele."""
    cutoff = issue_day + timedelta(days=1)
    hist_events = _field_events_before(known_events, field, cutoff)
    rh = _rate_history(known_events, field, cutoff)
    if not hist_events or not rh:
        return []
    # Kui kõige värskem korje on hinnanguline/ABC-ta, ei tohi vana rate'i nimetada värskeks state'iks.
    latest_event = hist_events[-1]
    latest_rate_event = rh[-1][0]
    if (latest_event.day, latest_event.order) != (latest_rate_event.day, latest_rate_event.order):
        return []
    return [float(rate) for _ev, rate in rh]


def _complete_day_rate_medians(known_events: List[Event], issue_day: date) -> List[Tuple[date, float, int]]:
    """Täielikud 3-põllu päevad ja nende rate-mediaanid, ainult issue-päevani."""
    cutoff = issue_day + timedelta(days=1)
    rates_by_day: Dict[date, List[float]] = {}
    fields_by_day: Dict[date, set] = {}
    for field in range(1, 15):
        for ev, rate in _rate_history(known_events, field, cutoff):
            if ev.day > issue_day or ev.source != "actual":
                continue
            rates_by_day.setdefault(ev.day, []).append(float(rate))
            fields_by_day.setdefault(ev.day, set()).add(field)
    rows: List[Tuple[date, float, int]] = []
    for dd in sorted(rates_by_day):
        n = len(fields_by_day.get(dd, set()))
        if n == EXPECTED_FIELDS_PER_DAY and len(rates_by_day[dd]) == EXPECTED_FIELDS_PER_DAY:
            rows.append((dd, float(np.median(rates_by_day[dd])), n))
    return rows


def _fallback_recent_rate(known_events: List[Event], issue_day: date) -> Optional[float]:
    """Cold-start fallback: esmalt BLOCK3; kui täielikke päevi pole, kõigi värskete rate'ide mediaan."""
    day_meds = _complete_day_rate_medians(known_events, issue_day)
    if day_meds:
        last3 = day_meds[-3:]
        weights = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
        weights /= weights.sum()
        return float(sum(w * row[1] for w, row in zip(weights, last3)))

    vals: List[float] = []
    cutoff = issue_day + timedelta(days=1)
    for field in range(1, 15):
        rh = _rate_history(known_events, field, cutoff)
        if not rh:
            continue
        ev, rate = rh[-1]
        if 0 <= (issue_day - ev.day).days <= BLOCK_LOOKBACK_DAYS:
            vals.append(float(rate))
    return float(np.median(vals)) if vals else None


def _benchmark_rates(known_events: List[Event], field: int, issue_day: date) -> Optional[Tuple[Dict[str, float], bool]]:
    day_meds = _complete_day_rate_medians(known_events, issue_day)
    fallback = _fallback_recent_rate(known_events, issue_day)
    if fallback is None or fallback <= 0:
        return None

    if day_meds:
        block1 = float(day_meds[-1][1])
        last3 = day_meds[-3:]
        weights = np.asarray([1.0, 2.0, 4.0], dtype=float)[-len(last3):]
        weights /= weights.sum()
        block3 = float(sum(w * row[1] for w, row in zip(weights, last3)))
    else:
        block1 = block3 = float(fallback)

    own = _own_rates_snapshot(known_events, field, issue_day)
    used_fallback = len(own) == 0
    own1 = float(own[-1]) if own else float(block3)
    if len(own) >= 2:
        own2 = float(np.median(own[-2:]))
    elif own:
        own2 = float(own[-1])
    else:
        own2 = float(block3)

    # 50/50 log-skaala segu = geomeetriline keskmine. Ei õpita replay tulemuse järgi.
    own_block = math.sqrt(max(1e-9, own2) * max(1e-9, block3))
    rates = {
        "OWN1": own1,
        "OWN2": own2,
        "BLOCK1": block1,
        "BLOCK3": block3,
        "OWN2+BLOCK3": own_block,
    }
    return rates, used_fallback


def _issue_dates(last_actual_day: date) -> List[date]:
    end = min(last_actual_day - timedelta(days=1), TODAY - timedelta(days=1))
    if end < REPLAY_START:
        return []
    return [REPLAY_START + timedelta(days=i) for i in range((end - REPLAY_START).days + 1)]


def _future_schedule_stubs(actual_lookup: Dict[Tuple[date, int], dict], issue_day: date) -> List[Tuple[date, int, int]]:
    out = []
    max_day = issue_day + timedelta(days=9)
    for (dd, field), row in actual_lookup.items():
        if issue_day < dd <= max_day:
            out.append((dd, field, int(row.get("_order") or 1)))
    out.sort(key=lambda x: (x[0], x[2], x[1]))
    return out


def _complete_actual_days(actual_lookup: Dict[Tuple[date, int], dict]) -> set:
    by_day: Dict[date, set] = {}
    for (dd, field) in actual_lookup:
        by_day.setdefault(dd, set()).add(field)
    return {dd for dd, fields in by_day.items() if len(fields) == EXPECTED_FIELDS_PER_DAY}


def _predict_issue(
    issue_day: date,
    all_events: List[Event],
    actual_lookup: Dict[Tuple[date, int], dict],
) -> List[dict]:
    known_events = [e for e in all_events if e.day <= issue_day]
    schedule_events = _copy_events(known_events)
    stubs = _future_schedule_stubs(actual_lookup, issue_day)
    out: List[dict] = []

    # Kõik 5 state'i külmutatakse issue-päeval põllu kaupa. Tuleviku prognoos ei muutu uueks sensoriks.
    frozen: Dict[int, Tuple[Dict[str, float], bool]] = {}

    for target_day, field, order in stubs:
        if field not in frozen:
            bench = _benchmark_rates(known_events, field, issue_day)
            if bench is None:
                continue
            frozen[field] = bench
        rates, used_fallback = frozen[field]
        growth = _growth_for_target(schedule_events, field, target_day, order)
        actual_row = actual_lookup.get((target_day, field))
        actual_abc = float(actual_row["_abc"]) if actual_row is not None else np.nan
        row = {
            "issue_day": issue_day,
            "target_day": target_day,
            "lead": int((target_day - issue_day).days),
            "field": field,
            "order": order,
            "growth_days": float(growth),
            "actual_abc": actual_abc,
            "own_fallback": int(used_fallback),
        }
        for name in MODEL_NAMES:
            row[name] = float(rates[name] * growth)
        out.append(row)

        # Ainult korjesündmuse ajamärk läheb edasi, saaginumbrit mitte.
        schedule_events.append(Event(target_day, field, order, None, False, "simulated_schedule"))
        schedule_events.sort(key=_event_sort_key)
    return out


def _daily_scores(field_df: pd.DataFrame, complete_days: set) -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame()
    # Skoorime ainult täielikud 3-põllu tegelikud korjepäevad.
    f = field_df[field_df["target_day"].isin(complete_days)].copy()
    if f.empty:
        return pd.DataFrame()
    agg = {
        "actual_abc": ("actual_abc", "sum"),
        "n_fields": ("field", "count"),
        "fallback_fields": ("own_fallback", "sum"),
    }
    for name in MODEL_NAMES:
        agg[name] = (name, "sum")
    daily = f.groupby(["issue_day", "target_day", "lead"], as_index=False).agg(**agg)
    daily = daily[daily["n_fields"] == EXPECTED_FIELDS_PER_DAY].copy()
    if daily.empty:
        return daily
    daily["fallback_pct"] = 100.0 * daily["fallback_fields"] / EXPECTED_FIELDS_PER_DAY
    for name in MODEL_NAMES:
        daily[f"{name}_err"] = daily[name] - daily["actual_abc"]
        daily[f"{name}_ape"] = daily[f"{name}_err"].abs() / daily["actual_abc"].clip(lower=0.1)
        daily[f"{name}_bias"] = 100.0 * daily[f"{name}_err"] / daily["actual_abc"].clip(lower=0.1)
    return daily


def _summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        rows.append({
            "Mudel": name,
            "MAPE %": 100.0 * float(daily[f"{name}_ape"].mean()),
            "±20% sees %": 100.0 * float((daily[f"{name}_ape"] <= 0.20).mean()),
            "Bias %": float(daily[f"{name}_bias"].mean()),
            "MAE ABC": float(daily[f"{name}_err"].abs().mean()),
            "N": int(len(daily)),
        })
    return pd.DataFrame(rows).sort_values(["MAPE %", "MAE ABC"])


def _lead_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in daily.groupby("lead"):
        row = {"Lead": int(lead), "N": int(len(g)), "Fallback %": float(g["fallback_pct"].mean())}
        for name in MODEL_NAMES:
            row[f"{name} MAPE %"] = 100.0 * float(g[f"{name}_ape"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Lead")


def _window_summary(daily: pd.DataFrame, start_day: date) -> pd.DataFrame:
    g = daily[daily["target_day"] >= start_day].copy()
    return _summary_table(g) if not g.empty else pd.DataFrame()


def _self_test() -> None:
    rows = []
    # 4 põldu, korduvad 5-päevased korjed; piisab state helperite testiks.
    for f in range(1, 5):
        for j, dd in enumerate([date(2026,7,10), date(2026,7,15), date(2026,7,20)]):
            rows.append({
                "harvest_date": dd.isoformat(), "field_no": f, "harvest_order": f if f <= 3 else 1,
                "a": 0.2, "b": 3.0+j, "c": 4.0+j, "xl": 1.0,
                "data_quality": "Kinnitatud",
            })
    events, lookup = _prepare_events(rows)
    assert len(events) == 12 and len(lookup) == 12
    rh = _rate_history(events, 1, date(2026,7,21))
    assert len(rh) == 2 and all(rate > 0 for _ev, rate in rh)
    bench = _benchmark_rates(events, 1, date(2026,7,20))
    assert bench is not None
    rates, _fb = bench
    assert set(rates) == set(MODEL_NAMES) and all(math.isfinite(v) and v > 0 for v in rates.values())

    # Tuleviku saak ei lähe state'i tagasi: lisame ainult schedule-eventi abc=None.
    schedule = _copy_events([e for e in events if e.day <= date(2026,7,20)])
    schedule.append(Event(date(2026,7,25), 1, 1, None, False, "simulated_schedule"))
    g = _growth_for_target(schedule, 1, date(2026,7,30), 1)
    assert 4.5 <= g <= 5.5
    print("LAB-138 SELF-TEST OK")


def main() -> None:
    import streamlit as st
    import db

    st.set_page_config(page_title="KurgiMootor LAB-138", layout="wide")
    st.error("🧪 LAB-138 STATE BENCHMARK · READ-ONLY · ILMA EI KASUTA")
    st.title("KurgiMootor · puhas STATE benchmark")
    st.caption(
        "Viis lihtsat korje-state'i varianti samas 22.07 → +1…+9 ajamasinas. "
        "Selles katses ei kasutata ilma ega õitsemist: otsime kõigepealt, milline state ise kannab infot."
    )

    with st.expander("Katse reeglid", expanded=False):
        st.markdown(
            """
- **OWN1:** sama põllu viimane ABC/kasvupäev.
- **OWN2:** sama põllu 2 viimase ABC/kasvupäev mediaan.
- **BLOCK1:** viimase täieliku 3-põllu korjepäeva rate-mediaan.
- **BLOCK3:** 3 viimase täieliku päeva rate-mediaanide kaalutud keskmine 1/2/4.
- **OWN2+BLOCK3:** OWN2 ja BLOCK3 geomeetriline keskmine.
- Kui põllu oma state pole veel olemas, kasutab OWN cold-startis BLOCK3 fallback'i; fallback kuvatakse.
- +1…+9 sees prognoositud saaki järgmise korje state'iks ei kasutata.
- Skoorimisse lähevad ainult päevad, kus on 3 usaldusväärset ABC-rida; tänane osaline korjepäev ei moonuta tulemust.
- Ainult **A+B+C**. XL/C-B ei ole selles katses.
            """
        )

    @st.cache_data(ttl=120, show_spinner=False)
    def _load_data():
        return db.get_harvest_history(limit=5000)

    if st.button("Värskenda DB andmed", type="secondary"):
        _load_data.clear()
        st.rerun()

    try:
        harvest_rows = _load_data()
    except Exception as exc:
        st.error(f"DB lugemine ebaõnnestus: {exc}")
        st.stop()

    events, actual_lookup = _prepare_events(harvest_rows)
    if not actual_lookup:
        st.error("Usaldusväärseid ABC ridu ei leitud.")
        st.stop()
    complete_days = _complete_actual_days(actual_lookup)
    last_actual_day = max(dd for dd, _fno in actual_lookup)
    issue_days = _issue_dates(last_actual_day)
    if not issue_days:
        st.error("22.07 järel pole replay jaoks piisavalt päevi.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Replay algus", REPLAY_START.strftime("%d.%m.%Y"))
    c2.metric("Usaldusväärseid ABC ridu", str(len(actual_lookup)))
    c3.metric("Täielikke 3-põllu päevi", str(len(complete_days)))
    c4.metric("Issue-päevi", str(len(issue_days)))

    if not st.button("▶ Jooksuta STATE benchmark", type="primary"):
        st.info("Vajuta üks kord. Ilma-API kutseid pole; arvutus on väike ja productionit ei puuduta.")
        st.stop()

    rows_out: List[dict] = []
    progress = st.progress(0.0, text="STATE replay…")
    for i, issue in enumerate(issue_days):
        rows_out.extend(_predict_issue(issue, events, actual_lookup))
        progress.progress((i + 1) / len(issue_days), text=f"STATE replay {issue.strftime('%d.%m')}…")
    progress.empty()

    field_df = pd.DataFrame(rows_out)
    daily = _daily_scores(field_df, complete_days)
    if daily.empty:
        st.error("Ühtegi täielikku 3-põllu päeva ei saanud skoorida.")
        st.stop()

    overall = _summary_table(daily)
    warm = _window_summary(daily, WARM_START)
    best = overall.iloc[0]

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Võitja · kogu MAPE", f"{best['MAPE %']:.1f}%", str(best["Mudel"]))
    if not warm.empty:
        warm_best = warm.iloc[0]
        top2.metric("Võitja · al 01.08", f"{warm_best['MAPE %']:.1f}%", str(warm_best["Mudel"]))
    else:
        top2.metric("Võitja · al 01.08", "—")
    plus1 = daily[daily["lead"] == 1]
    if not plus1.empty:
        p1 = _summary_table(plus1).iloc[0]
        top3.metric("+1p võitja", f"{p1['MAPE %']:.1f}%", str(p1["Mudel"]))
    else:
        top3.metric("+1p võitja", "—")
    top4.metric("Keskmine OWN fallback", f"{daily['fallback_pct'].mean():.0f}%")

    st.markdown("### 1. Kogu replay · mudelite võrdlus")
    st.dataframe(
        overall.style.format({
            "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%", "MAE ABC": "{:.1f}"
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### 2. Alates 01.08 · kui enamikul põldudel on juba oma state")
    if warm.empty:
        st.caption("Selles aknas pole skooritavaid ridu.")
    else:
        st.dataframe(
            warm.style.format({
                "MAPE %": "{:.1f}%", "±20% sees %": "{:.0f}%", "Bias %": "{:+.1f}%", "MAE ABC": "{:.1f}"
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### 3. Lead 1–9 · MAPE")
    lead = _lead_table(daily)
    fmt = {"Fallback %": "{:.0f}%"}
    fmt.update({f"{name} MAPE %": "{:.1f}%" for name in MODEL_NAMES})
    st.dataframe(lead.style.format(fmt), use_container_width=True, hide_index=True)

    with st.expander("Päevade kaupa · tegelik vs 5 benchmarki", expanded=False):
        show = daily.copy().sort_values(["target_day", "issue_day"])
        show["Issue"] = show["issue_day"].map(lambda d: d.strftime("%d.%m"))
        show["Siht"] = show["target_day"].map(lambda d: d.strftime("%d.%m"))
        show["Tegelik ABC"] = show["actual_abc"]
        cols = ["Issue", "Siht", "lead", "Tegelik ABC"] + MODEL_NAMES + ["fallback_pct"]
        st.dataframe(
            show[cols].style.format({
                "Tegelik ABC": "{:.1f}", **{name: "{:.1f}" for name in MODEL_NAMES}, "fallback_pct": "{:.0f}%"
            }),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        f"{LAB_VERSION} · NO WEATHER · 5 fikseeritud state benchmarki · "
        f"score only {EXPECTED_FIELDS_PER_DAY}-field complete days · DB read-only"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
