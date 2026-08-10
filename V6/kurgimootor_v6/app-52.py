from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import math
import hashlib
import json
import re

from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import db
from core import WeatherService

TODAY = datetime.now(ZoneInfo("Europe/Tallinn")).date()
SEASON_START = date(TODAY.year, 6, 15)
IDEA_FULL_SEARCH_EVERY_COMPLETE_DAYS = 3
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "kurgimootor_icon.png"
APP_ICON = Image.open(APP_ICON_PATH)
st.set_page_config(page_title="KurgiMootor V6.4", page_icon=APP_ICON, layout="wide")

# PWA / iOS Home Screen metadata. Streamlit does not expose <head> directly,
# so a tiny same-origin component appends the standard tags to the parent page.
components.html(
    """
    <script>
    (() => {
      const doc = window.parent.document;

      function ensureLink(rel, href, attrs = {}) {
        let el = doc.head.querySelector(`link[rel="${rel}"]`);
        if (!el) {
          el = doc.createElement("link");
          el.rel = rel;
          doc.head.appendChild(el);
        }
        el.href = href;
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      }

      function ensureMeta(name, content, byProperty = false) {
        const attr = byProperty ? "property" : "name";
        let el = doc.head.querySelector(`meta[${attr}="${name}"]`);
        if (!el) {
          el = doc.createElement("meta");
          el.setAttribute(attr, name);
          doc.head.appendChild(el);
        }
        el.setAttribute("content", content);
      }

      ensureLink("manifest", "/app/static/manifest.webmanifest");
      ensureLink("apple-touch-icon", "/app/static/apple-touch-icon.png", {"sizes": "180x180"});
      ensureLink("icon", "/app/static/kurgimootor-192.png", {"type": "image/png", "sizes": "192x192"});

      ensureMeta("theme-color", "#0b3b24");
      ensureMeta("mobile-web-app-capable", "yes");
      ensureMeta("apple-mobile-web-app-capable", "yes");
      ensureMeta("apple-mobile-web-app-status-bar-style", "black-translucent");
      ensureMeta("apple-mobile-web-app-title", "KurgiMootor");

      doc.title = "KurgiMootor";
    })();
    </script>
    """,
    height=0,
    width=0,
)



def _n(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _mark_model_dirty(reason: str) -> None:
    """Märgi, et järgmine teadlik mudeliring peab kasutama uusi andmeid."""
    db.set_app_setting("model_dirty", "1")
    db.set_app_setting("model_dirty_reason", str(reason))
    db.set_app_setting("model_dirty_at", datetime.now(ZoneInfo("Europe/Tallinn")).isoformat())


def _model_is_dirty() -> bool:
    return db.get_app_setting("model_dirty", "1") == "1"


def _temperature_curve_features(night_avgs, day_avgs):
    """Literature-informed, data-fitted nonlinear cucumber temperature basis.

    Breakpoints encode only where physiology plausibly changes regime.
    The coefficients/strength are learned from this farm's harvest data.

    Night:
      <16 C  : cold deficit + squared cold deficit (curvature; 12 can hurt far more than 14)
      16-20 C: warm-night band
      >20 C  : hot-night excess

    Day:
      <20 C  : cool-day deficit + squared deficit
      20-28 C: productive warm band
      >30 C  : heat excess + squared heat excess

    Values are interval means so harvest interval length remains a separate feature.
    """
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




# Astronoomiline päevapikkus Pärnu piirkonna laiuskraadil.
# Longitude pole päevapikkuse kestuse jaoks vajalik; kasutame geograafilist laiust ~58.38 N.
DAYLENGTH_LAT = 58.38

def _daylength_hours(day_value):
    n = int(day_value.timetuple().tm_yday)
    lat = math.radians(DAYLENGTH_LAT)
    # NOAA-tüüpi lihtsustatud päikesedeklinatsioon; piisav fotoperioodi tunnuseks.
    decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + n) / 365.0)
    cos_omega = -math.tan(lat) * math.tan(decl)
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = math.acos(cos_omega)
    return 24.0 * omega / math.pi

def _daylength_change_7d(day_value):
    return _daylength_hours(day_value) - _daylength_hours(day_value - timedelta(days=7))

def _format_field_value(value, digits=1):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return _fmt(float(value), digits)
    except (TypeError, ValueError):
        return str(value)

def _short_date(value: date) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{value.day:02d}. {months[value.month - 1]}"


def _weekday_letter(value: date) -> str:
    return ["E", "T", "K", "N", "R", "L", "P"][value.weekday()]


def _daily_summary(rows):
    count = len(rows)
    a = sum(_n(r.get("a")) for r in rows)
    b = sum(_n(r.get("b")) for r in rows)
    c = sum(_n(r.get("c")) for r in rows)
    xl = sum(_n(r.get("xl")) for r in rows)
    total = sum(_n(r.get("total")) for r in rows)
    cb = c / b if b > 0 else None
    return {
        "count": count,
        "a": a,
        "b": b,
        "c": c,
        "xl": xl,
        "total": total,
        "cb": cb,
    }


def _next_field(field_no: int) -> int:
    return 1 if field_no >= 14 else field_no + 1


def _planned_fields_for_day(day: date, today_rows, history_rows):
    # Kui tänasest on vähemalt üks õige korje sisestatud, kasutame päeva esimese
    # korje põldu ploki algusena. Nii jääb plaan kooskõlla päriselt sisestatud andmetega.
    ordered_today = sorted(today_rows, key=lambda r: int(r.get("harvest_order") or 99))
    if ordered_today:
        first = int(ordered_today[0].get("field_no"))
        return [first, _next_field(first), _next_field(_next_field(first))]

    # Muidu võtame viimase varasema päeva viimase korjepõllu ja liigume sealt edasi.
    previous = []
    for r in history_rows:
        try:
            rday = date.fromisoformat(str(r.get("harvest_date")))
        except (TypeError, ValueError):
            continue
        if rday < day:
            previous.append(r)

    if previous:
        latest_day = max(date.fromisoformat(str(r.get("harvest_date"))) for r in previous)
        latest_rows = [r for r in previous if str(r.get("harvest_date")) == latest_day.isoformat()]
        latest_rows.sort(key=lambda r: int(r.get("harvest_order") or 99))
        last_field = int(latest_rows[-1].get("field_no"))
        first = _next_field(last_field)
        return [first, _next_field(first), _next_field(_next_field(first))]

    return [1, 2, 3]


def _field_table(rows, planned_fields=None):
    by_field = {int(r.get("field_no")): r for r in rows if r.get("field_no") is not None}
    fields = planned_fields or [int(r.get("field_no")) for r in rows if r.get("field_no") is not None]
    table = []
    missing_rows = set()

    for field_no in fields:
        r = by_field.get(int(field_no))
        if r is None:
            missing_rows.add(len(table))
            table.append({
                "Põld": int(field_no),
                "A": None,
                "B": None,
                "C": None,
                "XL": None,
                "Kokku": "Andmed puuduvad",
                "C/B": None,
            })
            continue

        b = _n(r.get("b"))
        c = _n(r.get("c"))
        table.append({
            "Põld": r.get("field_no"),
            "A": _n(r.get("a")),
            "B": b,
            "C": c,
            "XL": _n(r.get("xl")),
            "Kokku": _n(r.get("total")),
            "C/B": round(c / b, 2) if b > 0 else None,
        })

    # Kui mingil põhjusel on sisestatud põld, mida plaanis polnud, näitame ka selle välja.
    planned_set = {int(f) for f in fields}
    for field_no, r in by_field.items():
        if field_no in planned_set:
            continue
        b = _n(r.get("b"))
        c = _n(r.get("c"))
        table.append({
            "Põld": field_no,
            "A": _n(r.get("a")),
            "B": b,
            "C": c,
            "XL": _n(r.get("xl")),
            "Kokku": _n(r.get("total")),
            "C/B": round(c / b, 2) if b > 0 else None,
        })

    return pd.DataFrame(table), missing_rows


def _render_day_block(day_label, rows, show_quality=False, planned_fields=None):
    s = _daily_summary(rows)
    cb_text = "—" if s["cb"] is None else _fmt(s["cb"], 2)
    st.markdown(f"### {day_label}\u2003\u2003{_fmt(s['total'])} kasti")
    st.caption(
        f"A {_fmt(s['a'])} · B {_fmt(s['b'])} · C {_fmt(s['c'])} · "
        f"XL {_fmt(s['xl'])} · C/B {cb_text} · {s['count']}/3 põldu"
    )

    df, missing_rows = _field_table(rows, planned_fields=planned_fields)

    formatters = {
        "A": lambda v: _format_field_value(v, 1),
        "B": lambda v: _format_field_value(v, 1),
        "C": lambda v: _format_field_value(v, 1),
        "XL": lambda v: _format_field_value(v, 1),
        "Kokku": lambda v: _format_field_value(v, 1),
        "C/B": lambda v: _format_field_value(v, 2),
    }
    styled = df.style.format(formatters)

    if missing_rows:
        def _highlight_missing(row):
            if row.name in missing_rows:
                return ["background-color: rgba(255, 193, 7, 0.28)" for _ in row]
            return ["" for _ in row]
        styled = styled.apply(_highlight_missing, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("Kollane = tänase põllu korjeandmed on veel puudu.")
    else:
        st.dataframe(styled, use_container_width=True, hide_index=True)

    if show_quality:
        notes = []
        for r in rows:
            quality = r.get("data_quality") or ""
            note = r.get("note") or ""
            if quality or note:
                notes.append({"Põld": r.get("field_no"), "Andmekvaliteet": quality, "Märkus": note})
        if notes:
            with st.expander("Andmekvaliteet ja märkused"):
                st.dataframe(pd.DataFrame(notes), use_container_width=True, hide_index=True)


# Ilm kontrollitakse automaatselt üks kord päevas. Viga ei takista korjete kasutamist.
# Kui päris ilmaandmete seis muutus, märgime mudeli uuendamist vajavaks, kuid
# EI käivita siin rasket mudeliarvutust.
try:
    _weather_refresh_before = db.get_app_setting("weather_last_refresh_at", "")
    WeatherService().auto_refresh_if_needed(TODAY)
    _weather_refresh_after = db.get_app_setting("weather_last_refresh_at", "")
    if _weather_refresh_after and _weather_refresh_after != _weather_refresh_before:
        _mark_model_dirty("uus või uuendatud ilm")
except Exception as exc:
    db.set_app_setting("weather_last_error", f"Automaatne ilmauuendus: {exc}")

st.title("KurgiMootor V6.4")
st.caption("Saagi ennustamise tööriist. Avaleht on töövoog, mitte ilmarakendus.")

page = st.radio(
    "Vaade",
    ["Täna", "Korjed", "Ilm", "Prognoos", "Mootori tähelepanekud"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_page",
)

if page == "Täna":
    today_rows = db.get_harvest_for_day(TODAY)
    harvest_history_for_plan = db.get_harvest_history()
    today_planned_fields = _planned_fields_for_day(TODAY, today_rows, harvest_history_for_plan)

    # Päeva vahetudes lähtestame tänase plaani automaatselt.
    current_home_day = TODAY.isoformat()
    if st.session_state.get("home_plan_day") != current_home_day:
        st.session_state["home_plan_day"] = current_home_day
        st.session_state["home_today_fields"] = list(today_planned_fields)

    # Turvavõrk: kui sessionis on tühi/puuduv valik ilma kasutaja teadliku muutmiseta,
    # alustame tänase automaatse plaaniga.
    if "home_today_fields" not in st.session_state:
        st.session_state["home_today_fields"] = list(today_planned_fields)

    selected_today_fields = st.multiselect(
        "Täna korjatavad põllud",
        options=list(range(1, 15)),
        key="home_today_fields",
        max_selections=4,
        help="Muuda ainult tänase tööplaani. Saagi sisestamine käib Korjed-menüüs.",
    )

    if selected_today_fields:
        if len(selected_today_fields) >= 2:
            fields_text = ", ".join(str(f) for f in selected_today_fields[:-1]) + f" ja {selected_today_fields[-1]}"
        else:
            fields_text = str(selected_today_fields[0])
        plan_text = f"Täna korjatakse põllud nr {fields_text}"
    else:
        plan_text = "Täna korjet ei ole"

    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">
          <span style="font-size:2.0rem;font-weight:950;opacity:.35;">{_weekday_letter(TODAY)}</span>
          <span style="font-size:2.15rem;font-weight:900;line-height:1.05;">{_short_date(TODAY)}</span>
        </div>
        <div style="font-size:1.10rem;font-weight:650;opacity:.82;margin-bottom:10px;">{plan_text}</div>
        """,
        unsafe_allow_html=True,
    )

    today_top_left, today_top_right = st.columns(2, gap="large")
    with today_top_left:
        home_today_actual_slot = st.empty()
    with today_top_right:
        home_today_forecast_slot = st.empty()

    st.markdown("#### Tänased põllud")
    today_by_field = {int(r.get("field_no")): r for r in today_rows if r.get("field_no") is not None}

    if not selected_today_fields:
        st.info("Täna korjet ei ole.")
    else:
        for field_no in selected_today_fields:
            row = today_by_field.get(int(field_no))
            if row:
                b = _n(row.get("b"))
                c = _n(row.get("c"))
                cb = c / b if b > 0 else None
                cb_text = "—" if cb is None else _fmt(cb, 2)
                total = _n(row.get("total"))
                st.markdown(
                    f"""
                    <div style="background:rgba(40,167,69,.10);border:1px solid rgba(40,167,69,.38);
                                border-radius:10px;padding:9px 11px;margin:5px 0;">
                      <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">
                        <strong>Põld {int(field_no)}</strong>
                        <strong>{_fmt(total)} kasti</strong>
                      </div>
                      <div style="font-size:.92rem;opacity:.78;margin-top:3px;">
                        A {_fmt(_n(row.get('a')))} · B {_fmt(b)} · C {_fmt(c)} · XL {_fmt(_n(row.get('xl')))} · C/B {cb_text}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:#fff3cd;border:1px solid #ffe69c;
                                border-radius:10px;padding:9px 11px;margin:5px 0;">
                      <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">
                        <strong>Põld {int(field_no)}</strong>
                        <strong>korje sisestamata</strong>
                      </div>
                      <div style="font-size:.92rem;opacity:.72;margin-top:3px;">
                        A — · B — · C — · XL — · C/B —
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("#### Mootori info")
    home_engine_info_slot = st.empty()

    st.divider()
    st.markdown("#### Järgmised päevad")
    home_future_forecast_slot = st.empty()

    # Avaleht EI käivita rasket walk-forward arvutust.
    # Vana visuaalne Avaleht säilib, kuid prognoosid ja täpsus loetakse
    # ainult salvestatud yield_forecasts snapshotidest.
    try:
        _home_saved = db.get_yield_forecasts(limit=5000) if db.yield_forecasts_available() else []
    except db.DatabaseError:
        _home_saved = []

    def _home_snapshot_map(target_day_value):
        """target_date kõik snapshotid: forecast_date -> field_no -> viimane rida."""
        picked = {}
        for r in _home_saved:
            if str(r.get("target_date")) != target_day_value.isoformat():
                continue
            fdate = str(r.get("forecast_date") or "")
            try:
                fno = int(r.get("field_no"))
            except (TypeError, ValueError):
                continue
            key = (fdate, fno)
            old = picked.get(key)
            if old is None or str(r.get("generated_at") or "") > str(old.get("generated_at") or ""):
                picked[key] = r
        by_date = {}
        for (fdate, fno), row in picked.items():
            by_date.setdefault(fdate, {})[fno] = row
        return by_date

    def _latest_snapshot_rows(target_day_value):
        by_date = _home_snapshot_map(target_day_value)
        if not by_date:
            return []
        latest_date = sorted(by_date.keys())[-1]
        return list(by_date[latest_date].values())

    def _snapshot_totals(rows_day):
        if not rows_day:
            return None
        try:
            total = sum(float(r.get("total_forecast")) for r in rows_day)
            abc = sum(float(r.get("abc_forecast")) for r in rows_day)
            xl = sum(float(r.get("xl_forecast")) for r in rows_day)
        except (TypeError, ValueError):
            return None
        cb_vals = []
        for r in rows_day:
            try:
                if r.get("cb_forecast") is not None:
                    cb_vals.append(float(r.get("cb_forecast")))
            except (TypeError, ValueError):
                pass
        cb = float(np.mean(cb_vals)) if cb_vals else None
        return {"total": total, "abc": abc, "xl": xl, "cb": cb}

    def _home_forecast_adjustment(target_day_value, current_rows):
        """Eelmine täielik snapshot vs hetkel kuvatav viimane snapshot."""
        current = _snapshot_totals(current_rows)
        if not current or current["total"] <= 0:
            return None
        expected = {int(r.get("field_no")) for r in current_rows if r.get("field_no") is not None}
        by_date = _home_snapshot_map(target_day_value)
        complete = []
        for fdate, rows_for_date in by_date.items():
            if set(rows_for_date.keys()) != expected:
                continue
            try:
                total = sum(float(rows_for_date[f]["total_forecast"]) for f in expected)
            except (TypeError, ValueError, KeyError):
                continue
            complete.append((fdate, total))
        if len(complete) < 2:
            return None
        complete.sort(key=lambda x: x[0])
        previous_total = complete[-2][1]
        if previous_total <= 0:
            return None
        return (current["total"] / previous_total - 1.0) * 100.0

    def _home_motor_accuracy_3p():
        """Vana 3P loogika, kuid ainult DB snapshotidest; mudelit ei treenita."""
        harvest_rows_home = db.get_harvest_history(limit=1000)
        actual_by_day = {}
        for hr in harvest_rows_home:
            try:
                d = date.fromisoformat(str(hr.get("harvest_date") or ""))
                fno = int(hr.get("field_no"))
                total = float(hr.get("total"))
            except (TypeError, ValueError):
                continue
            if d >= TODAY:
                continue
            actual_by_day.setdefault(d, {})[fno] = total

        complete_days = sorted(
            [d for d, rows_for_day in actual_by_day.items()
             if len(rows_for_day) == 3 and len(set(rows_for_day.keys())) == 3],
            reverse=True,
        )

        evaluated = []
        for d in complete_days:
            expected = set(actual_by_day[d].keys())
            actual_total = sum(actual_by_day[d].values())
            if actual_total <= 0:
                continue
            by_date = _home_snapshot_map(d)
            candidates = []
            for fdate, rows_for_date in by_date.items():
                if not fdate or fdate > d.isoformat():
                    continue
                if set(rows_for_date.keys()) != expected:
                    continue
                try:
                    forecast_total = sum(float(rows_for_date[f]["total_forecast"]) for f in expected)
                except (TypeError, ValueError, KeyError):
                    continue
                candidates.append((fdate, forecast_total))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            fdate, forecast_total = candidates[-1]
            err = abs(forecast_total / actual_total - 1.0) * 100.0
            evaluated.append({"day": d, "abs_pct_error": err})
            if len(evaluated) == 3:
                break

        if len(evaluated) < 3:
            return None, evaluated
        mean_err = float(np.mean([r["abs_pct_error"] for r in evaluated]))
        return max(0.0, min(100.0, 100.0 - mean_err)), evaluated

    # Tänane tegelik kaart – vana disain.
    actual_sum = _daily_summary(today_rows) if today_rows else {
        "total": 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "xl": 0.0, "cb": None
    }
    actual_cb_text = "—" if actual_sum["cb"] is None else _fmt(actual_sum["cb"], 2)

    with home_today_actual_slot.container():
        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px 14px;">
              <div style="font-size:.80rem;font-weight:900;letter-spacing:.10em;opacity:.58;">TEGELIK</div>
              <div style="font-size:2.15rem;font-weight:900;line-height:1.05;margin-top:3px;">{_fmt(actual_sum['total'])} kasti</div>
              <div style="font-size:.92rem;opacity:.74;margin-top:6px;">
                A {_fmt(actual_sum['a'])} · B {_fmt(actual_sum['b'])} · C {_fmt(actual_sum['c'])} · XL {_fmt(actual_sum['xl'])} · C/B {actual_cb_text}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tänane prognoos – sama visuaal mis enne, aga snapshotist.
    today_saved_rows = _latest_snapshot_rows(TODAY)
    today_saved = _snapshot_totals(today_saved_rows)
    with home_today_forecast_slot.container():
        if today_saved:
            adj_pct = _home_forecast_adjustment(TODAY, today_saved_rows)
            adj_text = "muutus —" if adj_pct is None else (
                f"{'↑' if adj_pct > 0 else '↓' if adj_pct < 0 else '→'} {adj_pct:+.0f}%"
            )
            cb_text = "—" if today_saved["cb"] is None else _fmt(today_saved["cb"], 2)
            st.markdown(
                f"""
                <div style="border:2px solid rgba(76,160,92,.38);border-radius:12px;padding:12px 14px;">
                  <div style="font-size:.80rem;font-weight:900;letter-spacing:.10em;opacity:.58;">PROGNOOS</div>
                  <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-top:3px;">
                    <div style="font-size:2.15rem;font-weight:950;line-height:1.05;">{_fmt(today_saved['total'])} kasti</div>
                    <div style="font-size:1.02rem;font-weight:800;opacity:.76;">{adj_text}</div>
                  </div>
                  <div style="font-size:.92rem;opacity:.76;margin-top:6px;">
                    A+B+C {_fmt(today_saved['abc'])} · XL {_fmt(today_saved['xl'])} · C/B ~{cb_text}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Tänast salvestatud prognoosi pole veel.")

    # Mootori info ja 3P kaart – vana disain.
    basis_rows = today_saved_rows
    if not basis_rows:
        for lead in range(1, 10):
            basis_rows = _latest_snapshot_rows(TODAY + timedelta(days=lead))
            if basis_rows:
                break
    basis = str(basis_rows[0].get("basis") or "") if basis_rows else ""
    abc_name = "weather-first"
    xl_name = "weather-first"
    cb_name = "weather-first baas"

    m = re.search(r"champion=([^;]+)", basis)
    if m:
        abc_name = m.group(1).strip()
    m = re.search(r"xl_champion=([^;]+)", basis)
    if m:
        xl_name = m.group(1).strip()
    m = re.search(r"cb_champion=([^;]+)", basis)
    if m:
        cb_name = m.group(1).strip()

    # Vanade snapshotide puhul asenda umbmäärane nimi selge hinnangutekstiga.
    if cb_name.lower() in {"c/b baasmudel", "baasmudel", "weather-first"}:
        cb_name = "weather-first baas"

    motor_accuracy_3p, motor_accuracy_days = _home_motor_accuracy_3p()
    with home_engine_info_slot.container():
        if motor_accuracy_3p is None:
            accuracy_value = "—"
            accuracy_sub = f"{len(motor_accuracy_days)}/3 päeva"
        else:
            accuracy_value = f"{motor_accuracy_3p:.0f}%"
            accuracy_sub = "viimased 3 täielikku korjepäeva"

        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,.24);border-radius:11px;
                        padding:10px 13px;margin:2px 0 7px 0;">
              <div style="font-size:.80rem;font-weight:900;letter-spacing:.08em;opacity:.60;">
                MOOTORI TÄPSUS · 3P
              </div>
              <div style="font-size:1.85rem;font-weight:950;line-height:1.05;margin-top:3px;">
                {accuracy_value}
              </div>
              <div style="font-size:.80rem;opacity:.62;margin-top:3px;">
                {accuracy_sub}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"A+B+C mootor: {abc_name} · XL mootor: {xl_name} · "
            f"C/B hinnang: {cb_name}"
        )

    # Järgmised päevad – vana kaardidisain, andmed ainult snapshotidest.
    with home_future_forecast_slot.container():
        shown = 0
        for lead in range(1, 10):
            target_day = TODAY + timedelta(days=lead)
            rows_day = _latest_snapshot_rows(target_day)
            totals = _snapshot_totals(rows_day)
            if not totals:
                continue

            cb_text = "—" if totals["cb"] is None else _fmt(totals["cb"], 2)
            adj_pct = _home_forecast_adjustment(target_day, rows_day)
            adj_text = "muutus —" if adj_pct is None else (
                f"{'↑' if adj_pct > 0 else '↓' if adj_pct < 0 else '→'} {adj_pct:+.0f}%"
            )
            trend = lead >= 6
            bg = "#fff3cd" if trend else "rgba(0,0,0,0.025)"
            border = "#ffe69c" if trend else "rgba(128,128,128,0.20)"
            badge = f" · 🟡 trend {lead} p" if trend else ""

            st.markdown(
                f"""
                <div style="background:{bg};border:1px solid {border};border-radius:10px;
                            padding:10px 12px;margin:7px 0;">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-size:1.65rem;font-weight:950;opacity:.38;">{_weekday_letter(target_day)}</span>
                      <strong style="font-size:1.05rem;">{_short_date(target_day)}</strong>
                    </div>
                    <strong style="font-size:1.15rem;">{_fmt(totals['total'])} kasti · {adj_text}{badge}</strong>
                  </div>
                  <div style="font-size:0.90rem;opacity:0.78;margin-top:3px;">
                    A+B+C {_fmt(totals['abc'])} · XL {_fmt(totals['xl'])} · C/B ~{cb_text}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            shown += 1

        if shown == 0:
            st.caption("Järgmiste päevade salvestatud prognoosi pole veel.")

elif page == "Korjed":
    st.subheader("Korjed")
    st.caption("Andmebaasis hoitakse iga põllu korje eraldi. Äpis vaatame saaki eelkõige päevade kaupa.")

    # Järgmise korje vaikimisi valikud tuletatakse esmalt päris tänastest andmetest.
    # Nii avaneb äpp pärast refreshi/redeploy'd kohe järgmise puuduva põllu ja järjekorraga,
    # mitte ei kuku tagasi 1 / 1 peale. Session state jääb siiski sama sessiooni jooksul
    # pärast salvestust esmaseks, et järgmisele reale liikumine oleks kohene.
    today_rows_for_form = db.get_harvest_for_day(TODAY)
    history_for_form = db.get_harvest_history()
    planned_for_form = _planned_fields_for_day(TODAY, today_rows_for_form, history_for_form)

    used_orders = {
        int(r.get("harvest_order"))
        for r in today_rows_for_form
        if r.get("harvest_order") is not None and int(r.get("harvest_order")) in (1, 2, 3)
    }
    missing_orders = [order for order in (1, 2, 3) if order not in used_orders]

    if missing_orders:
        inferred_order = missing_orders[0]
        inferred_field = int(planned_for_form[inferred_order - 1])
    elif today_rows_for_form:
        # Päev on 3/3 valmis. Kui kasutaja siiski lisab/parandab uut rida,
        # alustame järgmisest põllust ja järjekorrast 1.
        ordered_complete = sorted(today_rows_for_form, key=lambda r: int(r.get("harvest_order") or 99))
        inferred_field = _next_field(int(ordered_complete[-1].get("field_no")))
        inferred_order = 1
    else:
        inferred_field = int(planned_for_form[0])
        inferred_order = 1

    default_field = int(st.session_state.get("next_harvest_field", inferred_field))
    default_order = int(st.session_state.get("next_harvest_order", inferred_order))
    form_version = int(st.session_state.get("harvest_form_version", 0))

    st.markdown("#### Lisa või paranda korje")

    if st.session_state.get("harvest_saved_message"):
        st.success(st.session_state.pop("harvest_saved_message"))

    # Kuupäev ja põld valitakse enne vormi. Nii saab äpp kohe kontrollida,
    # kas sama kuupäeva + põllu kirje on juba olemas, ning laadida olemasolevad
    # väärtused parandamiseks nähtavale.
    c1, c2, c3 = st.columns(3)
    entry_date = c1.date_input("Kuupäev", value=TODAY, key="manual_harvest_date")
    entry_field = c2.selectbox(
        "Põld",
        list(range(1, 15)),
        index=max(0, min(13, default_field - 1)),
        key=f"manual_harvest_field_{form_version}",
    )

    selected_day_rows = db.get_harvest_for_day(entry_date)
    existing_row = next(
        (
            r for r in selected_day_rows
            if int(r.get("field_no") or r.get("field_id") or 0) == int(entry_field)
        ),
        None,
    )

    existing_order = (
        int(existing_row.get("harvest_order") or default_order)
        if existing_row else default_order
    )
    entry_order = c3.selectbox(
        "Järjekord",
        [1, 2, 3],
        index=max(0, min(2, existing_order - 1)),
        key=f"manual_harvest_order_{form_version}",
    )

    if existing_row:
        st.warning(
            f"⚠️ {entry_date} · põld {entry_field}: korje on juba olemas. "
            "Allolevad väärtused on praegu andmebaasis. Muutmine või kustutamine "
            "vajab eraldi kinnitust."
        )

    with st.form("manual_harvest_form"):
        q1, q2, q3, q4 = st.columns(4)
        entry_a_raw = q1.number_input(
            "A", min_value=0.0,
            value=float(existing_row.get("a") or 0.0) if existing_row else None,
            step=0.1, format="%.1f", placeholder="0,0",
            key=f"manual_a_{form_version}_{entry_date.isoformat()}_{entry_field}"
        )
        entry_b_raw = q2.number_input(
            "B", min_value=0.0,
            value=float(existing_row.get("b") or 0.0) if existing_row else None,
            step=0.1, format="%.1f", placeholder="0,0",
            key=f"manual_b_{form_version}_{entry_date.isoformat()}_{entry_field}"
        )
        entry_c_raw = q3.number_input(
            "C", min_value=0.0,
            value=float(existing_row.get("c") or 0.0) if existing_row else None,
            step=0.1, format="%.1f", placeholder="0,0",
            key=f"manual_c_{form_version}_{entry_date.isoformat()}_{entry_field}"
        )
        entry_xl_raw = q4.number_input(
            "XL", min_value=0.0,
            value=float(existing_row.get("xl") or 0.0) if existing_row else None,
            step=0.1, format="%.1f", placeholder="0,0",
            key=f"manual_xl_{form_version}_{entry_date.isoformat()}_{entry_field}"
        )

        entry_a = float(entry_a_raw or 0.0)
        entry_b = float(entry_b_raw or 0.0)
        entry_c = float(entry_c_raw or 0.0)
        entry_xl = float(entry_xl_raw or 0.0)

        total_preview = entry_a + entry_b + entry_c + entry_xl
        cb_preview = entry_c / entry_b if entry_b > 0 else None
        preview_text = f"Kokku: {_fmt(total_preview)}"
        if cb_preview is not None:
            preview_text += f" · C/B: {_fmt(cb_preview, 2)}"
        st.caption(preview_text)

        change_confirmed = True
        if existing_row:
            change_confirmed = st.checkbox(
                "Kinnitan olemasoleva korje muutmise või kustutamise",
                value=False,
            )
            st.caption(
                "Kui A, B, C ja XL on kõik 0, kustutatakse see korjerida andmebaasist."
            )

        submitted = st.form_submit_button(
            "Kustuta korjerida" if existing_row and total_preview <= 0 else "Salvesta korje"
        )

        if submitted:
            if existing_row and not change_confirmed:
                st.warning("Olemasoleva korje muutmiseks või kustutamiseks märgi kinnitus.")
            elif existing_row and total_preview <= 0:
                db.delete_harvest(entry_date, entry_field)
                _mark_model_dirty(f"korje kustutatud {entry_date} põld {entry_field}")
                st.session_state["harvest_form_version"] = form_version + 1
                st.session_state["harvest_saved_message"] = (
                    f"Kustutatud: {entry_date} · põld {entry_field}"
                )
                st.rerun()
            elif total_preview <= 0:
                st.warning("Uut 0-korjet ei salvestata. Sisesta vähemalt üks kogus.")
            else:
                db.save_harvest(
                    entry_date,
                    entry_field,
                    0,
                    entry_a,
                    entry_b,
                    entry_c,
                    entry_xl,
                    harvest_order=entry_order,
                )

                # Esimene ja teine korje ainult salvestuvad. Raske mudeliring muutub
                # vajalikuks alles siis, kui päev on terviklik 3/3.
                _saved_day_rows = db.get_harvest_for_day(entry_date)
                _saved_day_fields = {
                    int(r.get("field_no"))
                    for r in _saved_day_rows
                    if r.get("field_no") is not None
                }
                if len(_saved_day_rows) == 3 and len(_saved_day_fields) == 3:
                    _mark_model_dirty(f"täielik 3/3 korjepäev {entry_date}")

                st.session_state["next_harvest_field"] = 1 if entry_field >= 14 else entry_field + 1
                st.session_state["next_harvest_order"] = 1 if entry_order >= 3 else entry_order + 1
                st.session_state["harvest_form_version"] = form_version + 1
                action = "Muudetud" if existing_row else "Salvestatud"
                st.session_state["harvest_saved_message"] = (
                    f"{action}: {entry_date} · põld {entry_field} · kokku {_fmt(total_preview)}"
                )
                st.rerun()

    st.divider()
    st.markdown("#### Korjeajalugu päevade kaupa")
    rows = db.get_harvest_history()
    if rows:
        by_day = {}
        for row in rows:
            by_day.setdefault(str(row.get("harvest_date")), []).append(row)

        for day_str, day_rows in by_day.items():
            try:
                day_date = date.fromisoformat(day_str)
                day_label = f"{_weekday_letter(day_date)} {_short_date(day_date)}"
            except ValueError:
                day_date = None
                day_label = day_str

            planned_fields = None
            if day_date == TODAY:
                planned_fields = _planned_fields_for_day(TODAY, day_rows, rows)

            _render_day_block(day_label, day_rows, show_quality=True, planned_fields=planned_fields)
            st.divider()

        with st.expander("Näita kõiki korjeridu ühe tabelina"):
            df = pd.DataFrame(rows).rename(columns={
                "harvest_date": "Kuupäev",
                "field_no": "Põld",
                "harvest_order": "Järjekord",
                "interval_days": "Intervall",
                "a": "A",
                "b": "B",
                "c": "C",
                "xl": "XL",
                "total": "Kokku",
                "data_quality": "Andmekvaliteet",
                "note": "Märkus",
            })
            if not df.empty and "B" in df.columns and "C" in df.columns:
                df["C/B"] = df.apply(lambda r: round(_n(r["C"]) / _n(r["B"]), 2) if _n(r["B"]) > 0 else None, axis=1)
            all_rows_formatters = {
                "A": lambda v: _format_field_value(v, 1),
                "B": lambda v: _format_field_value(v, 1),
                "C": lambda v: _format_field_value(v, 1),
                "XL": lambda v: _format_field_value(v, 1),
                "Kokku": lambda v: _format_field_value(v, 1),
                "C/B": lambda v: _format_field_value(v, 2),
            }
            st.dataframe(df.style.format(all_rows_formatters), use_container_width=True, hide_index=True)
    else:
        st.info("Korjeid pole veel sisestatud.")

elif page == "Ilm":
    st.subheader("Ilmaklots")
    st.caption("Mõõdetud temperatuur, tuul, globaalradiatsioon, õhuniiskus ja sademed Pärnu jaamast. ET0 arvutatakse automaatselt.")
    counts = db.get_weather_counts()
    a, b, c = st.columns(3)
    a.metric("Mõõdetud päevi", counts["measured"])
    b.metric("Rohelisi päevi", counts["checked"])
    c.metric("Prognoosipäevi", counts["forecast"])
    st.caption(f"Viimane automaatne uuendus: {db.get_app_setting('weather_last_refresh_at', '—')}")
    last_result = db.get_app_setting("weather_last_result", "")
    if last_result:
        st.caption(last_result)
    error = db.get_app_setting("weather_last_error", "")
    if error:
        st.error(error)
    if st.button("Uuenda ilm kohe"):
        with st.spinner("Laen mõõteandmeid ja 9 päeva prognoosi..."):
            result = WeatherService().safe_refresh_all(TODAY)
        _mark_model_dirty("käsitsi uuendatud ilm")
        if result.get("error"):
            st.warning(f"Uuendus tehti osaliselt: {result['error']}")
        else:
            st.success("Ilmaandmed uuendatud.")
        st.rerun()

    history_default_start = SEASON_START
    history_default_end = TODAY
    st.subheader("Mõõdetud ilma ajalugu")
    d1, d2 = st.columns(2)
    history_start = d1.date_input(
        "Alguskuupäev",
        value=history_default_start,
        max_value=TODAY,
        key="weather_history_start",
    )
    history_end = d2.date_input(
        "Lõppkuupäev",
        value=history_default_end,
        max_value=TODAY,
        key="weather_history_end",
    )
    if history_start > history_end:
        st.warning("Alguskuupäev peab olema lõppkuupäevast varasem.")
        measured_rows = []
    else:
        history_rows = db.get_weather_rows(history_start, history_end)
        measured_rows = [r for r in history_rows if r.get("data_kind") == "measured"][::-1]

    forecast_rows = [
        r for r in db.get_weather_rows(TODAY, TODAY + timedelta(days=8))
        if r.get("data_kind") == "forecast"
    ]

    st.caption(
        f"Kuvatakse {len(measured_rows)} päeva. 🟢 = täielikult mõõdetud/kontrollitud; "
        "🔴 = mõni vajalik komponent on puudu või ajutine. "
        "Puuduvat mõõdetud ilma ei täideta enam varasemate päevade keskmisega."
    )

    weather_columns = [
        "temp_night_avg_c", "temp_day_avg_c", "temp_min_c", "temp_max_c",
        "wind_avg_ms", "humidity_avg_pct",
        "precipitation_mm", "et0_mm", "radiation_mj_m2",
    ]

    # Vajame fallbackiks ka kuni 3 päeva enne valitud ajaloo algust.
    fallback_start = history_start - timedelta(days=3)
    fallback_rows = db.get_weather_rows(fallback_start, history_end)
    fallback_by_day = {
        str(r.get("weather_date")): r
        for r in fallback_rows
        if r.get("data_kind") == "measured"
    }

    def _history_effective_value(day_value, feature):
        """Mõõdetud ajaloo väärtust ei leiutata teistest päevadest juurde."""
        row = fallback_by_day.get(day_value.isoformat()) or {}
        raw = row.get(feature)
        try:
            if raw is not None:
                return float(raw), False
        except (TypeError, ValueError):
            pass
        return None, False

    measured_display = []
    problem_display_cells = set()

    display_col_by_feature = {
        "temp_night_avg_c": "Öö kesk °C",
        "temp_day_avg_c": "Päev kesk °C",
        "temp_min_c": "Min °C",
        "temp_max_c": "Max °C",
        "wind_avg_ms": "Tuul m/s",
        "humidity_avg_pct": "Niiskus %",
        "precipitation_mm": "Sademed mm",
        "et0_mm": "ET0 mm",
        "radiation_mj_m2": "Radiatsioon MJ/m²",
    }
    problem_name_to_feature = {
        "öö": "temp_night_avg_c",
        "päev": "temp_day_avg_c",
        "tmin": "temp_min_c",
        "tmax": "temp_max_c",
        "tuul": "wind_avg_ms",
        "niiskus": "humidity_avg_pct",
        "rh": "humidity_avg_pct",
        "sademed": "precipitation_mm",
        "et0": "et0_mm",
        "radiatsioon": "radiation_mj_m2",
    }

    for idx, r in enumerate(measured_rows):
        try:
            row_day = date.fromisoformat(str(r.get("weather_date")))
        except ValueError:
            row_day = None

        effective = {}
        for feature in weather_columns:
            if row_day is None:
                effective[feature] = r.get(feature)
            else:
                value, _ = _history_effective_value(row_day, feature)
                effective[feature] = value

        check_message = str(r.get("check_message") or "")
        msg_lower = check_message.lower()

        # Puuduv väärtus: värvi täpselt selle välja lahter roosaks.
        for feature, col_name in display_col_by_feature.items():
            if effective.get(feature) is None:
                problem_display_cells.add((idx, col_name))

        # Ajutine päev: check_message ütleb, millised komponendid tulid veel
        # varem salvestatud prognoosist. Värvime ainult need komponendid.
        temporary_features = set()
        if "ajutine" in msg_lower or "varem salvestatud prognoos" in msg_lower:
            for name, feature in problem_name_to_feature.items():
                if name in msg_lower:
                    temporary_features.add(feature)

            # Kui radiatsioon või RH on ajutine, on ka neist arvutatud ET0 ajutine.
            if temporary_features.intersection(
                {"radiation_mj_m2", "humidity_avg_pct", "wind_avg_ms", "temp_min_c", "temp_max_c"}
            ):
                temporary_features.add("et0_mm")

        for feature in temporary_features:
            col_name = display_col_by_feature.get(feature)
            if col_name:
                problem_display_cells.add((idx, col_name))

        if r.get("checked"):
            status = "🟢 Kontrollitud"
        elif "ajutine" in msg_lower:
            status = "Ajutine"
        else:
            status = "Puudulik"

        measured_display.append({
            "Kuupäev": r["weather_date"],
            "Öö kesk °C": effective.get("temp_night_avg_c"),
            "Päev kesk °C": effective.get("temp_day_avg_c"),
            "Min °C": effective.get("temp_min_c"),
            "Max °C": effective.get("temp_max_c"),
            "Tuul m/s": effective.get("wind_avg_ms"),
            "Niiskus %": effective.get("humidity_avg_pct"),
            "Sademed mm": effective.get("precipitation_mm"),
            "ET0 mm": effective.get("et0_mm"),
            "Radiatsioon MJ/m²": effective.get("radiation_mj_m2"),
            "Allikas": r.get("source_station"),
            "Kontroll": check_message,
            "Olek": status,
        })

    measured_df = pd.DataFrame(measured_display)
    if not measured_df.empty:
        styled_weather = measured_df.style

        def _highlight_problem_cells(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            for row_idx, col_name in problem_display_cells:
                if row_idx in styles.index and col_name in styles.columns:
                    styles.loc[row_idx, col_name] = (
                        "background-color: rgba(255, 182, 193, 0.48);"
                    )
            return styles

        styled_weather = styled_weather.apply(_highlight_problem_cells, axis=None)

        # Ainult kuvamisvorming; arvutustes jäävad täpsed väärtused alles.
        styled_weather = styled_weather.format({
            "Öö kesk °C": "{:.1f}",
            "Päev kesk °C": "{:.1f}",
            "Min °C": "{:.1f}",
            "Max °C": "{:.1f}",
            "Tuul m/s": "{:.2f}",
            "Niiskus %": "{:.0f}",
            "Sademed mm": "{:.1f}",
            "ET0 mm": "{:.2f}",
            "Radiatsioon MJ/m²": "{:.2f}",
        }, na_rep="—")

        st.dataframe(styled_weather, use_container_width=True, hide_index=True)
    else:
        st.dataframe(measured_df, use_container_width=True, hide_index=True)

    st.subheader("9 päeva prognoos")
    forecast_display = [{
        "Kuupäev": r["weather_date"],
        "Öö kesk °C": r.get("temp_night_avg_c"),
        "Päev kesk °C": r.get("temp_day_avg_c"),
        "Min °C": r.get("temp_min_c"),
        "Max °C": r.get("temp_max_c"),
        "Tuul m/s": r.get("wind_avg_ms"),
        "Niiskus %": r.get("humidity_avg_pct"),
        "Sademed mm": r.get("precipitation_mm"),
        "ET0 mm": r.get("et0_mm"),
        "Radiatsioon MJ/m²": r.get("radiation_mj_m2"),
        "Kontroll": r.get("check_message"),
        "Olek": "🔵 Prognoos" if r.get("checked") else "🔴 Vigane prognoos",
    } for r in forecast_rows]
    st.dataframe(pd.DataFrame(forecast_display), use_container_width=True, hide_index=True)

if page in ("Prognoos", "Mootori tähelepanekud"):
    _forecast_page_placeholder = st.empty()
    with _forecast_page_placeholder.container():
        st.subheader("Prognoos")
        st.markdown("#### Andmete valmisolek")
        st.caption(
            "Kontrollime, kas ajaloolistest korjetest saab moodustada päris õppimisnäited: "
            "sama põllu kahe järjestikuse korje vahel peab olema täielik mõõdetud ilm. "
            "Tänane pooleliolev korjepäev ei blokeeri õppimist."
        )

        readiness_start = SEASON_START
        harvest_rows = db.get_harvest_history(limit=2000)

        # Korjepäevad ja viimane täielik 3/3 päev. Pooleliolevat tänast päeva õppesse ei võeta.
        harvest_by_day = {}
        for row in harvest_rows:
            day_str = str(row.get("harvest_date") or "")
            if day_str:
                harvest_by_day.setdefault(day_str, []).append(row)

        complete_harvest_days = []
        for day_str, day_rows in harvest_by_day.items():
            fields = {int(r.get("field_no")) for r in day_rows if r.get("field_no") is not None}
            if len(day_rows) == 3 and len(fields) == 3:
                try:
                    complete_harvest_days.append(date.fromisoformat(day_str))
                except ValueError:
                    pass

        last_complete_harvest = max(complete_harvest_days) if complete_harvest_days else None
        latest_harvest = None
        valid_days = []
        for day_str in harvest_by_day:
            try:
                valid_days.append(date.fromisoformat(day_str))
            except ValueError:
                pass
        if valid_days:
            latest_harvest = max(valid_days)

        # Korjeridade baaskvaliteet.
        harvest_problems = []
        estimated_harvest_rows = 0
        seen_keys = set()
        represented_fields = set()
        parsed_rows = []
        for row in harvest_rows:
            day_str = str(row.get("harvest_date") or "")
            field_no = row.get("field_no")
            try:
                field_no_int = int(field_no)
            except (TypeError, ValueError):
                field_no_int = None
            try:
                harvest_day = date.fromisoformat(day_str) if day_str else None
            except ValueError:
                harvest_day = None

            if field_no_int is not None:
                represented_fields.add(field_no_int)

            key = (day_str, field_no_int)
            if key in seen_keys:
                harvest_problems.append(f"Duplikaat: {day_str}, põld {field_no_int}")
            seen_keys.add(key)

            if harvest_day is None:
                harvest_problems.append(f"Vigane või puuduv korjekuupäev: {day_str or '—'}")
            if field_no_int is None or not 1 <= field_no_int <= 14:
                harvest_problems.append(f"Vigane põld: {field_no}")
            quality = str(row.get("data_quality") or "").strip().lower()
            is_estimated = quality in {"hinnanguline", "ligikaudne"}
            if is_estimated:
                estimated_harvest_rows += 1
            else:
                for field_name in ("a", "b", "c", "xl", "total"):
                    if row.get(field_name) is None:
                        harvest_problems.append(f"{day_str} põld {field_no}: {field_name.upper()} puudub")

            if harvest_day is not None and field_no_int is not None:
                parsed_rows.append((field_no_int, harvest_day, row))

        missing_fields = [f for f in range(1, 15) if f not in represented_fields]

        # Mõõdetud ööpäevailma saab ausalt nõuda ainult lõpetatud kalendripäevadelt.
        # Tänast päeva ei märgita enam puuduvaks isegi siis, kui tänane 3/3 korje on juba sisestatud.
        weather_target_end = min(last_complete_harvest, TODAY - timedelta(days=1)) if last_complete_harvest else None
        weather_missing = []
        weather_rows = []
        weather_by_day = {}
        required_weather = (
            "temp_night_avg_c", "temp_day_avg_c",
            "temp_min_c", "temp_max_c", "wind_avg_ms", "radiation_mj_m2",
            "humidity_avg_pct", "precipitation_mm", "et0_mm",
        )
        if weather_target_end and weather_target_end >= readiness_start:
            weather_missing = db.get_incomplete_measured_dates(readiness_start, weather_target_end)
            weather_rows = db.get_weather_rows(readiness_start, weather_target_end)
            for wr in weather_rows:
                if wr.get("data_kind") == "measured":
                    weather_by_day[str(wr.get("weather_date"))] = wr

        def _weather_day_ok(day_value: date) -> bool:
            wr = weather_by_day.get(day_value.isoformat())
            return bool(
                wr
                and wr.get("checked")
                and all(wr.get(name) is not None for name in required_weather)
            )

        # Päris õppimisnäide = ühe põllu korje, millele eelneb sama põllu varasem korje.
        # Ilmaaken on päev pärast eelmist korjet kuni jooksva korjepäevani (kaasa arvatud).
        rows_by_field = {f: [] for f in range(1, 15)}
        if weather_target_end:
            for field_no, harvest_day, row in parsed_rows:
                if harvest_day <= weather_target_end:
                    rows_by_field[field_no].append((harvest_day, row))
        for field_no in rows_by_field:
            rows_by_field[field_no].sort(key=lambda item: item[0])

        usable_samples = []
        incomplete_samples = []
        first_harvest_rows = []
        for field_no, items in rows_by_field.items():
            for idx, (current_day, row) in enumerate(items):
                if idx == 0:
                    first_harvest_rows.append((field_no, current_day))
                    continue
                previous_day, previous_row = items[idx - 1]
                window_start = previous_day + timedelta(days=1)
                window_end = current_day
                expected_days = max(0, (window_end - window_start).days + 1)
                missing_days = []
                cursor = window_start
                while cursor <= window_end:
                    if not _weather_day_ok(cursor):
                        missing_days.append(cursor)
                    cursor += timedelta(days=1)

                sample = {
                    "field_no": field_no,
                    "previous_day": previous_day,
                    "current_day": current_day,
                    "interval_days": (current_day - previous_day).days,
                    "weather_days": expected_days,
                    "missing_days": missing_days,
                    "current_row": row,
                    "previous_row": previous_row,
                    "previous2_row": items[idx - 2][1] if idx >= 2 else None,
                }
                if missing_days:
                    incomplete_samples.append(sample)
                else:
                    usable_samples.append(sample)

        weather_ready = bool(weather_target_end) and not weather_missing
        harvest_ready = bool(last_complete_harvest) and not harvest_problems and not missing_fields
        sample_ready = bool(usable_samples) and not incomplete_samples
        training_ready = weather_ready and harvest_ready and sample_ready

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Õppimisnäiteid", len(usable_samples))
        m2.metric("Ilmaauguga näiteid", len(incomplete_samples))
        m3.metric("Põlde esindatud", f"{len(represented_fields)}/14")
        m4.metric("Ilma kontroll kuni", weather_target_end.strftime("%d.%m") if weather_target_end else "—")

        if training_ready:
            st.success(
                f"✅ Andmestik on õppimiseks tehniliselt valmis kuni {last_complete_harvest.strftime('%d.%m.%Y')}. "
                f"Täieliku ilmavahemikuga õppimisnäiteid on {len(usable_samples)}."
            )
        else:
            st.warning("Õppimisandmestik ei ole veel täielikult valmis. Allpool on näha, mis piirab õppimisnäiteid.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Ilm")
            if weather_target_end is None or weather_target_end < readiness_start:
                st.info("Mõõdetud ilma kontroll algab pärast esimest lõpetatud korjepäeva.")
            elif weather_ready:
                days_count = (weather_target_end - readiness_start).days + 1
                st.success(
                    f"🟢 Mõõdetud ilm {readiness_start.strftime('%d.%m')}–{weather_target_end.strftime('%d.%m')} on täielik: "
                    f"{days_count}/{days_count} päeva."
                )
            else:
                expected = (weather_target_end - readiness_start).days + 1
                ok_count = max(0, expected - len(weather_missing))
                old_missing = [d for d in weather_missing if d < TODAY - timedelta(days=2)]
                status_text = (
                    f"Mõõdetud ilm {readiness_start.strftime('%d.%m')}–{weather_target_end.strftime('%d.%m')}: "
                    f"{ok_count}/{expected} päeva valmis."
                )
                if old_missing:
                    st.error("🔴 " + status_text)
                else:
                    st.warning("🟡 " + status_text + " Pärnu radiatsiooni/RH/sademete päevaelemendid võivad saabuda viitega.")

                friendly = {
                    "temp_night_avg_c": "öö kesk T", "temp_day_avg_c": "päeva kesk T",
                    "temp_min_c": "Tmin", "temp_max_c": "Tmax", "wind_avg_ms": "tuul",
                    "radiation_mj_m2": "radiatsioon", "humidity_avg_pct": "niiskus",
                    "precipitation_mm": "sademed", "et0_mm": "ET0",
                }
                details = []
                for d in weather_missing:
                    wr = weather_by_day.get(d.isoformat())
                    if not wr:
                        details.append(f"{d.strftime('%d.%m')}: mõõdetud päeva rida puudub")
                        continue
                    missing_parts = [friendly[k] for k in required_weather if wr.get(k) is None]
                    details.append(f"{d.strftime('%d.%m')}: " + ", ".join(missing_parts))
                if details:
                    st.caption("Puudu: " + " | ".join(details[:10]))
                st.caption("„Uuenda ilm kohe” kontrollib puudulikud päevad uuesti; tänast päeva siia enam ei arvestata.")

        with c2:
            st.markdown("##### Korjed")
            if harvest_ready:
                st.success("🟢 Korjeread korras ja kõik 14 põldu on ajaloos esindatud.")
            else:
                if missing_fields:
                    st.warning("Ajaloost puuduvad põllud: " + ", ".join(map(str, missing_fields)))
                if harvest_problems:
                    st.warning(f"Korjeandmetes on {len(harvest_problems)} tehnilist kontrollkohta.")
                    with st.expander("Näita korjeandmete kontrollkohti"):
                        for problem in harvest_problems[:100]:
                            st.write("• " + problem)
            if estimated_harvest_rows:
                st.caption(
                    f"{estimated_harvest_rows} hinnangulist/ligikaudset ajaloorida kasutatakse ajajooneks, "
                    "kuid mitte mudeli sihtreana."
                )

        st.markdown("##### Õppimisnäited põldude kaupa")
        st.caption(
            "Esimene teadaolev korje igal põllul on lähtepunkt, mitte õppimisnäide. "
            "Järgmise korje näide kasutab selle põllu kahe korje vahele jäävat ilma."
        )

        field_summary = []
        for field_no in range(1, 15):
            usable_n = sum(1 for s in usable_samples if s["field_no"] == field_no)
            bad_n = sum(1 for s in incomplete_samples if s["field_no"] == field_no)
            harvest_n = len(rows_by_field.get(field_no, []))
            field_summary.append({
                "Põld": field_no,
                "Korjeid": harvest_n,
                "Õppimisnäiteid": usable_n,
                "Ilmaauguga": bad_n,
            })
        st.dataframe(pd.DataFrame(field_summary), use_container_width=True, hide_index=True)

        if incomplete_samples:
            with st.expander("Näita ilmaauguga õppimisnäiteid"):
                for sample in incomplete_samples[:100]:
                    missing_text = ", ".join(d.strftime("%d.%m") for d in sample["missing_days"])
                    st.write(
                        f"• Põld {sample['field_no']}: {sample['previous_day'].strftime('%d.%m')} → "
                        f"{sample['current_day'].strftime('%d.%m')} — puuduv ilm: {missing_text}"
                    )

        st.markdown("##### Õppimisandmestik")
        st.caption(
            "Üks rida = ühe põllu üks järgmine korje. Ilmatunnused arvutatakse ainult selle põllu "
            "eelmise ja järgmise korje vahele jäävatest mõõdetud päevadest. A+B+C põhimudeli siht on tegelik A+B+C; XL hinnatakse eraldi."
        )

        # Bioloogilise koormuse jäljed arvutatakse ainult varasematest päriselt mõõdetud
        # sama põllu korjetest. Toorest eelmist saaki ei kasutata ankru ega prognoosi alusena.
        # Koormusindeks küsib: kas konkreetne korje oli selle põllu enda varasema tasemega
        # võrreldes ebatavaliselt suur, ning kas sellel on 1–2 korje järel stabiilne mõju.
        load_state_by_field_date = {}
        for _field_no, _rows in rows_by_field.items():
            _abc_hist = []
            _peak_hist = []
            # rows_by_field sisaldab (harvest_day, row) paare ja on juba kuupäeva järgi sorteeritud.
            # Ära käsitle paari dict-ina: biokoormuse arvutus kasutab päris harvest-rida.
            for _d, _row in _rows:
                try:
                    _a = float(_row.get("a")); _b = float(_row.get("b")); _c = float(_row.get("c"))
                except (TypeError, ValueError, AttributeError):
                    continue
                _quality = str(_row.get("data_quality") or "").strip().lower()
                if _quality in {"hinnanguline", "ligikaudne"}:
                    continue
                _abc = _a + _b + _c
                _prior = _abc_hist[-3:]
                _baseline = float(np.mean(_prior)) if _prior else None
                _load_index = (_abc / _baseline) if (_baseline is not None and _baseline > 0) else None
                _overload = max(0.0, _load_index - 1.0) if _load_index is not None else None
                _two_load = None
                if _prior:
                    _two_mean = float(np.mean([_abc, _prior[-1]]))
                    _two_load = (_two_mean / _baseline) if _baseline > 0 else None
                _peak = 1.0 if (_load_index is not None and _load_index >= 1.25) else 0.0 if _load_index is not None else None
                load_state_by_field_date[(int(_field_no), _d)] = {
                    "Koormusindeks -1": _load_index,
                    "Ülekoormus -1": _overload,
                    "2 korje koormus": _two_load,
                    "Tipukorje -1": _peak,
                    "Tipukorje -2": _peak_hist[-1] if _peak_hist else None,
                }
                _abc_hist.append(_abc)
                _peak_hist.append(_peak)

        training_rows = []
        for sample in usable_samples:
            current_row = sample.get("current_row") or {}
            previous_row = sample.get("previous_row") or {}
            previous2_row = sample.get("previous2_row") or {}

            # Õpperea siht peab olema päriselt numbriline. Osaline vana kirje võib olla
            # kasvuperioodi alguspunkt, kuid seda ei kasutata ise sihtreana.
            try:
                target_total = float(current_row.get("total"))
            except (TypeError, ValueError):
                continue

            # Hinnangulist vana korjet ei kasuta mudeli sihtreana. See võib jääda
            # ajajoone lähtepunktiks, kuid oletuslik saak ei tohi õpetada mudelit nagu tegelik mõõtmine.
            current_quality = str(current_row.get("data_quality") or "").strip().lower()
            if current_quality == "hinnanguline":
                continue

            window_weather = []
            cursor = sample["previous_day"] + timedelta(days=1)
            while cursor <= sample["current_day"]:
                wr = weather_by_day.get(cursor.isoformat())
                if wr:
                    window_weather.append(wr)
                cursor += timedelta(days=1)

            if len(window_weather) != sample["weather_days"] or not window_weather:
                continue

            tmin = [_n(w.get("temp_min_c")) for w in window_weather]
            tmax = [_n(w.get("temp_max_c")) for w in window_weather]
            night_t = [_n(w.get("temp_night_avg_c")) for w in window_weather]
            day_t = [_n(w.get("temp_day_avg_c")) for w in window_weather]
            daily_mean_t = [(night + day) / 2 for night, day in zip(night_t, day_t)]
            rad = [_n(w.get("radiation_mj_m2")) for w in window_weather]
            rain = [_n(w.get("precipitation_mm")) for w in window_weather]
            hum = [_n(w.get("humidity_avg_pct")) for w in window_weather]
            et0 = [_n(w.get("et0_mm")) for w in window_weather]
            wind = [_n(w.get("wind_avg_ms")) for w in window_weather]

            previous_total = previous_row.get("total")
            try:
                previous_total = float(previous_total) if previous_total is not None else None
            except (TypeError, ValueError):
                previous_total = None

            def _maybe_float(value):
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            current_a = _maybe_float(current_row.get("a"))
            current_b = _maybe_float(current_row.get("b"))
            current_c = _maybe_float(current_row.get("c"))
            current_xl = _maybe_float(current_row.get("xl"))
            if current_a is None or current_b is None or current_c is None or current_xl is None:
                continue
            target_abc = current_a + current_b + current_c
            target_cb = (current_c / current_b) if current_b > 0 else None

            previous_xl = _maybe_float(previous_row.get("xl"))
            previous2_xl = _maybe_float(previous2_row.get("xl"))
            previous2_total = _maybe_float(previous2_row.get("total"))
            previous_a = _maybe_float(previous_row.get("a"))
            previous_b = _maybe_float(previous_row.get("b"))
            previous_c = _maybe_float(previous_row.get("c"))
            previous2_a = _maybe_float(previous2_row.get("a"))
            previous2_b = _maybe_float(previous2_row.get("b"))
            previous2_c = _maybe_float(previous2_row.get("c"))
            previous_abc = (previous_a + previous_b + previous_c) if None not in (previous_a, previous_b, previous_c) else None
            previous2_abc = (previous2_a + previous2_b + previous2_c) if None not in (previous2_a, previous2_b, previous2_c) else None

            # Kui eelmine saak oli ainult hinnanguline, kasutame selle kuupäeva intervalli
            # määramiseks, kuid mitte saagi/XL lag-tunnusena. Mudeli missing-indikaator
            # saab siis ausalt aru, et eelmine saagitase pole usaldusväärne.
            prev_quality = str(previous_row.get("data_quality") or "").strip().lower()
            prev2_quality = str(previous2_row.get("data_quality") or "").strip().lower()
            if prev_quality == "hinnanguline":
                previous_total = previous_abc = previous_xl = None
            if prev2_quality == "hinnanguline":
                previous2_total = previous2_abc = previous2_xl = None

            previous_xl_share = (previous_xl / previous_total) if (previous_xl is not None and previous_total and previous_total > 0) else None
            previous2_xl_share = (previous2_xl / previous2_total) if (previous2_xl is not None and previous2_total and previous2_total > 0) else None
            previous_cb = None if prev_quality == "hinnanguline" else ((previous_c / previous_b) if (previous_c is not None and previous_b and previous_b > 0) else None)
            previous2_cb = None if prev2_quality == "hinnanguline" else ((previous2_c / previous2_b) if (previous2_c is not None and previous2_b and previous2_b > 0) else None)
            yield_trend = (previous_abc - previous2_abc) if (previous_abc is not None and previous2_abc is not None) else None
            prev_load = load_state_by_field_date.get((int(sample["field_no"]), sample["previous_day"]), {})

            def _tail_weather(n):
                tail = window_weather[-min(n, len(window_weather)):]
                tmins = [_n(w.get("temp_min_c")) for w in tail]
                tmaxs = [_n(w.get("temp_max_c")) for w in tail]
                mt = [(lo + hi) / 2 for lo, hi in zip(tmins, tmaxs)]
                rad_sum = sum(_n(w.get("radiation_mj_m2")) for w in tail)
                rain_sum = sum(_n(w.get("precipitation_mm")) for w in tail)
                et0_sum = sum(_n(w.get("et0_mm")) for w in tail)
                rh_mean = sum(_n(w.get("humidity_avg_pct")) for w in tail) / len(tail)
                wind_mean = sum(_n(w.get("wind_avg_ms")) for w in tail) / len(tail)
                tmax_mean = sum(tmaxs) / len(tmaxs)

                return {
                    f"T viim{n}": sum(mt) / len(mt),
                    f"Tmin viim{n}": sum(tmins) / len(tmins),
                    f"Tmax viim{n}": tmax_mean,
                    f"Rad viim{n}": rad_sum,
                    f"Sade viim{n}": rain_sum,
                    f"ET0 viim{n}": et0_sum,
                    f"Niiskus viim{n}": rh_mean,
                    f"Tuul viim{n}": wind_mean,
                    f"Tuul×Tmax viim{n}": wind_mean * tmax_mean,
                    f"Tuul×Rad/p viim{n}": wind_mean * (rad_sum / len(tail)),
                    f"Tuul×ET0/p viim{n}": wind_mean * (et0_sum / len(tail)),
                    f"Tuul×Kuivus viim{n}": wind_mean * (100.0 - rh_mean),
                    f"Päevapikkus viim{n}": float(np.mean([
                        _daylength_hours(sample["current_day"] - timedelta(days=i))
                        for i in range(min(n, sample["interval_days"]))
                    ])) if sample["interval_days"] > 0 else _daylength_hours(sample["current_day"]),
                }

            tail1 = _tail_weather(1)
            tail2 = _tail_weather(2)
            tail3 = _tail_weather(3)

            temp_curve = _temperature_curve_features(night_t, day_t)

            training_rows.append({
                "Kuupäev": sample["current_day"],
                "Põld": sample["field_no"],
                "Intervall p": sample["interval_days"],
                "Saak": target_total,
                "ABC saak": target_abc,
                "C/B siht": target_cb,
                "Eelmine ABC": previous_abc,
                "Eelmine saak": previous_total,
                "XL -1": previous_xl,
                "XL -2": previous2_xl,
                "T kesk": sum(daily_mean_t) / len(daily_mean_t),
                "ÖöT kesk": sum(night_t) / len(night_t),
                "PäevT kesk": sum(day_t) / len(day_t),
                "Tmin kesk": sum(tmin) / len(night_t),
                "Tmax kesk": sum(tmax) / len(tmax),
                "Tmin min": min(tmin),
                "Tmax max": max(tmax),
                **temp_curve,
                "Soojad ööd 16+": sum(1 for v in night_t if v >= 16.0),
                "Soojad ööd 18+": sum(1 for v in night_t if v >= 18.0),
                "Jahedad ööd 12-": sum(1 for v in night_t if v <= 12.0),
                "Soojad ööd 16+ %": 100.0 * sum(1 for v in night_t if v >= 16.0) / len(night_t),
                "Soojad ööd 18+ %": 100.0 * sum(1 for v in night_t if v >= 18.0) / len(night_t),
                "Jahedad ööd 12- %": 100.0 * sum(1 for v in night_t if v <= 12.0) / len(night_t),
                "Radiatsioon Σ": sum(rad),
                "Radiatsioon/p": sum(rad) / len(rad),
                "Sademed Σ": sum(rain),
                "Niiskus kesk": sum(hum) / len(hum),
                "ET0 Σ": sum(et0),
                "Tuul kesk": sum(wind) / len(wind),

                # Tuule koostoimed. Need EI lähe automaatselt baasmudelisse,
                # vaid Jäljeotsija testib neid walk-forward meetodil.
                "Tuul×Tmax": (sum(wind) / len(wind)) * (sum(tmax) / len(tmax)),
                "Tuul×Rad/p": (sum(wind) / len(wind)) * (sum(rad) / len(rad)),
                "Tuul×ET0/p": (sum(wind) / len(wind)) * (sum(et0) / len(et0)),
                "Tuul×Kuivus": (sum(wind) / len(wind)) * (100.0 - (sum(hum) / len(hum))),

                # Fotoperiood / päevapikkus.
                "Päevapikkus": _daylength_hours(sample["current_day"]),
                "Päevapikkus Δ7p": _daylength_change_7d(sample["current_day"]),
                "Päevapikkus kasvukesk": float(np.mean([
                    _daylength_hours(sample["previous_day"] + timedelta(days=i))
                    for i in range(1, sample["interval_days"] + 1)
                ])) if sample["interval_days"] > 0 else _daylength_hours(sample["current_day"]),

                "A": current_row.get("a"),
                "B": current_row.get("b"),
                "C": current_row.get("c"),
                "XL": current_row.get("xl"),
                # Jäljeotsija kandidaadid. Neid ei näidata põhitabelis ega kasutata baasmudelis.
                "Eelmine2 ABC": previous2_abc,
                "ABC trend": yield_trend,
                "Eelmine2 saak": previous2_total,
                "XL osakaal -1": previous_xl_share,
                "XL osakaal -2": previous2_xl_share,
                "C/B -1": previous_cb,
                "C/B -2": previous2_cb,
                "Koormusindeks -1": prev_load.get("Koormusindeks -1"),
                "Ülekoormus -1": prev_load.get("Ülekoormus -1"),
                "2 korje koormus": prev_load.get("2 korje koormus"),
                "Tipukorje -1": prev_load.get("Tipukorje -1"),
                "Tipukorje -2": prev_load.get("Tipukorje -2"),
                **tail1, **tail2, **tail3,
                "Andmekvaliteet": current_row.get("data_quality") or "",
            })

        if training_rows:
            training_df = pd.DataFrame(training_rows).sort_values(["Kuupäev", "Põld"], ascending=[False, True])
            t1, t2, t3 = st.columns(3)
            t1.metric("Valmis õppimisridu", len(training_df))
            t2.metric("Keskmine intervall", f"{training_df['Intervall p'].mean():.1f} p")
            t3.metric("Keskmine A+B+C", f"{training_df['ABC saak'].mean():.1f} kasti")

            visible_training_cols = [
                "Kuupäev", "Põld", "Intervall p", "ABC saak", "C/B siht", "XL", "Saak", "Eelmine ABC",
                "T kesk", "ÖöT kesk", "PäevT kesk", "Tmin kesk", "Tmax kesk", "Tmin min", "Tmax max", "Radiatsioon Σ", "Radiatsioon/p", "Sademed Σ",
                "Niiskus kesk", "ET0 Σ", "Tuul kesk", "A", "B", "C", "Andmekvaliteet",
            ]
            display_df = training_df[visible_training_cols].copy()
            display_df["Kuupäev"] = display_df["Kuupäev"].map(lambda d: d.strftime("%d.%m"))
            st.dataframe(
                display_df.style.format({
                    "ABC saak": "{:.1f}",
                    "C/B siht": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                    "Saak": "{:.1f}",
                    "Eelmine ABC": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                    "T kesk": "{:.1f}",
                    "ÖöT kesk": "{:.1f}",
                    "PäevT kesk": "{:.1f}",
                    "Tmin kesk": "{:.1f}",
                    "Tmax kesk": "{:.1f}",
                    "Tmin min": "{:.1f}",
                    "Tmax max": "{:.1f}",
                    "Radiatsioon Σ": "{:.1f}",
                    "Radiatsioon/p": "{:.1f}",
                    "Sademed Σ": "{:.1f}",
                    "Niiskus kesk": "{:.1f}",
                    "ET0 Σ": "{:.1f}",
                    "Tuul kesk": "{:.1f}",
                    "A": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "B": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "XL": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Radiatsioon Σ, sademed Σ ja ET0 Σ on kogu korjevahemiku summad; T, niiskus ja tuul on "
                "sama vahemiku päevade keskmised. Põhimudeli siht on A+B+C; XL käsitletakse eraldi mürasema korjejäägi komponendina."
            )

            csv_df = training_df.copy()
            csv_df["Kuupäev"] = csv_df["Kuupäev"].map(lambda d: d.isoformat())
            st.download_button(
                "Laadi õppimisandmestik CSV-na",
                data=csv_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="kurgimootor_training_dataset.csv",
                mime="text/csv",
            )

            st.markdown("##### A+B+C kasvupotentsiaali mudel + eraldi XL-komponent")
            st.caption(
                "Temperatuur ja tuul: Häädemeeste mõõdetud tunniandmed. "
                "Radiatsioon, RH ja sademed: Pärnu. Tmin/Tmax säilivad ET0 jaoks, "
                "aga saagimudeli temperatuuriloogika kasutab öö- ja päevakeskmist."
            )
            st.caption(
                "Temperatuurivastus on mittelineaarne: külma- ja kuumastressil on eraldi tsoonid ning ruutliikmed. "
                "Seetõttu ei käsitle mootor näiteks 12→14 °C ja 16→18 °C öist muutust võrdse mõjuna. "
                "Murdepunktid on bioloogiliselt informeeritud; mõju tugevuse õpib mudel meie enda korjetest."
            )
            st.caption(
                "Põhimudel ennustab ainult A+B+C saaki positiivsel kasvupotentsiaali skaalal. "
                "Baasmudel kasutab Häädemeeste mõõdetud öö/päeva keskmise temperatuuri mittelineaarset kasvukõverat, muud ilma, korjeintervalli, hooaja faasi ja põllu identiteeti; eelmine saak ei ole prognoosi ankur. "
                "XL prognoositakse eraldi mürasema korjejäägi komponendina. Mõlemat hinnatakse ajaliselt ausa walk-forward testiga."
            )

            # Baasmudel on teadlikult puhas bioloogiline mudel: ilm + kasvuaeg + põld + hooaja faas.
            # Eelmise korje saak EI ole baasmudeli kohustuslik sisend; Jäljeotsija võib selle
            # eraldi kandidaadina sisse lubada ainult siis, kui aus walk-forward test tõestab kasu.
            base_cont_cols = [
                "Intervall p", "Hooajapäev",

                # V6.4 nonlinear-temperature:
                # temperatuur EI sisene enam ühe lineaarse Tmin/Tmax koefitsiendina.
                # Külmastressi ruutliige võimaldab nt 12 C ööl olla ebaproportsionaalselt
                # halvem kui 14 C, samal ajal kui 16 -> 18 ei pea andma sama suurt võitu.
                "Öö jahedus <16", "Öö jahedus² <16",
                "Öö soojus 16-20", "Öö kuumus >20",
                "Päeva jahedus <20", "Päeva jahedus² <20",
                "Päeva soojus 20-28",
                "Päeva kuumus >30", "Päeva kuumus² >30",

                "Radiatsioon Σ", "Radiatsioon/p",
                "Sademed Σ", "Niiskus kesk", "ET0 Σ", "Tuul kesk",
            ]
            model_df = training_df.copy().sort_values(["Kuupäev", "Põld"]).reset_index(drop=True)
            model_df["Hooajapäev"] = pd.to_datetime(model_df["Kuupäev"]).map(
                lambda d: (d.date() - SEASON_START).days
            )
            fields = model_df["Põld"].astype(int).to_numpy()
            dates = pd.to_datetime(model_df["Kuupäev"]).dt.date.to_numpy()
            X_base = model_df[base_cont_cols].astype(float).to_numpy()

            y_abc = pd.to_numeric(model_df["ABC saak"], errors="coerce").to_numpy(dtype=float)
            # V6.4: A+B+C operatiivne mudel õpib multiplikatiivsel positiivsel kasvupotentsiaali skaalal.
            # EPS on ainult log(0) numbriline kaitse, mitte bioloogiline saagipõrand.
            ABC_LOG_EPS = 0.05
            log_y_abc = np.where(
                np.isfinite(y_abc) & (y_abc >= 0),
                np.log(np.maximum(y_abc, ABC_LOG_EPS)),
                np.nan,
            )
            y_xl = pd.to_numeric(model_df["XL"], errors="coerce").to_numpy(dtype=float)
            y_total = pd.to_numeric(model_df["Saak"], errors="coerce").to_numpy(dtype=float)
            y_cb = pd.to_numeric(model_df["C/B siht"], errors="coerce").to_numpy(dtype=float)
            log_y_cb = np.where(np.isfinite(y_cb) & (y_cb > 0), np.log(y_cb), np.nan)
            raw_prev_abc = pd.to_numeric(model_df["Eelmine ABC"], errors="coerce").to_numpy(dtype=float)
            raw_prev_total = pd.to_numeric(model_df["Eelmine saak"], errors="coerce").to_numpy(dtype=float)
            raw_xl1 = pd.to_numeric(model_df["XL -1"], errors="coerce").to_numpy(dtype=float)
            raw_xl2 = pd.to_numeric(model_df["XL -2"], errors="coerce").to_numpy(dtype=float)
            raw_cb1 = pd.to_numeric(model_df["C/B -1"], errors="coerce").to_numpy(dtype=float)
            raw_cb2 = pd.to_numeric(model_df["C/B -2"], errors="coerce").to_numpy(dtype=float)

            def _build_ridge_design(train_idx, test_idx, extra_arrays):
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
                ftr = np.zeros((len(train_idx), 14), dtype=float)
                fte = np.zeros((len(test_idx), 14), dtype=float)
                for ri, f in enumerate(fields[train_idx]):
                    if 1 <= int(f) <= 14:
                        ftr[ri, int(f) - 1] = 1.0
                for ri, f in enumerate(fields[test_idx]):
                    if 1 <= int(f) <= 14:
                        fte[ri, int(f) - 1] = 1.0
                tr_missing = np.column_stack(tr_missing_parts) if tr_missing_parts else np.empty((len(train_idx), 0))
                te_missing = np.column_stack(te_missing_parts) if te_missing_parts else np.empty((len(test_idx), 0))
                Xtr = np.column_stack([np.ones(len(train_idx)), ztr, tr_missing, ftr])
                Xte = np.column_stack([np.ones(len(test_idx)), zte, te_missing, fte])
                return Xtr, Xte, means, scales, fills

            def _ridge_walk_predict(target, extra_arrays, train_idx, test_idx, alpha=10.0, floor_zero=True, field_alpha=80.0):
                Xtr, Xte, _, _, _ = _build_ridge_design(train_idx, test_idx, extra_arrays)
                penalty = np.eye(Xtr.shape[1]) * alpha
                penalty[0, 0] = 0.0
                # 14 viimast veergu on põllu one-hot tunnused. Õppimisandmestik on veel väike,
                # seetõttu kasutame siin tugevat partial-pooling kahandamist: põllu eripära jääb
                # alles, kuid ei tohi üksinda weather-first prognoosi nulli/äärmusse suruda.
                penalty[-14:, -14:] = np.eye(14) * field_alpha
                beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ target[train_idx]
                values = Xte @ beta
                return np.maximum(values, 0.0) if floor_zero else values

            def _abc_growth_walk_predict(extra_arrays, train_idx, test_idx, alpha=10.0, field_alpha=80.0, z_clip=2.5):
                """V6.4 A+B+C: multiplicative/positive growth-potential model.

                Õpib log(A+B+C) multiplikatiivsel skaalal. Ilma ja muude pidevate tunnuste z-skoorid
                piiratakse treeningu mõistlikku vahemikku, et 6–9 päeva prognoosi
                ekstreemne ilm ei saaks lineaarset latentset mudelit absurdselt ekstrapoleerida.
                """
                valid_train = train_idx[np.isfinite(log_y_abc[train_idx])]
                if len(valid_train) < min_train_rows:
                    return np.full(len(test_idx), np.nan, dtype=float)
                Xtr, Xte, _, _, _ = _build_ridge_design(valid_train, test_idx, extra_arrays)
                n_numeric = X_base.shape[1] + len(extra_arrays)
                # intercept on veerg 0; sellele järgnevad standardiseeritud arvulised tunnused
                Xtr[:, 1:1+n_numeric] = np.clip(Xtr[:, 1:1+n_numeric], -z_clip, z_clip)
                Xte[:, 1:1+n_numeric] = np.clip(Xte[:, 1:1+n_numeric], -z_clip, z_clip)
                penalty = np.eye(Xtr.shape[1]) * alpha
                penalty[0, 0] = 0.0
                penalty[-14:, -14:] = np.eye(14) * field_alpha
                beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ log_y_abc[valid_train]
                latent = Xte @ beta
                # exp() teeb mudeli multiplikatiivseks ja positiivseks. Väga halb ilm võib
                # viia prognoosi väga madalale, kuid mitte lineaarse algebra tõttu negatiivseks.
                return np.exp(np.clip(latent, np.log(ABC_LOG_EPS), 6.0))

            min_train_rows = 10
            abc_predictions = np.full(len(model_df), np.nan, dtype=float)
            xl_predictions = np.full(len(model_df), np.nan, dtype=float)
            for test_day in sorted(set(dates)):
                test_idx = np.where(dates == test_day)[0]
                train_idx = np.where(dates < test_day)[0]
                if len(train_idx) < min_train_rows:
                    continue
                # A+B+C BAASMudel: ainult ilm + intervall + hooajapäev + põllu identiteet.
                # Eelmist saaki siin tahtlikult EI kasutata.
                abc_predictions[test_idx] = _abc_growth_walk_predict([], train_idx, test_idx)
                # XL BAASMudel: ainult ilm + intervall + hooajapäev + põllu identiteet.
                # Toores eelmine XL/ABC/kogusaak EI kuulu baasi; need saavad ainult
                # walk-forward kandidaadina tõestada, kas neil on päriselt lisaväärtust.
                xl_predictions[test_idx] = _ridge_walk_predict(
                    y_xl, [], train_idx, test_idx
                )

            # C/B BAASMudel: ainult ilm + intervall + hooajapäev + põllu identiteet.
            # Eelmine C/B ei ole enam vaikimisi ankur, vaid eraldi testitav kandidaat.
            cb_predictions = np.full(len(model_df), np.nan, dtype=float)
            for test_day in sorted(set(dates)):
                test_idx = np.where(dates == test_day)[0]
                train_idx = np.where((dates < test_day) & np.isfinite(log_y_cb))[0]
                if len(train_idx) < min_train_rows:
                    continue
                log_pred = _ridge_walk_predict(log_y_cb, [], train_idx, test_idx, floor_zero=False)
                cb_predictions[test_idx] = np.exp(np.clip(log_pred, np.log(0.10), np.log(10.0)))

            valid_abc = np.isfinite(abc_predictions)
            valid_xl = np.isfinite(xl_predictions)
            valid_both = valid_abc & valid_xl
            predictions = abc_predictions  # Jäljeotsija baasiks jääb objektiivne A+B+C mudel.
            y = y_abc
            valid_pred = valid_abc
            current_test_mae = None
            current_total_test_mae = None

            if valid_abc.any():
                abc_err = abc_predictions - y_abc
                abc_mae = float(np.mean(np.abs(abc_err[valid_abc])))
                current_test_mae = abc_mae
                abc_rmse = float(np.sqrt(np.mean(abc_err[valid_abc] ** 2)))
                abc_bias = float(np.mean(abc_err[valid_abc]))
                abc_within2 = float(np.mean(np.abs(abc_err[valid_abc]) <= 2.0) * 100.0)

                xl_mae = float(np.mean(np.abs(xl_predictions[valid_xl] - y_xl[valid_xl]))) if valid_xl.any() else None
                total_predictions = abc_predictions + xl_predictions
                if valid_both.any():
                    total_err = total_predictions - y_total
                    total_mae = float(np.mean(np.abs(total_err[valid_both])))
                    current_total_test_mae = total_mae
                else:
                    total_mae = None

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("A+B+C MAE", f"{abc_mae:.1f} kasti")
                r2.metric("XL MAE", "—" if xl_mae is None else f"{xl_mae:.1f} kasti")
                r3.metric("Kogu MAE", "—" if total_mae is None else f"{total_mae:.1f} kasti")
                r4.metric("A+B+C ±2 sees", f"{abc_within2:.0f}%")
                tested_days_abc = len(set(dates[np.where(valid_abc)[0]]))
                st.caption(
                    f"A+B+C testitud: {int(valid_abc.sum())}/{len(model_df)} rida · {tested_days_abc} testipäeva · "
                    f"RMSE {abc_rmse:.1f} · nihe {abc_bias:+.1f}. Kogu = A+B+C prognoos + eraldi XL prognoos."
                )

                baseline_mask = valid_abc & np.isfinite(raw_prev_abc)
                if baseline_mask.any():
                    baseline_mae = float(np.mean(np.abs(raw_prev_abc[baseline_mask] - y_abc[baseline_mask])))
                    model_same = float(np.mean(np.abs(abc_predictions[baseline_mask] - y_abc[baseline_mask])))
                    delta = baseline_mae - model_same
                    if delta > 0:
                        st.success(f"A+B+C mudel: MAE {model_same:.1f} vs lihtne 'eelmine A+B+C' {baseline_mae:.1f}. Eelis {delta:.1f} kasti.")
                    else:
                        st.warning(f"A+B+C mudel: MAE {model_same:.1f} vs lihtne 'eelmine A+B+C' {baseline_mae:.1f}. Baas on praegu parem.")

                eval_df = model_df[["Kuupäev", "Põld", "ABC saak", "XL", "Saak", "Eelmine ABC"]].copy()
                eval_df["ABC prognoos"] = abc_predictions
                eval_df["XL prognoos"] = xl_predictions
                eval_df["Kogu prognoos"] = abc_predictions + xl_predictions
                eval_df["ABC viga"] = eval_df["ABC prognoos"] - eval_df["ABC saak"]
                eval_df["Kogu viga"] = eval_df["Kogu prognoos"] - eval_df["Saak"]
                eval_df = eval_df.sort_values(["Kuupäev", "Põld"], ascending=[False, True])
                eval_df["Kuupäev"] = eval_df["Kuupäev"].map(lambda d: d.strftime("%d.%m"))
                st.dataframe(
                    eval_df.style.format({
                        "ABC saak": "{:.1f}", "XL": "{:.1f}", "Saak": "{:.1f}",
                        "Eelmine ABC": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                        "ABC prognoos": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                        "XL prognoos": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                        "Kogu prognoos": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                        "ABC viga": lambda v: "—" if pd.isna(v) else f"{v:+.1f}",
                        "Kogu viga": lambda v: "—" if pd.isna(v) else f"{v:+.1f}",
                    }), use_container_width=True, hide_index=True,
                )
                st.caption(
                    "ABC viga näitab taime tootmise prognoosi viga. Kogu viga sisaldab lisaks XL-komponendi müra. "
                    "XL-mudel on eraldi just selleks, et korjejääk ei moonutaks põhimudelit."
                )
            else:
                st.info(f"Ajaliselt ausaks testiks on vaja vähemalt {min_train_rows} varasemat õppimisrida.")

            # -------------------------------------------------------------------------
            # Jäljeotsija v2 — automaatne champion-valik A+B+C põhimudelile
            # -------------------------------------------------------------------------
            # WEATHER-FIRST + BIOLOOGILISE KOORMUSE LUKK:
            # A+B+C baas jääb ilm + intervall + hooaja faas + põld. Operatiivseks korrigeerijaks
            # võivad saada kas ilmast tuletatud jäljed või normaliseeritud bioloogilise koormuse
            # jäljed, kui need läbivad sama range walk-forward stabiilsustesti. Toores eelmine
            # saak/trend/XL/C-B ei saa prognoosi ankurdada ja jäävad diagnostikasse.
            weather_candidate_groups = {
                "Viimase 1 päeva ilm": ["ÖöT viim1", "PäevT viim1", "Rad viim1", "Sade viim1", "ET0 viim1", "Niiskus viim1"],
                "Viimase 2 päeva ilm": ["ÖöT viim2", "PäevT viim2", "Rad viim2", "Sade viim2", "ET0 viim2", "Niiskus viim2"],
                "Viimase 3 päeva ilm": ["ÖöT viim3", "PäevT viim3", "Rad viim3", "Sade viim3", "ET0 viim3", "Niiskus viim3"],

                # Ööd / temperatuur
                "Soojad ööd 16+": ["Soojad ööd 16+ %"],
                "Väga soojad ööd 18+": ["Soojad ööd 18+ %"],
                "Jahedad ööd 12-": ["Jahedad ööd 12- %"],
                "Öötemperatuuri kuju": ["ÖöT kesk", "Soojad ööd 16+ %", "Jahedad ööd 12- %"],
                "Päev/öö keskmised": ["ÖöT kesk", "PäevT kesk"],
                # Temperatuuri õppimine põhineb Häädemeeste öö/päeva keskmistel.
                # Tmin/Tmax säilivad ilmaajaloos ja ET0 arvutuses, kuid ei kandideeri champion-tunnuseks.

                # Tuule koostoimed
                "Tuul × Tmax": ["Tuul×Tmax"],
                "Tuul × radiatsioon": ["Tuul×Rad/p"],
                "Tuul × ET0": ["Tuul×ET0/p"],
                "Tuul × kuivus": ["Tuul×Kuivus"],
                "Tuulestress kombineeritud": ["Tuul×Tmax", "Tuul×Rad/p", "Tuul×ET0/p", "Tuul×Kuivus"],

                # Viimase 1–3 päeva tuulestress
                "Tuulestress viim1": ["Tuul×Tmax viim1", "Tuul×Rad/p viim1", "Tuul×ET0/p viim1", "Tuul×Kuivus viim1"],
                "Tuulestress viim2": ["Tuul×Tmax viim2", "Tuul×Rad/p viim2", "Tuul×ET0/p viim2", "Tuul×Kuivus viim2"],
                "Tuulestress viim3": ["Tuul×Tmax viim3", "Tuul×Rad/p viim3", "Tuul×ET0/p viim3", "Tuul×Kuivus viim3"],

                # Fotoperiood
                "Päevapikkus": ["Päevapikkus"],
                "Päevapikkuse trend": ["Päevapikkus", "Päevapikkus Δ7p"],
                "Kasvuperioodi päevapikkus": ["Päevapikkus kasvukesk"],
                "Päevapikkus viim3": ["Päevapikkus viim3", "Päevapikkus Δ7p"],
            }
            biological_load_candidate_groups = {
                "Ebatavaline koormus -1": ["Koormusindeks -1", "Ülekoormus -1"],
                "Kahe korje koormus": ["2 korje koormus"],
                "Tipukorje järelmõju": ["Tipukorje -1", "Tipukorje -2"],
            }
            operational_candidate_groups = {**weather_candidate_groups, **biological_load_candidate_groups}
            memory_diagnostic_groups = {
                "Eelmine A+B+C": ["Eelmine ABC"],
                "A+B+C trend 2 korjet": ["Eelmine ABC", "Eelmine2 ABC", "ABC trend"],
                "Korjejääk / kõrge eelkorje": ["Eelmine saak", "XL -1", "XL -2", "XL osakaal -1", "XL osakaal -2"],
                "XL osakaal 2 korjet": ["XL osakaal -1", "XL osakaal -2"],
            }
            diagnostic_only_groups = {
                **memory_diagnostic_groups,
                "C/B mälu 2 korjet (diagnostika)": ["C/B -1", "C/B -2"],
            }

            def _walk_forward_with_extra(extra_cols, alpha=10.0):
                raw_extra = model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where(dates < test_day)[0]
                    if len(train_idx) < min_train_rows:
                        continue
                    # Kandidaat saab ainult need lisatunnused, mis tema grupis on nimetatud.
                    # Eelmist saaki ei lisata enam vaikimisi ühelegi kandidaadile.
                    extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]
                    preds[test_idx] = _abc_growth_walk_predict(extra_arrays, train_idx, test_idx)
                return preds

            def _stability_stats(candidate_pred):
                mask = np.isfinite(predictions) & np.isfinite(candidate_pred)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return None
                base_abs = np.abs(predictions[idx] - y[idx])
                cand_abs = np.abs(candidate_pred[idx] - y[idx])
                base_mae = float(base_abs.mean())
                cand_mae = float(cand_abs.mean())
                improvement = base_mae - cand_mae
                win_share = float(np.mean(cand_abs < base_abs))
                # Ajalise stabiilsuse kontroll: sama testipäeva põllud jäävad alati samasse poolde.
                unique_test_days = sorted(set(dates[idx]))
                day_halves = np.array_split(np.array(unique_test_days, dtype=object), 2)
                half_improvements = []
                for day_half in day_halves:
                    day_set = set(day_half.tolist())
                    h = np.array([i for i in idx if dates[i] in day_set], dtype=int)
                    if len(h) == 0:
                        continue
                    half_improvements.append(float(
                        np.mean(np.abs(predictions[h] - y[h])) - np.mean(np.abs(candidate_pred[h] - y[h]))
                    ))
                min_half = min(half_improvements) if half_improvements else -999.0
                stable = bool(len(idx) >= 12 and improvement >= 0.10 and win_share >= 0.50 and min_half >= -0.05)
                return {
                    "Baas MAE": base_mae, "Katse MAE": cand_mae, "Paranemine": improvement,
                    "Võidab ridu %": win_share * 100.0, "Halvim pool": min_half,
                    "Testiridu": int(len(idx)), "Stabiilne": stable,
                }

            trace_results = []
            candidate_predictions = {}
            for name, cols in {**operational_candidate_groups, **diagnostic_only_groups}.items():
                cp = _walk_forward_with_extra(cols)
                candidate_predictions[name] = cp
                stats = _stability_stats(cp)
                if stats:
                    trace_results.append({"Jälg": name, **stats})

            trace_df = pd.DataFrame(trace_results)
            if not trace_df.empty:
                trace_df = trace_df.sort_values(["Stabiilne", "Paranemine"], ascending=[False, False])

            champion_name = "Baasmudel"
            champion_cols = []
            champion_pred = predictions.copy()
            champion_mae = current_test_mae
            stable_operational = []
            for name, cols in operational_candidate_groups.items():
                if name not in candidate_predictions:
                    continue
                stats = _stability_stats(candidate_predictions[name])
                if stats and stats["Stabiilne"]:
                    stable_operational.append((stats["Katse MAE"], name, cols, candidate_predictions[name], stats))
            if stable_operational:
                stable_operational.sort(key=lambda x: x[0])
                champion_mae, champion_name, champion_cols, champion_pred, champion_stats = stable_operational[0]
            else:
                champion_stats = None


            # -------------------------------------------------------------------------
            # Avastusmootori CPU-turvarežiim
            # -------------------------------------------------------------------------
            # Community Cloudi tavakasutuses ei käivita rasket autonoomset ideegeneraatorit
            # iga Streamliti rerun'i ajal. Operatiivne prognoos ja tavapärane Jäljeotsija
            # töötavad edasi. Autonoomne avastus ehitatakse järgmises etapis püsiva
            # "ainult uute andmete korral" käivitusega.
            _complete_day_count = len(complete_harvest_days)
            try:
                _last_full_idea_count = int(
                    db.get_app_setting("idea_full_search_complete_day_count", "0") or 0
                )
            except (TypeError, ValueError):
                _last_full_idea_count = 0

            # Lai loominguline otsing: alguses iga 3 uue täieliku 3/3 korjepäeva järel.
            # Vahepeal kontrollib tavaline Jäljeotsija olemasolevaid kandidaate.
            AUTONOMOUS_DISCOVERY_ENABLED = bool(
                _complete_day_count >= _last_full_idea_count + IDEA_FULL_SEARCH_EVERY_COMPLETE_DAYS
            )

            # -------------------------------------------------------------------------
            # Avastusmootori sessioon-cache
            # -------------------------------------------------------------------------
            # Sama õppimisandmestiku ja sama championi korral ei ole vaja kümneid/sadu
            # walk-forward ideekatseid igal Streamliti rerun'il uuesti teha.
            # Uus korje või muutunud ilm muudab model_df sisu ja seega cache-võtit.
            _auto_hash_cols = [
                c for c in model_df.columns
                if c not in {"Andmekvaliteet"}
            ]
            _auto_hash_frame = model_df[_auto_hash_cols].copy()
            for _c in _auto_hash_frame.columns:
                if _c == "Kuupäev":
                    _auto_hash_frame[_c] = _auto_hash_frame[_c].map(
                        lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v)
                    )
            _auto_data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(
                    _auto_hash_frame.astype(str),
                    index=True,
                ).values.tobytes()
            ).hexdigest()
            _auto_cache_key = hashlib.sha256(
                (
                    _auto_data_hash
                    + "|champion=" + str(champion_name)
                    + "|cols=" + ",".join(map(str, champion_cols))
                ).encode("utf-8")
            ).hexdigest()

            _auto_cache = st.session_state.get("_autonomous_discovery_cache")
            _auto_cache_hit = bool(
                isinstance(_auto_cache, dict)
                and _auto_cache.get("key") == _auto_cache_key
            )

            if not AUTONOMOUS_DISCOVERY_ENABLED:
                _saved_auto_raw = db.get_app_setting("autonomous_discovery_trace_json", "")
                _saved_counts_raw = db.get_app_setting("autonomous_discovery_category_counts_json", "")
                try:
                    autonomous_trace_df = pd.DataFrame(json.loads(_saved_auto_raw)) if _saved_auto_raw else pd.DataFrame()
                except Exception:
                    autonomous_trace_df = pd.DataFrame()
                try:
                    autonomous_category_counts = json.loads(_saved_counts_raw) if _saved_counts_raw else {}
                except Exception:
                    autonomous_category_counts = {}
                try:
                    autonomous_candidate_count = int(
                        db.get_app_setting("autonomous_discovery_candidate_count", "0") or 0
                    )
                except (TypeError, ValueError):
                    autonomous_candidate_count = 0
                _auto_discovery_days = set()
                _auto_confirm_days = set()
                _auto_cache_hit = False
            elif _auto_cache_hit:
                autonomous_trace_df = _auto_cache.get("trace_df", pd.DataFrame()).copy()
                autonomous_candidate_count = int(_auto_cache.get("candidate_count", 0))
                autonomous_category_counts = dict(_auto_cache.get("category_counts", {}))
                _auto_discovery_days = set(_auto_cache.get("discovery_days", []))
                _auto_confirm_days = set(_auto_cache.get("confirm_days", []))
            else:
                # Need kandidaadid EI SAA automaatselt championiks. Nad on ainult avastamiseks:
                # iga idee lisatakse tänasele championile ühe lisatunnusena ja testitakse
                # sama ajaliselt ausa walk-forward loogikaga. Kasutuselevõtt jääb eraldi otsuseks.
                autonomous_discovery_rows = []
                autonomous_candidate_count = 0

                def _safe_ratio(num, den):
                    num = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
                    den = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
                    out = np.full(len(num), np.nan, dtype=float)
                    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-6)
                    out[mask] = num[mask] / den[mask]
                    return out

                autonomous_features = []
                autonomous_feature_meta = []
                autonomous_feature_values = {}
                autonomous_category_counts = {
                    "Mittelineaarsed": 0,
                    "Koostoimed": 0,
                    "Suhtarvud": 0,
                    "Ajamuutused": 0,
                    "Temperatuuriläved": 0,
                    "Teise ringi kombinatsioonid": 0,
                }

                def _register_discovery_feature(name, values, category):
                    col = f"__AUTO__{len(autonomous_features)}"
                    arr = np.asarray(values, dtype=float)
                    finite = arr[np.isfinite(arr)]
                    if len(finite) < max(12, min_train_rows):
                        return
                    if np.nanstd(finite) < 1e-9:
                        return
                    autonomous_feature_values[col] = arr
                    autonomous_features.append((name, col))
                    autonomous_feature_meta.append({"Idee": name, "Veerg": col, "Kategooria": category})
                    autonomous_category_counts[category] = autonomous_category_counts.get(category, 0) + 1

                def _past_only_abs_deviation(values):
                    """Ajaliselt aus |x - mediaan| tunnus.

                    Iga kuupäeva rea võrdlusmediaan arvutatakse ainult VARASEMATE
                    kuupäevade ridadest. Sama testipäeva ega tulevasi ridu ei kasutata.
                    """
                    arr = np.asarray(values, dtype=float)
                    out = np.full(len(arr), np.nan, dtype=float)

                    for _day in sorted(set(dates)):
                        _test_idx = np.where(dates == _day)[0]
                        _past_idx = np.where(dates < _day)[0]
                        _past_vals = arr[_past_idx]
                        _past_vals = _past_vals[np.isfinite(_past_vals)]
                        if len(_past_vals) == 0:
                            continue
                        _past_median = float(np.median(_past_vals))
                        _today_vals = arr[_test_idx]
                        _finite_today = np.isfinite(_today_vals)
                        out[_test_idx[_finite_today]] = np.abs(
                            _today_vals[_finite_today] - _past_median
                        )

                    return out

                # 1) Mittelineaarsus: ruut ja kõrvalekalle varasemast tavatasemest.
                # Mediaanipõhine kõrvalekalle on expanding-past: testipäev ega tulevik
                # ei osale oma võrdlusmediaani arvutamises.
                nonlinear_sources = [
                    "T kesk", "ÖöT kesk", "PäevT kesk",
                    "Radiatsioon/p", "Sademed Σ", "Niiskus kesk", "ET0 Σ", "Tuul kesk",
                    "Päevapikkus", "Päevapikkus Δ7p",
                ]
                for _c in nonlinear_sources:
                    if _c not in model_df.columns:
                        continue
                    _x = pd.to_numeric(model_df[_c], errors="coerce").to_numpy(dtype=float)
                    _register_discovery_feature(f"{_c}²", _x ** 2, "Mittelineaarsed")
                    _register_discovery_feature(
                        f"|{_c} − varasem mediaan|",
                        _past_only_abs_deviation(_x),
                        "Mittelineaarsed",
                    )

                # 2) Koostoimed: ristmõjud ilma käsitsi ette kirjutatud hüpoteesita.
                interaction_sources = [
                    "ÖöT kesk", "PäevT kesk", "Radiatsioon/p", "Sademed Σ",
                    "Niiskus kesk", "ET0 Σ", "Tuul kesk", "Päevapikkus",
                    "Tmin viim1", "Tmax viim1", "Rad viim1", "Sade viim1",
                    "ET0 viim1", "Niiskus viim1", "Tuul viim1",
                ]
                interaction_sources = [c for c in interaction_sources if c in model_df.columns]
                for _i, _a in enumerate(interaction_sources):
                    _xa = pd.to_numeric(model_df[_a], errors="coerce").to_numpy(dtype=float)
                    for _b in interaction_sources[_i + 1:]:
                        _xb = pd.to_numeric(model_df[_b], errors="coerce").to_numpy(dtype=float)
                        _register_discovery_feature(f"{_a} × {_b}", _xa * _xb, "Koostoimed")

                # 3) Suhtarvud.
                ratio_pairs = [
                    ("ET0 Σ", "Sademed Σ"),
                    ("Radiatsioon/p", "ET0 Σ"),
                    ("Radiatsioon/p", "Niiskus kesk"),
                    ("Tuul kesk", "Niiskus kesk"),
                    ("ET0 viim1", "Sade viim1"),
                    ("Rad viim1", "ET0 viim1"),
                    ("Tuul viim1", "Niiskus viim1"),
                    ("Rad viim2", "ET0 viim2"),
                    ("Rad viim3", "ET0 viim3"),
                ]
                for _a, _b in ratio_pairs:
                    if _a in model_df.columns and _b in model_df.columns:
                        _register_discovery_feature(
                            f"{_a} / {_b}",
                            _safe_ratio(model_df[_a], model_df[_b]),
                            "Suhtarvud",
                        )

                # 4) Ajamuutus: värske ilm võrreldes 2–3 päeva fooniga.
                temporal_bases = ["T", "Tmin", "Tmax", "Rad", "Sade", "ET0", "Niiskus", "Tuul"]
                for _base in temporal_bases:
                    for _short, _long in ((1, 2), (1, 3), (2, 3)):
                        _a = f"{_base} viim{_short}"
                        _b = f"{_base} viim{_long}"
                        if _a in model_df.columns and _b in model_df.columns:
                            _xa = pd.to_numeric(model_df[_a], errors="coerce").to_numpy(dtype=float)
                            _xb = pd.to_numeric(model_df[_b], errors="coerce").to_numpy(dtype=float)
                            _register_discovery_feature(f"{_a} − {_b}", _xa - _xb, "Ajamuutused")

                # 5) Temperatuuriläved: generaator proovib ka murdepunkte, mida me ette ei valinud.
                if "Tmin min" in model_df.columns:
                    _x = pd.to_numeric(model_df["Tmin min"], errors="coerce").to_numpy(dtype=float)
                    for _thr in range(10, 20, 2):
                        _register_discovery_feature(
                            f"Öö puudujääk alla {_thr}°",
                            np.maximum(0.0, float(_thr) - _x),
                            "Temperatuuriläved",
                        )
                if "Tmax max" in model_df.columns:
                    _x = pd.to_numeric(model_df["Tmax max"], errors="coerce").to_numpy(dtype=float)
                    for _thr in range(24, 35, 2):
                        _register_discovery_feature(
                            f"Päeva ülejääk üle {_thr}°",
                            np.maximum(0.0, _x - float(_thr)),
                            "Temperatuuriläved",
                        )

                # Lisa kõik genereeritud tunnused DataFrame'i ühe korraga.
                # See väldib pandas DataFrame fragmentation'it ja sadu aeglaseid insert-operatsioone.
                if autonomous_feature_values:
                    _auto_feature_df = pd.DataFrame(
                        autonomous_feature_values,
                        index=model_df.index,
                    )
                    model_df = pd.concat([model_df, _auto_feature_df], axis=1).copy()

                # Ideegeneraatori valikukallutatuse kaitse:
                # vanemad testipäevad = AVASTUS, hilisemad testipäevad = KINNITUS.
                # Kinnitusplokki EI kasutata ideede ega 2. ringi seemnete valimiseks.
                _auto_valid_mask = np.isfinite(champion_pred) & np.isfinite(y)
                _auto_test_days = sorted(set(dates[np.where(_auto_valid_mask)[0]]))
                if len(_auto_test_days) >= 4:
                    _confirm_day_count = max(2, int(round(len(_auto_test_days) * 0.30)))
                    _confirm_day_count = min(_confirm_day_count, len(_auto_test_days) - 2)
                else:
                    _confirm_day_count = 1 if len(_auto_test_days) >= 2 else 0

                _auto_confirm_days = set(_auto_test_days[-_confirm_day_count:]) if _confirm_day_count else set()
                _auto_discovery_days = set(_auto_test_days[:-_confirm_day_count]) if _confirm_day_count else set(_auto_test_days)

                def _autonomous_stats(candidate_pred, allowed_days=None):
                    mask = np.isfinite(champion_pred) & np.isfinite(candidate_pred) & np.isfinite(y)
                    idx = np.where(mask)[0]
                    if allowed_days is not None:
                        idx = np.array([i for i in idx if dates[i] in allowed_days], dtype=int)
                    if len(idx) == 0:
                        return None

                    champion_abs = np.abs(champion_pred[idx] - y[idx])
                    cand_abs = np.abs(candidate_pred[idx] - y[idx])
                    champion_mae_same = float(champion_abs.mean())
                    cand_mae = float(cand_abs.mean())
                    improvement = champion_mae_same - cand_mae
                    win_share = float(np.mean(cand_abs < champion_abs))

                    unique_days = sorted(set(dates[idx]))
                    halves = np.array_split(np.array(unique_days, dtype=object), 2)
                    half_improvements = []
                    for half in halves:
                        day_set = set(half.tolist())
                        h = np.array([i for i in idx if dates[i] in day_set], dtype=int)
                        if len(h):
                            half_improvements.append(float(
                                np.mean(np.abs(champion_pred[h] - y[h]))
                                - np.mean(np.abs(candidate_pred[h] - y[h]))
                            ))
                    min_half = min(half_improvements) if half_improvements else -999.0
                    stable = bool(
                        len(idx) >= 6
                        and improvement > 0.0
                        and win_share >= 0.50
                        and min_half >= -0.10
                    )
                    return {
                        "Championi MAE": champion_mae_same,
                        "Idee MAE": cand_mae,
                        "Paranemine": improvement,
                        "Võidab ridu %": win_share * 100.0,
                        "Halvim pool": min_half,
                        "Testiridu": int(len(idx)),
                        "Stabiilne": stable,
                    }

                # ESIMENE RING: ideed hinnatakse valikuks AINULT avastusplokis.
                first_round_rows = []
                _auto_prediction_cache = {}
                for _idea_name, _idea_col in autonomous_features:
                    _cols = list(champion_cols) + [_idea_col]
                    _pred = _walk_forward_with_extra(_cols)
                    _auto_prediction_cache[_idea_col] = _pred
                    _disc = _autonomous_stats(_pred, _auto_discovery_days)
                    autonomous_candidate_count += 1
                    if _disc:
                        _meta = next((m for m in autonomous_feature_meta if m["Veerg"] == _idea_col), {})
                        _row = {
                            "Idee": _idea_name,
                            "Kategooria": _meta.get("Kategooria", "—"),
                            "Ring": 1,
                            "_Veerg": _idea_col,
                            "Avastus MAE": _disc["Idee MAE"],
                            "Avastus champion": _disc["Championi MAE"],
                            "Avastus paranemine": _disc["Paranemine"],
                            "Avastus võidab %": _disc["Võidab ridu %"],
                            "Avastus ridu": _disc["Testiridu"],
                        }
                        first_round_rows.append(_row)

                first_round_df = pd.DataFrame(first_round_rows)
                if not first_round_df.empty:
                    first_round_df = first_round_df.sort_values(
                        ["Avastus paranemine", "Avastus võidab %"],
                        ascending=[False, False],
                    ).reset_index(drop=True)

                # TEINE RING: seemned valitakse ainult AVASTUSploki tulemuste järgi.
                second_round_seed = []
                if not first_round_df.empty:
                    promising = first_round_df[first_round_df["Avastus paranemine"] > -0.05].head(8)
                    second_round_seed = promising.to_dict("records")

                candidate_specs = [(r["Idee"], [r["_Veerg"]], 1, r["Kategooria"]) for r in first_round_rows]

                for _i, _left in enumerate(second_round_seed):
                    for _right in second_round_seed[_i + 1:]:
                        _left_col = _left.get("_Veerg")
                        _right_col = _right.get("_Veerg")
                        if not _left_col or not _right_col:
                            continue
                        _combo_name = f"{_left['Idee']} + {_right['Idee']}"
                        candidate_specs.append(
                            (_combo_name, [_left_col, _right_col], 2, "Teise ringi kombinatsioonid")
                        )
                        autonomous_candidate_count += 1
                        autonomous_category_counts["Teise ringi kombinatsioonid"] += 1

                # Alles nüüd vaatame valitud ideid hilisemas KINNITUSplokis.
                for _idea_name, _idea_cols, _ring, _category in candidate_specs:
                    _pred = _walk_forward_with_extra(list(champion_cols) + list(_idea_cols))
                    _disc = _autonomous_stats(_pred, _auto_discovery_days)
                    _conf = _autonomous_stats(_pred, _auto_confirm_days) if _auto_confirm_days else None
                    if not _disc:
                        continue

                    # "Stabiilne leid" nõuab positiivset tulemust eraldi hilisemas plokis.
                    _confirmed = bool(
                        _conf
                        and _disc["Paranemine"] > 0.0
                        and _disc["Võidab ridu %"] >= 50.0
                        and _conf["Paranemine"] > 0.0
                        and _conf["Võidab ridu %"] >= 50.0
                    )
                    autonomous_discovery_rows.append({
                        "Idee": _idea_name,
                        "Kategooria": _category,
                        "Ring": _ring,
                        "Avastus MAE": _disc["Idee MAE"],
                        "Avastus champion": _disc["Championi MAE"],
                        "Avastus paranemine": _disc["Paranemine"],
                        "Avastus võidab %": _disc["Võidab ridu %"],
                        "Kinnitus MAE": _conf["Idee MAE"] if _conf else np.nan,
                        "Kinnitus champion": _conf["Championi MAE"] if _conf else np.nan,
                        "Kinnitus paranemine": _conf["Paranemine"] if _conf else np.nan,
                        "Kinnitus võidab %": _conf["Võidab ridu %"] if _conf else np.nan,
                        "Kinnitus ridu": _conf["Testiridu"] if _conf else 0,
                        "Stabiilne": _confirmed,
                        "_Veerg": " + ".join(_idea_cols),
                    })

                autonomous_trace_df = pd.DataFrame(autonomous_discovery_rows)
                if not autonomous_trace_df.empty:
                    autonomous_trace_df = autonomous_trace_df.sort_values(
                        ["Stabiilne", "Kinnitus paranemine", "Avastus paranemine"],
                        ascending=[False, False, False],
                    ).reset_index(drop=True)


                # Salvesta ainult avastusmootori väljundid. Prognoosimudel ise ei tule cache'ist.
                st.session_state["_autonomous_discovery_cache"] = {
                    "key": _auto_cache_key,
                    "trace_df": autonomous_trace_df.copy(),
                    "candidate_count": int(autonomous_candidate_count),
                    "category_counts": dict(autonomous_category_counts),
                    "discovery_days": list(_auto_discovery_days),
                    "confirm_days": list(_auto_confirm_days),
                }

                # Lai otsing on kallis: säilita tulemus püsivalt, et järgmised
                # leheavamised ei peaks sama ideeruumi uuesti läbi arvutama.
                try:
                    _auto_store_df = autonomous_trace_df.drop(columns=["_Veerg"], errors="ignore").copy()
                    _auto_store_df = _auto_store_df.where(pd.notna(_auto_store_df), None)
                    db.set_app_setting(
                        "autonomous_discovery_trace_json",
                        json.dumps(_auto_store_df.to_dict(orient="records"), ensure_ascii=False),
                    )
                    db.set_app_setting(
                        "autonomous_discovery_category_counts_json",
                        json.dumps(autonomous_category_counts, ensure_ascii=False),
                    )
                    db.set_app_setting(
                        "autonomous_discovery_candidate_count",
                        str(int(autonomous_candidate_count)),
                    )
                    db.set_app_setting(
                        "idea_full_search_complete_day_count",
                        str(int(_complete_day_count)),
                    )
                    db.set_app_setting(
                        "idea_full_search_last_at",
                        datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                    )
                except Exception as _auto_store_exc:
                    db.set_app_setting("idea_full_search_last_error", str(_auto_store_exc))

            st.markdown("##### Tänane champion-mootor")
            ch1, ch2, ch3 = st.columns(3)
            ch1.metric("Champion", champion_name)
            ch2.metric("A+B+C MAE", "—" if champion_mae is None else f"{champion_mae:.2f} kasti")
            if champion_stats:
                ch3.metric("Eelis baasi ees", f"{champion_stats['Paranemine']:+.2f} kasti")
                st.success(
                    f"9 päeva prognoos kasutab automaatselt championit **{champion_name}**. "
                    f"See võitis {champion_stats['Võidab ridu %']:.0f}% samadest testiridadest ja "
                    f"parandas MAE-d {champion_stats['Paranemine']:.2f} kasti. "
                    "Champion vaadatakse iga uue korjega uuesti üle."
                )
            else:
                ch3.metric("Eelis baasi ees", "0.00 kasti")
                st.info(
                    "Ükski ilmast ega bioloogilisest koormusest tuletatud lisajälg ei läbinud täna stabiilsuslävendit. "
                    "9 päeva prognoos kasutab seetõttu puhast ilma + intervalli + hooaja faasi + põllu baasmudelit. "
                    "Toorest eelmist saaki, saagitrendi, XL-i ega C/B-d Jäljeotsija prognoosi ankruks ei luba."
                )

            # -------------------------------------------------------------------------
            # XL kvaliteedimootor — weather-first baas + ainult tõestatud mälutunnused
            # -------------------------------------------------------------------------
            st.markdown("##### XL komponent")
            st.caption(
                "XL baasmudel kasutab ilma, intervalli, hooaja faasi ja põllu identiteeti. "
                "Eelmise korje XL, A+B+C või kogusaak ei ole ankur; need võivad kasutusse "
                "pääseda ainult eraldi walk-forward stabiilsustesti kaudu."
            )

            xl_candidate_groups = {
                "Eelmine XL": ["XL -1"],
                "XL mälu 2 korjet": ["XL -1", "XL -2"],
                "Eelmine A+B+C": ["Eelmine ABC"],
                "Eelmine kogusaak": ["Eelmine saak"],
                "Korjejääk kombineeritud": ["XL -1", "XL -2", "Eelmine ABC", "Eelmine saak"],
                "Viimase 1 päeva ilm": ["T viim1", "Rad viim1", "Sade viim1", "ET0 viim1", "Niiskus viim1"],
                "Viimase 2 päeva ilm": ["T viim2", "Rad viim2", "Sade viim2", "ET0 viim2", "Niiskus viim2"],
                "Viimase 3 päeva ilm": ["T viim3", "Rad viim3", "Sade viim3", "ET0 viim3", "Niiskus viim3"],
            }

            def _xl_walk_with_extra(extra_cols, alpha=10.0):
                raw_extra = (
                    model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    if extra_cols else np.empty((len(model_df), 0))
                )
                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where((dates < test_day) & np.isfinite(y_xl))[0]
                    if len(train_idx) < min_train_rows:
                        continue
                    extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]
                    preds[test_idx] = _ridge_walk_predict(
                        y_xl, extra_arrays, train_idx, test_idx, alpha=alpha
                    )
                return preds

            def _xl_stability_stats(candidate_pred):
                mask = np.isfinite(xl_predictions) & np.isfinite(candidate_pred) & np.isfinite(y_xl)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return None
                base_abs = np.abs(xl_predictions[idx] - y_xl[idx])
                cand_abs = np.abs(candidate_pred[idx] - y_xl[idx])
                improvement = float(base_abs.mean() - cand_abs.mean())
                win_share = float(np.mean(cand_abs < base_abs))
                unique_days = sorted(set(dates[idx]))
                halves = np.array_split(np.array(unique_days, dtype=object), 2)
                half_improvements = []
                for half in halves:
                    day_set = set(half.tolist())
                    h = np.array([i for i in idx if dates[i] in day_set], dtype=int)
                    if len(h):
                        half_improvements.append(float(
                            np.mean(np.abs(xl_predictions[h] - y_xl[h]))
                            - np.mean(np.abs(candidate_pred[h] - y_xl[h]))
                        ))
                min_half = min(half_improvements) if half_improvements else -999.0
                stable = bool(
                    len(idx) >= 12
                    and improvement >= 0.05
                    and win_share >= 0.50
                    and min_half >= -0.05
                )
                return {
                    "Baas MAE": float(base_abs.mean()),
                    "Katse MAE": float(cand_abs.mean()),
                    "Paranemine": improvement,
                    "Võidab ridu %": win_share * 100.0,
                    "Halvim pool": min_half,
                    "Testiridu": int(len(idx)),
                    "Stabiilne": stable,
                }

            xl_candidate_predictions = {}
            xl_trace_rows = []
            for name, cols in xl_candidate_groups.items():
                pred = _xl_walk_with_extra(cols)
                xl_candidate_predictions[name] = pred
                stats = _xl_stability_stats(pred)
                if stats:
                    xl_trace_rows.append({"Jälg": name, **stats})

            xl_champion_name = "XL weather-first baasmudel"
            xl_champion_cols = []
            xl_champion_mae = (
                float(np.mean(np.abs(xl_predictions[valid_xl] - y_xl[valid_xl])))
                if valid_xl.any() else None
            )
            xl_champion_stats = None
            xl_stable = []
            for name, cols in xl_candidate_groups.items():
                stats = _xl_stability_stats(xl_candidate_predictions.get(name))
                if stats and stats["Stabiilne"]:
                    xl_stable.append((stats["Katse MAE"], name, cols, stats))
            if xl_stable:
                xl_stable.sort(key=lambda x: x[0])
                xl_champion_mae, xl_champion_name, xl_champion_cols, xl_champion_stats = xl_stable[0]

            xl1, xl2 = st.columns(2)
            xl1.metric("XL champion", xl_champion_name)
            xl2.metric("XL MAE", "—" if xl_champion_mae is None else f"{xl_champion_mae:.2f}")
            if xl_champion_stats:
                st.success(
                    f"XL kasutab **{xl_champion_name}**: MAE paranemine "
                    f"{xl_champion_stats['Paranemine']:.2f}, võitis "
                    f"{xl_champion_stats['Võidab ridu %']:.0f}% testiridadest."
                )
            else:
                st.info(
                    "Ükski XL mälutunnus ega lisajälg ei läbinud stabiilsuslävendit; "
                    "XL kasutab weather-first baasmudelit."
                )

            # -------------------------------------------------------------------------
            # C/B kvaliteedimootor — eraldi champion, ei mõjuta A+B+C championit
            # -------------------------------------------------------------------------
            st.markdown("##### C/B kvaliteedimootor")
            st.caption(
                "C/B prognoositakse eraldi. Baasmudel kasutab ilma, korjeintervalli, "
                "hooaja faasi ja põllu identiteeti. Eelmine C/B on ainult kandidaat ning "
                "pääseb kasutusse üksnes walk-forward testiga tõestatud stabiilse kasu korral."
            )

            valid_cb_base = np.isfinite(cb_predictions) & np.isfinite(y_cb)
            cb_base_mae = float(np.mean(np.abs(cb_predictions[valid_cb_base] - y_cb[valid_cb_base]))) if valid_cb_base.any() else None

            cb_candidate_groups = {
                "Eelmine C/B": ["C/B -1"],
                "C/B mälu 2 korjet": ["C/B -1", "C/B -2"],
                "A+B+C trend": ["Eelmine2 ABC", "ABC trend"],
                "XL / korjejääk": ["XL -1", "XL -2", "XL osakaal -1", "XL osakaal -2"],
                "Viimase 1 päeva ilm": ["T viim1", "Rad viim1", "Sade viim1", "ET0 viim1", "Niiskus viim1"],
                "Viimase 2 päeva ilm": ["T viim2", "Rad viim2", "Sade viim2", "ET0 viim2", "Niiskus viim2"],
                "Viimase 3 päeva ilm": ["T viim3", "Rad viim3", "Sade viim3", "ET0 viim3", "Niiskus viim3"],
            }

            def _cb_walk_with_extra(extra_cols, alpha=10.0):
                raw_extra = model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if extra_cols else np.empty((len(model_df), 0))
                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where((dates < test_day) & np.isfinite(log_y_cb))[0]
                    if len(train_idx) < min_train_rows:
                        continue
                    extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]
                    log_pred = _ridge_walk_predict(
                        log_y_cb, extra_arrays, train_idx, test_idx,
                        alpha=alpha, floor_zero=False
                    )
                    preds[test_idx] = np.exp(np.clip(log_pred, np.log(0.10), np.log(10.0)))
                return preds

            def _cb_stability_stats(candidate_pred):
                mask = np.isfinite(cb_predictions) & np.isfinite(candidate_pred) & np.isfinite(y_cb)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return None
                base_abs = np.abs(cb_predictions[idx] - y_cb[idx])
                cand_abs = np.abs(candidate_pred[idx] - y_cb[idx])
                improvement = float(base_abs.mean() - cand_abs.mean())
                win_share = float(np.mean(cand_abs < base_abs))
                unique_days = sorted(set(dates[idx]))
                halves = np.array_split(np.array(unique_days, dtype=object), 2)
                half_improvements = []
                for half in halves:
                    day_set = set(half.tolist())
                    h = np.array([i for i in idx if dates[i] in day_set], dtype=int)
                    if len(h):
                        half_improvements.append(float(
                            np.mean(np.abs(cb_predictions[h] - y_cb[h])) - np.mean(np.abs(candidate_pred[h] - y_cb[h]))
                        ))
                min_half = min(half_improvements) if half_improvements else -999.0
                stable = bool(len(idx) >= 12 and improvement >= 0.05 and win_share >= 0.50 and min_half >= -0.03)
                return {
                    "Baas MAE": float(base_abs.mean()), "Katse MAE": float(cand_abs.mean()),
                    "Paranemine": improvement, "Võidab ridu %": win_share * 100.0,
                    "Halvim pool": min_half, "Testiridu": int(len(idx)), "Stabiilne": stable,
                }

            cb_candidate_predictions = {}
            cb_trace_rows = []
            for name, cols in cb_candidate_groups.items():
                pred = _cb_walk_with_extra(cols)
                cb_candidate_predictions[name] = pred
                stats = _cb_stability_stats(pred)
                if stats:
                    cb_trace_rows.append({"Jälg": name, **stats})

            cb_champion_name = "C/B baasmudel"
            cb_champion_cols = []
            cb_champion_mae = cb_base_mae
            cb_champion_stats = None
            cb_stable = []
            for name, cols in cb_candidate_groups.items():
                stats = _cb_stability_stats(cb_candidate_predictions.get(name)) if name in cb_candidate_predictions else None
                if stats and stats["Stabiilne"]:
                    cb_stable.append((stats["Katse MAE"], name, cols, stats))
            if cb_stable:
                cb_stable.sort(key=lambda x: x[0])
                cb_champion_mae, cb_champion_name, cb_champion_cols, cb_champion_stats = cb_stable[0]

            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("C/B champion", cb_champion_name)
            cb2.metric("C/B MAE", "—" if cb_champion_mae is None else f"{cb_champion_mae:.2f}")
            cb_test_rows = int(valid_cb_base.sum())
            cb_test_days = len(set(dates[np.where(valid_cb_base)[0]])) if valid_cb_base.any() else 0
            cb3.metric("Testitud", f"{cb_test_rows}/{len(model_df)} rida")
            st.caption(f"C/B aus test hõlmab {cb_test_days} testipäeva. Madalam MAE on parem.")
            if cb_champion_stats:
                st.success(
                    f"C/B prognoos kasutab **{cb_champion_name}**: MAE paranemine {cb_champion_stats['Paranemine']:.2f}, "
                    f"võitis {cb_champion_stats['Võidab ridu %']:.0f}% samadest testiridadest."
                )
            else:
                st.info("Ükski C/B lisajälg ei läbinud stabiilsuslävendit; kvaliteediprognoos kasutab C/B baasmudelit.")

            if cb_trace_rows:
                with st.expander("Näita C/B jäljeotsija tulemusi"):
                    cb_trace_df = pd.DataFrame(cb_trace_rows).sort_values(["Stabiilne", "Paranemine"], ascending=[False, False])
                    st.dataframe(cb_trace_df.style.format({
                        "Baas MAE": "{:.2f}", "Katse MAE": "{:.2f}", "Paranemine": "{:+.2f}",
                        "Võidab ridu %": "{:.0f}%", "Halvim pool": "{:+.2f}",
                    }), use_container_width=True, hide_index=True)

            def _fit_full_generic(target, extra_arrays, alpha=10.0, field_alpha=80.0):
                idx = np.where(np.isfinite(target))[0]
                Xtr, _, means, scales, fills = _build_ridge_design(idx, idx, extra_arrays)
                penalty = np.eye(Xtr.shape[1]) * alpha
                penalty[0, 0] = 0.0
                # Partial pooling põldudele: 14 viimast koefitsienti saavad tugevama
                # regulaaristuse kui ilma/hooaja tunnused. Nii ei õpi 2–3 korjerea põhjal
                # mõne põllu jaoks kunstlikku suurt negatiivset/positiivset püsiefekti.
                penalty[-14:, -14:] = np.eye(14) * field_alpha
                beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ target
                return {"beta": beta, "means": means, "scales": scales, "fills": fills, "n_extra": len(extra_arrays)}

            def _predict_full_generic(model, field_no, base_values, extra_values, floor_zero=True):
                x = list(base_values)
                miss = []
                for i, value in enumerate(extra_values):
                    if value is None or not np.isfinite(float(value)):
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

            def _fit_full_abc_growth(extra_arrays, alpha=10.0, field_alpha=80.0, z_clip=2.5):
                idx = np.where(np.isfinite(log_y_abc))[0]
                Xtr, _, means, scales, fills = _build_ridge_design(idx, idx, extra_arrays)
                n_numeric = X_base.shape[1] + len(extra_arrays)
                Xtr[:, 1:1+n_numeric] = np.clip(Xtr[:, 1:1+n_numeric], -z_clip, z_clip)
                penalty = np.eye(Xtr.shape[1]) * alpha
                penalty[0, 0] = 0.0
                penalty[-14:, -14:] = np.eye(14) * field_alpha
                beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ log_y_abc[idx]
                return {
                    "beta": beta, "means": means, "scales": scales, "fills": fills,
                    "n_extra": len(extra_arrays), "z_clip": z_clip,
                }

            def _predict_full_abc_growth(model, field_no, base_values, extra_values):
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
                return float(np.exp(np.clip(latent, np.log(ABC_LOG_EPS), 6.0)))

            def _abc_growth_explain(model, field_no, base_values, extra_values, extra_names):
                """Jaga V6.4 log-mudeli prognoos täpselt +/- kastipanusteks.

                Referents on mudeli neutraalne treeningtase: standardiseeritud pidevad tunnused = 0,
                puuduvad lisatunnused = 0 ja põlluefekt = 0. Kuna exp() on mittelineaarne,
                skaleerime latentse lineaarse panuse ühiselt tagasi kastiskaalale. Nii kehtib:
                mudelibaas + kõigi tegurite panused = A+B+C prognoos.
                """
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

                x = np.array(x, dtype=float)
                z = (x - model["means"]) / model["scales"]
                z = np.clip(z, -model.get("z_clip", 2.5), model.get("z_clip", 2.5))

                beta = model["beta"]
                n_base = len(base_cont_cols)
                n_extra = len(extra_values)
                groups = {
                    "Temperatuur": 0.0, "Radiatsioon": 0.0, "Sademed": 0.0,
                    "Niiskus": 0.0, "ET0": 0.0, "Tuul": 0.0, "Tuul+stress": 0.0,
                    "Päevapikkus": 0.0,
                    "Intervall": 0.0, "Hooaeg": 0.0, "Põlluefekt": 0.0,
                    "Biokoormus": 0.0, "Muu": 0.0,
                }

                def _group_for_feature(name):
                    if name == "Intervall p":
                        return "Intervall"
                    if name == "Hooajapäev":
                        return "Hooaeg"
                    if (
                        name in {"T kesk", "Tmin kesk", "Tmax kesk", "Tmin min", "Tmax max"}
                        or str(name).startswith(("T viim", "ÖöT", "PäevT", "Tmin viim", "Tmax viim", "Öö ", "Päeva "))
                    ):
                        return "Temperatuur"
                    if str(name).startswith("Radiatsioon") or str(name).startswith("Rad viim"):
                        return "Radiatsioon"
                    if str(name).startswith("Sademed") or str(name).startswith("Sade viim"):
                        return "Sademed"
                    if str(name).startswith("Niiskus"):
                        return "Niiskus"
                    if str(name).startswith("ET0"):
                        return "ET0"
                    if str(name).startswith("Päevapikkus"):
                        return "Päevapikkus"
                    if str(name).startswith("Tuul×"):
                        return "Tuul+stress"
                    if str(name).startswith("Tuul"):
                        return "Tuul"
                    if name in {"Koormusindeks -1", "Ülekoormus -1", "2 korje koormus", "Tipukorje -1", "Tipukorje -2"}:
                        return "Biokoormus"
                    return "Muu"

                numeric_names = list(base_cont_cols) + list(extra_names)
                for j, name in enumerate(numeric_names):
                    groups[_group_for_feature(name)] += float(beta[1 + j] * z[j])

                # Lisatunnuste missing-indikaatorid kuuluvad sama teguri panuse sisse.
                miss_start = 1 + len(numeric_names)
                for j, name in enumerate(extra_names):
                    groups[_group_for_feature(name)] += float(beta[miss_start + j] * miss[j])

                field_start = miss_start + n_extra
                if 1 <= int(field_no) <= 14:
                    groups["Põlluefekt"] += float(beta[field_start + int(field_no) - 1])

                intercept = float(beta[0])
                latent_delta = float(sum(groups.values()))
                latent = intercept + latent_delta
                pred = float(np.exp(np.clip(latent, np.log(ABC_LOG_EPS), 6.0)))
                baseline = float(np.exp(np.clip(intercept, np.log(ABC_LOG_EPS), 6.0)))

                # Monotoonse exp() korral on see ühine positiivne skaalategur.
                # Panused säilitavad märgi ja summeeruvad täpselt prognoosini.
                if abs(latent_delta) > 1e-12:
                    scale = (pred - baseline) / latent_delta
                    crate_effects = {k: float(v * scale) for k, v in groups.items()}
                else:
                    crate_effects = {k: 0.0 for k in groups}

                return {
                    "baseline": baseline,
                    "effects": crate_effects,
                    "reconstructed": baseline + sum(crate_effects.values()),
                    "prediction": pred,
                }

            champion_extra_arrays = [
                pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float) for c in champion_cols
            ]
            full_abc_base_model = _fit_full_abc_growth([])
            full_abc_model = _fit_full_abc_growth(champion_extra_arrays)
            biological_load_feature_names = {
                c for cols in biological_load_candidate_groups.values() for c in cols
            }
            champion_uses_biological_load = any(c in biological_load_feature_names for c in champion_cols)
            xl_champion_extra_arrays = [
                pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float)
                for c in xl_champion_cols
            ]
            full_xl_model = _fit_full_generic(y_xl, xl_champion_extra_arrays)

            cb_champion_extra_arrays = [
                pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float)
                for c in cb_champion_cols
            ]
            full_cb_model = _fit_full_generic(log_y_cb, cb_champion_extra_arrays)

            # -------------------------------------------------------------------------
            # 9 päeva ette: A+B+C + eraldi XL
            # -------------------------------------------------------------------------
            st.markdown("##### 9 päeva saagiprognoos")
            st.caption(
                f"A+B+C kasutab tänast champion-mootorit: {champion_name}. Baasmudel põhineb ilmal, intervallil, "
                f"hooaja faasil ja põllu identiteedil. Tõestatud normaliseeritud bioloogiline koormus võib baasi korrigeerida, "
                f"kuid toores eelmine saak, saagitrend ja muud korjeajaloo mälutunnused ei saa prognoosi ankurdada. "
                f"XL kasutab eraldi championit: {xl_champion_name}. "
                f"C/B kasutab eraldi championit: {cb_champion_name}. "
                "Korjevahemiku möödunud päevadel kasutatakse mõõdetud ilma ja tulevastel päevadel 9 päeva ilmaprognoosi. "
                "Põllu eripära on mudelis partial-pooling kujul: väikese põllupõhise valimi korral ei lasta põlluefektil ilma mõju üle sõita."
            )
            st.info(
                "Oluline: ajalooline walk-forward MAE mõõdab saagimudelit realiseerunud mõõdetud ilma peal. "
                "Päris 1–9 päeva prognoos sisaldab lisaks ilmaprognoosi viga. Operatiivset täpsust hakkame mõõtma "
                "yield_forecasts snapshot'ide põhjal, kui tegelikud korjed saabuvad."
            )

            future_weather_end = TODAY + timedelta(days=9)
            all_weather_rows = db.get_weather_rows(readiness_start, future_weather_end)
            all_weather_by_day = {str(r.get("weather_date")): r for r in all_weather_rows}
            weather_feature_names = [
                "temp_night_avg_c", "temp_day_avg_c",
                "temp_min_c", "temp_max_c", "wind_avg_ms", "radiation_mj_m2",
                "humidity_avg_pct", "precipitation_mm", "et0_mm",
            ]

            def _nearest_weather_value(day_value, feature):
                """Puuduvat ilma ei asendata teiste päevade keskmisega."""
                return None

            def _weather_window_for_prediction(previous_day, target_day):
                rows, estimated_days = [], set()
                d = previous_day + timedelta(days=1)
                while d <= target_day:
                    src = all_weather_by_day.get(d.isoformat())
                    if not src:
                        return None, None
                    clean = {}
                    for feature in weather_feature_names:
                        value = src.get(feature)
                        try:
                            value = float(value) if value is not None else None
                        except (TypeError, ValueError):
                            value = None
                        if value is None:
                            value = _nearest_weather_value(d, feature)
                            if value is not None:
                                estimated_days.add(d)
                        if value is None:
                            return None, None
                        clean[feature] = value
                    rows.append(clean)
                    d += timedelta(days=1)
                if not rows:
                    return None, None
                tmins_all = [r["temp_min_c"] for r in rows]
                tmaxs_all = [r["temp_max_c"] for r in rows]
                night_all = [r["temp_night_avg_c"] for r in rows]
                day_all = [r["temp_day_avg_c"] for r in rows]
                mean_t = [(night + day) / 2 for night, day in zip(night_all, day_all)]
                wx = {
                    "Intervall p": (target_day - previous_day).days,
                    "Hooajapäev": (target_day - SEASON_START).days,
                    "T kesk": float(np.mean(mean_t)),
                    "ÖöT kesk": float(np.mean(night_all)),
                    "PäevT kesk": float(np.mean(day_all)),
                    "Tmin kesk": float(np.mean(tmins_all)),
                    "Tmax kesk": float(np.mean(tmaxs_all)),
                    "Tmin min": float(np.min(tmins_all)),
                    "Tmax max": float(np.max(tmaxs_all)),
                    "Soojad ööd 16+": int(sum(1 for v in night_all if v >= 16.0)),
                    "Soojad ööd 18+": int(sum(1 for v in night_all if v >= 18.0)),
                    "Jahedad ööd 12-": int(sum(1 for v in night_all if v <= 12.0)),
                    "Soojad ööd 16+ %": 100.0 * sum(1 for v in night_all if v >= 16.0) / len(night_all),
                    "Soojad ööd 18+ %": 100.0 * sum(1 for v in night_all if v >= 18.0) / len(night_all),
                    "Jahedad ööd 12- %": 100.0 * sum(1 for v in night_all if v <= 12.0) / len(night_all),
                    "Radiatsioon Σ": float(np.sum([r["radiation_mj_m2"] for r in rows])),
                    "Radiatsioon/p": float(np.mean([r["radiation_mj_m2"] for r in rows])),
                    "Sademed Σ": float(np.sum([r["precipitation_mm"] for r in rows])),
                    "Niiskus kesk": float(np.mean([r["humidity_avg_pct"] for r in rows])),
                    "ET0 Σ": float(np.sum([r["et0_mm"] for r in rows])),
                    "Tuul kesk": float(np.mean([r["wind_avg_ms"] for r in rows])),
                    "Päevapikkus": _daylength_hours(target_day),
                    "Päevapikkus Δ7p": _daylength_change_7d(target_day),
                    "Päevapikkus kasvukesk": float(np.mean([
                        _daylength_hours(previous_day + timedelta(days=i))
                        for i in range(1, (target_day - previous_day).days + 1)
                    ])) if (target_day - previous_day).days > 0 else _daylength_hours(target_day),
                }

                # Sama tuule koostoime arvutus peab olema õppimisel ja prognoosis identne.
                # Täpselt sama mittelineaarne temperatuuribaas nagu õppimisandmestikus.
                wx.update(_temperature_curve_features(night_all, day_all))

                wx["Tuul×Tmax"] = wx["Tuul kesk"] * wx["Tmax kesk"]
                wx["Tuul×Rad/p"] = wx["Tuul kesk"] * wx["Radiatsioon/p"]
                wx["Tuul×ET0/p"] = wx["Tuul kesk"] * (wx["ET0 Σ"] / max(1, len(rows)))
                wx["Tuul×Kuivus"] = wx["Tuul kesk"] * (100.0 - wx["Niiskus kesk"])
                for n in (1, 2, 3):
                    tail = rows[-min(n, len(rows)):]
                    tmins = [r["temp_min_c"] for r in tail]
                    tmaxs = [r["temp_max_c"] for r in tail]
                    nights = [r["temp_night_avg_c"] for r in tail]
                    days = [r["temp_day_avg_c"] for r in tail]
                    tvals = [(night + day) / 2 for night, day in zip(nights, days)]
                    wx[f"T viim{n}"] = float(np.mean(tvals))
                    wx[f"ÖöT viim{n}"] = float(np.mean(nights))
                    wx[f"PäevT viim{n}"] = float(np.mean(days))
                    wx[f"Tmin viim{n}"] = float(np.mean(tmins))
                    wx[f"Tmax viim{n}"] = float(np.mean(tmaxs))
                    wx[f"Rad viim{n}"] = float(np.sum([r["radiation_mj_m2"] for r in tail]))
                    wx[f"Sade viim{n}"] = float(np.sum([r["precipitation_mm"] for r in tail]))
                    wx[f"ET0 viim{n}"] = float(np.sum([r["et0_mm"] for r in tail]))
                    wx[f"Niiskus viim{n}"] = float(np.mean([r["humidity_avg_pct"] for r in tail]))
                    wx[f"Tuul viim{n}"] = float(np.mean([r["wind_avg_ms"] for r in tail]))
                    wx[f"Tuul×Tmax viim{n}"] = wx[f"Tuul viim{n}"] * wx[f"Tmax viim{n}"]
                    wx[f"Tuul×Rad/p viim{n}"] = wx[f"Tuul viim{n}"] * (wx[f"Rad viim{n}"] / len(tail))
                    wx[f"Tuul×ET0/p viim{n}"] = wx[f"Tuul viim{n}"] * (wx[f"ET0 viim{n}"] / len(tail))
                    wx[f"Tuul×Kuivus viim{n}"] = wx[f"Tuul viim{n}"] * (100.0 - wx[f"Niiskus viim{n}"])
                    wx[f"Päevapikkus viim{n}"] = float(np.mean([
                        _daylength_hours(target_day - timedelta(days=i))
                        for i in range(min(n, len(rows)))
                    ]))
                return wx, estimated_days

            def _champion_feature_values(state, wx):
                values = {
                    # Toored saagimälu väärtused on siin ainult diagnostika ühilduvuse jaoks;
                    # operatiivse championi nimekiri neid ei sisalda.
                    "Eelmine ABC": state.get("abc"),
                    "Eelmine2 ABC": state.get("abc_prev"),
                    "ABC trend": (state.get("abc") - state.get("abc_prev")) if state.get("abc") is not None and state.get("abc_prev") is not None else None,
                    "Eelmine saak": state.get("total"),
                    "XL -1": state.get("xl"),
                    "XL -2": state.get("xl_prev"),
                    "XL osakaal -1": (state.get("xl") / state.get("total")) if state.get("xl") is not None and state.get("total") not in (None, 0) else None,
                    "XL osakaal -2": (state.get("xl_prev") / state.get("total_prev")) if state.get("xl_prev") is not None and state.get("total_prev") not in (None, 0) else None,
                    "Koormusindeks -1": state.get("load_index"),
                    "Ülekoormus -1": state.get("overload"),
                    "2 korje koormus": state.get("load2_index"),
                    "Tipukorje -1": state.get("peak"),
                    "Tipukorje -2": state.get("peak_prev"),
                }
                for k, v in wx.items():
                    if (
                        k.startswith((
                            "T viim", "Tmin viim", "Tmax viim", "Rad viim", "Sade viim",
                            "ET0 viim", "Niiskus viim", "Tuul viim", "Tuul×"
                        ))
                        or k in {
                            "Tmin min", "Tmax max", "Soojad ööd 16+ %",
                            "Soojad ööd 18+ %", "Jahedad ööd 12- %",
                            "Tuul×Tmax", "Tuul×Rad/p", "Tuul×ET0/p", "Tuul×Kuivus",
                            "Päevapikkus", "Päevapikkus Δ7p", "Päevapikkus kasvukesk"
                        }
                        or k.startswith("Päevapikkus viim")
                    ):
                        values[k] = v
                return [values.get(c) for c in champion_cols]

            def _xl_champion_feature_values(state, wx):
                values = {
                    "XL -1": state.get("xl"),
                    "XL -2": state.get("xl_prev"),
                    "Eelmine ABC": state.get("abc"),
                    "Eelmine saak": state.get("total"),
                }
                for k, v in wx.items():
                    if k.startswith((
                        "T viim", "Tmin viim", "Tmax viim", "Rad viim", "Sade viim",
                        "ET0 viim", "Niiskus viim", "Tuul viim", "Tuul×"
                    )):
                        values[k] = v
                return [values.get(c) for c in xl_champion_cols]

            def _cb_champion_feature_values(state, wx):
                values = {
                    "C/B -1": state.get("cb"),
                    "C/B -2": state.get("cb_prev"),
                    "Eelmine2 ABC": state.get("abc_prev"),
                    "ABC trend": (state.get("abc") - state.get("abc_prev")) if state.get("abc") is not None and state.get("abc_prev") is not None else None,
                    "XL -1": state.get("xl"),
                    "XL -2": state.get("xl_prev"),
                    "XL osakaal -1": (state.get("xl") / state.get("total")) if state.get("xl") is not None and state.get("total") not in (None, 0) else None,
                    "XL osakaal -2": (state.get("xl_prev") / state.get("total_prev")) if state.get("xl_prev") is not None and state.get("total_prev") not in (None, 0) else None,
                }
                for k, v in wx.items():
                    if (
                        k.startswith((
                            "T viim", "Tmin viim", "Tmax viim", "Rad viim", "Sade viim",
                            "ET0 viim", "Niiskus viim", "Tuul viim", "Tuul×"
                        ))
                        or k in {
                            "Tmin min", "Tmax max", "Soojad ööd 16+ %",
                            "Soojad ööd 18+ %", "Jahedad ööd 12- %",
                            "Tuul×Tmax", "Tuul×Rad/p", "Tuul×ET0/p", "Tuul×Kuivus",
                            "Päevapikkus", "Päevapikkus Δ7p", "Päevapikkus kasvukesk"
                        }
                        or k.startswith("Päevapikkus viim")
                    ):
                        values[k] = v
                return [values.get(c) for c in cb_champion_cols]

            def _predict_one(field_no, state, target_day):
                wx, estimated_days = _weather_window_for_prediction(state["date"], target_day)
                if wx is None:
                    return None
                base_values = [wx[c] for c in base_cont_cols]
                # Bioloogilise koormuse korrektsioon on lubatud ainult siis, kui sihtkorje
                # eelmine sama põllu korje on päriselt mõõdetud. Kui vahepealne eelkorje on
                # ise tulevikuprognoos, ei toida me mudelit tema enda väljundiga tagasi.
                if champion_uses_biological_load and state.get("source") != "tegelik":
                    # Sama põld võib 9 päeva aknas tulla uuesti korjesse. Prognoositud eelkorjet
                    # ei kasutata järgmise korje biokoormuse sisendina.
                    abc_extra_values = [None for _ in champion_cols]
                    abc_mode = f"{champion_name} · koormus neutraalne"
                else:
                    abc_extra_values = _champion_feature_values(state, wx)
                    abc_mode = champion_name

                abc_pred = _predict_full_abc_growth(
                    full_abc_model, field_no, base_values, abc_extra_values,
                )
                abc_explain = _abc_growth_explain(
                    full_abc_model, field_no, base_values, abc_extra_values, champion_cols,
                )
                xl_pred = _predict_full_generic(
                    full_xl_model, field_no, base_values,
                    _xl_champion_feature_values(state, wx),
                )
                cb_log_pred = _predict_full_generic(
                    full_cb_model, field_no, base_values,
                    _cb_champion_feature_values(state, wx),
                    floor_zero=False,
                )
                cb_pred = float(np.exp(np.clip(cb_log_pred, np.log(0.10), np.log(10.0))))
                return {
                    "abc": abc_pred, "xl": xl_pred, "cb": cb_pred, "total": abc_pred + xl_pred,
                    "interval": wx["Intervall p"], "estimated_days": estimated_days or set(), "abc_mode": abc_mode,
                    "abc_explain": abc_explain, "wx": wx,
                }

            field_state = {}
            field_actual_abc_hist = {}
            field_peak_hist = {}
            for row in sorted(harvest_rows, key=lambda r: (str(r.get("harvest_date") or ""), int(r.get("harvest_order") or 99))):
                try:
                    d = date.fromisoformat(str(row.get("harvest_date")))
                    f = int(row.get("field_no"))
                    a = float(row.get("a")); b = float(row.get("b")); c = float(row.get("c")); xl = float(row.get("xl"))
                    total_value = float(row.get("total"))
                except (TypeError, ValueError):
                    continue
                if d < TODAY:
                    quality = str(row.get("data_quality") or "").strip().lower()
                    old = field_state.get(f)
                    abc_value = a + b + c
                    hist = field_actual_abc_hist.setdefault(f, [])
                    peak_hist = field_peak_hist.setdefault(f, [])
                    prior = hist[-3:]
                    baseline = float(np.mean(prior)) if prior else None
                    load_index = (abc_value / baseline) if (baseline is not None and baseline > 0) else None
                    overload = max(0.0, load_index - 1.0) if load_index is not None else None
                    load2_index = None
                    if prior:
                        load2_index = (float(np.mean([abc_value, prior[-1]])) / baseline) if baseline > 0 else None
                    peak = 1.0 if (load_index is not None and load_index >= 1.25) else 0.0 if load_index is not None else None
                    field_state[f] = {
                        "date": d, "abc": abc_value,
                        "abc_prev": old.get("abc") if old else None,
                        "xl": xl, "xl_prev": old.get("xl") if old else None,
                        "cb": (c / b) if b > 0 else None, "cb_prev": old.get("cb") if old else None,
                        "total": total_value, "total_prev": old.get("total") if old else None,
                        "load_index": load_index, "overload": overload, "load2_index": load2_index,
                        "peak": peak, "peak_prev": peak_hist[-1] if peak_hist else None,
                        "source": "tegelik",
                    }
                    if quality not in {"hinnanguline", "ligikaudne"}:
                        hist.append(abc_value)
                        peak_hist.append(peak)

            today_rows_live = db.get_harvest_for_day(TODAY)
            today_plan_default = _planned_fields_for_day(TODAY, today_rows_live, harvest_rows)

            # Avalehel võib kasutaja tänase 3 põllu valikust ühe eemaldada.
            # Kui sessionis pole valikut, kasutame automaatset plaani.
            selected_home = st.session_state.get("home_today_fields")
            if selected_home is not None:
                today_plan = [int(f) for f in list(selected_home)[:4]]
            else:
                today_plan = [int(f) for f in today_plan_default]

            today_actual = {int(r.get("field_no")): r for r in today_rows_live if r.get("field_no") is not None}

            # Tänane prognoos arvutatakse kõigile valitud põldudele ENNE tänaste
            # tegelike ridade rakendamist. Nii saab avalehel võrrelda prognoosi ja tegelikku.
            today_forecast_rows = []
            today_predictions_by_field = {}
            for f in today_plan:
                prev = field_state.get(int(f))
                if not prev:
                    continue
                pred = _predict_one(int(f), prev, TODAY)
                if not pred:
                    continue
                today_predictions_by_field[int(f)] = pred
                today_forecast_rows.append({
                    "Põld": int(f),
                    "A+B+C": pred["abc"],
                    "C/B": pred["cb"],
                    "XL": pred["xl"],
                    "Kokku": pred["total"],
                    "Intervall": pred["interval"],
                    "Alus": "tegelik eelkorje" if prev.get("source") == "tegelik" else "prognoositud eelkorje",
                    "Hinnanguline ilm": ", ".join(sorted(d.strftime("%d.%m") for d in pred["estimated_days"])) or "—",
                    "_ABC_selgitus": pred.get("abc_explain"),
                    "_WX": pred.get("wx"),
                })

            internal_today = []
            for f in today_plan:
                actual = today_actual.get(int(f))
                if actual:
                    try:
                        old = field_state.get(int(f))
                        a=float(actual.get("a")); b=float(actual.get("b")); c=float(actual.get("c")); xl=float(actual.get("xl")); total=float(actual.get("total"))
                        abc_value = a + b + c
                        hist = field_actual_abc_hist.setdefault(int(f), [])
                        peak_hist = field_peak_hist.setdefault(int(f), [])
                        prior = hist[-3:]
                        baseline = float(np.mean(prior)) if prior else None
                        load_index = (abc_value / baseline) if (baseline is not None and baseline > 0) else None
                        overload = max(0.0, load_index - 1.0) if load_index is not None else None
                        load2_index = (float(np.mean([abc_value, prior[-1]])) / baseline) if (prior and baseline and baseline > 0) else None
                        peak = 1.0 if (load_index is not None and load_index >= 1.25) else 0.0 if load_index is not None else None
                        field_state[int(f)] = {
                            "date": TODAY, "abc": abc_value,
                            "abc_prev": old.get("abc") if old else None,
                            "xl": xl, "xl_prev": old.get("xl") if old else None,
                            "cb": (c / b) if b > 0 else None, "cb_prev": old.get("cb") if old else None,
                            "total": total, "total_prev": old.get("total") if old else None,
                            "load_index": load_index, "overload": overload, "load2_index": load2_index,
                            "peak": peak, "peak_prev": peak_hist[-1] if peak_hist else None,
                            "source": "tegelik",
                        }
                        hist.append(abc_value); peak_hist.append(peak)
                        continue
                    except (TypeError, ValueError):
                        pass

                prev = field_state.get(int(f))
                pred = today_predictions_by_field.get(int(f))
                if not prev or not pred:
                    continue
                field_state[int(f)] = {
                    "date": TODAY, "abc": pred["abc"], "abc_prev": prev.get("abc"),
                    "xl": pred["xl"], "xl_prev": prev.get("xl"),
                    "cb": pred["cb"], "cb_prev": prev.get("cb"),
                    "total": pred["total"], "total_prev": prev.get("total"),
                    "load_index": None, "overload": None, "load2_index": None,
                    "peak": None, "peak_prev": prev.get("peak"),
                    "source": "prognoos",
                }
                internal_today.append((int(f), pred["total"]))

            forecast_days = []
            first_field = _next_field(today_plan[-1]) if today_plan else 1
            current_first = first_field
            any_weather_imputation = set()
            for offset in range(1, 10):
                target_day = TODAY + timedelta(days=offset)
                day_fields = [current_first, _next_field(current_first), _next_field(_next_field(current_first))]
                day_rows = []
                for f in day_fields:
                    prev = field_state.get(int(f))
                    if not prev:
                        day_rows.append({"Põld": f, "A+B+C": None, "C/B": None, "XL": None, "Kokku": None, "Intervall": None, "Alus": "puudub", "Hinnanguline ilm": "—"})
                        continue
                    result = _predict_one(int(f), prev, target_day)
                    if not result:
                        day_rows.append({"Põld": f, "A+B+C": None, "C/B": None, "XL": None, "Kokku": None, "Intervall": (target_day-prev["date"]).days, "Alus": prev["source"], "Hinnanguline ilm": "puudub"})
                        continue
                    any_weather_imputation.update(result["estimated_days"])
                    source_label = "tegelik eelkorje" if prev["source"] == "tegelik" else "prognoositud eelkorje"
                    if result.get("abc_mode") == "weather-first fallback":
                        source_label += " · ABC weather-first fallback"
                    day_rows.append({
                        "Põld": f, "A+B+C": result["abc"], "C/B": result["cb"], "XL": result["xl"], "Kokku": result["total"],
                        "Intervall": result["interval"], "Alus": source_label,
                        "Hinnanguline ilm": ", ".join(sorted(d.strftime("%d.%m") for d in result["estimated_days"])) or "—",
                        "_ABC_selgitus": result.get("abc_explain"), "_WX": result.get("wx"),
                    })
                    field_state[int(f)] = {
                        "date": target_day, "abc": result["abc"], "abc_prev": prev.get("abc"),
                        "xl": result["xl"], "xl_prev": prev.get("xl"),
                        "cb": result["cb"], "cb_prev": prev.get("cb"),
                        "total": result["total"], "total_prev": prev.get("total"),
                        "load_index": None, "overload": None, "load2_index": None,
                        "peak": None, "peak_prev": prev.get("peak"),
                        "source": "prognoos",
                    }
                forecast_days.append((target_day, day_rows))
                current_first = _next_field(day_fields[-1])

            # Salvestame 9 päeva prognoosid eraldi ajalukku. Sama päeva rerun uuendab
            # olemasolevat snapshot'i; järgmine päev loob uue lead-time snapshot'i.
            # Mudeliversioon tähistab champion-valiku raamistikku, mitte tänase võitja nime.
            # Nii uuendab sama päeva rerun sama operatiivset snapshot'i ka siis, kui champion
            # uue korje järel päeva jooksul muutub. Võitja nimi salvestub basis-väljale.
            MODEL_VERSION = "v6.4-growth-nonlinear-temp-nights-wind-daylength-v5"
            forecast_payloads = []

            # Salvesta ka tänase päeva prognoos lead=0 snapshotina.
            # See võimaldab hiljem näha, kuhu 5p/3p/1p/täna prognoos korjepäeva lähenedes liikus.
            # Tänase päeva snapshot lukustatakse enne esimese tegeliku korje sisestamist.
            if not today_rows_live:
                for row in today_forecast_rows:
                    if (
                        row.get("A+B+C") is None
                        or row.get("XL") is None
                        or row.get("C/B") is None
                        or row.get("Kokku") is None
                    ):
                        continue
                    forecast_payloads.append({
                        "forecast_date": TODAY.isoformat(),
                        "target_date": TODAY.isoformat(),
                        "field_no": int(row["Põld"]),
                        "lead_days": 0,
                        "abc_forecast": float(row["A+B+C"]),
                        "cb_forecast": float(row["C/B"]),
                        "xl_forecast": float(row["XL"]),
                        "total_forecast": float(row["Kokku"]),
                        "interval_days": row.get("Intervall"),
                        "basis": (
                            f"tänane tööprognoos; champion={champion_name}; "
                            f"xl_champion={xl_champion_name}; cb_champion={cb_champion_name}"
                        ),
                        "estimated_weather_days": row.get("Hinnanguline ilm") or "",
                        "model_version": MODEL_VERSION,
                    })

            for target_day, rows_day in forecast_days:
                for row in rows_day:
                    if (
                        row.get("A+B+C") is None
                        or row.get("XL") is None
                        or row.get("C/B") is None
                        or row.get("Kokku") is None
                    ):
                        continue
                    forecast_payloads.append({
                        "forecast_date": TODAY.isoformat(),
                        "target_date": target_day.isoformat(),
                        "field_no": int(row["Põld"]),
                        "lead_days": (target_day - TODAY).days,
                        "abc_forecast": float(row["A+B+C"]),
                        "cb_forecast": float(row["C/B"]),
                        "xl_forecast": float(row["XL"]),
                        "total_forecast": float(row["Kokku"]),
                        "interval_days": row.get("Intervall"),
                        "basis": (
                            f"{row.get('Alus') or ''}; champion={champion_name}; "
                            f"xl_champion={xl_champion_name}; cb_champion={cb_champion_name}"
                        ),
                        "estimated_weather_days": row.get("Hinnanguline ilm") or "",
                        "model_version": MODEL_VERSION,
                    })

            forecast_store_ok = False
            try:
                if db.yield_forecasts_available():
                    db.save_yield_forecasts(forecast_payloads)
                    forecast_store_ok = True
                else:
                    st.warning("Prognoosid arvutatakse, kuid neid ei salvestata veel: Supabase'is puudub yield_forecasts tabel. Käivita ZIP-is olev supabase_yield_forecasts.sql üks kord.")
            except db.DatabaseError as exc:
                st.warning(f"Prognoosid arvutatakse, kuid ajaloo salvestamine ebaõnnestus: {exc}")

            # Prognoosi liikumine: eelmine sama korjepäeva salvestatud snapshot vs praegune prognoos.
            # Kasutame ainult sama põllukomplekti täielikke snapshot'e.
            forecast_history_rows = []
            try:
                if db.yield_forecasts_available():
                    forecast_history_rows = db.get_yield_forecasts(limit=5000)
            except db.DatabaseError:
                forecast_history_rows = []

            def _forecast_adjustment(target_day_value, field_numbers, current_total):
                expected = {int(f) for f in field_numbers}
                if not expected or current_total is None:
                    return None, None, None

                # Kui sama forecast_date/field kohta on mitu model_version rida,
                # eelista kõige hiljem genereeritut.
                picked = {}
                for hist in forecast_history_rows:
                    if str(hist.get("target_date")) != target_day_value.isoformat():
                        continue
                    try:
                        fno = int(hist.get("field_no"))
                    except (TypeError, ValueError):
                        continue
                    if fno not in expected:
                        continue
                    fdate = str(hist.get("forecast_date") or "")
                    key = (fdate, fno)
                    old_hist = picked.get(key)
                    if old_hist is None or str(hist.get("generated_at") or "") > str(old_hist.get("generated_at") or ""):
                        picked[key] = hist

                by_date = {}
                for (fdate, fno), hist in picked.items():
                    by_date.setdefault(fdate, {})[fno] = hist

                complete = []
                for fdate, rows_for_date in by_date.items():
                    if set(rows_for_date.keys()) != expected:
                        continue
                    try:
                        total = sum(float(rows_for_date[f]["total_forecast"]) for f in expected)
                    except (TypeError, ValueError, KeyError):
                        continue
                    complete.append((fdate, total))

                if not complete:
                    return None, None, None

                complete.sort(key=lambda x: x[0])
                earlier = [(fdate, total) for fdate, total in complete if fdate < TODAY.isoformat()]
                if not earlier:
                    return None, None, None

                previous_date, previous_total = earlier[-1]
                if previous_total <= 0:
                    return None, previous_total, previous_date

                pct = (float(current_total) / previous_total - 1.0) * 100.0
                return pct, previous_total, previous_date


            def _motor_accuracy_3p():
                """Viimase 3 täieliku korjepäeva operatiivne prognoositäpsus.

                Iga päeva jaoks võetakse viimane täielik salvestatud prognoosisnapshot,
                mis oli olemas hiljemalt selle korjepäeva ajal. Sama päeva lead=0 snapshot
                on lubatud, sest see salvestatakse ainult enne esimese tegeliku korje sisestamist.

                Päeva täpsus = 100 - absoluutne protsentuaalne viga.
                Avalehe 3P näit = kolme päeva absoluutsete protsentuaalsete vigade keskmise
                pöördväärtus ehk 100 - keskmine absoluutne protsentuaalne viga.
                See näit ei lähe ühegi tulevase prognoosi sisendiks.
                """
                # Päris korjed päevade kaupa.
                actual_by_day = {}
                for hr in harvest_rows:
                    day_str = str(hr.get("harvest_date") or "")
                    try:
                        d = date.fromisoformat(day_str)
                        fno = int(hr.get("field_no"))
                        total = float(hr.get("total"))
                    except (TypeError, ValueError):
                        continue
                    if d >= TODAY:
                        continue
                    actual_by_day.setdefault(d, {})[fno] = total

                # Ainult täielikud 3 põllu korjepäevad.
                complete_days = [
                    d for d, rows_for_day in actual_by_day.items()
                    if len(rows_for_day) == 3 and len(set(rows_for_day.keys())) == 3
                ]
                complete_days.sort(reverse=True)

                evaluated = []
                for d in complete_days:
                    expected = set(actual_by_day[d].keys())
                    actual_total = sum(actual_by_day[d].values())
                    if actual_total <= 0:
                        continue

                    # Dedupe: sama forecast_date/põld puhul võta viimane genereeritud rida.
                    picked = {}
                    for hist in forecast_history_rows:
                        if str(hist.get("target_date")) != d.isoformat():
                            continue
                        fdate = str(hist.get("forecast_date") or "")
                        if not fdate or fdate > d.isoformat():
                            continue
                        try:
                            fno = int(hist.get("field_no"))
                        except (TypeError, ValueError):
                            continue
                        if fno not in expected:
                            continue
                        key = (fdate, fno)
                        old = picked.get(key)
                        if old is None or str(hist.get("generated_at") or "") > str(old.get("generated_at") or ""):
                            picked[key] = hist

                    by_fdate = {}
                    for (fdate, fno), hist in picked.items():
                        by_fdate.setdefault(fdate, {})[fno] = hist

                    complete_snapshots = []
                    for fdate, rows_for_date in by_fdate.items():
                        if set(rows_for_date.keys()) != expected:
                            continue
                        try:
                            forecast_total = sum(
                                float(rows_for_date[f]["total_forecast"]) for f in expected
                            )
                        except (TypeError, ValueError, KeyError):
                            continue
                        complete_snapshots.append((fdate, forecast_total))

                    if not complete_snapshots:
                        continue

                    # Korjele kõige lähem, aga mitte korjepäevast hilisem snapshot.
                    complete_snapshots.sort(key=lambda x: x[0])
                    forecast_date, forecast_total = complete_snapshots[-1]

                    abs_pct_error = abs(forecast_total / actual_total - 1.0) * 100.0
                    evaluated.append({
                        "day": d,
                        "forecast_date": forecast_date,
                        "actual_total": actual_total,
                        "forecast_total": forecast_total,
                        "abs_pct_error": abs_pct_error,
                    })

                    if len(evaluated) == 3:
                        break

                if len(evaluated) < 3:
                    return None, evaluated

                mean_abs_pct_error = float(np.mean([r["abs_pct_error"] for r in evaluated]))
                accuracy = max(0.0, min(100.0, 100.0 - mean_abs_pct_error))
                return accuracy, evaluated

            # Avalehe renderdus on teadlikult sellest raskest plokist eemaldatud.
            # Täna-vaade loeb ainult salvestatud yield_forecasts snapshoti ega käivita
            # walk-forward treeningut ega kandidaadikatseid.

            if internal_today:
                st.caption("Tänase poolelioleva korje sisemine tööprognoos: " + ", ".join(f"põld {f} ≈ {p:.1f}" for f,p in internal_today) + ". Tegelik kirje asendab selle automaatselt.")
            if any_weather_imputation:
                st.error(
                    "⚠️ Prognoos jäi puuduliku ilma tõttu hinnanguliseks. "
                    "Teiste päevade keskmisega ilma enam ei täideta."
                )

            # Prognoos-menüü detail algab tänasest päevast (lead=0), seejärel +1...+9 päeva.
            forecast_detail_days = [(TODAY, today_forecast_rows)] + forecast_days

            for target_day, rows_day in forecast_detail_days:
                vals = [r["Kokku"] for r in rows_day if r.get("Kokku") is not None]
                expected_count = len(rows_day)
                total_day = sum(vals) if expected_count > 0 and len(vals) == expected_count else None
                total_text = f"{_fmt(total_day)} kasti" if total_day is not None else ("korjet ei ole" if expected_count == 0 else "prognoos puudulik")
                lead = (target_day - TODAY).days
                if target_day == TODAY:
                    st.markdown(f"### {_weekday_letter(target_day)} {_short_date(target_day)}  {total_text}")
                    st.caption("täna · champion-mootori tööprognoos enne tänaste tegelike korjete mõju")
                elif lead >= 6:
                    header_html = (
                        '<div style="background:#fff3cd;border:1px solid #ffe69c;border-radius:10px;'
                        'padding:8px 12px;margin:12px 0 6px 0;">'
                        f'<span style="font-size:1.45rem;font-weight:700;">{_short_date(target_day)}&emsp;&emsp;{total_text}</span>'
                        f'<span style="margin-left:10px;font-size:0.92rem;">🟡 trendiprognoos · {lead} p ette</span>'
                        '</div>'
                    )
                    st.markdown(header_html, unsafe_allow_html=True)
                else:
                    confidence = "kõrgem kindlus" if lead <= 3 else "keskmine kindlus"
                    st.markdown(f"### {_short_date(target_day)}  {total_text}")
                    st.caption(f"{lead} p ette · {confidence}")
                day_df = pd.DataFrame(rows_day)
                visible_cols = [c for c in day_df.columns if not str(c).startswith("_")]
                day_df_visible = day_df[visible_cols]
                st.dataframe(day_df_visible.style.format({
                    "A+B+C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "C/B": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                    "XL": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "Kokku": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "Intervall": lambda v: "—" if pd.isna(v) else f"{int(v)} p",
                }), use_container_width=True, hide_index=True)

                explain_rows = []
                input_lines = []
                for r in rows_day:
                    ex = r.get("_ABC_selgitus")
                    wxr = r.get("_WX") or {}
                    if not ex:
                        continue
                    effects = ex.get("effects") or {}
                    erow = {
                        "Põld": r.get("Põld"),
                        "Mudelibaas": ex.get("baseline"),
                        "Temperatuur": effects.get("Temperatuur", 0.0),
                        "Radiatsioon": effects.get("Radiatsioon", 0.0),
                        "Sademed": effects.get("Sademed", 0.0),
                        "Niiskus": effects.get("Niiskus", 0.0),
                        "ET0": effects.get("ET0", 0.0),
                        "Tuul": effects.get("Tuul", 0.0),
                        "Intervall": effects.get("Intervall", 0.0),
                        "Hooaeg": effects.get("Hooaeg", 0.0),
                        "Põlluefekt": effects.get("Põlluefekt", 0.0),
                        "Biokoormus": effects.get("Biokoormus", 0.0),
                        "A+B+C": ex.get("prediction"),
                    }
                    if abs(float(effects.get("Muu", 0.0) or 0.0)) >= 0.01:
                        erow["Muu"] = effects.get("Muu", 0.0)
                    explain_rows.append(erow)
                    try:
                        input_lines.append(
                            f"põld {int(r.get('Põld'))}: Tmin {float(wxr.get('Tmin kesk')):.1f} °C (min {float(wxr.get('Tmin min')):.1f}) · "
                            f"Tmax {float(wxr.get('Tmax kesk')):.1f} °C (max {float(wxr.get('Tmax max')):.1f}) · "
                            f"sooje öid ≥16 °C {int(wxr.get('Soojad ööd 16+'))}/{int(wxr.get('Intervall p'))} · "
                            f"päev {float(wxr.get('Päevapikkus')):.2f} h ({float(wxr.get('Päevapikkus Δ7p')):+.2f} h/7p) · "
                            f"rad {float(wxr.get('Radiatsioon Σ')):.1f} MJ/m² ({float(wxr.get('Radiatsioon/p')):.1f}/p) · "
                            f"sade {float(wxr.get('Sademed Σ')):.1f} mm · RH {float(wxr.get('Niiskus kesk')):.0f}% · "
                            f"ET0 {float(wxr.get('ET0 Σ')):.1f} mm · intervall {int(wxr.get('Intervall p'))} p"
                        )
                    except (TypeError, ValueError):
                        pass

                if explain_rows:
                    st.caption("A+B+C selgitus · +/− = teguri panus kastides võrreldes mudeli neutraalse treeningtasemega")
                    explain_df = pd.DataFrame(explain_rows)
                    effect_cols = [c for c in explain_df.columns if c not in {"Põld", "Mudelibaas", "A+B+C"}]
                    fmt = {
                        "Mudelibaas": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                        "A+B+C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    }
                    for c in effect_cols:
                        fmt[c] = lambda v: "—" if pd.isna(v) else f"{float(v):+.1f}"
                    st.dataframe(explain_df.style.format(fmt), use_container_width=True, hide_index=True)
                    if input_lines:
                        st.caption("Sisendid · " + "  |  ".join(input_lines))

            abc_mae_text = f"{current_test_mae:.1f} kasti/põld" if current_test_mae is not None else "veel hindamata"
            total_mae_text = f"{current_total_test_mae:.1f} kasti/põld" if current_total_test_mae is not None else "veel hindamata"
            st.caption(
                f"A+B+C ajaline testviga: {abc_mae_text}. Kogusaagi (ABC+XL) testviga: {total_mae_text}. "
                f"1–3 päeva on operatiivne vaade, 4–5 päeva planeerimisvaade ja 6–9 päeva kollasega märgitud trendiprognoos. "
                "A+B+C prognoosi baas on weather-first. Tõestatud bioloogiline koormus võib korrigeerida ainult siis, kui eelkorje on päriselt mõõdetud; "
                "prognoositud eelkorjet ei kasutata koormuse tagasisidena. Kaugematel päevadel kasvab ebakindlus eelkõige ilmaprognoosi tõttu."
            )

            # -------------------------------------------------------------------------
            # Salvestatud prognoos vs tegelik
            # -------------------------------------------------------------------------
            if forecast_store_ok:
                st.markdown("##### Prognooside ajalugu")
                st.caption("Iga päev jääb alles uus 1–9 päeva ette tehtud snapshot. Tegelik korje liidetakse kuvamisel automaatselt harvests tabelist.")
                try:
                    saved_forecasts_all = db.get_yield_forecasts(limit=1000)
                    saved_forecasts = [r for r in saved_forecasts_all if str(r.get("model_version") or "") == MODEL_VERSION]
                except db.DatabaseError:
                    saved_forecasts_all = []
                    saved_forecasts = []
                actual_lookup = {}
                for hr in harvest_rows:
                    try:
                        key = (str(hr.get("harvest_date")), int(hr.get("field_no")))
                        a = float(hr.get("a")); b = float(hr.get("b")); c = float(hr.get("c")); xl = float(hr.get("xl")); total = float(hr.get("total"))
                        actual_lookup[key] = {"abc": a+b+c, "cb": (c/b) if b > 0 else None, "xl": xl, "total": total}
                    except (TypeError, ValueError):
                        continue

                history_rows = []
                for fr in saved_forecasts:
                    if str(fr.get("model_version")) != MODEL_VERSION:
                        continue
                    key = (str(fr.get("target_date")), int(fr.get("field_no")))
                    actual = actual_lookup.get(key)
                    try:
                        abc_f = float(fr.get("abc_forecast")); xl_f = float(fr.get("xl_forecast")); total_f = float(fr.get("total_forecast"))
                    except (TypeError, ValueError):
                        continue
                    history_rows.append({
                        "Prognoositud": str(fr.get("forecast_date")),
                        "Korjepäev": str(fr.get("target_date")),
                        "Põld": int(fr.get("field_no")),
                        "Ette": int(fr.get("lead_days") or 0),
                        "ABC prognoos": abc_f,
                        "C/B prognoos": float(fr.get("cb_forecast")) if fr.get("cb_forecast") is not None else None,
                        "XL prognoos": xl_f,
                        "Kokku prognoos": total_f,
                        "ABC tegelik": actual.get("abc") if actual else None,
                        "C/B tegelik": actual.get("cb") if actual else None,
                        "XL tegelik": actual.get("xl") if actual else None,
                        "Kokku tegelik": actual.get("total") if actual else None,
                        "Kogu viga": (total_f - actual["total"]) if actual else None,
                    })
                hist_df = pd.DataFrame(history_rows)
                if not hist_df.empty:
                    matured = hist_df[pd.notna(hist_df["Kokku tegelik"])].copy()
                    if not matured.empty:
                        hm1, hm2, hm3, hm4 = st.columns(4)
                        abs_err = matured["Kogu viga"].abs()
                        hm1.metric("Pärisprognoose hinnatud", len(matured))
                        hm2.metric("MAE", f"{abs_err.mean():.1f} kasti")
                        hm3.metric("±2 sees", f"{(abs_err <= 2).mean()*100:.0f}%")
                        one_day = matured[matured["Ette"] == 1]
                        hm4.metric("1 p MAE", "—" if one_day.empty else f"{one_day['Kogu viga'].abs().mean():.1f} kasti")
                    hist_show = hist_df.sort_values(["Korjepäev", "Põld", "Ette"], ascending=[False, True, True]).head(120).copy()
                    for dc in ("Prognoositud", "Korjepäev"):
                        hist_show[dc] = hist_show[dc].map(lambda x: date.fromisoformat(x).strftime("%d.%m") if x else "—")
                    st.dataframe(
                        hist_show.style.format({
                            "Ette": lambda v: f"{int(v)} p",
                            "ABC prognoos": "{:.1f}", "C/B prognoos": lambda v: "—" if pd.isna(v) else f"{v:.2f}", "XL prognoos": "{:.1f}", "Kokku prognoos": "{:.1f}",
                            "ABC tegelik": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                            "C/B tegelik": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                            "XL tegelik": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                            "Kokku tegelik": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                            "Kogu viga": lambda v: "—" if pd.isna(v) else f"{v:+.1f}",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("Esimesed prognoosisnapshot'id salvestatakse nüüd. Tegelike korjete lisandudes tekib siia võrdlus.")

            # -------------------------------------------------------------------------
            # Jäljeotsija — uurimisraport päriselt tehtud testide põhjal
            # -------------------------------------------------------------------------
            if valid_pred.any():
                st.markdown("##### Jäljeotsija — tänane uurimisraport")
                st.caption(
                    "Kõik allolevad väited tulevad samast ajaliselt ausast walk-forward testist. "
                    "Jäljeotsija ei lisa tunnust mootorisse pelgalt hea ühe tulemuse pärast: kasu peab olema piisav ja ajaliselt stabiilne."
                )

                if not trace_df.empty:
                    # Inimloetavad hüpoteesid. Need ei ole järeldused, vaid kirjeldavad, mida test päriselt proovis.
                    trace_hypotheses = {
                        "Eelmine A+B+C": "kas eelmise korje A+B+C annab ilma ja kasvudünaamika kõrval veel lisainfot",
                        "A+B+C trend 2 korjet": "kas kahe viimase korje saagisuund annab järgmise korje kohta lisasignaali",
                        "Korjejääk / kõrge eelkorje": "kas kõrge eelkorje koos XL-jäljega viitab vahele jäänud viljale või järelmõjule",
                        "XL osakaal 2 korjet": "kas XL osakaalu mälu kannab järgmisse korjesse kasutatavat signaali",
                        "Ebatavaline koormus -1": "kas eelmise korje ebatavaliselt suur A+B+C võrreldes sama põllu varasema tasemega jätab järgmisse korjesse taastumisefekti",
                        "Kahe korje koormus": "kas kahe järjestikuse korje keskmisest suurem koormus vähendab järgmise korje potentsiaali",
                        "Tipukorje järelmõju": "kas selge tipukorje mõju avaldub 1–2 korjet hiljem, mitte toore eelmise saagi ankurdusena",
                        "Viimase 1 päeva ilm": "kas korje-eelse viimase päeva ilm on tähtsam kui kogu korjevahemiku keskmine",
                        "Viimase 2 päeva ilm": "kas korje-eelse kahe päeva ilm annab eraldi lisasignaali",
                        "Viimase 3 päeva ilm": "kas korje-eelse kolme päeva ilm annab eraldi lisasignaali",
                        "C/B mälu 2 korjet (diagnostika)": "kas varasem suurusjaotus kannab järgmise korje A+B+C kohta diagnostilist infot",
                    }

                    def _why_not_stable(row):
                        reasons = []
                        if int(row.get("Testiridu", 0)) < 12:
                            reasons.append("testiridu on veel vähe")
                        if float(row.get("Paranemine", 0.0)) < 0.10:
                            reasons.append("MAE paranemine on alla 0,10 kasti")
                        if float(row.get("Võidab ridu %", 0.0)) < 50.0:
                            reasons.append("ei võida vähemalt pooli testiridu")
                        if float(row.get("Halvim pool", -999.0)) < -0.05:
                            reasons.append("mõju ei püsi testi mõlemas ajapooles")
                        return ", ".join(reasons) if reasons else "ei läbinud stabiilsusreeglit"

                    operational_trace = trace_df[trace_df["Jälg"].isin(operational_candidate_groups.keys())].copy()
                    operational_trace = operational_trace.sort_values("Paranemine", ascending=False)
                    memory_trace = trace_df[trace_df["Jälg"].isin(memory_diagnostic_groups.keys())].copy()
                    memory_trace = memory_trace.sort_values("Paranemine", ascending=False)

                    st.markdown("**Mida ma täna kaalusin**")
                    considered = []
                    for _, r in trace_df.sort_values("Paranemine", ascending=False).iterrows():
                        name = r["Jälg"]
                        role = ("weather-first kandidaat" if name in weather_candidate_groups else
                                "bioloogilise koormuse kandidaat" if name in biological_load_candidate_groups else
                                "uurimisjälg (ei juhi prognoosi)")
                        considered.append(f"**{name}** — {trace_hypotheses.get(name, 'lisatunnuse võimalik mõju')} · {role}")
                    if considered:
                        st.markdown(";  ".join(considered) + ".")

                    st.markdown("**Praegu parim mootor**")
                    if champion_stats:
                        st.success(
                            f"**{champion_name}** on tänane champion. Samadel testiridadel on MAE "
                            f"{champion_stats['Katse MAE']:.2f} kasti võrreldes baasi {champion_stats['Baas MAE']:.2f}-ga "
                            f"(paranemine {champion_stats['Paranemine']:.2f}). See võitis "
                            f"{champion_stats['Võidab ridu %']:.0f}% testiridadest ja mõju püsis ajaliselt piisavalt stabiilne."
                        )
                    else:
                        st.info(
                            f"**{champion_name}** jääb championiks. Ükski lisajälg ei tõestanud täna piisavalt stabiilset eelist. "
                            f"Weather-first baasi MAE on {champion_mae:.2f} kasti."
                        )

                    if not memory_trace.empty:
                        best_memory = memory_trace.iloc[0]
                        st.markdown("**Tugevaim ajalooline mälusignaal (ainult uurimiseks)**")
                        direction = "parandas" if float(best_memory["Paranemine"]) > 0 else "ei parandanud"
                        st.write(
                            f"**{best_memory['Jälg']}** {direction} samadel testiridadel MAE-d "
                            f"{best_memory['Paranemine']:+.2f} kasti ja võitis {best_memory['Võidab ridu %']:.0f}% ridadest. "
                            "Seda ei kasutata A+B+C operatiivse championina isegi siis, kui ajalooline sobivus on hea, "
                            "sest järsu ilmamuutuse korral peab prognoosi juhtima kasvuilm, mitte eelmise korje tase."
                        )

                    # Lähim kandidaat = parima paranemisega lubatud (ilm või bioloogiline koormus) kandidaat, mis pole champion.
                    nearest = None
                    for _, r in operational_trace.iterrows():
                        if r["Jälg"] != champion_name:
                            nearest = r
                            break
                    if nearest is not None:
                        st.markdown("**Lähim kandidaat**")
                        name = nearest["Jälg"]
                        if bool(nearest["Stabiilne"]):
                            status_text = "läbis stabiilsusreegli, kuid jäi tänasele championile alla"
                        else:
                            status_text = _why_not_stable(nearest)
                        st.write(
                            f"**{name}**: MAE {nearest['Katse MAE']:.2f} vs baas {nearest['Baas MAE']:.2f}; "
                            f"muutus {nearest['Paranemine']:+.2f} kasti, võitis {nearest['Võidab ridu %']:.0f}% ridadest. "
                            f"Praegu ei tõsta ma seda championiks, sest {status_text}."
                        )

                    watch = trace_df[(trace_df["Paranemine"] > 0) & (~trace_df["Stabiilne"])].sort_values("Paranemine", ascending=False).head(4)
                    if not watch.empty:
                        st.markdown("**Hoian silma peal**")
                        for _, r in watch.iterrows():
                            role = ("weather-first kandidaat" if r["Jälg"] in weather_candidate_groups else
                                    "bioloogilise koormuse kandidaat" if r["Jälg"] in biological_load_candidate_groups else
                                    "uurimisjälg")
                            st.write(
                                f"• **{r['Jälg']}** ({role}) — praegu {r['Paranemine']:+.2f} kasti MAE muutus; "
                                f"{_why_not_stable(r)}. Uute korjetega kontrollin seda uuesti."
                            )

                    rejected = trace_df[trace_df["Paranemine"] <= 0].sort_values("Paranemine", ascending=False).head(5)
                    if not rejected.empty:
                        st.markdown("**Praegu kõrvale jäetud**")
                        for _, r in rejected.iterrows():
                            st.write(
                                f"• **{r['Jälg']}** — ei parandanud tänases testis baasi "
                                f"(MAE muutus {r['Paranemine']:+.2f} kasti). See ei tähenda, et seost kindlasti pole; "
                                "praegused andmed ei toeta selle kasutamist championis."
                            )

                    with st.expander("Näita Jäljeotsija kõiki teste ja numbreid"):
                        show_trace = trace_df.copy()
                        st.dataframe(
                            show_trace.style.format({
                                "Baas MAE": "{:.2f}", "Katse MAE": "{:.2f}", "Paranemine": "{:+.2f}",
                                "Võidab ridu %": "{:.0f}%", "Halvim pool": "{:+.2f}",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                        st.caption(
                            f"Aktiivne A+B+C champion: {champion_name}. Valik arvutatakse uuesti iga uue korje järel. "
                            "Toored saagimälu tunnused on raportis diagnostilised; normaliseeritud bioloogiline koormus võib championiks saada ainult stabiilse tõendi korral. "
                            "Toortabel on alles kontrolliks; põhiinfo on ülal uurimisraportis."
                        )
        else:
            st.info("Täieliku ilmavahemiku ja numbrilise saagiga õppimisridu ei ole veel piisavalt.")

        st.markdown("##### Mida mudel hiljem sellest kasutab")
        st.write(
            "Põhimudeli õppimisnäite siht on konkreetse põllu järgmise korje A+B+C. XL õpitakse eraldi korjejäägi komponendina. Sisenditesse saab ilmavahemikust "
            "arvutada näiteks temperatuuri, radiatsiooni, sademete, õhuniiskuse ja ET0 summad/keskmised ning "
            "korjeintervalli. Toores eelmine saak ei ole prognoosi sisendankur; ainult sellest tuletatud ja walk-forward testiga tõestatud bioloogiline koormus võib weather-first baasi korrigeerida."
        )

        if last_complete_harvest and latest_harvest and latest_harvest > last_complete_harvest:
            st.caption(
                "Uuem pooleliolev korjepäev jääb õppimisest ajutiselt välja, kuni päeva korjeplokk on täielik."
            )

        st.divider()
        st.info("Mudel töötab kihiliselt: weather-first A+B+C baas + ainult tõestatud bioloogilise koormuse korrektsioon + eraldi XL ja C/B komponendid. Toores eelmine saak ei ole prognoosi ankur ega champion-tunnus.")

        # See Prognoosi/Mootori tähelepanekute arvutusring kontrollis olemasolevad
        # kandidaadid uute andmete peal üle. Märgi seis ajakohaseks.
        db.set_app_setting("model_dirty", "0")
        db.set_app_setting("model_last_checked_complete_day_count", str(len(complete_harvest_days)))
        db.set_app_setting("model_last_checked_at", datetime.now(ZoneInfo("Europe/Tallinn")).isoformat())

    if page == "Mootori tähelepanekud":
        _forecast_page_placeholder.empty()
if page == "Mootori tähelepanekud":
    st.subheader("Mootori tähelepanekud")
    if "AUTONOMOUS_DISCOVERY_ENABLED" in locals():
        if AUTONOMOUS_DISCOVERY_ENABLED:
            st.info("Autonoomse ideegeneraatori lai ring käivitus: täitus 3 uue täieliku korjepäeva intervall.")
        else:
            _last_full = db.get_app_setting("idea_full_search_complete_day_count", "0")
            st.info(
                f"Autonoomne ideegeneraator: lai uus otsing iga {IDEA_FULL_SEARCH_EVERY_COMPLETE_DAYS}. "
                f"täieliku 3/3 korjepäeva järel. Viimane lai ring oli täielike päevade arvu {_last_full} juures. "
                "Vahepeal kontrollib tavaline Jäljeotsija olemasolevaid kandidaate."
            )
    st.caption(
        "See leht tõlgendab walk-forward teste ja mootori enda genereeritud uusi tunnuseideid. "
        "Ideegeneraator võib otsida laialt; kasutuselevõtt jääb eraldi rangelt kontrollitud otsuseks."
    )

    _obs_ready = bool(training_rows) and "trace_df" in locals() and "champion_name" in locals()

    if not _obs_ready:
        st.info("Mootoril pole veel piisavalt ausaid walk-forward teste, et tähelepanekuid teha.")
    else:
        st.markdown("### ✅ Praegu usaldan")

        if champion_stats:
            st.success(
                f"**A+B+C champion: {champion_name}.** "
                f"Walk-forward testis on MAE {champion_stats['Katse MAE']:.2f} kasti/põld, "
                f"baasmudelil {champion_stats['Baas MAE']:.2f}. "
                f"Kandidaat võitis {champion_stats['Võidab ridu %']:.0f}% samadest testiridadest."
            )
        else:
            st.success(
                f"**A+B+C champion: {champion_name}.** "
                f"Weather-first baas püsib parim; selle walk-forward MAE on {champion_mae:.2f} kasti/põld. "
                "Ükski lubatud lisajälg pole veel tõestanud piisavalt stabiilset eelist."
            )

        if "cb_champion_name" in locals():
            if cb_champion_stats:
                st.success(
                    f"**C/B champion: {cb_champion_name}.** "
                    f"Walk-forward MAE {cb_champion_mae:.2f}; "
                    f"paranemine võrreldes baasiga {cb_champion_stats['Paranemine']:.2f}."
                )
            else:
                st.info(
                    f"**C/B: {cb_champion_name}.** "
                    "Ükski lisajälg ei ole veel läbinud C/B stabiilsusreeglit."
                )

        _operational_trace = trace_df[
            trace_df["Jälg"].isin(operational_candidate_groups.keys())
        ].copy()
        _operational_trace = _operational_trace.sort_values("Paranemine", ascending=False)

        _watch = _operational_trace[
            (_operational_trace["Paranemine"] > 0)
            & (~_operational_trace["Stabiilne"])
        ].head(4)


        st.markdown("### 🔎 Mootori enda avastused")

        if "autonomous_trace_df" not in locals() or autonomous_trace_df.empty:
            st.caption("Ideegeneraator ei saanud praegu piisava andmekvaliteediga uusi kandidaate testida.")
        else:
            st.markdown("#### Avastusruum")
            st.caption(
                "Avastusmootor: " + (
                    "kasutati sama andmestiku cache'i"
                    if "_auto_cache_hit" in locals() and _auto_cache_hit
                    else "arvutati uuesti, sest andmestik/champion muutus või cache puudus"
                )
            )
            _space_parts = [
                f"testitud kokku **{autonomous_candidate_count}** ideed",
                f"mittelineaarseid **{autonomous_category_counts.get('Mittelineaarsed', 0)}**",
                f"koostoimeid **{autonomous_category_counts.get('Koostoimed', 0)}**",
                f"suhtarve **{autonomous_category_counts.get('Suhtarvud', 0)}**",
                f"ajamuutusi **{autonomous_category_counts.get('Ajamuutused', 0)}**",
                f"temperatuurilävesid **{autonomous_category_counts.get('Temperatuuriläved', 0)}**",
                f"2. ringi kombinatsioone **{autonomous_category_counts.get('Teise ringi kombinatsioonid', 0)}**",
            ]
            st.info(" · ".join(_space_parts))
            st.caption(
                f"1. ring ja 2. ring valitakse ainult vanemas avastusplokis "
                f"({_fmt(len(_auto_discovery_days), 0)} testipäeva). "
                f"Hilisemad {_fmt(len(_auto_confirm_days), 0)} testipäeva on eraldi kinnitusplokk. "
                "Ühtegi enda leitud ideed EI võeta automaatselt kasutusse."
            )

            _auto_top = autonomous_trace_df.head(10)
            for _, _r in _auto_top.iterrows():
                _status = "✅ kinnitatud leid" if bool(_r["Stabiilne"]) else "🧪 uurimisleid"
                st.write(
                    f"• **{_r['Idee']}** — {_status} · ring {int(_r.get('Ring', 1))}; "
                    f"avastus MAE {_r['Avastus MAE']:.2f} "
                    f"(Δ {_r['Avastus paranemine']:+.2f}), "
                    f"kinnitus MAE {_r['Kinnitus MAE']:.2f} "
                    f"(Δ {_r['Kinnitus paranemine']:+.2f}), "
                    f"kinnituses võitis {_r['Kinnitus võidab %']:.0f}% ridadest."
                )

            _better = autonomous_trace_df[autonomous_trace_df["Stabiilne"] == True]
            if _better.empty:
                st.info(
                    "Praegu ei ole ühtegi enda leitud ideed, mis oleks eraldi hilisemas "
                    "kinnitusplokis championi üle löönud."
                )
            else:
                _best = _better.iloc[0]
                st.success(
                    f"Parim eraldi kinnitatud mootori enda leid: **{_best['Idee']}**. "
                    f"Kinnitus-MAE {_best['Kinnitus MAE']:.2f} vs champion "
                    f"{_best['Kinnitus champion']:.2f}; paranemine "
                    f"{_best['Kinnitus paranemine']:.2f} kasti. "
                    "See jääb endiselt ainult kandidaadiks."
                )

            with st.expander("Näita kõiki mootori enda kandidaatide teste"):
                _show_auto = autonomous_trace_df.drop(columns=["_Veerg"], errors="ignore")
                st.dataframe(
                    _show_auto.style.format({
                        "Avastus MAE": "{:.2f}",
                        "Avastus champion": "{:.2f}",
                        "Avastus paranemine": "{:+.2f}",
                        "Avastus võidab %": "{:.0f}%",
                        "Kinnitus MAE": "{:.2f}",
                        "Kinnitus champion": "{:.2f}",
                        "Kinnitus paranemine": "{:+.2f}",
                        "Kinnitus võidab %": "{:.0f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### 👀 Hoian silma peal")

        if _watch.empty:
            st.caption(
                "Praegu pole ühtegi lubatud lisajälge, mis parandaks baasi, "
                "kuid jääks napilt stabiilsuslävendi alla."
            )
        else:
            for _, _r in _watch.iterrows():
                _name = str(_r["Jälg"])

                _role = (
                    "ilmastikujälg"
                    if _name in weather_candidate_groups
                    else "bioloogilise koormuse jälg"
                    if _name in biological_load_candidate_groups
                    else "lisajälg"
                )

                st.write(
                    f"• **{_name}** ({_role}) — "
                    f"MAE muutus {_r['Paranemine']:+.2f} kasti, "
                    f"võitis {_r['Võidab ridu %']:.0f}% testiridadest, "
                    "kuid ei ole veel piisavalt stabiilne."
                )

        _rejected = _operational_trace[
            _operational_trace["Paranemine"] <= 0
        ].head(5)

        st.markdown("### ⛔ Praegu ei kasuta")

        if _rejected.empty:
            st.caption(
                "Praegu pole lubatud kandidaate, "
                "mis oleksid walk-forward testis baasist selgelt halvemad."
            )
        else:
            for _, _r in _rejected.iterrows():
                st.write(
                    f"• **{_r['Jälg']}** — ei parandanud baasmudelit "
                    f"(MAE muutus {_r['Paranemine']:+.2f} kasti)."
                )

        if "memory_diagnostic_groups" in locals():
            _memory_trace = trace_df[
                trace_df["Jälg"].isin(memory_diagnostic_groups.keys())
            ].copy()

            _memory_trace = _memory_trace.sort_values(
                "Paranemine",
                ascending=False
            )

            if not _memory_trace.empty:
                _best_memory = _memory_trace.iloc[0]

                st.divider()
                st.markdown("#### 🧪 Ainult uurimiseks")
                st.write(
                    f"Tugevaim saagimälu diagnostiline signaal on praegu "
                    f"**{_best_memory['Jälg']}** "
                    f"(MAE muutus {_best_memory['Paranemine']:+.2f} kasti; "
                    f"võitis {_best_memory['Võidab ridu %']:.0f}% ridadest). "
                    "Seda ei kasutata A+B+C operatiivse prognoosi ankruna."
                )

        st.caption(
            "Tähelepanekud arvutatakse uuesti iga uue korje järel "
            "samadest walk-forward tulemustest. "
            "Leht ise ei õpeta ega muuda mootorit."
        )
