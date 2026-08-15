from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import math
import hashlib
import json
import time
import re

from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import db
from core import WeatherService
from model_engine import (
    temperature_curve_features as _temperature_curve_features,
    daylength_hours as _daylength_hours,
    daylength_change_7d as _daylength_change_7d,
    season_curve_features as _season_curve_features,
    chronological_discovery_confirmation_days as _engine_discovery_confirmation_days,
    build_ridge_design as _engine_build_ridge_design,
    ridge_walk_predict as _engine_ridge_walk_predict,
    abc_growth_walk_predict as _engine_abc_growth_walk_predict,
    fit_full_generic as _engine_fit_full_generic,
    predict_full_generic as _engine_predict_full_generic,
    fit_full_abc_growth as _engine_fit_full_abc_growth,
    predict_full_abc_growth as _engine_predict_full_abc_growth,
)

# V19 AUDIT: Pärnu sade=proxy; ET0 koond; radar taastatud; feature-signature dedupe; aus laia vooru edukus.
TODAY = datetime.now(ZoneInfo("Europe/Tallinn")).date()
SEASON_START = date(TODAY.year, 6, 15)
IDEA_FULL_SEARCH_EVERY_COMPLETE_DAYS = 3
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "kurgimootor_icon.png"
APP_PATCH_VERSION = "app-120"  # sisemine failiversioon; DB forecast MODEL_VERSION jääb teadlikult samaks
APP_ICON = Image.open(APP_ICON_PATH)
st.set_page_config(page_title="KurgiMootor V6.4", page_icon=APP_ICON, layout="wide")

st.markdown("""<style>
/* Tänaste põldude valitud sildid: kurgiroheline.
   Streamlit/BaseWeb võib renderdada tag'i div või span elemendina. */
.stMultiSelect [data-baseweb="tag"],
div[data-baseweb="select"] [data-baseweb="tag"],
span[data-baseweb="tag"],
div[data-baseweb="tag"] {
    background: #9BCB7A !important;
    background-color: #9BCB7A !important;
    color: #1F3B22 !important;
    border-color: #7FB35F !important;
}

/* Tag'i tekst ja eemaldamise X */
.stMultiSelect [data-baseweb="tag"] *,
div[data-baseweb="select"] [data-baseweb="tag"] *,
span[data-baseweb="tag"] *,
div[data-baseweb="tag"] * {
    color: #1F3B22 !important;
    fill: #355C37 !important;
}

/* Hover */
.stMultiSelect [data-baseweb="tag"]:hover,
div[data-baseweb="select"] [data-baseweb="tag"]:hover,
span[data-baseweb="tag"]:hover,
div[data-baseweb="tag"]:hover {
    background: #8FC46D !important;
    background-color: #8FC46D !important;
}
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Tänaste põldude valitud sildid: kurgiroheline, mitte veapunane. */
div[data-baseweb="tag"] {
    background-color: #9BCB7A !important;
    color: #1F3B22 !important;
    border: 1px solid #7FB35F !important;
}
div[data-baseweb="tag"] svg {
    fill: #355C37 !important;
}
div[data-baseweb="tag"]:hover {
    background-color: #8FC46D !important;
}
</style>
""", unsafe_allow_html=True)

# Streamliti multiselecti tag'i tegelik sisemine DOM võib versiooniti muutuda.
# CSS selector ei kata kõiki versioone, seega värvime ainult tänaste põldude
# valitud numbriklotsid parent-DOM-is ja jälgime rerun'e MutationObserveriga.
components.html(
    """
    <script>
    (() => {
      const doc = window.parent.document;
      const GREEN = '#9BCB7A';
      const GREEN_HOVER = '#8FC46D';
      const BORDER = '#7FB35F';
      const TEXT = '#1F3B22';
      const ICON = '#355C37';

      function paintTag(tag) {
        if (!tag || tag.dataset.kurgiGreen === '1') return;
        tag.dataset.kurgiGreen = '1';
        tag.style.setProperty('background', GREEN, 'important');
        tag.style.setProperty('background-color', GREEN, 'important');
        tag.style.setProperty('color', TEXT, 'important');
        tag.style.setProperty('border-color', BORDER, 'important');

        tag.querySelectorAll('*').forEach(el => {
          el.style.setProperty('color', TEXT, 'important');
          if (el.tagName && el.tagName.toLowerCase() === 'svg') {
            el.style.setProperty('fill', ICON, 'important');
          }
        });

        tag.addEventListener('mouseenter', () => {
          tag.style.setProperty('background', GREEN_HOVER, 'important');
          tag.style.setProperty('background-color', GREEN_HOVER, 'important');
        });
        tag.addEventListener('mouseleave', () => {
          tag.style.setProperty('background', GREEN, 'important');
          tag.style.setProperty('background-color', GREEN, 'important');
        });
      }

      function findTagFromNumberNode(node, root) {
        let el = node;
        for (let i = 0; i < 7 && el && el !== root; i++, el = el.parentElement) {
          const txt = (el.textContent || '').trim().replace(/\\s+/g, ' ');
          const hasRemove = !!el.querySelector('svg, button');
          // Tag on väike element: üks põllunumber + eemaldamise ikoon.
          if (/^(?:[1-9]|1[0-4])(?:\\s*[×xX])?$/.test(txt) && hasRemove) {
            return el;
          }
        }
        return null;
      }

      function apply() {
        const roots = doc.querySelectorAll('.stMultiSelect, [data-testid="stMultiSelect"]');
        roots.forEach(root => {
          // Piirame skripti just selle multiselectiga, mille label on Täna korjatavad põllud.
          const block = root.closest('[data-testid="stElementContainer"]') || root.parentElement;
          const context = (block && block.textContent ? block.textContent : root.textContent || '');
          if (!context.includes('Täna korjatavad põllud')) return;

          root.querySelectorAll('span, div').forEach(node => {
            const txt = (node.textContent || '').trim();
            if (!/^(?:[1-9]|1[0-4])$/.test(txt)) return;
            const tag = findTagFromNumberNode(node, root);
            if (tag) paintTag(tag);
          });
        });
      }

      apply();
      const observer = new MutationObserver(() => apply());
      observer.observe(doc.body, {childList: true, subtree: true});
      window.setInterval(apply, 1200);
    })();
    </script>
    """,
    height=0,
    width=0,
)

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


def _mark_forecast_dirty(reason: str) -> None:
    """Märgi, et operatiivne 9 päeva prognoos vajab värskendust ilma Jäljeotsijata."""
    db.set_app_setting("forecast_dirty", "1")
    db.set_app_setting("forecast_dirty_reason", str(reason))
    db.set_app_setting("forecast_dirty_at", datetime.now(ZoneInfo("Europe/Tallinn")).isoformat())


def _forecast_is_dirty() -> bool:
    return db.get_app_setting("forecast_dirty", "1") == "1"


def _load_json_setting(key: str, default):
    try:
        raw = db.get_app_setting(key, "")
        value = json.loads(raw) if raw else default
        return value
    except Exception:
        return default


def _save_json_setting(key: str, value) -> None:
    try:
        db.set_app_setting(key, json.dumps(value, ensure_ascii=False))
    except Exception:
        pass


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


def _dateday_label(value) -> str:
    """Kuva kuupäev nädalapäeva ühetähelise lühendiga."""
    try:
        d = value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
        return f"{_weekday_letter(d)} {_short_date(d)}"
    except Exception:
        return str(value)


def _snapshot_batch_map(rows, target_day_value, max_forecast_date=None):
    """Tagasta forecast_date -> field_no -> row ühe ja sama salvestusbatch'i kaupa.

    yield_forecasts upsert uuendab ainult samade põldude ridu. Kui päeva plaan muutub,
    võivad vana batch'i põllud tabelisse alles jääda. generated_at on ühe save-kutse
    kõigil ridadel sama, seega EI tohi eri generated_at batch'e põllu kaupa kokku liita.
    """
    target_iso = target_day_value.isoformat() if hasattr(target_day_value, "isoformat") else str(target_day_value)
    max_fdate = (
        max_forecast_date.isoformat()
        if hasattr(max_forecast_date, "isoformat")
        else (str(max_forecast_date) if max_forecast_date else None)
    )
    grouped = {}
    for row in rows or []:
        if str(row.get("target_date") or "") != target_iso:
            continue
        fdate = str(row.get("forecast_date") or "")
        if not fdate or (max_fdate and fdate > max_fdate):
            continue
        try:
            fno = int(row.get("field_no"))
        except (TypeError, ValueError):
            continue
        generated_at = str(row.get("generated_at") or "")
        model_version = str(row.get("model_version") or "")
        batch_key = (fdate, generated_at, model_version)
        grouped.setdefault(batch_key, {})[fno] = row

    by_fdate = {}
    rank_by_fdate = {}
    for (fdate, generated_at, model_version), field_rows in grouped.items():
        rank = (generated_at, model_version)
        if fdate not in rank_by_fdate or rank > rank_by_fdate[fdate]:
            rank_by_fdate[fdate] = rank
            by_fdate[fdate] = field_rows
    return by_fdate


def _latest_snapshot_batch_rows(rows, target_day_value):
    """Viimase forecast_date viimane terviklik salvestusbatch; batch'e ei segata."""
    by_fdate = _snapshot_batch_map(rows, target_day_value)
    if not by_fdate:
        return []
    latest_fdate = sorted(by_fdate.keys())[-1]
    return list(by_fdate[latest_fdate].values())

def _temperature_cell_style(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    degree = int(round(v))
    degree = max(-10, min(40, degree))
    palette = {
        -10: "background-color: #5B2C83; color: #111111;",
        -9: "background-color: #61358E; color: #111111;",
        -8: "background-color: #684099; color: #111111;",
        -7: "background-color: #704AA4; color: #111111;",
        -6: "background-color: #7854AF; color: #111111;",
        -5: "background-color: #805FBA; color: #111111;",
        -4: "background-color: #6B6FD1; color: #111111;",
        -3: "background-color: #6279DB; color: #111111;",
        -2: "background-color: #5984E5; color: #111111;",
        -1: "background-color: #508EEF; color: #111111;",
        0: "background-color: #4899F7; color: #111111;",
        1: "background-color: #53A6FA; color: #111111;",
        2: "background-color: #5FB2FC; color: #111111;",
        3: "background-color: #6BBEFD; color: #111111;",
        4: "background-color: #78C9FE; color: #111111;",
        5: "background-color: #86D4FE; color: #111111;",
        6: "background-color: #95DEFF; color: #111111;",
        7: "background-color: #A7E6FF; color: #111111;",
        8: "background-color: #B9EDFF; color: #111111;",
        9: "background-color: #CCF3FF; color: #111111;",
        10: "background-color: #E0F7FF; color: #111111;",
        11: "background-color: #F2F4F5; color: #111111;",
        12: "background-color: #F8F0DA; color: #111111;",
        13: "background-color: #FBE7B7; color: #111111;",
        14: "background-color: #FCDE92; color: #111111;",
        15: "background-color: #FDD66E; color: #111111;",
        16: "background-color: #FCCD4E; color: #111111;",
        17: "background-color: #FBC338; color: #111111;",
        18: "background-color: #F8B526; color: #111111;",
        19: "background-color: #F6A51B; color: #111111;",
        20: "background-color: #F49517; color: #111111;",
        21: "background-color: #F18418; color: #111111;",
        22: "background-color: #ED731B; color: #111111;",
        23: "background-color: #E96020; color: #111111;",
        24: "background-color: #E44E27; color: #111111;",
        25: "background-color: #DE3D30; color: #111111;",
        26: "background-color: #D92F39; color: #111111;",
        27: "background-color: #D22342; color: #111111;",
        28: "background-color: #CA194A; color: #111111;",
        29: "background-color: #C01152; color: #111111;",
        30: "background-color: #B70A58; color: #111111;",
        31: "background-color: #AD075E; color: #111111;",
        32: "background-color: #A30563; color: #111111;",
        33: "background-color: #990468; color: #111111;",
        34: "background-color: #8F046D; color: #111111;",
        35: "background-color: #850572; color: #111111;",
        36: "background-color: #7A0677; color: #111111;",
        37: "background-color: #70077B; color: #111111;",
        38: "background-color: #65097F; color: #111111;",
        39: "background-color: #5B0B83; color: #111111;",
        40: "background-color: #510D87; color: #111111;",
    }
    return palette.get(degree, "")



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


# Ilm kontrollitakse automaatselt üks kord päevas. Ilm ja operatiivne 9 päeva
# prognoos on eraldatud Jäljeotsijast: ilma muutus EI märgi uurimismootorit dirty.
# Ametlikud päevaelemendid (eriti radiatsioon) võivad pärast südaööd viibida,
# seetõttu kontrollime puudulikku eilset päeva sama päeva jooksul uuesti, kuid
# mitte sagedamini kui kord 2 tunni jooksul.
try:
    _weather_service = WeatherService()
    _weather_refresh_before = db.get_app_setting("weather_last_refresh_at", "")
    _weather_service.auto_refresh_if_needed(TODAY)
    _weather_refresh_after = db.get_app_setting("weather_last_refresh_at", "")
    _auto_weather_changed = bool(
        _weather_refresh_after and _weather_refresh_after != _weather_refresh_before
    )
    if _auto_weather_changed:
        _mark_forecast_dirty("uus või uuendatud ilm")

    _yesterday = TODAY - timedelta(days=1)
    _incomplete_yesterday = db.get_incomplete_measured_dates(_yesterday, _yesterday)
    if _incomplete_yesterday:
        # Kui sama skriptikäivitus just tegi päeva esimese automaatse refresh'i,
        # ei tee me kohe teist identset võrguringi otsa. Esimene hilinenud-komponendi
        # korduskontroll saab toimuda 2 h pärast.
        _retry_due = not _auto_weather_changed
        if _auto_weather_changed:
            db.set_app_setting(
                "weather_incomplete_retry_at",
                datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
            )
        _last_retry_raw = db.get_app_setting("weather_incomplete_retry_at", "")
        if _retry_due and _last_retry_raw:
            try:
                _last_retry_dt = datetime.fromisoformat(_last_retry_raw).astimezone(ZoneInfo("Europe/Tallinn"))
                _retry_due = (datetime.now(ZoneInfo("Europe/Tallinn")) - _last_retry_dt).total_seconds() >= 2 * 3600
            except Exception:
                _retry_due = True
        if _retry_due:
            # Korduskontroll peab olema odav: kontrollime AINULT puuduvat mõõdetud
            # eilset ilma. 10 päeva ilmaprognoosi siin ei värskendata ja operatiivset
            # saagiprognoosi ei käivitata enne, kui eilne päev päriselt täielikuks saab.
            _incomplete_before = set(
                db.get_incomplete_measured_dates(_yesterday, _yesterday) or []
            )
            try:
                _retry_result = _weather_service._refresh_measured_incremental(TODAY)
            except Exception as _retry_exc:
                _retry_result = {"error": str(_retry_exc)}
                db.set_app_setting("weather_last_error", f"Mõõdetud ilma korduskontroll: {_retry_exc}")

            _retry_now = datetime.now(ZoneInfo("Europe/Tallinn"))
            db.set_app_setting("weather_incomplete_retry_at", _retry_now.isoformat())
            _incomplete_after = set(
                db.get_incomplete_measured_dates(_yesterday, _yesterday) or []
            )

            # Alles siis, kui varem puudulik eilne päev sai nüüd päriselt roheliseks,
            # on uuel mõõdetud infol mõtet käivitada üks uus operatiivprognoos.
            if _incomplete_before and not _incomplete_after:
                db.set_app_setting("weather_last_refresh_at", _retry_now.isoformat(timespec="seconds"))
                db.set_app_setting(
                    "weather_last_result",
                    f"Hilinenud mõõdetud ilm valmis; lisatud {_retry_result.get('saved', 0)} päeva",
                )
                _mark_forecast_dirty("hilinenud mõõdetud ilm saabus")
except Exception as exc:
    db.set_app_setting("weather_last_error", f"Automaatne ilmauuendus: {exc}")

_hdr_logo, _hdr_title = st.columns([1, 7])
with _hdr_logo:
    st.image(str(APP_ICON_PATH), width=72)
with _hdr_title:
    st.markdown("## KurgiMootor V6.4")
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

    # Päeva tööplaan säilitatakse andmebaasis. Nii tähendab "täielik korjepäev"
    # päriselt seda, et kõik selleks päevaks valitud põllud on sisestatud — mitte 3/3.
    saved_today_plan = db.get_harvest_plan(TODAY)
    initial_today_plan = list(saved_today_plan) if saved_today_plan is not None else list(today_planned_fields)

    # Päeva vahetudes taastame salvestatud plaani; vanal/veel salvestamata päeval
    # kasutame senist automaatset põlluplaani.
    current_home_day = TODAY.isoformat()
    if st.session_state.get("home_plan_day") != current_home_day:
        st.session_state["home_plan_day"] = current_home_day
        st.session_state["home_today_fields"] = list(initial_today_plan)

    if "home_today_fields" not in st.session_state:
        st.session_state["home_today_fields"] = list(initial_today_plan)

    selected_today_fields = st.multiselect(
        "Täna korjatavad põllud",
        options=list(range(1, 15)),
        key="home_today_fields",
        max_selections=4,
        help="Muuda ainult tänase tööplaani. Saagi sisestamine käib Korjed-menüüs.",
    )

    selected_today_fields = [int(f) for f in selected_today_fields]
    if saved_today_plan != selected_today_fields:
        db.save_harvest_plan(TODAY, selected_today_fields)

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
        """target_date snapshotid batch-kaupa; eri generated_at ringe ei segata."""
        return _snapshot_batch_map(_home_saved, target_day_value)

    def _latest_snapshot_rows(target_day_value):
        return _latest_snapshot_batch_rows(_home_saved, target_day_value)

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
        """Muutus sama korjepäeva esimese 9 päeva ette tehtud snapshot'i suhtes."""
        current = _snapshot_totals(current_rows)
        if not current or current["total"] <= 0:
            return None
        expected = {int(r.get("field_no")) for r in current_rows if r.get("field_no") is not None}
        by_date = _home_snapshot_map(target_day_value)

        base_rows = []
        for fdate, rows_for_date in by_date.items():
            if set(rows_for_date.keys()) != expected:
                continue
            try:
                forecast_day = date.fromisoformat(str(fdate))
                total = sum(float(rows_for_date[f]["total_forecast"]) for f in expected)
            except (TypeError, ValueError, KeyError):
                continue
            if (target_day_value - forecast_day).days == 9:
                base_rows.append((str(fdate), total))

        if not base_rows:
            return None
        base_rows.sort(key=lambda x: x[0])
        first_total = base_rows[0][1]
        if first_total <= 0:
            return None
        return (current["total"] / first_total - 1.0) * 100.0


    def _home_motor_accuracy_3p():
        """3P täpsus viimase 3 TÄIELIKU korjepäeva põhjal, sh tänane kui valmis."""
        harvest_rows_home = db.get_harvest_history(limit=1000)
        try:
            harvest_plans_home = db.get_harvest_plans()
        except Exception:
            harvest_plans_home = {}

        actual_by_day = {}
        for hr in harvest_rows_home:
            try:
                d = date.fromisoformat(str(hr.get("harvest_date") or ""))
                fno = int(hr.get("field_no"))
                total = float(hr.get("total"))
            except (TypeError, ValueError):
                continue
            # Tulevikku ei vaata, aga tänane päev on lubatud kohe, kui päev on täielik.
            if d > TODAY:
                continue
            actual_by_day.setdefault(d, {})[fno] = total

        complete_days = []
        for d, rows_for_day in actual_by_day.items():
            day_key = d.isoformat()
            saved_plan = harvest_plans_home.get(day_key)
            if saved_plan is not None:
                expected_fields = {int(f) for f in saved_plan}
                is_complete = (
                    bool(expected_fields)
                    and set(rows_for_day.keys()) == expected_fields
                    and len(rows_for_day) == len(expected_fields)
                )
            else:
                # Vanad päevad, millele tööplaani DB-s ei salvestatud.
                is_complete = (
                    len(rows_for_day) == 3
                    and len(set(rows_for_day.keys())) == 3
                )
            if is_complete:
                complete_days.append(d)

        complete_days = sorted(complete_days, reverse=True)

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
            adj_text = "—" if adj_pct is None else (
                "0%" if abs(adj_pct) < 0.5 else f"{'↑' if adj_pct > 0 else '↓'} {adj_pct:+.0f}%"
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
            adj_text = "—" if adj_pct is None else (
                "0%" if abs(adj_pct) < 0.5 else f"{'↑' if adj_pct > 0 else '↓'} {adj_pct:+.0f}%"
            )
            far_forecast = lead >= 6
            bg = "#fff3cd" if far_forecast else "rgba(0,0,0,0.025)"
            border = "#ffe69c" if far_forecast else "rgba(128,128,128,0.20)"
            badge = ""

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

    # Korjevormi valik sõltub valitud kuupäevast. Tänasel päeval võib sisestada
    # ainult tänases tööplaanis olevaid põlde; minevikus saab kõiki põlde parandada.
    # Kuupäeva vahetamisel ei kanta eelmise päeva "järgmise põllu" session-state'i üle.
    today_rows_for_form = db.get_harvest_for_day(TODAY)
    history_for_form = db.get_harvest_history()
    auto_planned_for_form = _planned_fields_for_day(TODAY, today_rows_for_form, history_for_form)
    saved_plan_for_form = db.get_harvest_plan(TODAY)
    planned_for_form = (
        list(saved_plan_for_form)
        if saved_plan_for_form is not None and saved_plan_for_form
        else list(auto_planned_for_form)
    )
    form_version = int(st.session_state.get("harvest_form_version", 0))

    st.markdown("#### Lisa või paranda korje")

    if st.session_state.get("harvest_saved_message"):
        st.success(st.session_state.pop("harvest_saved_message"))

    c1, c2, c3 = st.columns(3)
    entry_date = c1.date_input(
        "Kuupäev", value=TODAY, max_value=TODAY, key="manual_harvest_date"
    )
    selected_day_rows = db.get_harvest_for_day(entry_date)
    ordered_selected_rows = sorted(
        selected_day_rows,
        key=lambda r: (int(r.get("harvest_order") or 99), int(r.get("field_no") or 99)),
    )

    _context_day = entry_date.isoformat()
    _context_changed = st.session_state.get("harvest_form_context_date") != _context_day
    st.session_state["harvest_form_context_date"] = _context_day

    if entry_date == TODAY:
        # Tänase valiku allikas on salvestatud tööplaan; kui seda veel pole, kasutame
        # sama automaatset plaani, mida Avaleht. Põld 8 ei saa pärast 5/6/7 lõppu
        # lihtsalt "järgmise numbrina" vormi tekkida.
        today_allowed_fields = [int(f) for f in planned_for_form]
        if not today_allowed_fields:
            today_allowed_fields = [
                int(r.get("field_no")) for r in ordered_selected_rows
                if r.get("field_no") is not None
            ]
        if not today_allowed_fields:
            today_allowed_fields = [1]
        field_options = list(dict.fromkeys(today_allowed_fields))

        entered_today = {
            int(r.get("field_no")) for r in selected_day_rows
            if r.get("field_no") is not None
        }
        missing_today = [f for f in field_options if f not in entered_today]
        if missing_today:
            inferred_field = int(missing_today[0])
        elif ordered_selected_rows:
            inferred_field = int(ordered_selected_rows[0].get("field_no"))
        else:
            inferred_field = int(field_options[0])

        _session_next = st.session_state.get("next_harvest_field")
        if (not _context_changed) and _session_next is not None and int(_session_next) in field_options:
            default_field = int(_session_next)
        else:
            default_field = inferred_field
    else:
        # Minevikus võib kõiki põlde parandada/lisada, kuid kuupäeva vahetamisel
        # avame vaikimisi selle päeva esimese päriselt korjatud põllu.
        field_options = list(range(1, 15))
        if ordered_selected_rows:
            default_field = int(ordered_selected_rows[0].get("field_no"))
        else:
            default_field = 1

    field_index = field_options.index(default_field) if default_field in field_options else 0
    entry_field = c2.selectbox(
        "Põld",
        field_options,
        index=field_index,
        key=f"manual_harvest_field_{form_version}_{entry_date.isoformat()}",
    )

    existing_row = next(
        (
            r for r in selected_day_rows
            if int(r.get("field_no") or r.get("field_id") or 0) == int(entry_field)
        ),
        None,
    )

    if existing_row:
        existing_order = int(existing_row.get("harvest_order") or 1)
    elif entry_date == TODAY and int(entry_field) in field_options:
        existing_order = field_options.index(int(entry_field)) + 1
    else:
        existing_order = 1

    entry_order = c3.selectbox(
        "Järjekord",
        [1, 2, 3, 4],
        index=max(0, min(3, existing_order - 1)),
        key=f"manual_harvest_order_{form_version}_{entry_date.isoformat()}_{entry_field}",
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
            _today_allowed = set(int(f) for f in planned_for_form)
            if entry_date > TODAY:
                st.error("Tuleviku kuupäevale korjet sisestada ei saa.")
            elif entry_date == TODAY and (not _today_allowed or int(entry_field) not in _today_allowed):
                st.error("Tänasele päevale saab sisestada ainult tänases tööplaanis olevaid põlde.")
            elif existing_row and not change_confirmed:
                st.warning("Olemasoleva korje muutmiseks või kustutamiseks märgi kinnitus.")
            elif existing_row and total_preview <= 0:
                db.delete_harvest(entry_date, entry_field)
                _mark_forecast_dirty(f"korje kustutatud {entry_date} põld {entry_field}")
                st.session_state["harvest_form_version"] = form_version + 1
                st.session_state["harvest_saved_message"] = (
                    f"Kustutatud: {entry_date} · põld {entry_field}"
                )
                st.rerun()
            elif total_preview <= 0:
                st.warning("Uut 0-korjet ei salvestata. Sisesta vähemalt üks kogus.")
            else:
                db.save_harvest(
                    entry_date, entry_field, 0,
                    entry_a, entry_b, entry_c, entry_xl,
                    harvest_order=entry_order,
                )
                _mark_forecast_dirty(f"korjeandmed uuendatud {entry_date}")

                _saved_day_rows = db.get_harvest_for_day(entry_date)
                _saved_day_fields = {
                    int(r.get("field_no"))
                    for r in _saved_day_rows
                    if r.get("field_no") is not None
                }
                _saved_plan = db.get_harvest_plan(entry_date)
                if _saved_plan is not None:
                    _expected_fields = set(int(f) for f in _saved_plan)
                    _day_complete = (
                        bool(_expected_fields)
                        and _saved_day_fields == _expected_fields
                        and len(_saved_day_rows) == len(_expected_fields)
                    )
                else:
                    _day_complete = len(_saved_day_rows) == 3 and len(_saved_day_fields) == 3
                if _day_complete:
                    _mark_model_dirty(f"täielik korjepäev {entry_date}")

                # Järgmine vaikimisi valik jääb sama päeva plaani sisse. Kui kõik
                # tänased põllud on valmis, läheme parandamisrežiimis plaani esimesele põllule.
                if entry_date == TODAY and _saved_plan:
                    _plan_list = [int(f) for f in _saved_plan]
                    _missing_after = [f for f in _plan_list if f not in _saved_day_fields]
                    _next_field_value = _missing_after[0] if _missing_after else _plan_list[0]
                    st.session_state["next_harvest_field"] = int(_next_field_value)
                    st.session_state["next_harvest_order"] = _plan_list.index(int(_next_field_value)) + 1
                else:
                    _ordered_after = sorted(
                        _saved_day_rows,
                        key=lambda r: (int(r.get("harvest_order") or 99), int(r.get("field_no") or 99)),
                    )
                    _fields_after = [int(r.get("field_no")) for r in _ordered_after if r.get("field_no") is not None]
                    if _fields_after:
                        try:
                            _pos = _fields_after.index(int(entry_field))
                            _next_field_value = _fields_after[min(_pos + 1, len(_fields_after) - 1)]
                        except ValueError:
                            _next_field_value = _fields_after[0]
                        st.session_state["next_harvest_field"] = int(_next_field_value)
                        _next_row = next((r for r in _ordered_after if int(r.get("field_no") or 0) == int(_next_field_value)), None)
                        st.session_state["next_harvest_order"] = int((_next_row or {}).get("harvest_order") or 1)

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
    st.caption("Temperatuur ja tuul: Häädemeeste jaam. Radiatsioon, õhuniiskus ja sademed: Pärnu jaam. ET0 arvutab KurgiMootor.")
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
        _mark_forecast_dirty("käsitsi uuendatud ilm")
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
            # Ajutise staatuse puhul märgime ainult need komponendid, mis võivad
            # pärineda Pärnu asemel varem salvestatud prognoosist.
            if "radiatsioon" in msg_lower:
                temporary_features.add("radiation_mj_m2")
            if "niiskus" in msg_lower or "rh" in msg_lower:
                temporary_features.add("humidity_avg_pct")
            if "sademed" in msg_lower or "sade" in msg_lower:
                temporary_features.add("precipitation_mm")

            # ET0 on ajutine, kui mõni selle kasutatud väline sisend on ajutine.
            if temporary_features.intersection(
                {"radiation_mj_m2", "humidity_avg_pct"}
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
            "Kuupäev": _dateday_label(r["weather_date"]),
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
        "Kuupäev": _dateday_label(r["weather_date"]),
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
        "Olek": "🔵 Prognoos" if r.get("checked") else "Vigane prognoos",
    } for r in forecast_rows]
    forecast_df = pd.DataFrame(forecast_display)
    if not forecast_df.empty:
        forecast_styled = forecast_df.style.format({
            "Öö kesk °C": "{:.1f}",
            "Päev kesk °C": "{:.1f}",
            "Min °C": "{:.1f}",
            "Max °C": "{:.1f}",
            "Tuul m/s": "{:.1f}",
            "Niiskus %": "{:.0f}",
            "Sademed mm": "{:.1f}",
            "ET0 mm": "{:.2f}",
            "Radiatsioon MJ/m²": "{:.2f}",
        }, na_rep="—")
        forecast_styled = forecast_styled.map(
            _temperature_cell_style,
            subset=["Min °C", "Max °C"],
        )
        st.dataframe(forecast_styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# V20 CPU-kaitse: Jäljeotsija ja lai uuring jagavad ühist, kasutaja muudetavat
# päevast uurimisaja eelarvet. Need on KurgiMootori enda wall-clock sekundid,
# mitte Streamlit Cloudi ametlik CPU-kvoot. Operatiivne ilm + 9 päeva prognoos
# ei kuulu sellesse uurimispotti.
CPU_RESEARCH_DAILY_DEFAULT_S = 40.0
CPU_LIGHT_TARGET_DEFAULT_S = 20.0
CPU_LAYERED_MAX_DEFAULT_S = 20.0
CPU_RESEARCH_DAILY_HARD_CAP_S = 50.0
CPU_COMPONENT_HARD_CAP_S = 35.0
CPU_LAYERED_MIN_LAUNCH_S = 12.0
CPU_AUTO_SAFETY_RESERVE_S = 5.0

def _cpu_setting_float(key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(db.get_app_setting(key, str(default)) or default)
    except Exception:
        value = float(default)
    return max(float(lo), min(float(hi), value))

CPU_RESEARCH_DAILY_BUDGET_S = _cpu_setting_float(
    "cpu_research_daily_budget_s", CPU_RESEARCH_DAILY_DEFAULT_S, 10.0, CPU_RESEARCH_DAILY_HARD_CAP_S
)
CPU_LIGHT_TARGET_S = _cpu_setting_float(
    "cpu_light_target_s", CPU_LIGHT_TARGET_DEFAULT_S, 5.0, CPU_COMPONENT_HARD_CAP_S
)
CPU_LAYERED_MAX_S = _cpu_setting_float(
    "cpu_layered_max_s", CPU_LAYERED_MAX_DEFAULT_S, 5.0, CPU_COMPONENT_HARD_CAP_S
)

# Päevane kasutus loetakse püsivalt app_settings-ist. Päeva vahetudes nullime
# ainult KurgiMootori enda uurimisaja arvesti.
_cpu_day_key = TODAY.isoformat()
try:
    _cpu_saved_day = str(db.get_app_setting("cpu_research_usage_day", "") or "")
except Exception:
    _cpu_saved_day = ""

if _cpu_saved_day != _cpu_day_key:
    _cpu_used_today_s = 0.0
    _cpu_light_today_s = 0.0
    _cpu_layered_today_s = 0.0
    try:
        db.set_app_setting("cpu_research_usage_day", _cpu_day_key)
        db.set_app_setting("cpu_research_used_s", "0.0")
        db.set_app_setting("cpu_research_light_s", "0.0")
        db.set_app_setting("cpu_research_layered_s", "0.0")
    except Exception:
        pass
else:
    _cpu_used_today_s = _cpu_setting_float("cpu_research_used_s", 0.0, 0.0, 9999.0)
    _cpu_light_today_s = _cpu_setting_float("cpu_research_light_s", 0.0, 0.0, 9999.0)
    _cpu_layered_today_s = _cpu_setting_float("cpu_research_layered_s", 0.0, 0.0, 9999.0)

def _cpu_remaining_s() -> float:
    return max(0.0, float(CPU_RESEARCH_DAILY_BUDGET_S) - float(_cpu_used_today_s))

def _cpu_record_research(kind: str, seconds: float) -> None:
    global _cpu_used_today_s, _cpu_light_today_s, _cpu_layered_today_s
    elapsed = max(0.0, float(seconds or 0.0))
    if elapsed <= 0:
        return
    _cpu_used_today_s += elapsed
    if kind == "light":
        _cpu_light_today_s += elapsed
    elif kind == "layered":
        _cpu_layered_today_s += elapsed
    try:
        db.set_app_setting("cpu_research_usage_day", _cpu_day_key)
        db.set_app_setting("cpu_research_used_s", f"{_cpu_used_today_s:.1f}")
        db.set_app_setting("cpu_research_light_s", f"{_cpu_light_today_s:.1f}")
        db.set_app_setting("cpu_research_layered_s", f"{_cpu_layered_today_s:.1f}")
        if _cpu_used_today_s > CPU_RESEARCH_DAILY_BUDGET_S:
            db.set_app_setting(
                "cpu_research_last_warning",
                f"{_cpu_day_key}: uurimisaeg {_cpu_used_today_s:.1f}s ületas sihtpiiri {CPU_RESEARCH_DAILY_BUDGET_S:.0f}s; uusi laiu katseid ei käivitata.",
            )
    except Exception:
        pass

# Lai uuring saab igal käivitamisel dünaamilise ülempiiri. 105 s fikseeritud
# vooru enam ei ole. Ühine päevapiir võidab alati individuaalseid seadeid.
LAYERED_DAILY_BUDGET_S = min(CPU_LAYERED_MAX_S, _cpu_remaining_s())
_layered_run_now = False
_layered_auto_due = False
_layered_skip_reason = ""
_layered_elapsed_s_this_cycle = 0.0
_layered_last_at_raw = db.get_app_setting("layered_research_last_at", "")
_layered_last_attempt_raw = db.get_app_setting("layered_research_last_attempt_at", "")
_layered_last_error_raw = db.get_app_setting("layered_research_last_error", "")
_layered_last_day = str(_layered_last_at_raw)[:10] if _layered_last_at_raw else ""
_layered_success_today = _layered_last_day == TODAY.isoformat()

# Automaatne lai otsing võib käivituda ainult uue täieliku korjepäeva tõttu.
# Ilmauuendus üksi EI tohi seda käivitada.
try:
    _last_research_harvest_day = db.get_app_setting("layered_research_last_harvest_day", "")
except Exception:
    _last_research_harvest_day = ""

if page == "Mootori tähelepanekud":
    with st.expander("⚙️ Uurimise CPU piirid"):
        _cpu_c1, _cpu_c2, _cpu_c3 = st.columns(3)
        with _cpu_c1:
            _cpu_daily_input = st.number_input(
                "Päevane uurimispott (s)",
                min_value=10, max_value=int(CPU_RESEARCH_DAILY_HARD_CAP_S), step=5,
                value=int(round(CPU_RESEARCH_DAILY_BUDGET_S)),
                help="Jäljeotsija + lai otsing jagavad seda potti. Ilm ja 9 päeva operatiivne prognoos siia ei kuulu.",
                key="cpu_daily_budget_input",
            )
        with _cpu_c2:
            _cpu_light_input = st.number_input(
                "Jäljeotsija sihtlagi (s)",
                min_value=5, max_value=int(CPU_COMPONENT_HARD_CAP_S), step=5,
                value=int(round(CPU_LIGHT_TARGET_S)),
                help="Pehme siht/reserv. Jäljeotsijat ei katkestata poole cache-kirjutuse pealt; ülejooks vähendab või nullib laia otsingu aega.",
                key="cpu_light_target_input",
            )
        with _cpu_c3:
            _cpu_layered_input = st.number_input(
                "Lai otsing max (s)",
                min_value=5, max_value=int(CPU_COMPONENT_HARD_CAP_S), step=5,
                value=int(round(CPU_LAYERED_MAX_S)),
                help="Kõva käivituspiir: lai otsing saab kõige rohkem selle aja, aga ühine päevapott võib anda vähem.",
                key="cpu_layered_max_input",
            )

        if (_cpu_light_input + _cpu_layered_input) > _cpu_daily_input:
            st.info(
                "Jäljeotsija ja laia otsingu individuaalsed soovid ületavad ühist potti. "
                "See on lubatud: ühine päevapiir võidab alati, seega ühe suurem aeg jätab teisele automaatselt vähem."
            )
        if _cpu_daily_input > 40:
            st.warning(
                "Üle 40 s päevast uurimispotti kasutaksin ainult siis, kui tehniline koormus näitab mitu päeva järjest head varu. "
                "See piir ei ole Streamliti ametlik CPU-kvoot, vaid KurgiMootori enda konservatiivne wall-clock kaitse."
            )

        st.caption(
            f"Täna mõõdetud uurimisaega: {_cpu_used_today_s:.1f}/{CPU_RESEARCH_DAILY_BUDGET_S:.0f} s "
            f"(Jäljeotsija {_cpu_light_today_s:.1f} s · lai {_cpu_layered_today_s:.1f} s) · "
            f"alles {_cpu_remaining_s():.1f} s."
        )
        if st.button("Salvesta CPU piirid", key="save_cpu_limits"):
            try:
                db.set_app_setting("cpu_research_daily_budget_s", str(float(_cpu_daily_input)))
                db.set_app_setting("cpu_light_target_s", str(float(_cpu_light_input)))
                db.set_app_setting("cpu_layered_max_s", str(float(_cpu_layered_input)))
                st.success("CPU piirid salvestatud.")
                st.rerun()
            except Exception as _cpu_save_exc:
                st.error(f"CPU piiride salvestamine ebaõnnestus: {_cpu_save_exc}")

        st.warning(
            "Need sekundid mõõdavad KurgiMootori enda arvutusaega, mitte Streamlit Cloudi tegelikku CPU-arvestust. "
            "Kaitse eesmärk on vältida pikki ja korduvaid uurimisringe; operatiivset ilma- ja saagiprognoosi see ei blokeeri."
        )

    if _layered_last_at_raw:
        try:
            _ldt = datetime.fromisoformat(_layered_last_at_raw).astimezone(ZoneInfo("Europe/Tallinn"))
            _age_h = max(0, int((datetime.now(ZoneInfo("Europe/Tallinn")) - _ldt).total_seconds() // 3600))
            _last_txt = f"{_ldt.strftime('%d.%m %H:%M')} · {_age_h} h tagasi"
        except Exception:
            _last_txt = str(_layered_last_at_raw)
    else:
        _last_txt = "pole veel edukalt lõpetatud"
    st.caption(f"🔬 Viimane edukas lai kihiline uuring: {_last_txt}")

    # Käsitsi lai voor ei ole eraldi "veel üks katse". Seda saab kasutada ainult
    # viimase uue täieliku korjepäeva jaoks juhul, kui Jäljeotsija on sama õppimisnäite
    # juba läbi töötanud, aga lai voor jäi nt CPU-puuduse tõttu tegemata.
    _manual_latest_complete_harvest = None
    try:
        _mh_rows = db.get_harvest_history(limit=1000)
        _mh_plans = db.get_harvest_plans()
        _mh_by_day = {}
        for _r in _mh_rows:
            _dkey = str(_r.get("harvest_date") or "")
            if _dkey:
                _mh_by_day.setdefault(_dkey, []).append(_r)
        _mh_complete = []
        for _dkey, _drows in _mh_by_day.items():
            try:
                _dd = date.fromisoformat(_dkey)
            except ValueError:
                continue
            if _dd > TODAY:
                continue
            _fields = {int(_r.get("field_no")) for _r in _drows if _r.get("field_no") is not None}
            _plan = _mh_plans.get(_dkey)
            if _plan is not None:
                _expected = {int(_f) for _f in _plan}
                _complete = bool(_expected) and _fields == _expected and len(_drows) == len(_expected)
            else:
                _complete = len(_drows) == 3 and len(_fields) == 3
            if _complete:
                _mh_complete.append(_dd)
        if _mh_complete:
            _manual_latest_complete_harvest = max(_mh_complete)
    except Exception:
        _manual_latest_complete_harvest = None

    _manual_latest_key = (
        _manual_latest_complete_harvest.isoformat() if _manual_latest_complete_harvest else ""
    )
    _manual_light_done_key = str(db.get_app_setting("cpu_light_last_complete_harvest_day", "") or "")

    # Automaatne lai ring on endiselt ainult üks kord uue täieliku korjepäeva järel.
    # Käsitsi võib sama andmeseisu uurimist aga jätkata seni, kuni tänases ühises
    # uurimispotis on turvavaruga piisavalt aega. Püsiv research-state tagab, et
    # juba testitud kombinatsioone sama data_hash'i peal uuesti ei arvutata.
    _manual_layered_available_s = min(
        float(CPU_LAYERED_MAX_S),
        max(
            0.0,
            _cpu_remaining_s() - float(CPU_AUTO_SAFETY_RESERVE_S),
        ),
    )
    _manual_pot_remaining_s = _cpu_remaining_s()

    if not _manual_latest_key:
        st.button(
            "🔬 Lai uurimisvoor ootab täielikku korjepäeva",
            disabled=True,
            key="layered_manual_no_harvest_disabled",
        )
    elif _manual_light_done_key != _manual_latest_key:
        st.button(
            "🔬 Lai uurimisvoor ootab Jäljeotsijat",
            disabled=True,
            key="layered_manual_wait_light_disabled",
        )
        st.caption("Lai otsing saab järgneda alles siis, kui sama täieliku korjepäeva Jäljeotsija ring on tehtud.")
    elif _manual_layered_available_s < CPU_LAYERED_MIN_LAUNCH_S:
        st.button(
            "🔬 Tänane uurimispott kasutatud",
            disabled=True,
            key="layered_manual_budget_disabled",
        )
        st.caption(
            f"Potis on {_manual_pot_remaining_s:.1f} s; uue kandidaadi turvaliseks käivituseks "
            f"on vaja vähemalt {CPU_LAYERED_MIN_LAUNCH_S:.0f} s ning {CPU_AUTO_SAFETY_RESERVE_S:.0f} s turvavaru."
        )
    else:
        _layered_run_now = st.button(
            f"🔬 Jätka laia uuringut · {_manual_pot_remaining_s:.0f} s potis alles",
            key="layered_manual_run",
            help=(
                "Jätkab sama andmeseisu uurimist sealt, kuhu eelmine ring jäi. "
                "Juba testitud kombinatsioone ei korrata. Automaatika käivitab endiselt "
                "ainult ühe laia ringi uue täieliku korjepäeva järel."
            ),
        )
        st.caption(
            f"Viimane täielik korjepäev: {_manual_latest_complete_harvest.strftime('%d.%m.%Y')} · "
            f"selle vajutuse ülempiir {_manual_layered_available_s:.1f} s."
        )
        if _layered_run_now:
            LAYERED_DAILY_BUDGET_S = _manual_layered_available_s

    # Kiire uurimisseisu kokkuvõte otse nupu juures.
    _manual_tested_n = str(db.get_app_setting("layered_research_last_candidates", "") or "")
    _manual_best = _load_json_setting("layered_best_challenger_json", {})
    _manual_bits = []
    if _manual_tested_n:
        _manual_bits.append(f"uuritud kombinatsioone sellel andmeseisul: {_manual_tested_n}")
    if isinstance(_manual_best, dict) and _manual_best:
        try:
            _manual_imp = float(_manual_best.get("improvement"))
            _manual_bits.append(f"parim challenger Δ {_manual_imp:+.2f}")
        except Exception:
            pass
    if _manual_bits:
        st.caption(" · ".join(_manual_bits))
    if isinstance(_manual_best, dict) and (_manual_best.get("parts") or []):
        st.caption("Praegune parim: " + " | ".join(map(str, _manual_best.get("parts") or [])))

    # Variprognoos on selle lehe põhisisu, mitte diagnostika: näita see kohe
    # uurimisnupu all, et championit ja parimat alternatiivi saaks kiiresti võrrelda.
    _top_shadow = _load_json_setting("shadow_challenger_forecast_json", {})
    if isinstance(_top_shadow, dict) and _top_shadow.get("target_date") and _top_shadow.get("challenger_total") is not None:
        try:
            _top_shadow_day = date.fromisoformat(str(_top_shadow.get("target_date")))
            _top_official_total = float(_top_shadow.get("official_total"))
            _top_challenger_total = float(_top_shadow.get("challenger_total"))
            _top_shadow_diff = _top_challenger_total - _top_official_total
            st.markdown("### 🌓 Homne variprognoos")
            st.info(
                f"**{_weekday_letter(_top_shadow_day)} {_short_date(_top_shadow_day)}** · "
                f"ametlik {str(_top_shadow.get('official_model') or 'champion')} **{_top_official_total:.1f} kasti** · "
                f"challenger **{_top_challenger_total:.1f} kasti** · vahe **{_top_shadow_diff:+.1f}**."
            )
            _top_shadow_parts = _top_shadow.get("challenger_parts") or []
            if _top_shadow_parts:
                st.caption("Alternatiiv: " + " | ".join(map(str, _top_shadow_parts)))
            st.caption(
                ("Kinnitus: ✅ kinnitatud. " if bool(_top_shadow.get("confirmed")) else "Kinnitus: veel kinnitamata. ")
                + "Variprognoos ei muuda ametlikku prognoosi ega championit."
            )
        except Exception:
            pass

    if _layered_last_error_raw:
        st.caption(
            "Viimane katse ei lõpetanud edukalt: "
            + str(_layered_last_error_raw)
            + ". See ei lukusta uut katset."
        )

_auto_research_dirty = _model_is_dirty()
_research_dirty_reason = db.get_app_setting("model_dirty_reason", "") if _auto_research_dirty else ""
_new_complete_harvest_trigger = bool(
    _auto_research_dirty and str(_research_dirty_reason).startswith("täielik korjepäev")
)
# app-117-st üle tulnud vana ilma-/prognoosi dirty-märge ei tohi pärast
# triggerite lahutamist jääda igaveseks research-dirty olekuks.
if _auto_research_dirty and not _new_complete_harvest_trigger:
    try:
        db.set_app_setting("model_dirty", "0")
    except Exception:
        pass
    _auto_research_dirty = False
    _research_dirty_reason = ""

_forecast_refresh_due = _forecast_is_dirty()
_forecast_dirty_reason = db.get_app_setting("forecast_dirty_reason", "") if _forecast_refresh_due else ""

# Prognoosi LEHE AVAMINE ei käivita enam operatiivset arvutusringi.
# Uus ametlik prognoos arvutatakse ainult siis, kui sisend päriselt muutus
# (ilm/uus päev), tuli uus täielik korjepäev või kasutaja käivitas käsitsi laia uuringu.
# Jäljeotsija saab automaatselt õiguse ainult uue täieliku korjepäeva järel.
_light_skip_day = str(db.get_app_setting("cpu_light_skip_day", "") or "")
_run_light_research_requested = bool(
    _new_complete_harvest_trigger and _light_skip_day != TODAY.isoformat()
)
_run_layered_manual_requested = bool(_layered_run_now)
_run_operational_cycle = bool(
    _forecast_refresh_due or _run_light_research_requested or _run_layered_manual_requested
)

if _run_operational_cycle:
    _light_cycle_t0 = time.perf_counter()
    if _run_light_research_requested:
        _light_cycle_reason = str(_research_dirty_reason or "täielik korjepäev")
    elif _run_layered_manual_requested:
        _light_cycle_reason = "käsitsi lai uuring"
    elif _forecast_refresh_due:
        _light_cycle_reason = "operatiivne prognoos · " + str(_forecast_dirty_reason or "uus ilm")
    else:
        _light_cycle_reason = "operatiivne prognoos"

    _run_light_research = False

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

        # Korjepäevad ja viimane täielik päev. Uutel päevadel tähendab "täielik", et
        # kõik selleks päevaks valitud põllud on sisestatud. Vanadel päevadel, mille
        # tööplaani veel ei salvestatud, kasutame tagasiühilduvuseks senist 3-põllu reeglit.
        harvest_by_day = {}
        for row in harvest_rows:
            day_str = str(row.get("harvest_date") or "")
            if day_str:
                harvest_by_day.setdefault(day_str, []).append(row)
        saved_harvest_plans = db.get_harvest_plans()

        complete_harvest_days = []
        for day_str, day_rows in harvest_by_day.items():
            fields = {int(r.get("field_no")) for r in day_rows if r.get("field_no") is not None}
            expected_plan = saved_harvest_plans.get(day_str)
            if expected_plan is not None:
                expected_fields = set(int(f) for f in expected_plan)
                day_complete = bool(expected_fields) and fields == expected_fields and len(day_rows) == len(expected_fields)
            else:
                day_complete = len(day_rows) == 3 and len(fields) == 3
            if day_complete:
                try:
                    complete_harvest_days.append(date.fromisoformat(day_str))
                except ValueError:
                    pass

        last_complete_harvest = max(complete_harvest_days) if complete_harvest_days else None

        # Uue täieliku korjepäeva Jäljeotsija käib maksimaalselt üks kord selle
        # õppimisnäite kohta ja ainult siis, kui ühises uurimispotis on enne starti
        # mõistlik varu. Ilm, lehe avamine ja Streamliti rerun ei anna uurimisõigust.
        _light_last_harvest_day = str(db.get_app_setting("cpu_light_last_complete_harvest_day", "") or "")
        _light_min_launch_s = min(float(CPU_LIGHT_TARGET_S), 10.0)
        if _run_light_research_requested and last_complete_harvest is not None:
            _complete_key = last_complete_harvest.isoformat()
            if _complete_key == _light_last_harvest_day:
                _run_light_research = False
                db.set_app_setting("model_dirty", "0")
            elif _cpu_remaining_s() < _light_min_launch_s:
                _run_light_research = False
                db.set_app_setting("cpu_light_skip_day", TODAY.isoformat())
                db.set_app_setting(
                    "cpu_light_last_warning",
                    f"{TODAY.isoformat()}: Jäljeotsija jäi käivitamata; ühises potis oli "
                    f"{_cpu_remaining_s():.1f}s, stardiks on vaja vähemalt {_light_min_launch_s:.0f}s. "
                    "Täna uuesti ei proovita; järgmise päeva esimesel käivitusel proovitakse uuesti.",
                )
            else:
                _run_light_research = True

        # Lai kihiline otsing võib automaatselt järgneda ainult samale uuele
        # täielikule korjepäevale. Tema päris ajabubudžett arvutatakse alles vahetult
        # enne laia ringi, kui Jäljeotsija tegelik kestus on juba teada.
        _layered_auto_due = bool(
            _run_light_research
            and last_complete_harvest is not None
            and last_complete_harvest.isoformat() != str(_last_research_harvest_day or "")
        )

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
        # Tänast päeva ei märgita enam puuduvaks isegi siis, kui tänaseks valitud korjed on juba sisestatud.
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
                    "previous_order": int(previous_row.get("harvest_order") or 1),
                    "current_order": int(row.get("harvest_order") or 1),
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

            # V6.5 BIO kandidaatkiht.
            # Ainult bioloogiliselt interpreteeritavad tunnused; champion'i loogikat ei muudeta.
            # GDD base 10 C on siin kandidaat, mitte etteantud "õige" füsioloogiline konstant.
            bio_gdd10 = sum(max(0.0, tm - 10.0) for tm in daily_mean_t)
            bio_rad_sum = sum(rad)
            # Soojusnõudluse ja assimilaatide pakkumise lihtne tasakaaluproxy.
            bio_rad_per_gdd10 = bio_rad_sum / bio_gdd10 if bio_gdd10 > 0 else 0.0

            # Hilisematele kasvupäevadele suurem kaal. See ei eelda, et just selline
            # kaal on õige; Jäljeotsija peab selle väärtuse walk-forward testis tõestama.
            bio_weights = np.arange(1.0, len(window_weather) + 1.0, dtype=float)
            bio_weight_sum = float(np.sum(bio_weights))
            bio_weighted_temp = (
                float(np.dot(np.asarray(daily_mean_t, dtype=float), bio_weights) / bio_weight_sum)
                if bio_weight_sum > 0 else 0.0
            )
            bio_weighted_rad = (
                float(np.dot(np.asarray(rad, dtype=float), bio_weights) / bio_weight_sum)
                if bio_weight_sum > 0 else 0.0
            )

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
                nights = [_n(w.get("temp_night_avg_c")) for w in tail]
                days = [_n(w.get("temp_day_avg_c")) for w in tail]
                mt = [(lo + hi) / 2 for lo, hi in zip(tmins, tmaxs)]
                rad_sum = sum(_n(w.get("radiation_mj_m2")) for w in tail)
                rain_sum = sum(_n(w.get("precipitation_mm")) for w in tail)
                et0_sum = sum(_n(w.get("et0_mm")) for w in tail)
                rh_mean = sum(_n(w.get("humidity_avg_pct")) for w in tail) / len(tail)
                wind_mean = sum(_n(w.get("wind_avg_ms")) for w in tail) / len(tail)
                tmax_mean = sum(tmaxs) / len(tmaxs)

                return {
                    f"T viim{n}": sum(mt) / len(mt),
                    f"ÖöT viim{n}": sum(nights) / len(nights),
                    f"PäevT viim{n}": sum(days) / len(days),
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

            def _lookback_weather(n):
                # Pikem ilmamälu võib ulatuda üle eelmise korje piiri. See on ainult
                # Jäljeotsija kandidaat, mitte baasmudeli kohustuslik sisend.
                lookback = []
                d0 = sample["current_day"] - timedelta(days=n - 1)
                d = d0
                while d <= sample["current_day"]:
                    wr = weather_by_day.get(d.isoformat())
                    if not wr or not _weather_day_ok(d):
                        return {
                            f"ÖöT {n}p": None, f"PäevT {n}p": None, f"Rad {n}p": None,
                            f"Sade {n}p": None, f"ET0 {n}p": None, f"Niiskus {n}p": None,
                        }
                    lookback.append(wr)
                    d += timedelta(days=1)
                return {
                    f"ÖöT {n}p": float(np.mean([_n(w.get("temp_night_avg_c")) for w in lookback])),
                    f"PäevT {n}p": float(np.mean([_n(w.get("temp_day_avg_c")) for w in lookback])),
                    f"Rad {n}p": float(np.sum([_n(w.get("radiation_mj_m2")) for w in lookback])),
                    f"Sade {n}p": float(np.sum([_n(w.get("precipitation_mm")) for w in lookback])),
                    f"ET0 {n}p": float(np.sum([_n(w.get("et0_mm")) for w in lookback])),
                    f"Niiskus {n}p": float(np.mean([_n(w.get("humidity_avg_pct")) for w in lookback])),
                }

            tail1 = _tail_weather(1)
            tail2 = _tail_weather(2)
            tail3 = _tail_weather(3)
            lookback7 = _lookback_weather(7)
            lookback10 = _lookback_weather(10)
            lookback14 = _lookback_weather(14)

            temp_curve = _temperature_curve_features(night_t, day_t)

            training_rows.append({
                "BIO GDD10": bio_gdd10,
                "BIO radiatsioonisumma": bio_rad_sum,
                "BIO rad/GDD10": bio_rad_per_gdd10,
                "BIO hiline T": bio_weighted_temp,
                "BIO hiline radiatsioon": bio_weighted_rad,

                "Kuupäev": sample["current_day"],
                "Põld": sample["field_no"],
                "Intervall p": sample["interval_days"],
                # Ligikaudne tegelik kasvukestus: korjejärjekorra üks samm ≈ 3 h.
                # Ei asenda veel baasmudeli intervalli; Jäljeotsija peab kasu tõestama.
                "Kasvuaeg p": float(sample["interval_days"])
                    + (float(sample.get("current_order", 1)) - float(sample.get("previous_order", 1))) * (3.0 / 24.0),
                **_season_curve_features(sample["current_day"], SEASON_START),
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
                **tail1, **tail2, **tail3, **lookback7, **lookback10, **lookback14,
                "Andmekvaliteet": current_row.get("data_quality") or "",
            })

        if training_rows:
            training_df = pd.DataFrame(training_rows).sort_values(["Kuupäev", "Põld"], ascending=[False, True])
            t1, t2, t3 = st.columns(3)
            t1.metric("Valmis õppimisridu", len(training_df))
            t2.metric("Keskmine intervall", f"{training_df['Intervall p'].mean():.1f} p")
            t3.metric("Keskmine A+B+C", f"{training_df['ABC saak'].mean():.1f} kasti")

            visible_training_cols = [
                "Kuupäev", "Põld", "Intervall p", "Kasvuaeg p", "ABC saak", "C/B siht", "XL", "Saak", "Eelmine ABC",
                "T kesk", "ÖöT kesk", "PäevT kesk", "Tmin kesk", "Tmax kesk", "Tmin min", "Tmax max", "Radiatsioon Σ", "Radiatsioon/p", "Sademed Σ",
                "Niiskus kesk", "ET0 Σ", "Tuul kesk", "A", "B", "C", "Andmekvaliteet",
            ]
            display_df = training_df[visible_training_cols].copy()
            display_df["Kuupäev"] = display_df["Kuupäev"].map(lambda d: d.strftime("%d.%m"))
            st.dataframe(
                display_df.style.format({
                    "Kasvuaeg p": "{:.1f}",
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

            # -------------------------------------------------------------
            # KERGE JÄLJEOTSIJA PÜSIV CACHE
            #
            # Sama ajaloolise õppimisandmestiku korral (nt ainult tuleviku ilma
            # uuendus) EI arvutata 36+ kandidaadi walk-forward'i uuesti.
            # Uue täieliku korjepäeva lisandumisel säilivad vanad ausad
            # walk-forward ennustused ja arvutatakse juurde ainult uue päeva samm.
            # -------------------------------------------------------------
            _light_cache_version = "v16-light-wf-cache-1"

            _light_hash_frame = model_df.copy()
            for _c in _light_hash_frame.columns:
                if _c == "Kuupäev":
                    _light_hash_frame[_c] = _light_hash_frame[_c].map(
                        lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v)
                    )
            _light_row_hash_values = pd.util.hash_pandas_object(
                _light_hash_frame.astype(str), index=False
            ).to_numpy(dtype=np.uint64)
            _light_row_hashes = [f"{int(v):016x}" for v in _light_row_hash_values]
            _light_data_hash = hashlib.sha256(
                "|".join(_light_row_hashes).encode("utf-8")
            ).hexdigest()

            _light_cache_raw = db.get_app_setting("light_walk_cache_json", "")
            try:
                _light_cache = json.loads(_light_cache_raw) if _light_cache_raw else {}
                if not isinstance(_light_cache, dict):
                    _light_cache = {}
            except Exception:
                _light_cache = {}

            if _light_cache.get("version") != _light_cache_version:
                _light_cache = {}

            _cached_row_hashes = list(_light_cache.get("row_hashes") or [])
            _light_exact_cache = (
                _cached_row_hashes == _light_row_hashes
                and _light_cache.get("data_hash") == _light_data_hash
            )
            _light_incremental_cache = (
                bool(_cached_row_hashes)
                and len(_cached_row_hashes) < len(_light_row_hashes)
                and _light_row_hashes[:len(_cached_row_hashes)] == _cached_row_hashes
            )
            _light_cached_n = len(_cached_row_hashes) if (_light_exact_cache or _light_incremental_cache) else 0

            _light_cache.setdefault("groups", {})
            _light_cache_hits = 0
            _light_cache_incremental = 0
            _light_cache_misses = 0

            def _light_decode_array(values, expected_len):
                if not isinstance(values, list):
                    return None
                if len(values) > expected_len:
                    return None
                arr = np.full(expected_len, np.nan, dtype=float)
                for i, value in enumerate(values):
                    if value is None:
                        continue
                    try:
                        arr[i] = float(value)
                    except (TypeError, ValueError):
                        pass
                return arr

            def _light_encode_array(arr):
                return [
                    None if not np.isfinite(v) else float(v)
                    for v in np.asarray(arr, dtype=float)
                ]

            def _light_walk_cached(group_name, item_name, day_predictor):
                """Taasta vana WF tulemus ja arvuta ainult puuduva lõpu testipäevad."""
                nonlocal_holder = None  # ainult selguse jaoks; Python scope'i pole vaja muuta
                group = _light_cache["groups"].setdefault(group_name, {})
                stored = group.get(item_name)
                n = len(model_df)

                use_prefix = False
                start_idx = 0
                preds = None

                if _light_exact_cache and stored is not None:
                    restored = _light_decode_array(stored, n)
                    if restored is not None and len(stored) == n:
                        preds = restored
                        use_prefix = True
                        start_idx = n
                elif _light_incremental_cache and stored is not None:
                    restored = _light_decode_array(stored, n)
                    if restored is not None and len(stored) == _light_cached_n:
                        preds = restored
                        use_prefix = True
                        start_idx = _light_cached_n

                if preds is None:
                    preds = np.full(n, np.nan, dtype=float)
                    start_idx = 0

                if start_idx >= n:
                    mode = "hit"
                    test_days = []
                elif use_prefix and start_idx > 0:
                    mode = "incremental"
                    test_days = sorted(set(dates[start_idx:]))
                else:
                    mode = "miss"
                    test_days = sorted(set(dates))

                for _test_day in test_days:
                    day_predictor(_test_day, preds)

                group[item_name] = _light_encode_array(preds)
                return preds, mode

            def _save_light_walk_cache():
                try:
                    _light_cache["version"] = _light_cache_version
                    _light_cache["data_hash"] = _light_data_hash
                    _light_cache["row_hashes"] = list(_light_row_hashes)
                    _light_cache["saved_at"] = datetime.now(ZoneInfo("Europe/Tallinn")).isoformat()
                    db.set_app_setting(
                        "light_walk_cache_json",
                        json.dumps(_light_cache, ensure_ascii=False),
                    )
                except Exception:
                    pass

            # Mudeli lineaaralgebra elab model_engine.py-s; siin seome selle tänase andmestikuga.
            def _build_ridge_design(train_idx, test_idx, extra_arrays):
                return _engine_build_ridge_design(X_base, fields, train_idx, test_idx, extra_arrays)

            def _ridge_walk_predict(target, extra_arrays, train_idx, test_idx, alpha=10.0, floor_zero=True, field_alpha=80.0):
                return _engine_ridge_walk_predict(
                    X_base, fields, target, extra_arrays, train_idx, test_idx,
                    alpha=alpha, floor_zero=floor_zero, field_alpha=field_alpha,
                )

            def _abc_growth_walk_predict(extra_arrays, train_idx, test_idx, alpha=10.0, field_alpha=80.0, z_clip=2.5):
                return _engine_abc_growth_walk_predict(
                    X_base, fields, log_y_abc, extra_arrays, train_idx, test_idx,
                    min_train_rows=min_train_rows, alpha=alpha, field_alpha=field_alpha,
                    z_clip=z_clip, log_eps=ABC_LOG_EPS,
                )

            min_train_rows = 10

            def _base_abc_day(test_day, preds):
                test_idx = np.where(dates == test_day)[0]
                train_idx = np.where(dates < test_day)[0]
                if len(train_idx) < min_train_rows:
                    return
                preds[test_idx] = _abc_growth_walk_predict([], train_idx, test_idx)

            def _base_xl_day(test_day, preds):
                test_idx = np.where(dates == test_day)[0]
                train_idx = np.where(dates < test_day)[0]
                if len(train_idx) < min_train_rows:
                    return
                preds[test_idx] = _ridge_walk_predict(y_xl, [], train_idx, test_idx)

            def _base_cb_day(test_day, preds):
                test_idx = np.where(dates == test_day)[0]
                train_idx = np.where((dates < test_day) & np.isfinite(log_y_cb))[0]
                if len(train_idx) < min_train_rows:
                    return
                log_pred = _ridge_walk_predict(log_y_cb, [], train_idx, test_idx, floor_zero=False)
                preds[test_idx] = np.exp(np.clip(log_pred, np.log(0.10), np.log(10.0)))

            abc_predictions, _base_abc_mode = _light_walk_cached(
                "base", "abc", _base_abc_day
            )
            xl_predictions, _base_xl_mode = _light_walk_cached(
                "base", "xl", _base_xl_day
            )
            cb_predictions, _base_cb_mode = _light_walk_cached(
                "base", "cb", _base_cb_day
            )

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
                # Pikem ilmamälu ulatub teadlikult üle eelmise korje piiri.
                # Need on ainult kandidaadid; baasmudel jääb korjest-korjeni weather-first.
                "7 päeva ilmamälu": ["ÖöT 7p", "PäevT 7p", "Rad 7p", "Sade 7p", "ET0 7p", "Niiskus 7p"],
                "10 päeva ilmamälu": ["ÖöT 10p", "PäevT 10p", "Rad 10p", "Sade 10p", "ET0 10p", "Niiskus 10p"],
                "14 päeva ilmamälu": ["ÖöT 14p", "PäevT 14p", "Rad 14p", "Sade 14p", "ET0 14p", "Niiskus 14p"],

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

                # Hooaja / taime kulumise mittelineaarne kuju.
                # Languse märki ei kirjutata ette; avastus+kinnitus peab kasu tõestama.
                "Hooaja kaar": ["Hooajapäev²"],
                "Hooaja faasivahetused": ["Hooaeg 35+", "Hooaeg 50+", "Hooaeg 65+"],
                "Hilishooaja kulumiskõver": ["Hooaeg 50+", "Hooaeg 50+²", "Hooaeg 65+"],
                "Paindlik hooajakõver": ["Hooajapäev²", "Hooaeg 35+", "Hooaeg 50+", "Hooaeg 65+"],

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

            # V6.5 LAYERED:
            # Üksiktunnused on ainult sõelumiseks. Nad EI saa enam A+B+C championiks.
            # Championivõistlusse pääsevad alles vähemalt kahe eri bioloogilise lisakihi
            # kombinatsioonid weather-first baasi peal.
            screening_candidate_groups = {**weather_candidate_groups, **biological_load_candidate_groups}

            # Kihid on bioloogilised rollid, mitte üksikud valemid. Iga kihi parim
            # esindaja valitakse ainult AVASTUSperioodil; kinnitusaega selleks ei vaadata.
            layered_screen_layers = {
                "Termiline areng": {
                    "BIO GDD10": ["BIO GDD10"],
                    "BIO hiline T": ["BIO hiline T"],
                    "7p temperatuur": ["ÖöT 7p", "PäevT 7p"],
                    "10p temperatuur": ["ÖöT 10p", "PäevT 10p"],
                    "14p temperatuur": ["ÖöT 14p", "PäevT 14p"],
                    "Öötemperatuuri kuju": ["ÖöT kesk", "Soojad ööd 16+ %", "Jahedad ööd 12- %"],
                },
                "Assimilaatide pakkumine": {
                    "BIO radiatsioonisumma": ["BIO radiatsioonisumma"],
                    "BIO hiline radiatsioon": ["BIO hiline radiatsioon"],
                    "BIO rad/GDD10": ["BIO rad/GDD10"],
                    "7p radiatsioon": ["Rad 7p"],
                    "10p radiatsioon": ["Rad 10p"],
                    "14p radiatsioon": ["Rad 14p"],
                },
                "Bioloogiline koormus": {
                    "Ebatavaline koormus -1": ["Koormusindeks -1", "Ülekoormus -1"],
                    "Kahe korje koormus": ["2 korje koormus"],
                    "Tipukorje järelmõju": ["Tipukorje -1", "Tipukorje -2"],
                },
                "Mikrokliima": {
                    "Niiskus": ["Niiskus kesk"],
                    "Viimase 3p niiskus": ["Niiskus viim3"],
                    "7p niiskus": ["Niiskus 7p"],
                    "10p niiskus": ["Niiskus 10p"],
                    "14p niiskus": ["Niiskus 14p"],
                    # ET0 on KurgiMootori arvutatud koondnäitaja. Seda võib eraldi
                    # testida, kuid see ei ole puhas kohalik mõõtmine.
                    "ET0 koond": ["ET0 Σ"],
                    # Sade tuleb Pärnu jaamast (~40 km). Käsitle seda piirkondliku
                    # ilmamustri proxy-na, mitte väitena kohaliku põllu sademest.
                    "Sade + niiskus (Pärnu proxy)": ["Sademed Σ", "Niiskus kesk"],
                    "Sade + ET0 (Pärnu proxy)": ["Sademed Σ", "ET0 Σ"],
                },
                "Hooaja kulumine": {
                    "Hooaja kaar": ["Hooajapäev²"],
                    "Hooaja faasivahetused": ["Hooaeg 35+", "Hooaeg 50+", "Hooaeg 65+"],
                    "Hilishooaja kulumiskõver": ["Hooaeg 50+", "Hooaeg 50+²", "Hooaeg 65+"],
                    "Päevapikkuse trend": ["Päevapikkus", "Päevapikkus Δ7p"],
                },
                "Kasvuaeg": {
                    "Täpne kasvuaeg ~3h/põld": ["Kasvuaeg p"],
                    "Kasvuaeg + hiline T": ["Kasvuaeg p", "BIO hiline T"],
                    "Kasvuaeg + hiline radiatsioon": ["Kasvuaeg p", "BIO hiline radiatsioon"],
                },
                "Tuul ja kuivatusstress": {
                    "Tuul kesk": ["Tuul kesk"],
                    "Tuul × Tmax": ["Tuul×Tmax"],
                    "Tuul × radiatsioon": ["Tuul×Rad/p"],
                    "Tuul × ET0": ["Tuul×ET0/p"],
                    "Tuul × kuivus": ["Tuul×Kuivus"],
                    "Tuulestress kombineeritud": ["Tuul×Tmax", "Tuul×Rad/p", "Tuul×ET0/p", "Tuul×Kuivus"],
                    "Tuulestress viim1": ["Tuul×Tmax viim1", "Tuul×Rad/p viim1", "Tuul×ET0/p viim1", "Tuul×Kuivus viim1"],
                    "Tuulestress viim2": ["Tuul×Tmax viim2", "Tuul×Rad/p viim2", "Tuul×ET0/p viim2", "Tuul×Kuivus viim2"],
                    "Tuulestress viim3": ["Tuul×Tmax viim3", "Tuul×Rad/p viim3", "Tuul×ET0/p viim3", "Tuul×Kuivus viim3"],
                },
            }

            # Üksiktunnused jäävad päeviku ja "Hoian silma peal" RADARIKS.
            # See EI anna neile champion-õigust: A+B+C championiks saab endiselt
            # ainult vähemalt kahe kihi kombinatsioon pärast eraldi kinnitust.
            operational_candidate_groups = dict(screening_candidate_groups)

            memory_diagnostic_groups = {
                "Eelmine A+B+C": ["Eelmine ABC"],
                "A+B+C trend 2 korjet": ["Eelmine ABC", "Eelmine2 ABC", "ABC trend"],
                "Korjejääk / kõrge eelkorje": ["Eelmine saak", "XL -1", "XL -2", "XL osakaal -1", "XL osakaal -2"],
                "XL osakaal 2 korjet": ["XL osakaal -1", "XL osakaal -2"],
            }
            diagnostic_only_groups = {
                **memory_diagnostic_groups,
                "Täpne kasvuaeg ~3h/põld (diagnostika)": ["Kasvuaeg p"],
                "C/B mälu 2 korjet (diagnostika)": ["C/B -1", "C/B -2"],
            }

            def _walk_forward_with_extra(extra_cols, alpha=10.0, cache_key=None):
                raw_extra = model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]

                def _predict_day(test_day, preds):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where(dates < test_day)[0]
                    if len(train_idx) < min_train_rows:
                        return
                    preds[test_idx] = _abc_growth_walk_predict(extra_arrays, train_idx, test_idx)

                if cache_key is not None:
                    preds, mode = _light_walk_cached("abc", str(cache_key), _predict_day)
                    return preds

                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    _predict_day(test_day, preds)
                return preds

            def _chronological_discovery_confirmation_days(base_pred, target):
                # Ühine statistiline valikupoliitika asub model_engine.py-s.
                return _engine_discovery_confirmation_days(dates, base_pred, target)

            def _period_stability_stats(
                base_pred, candidate_pred, target, day_set,
                *, min_rows, improvement_threshold, min_half_threshold,
            ):
                if not day_set:
                    return None
                mask = (
                    np.isfinite(base_pred)
                    & np.isfinite(candidate_pred)
                    & np.isfinite(target)
                    & np.array([d in day_set for d in dates], dtype=bool)
                )
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return None
                base_abs = np.abs(base_pred[idx] - target[idx])
                cand_abs = np.abs(candidate_pred[idx] - target[idx])
                base_mae = float(base_abs.mean())
                cand_mae = float(cand_abs.mean())
                improvement = base_mae - cand_mae
                win_share = float(np.mean(cand_abs < base_abs))
                unique_test_days = sorted(set(dates[idx]))
                day_halves = np.array_split(np.array(unique_test_days, dtype=object), 2)
                half_improvements = []
                for day_half in day_halves:
                    day_subset = set(day_half.tolist())
                    h = np.array([i for i in idx if dates[i] in day_subset], dtype=int)
                    if len(h) == 0:
                        continue
                    half_improvements.append(float(
                        np.mean(np.abs(base_pred[h] - target[h]))
                        - np.mean(np.abs(candidate_pred[h] - target[h]))
                    ))
                min_half = min(half_improvements) if half_improvements else -999.0
                stable = bool(
                    len(idx) >= min_rows
                    and improvement >= improvement_threshold
                    and win_share >= 0.50
                    and min_half >= min_half_threshold
                )
                return {
                    "Baas MAE": base_mae, "Katse MAE": cand_mae, "Paranemine": improvement,
                    "Võidab ridu %": win_share * 100.0, "Halvim pool": min_half,
                    "Testiridu": int(len(idx)), "Testipäevi": int(len(unique_test_days)), "Stabiilne": stable,
                }

            _abc_discovery_days, _abc_confirmation_days = _chronological_discovery_confirmation_days(predictions, y)

            def _stability_stats(candidate_pred):
                # Üldine diagnostika kogu walk-forward ajalool; championit selle järgi enam ei valita.
                all_days = set(sorted(set(dates[np.where(np.isfinite(predictions) & np.isfinite(y))[0]])))
                return _period_stability_stats(
                    predictions, candidate_pred, y, all_days,
                    min_rows=12, improvement_threshold=0.10, min_half_threshold=-0.05,
                )

            trace_results = []
            candidate_predictions = {}
            _all_candidate_groups = {**screening_candidate_groups, **diagnostic_only_groups}
            _missing_candidate_columns = sorted({
                col
                for cols in _all_candidate_groups.values()
                for col in cols
                if col not in model_df.columns
            })
            if _missing_candidate_columns:
                raise RuntimeError(
                    "Mudeli kandidaattunnused puuduvad õppimisandmestikust: "
                    + ", ".join(_missing_candidate_columns)
                )

            if _run_light_research:
                for name, cols in _all_candidate_groups.items():
                    cp = _walk_forward_with_extra(cols, cache_key=name)
                    candidate_predictions[name] = cp
                    stats = _stability_stats(cp)
                    if stats:
                        trace_results.append({"Jälg": name, **stats})

                trace_df = pd.DataFrame(trace_results)
                if not trace_df.empty:
                    trace_df = trace_df.sort_values(["Stabiilne", "Paranemine"], ascending=[False, False])

                # Põllu seisundi valve arvutatakse ainult päris Jäljeotsija ringis.
                field_state_watch = []
                try:
                    for _f in sorted(set(int(v) for v in fields)):
                        _idx = np.where(
                            (fields == _f) & np.isfinite(predictions) & np.isfinite(y)
                        )[0]
                        if len(_idx) < 2:
                            continue
                        _idx = _idx[np.argsort(dates[_idx])]
                        _last2 = _idx[-2:]
                        _res = y[_last2] - predictions[_last2]
                        if np.all(_res <= -0.75):
                            field_state_watch.append({
                                "Põld": _f,
                                "Viimane kõrvalekalle": float(_res[-1]),
                                "Eelmine kõrvalekalle": float(_res[-2]),
                                "Märge": "võimalik põllu potentsiaali muutus",
                            })
                except Exception:
                    field_state_watch = []
            else:
                # Operatiivne prognoosiring ei käivita kandidaatide walk-forward'i.
                # Diagnostika jaoks kasutame viimast päris uurimisringi snapshot'i.
                _saved_obs_for_forecast = _load_json_setting("motor_observation_snapshot_json", {})
                trace_df = pd.DataFrame((_saved_obs_for_forecast or {}).get("trace") or [])
                field_state_watch = list((_saved_obs_for_forecast or {}).get("field_state_watch") or [])

            # Kerge Jäljeotsija teavitus Avalehele tekib ainult päris uurimisringis.
            # Operatiivne ilma/prognoosi refresh ei lisa discovery ajalukku uusi kirjeid.
            if _run_light_research:
                try:
                    _alert_candidates = trace_df[
                        (trace_df["Stabiilne"] == True)
                        & (pd.to_numeric(trace_df["Paranemine"], errors="coerce") >= 0.10)
                    ].copy() if not trace_df.empty else pd.DataFrame()

                    if not _alert_candidates.empty:
                        _alert_candidates = _alert_candidates.sort_values(
                            ["Paranemine", "Võidab ridu %"], ascending=[False, False]
                        )
                        _ar = _alert_candidates.iloc[0]
                        _alert_payload = {
                            "name": str(_ar["Jälg"]),
                            "improvement": float(_ar["Paranemine"]),
                            "win_share": float(_ar["Võidab ridu %"]),
                            "detected_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                            "dirty_reason": db.get_app_setting("model_dirty_reason", ""),
                        }
                        # Säilita tähelepanekute ajalugu. Sama jälge ei dubleerita iga
                        # kontrollringiga; uus kirje tekib siis, kui sama jälje tugevus on
                        # muutunud vähemalt 0.05 MAE või viimasest kirjest on möödas 7 päeva.
                        _history_raw = db.get_app_setting("discovery_history_json", "")
                        try:
                            _history = json.loads(_history_raw) if _history_raw else []
                            if not isinstance(_history, list):
                                _history = []
                        except Exception:
                            _history = []
                        _append = True
                        _same = [x for x in _history if str(x.get("name")) == _alert_payload["name"]]
                        if _same:
                            _prev = _same[-1]
                            try:
                                _prev_dt = datetime.fromisoformat(str(_prev.get("detected_at")))
                                _now_dt = datetime.fromisoformat(_alert_payload["detected_at"])
                                _age_days = (_now_dt - _prev_dt).total_seconds() / 86400.0
                                _delta_imp = abs(float(_prev.get("improvement", 0)) - _alert_payload["improvement"])
                                if _age_days < 7.0 and _delta_imp < 0.05:
                                    _append = False
                            except Exception:
                                pass
                        if _append:
                            _history.append(_alert_payload)
                            _history = _history[-30:]
                            db.set_app_setting(
                                "discovery_history_json",
                                json.dumps(_history, ensure_ascii=False),
                            )
                        # Vana ühe-jälje Avalehe märguanne pole enam kasutusel.
                        db.set_app_setting("light_discovery_alert_json", "")
                    else:
                        db.set_app_setting("light_discovery_alert_json", "")
                except Exception:
                    pass

            # Operatiivne prognoos kasutab viimast kinnitatud champion-olekut.
            # Ilma/prognoosi refresh ei tee uut champion-valikut.
            _saved_abc_champion = _load_json_setting("abc_champion_state_json", {})
            champion_name = str((_saved_abc_champion or {}).get("name") or "Baasmudel")
            champion_cols = [
                str(c) for c in ((_saved_abc_champion or {}).get("cols") or [])
                if str(c) in model_df.columns
            ]
            champion_pred = predictions.copy()
            champion_mae = (_saved_abc_champion or {}).get("mae")
            if champion_mae is None:
                champion_mae = current_test_mae
            champion_stats = (_saved_abc_champion or {}).get("stats")
            champion_discovery_stats = None
            layered_trace_rows = []

            layered_trace_df = pd.DataFrame()
            _saved_layered_json = db.get_app_setting("layered_research_trace_json", "")
            if _saved_layered_json:
                try:
                    _rows = json.loads(_saved_layered_json)
                    if isinstance(_rows, list):
                        layered_trace_df = pd.DataFrame(_rows)
                except Exception:
                    layered_trace_df = pd.DataFrame()

            # Automaatse laia ringi päris eelarve arvutatakse alles nüüd, pärast
            # Jäljeotsija tööd. Nii jagavad nad päriselt sama päevast potti.
            if _layered_auto_due and not _run_layered_manual_requested:
                _light_elapsed_before_wide_s = max(0.0, time.perf_counter() - _light_cycle_t0)
                # XL/C-B Jäljeotsija töö tuleb koodis pärast laia ABC-plokki, seega
                # reserveerime automaatselt vähemalt kogu Jäljeotsija sihtaja, mitte
                # ainult seni kulunud ABC-osa. Ühine 40 s pott jääb nii päriselt kaitstuks.
                _light_reserve_for_shared_pot_s = max(
                    float(CPU_LIGHT_TARGET_S), _light_elapsed_before_wide_s
                )
                _auto_layered_available_s = min(
                    float(CPU_LAYERED_MAX_S),
                    max(
                        0.0,
                        float(CPU_RESEARCH_DAILY_BUDGET_S)
                        - float(_cpu_used_today_s)
                        - _light_reserve_for_shared_pot_s
                        - float(CPU_AUTO_SAFETY_RESERVE_S),
                    ),
                )
                if _auto_layered_available_s >= CPU_LAYERED_MIN_LAUNCH_S:
                    LAYERED_DAILY_BUDGET_S = _auto_layered_available_s
                    _layered_run_now = True
                else:
                    _layered_run_now = False
                    _layered_skip_reason = (
                        f"Lai uuring jäi vahele: Jäljeotsija järel jäi ühises potis "
                        f"turvavaruga {_auto_layered_available_s:.1f}s; käivituseks on vaja vähemalt "
                        f"{CPU_LAYERED_MIN_LAUNCH_S:.0f}s."
                    )
                    try:
                        db.set_app_setting("layered_research_last_skip_reason", _layered_skip_reason)
                        db.set_app_setting("layered_research_last_skip_at", datetime.now(ZoneInfo("Europe/Tallinn")).isoformat())
                    except Exception:
                        pass

            if _layered_run_now:
                _layered_t0 = time.perf_counter()
                _layered_started_at = datetime.now(ZoneInfo("Europe/Tallinn"))
                _layered_deadline = _layered_t0 + float(LAYERED_DAILY_BUDGET_S)
                _layered_stop_margin_s = 6.0
                _layered_saved_candidate_avg_s = _cpu_setting_float(
                    "layered_research_last_candidate_avg_s", 8.0, 1.0, 30.0
                )

                try:
                    db.set_app_setting("layered_research_last_attempt_at", _layered_started_at.isoformat())
                    db.set_app_setting("layered_research_last_error", "")
                except Exception:
                    pass

                # -----------------------------------------------------------------
                # V6.5+ PÜSIV SUUNATUD AVATUD OTSING
                #
                # - uurimisruumi EI kärbita üksiktunnuse tulemuse järgi;
                # - kõik kihid/variandid jäävad lubatuks;
                # - ~75% katsetest jätkab seni lootustandvate kombinatsioonide ümbruses;
                # - ~25% katsetest uurib uusi kombinatsioone üle kogu ruumi;
                # - sama kombinatsiooni sama andmeseisu juures ei korrata;
                # - iga lõpetatud katse salvestatakse kohe;
                # - järgmine päev jätkab kogutud uurimisajaloo pealt.
                # -----------------------------------------------------------------

                def _variant_specs(layer_name):
                    specs = []
                    for variant_name, cols in layered_screen_layers.get(layer_name, {}).items():
                        if any(c not in model_df.columns for c in cols):
                            continue
                        specs.append({
                            "layer": layer_name,
                            "variant": variant_name,
                            "cols": list(cols),
                        })
                    return specs

                def _merge_cols_many(specs):
                    merged = []
                    for spec in specs:
                        for c in spec["cols"]:
                            if c not in merged:
                                merged.append(c)
                    return merged

                def _combo_key(specs):
                    # Kihi- ja variandinimed teevad võtme loetavaks ja deterministlikuks.
                    parts = sorted(f"{s['layer']}::{s['variant']}" for s in specs)
                    return " || ".join(parts)

                def _feature_signature(specs):
                    # Eri nimega kombinatsioonid võivad anda täpselt sama lõpliku
                    # tunnusekomplekti. Sellist mudelit ei treenita teist korda.
                    cols = sorted(_merge_cols_many(specs))
                    return hashlib.sha256(
                        "||".join(cols).encode("utf-8")
                    ).hexdigest()[:24]

                def _combo_from_key(key):
                    wanted = []
                    lookup = {}
                    for layer_name in layered_screen_layers:
                        for spec in _variant_specs(layer_name):
                            lookup[f"{spec['layer']}::{spec['variant']}"] = spec
                    for token in str(key or "").split(" || "):
                        token = token.strip()
                        if token in lookup:
                            wanted.append(lookup[token])
                    return wanted

                def _serialize_research_state(state):
                    # Detailtulemused jäävad kompaktseks, kuid testitud kombinatsiooni-
                    # võtmeid/signatuure ühe andmeseisu sees ei kärbita, et vana CPU-töö
                    # hiljem uuesti järjekorda ei satuks.
                    state = dict(state)
                    state["results"] = list(state.get("results", []))[-1000:]
                    state["tested_by_hash"] = {
                        str(k): list(v)
                        for k, v in dict(state.get("tested_by_hash", {})).items()
                    }
                    state["signatures_by_hash"] = {
                        str(k): list(v)
                        for k, v in dict(state.get("signatures_by_hash", {})).items()
                    }
                    return state

                _research_state_raw = db.get_app_setting("layered_research_state_json", "")
                try:
                    _research_state = json.loads(_research_state_raw) if _research_state_raw else {}
                    if not isinstance(_research_state, dict):
                        _research_state = {}
                except Exception:
                    _research_state = {}

                _research_state.setdefault("results", [])
                _research_state.setdefault("tested_by_hash", {})
                _research_state.setdefault("signatures_by_hash", {})
                _research_state.setdefault("run_no", 0)
                # Värske avastuse kihiring liigub püsivalt edasi ka üle eri päevade.
                # Nii ei satu väike päevane katsete arv juhuslikult kogu aeg sama kihi ümber.
                _research_state.setdefault("fresh_layer_cursor", 0)
                _research_state["run_no"] = int(_research_state.get("run_no", 0) or 0) + 1

                # Andmeseisu võti: uue korjega tekib uus hash. Vana uurimisajalugu jääb
                # suuna andmiseks alles, kuid kombinatsioonid saavad uue andmestiku peal
                # vajadusel uuesti kinnitust.
                _research_hash_cols = [
                    c for c in model_df.columns
                    if c not in {"Andmekvaliteet"}
                ]
                _research_hash_frame = model_df[_research_hash_cols].copy()
                for _c in _research_hash_frame.columns:
                    if _c == "Kuupäev":
                        _research_hash_frame[_c] = _research_hash_frame[_c].map(
                            lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v)
                        )
                _research_data_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(
                        _research_hash_frame.astype(str), index=True
                    ).values.tobytes()
                ).hexdigest()[:20]

                _tested = set(_research_state["tested_by_hash"].get(_research_data_hash, []))
                _tested_signatures = set(
                    _research_state["signatures_by_hash"].get(_research_data_hash, [])
                )

                # Kõik saadaval olevad variandid. Ükski ilmafaktor ega muu kiht pole
                # "halbade üksiktulemuste" tõttu siit eemaldatud.
                _all_specs = []
                for _layer_name in layered_screen_layers:
                    _all_specs.extend(_variant_specs(_layer_name))

                _specs_by_layer = {}
                for _spec in _all_specs:
                    _specs_by_layer.setdefault(_spec["layer"], []).append(_spec)

                # Pseudo-juhuslik, kuid korratav päevane järjestus.
                _seed_material = (
                    f"{TODAY.isoformat()}|{_research_state['run_no']}|{_research_data_hash}"
                ).encode("utf-8")
                _seed = int(hashlib.sha256(_seed_material).hexdigest()[:16], 16) % (2**32 - 1)
                _rng = np.random.default_rng(_seed)

                def _wide_day_level_series(pred_arr, target_arr, allowed_days):
                    rows = []
                    for _day in sorted(allowed_days):
                        idx = np.where(
                            (dates == _day)
                            & np.isfinite(pred_arr)
                            & np.isfinite(target_arr)
                        )[0]
                        if len(idx) == 0:
                            continue
                        rows.append({
                            "day": _day,
                            "pred": float(np.mean(pred_arr[idx])),
                            "actual": float(np.mean(target_arr[idx])),
                        })
                    return rows

                def _wide_weather_regime_days(allowed_days):
                    # Režiimimuutuse diagnostika kasutab eri ilmakanaleid koos.
                    # See ei anna ühelegi faktorile eelisõigust, vaid märgib päevi,
                    # mil kasvukeskkond muutus võrreldes eelneva päevaga palju.
                    usable = []
                    regime_cols = [
                        "BIO hiline T", "BIO hiline radiatsioon",
                        "Niiskus kesk", "ET0 Σ", "Tuul kesk",
                    ]
                    present_cols = [c for c in regime_cols if c in model_df.columns]
                    if len(present_cols) < 3:
                        return set()
                    for _day in sorted(allowed_days):
                        idx = np.where(dates == _day)[0]
                        if len(idx) == 0:
                            continue
                        vals = []
                        ok = True
                        for col in present_cols:
                            arr = pd.to_numeric(model_df.loc[idx, col], errors="coerce").to_numpy(dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if len(arr) == 0:
                                ok = False
                                break
                            vals.append(float(np.mean(arr)))
                        if ok:
                            usable.append((_day, vals))
                    if len(usable) < 4:
                        return set()
                    arr = np.asarray([
                        np.asarray(usable[i][1], dtype=float) - np.asarray(usable[i-1][1], dtype=float)
                        for i in range(1, len(usable))
                    ], dtype=float)
                    scale = np.nanstd(arr, axis=0)
                    scale = np.where(scale < 1e-6, 1.0, scale)
                    scores = np.sqrt(np.sum((arr / scale) ** 2, axis=1))
                    threshold = float(np.nanquantile(scores, 0.67))
                    return {
                        usable[i + 1][0]
                        for i, score in enumerate(scores)
                        if np.isfinite(score) and score >= threshold
                    }

                def _wide_regime_improvement(base_pred, cand_pred, target_arr, allowed_days):
                    regime_days = _wide_weather_regime_days(allowed_days)
                    if not regime_days:
                        return None
                    mask = (
                        np.isfinite(base_pred)
                        & np.isfinite(cand_pred)
                        & np.isfinite(target_arr)
                        & np.array([d in regime_days for d in dates], dtype=bool)
                    )
                    idx = np.where(mask)[0]
                    if len(idx) < 3:
                        return None
                    base_mae = float(np.mean(np.abs(base_pred[idx] - target_arr[idx])))
                    cand_mae = float(np.mean(np.abs(cand_pred[idx] - target_arr[idx])))
                    return base_mae - cand_mae

                def _wide_overreaction_penalty(pred_arr, target_arr, allowed_days):
                    rows = _wide_day_level_series(pred_arr, target_arr, allowed_days)
                    if len(rows) < 3:
                        return 0.0
                    excess = []
                    for i in range(1, len(rows)):
                        p_move = abs(rows[i]["pred"] - rows[i - 1]["pred"])
                        a_move = abs(rows[i]["actual"] - rows[i - 1]["actual"])
                        excess.append(max(0.0, p_move - a_move - 0.25))
                    return float(np.mean(excess)) if excess else 0.0

                def _eval_open_candidate(specs):
                    cols = _merge_cols_many(specs)
                    if not cols:
                        return None
                    cp = _walk_forward_with_extra(cols)
                    stats = _period_stability_stats(
                        predictions, cp, y, _abc_discovery_days,
                        min_rows=6,
                        improvement_threshold=-999.0,  # uurimisruumis ei keela midagi
                        min_half_threshold=-999.0,
                    )
                    if not stats:
                        return None

                    # Sama walk-forward ennustus on juba olemas. Arvuta kohe ka hilisema
                    # kinnituse statistika; selleks EI tehta teist kallist walk-forward ringi.
                    confirm_stats = _period_stability_stats(
                        predictions, cp, y, _abc_confirmation_days,
                        min_rows=6,
                        improvement_threshold=0.05,
                        min_half_threshold=-0.05,
                    ) if _abc_confirmation_days else None

                    # Päevataseme lihtsad lisadiagnostikad.
                    trend_acc = None
                    try:
                        rows = []
                        for _day in sorted(_abc_discovery_days):
                            idx = np.where(
                                (dates == _day)
                                & np.isfinite(cp)
                                & np.isfinite(y)
                            )[0]
                            if len(idx):
                                rows.append((
                                    _day,
                                    float(np.mean(cp[idx])),
                                    float(np.mean(y[idx])),
                                ))
                        if len(rows) >= 3:
                            hit = total = 0
                            for i in range(1, len(rows)):
                                pdiff = rows[i][1] - rows[i-1][1]
                                adiff = rows[i][2] - rows[i-1][2]
                                ps = 0 if abs(pdiff) < 0.15 else (1 if pdiff > 0 else -1)
                                ac = 0 if abs(adiff) < 0.15 else (1 if adiff > 0 else -1)
                                hit += int(ps == ac)
                                total += 1
                            trend_acc = 100.0 * hit / total if total else None
                    except Exception:
                        trend_acc = None

                    return {
                        "key": _combo_key(specs),
                        "parts": [f"{s['layer']}: {s['variant']}" for s in specs],
                        "layers": sorted({s["layer"] for s in specs}),
                        "cols": cols,
                        "pred": cp,
                        "stats": stats,
                        "confirm_stats": confirm_stats,
                        "trend_accuracy": trend_acc,
                        "regime_improvement": _wide_regime_improvement(
                            predictions, cp, y, _abc_discovery_days
                        ),
                        "overreaction": _wide_overreaction_penalty(
                            cp, y, _abc_discovery_days
                        ),
                    }

                def _result_score(row):
                    # Suund ei ole ainult MAE. Hea kandidaat võib olla väärtuslik ka
                    # stabiliseerijana: halvim pool, trend, ilmamuutuse režiimid ja
                    # ülereageerimine mõjutavad seda, kuhu lai otsing edasi liigub.
                    try:
                        improvement = float(row.get("improvement", -999.0))
                    except Exception:
                        improvement = -999.0
                    try:
                        win_share = float(row.get("win_share", 0.0))
                    except Exception:
                        win_share = 0.0
                    try:
                        worst_half = float(row.get("worst_half", -999.0))
                    except Exception:
                        worst_half = -999.0
                    try:
                        trend = float(row.get("trend_accuracy")) if row.get("trend_accuracy") is not None else 50.0
                    except Exception:
                        trend = 50.0
                    try:
                        regime = float(row.get("regime_improvement")) if row.get("regime_improvement") is not None else 0.0
                    except Exception:
                        regime = 0.0
                    try:
                        overreaction = float(row.get("overreaction", 0.0) or 0.0)
                    except Exception:
                        overreaction = 0.0

                    # Positiivne halvim-pool saab samuti väikese boonuse. Lisaks
                    # on väga väike keerukustrahv: võrdse sisulise tulemuse korral suunab
                    # see otsingu pigem lihtsama 2–3-kihilise mudeli poole, kuid ei keela
                    # tugevamal 4+ kihilisel kandidaadil võita.
                    stability_component = 0.10 * max(-1.5, min(1.5, worst_half))
                    layer_count = len(row.get("layers") or [])
                    complexity_penalty = 0.01 * max(0, layer_count - 2)
                    return (
                        improvement
                        + 0.004 * (win_share - 50.0)
                        + stability_component
                        + 0.001 * (trend - 50.0)
                        + 0.12 * regime
                        - 0.15 * overreaction
                        - complexity_penalty
                    )

                def _historical_best_keys(limit=40):
                    # Kasuta suuna andmiseks eri varasemate andmeseisude parimaid,
                    # mitte ainult tänase hetke top-1 tulemust.
                    rows = [
                        r for r in _research_state.get("results", [])
                        if r.get("combo_key")
                    ]
                    rows.sort(key=_result_score, reverse=True)
                    out = []
                    seen = set()
                    for row in rows:
                        key = str(row["combo_key"])
                        if key not in seen:
                            out.append(key)
                            seen.add(key)
                        if len(out) >= limit:
                            break
                    return out

                def _mutate_from_key(parent_key):
                    parent = _combo_from_key(parent_key)
                    if not parent:
                        return None

                    # Üks variant ühe kihi kohta.
                    by_layer = {s["layer"]: s for s in parent}
                    layers_present = list(by_layer.keys())
                    all_layers = list(_specs_by_layer.keys())

                    # Suunatud otsing peab saama head kombinatsiooni kasvatada kuni
                    # kõigi olemasolevate rollideni. Lisamine saab veidi suurema kaalu,
                    # et 4-kihiline hea vanem jõuaks päriselt 5., 6., 7. ... stabiliseeriva kihiga proovini.
                    action = int(_rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
                    if action == 0 and len(layers_present) < len(all_layers):
                        # Lisa uus kiht.
                        candidates = [l for l in all_layers if l not in by_layer]
                        if candidates:
                            layer = str(_rng.choice(candidates))
                            by_layer[layer] = _rng.choice(_specs_by_layer[layer])
                    elif action == 1 and layers_present:
                        # Vaheta ühe olemasoleva kihi variant.
                        layer = str(_rng.choice(layers_present))
                        opts = _specs_by_layer[layer]
                        if opts:
                            by_layer[layer] = _rng.choice(opts)
                    elif len(layers_present) >= 3:
                        # Vaheta üks kiht täiesti teise vastu.
                        drop_layer = str(_rng.choice(layers_present))
                        candidates = [l for l in all_layers if l not in by_layer]
                        if candidates:
                            del by_layer[drop_layer]
                            new_layer = str(_rng.choice(candidates))
                            by_layer[new_layer] = _rng.choice(_specs_by_layer[new_layer])

                    specs = list(by_layer.values())
                    return specs if len(specs) >= 2 else None

                def _fresh_explore_combo():
                    # Värske avastus peab päriselt katma eri bioloogilisi rolle.
                    # Iga värske katse saab ühe sund-ankurkihi; ankur liigub püsiva
                    # ringina läbi kõigi kihtide. Ülejäänud kihid valitakse juhuslikult.
                    # See ei lisa ühtegi walk-forward katset ega CPU-kulu, vaid muudab
                    # sama katsete arvu mitmekesisemaks.
                    layers = list(_specs_by_layer.keys())
                    if len(layers) < 2:
                        return None

                    try:
                        cursor = int(_research_state.get("fresh_layer_cursor", 0) or 0)
                    except Exception:
                        cursor = 0
                    anchor_layer = str(layers[cursor % len(layers)])
                    _research_state["fresh_layer_cursor"] = cursor + 1

                    max_k = len(layers)
                    choices = np.arange(2, max_k + 1)

                    # Kahanev, mitte keelav jaotus. Esimesed kaalud on tahtlikult
                    # pehmed, et 4–7 kihi kombinatsioonid ei muutuks praktiliselt olematuks.
                    base_weights = {
                        2: 0.28, 3: 0.25, 4: 0.18, 5: 0.12,
                        6: 0.08, 7: 0.05, 8: 0.025, 9: 0.015,
                    }
                    weights = np.array([
                        base_weights.get(int(k), max(0.005, 0.015 * (0.72 ** max(0, int(k) - 9))))
                        for k in choices
                    ], dtype=float)
                    weights = weights / weights.sum()
                    k = int(_rng.choice(choices, p=weights))

                    other_layers = [str(layer) for layer in layers if str(layer) != anchor_layer]
                    extra_count = max(1, k - 1)
                    extra_count = min(extra_count, len(other_layers))
                    picked_layers = [anchor_layer]
                    if extra_count:
                        picked_layers.extend(
                            str(layer)
                            for layer in _rng.choice(other_layers, size=extra_count, replace=False)
                        )
                    return [
                        _rng.choice(_specs_by_layer[str(layer)])
                        for layer in picked_layers
                    ]

                # Väikese päevase katsearvu juures hoiame avastuse laiemana:
                # kõige rohkem 2 vana tugevat kombinatsiooni korduskinnituseks ning
                # pärast seda 50/50 suunatud otsing vs täiesti värske avastus.
                _best_parent_keys = _historical_best_keys(limit=40)
                _revalidate_queue = []
                for _key in _best_parent_keys[:2]:
                    _specs = _combo_from_key(_key)
                    if _specs and _combo_key(_specs) not in _tested:
                        _revalidate_queue.append(_specs)

                _run_results = []
                _attempts = 0
                _candidate_durations = []
                _max_attempts = 500  # kaitse juhul, kui palju duplikaate tekib

                while (
                    time.perf_counter() < (_layered_deadline - _layered_stop_margin_s)
                    and _attempts < _max_attempts
                ):
                    _attempts += 1

                    if _revalidate_queue:
                        _specs = _revalidate_queue.pop(0)
                        _mode = "kinnitus"
                    else:
                        exploit = bool(_best_parent_keys) and (_rng.random() < 0.50)
                        if exploit:
                            parent = str(_rng.choice(_best_parent_keys))
                            _specs = _mutate_from_key(parent)
                            _mode = "suunatud"
                        else:
                            _specs = _fresh_explore_combo()
                            _mode = "avastus"

                    if not _specs:
                        continue
                    _key = _combo_key(_specs)
                    _signature = _feature_signature(_specs)
                    if not _key or _key in _tested or _signature in _tested_signatures:
                        continue

                    # Kui aega on liiga vähe, uut kallist walk-forward'i ei alusta.
                    # Pärast esimesi katseid kasutame päriselt mõõdetud kandidaadi kestust,
                    # et dünaamilisest CPU-piirist ühe pika viimase katsega üle ei sõidaks.
                    _now = time.perf_counter()
                    _remaining = _layered_deadline - _now
                    _expected_next = (
                        max(_layered_stop_margin_s, 1.35 * float(np.mean(_candidate_durations)))
                        if _candidate_durations
                        else max(_layered_stop_margin_s, 1.35 * float(_layered_saved_candidate_avg_s))
                    )
                    if _remaining <= _expected_next:
                        break

                    _candidate_t0 = time.perf_counter()
                    _cand = _eval_open_candidate(_specs)
                    _candidate_durations.append(time.perf_counter() - _candidate_t0)
                    _tested.add(_key)
                    _tested_signatures.add(_signature)

                    # Ka statistilise raportireata katse on päriselt tehtud CPU-töö.
                    # Salvesta see kohe, et restart/järgmine voor seda ei kordaks.
                    _research_state["tested_by_hash"][_research_data_hash] = list(_tested)
                    _research_state["signatures_by_hash"][_research_data_hash] = list(_tested_signatures)
                    if _cand is None:
                        try:
                            db.set_app_setting(
                                "layered_research_state_json",
                                json.dumps(
                                    _serialize_research_state(_research_state),
                                    ensure_ascii=False,
                                ),
                            )
                        except Exception:
                            pass
                        continue

                    _s = _cand["stats"]
                    _row = {
                        "combo_key": _key,
                        "tested_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                        "data_hash": _research_data_hash,
                        "training_rows": int(len(model_df)),
                        "mode": _mode,
                        "layers": _cand["layers"],
                        "parts": _cand["parts"],
                        "cols": _cand["cols"],
                        "base_mae": float(_s["Baas MAE"]),
                        "candidate_mae": float(_s["Katse MAE"]),
                        "improvement": float(_s["Paranemine"]),
                        "win_share": float(_s["Võidab ridu %"]),
                        "worst_half": float(_s["Halvim pool"]),
                        "test_rows": int(_s["Testiridu"]),
                        "trend_accuracy": (
                            None if _cand["trend_accuracy"] is None
                            else float(_cand["trend_accuracy"])
                        ),
                        "regime_improvement": (
                            None if _cand.get("regime_improvement") is None
                            else float(_cand.get("regime_improvement"))
                        ),
                        "overreaction": float(_cand.get("overreaction", 0.0) or 0.0),
                        "confirmation_base_mae": (
                            None if not _cand.get("confirm_stats")
                            else float(_cand["confirm_stats"]["Baas MAE"])
                        ),
                        "confirmation_candidate_mae": (
                            None if not _cand.get("confirm_stats")
                            else float(_cand["confirm_stats"]["Katse MAE"])
                        ),
                        "confirmation_improvement": (
                            None if not _cand.get("confirm_stats")
                            else float(_cand["confirm_stats"]["Paranemine"])
                        ),
                        "confirmation_win_share": (
                            None if not _cand.get("confirm_stats")
                            else float(_cand["confirm_stats"]["Võidab ridu %"])
                        ),
                        "confirmation_worst_half": (
                            None if not _cand.get("confirm_stats")
                            else float(_cand["confirm_stats"]["Halvim pool"])
                        ),
                        "confirmation_test_rows": (
                            0 if not _cand.get("confirm_stats")
                            else int(_cand["confirm_stats"]["Testiridu"])
                        ),
                        "confirmation_stable": bool(
                            _cand.get("confirm_stats") and _cand["confirm_stats"].get("Stabiilne")
                        ),
                    }
                    _research_state["results"].append(_row)
                    _run_results.append(_row)

                    # IGA valmis katse järel progress püsivalt kirja.
                    _research_state["tested_by_hash"][_research_data_hash] = list(_tested)
                    _research_state["signatures_by_hash"][_research_data_hash] = list(_tested_signatures)
                    try:
                        db.set_app_setting(
                            "layered_research_state_json",
                            json.dumps(
                                _serialize_research_state(_research_state),
                                ensure_ascii=False,
                            ),
                        )
                        db.set_app_setting(
                            "layered_research_progress_json",
                            json.dumps({
                                "started_at": _layered_started_at.isoformat(),
                                "last_saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                                "data_hash": _research_data_hash,
                                "tested_this_run": len(_run_results),
                                "last_combo": _key,
                                "budget_s": LAYERED_DAILY_BUDGET_S,
                            }, ensure_ascii=False),
                        )
                    except Exception:
                        pass

                # Raport: tänase andmeseisu parimad + viimane jooks.
                _current_results = [
                    r for r in _research_state["results"]
                    if r.get("data_hash") == _research_data_hash
                ]
                _current_results.sort(key=_result_score, reverse=True)
                _top_rows = _current_results[:30]

                layered_trace_rows = []
                for _r in _top_rows:
                    layered_trace_rows.append({
                        "Mudel": "Avatud kihiline otsing",
                        "Kihte": len(_r.get("layers") or []),
                        "Koosseis": " | ".join(_r.get("parts") or []),
                        "Baas MAE": _r.get("base_mae"),
                        "Katse MAE": _r.get("candidate_mae"),
                        "Paranemine": _r.get("improvement"),
                        "Võidab ridu %": _r.get("win_share"),
                        "Halvim pool": _r.get("worst_half"),
                        "Testiridu": _r.get("test_rows"),
                        "Trend %": _r.get("trend_accuracy"),
                        "Režiim Δ": _r.get("regime_improvement"),
                        "Ülereageerimine": _r.get("overreaction"),
                        "Stabiilne": bool(
                            (_r.get("improvement") or -999) >= 0.05
                            and (_r.get("win_share") or 0) >= 50.0
                            and (_r.get("worst_half") or -999) >= -0.10
                        ),
                        "Kinnitus MAE": _r.get("confirmation_candidate_mae"),
                        "Kinnitus Δ": _r.get("confirmation_improvement"),
                        "Kinnitus võidab %": _r.get("confirmation_win_share"),
                        "Kinnitus ridu": _r.get("confirmation_test_rows"),
                        "Kinnitus stabiilne": bool(_r.get("confirmation_stable")),
                    })

                layered_trace_df = pd.DataFrame(layered_trace_rows)
                if not layered_trace_df.empty:
                    layered_trace_df = layered_trace_df.sort_values(
                        ["Stabiilne", "Paranemine", "Võidab ridu %"],
                        ascending=[False, False, False],
                    ).reset_index(drop=True)

                # Championiks pääseb ainult avastuse järel eraldi hilisemal
                # confirmation-perioodil kinnitust saanud kandidaat. App-120 ei kontrolli
                # enam ainult avastusjärjestuse nr 1 kandidaati: iga UUS kandidaat sai
                # kinnituse statistika juba sama walk-forward ennustuse pealt tasuta.
                # Vana app-119 top-challengeri kinnitus tuuakse ühekordselt state'ist kaasa,
                # kui see kuulub samasse andmeseisu.
                _prior_best_ch = _load_json_setting("layered_best_challenger_json", {})
                if isinstance(_prior_best_ch, dict) and _prior_best_ch.get("data_hash") == _research_data_hash:
                    _prior_cols = list(_prior_best_ch.get("cols") or [])
                    _prior_conf = _prior_best_ch.get("confirmation_stats")
                    if _prior_cols and isinstance(_prior_conf, dict) and _prior_conf:
                        for _r in _current_results:
                            if list(_r.get("cols") or []) == _prior_cols and not _r.get("confirmation_test_rows"):
                                _r["confirmation_base_mae"] = _prior_conf.get("Baas MAE")
                                _r["confirmation_candidate_mae"] = _prior_conf.get("Katse MAE")
                                _r["confirmation_improvement"] = _prior_conf.get("Paranemine")
                                _r["confirmation_win_share"] = _prior_conf.get("Võidab ridu %")
                                _r["confirmation_worst_half"] = _prior_conf.get("Halvim pool")
                                _r["confirmation_test_rows"] = _prior_conf.get("Testiridu", 0)
                                _r["confirmation_stable"] = bool(_prior_conf.get("Stabiilne"))
                                break

                _confirmed_rows = [
                    _r for _r in _current_results
                    if bool(_r.get("confirmation_stable"))
                    and _r.get("confirmation_candidate_mae") is not None
                ]
                _selected_confirmed_row = None
                if _confirmed_rows:
                    # Kõigepealt parim kinnitus-MAE. Kui tulemused on sisuliselt võrdsed
                    # (<=0.03 kasti), eelista lihtsamat mudelit; seejärel suuremat
                    # võiduprotsenti ja alles siis imepisikest MAE erinevust.
                    _best_confirm_mae = min(float(_r["confirmation_candidate_mae"]) for _r in _confirmed_rows)
                    _near_best = [
                        _r for _r in _confirmed_rows
                        if float(_r["confirmation_candidate_mae"]) <= _best_confirm_mae + 0.03
                    ]
                    _near_best.sort(key=lambda _r: (
                        len(_r.get("layers") or []),
                        -float(_r.get("confirmation_win_share") or 0.0),
                        float(_r.get("confirmation_candidate_mae") or 999.0),
                        -float(_r.get("improvement") or -999.0),
                    ))
                    _selected_confirmed_row = _near_best[0]

                if _selected_confirmed_row is not None:
                    _selected_cols = list(_selected_confirmed_row.get("cols") or [])
                    if _selected_cols:
                        # Üksainus walk-forward rekonstruktsioon on vajalik, et ülejäänud
                        # diagnostika saaks championi ajaloolist prediktsioonijada kasutada.
                        # Kandidaatide KINNITUSE valik ise uut walk-forward'i ei vajanud.
                        _selected_pred = _walk_forward_with_extra(_selected_cols)
                        champion_name = "Avatud kihiline champion"
                        champion_cols = _selected_cols
                        champion_pred = _selected_pred
                        champion_mae = float(_selected_confirmed_row["confirmation_candidate_mae"])
                        champion_stats = {
                            "Baas MAE": float(_selected_confirmed_row.get("confirmation_base_mae")),
                            "Katse MAE": float(_selected_confirmed_row.get("confirmation_candidate_mae")),
                            "Paranemine": float(_selected_confirmed_row.get("confirmation_improvement")),
                            "Võidab ridu %": float(_selected_confirmed_row.get("confirmation_win_share")),
                            "Halvim pool": float(_selected_confirmed_row.get("confirmation_worst_half")),
                            "Testiridu": int(_selected_confirmed_row.get("confirmation_test_rows") or 0),
                            "Stabiilne": True,
                            "Kinnitusperiood": True,
                            "Kihid": " | ".join(_selected_confirmed_row.get("parts") or []),
                        }

                # Variprognoosile salvesta parim ametlikust championist erinev alternatiiv.
                # Kui champion on endiselt baas, on see lihtsalt parim kihiline challenger.
                _alternative_rows = [
                    _r for _r in _current_results
                    if list(_r.get("cols") or []) != list(champion_cols)
                ]
                _alternative_rows.sort(key=_result_score, reverse=True)
                if _alternative_rows:
                    _best_challenger = _alternative_rows[0]
                    _best_challenger_confirm_stats = None
                    if _best_challenger.get("confirmation_test_rows"):
                        _best_challenger_confirm_stats = {
                            "Baas MAE": _best_challenger.get("confirmation_base_mae"),
                            "Katse MAE": _best_challenger.get("confirmation_candidate_mae"),
                            "Paranemine": _best_challenger.get("confirmation_improvement"),
                            "Võidab ridu %": _best_challenger.get("confirmation_win_share"),
                            "Halvim pool": _best_challenger.get("confirmation_worst_half"),
                            "Testiridu": _best_challenger.get("confirmation_test_rows"),
                            "Stabiilne": bool(_best_challenger.get("confirmation_stable")),
                        }
                    _save_json_setting("layered_best_challenger_json", {
                        "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                        "harvest_day": last_complete_harvest.isoformat() if last_complete_harvest else None,
                        "data_hash": _research_data_hash,
                        "name": "Avatud kihiline otsing",
                        "parts": list(_best_challenger.get("parts") or []),
                        "cols": list(_best_challenger.get("cols") or []),
                        "base_mae": _best_challenger.get("base_mae"),
                        "candidate_mae": _best_challenger.get("candidate_mae"),
                        "improvement": _best_challenger.get("improvement"),
                        "win_share": _best_challenger.get("win_share"),
                        "worst_half": _best_challenger.get("worst_half"),
                        "test_rows": _best_challenger.get("test_rows"),
                        "trend_accuracy": _best_challenger.get("trend_accuracy"),
                        "regime_improvement": _best_challenger.get("regime_improvement"),
                        "overreaction": _best_challenger.get("overreaction"),
                        "confirmed": bool(_best_challenger.get("confirmation_stable")),
                        "confirmation_stats": _best_challenger_confirm_stats,
                    })

                _layered_elapsed = max(0.0, time.perf_counter() - _layered_t0)
                _layered_elapsed_s_this_cycle = _layered_elapsed
                _cpu_record_research("layered", _layered_elapsed)
                try:
                    # State ja koormus salvestuvad alati. Päev märgitakse edukalt
                    # uurituks ainult siis, kui vähemalt üks uus hinnatav kandidaat lõpetati.
                    db.set_app_setting("layered_research_last_duration_s", f"{_layered_elapsed:.1f}")
                    db.set_app_setting("layered_research_last_candidates", str(len(_current_results)))
                    db.set_app_setting("layered_research_last_run_candidates", str(len(_run_results)))
                    db.set_app_setting("layered_research_last_budget_s", str(LAYERED_DAILY_BUDGET_S))
                    if _candidate_durations:
                        db.set_app_setting(
                            "layered_research_last_candidate_avg_s",
                            f"{float(np.mean(_candidate_durations)):.2f}",
                        )
                    db.set_app_setting(
                        "layered_research_state_json",
                        json.dumps(
                            _serialize_research_state(_research_state),
                            ensure_ascii=False,
                        ),
                    )

                    if _run_results:
                        _save_df = layered_trace_df.copy().where(pd.notna(layered_trace_df), None)
                        db.set_app_setting(
                            "layered_research_trace_json",
                            json.dumps(_save_df.to_dict(orient="records"), ensure_ascii=False),
                        )
                        db.set_app_setting("layered_research_last_at", _layered_started_at.isoformat())
                        if last_complete_harvest is not None:
                            db.set_app_setting(
                                "layered_research_last_harvest_day",
                                last_complete_harvest.isoformat(),
                            )
                        db.set_app_setting("layered_research_last_error", "")
                    else:
                        db.set_app_setting(
                            "layered_research_last_error",
                            "Uurimisvoor lõpetas ilma ühegi uue hinnatava kandidaadita; päev jäi uuesti proovitavaks.",
                        )
                except Exception as _e:
                    db.set_app_setting("layered_research_last_error", str(_e))

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

            # Lai loominguline/kihtide otsing on käsitsi. Päevade arv ei käivita seda.
            # Vahepeal kasutatakse viimati salvestatud uuringutulemust.
            # Avastusruumi versioonivõti. Uue BIO-kihi lisamisel peab vähemalt üks
            # lai ring päriselt uuesti jooksma; vana Fix06 "done" lipp ei tohi uut
            # kandidaatruumi vahele jätta.
            _force_brainstorm_key = "v65_bio_discovery_v1_done"
            _force_brainstorm_once = db.get_app_setting(_force_brainstorm_key, "0") != "1"
            # Lai vabade ideede otsing ei käivitu enam automaatselt rerun'idel.
            AUTONOMOUS_DISCOVERY_ENABLED = False

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
                    "v65bio-v1|"
                    + _auto_data_hash
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
                    "Ilmamälu": 0,
                    "Hooaeg × ilm": 0,
                    "Bioloogiline koormus": 0,
                    "BIO füsioloogia": 0,
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

                # BIO kandidaatkiht: teaduskirjandusest motiveeritud, kuid
                # endiselt ainult walk-forward testitavad kandidaadid.
                _bio_candidates = {
                    "BIO GDD10": "Termiline areng: temperatuurisumma üle 10 °C baasi.",
                    "BIO radiatsioonisumma": "Assimilaatide pakkumise lihtne radiatsioonisumma.",
                    "BIO rad/GDD10": "Assimilaatide pakkumise ja termilise arengunõudluse tasakaalu proxy.",
                    "BIO hiline T": "Korje-eelsema kasvuperioodi hilisematele päevadele rohkem kaalutud temperatuur.",
                    "BIO hiline radiatsioon": "Korje-eelsema kasvuperioodi hilisematele päevadele rohkem kaalutud radiatsioon.",
                }
                for _bio_name, _bio_hyp in _bio_candidates.items():
                    if _bio_name in model_df.columns:
                        _register_discovery_feature(
                            _bio_name,
                            pd.to_numeric(model_df[_bio_name], errors="coerce").to_numpy(dtype=float),
                            "BIO füsioloogia",
                        )

                # 6) Pikem ilmamälu: proovi 7/10/14 päeva tunnuseid ka üksikult ja
                # ilmastressi kombinatsioonidena. Need ulatuvad üle eelmise korje piiri,
                # kuid EI kasuta eelmist saaki ankruna.
                for _days in (7, 10, 14):
                    _cols = {
                        "ÖöT": f"ÖöT {_days}p",
                        "PäevT": f"PäevT {_days}p",
                        "Rad": f"Rad {_days}p",
                        "Sade": f"Sade {_days}p",
                        "ET0": f"ET0 {_days}p",
                        "Niiskus": f"Niiskus {_days}p",
                    }
                    for _label, _col in _cols.items():
                        if _col in model_df.columns:
                            _register_discovery_feature(
                                f"{_days}p {_label}",
                                pd.to_numeric(model_df[_col], errors="coerce").to_numpy(dtype=float),
                                "Ilmamälu",
                            )
                    if _cols["ÖöT"] in model_df.columns and _cols["Rad"] in model_df.columns:
                        _register_discovery_feature(
                            f"{_days}p ööT × radiatsioon",
                            pd.to_numeric(model_df[_cols["ÖöT"]], errors="coerce").to_numpy(dtype=float)
                            * pd.to_numeric(model_df[_cols["Rad"]], errors="coerce").to_numpy(dtype=float),
                            "Ilmamälu",
                        )
                    if _cols["ET0"] in model_df.columns and _cols["Sade"] in model_df.columns:
                        _register_discovery_feature(
                            f"{_days}p veebilanss ET0−sade",
                            pd.to_numeric(model_df[_cols["ET0"]], errors="coerce").to_numpy(dtype=float)
                            - pd.to_numeric(model_df[_cols["Sade"]], errors="coerce").to_numpy(dtype=float),
                            "Ilmamälu",
                        )

                # 7) Hooaja kulumine × ilm: kas sama kasvuilm annab hooaja hilisemas osas
                # väiksema vastuse. Märki ei sunnita ette; mudel peab seose leidma.
                _season_candidates = ["Hooajapäev²", "Hooaeg 50+", "Hooaeg 65+", "Päevapikkus"]
                _weather_response = ["ÖöT kesk", "PäevT kesk", "Radiatsioon/p", "ET0 Σ"]
                for _s in _season_candidates:
                    if _s not in model_df.columns:
                        continue
                    _xs = pd.to_numeric(model_df[_s], errors="coerce").to_numpy(dtype=float)
                    for _w in _weather_response:
                        if _w not in model_df.columns:
                            continue
                        _xw = pd.to_numeric(model_df[_w], errors="coerce").to_numpy(dtype=float)
                        _register_discovery_feature(
                            f"{_s} × {_w}",
                            _xs * _xw,
                            "Hooaeg × ilm",
                        )

                # 8) Pikem bioloogiline koormusmälu. Kasutame ainult normaliseeritud
                # koormusindekseid; toorest eelmist saaki ei muudeta operatiivseks ankruks.
                _load_cols = [c for c in ["Koormusindeks -1", "2 korje koormus", "Tipukorje -1", "Tipukorje -2"] if c in model_df.columns]
                for _c in _load_cols:
                    _register_discovery_feature(
                        f"Koormus: {_c}",
                        pd.to_numeric(model_df[_c], errors="coerce").to_numpy(dtype=float),
                        "Bioloogiline koormus",
                    )
                if "Koormusindeks -1" in model_df.columns and "2 korje koormus" in model_df.columns:
                    _k1 = pd.to_numeric(model_df["Koormusindeks -1"], errors="coerce").to_numpy(dtype=float)
                    _k2 = pd.to_numeric(model_df["2 korje koormus"], errors="coerce").to_numpy(dtype=float)
                    _register_discovery_feature(
                        "Koormusmälu kombineeritud",
                        0.6 * _k1 + 0.4 * _k2,
                        "Bioloogiline koormus",
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
                    if _force_brainstorm_once:
                        db.set_app_setting(_force_brainstorm_key, "1")
                except Exception as _auto_store_exc:
                    db.set_app_setting("idea_full_search_last_error", str(_auto_store_exc))

            st.markdown("##### Tänane champion-mootor")
            ch1, ch2, ch3 = st.columns(3)
            ch1.metric("Champion", champion_name)
            ch2.metric("A+B+C MAE", "—" if champion_mae is None else f"{champion_mae:.2f} kasti")
            if champion_stats:
                ch3.metric("Eelis baasi ees", f"{champion_stats['Paranemine']:+.2f} kasti")
                st.success(
                    f"9 päeva prognoos kasutab kinnitatud championit **{champion_name}**. "
                    f"Kandidaat valiti varasemal avastusperioodil ja alles seejärel kontrolliti "
                    f"hilisemal puutumata kinnitusaajal. Kinnituses võitis ta "
                    f"{champion_stats['Võidab ridu %']:.0f}% testiridadest ning parandas MAE-d "
                    f"{champion_stats['Paranemine']:.2f} kasti. Champion vaadatakse iga uue korjega uuesti üle."
                )
            else:
                ch3.metric("Eelis baasi ees", "0.00 kasti")
                st.info(
                    "Ükski vähemalt kahe lisakihiga BIO-mudel ei läbinud eraldi avastus- ja kinnitusetappi. "
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
                "Kasvuilm + kasvuaeg": ["Kasvuaeg p", "BIO hiline T", "BIO hiline radiatsioon"],
            }

            def _xl_walk_with_extra(extra_cols, alpha=10.0, cache_key=None):
                raw_extra = (
                    model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    if extra_cols else np.empty((len(model_df), 0))
                )
                extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]

                def _predict_day(test_day, preds):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where((dates < test_day) & np.isfinite(y_xl))[0]
                    if len(train_idx) < min_train_rows:
                        return
                    preds[test_idx] = _ridge_walk_predict(
                        y_xl, extra_arrays, train_idx, test_idx, alpha=alpha
                    )

                if cache_key is not None:
                    preds, mode = _light_walk_cached("xl", str(cache_key), _predict_day)
                    return preds

                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    _predict_day(test_day, preds)
                return preds

            _xl_discovery_days, _xl_confirmation_days = _chronological_discovery_confirmation_days(xl_predictions, y_xl)

            def _xl_stability_stats(candidate_pred):
                all_days = set(sorted(set(dates[np.where(np.isfinite(xl_predictions) & np.isfinite(y_xl))[0]])))
                return _period_stability_stats(
                    xl_predictions, candidate_pred, y_xl, all_days,
                    min_rows=12, improvement_threshold=0.05, min_half_threshold=-0.05,
                )

            xl_candidate_predictions = {}
            xl_trace_rows = []
            if _run_light_research:
                for name, cols in xl_candidate_groups.items():
                    pred = _xl_walk_with_extra(cols, cache_key=name)
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
                xl_discovery = []
                for name, cols in xl_candidate_groups.items():
                    cp = xl_candidate_predictions.get(name)
                    if cp is None:
                        continue
                    d_stats = _period_stability_stats(
                        xl_predictions, cp, y_xl, _xl_discovery_days,
                        min_rows=6, improvement_threshold=0.05, min_half_threshold=-0.05,
                    )
                    if d_stats and d_stats["Stabiilne"]:
                        xl_discovery.append((d_stats["Katse MAE"], name, cols, cp, d_stats))
                if xl_discovery and _xl_confirmation_days:
                    xl_discovery.sort(key=lambda x: x[0])
                    _, selected_name, selected_cols, selected_pred, selected_discovery_stats = xl_discovery[0]
                    confirm_stats = _period_stability_stats(
                        xl_predictions, selected_pred, y_xl, _xl_confirmation_days,
                        min_rows=6, improvement_threshold=0.03, min_half_threshold=-0.05,
                    )
                    if confirm_stats and confirm_stats["Stabiilne"]:
                        xl_champion_name = selected_name
                        xl_champion_cols = selected_cols
                        xl_champion_mae = confirm_stats["Katse MAE"]
                        xl_champion_stats = dict(confirm_stats)
                        xl_champion_stats["Avastus MAE"] = selected_discovery_stats["Katse MAE"]
                        xl_champion_stats["Avastus paranemine"] = selected_discovery_stats["Paranemine"]
                _save_json_setting("xl_champion_state_json", {
                    "name": xl_champion_name, "cols": list(xl_champion_cols),
                    "mae": xl_champion_mae, "stats": xl_champion_stats,
                    "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                })
            else:
                _saved_xl_champion = _load_json_setting("xl_champion_state_json", {})
                xl_champion_name = str((_saved_xl_champion or {}).get("name") or "XL weather-first baasmudel")
                xl_champion_cols = [
                    str(c) for c in ((_saved_xl_champion or {}).get("cols") or [])
                    if str(c) in model_df.columns
                ]
                xl_champion_mae = (_saved_xl_champion or {}).get("mae")
                if xl_champion_mae is None and valid_xl.any():
                    xl_champion_mae = float(np.mean(np.abs(xl_predictions[valid_xl] - y_xl[valid_xl])))
                xl_champion_stats = (_saved_xl_champion or {}).get("stats")

            xl1, xl2 = st.columns(2)
            xl1.metric("XL champion", xl_champion_name)
            xl2.metric("XL MAE", "—" if xl_champion_mae is None else f"{xl_champion_mae:.2f}")
            if xl_champion_stats:
                st.success(
                    f"XL kasutab kinnitatud championit **{xl_champion_name}**: hilisemal kinnitusaajal "
                    f"paranes MAE {xl_champion_stats['Paranemine']:.2f} ja kandidaat võitis "
                    f"{xl_champion_stats['Võidab ridu %']:.0f}% kinnitusridadest."
                )
            else:
                st.info(
                    "Ükski XL mälutunnus ega lisajälg ei läbinud eraldi avastus- ja kinnitusetappi; "
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

            def _cb_walk_with_extra(extra_cols, alpha=10.0, cache_key=None):
                raw_extra = model_df[extra_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if extra_cols else np.empty((len(model_df), 0))
                extra_arrays = [raw_extra[:, j] for j in range(raw_extra.shape[1])]

                def _predict_day(test_day, preds):
                    test_idx = np.where(dates == test_day)[0]
                    train_idx = np.where((dates < test_day) & np.isfinite(log_y_cb))[0]
                    if len(train_idx) < min_train_rows:
                        return
                    log_pred = _ridge_walk_predict(
                        log_y_cb, extra_arrays, train_idx, test_idx,
                        alpha=alpha, floor_zero=False
                    )
                    preds[test_idx] = np.exp(np.clip(log_pred, np.log(0.10), np.log(10.0)))

                if cache_key is not None:
                    preds, mode = _light_walk_cached("cb", str(cache_key), _predict_day)
                    return preds

                preds = np.full(len(model_df), np.nan, dtype=float)
                for test_day in sorted(set(dates)):
                    _predict_day(test_day, preds)
                return preds

            _cb_discovery_days, _cb_confirmation_days = _chronological_discovery_confirmation_days(cb_predictions, y_cb)

            def _cb_stability_stats(candidate_pred):
                all_days = set(sorted(set(dates[np.where(np.isfinite(cb_predictions) & np.isfinite(y_cb))[0]])))
                return _period_stability_stats(
                    cb_predictions, candidate_pred, y_cb, all_days,
                    min_rows=12, improvement_threshold=0.05, min_half_threshold=-0.03,
                )

            cb_candidate_predictions = {}
            cb_trace_rows = []
            if _run_light_research:
                for name, cols in cb_candidate_groups.items():
                    pred = _cb_walk_with_extra(cols, cache_key=name)
                    cb_candidate_predictions[name] = pred
                    stats = _cb_stability_stats(pred)
                    if stats:
                        cb_trace_rows.append({"Jälg": name, **stats})

                cb_champion_name = "C/B baasmudel"
                cb_champion_cols = []
                cb_champion_mae = cb_base_mae
                cb_champion_stats = None
                cb_discovery = []
                for name, cols in cb_candidate_groups.items():
                    cp = cb_candidate_predictions.get(name)
                    if cp is None:
                        continue
                    d_stats = _period_stability_stats(
                        cb_predictions, cp, y_cb, _cb_discovery_days,
                        min_rows=6, improvement_threshold=0.05, min_half_threshold=-0.03,
                    )
                    if d_stats and d_stats["Stabiilne"]:
                        cb_discovery.append((d_stats["Katse MAE"], name, cols, cp, d_stats))
                if cb_discovery and _cb_confirmation_days:
                    cb_discovery.sort(key=lambda x: x[0])
                    _, selected_name, selected_cols, selected_pred, selected_discovery_stats = cb_discovery[0]
                    confirm_stats = _period_stability_stats(
                        cb_predictions, selected_pred, y_cb, _cb_confirmation_days,
                        min_rows=6, improvement_threshold=0.03, min_half_threshold=-0.03,
                    )
                    if confirm_stats and confirm_stats["Stabiilne"]:
                        cb_champion_name = selected_name
                        cb_champion_cols = selected_cols
                        cb_champion_mae = confirm_stats["Katse MAE"]
                        cb_champion_stats = dict(confirm_stats)
                        cb_champion_stats["Avastus MAE"] = selected_discovery_stats["Katse MAE"]
                        cb_champion_stats["Avastus paranemine"] = selected_discovery_stats["Paranemine"]
                _save_json_setting("cb_champion_state_json", {
                    "name": cb_champion_name, "cols": list(cb_champion_cols),
                    "mae": cb_champion_mae, "stats": cb_champion_stats,
                    "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                })
            else:
                _saved_cb_champion = _load_json_setting("cb_champion_state_json", {})
                cb_champion_name = str((_saved_cb_champion or {}).get("name") or "C/B baasmudel")
                cb_champion_cols = [
                    str(c) for c in ((_saved_cb_champion or {}).get("cols") or [])
                    if str(c) in model_df.columns
                ]
                cb_champion_mae = (_saved_cb_champion or {}).get("mae")
                if cb_champion_mae is None:
                    cb_champion_mae = cb_base_mae
                cb_champion_stats = (_saved_cb_champion or {}).get("stats")

            # Cache'i täielik row-hash kirjutatakse ainult päris Jäljeotsija ringis
            # või täpsel cache-hit'il. Kui uus korjepäev jäi CPU tõttu uurimata,
            # säilitame vana prefixi, et järgmine lubatud ring saaks teha incremental sammu.
            if _run_light_research or _light_exact_cache:
                _save_light_walk_cache()

            if _run_light_research:
                _light_research_elapsed_measured_s = max(
                    0.0,
                    time.perf_counter() - _light_cycle_t0 - float(_layered_elapsed_s_this_cycle or 0.0),
                )

            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("C/B champion", cb_champion_name)
            cb2.metric("C/B MAE", "—" if cb_champion_mae is None else f"{cb_champion_mae:.2f}")
            cb_test_rows = int(valid_cb_base.sum())
            cb_test_days = len(set(dates[np.where(valid_cb_base)[0]])) if valid_cb_base.any() else 0
            cb3.metric("Testitud", f"{cb_test_rows}/{len(model_df)} rida")
            st.caption(f"C/B aus test hõlmab {cb_test_days} testipäeva. Madalam MAE on parem.")
            if cb_champion_stats:
                st.success(
                    f"C/B prognoos kasutab kinnitatud championit **{cb_champion_name}**: hilisemal kinnitusaajal "
                    f"paranes MAE {cb_champion_stats['Paranemine']:.2f} ja kandidaat võitis "
                    f"{cb_champion_stats['Võidab ridu %']:.0f}% kinnitusridadest."
                )
            else:
                st.info("Ükski C/B lisajälg ei läbinud eraldi avastus- ja kinnitusetappi; kvaliteediprognoos kasutab C/B baasmudelit.")

            if cb_trace_rows:
                with st.expander("Näita C/B jäljeotsija tulemusi"):
                    cb_trace_df = pd.DataFrame(cb_trace_rows).sort_values(["Stabiilne", "Paranemine"], ascending=[False, False])
                    st.dataframe(cb_trace_df.style.format({
                        "Baas MAE": "{:.2f}", "Katse MAE": "{:.2f}", "Paranemine": "{:+.2f}",
                        "Võidab ridu %": "{:.0f}%", "Halvim pool": "{:+.2f}",
                    }), use_container_width=True, hide_index=True)

            def _fit_full_generic(target, extra_arrays, alpha=10.0, field_alpha=80.0):
                return _engine_fit_full_generic(
                    X_base, fields, target, extra_arrays, alpha=alpha, field_alpha=field_alpha,
                )

            def _predict_full_generic(model, field_no, base_values, extra_values, floor_zero=True):
                return _engine_predict_full_generic(
                    model, field_no, base_values, extra_values, floor_zero=floor_zero,
                )

            def _fit_full_abc_growth(extra_arrays, alpha=10.0, field_alpha=80.0, z_clip=2.5):
                return _engine_fit_full_abc_growth(
                    X_base, fields, log_y_abc, extra_arrays, alpha=alpha,
                    field_alpha=field_alpha, z_clip=z_clip,
                )

            def _predict_full_abc_growth(model, field_no, base_values, extra_values):
                return _engine_predict_full_abc_growth(
                    model, field_no, base_values, extra_values, log_eps=ABC_LOG_EPS,
                )

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
                    if name == "Hooajapäev" or str(name).startswith("Hooaeg"):
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

            if champion_name != "Baasmudel" and not champion_cols:
                champion_name = "Baasmudel"
                champion_stats = None
                champion_mae = current_test_mae

            _save_json_setting("abc_champion_state_json", {
                "name": champion_name,
                "cols": list(champion_cols),
                "mae": champion_mae,
                "stats": champion_stats,
                "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
            })

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

            # Rekursiivses 9 päeva prognoosis ei tohi XL ega C/B omaenda varasemaid
            # prognoose toore mälutunnusena tagasi sisendisse sööta. Ilmapõhine champion
            # võib jätkuda; mäluchampioni korral kasutatakse prognoositud eelkorje järel baasmudelit.
            xl_memory_feature_names = {"XL -1", "XL -2", "Eelmine ABC", "Eelmine saak"}
            cb_memory_feature_names = {
                "C/B -1", "C/B -2", "Eelmine2 ABC", "ABC trend",
                "XL -1", "XL -2", "XL osakaal -1", "XL osakaal -2",
            }
            xl_champion_uses_memory = any(c in xl_memory_feature_names for c in xl_champion_cols)
            cb_champion_uses_memory = any(c in cb_memory_feature_names for c in cb_champion_cols)
            full_xl_base_model = _fit_full_generic(y_xl, [])
            full_cb_base_model = _fit_full_generic(log_y_cb, [])

            # -------------------------------------------------------------------------
            # 9 päeva ette: A+B+C + eraldi XL
            # -------------------------------------------------------------------------
            st.markdown("##### 9 päeva saagiprognoos")
            st.caption(
                f"A+B+C kasutab tänast champion-mootorit: {champion_name}. Baasmudel põhineb ilmal, intervallil, "
                f"hooaja faasil ja põllu identiteedil. Jäljeotsija testib eraldi ka õpitavat mittelineaarset hooaja/taime kulumise kõverat, "
                f"mis võib kajastada vananemise, lehehaiguste ja kahjurite kumulatiivset hilishooaja mõju. "
                f"Tõestatud normaliseeritud bioloogiline koormus võib baasi korrigeerida, "
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

                # Õppimise ja prognoosi hooajafunktsioon peab olema identne.
                wx.update(_season_curve_features(target_day, SEASON_START))

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

                # 7/10/14 päeva ilmamälu: erinevalt korjevahemiku tunnustest võib see
                # ulatuda üle eelmise korje kuupäeva. Puuduliku akna korral jääb kandidaat neutraalseks.
                for n in (7, 10, 14):
                    lb = []
                    ok = True
                    for i in range(n - 1, -1, -1):
                        day_key = target_day - timedelta(days=i)
                        src = all_weather_by_day.get(day_key.isoformat())
                        clean = {}
                        if not src:
                            ok = False
                            break
                        for feature in weather_feature_names:
                            try:
                                value = float(src.get(feature)) if src.get(feature) is not None else None
                            except (TypeError, ValueError):
                                value = None
                            if value is None:
                                ok = False
                                break
                            clean[feature] = value
                        if not ok:
                            break
                        lb.append(clean)
                    if ok and len(lb) == n:
                        wx[f"ÖöT {n}p"] = float(np.mean([r["temp_night_avg_c"] for r in lb]))
                        wx[f"PäevT {n}p"] = float(np.mean([r["temp_day_avg_c"] for r in lb]))
                        wx[f"Rad {n}p"] = float(np.sum([r["radiation_mj_m2"] for r in lb]))
                        wx[f"Sade {n}p"] = float(np.sum([r["precipitation_mm"] for r in lb]))
                        wx[f"ET0 {n}p"] = float(np.sum([r["et0_mm"] for r in lb]))
                        wx[f"Niiskus {n}p"] = float(np.mean([r["humidity_avg_pct"] for r in lb]))
                    else:
                        for prefix in ("ÖöT", "PäevT", "Rad", "Sade", "ET0", "Niiskus"):
                            wx[f"{prefix} {n}p"] = None
                return wx, estimated_days

            def _champion_feature_values(state, wx, cols=None):
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
                        or k == "Hooajapäev²"
                        or k.startswith("Hooaeg ")
                        or re.match(r"^(ÖöT|PäevT|Rad|Sade|ET0|Niiskus) (7|10|14)p$", str(k))
                    ):
                        values[k] = v
                use_cols = champion_cols if cols is None else list(cols)
                return [values.get(c) for c in use_cols]

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
                    # Sama põld võib 9 päeva aknas tulla uuesti korjesse. Kui biokoormuse
                    # champion vajaks prognoositud eelkorjet, läheme päriselt puhtale
                    # weather-first baasmudelile, mitte championile missing-väärtustega.
                    abc_model_used = full_abc_base_model
                    abc_extra_values = []
                    abc_extra_names = []
                    abc_mode = "weather-first baasmudel · prognoositud eelkorje"
                else:
                    abc_model_used = full_abc_model
                    abc_extra_values = _champion_feature_values(state, wx)
                    abc_extra_names = champion_cols
                    abc_mode = champion_name

                abc_pred = _predict_full_abc_growth(
                    abc_model_used, field_no, base_values, abc_extra_values,
                )
                abc_explain = _abc_growth_explain(
                    abc_model_used, field_no, base_values, abc_extra_values, abc_extra_names,
                )

                if xl_champion_uses_memory and state.get("source") != "tegelik":
                    xl_pred = _predict_full_generic(full_xl_base_model, field_no, base_values, [])
                else:
                    xl_pred = _predict_full_generic(
                        full_xl_model, field_no, base_values,
                        _xl_champion_feature_values(state, wx),
                    )

                if cb_champion_uses_memory and state.get("source") != "tegelik":
                    cb_log_pred = _predict_full_generic(
                        full_cb_base_model, field_no, base_values, [], floor_zero=False,
                    )
                else:
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

            # Variprognoosi jaoks säilitame seisundi täpselt enne homse rekursiivse
            # prognoosi rakendamist. Nii saab parimat kinnitamata challengerit võrrelda
            # sama ilma, intervalli ja sama XL-komponendi peal ametliku prognoosiga.
            _shadow_state_for_tomorrow = {int(k): dict(v) for k, v in field_state.items()}

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
                    if str(result.get("abc_mode") or "").startswith("weather-first baasmudel"):
                        source_label += " · ABC weather-first baasmudel"
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

            # Parima kinnitamata kihilise kandidaadi VARIPROGNOOS homseks.
            # See ei muuda ametlikku prognoosi ega championit ja ei käivita uut
            # walk-forward uuringut; kasutame juba salvestatud parima challengeri tunnuseid.
            try:
                _challenger = _load_json_setting("layered_best_challenger_json", {})

                # Ühekordne tagasiühilduv migratsioon app-117 salvestatud laiast uurimisest.
                # Nii saab esimene app-118 operatiivne ring näidata variprognoosi ilma
                # uut kallist laia walk-forward uuringut käivitamata.
                if not isinstance(_challenger, dict) or not (_challenger.get("cols") or []):
                    _old_research_state = _load_json_setting("layered_research_state_json", {})
                    _old_results = list((_old_research_state or {}).get("results") or []) if isinstance(_old_research_state, dict) else []
                    _old_results = [
                        _r for _r in _old_results
                        if isinstance(_r, dict) and (_r.get("cols") or [])
                    ]
                    if _old_results:
                        _newest_old = max(_old_results, key=lambda _r: str(_r.get("tested_at") or ""))
                        _latest_hash = str(_newest_old.get("data_hash") or "")
                        _same_state = [
                            _r for _r in _old_results
                            if (not _latest_hash or str(_r.get("data_hash") or "") == _latest_hash)
                        ]

                        def _saved_challenger_score(_r):
                            try:
                                _imp = float(_r.get("improvement", -999.0))
                            except Exception:
                                _imp = -999.0
                            try:
                                _win = float(_r.get("win_share", 0.0))
                            except Exception:
                                _win = 0.0
                            try:
                                _worst = float(_r.get("worst_half", -999.0))
                            except Exception:
                                _worst = -999.0
                            try:
                                _trend = float(_r.get("trend_accuracy")) if _r.get("trend_accuracy") is not None else 50.0
                            except Exception:
                                _trend = 50.0
                            try:
                                _regime = float(_r.get("regime_improvement")) if _r.get("regime_improvement") is not None else 0.0
                            except Exception:
                                _regime = 0.0
                            try:
                                _over = float(_r.get("overreaction", 0.0) or 0.0)
                            except Exception:
                                _over = 0.0
                            _stability = 0.10 * max(-1.5, min(1.5, _worst))
                            return _imp + 0.004 * (_win - 50.0) + _stability + 0.001 * (_trend - 50.0) + 0.12 * _regime - 0.15 * _over

                        _old_best = max(_same_state, key=_saved_challenger_score)
                        _challenger = {
                            "saved_at": str(_old_best.get("tested_at") or datetime.now(ZoneInfo("Europe/Tallinn")).isoformat()),
                            "harvest_day": str(_last_research_harvest_day or "") or None,
                            "data_hash": _old_best.get("data_hash"),
                            "name": "Avatud kihiline otsing",
                            "parts": list(_old_best.get("parts") or []),
                            "cols": list(_old_best.get("cols") or []),
                            "base_mae": _old_best.get("base_mae"),
                            "candidate_mae": _old_best.get("candidate_mae"),
                            "improvement": _old_best.get("improvement"),
                            "win_share": _old_best.get("win_share"),
                            "worst_half": _old_best.get("worst_half"),
                            "test_rows": _old_best.get("test_rows"),
                            "trend_accuracy": _old_best.get("trend_accuracy"),
                            "regime_improvement": _old_best.get("regime_improvement"),
                            "overreaction": _old_best.get("overreaction"),
                            "confirmed": False,
                            "confirmation_stats": None,
                            "migrated_from_v117": True,
                        }
                        _save_json_setting("layered_best_challenger_json", _challenger)

                _challenger_cols = [
                    str(c) for c in ((_challenger or {}).get("cols") or [])
                    if str(c) in model_df.columns
                ]
                _challenger_is_distinct = bool(_challenger_cols) and _challenger_cols != list(champion_cols)
                if forecast_days and _challenger_is_distinct:
                    _challenger_arrays = [
                        pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float)
                        for c in _challenger_cols
                    ]
                    _challenger_model = _fit_full_abc_growth(_challenger_arrays)
                    _challenger_uses_bio = any(c in biological_load_feature_names for c in _challenger_cols)
                    _tomorrow_day, _tomorrow_official_rows = forecast_days[0]
                    _challenger_rows = []
                    _official_rows_by_field = {int(r.get("Põld")): r for r in _tomorrow_official_rows if r.get("Põld") is not None}
                    for _f, _orow in _official_rows_by_field.items():
                        _state = _shadow_state_for_tomorrow.get(int(_f))
                        if not _state or _orow.get("XL") is None:
                            continue
                        _wx, _est = _weather_window_for_prediction(_state["date"], _tomorrow_day)
                        if _wx is None:
                            continue
                        _base_values = [_wx[c] for c in base_cont_cols]
                        if _challenger_uses_bio and _state.get("source") != "tegelik":
                            _abc_shadow = _predict_full_abc_growth(full_abc_base_model, int(_f), _base_values, [])
                            _shadow_mode = "baas prognoositud eelkorje tõttu"
                        else:
                            _extra_values = _champion_feature_values(_state, _wx, cols=_challenger_cols)
                            _abc_shadow = _predict_full_abc_growth(
                                _challenger_model, int(_f), _base_values, _extra_values
                            )
                            _shadow_mode = "challenger"
                        _challenger_rows.append({
                            "field_no": int(_f),
                            "abc": float(_abc_shadow),
                            "xl": float(_orow.get("XL")),
                            "total": float(_abc_shadow) + float(_orow.get("XL")),
                            "mode": _shadow_mode,
                        })

                    if _challenger_rows and len(_challenger_rows) == len(_official_rows_by_field):
                        _official_total = sum(float(r.get("Kokku")) for r in _tomorrow_official_rows if r.get("Kokku") is not None)
                        _shadow_total = sum(float(r["total"]) for r in _challenger_rows)
                        _save_json_setting("shadow_challenger_forecast_json", {
                            "generated_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                            "target_date": _tomorrow_day.isoformat(),
                            "official_model": champion_name,
                            "official_total": float(_official_total),
                            "challenger_name": str((_challenger or {}).get("name") or "Avatud kihiline otsing"),
                            "challenger_parts": list((_challenger or {}).get("parts") or []),
                            "challenger_cols": list(_challenger_cols),
                            "challenger_total": float(_shadow_total),
                            "difference": float(_shadow_total - _official_total),
                            "confirmed": bool((_challenger or {}).get("confirmed")),
                            "candidate_mae": (_challenger or {}).get("candidate_mae"),
                            "base_mae": (_challenger or {}).get("base_mae"),
                            "improvement": (_challenger or {}).get("improvement"),
                            "win_share": (_challenger or {}).get("win_share"),
                            "worst_half": (_challenger or {}).get("worst_half"),
                            "rows": _challenger_rows,
                        })
                elif not _challenger_is_distinct:
                    _save_json_setting("shadow_challenger_forecast_json", {})
            except Exception as _shadow_exc:
                db.set_app_setting("shadow_challenger_last_error", str(_shadow_exc))

            # Salvestame 9 päeva prognoosid eraldi ajalukku. Sama päeva rerun uuendab
            # olemasolevat snapshot'i; järgmine päev loob uue lead-time snapshot'i.
            # Mudeliversioon tähistab champion-valiku raamistikku, mitte tänase võitja nime.
            # Nii uuendab sama päeva rerun sama operatiivset snapshot'i ka siis, kui champion
            # uue korje järel päeva jooksul muutub. Võitja nimi salvestub basis-väljale.
            MODEL_VERSION = "v6.5-v18-complete-daily-research-observation-snapshot"
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

            # Säilita koos eduka ametliku snapshotiga ka A+B+C selgitus ja selle
            # ilmastiku sisendid. Prognoos-leht saab neid hiljem näidata puhta
            # lugemisvaatena, ilma uut ~10 s operatiivringi käivitamata.
            if forecast_store_ok:
                # Selgitus peab käima TÄPSELT sama ametliku snapshot-batch'i kohta.
                # Tänase selgitust ei tohi ära jätta ainult seepärast, et korje on juba alanud:
                # lead=0 snapshot ise on lukus, aga kui praegune pre-harvest arvutus vastab
                # sellele snapshotile, saame selgituse turvaliselt tagantjärele salvestada.
                _explain_days_to_save = [(TODAY, today_forecast_rows)] + list(forecast_days)
                try:
                    _official_rows_for_explain = (
                        db.get_yield_forecasts(limit=5000)
                        if db.yield_forecasts_available() else []
                    )
                except db.DatabaseError:
                    _official_rows_for_explain = []

                _wx_keep = [
                    "Tmin kesk", "Tmin min", "Tmax kesk", "Tmax max",
                    "Soojad ööd 16+", "Intervall p", "Päevapikkus", "Päevapikkus Δ7p",
                    "Radiatsioon Σ", "Radiatsioon/p", "Sademed Σ", "Niiskus kesk", "ET0 Σ",
                ]
                for _exp_day, _exp_day_rows in _explain_days_to_save:
                    # Vali andmebaasist sama päeva viimane terviklik salvestusbatch.
                    # Kui tänane snapshot on korje alustamise tõttu lukus ja praegune
                    # arvutus enam sellega ei ühti, EI kirjuta me eksitavat selgitust.
                    _official_batch = _latest_snapshot_batch_rows(
                        _official_rows_for_explain, _exp_day
                    )
                    _official_by_field = {
                        int(_r.get("field_no")): _r
                        for _r in _official_batch
                        if _r.get("field_no") is not None
                    }
                    _calc_by_field = {
                        int(_r.get("Põld")): _r
                        for _r in _exp_day_rows
                        if _r.get("Põld") is not None
                    }
                    if (
                        not _official_by_field
                        or set(_official_by_field.keys()) != set(_calc_by_field.keys())
                    ):
                        continue
                    _same_snapshot = True
                    for _f, _crow in _calc_by_field.items():
                        try:
                            if abs(
                                float(_official_by_field[_f].get("abc_forecast"))
                                - float(_crow.get("A+B+C"))
                            ) > 0.06:
                                _same_snapshot = False
                                break
                        except (TypeError, ValueError, KeyError):
                            _same_snapshot = False
                            break
                    if not _same_snapshot:
                        continue

                    _exp_rows = []
                    for _exp_row in _exp_day_rows:
                        _ex = _exp_row.get("_ABC_selgitus") or {}
                        _wx = _exp_row.get("_WX") or {}
                        if not _ex or _exp_row.get("Põld") is None:
                            continue
                        _effects = {}
                        for _k, _v in dict(_ex.get("effects") or {}).items():
                            try:
                                _effects[str(_k)] = float(_v)
                            except (TypeError, ValueError):
                                pass
                        _wx_small = {}
                        for _k in _wx_keep:
                            _v = _wx.get(_k)
                            if _v is None:
                                _wx_small[_k] = None
                                continue
                            try:
                                _wx_small[_k] = float(_v)
                            except (TypeError, ValueError):
                                _wx_small[_k] = None
                        try:
                            _baseline = float(_ex.get("baseline"))
                            _prediction = float(_ex.get("prediction"))
                        except (TypeError, ValueError):
                            continue
                        _exp_rows.append({
                            "field_no": int(_exp_row.get("Põld")),
                            "abc_forecast": float(_exp_row.get("A+B+C")),
                            "baseline": _baseline,
                            "prediction": _prediction,
                            "effects": _effects,
                            "wx": _wx_small,
                        })
                    if _exp_rows:
                        _save_json_setting(
                            f"forecast_explain_{_exp_day.isoformat()}",
                            {
                                "forecast_date": TODAY.isoformat(),
                                "target_date": _exp_day.isoformat(),
                                "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                                "rows": _exp_rows,
                            },
                        )

            # Prognoosi liikumine: eelmine sama korjepäeva salvestatud snapshot vs praegune prognoos.
            # Kasutame ainult sama põllukomplekti täielikke snapshot'e.
            forecast_history_rows = []
            try:
                if db.yield_forecasts_available():
                    forecast_history_rows = db.get_yield_forecasts(limit=5000)
            except db.DatabaseError:
                forecast_history_rows = []

            def _forecast_adjustment(target_day_value, field_numbers, current_total):
                """Muutus sama korjepäeva esimese 9 päeva ette tehtud prognoosi suhtes."""
                expected = {int(f) for f in field_numbers}
                if not expected or current_total is None:
                    return None, None, None

                # Ära filtreeri põlde enne batch'i moodustamist: muidu võiks vana
                # lisapõld kaduda filtris ja eri arvutusringidest tekiks näiliselt "täielik" snapshot.
                by_date = _snapshot_batch_map(forecast_history_rows, target_day_value)

                base_rows = []
                for fdate, rows_for_date in by_date.items():
                    if set(rows_for_date.keys()) != expected:
                        continue
                    try:
                        forecast_day = date.fromisoformat(fdate)
                        total = sum(float(rows_for_date[f]["total_forecast"]) for f in expected)
                    except (TypeError, ValueError, KeyError):
                        continue
                    if (target_day_value - forecast_day).days == 9:
                        base_rows.append((fdate, total))

                if not base_rows:
                    return None, None, None

                base_rows.sort(key=lambda x: x[0])
                first_date, first_total = base_rows[0]
                if first_total <= 0:
                    return None, first_total, first_date

                pct = (float(current_total) / first_total - 1.0) * 100.0
                return pct, first_total, first_date


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

                    # Üks snapshot = üks generated_at batch. Vanu ja uusi põllukomplekte
                    # ei tohi põllu kaupa üheks päevaks kokku liita.
                    by_fdate = _snapshot_batch_map(
                        forecast_history_rows, d, max_forecast_date=d
                    )

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
                        f'<span style="font-size:1.45rem;font-weight:700;">{_weekday_letter(target_day)} {_short_date(target_day)}&emsp;&emsp;{total_text}</span>'
                        f'<span style="margin-left:10px;font-size:0.92rem;">{lead} p ette</span>'
                        '</div>'
                    )
                    st.markdown(header_html, unsafe_allow_html=True)
                else:
                    confidence = "kõrgem kindlus" if lead <= 3 else "keskmine kindlus"
                    st.markdown(f"### {_weekday_letter(target_day)} {_short_date(target_day)}  {total_text}")
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
                            f"päev {float(wxr.get('Päevapikkus')):.1f} h ({float(wxr.get('Päevapikkus Δ7p')):+.1f} h/7p) · "
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
                f"1–3 päeva on operatiivne vaade, 4–5 päeva planeerimisvaade ja 6–9 päeva on kollasega märgitud kaugem prognoos. "
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
            # Jäljeotsija detailne uurimisraport kuulub Mootori tähelepanekutesse.
            # Prognoos näitab prognoosi, täpsust ja lühikest mudeliinfot.
            # -------------------------------------------------------------------------
            if page == "Mootori tähelepanekud" and valid_pred.any():
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

                    operational_trace = trace_df[trace_df["Jälg"].isin(screening_candidate_groups.keys())].copy()
                    operational_trace = operational_trace.sort_values("Paranemine", ascending=False)
                    memory_trace = trace_df[trace_df["Jälg"].isin(memory_diagnostic_groups.keys())].copy()
                    memory_trace = memory_trace.sort_values("Paranemine", ascending=False)

                    with st.expander("Mida Jäljeotsija täna kaalus"):
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
                            f"**{champion_name}** on tänane kinnitatud champion. Kandidaat valiti ainult varasemal "
                            f"avastusperioodil; hilisem kinnitusaeg jäi valiku ajal puutumata. Kinnituses on MAE "
                            f"{champion_stats['Katse MAE']:.2f} kasti võrreldes baasi {champion_stats['Baas MAE']:.2f}-ga "
                            f"(paranemine {champion_stats['Paranemine']:.2f}) ning kandidaat võitis "
                            f"{champion_stats['Võidab ridu %']:.0f}% kinnitusridadest."
                        )
                    else:
                        st.info(
                            f"**{champion_name}** jääb championiks. Ükski lisajälg ei läbinud eraldi avastus- ja kinnitusetappi. "
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
                        st.markdown("**Tugevaim üksiktunnuse sõelasignaal**")
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

                    rejected = trace_df[trace_df["Paranemine"] <= 0].sort_values("Paranemine", ascending=False)
                    if not rejected.empty:
                        with st.expander(f"Ootel üksiktunnused ({len(rejected)})"):
                            st.caption(
                                "Ootel ei tähenda keelatud. Need tunnused võivad kihilistes kombinatsioonides "
                                "endiselt osaleda ja uute ilma-/hooajatingimustega uuesti tugevneda."
                            )
                            for _, r in rejected.iterrows():
                                st.write(
                                    f"• **{r['Jälg']}** — üksikult ei parandanud praeguses testis baasi "
                                    f"(MAE muutus {r['Paranemine']:+.2f} kasti), kuid jääb kombinatsioonidele avatuks."
                                )

                    _response_fade = _response_fade_diagnostic() if "_response_fade_diagnostic" in locals() else None
                    if _response_fade and _response_fade["flag"]:
                        st.warning(
                            "🌿 **Taime reaktsioonivõime jälg:** viimased BIO kasvutingimused on olnud "
                            "vähemalt varasema taseme juures, kuid saagivastus on nõrgenenud "
                            f"({_response_fade['yield_change']:+.1f} kasti/põld). "
                            "See on hooaja/taime kulumise hoiatus, mitte automaatne prognoosikorrektsioon."
                        )

                    st.markdown("**Kihilised BIO-mudelid**")
                    if "layered_trace_df" in locals() and not layered_trace_df.empty:
                        _best_layered = layered_trace_df.iloc[0]
                        st.write(
                            f"Parim kihiline kombinatsioon avastusperioodil: **{_best_layered['Mudel']}** · "
                            f"MAE {_best_layered['Katse MAE']:.2f} vs baas {_best_layered['Baas MAE']:.2f} "
                            f"(Δ {_best_layered['Paranemine']:+.2f}); "
                            f"võitis {_best_layered['Võidab ridu %']:.0f}% ridadest."
                        )
                        st.caption("Koosseis: " + str(_best_layered["Koosseis"]))
                        st.caption(
                            "Valikul vaadatakse lisaks MAE-le ka trendi tabamist, käitumist suurematel "
                            "ilmamuutustel ja seda, et prognoos ei hüppaks päevast päeva põhjendamatult."
                        )
                        with st.expander("Näita kõiki kihilisi BIO-mudeleid"):
                            st.dataframe(
                                layered_trace_df.style.format({
                                    "Baas MAE": "{:.2f}",
                                    "Katse MAE": "{:.2f}",
                                    "Paranemine": "{:+.2f}",
                                    "Võidab ridu %": "{:.0f}%",
                                    "Halvim pool": "{:+.2f}",
                                    "Trend %": "{:.0f}%",
                                    "Režiim Δ": "{:+.2f}",
                                    "Ülereageerimine": "{:.2f}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.caption(
                            "Praegusest avastusperioodist ei saanud veel moodustada vähemalt kahe "
                            "usutava lisakihiga kihilist kandidaati."
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
                            "Operatiivne kandidaat valitakse varasemal avastusperioodil ja peab seejärel läbima hilisema puutumata kinnituse. "
                            "Toored saagimälu tunnused on raportis diagnostilised; normaliseeritud bioloogiline koormus võib championiks saada ainult kinnitatud tõendi korral. "
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

        # Mootori diagnostiline snapshot muutub ainult päris uurimisringi järel.
        # Ilma- ja operatiivprognoosi refresh ei tohi seda "uueks õppimiseks" muuta.

        # ---------------------------------------------------------------------
        # MOOTORI TÄHELEPANEKUTE PÜSIV SNAPSHOT
        # Lehe avamine ei tohi nõuda uut walk-forward ringi. Salvestame pärast
        # päris Jäljeotsija või laia uurimisringi kogu diagnostilise seisu.
        # ---------------------------------------------------------------------
        if _run_light_research or _layered_run_now:
            try:
                def _records_for_snapshot(df):
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        return []
                    clean = df.copy().where(pd.notna(df), None)
                    return clean.to_dict(orient="records")

                _obs_journal = {}
                if last_complete_harvest is not None:
                    _jday = last_complete_harvest
                    _jrows = harvest_by_day.get(_jday.isoformat(), [])
                    _jexpected = {
                        int(r.get("field_no")) for r in _jrows if r.get("field_no") is not None
                    }
                    if _jrows:
                        try:
                            _obs_journal["actual_total"] = float(sum(float(r.get("total")) for r in _jrows))
                            _obs_journal["actual_abc"] = float(sum(
                                float(r.get("a", 0) or 0) + float(r.get("b", 0) or 0) + float(r.get("c", 0) or 0)
                                for r in _jrows
                            ))
                        except Exception:
                            pass

                    # Korjepäeval olemas olnud viimane täielik snapshot.
                    try:
                        _by_fd = _snapshot_batch_map(
                            forecast_history_rows if "forecast_history_rows" in locals() else [],
                            _jday,
                            max_forecast_date=_jday,
                        )
                        _complete = []
                        for _fd, _rows in _by_fd.items():
                            if set(_rows.keys()) != _jexpected:
                                continue
                            _total = sum(float(_rows[f]["total_forecast"]) for f in _jexpected)
                            _abc = sum(float(_rows[f]["abc_forecast"]) for f in _jexpected)
                            _basis = [str(_rows[f].get("basis") or "") for f in _jexpected]
                            _complete.append((_fd, _total, _abc, _basis))
                        if _complete:
                            _complete.sort(key=lambda x: x[0])
                            _fd, _ftotal, _fabc, _fbasis = _complete[-1]
                            _obs_journal["forecast_date"] = _fd
                            _obs_journal["forecast_total"] = float(_ftotal)
                            _obs_journal["forecast_abc"] = float(_fabc)
                            for _basis in _fbasis:
                                _m = re.search(r"(?:^|;\s*)champion=([^;]+)", _basis)
                                if _m:
                                    _obs_journal["forecast_champion"] = _m.group(1).strip()
                                    break
                    except Exception:
                        pass

                _obs_snapshot = {
                    "saved_at": datetime.now(ZoneInfo("Europe/Tallinn")).isoformat(),
                    "last_complete_harvest": last_complete_harvest.isoformat() if last_complete_harvest else None,
                    "training_rows": len(training_rows) if "training_rows" in locals() else 0,
                    "champion_name": str(champion_name) if "champion_name" in locals() else "Baasmudel",
                    "champion_mae": float(champion_mae) if "champion_mae" in locals() and champion_mae is not None else None,
                    "champion_stats": champion_stats if "champion_stats" in locals() and isinstance(champion_stats, dict) else None,
                    "trace": _records_for_snapshot(trace_df if "trace_df" in locals() else None),
                    "layered": _records_for_snapshot(layered_trace_df if "layered_trace_df" in locals() else None),
                    "autonomous": _records_for_snapshot(autonomous_trace_df if "autonomous_trace_df" in locals() else None),
                    "autonomous_category_counts": autonomous_category_counts if "autonomous_category_counts" in locals() and isinstance(autonomous_category_counts, dict) else {},
                    "autonomous_candidate_count": int(autonomous_candidate_count) if "autonomous_candidate_count" in locals() else 0,
                    "field_state_watch": field_state_watch if "field_state_watch" in locals() else [],
                    "weather_groups": list(weather_candidate_groups.keys()) if "weather_candidate_groups" in locals() else [],
                    "bio_groups": list(biological_load_candidate_groups.keys()) if "biological_load_candidate_groups" in locals() else [],
                    "memory_groups": list(memory_diagnostic_groups.keys()) if "memory_diagnostic_groups" in locals() else [],
                    "screening_groups": list(screening_candidate_groups.keys()) if "screening_candidate_groups" in locals() else [],
                    "journal": _obs_journal,
                }
                db.set_app_setting(
                    "motor_observation_snapshot_json",
                    json.dumps(_obs_snapshot, ensure_ascii=False),
                )
            except Exception as _obs_save_error:
                db.set_app_setting("motor_observation_snapshot_error", str(_obs_save_error))

        _cycle_finished_at = datetime.now(ZoneInfo("Europe/Tallinn"))
        _cycle_total_duration_s = max(0.0, time.perf_counter() - _light_cycle_t0)

        # Iga edukas operatiivne prognoosiring lõpetab forecast-dirty seisundi,
        # kuid EI puuduta uurimismootori dirty-lippu, kui Jäljeotsija jäi CPU tõttu ootele.
        db.set_app_setting("forecast_dirty", "0")
        db.set_app_setting("forecast_last_checked_at", _cycle_finished_at.isoformat())
        db.set_app_setting("forecast_cycle_last_duration_s", f"{_cycle_total_duration_s:.1f}")
        db.set_app_setting("forecast_cycle_last_reason", str(_light_cycle_reason))

        if _run_light_research:
            _light_duration_s = float(
                locals().get(
                    "_light_research_elapsed_measured_s",
                    max(0.0, _cycle_total_duration_s - float(_layered_elapsed_s_this_cycle or 0.0)),
                )
            )
            db.set_app_setting("model_dirty", "0")
            db.set_app_setting("model_last_checked_complete_day_count", str(len(complete_harvest_days)))
            db.set_app_setting("model_last_checked_at", _cycle_finished_at.isoformat())
            db.set_app_setting("cpu_light_skip_day", "")

            db.set_app_setting("light_cycle_last_duration_s", f"{_light_duration_s:.1f}")
            db.set_app_setting("light_cycle_last_total_duration_s", f"{_cycle_total_duration_s:.1f}")
            db.set_app_setting("light_cycle_last_at", _cycle_finished_at.isoformat())
            db.set_app_setting("light_cycle_last_reason", str(_light_cycle_reason))
            db.set_app_setting(
                "light_cycle_last_abc_candidates",
                str(len(candidate_predictions)) if "candidate_predictions" in locals() else "0",
            )
            db.set_app_setting(
                "light_cycle_last_cb_candidates",
                str(len(cb_candidate_predictions)) if "cb_candidate_predictions" in locals() else "0",
            )
            db.set_app_setting(
                "light_cycle_last_training_rows",
                str(len(training_df)) if "training_df" in locals() else "0",
            )
            _cache_mode_txt = (
                "täielik cache"
                if "_light_exact_cache" in locals() and _light_exact_cache
                else "ainult uus walk-forward samm"
                if "_light_incremental_cache" in locals() and _light_incremental_cache
                else "täisarvutus"
            )
            db.set_app_setting("light_cycle_last_cache_mode", _cache_mode_txt)

            _cpu_record_research("light", _light_duration_s)
            db.set_app_setting(
                "cpu_light_last_complete_harvest_day",
                last_complete_harvest.isoformat() if last_complete_harvest else TODAY.isoformat(),
            )
            if _light_duration_s > CPU_LIGHT_TARGET_S:
                db.set_app_setting(
                    "cpu_light_last_warning",
                    f"Jäljeotsija kestis {_light_duration_s:.1f}s, üle sihtlae {CPU_LIGHT_TARGET_S:.0f}s; "
                    "lai otsing saab samal päeval automaatselt vähem või 0 sekundit.",
                )

    if page != "Prognoos":
        _forecast_page_placeholder.empty()

    # Täna-leht loeb snapshotid enne allpool jooksvat peidetud operatiivringi.
    # Kui ilm/korje muutis prognoosi, tee pärast edukat salvestust üks odav rerun,
    # et kasutaja ei jääks kuni järgmise klõpsuni vana snapshot'i vaatama.
    if page == "Täna" and _forecast_refresh_due:
        st.rerun()

# Prognoos-leht on tavavaates puhas LUGEMISVAADE: näita täpselt samu
# yield_forecasts snapshotte, mida kasutab Avalehe „Järgmised päevad“.
# Nii ei saa pelk Prognoos-lehe avamine ametlikku prognoosi muuta ega ~10 s
# operatiivset arvutusringi käivitada. Kui ülal jooksis päris operatiivring,
# renderdas see juba detailse Prognoos-vaate värskete arvutustega.
if page == "Prognoos" and not _run_operational_cycle:
    st.subheader("Prognoos")
    st.caption(
        "Viimane salvestatud ametlik prognoos. Selle lehe avamine ei arvuta mudelit uuesti; "
        "uus prognoos tekib ainult siis, kui muutub päris sisend (ilm, uus korje/champion või uus päev)."
    )

    try:
        _ro_saved = db.get_yield_forecasts(limit=5000) if db.yield_forecasts_available() else []
    except db.DatabaseError:
        _ro_saved = []

    def _ro_latest_snapshot_rows(target_day_value):
        return _latest_snapshot_batch_rows(_ro_saved, target_day_value)

    _ro_shown = 0
    for _lead in range(0, 10):
        _target_day = TODAY + timedelta(days=_lead)
        _rows = _ro_latest_snapshot_rows(_target_day)
        if not _rows:
            continue
        _rows = sorted(_rows, key=lambda _r: int(_r.get("field_no") or 999))
        try:
            _total_day = sum(float(_r.get("total_forecast")) for _r in _rows)
        except (TypeError, ValueError):
            _total_day = None
        _total_text = f"{_fmt(_total_day)} kasti" if _total_day is not None else "prognoos puudulik"

        if _lead >= 6:
            _header_html = (
                '<div style="background:#fff3cd;border:1px solid #ffe69c;border-radius:10px;'
                'padding:8px 12px;margin:12px 0 6px 0;">'
                f'<span style="font-size:1.45rem;font-weight:700;">{_weekday_letter(_target_day)} {_short_date(_target_day)}&emsp;&emsp;{_total_text}</span>'
                f'<span style="margin-left:10px;font-size:0.92rem;">{_lead} p ette</span>'
                '</div>'
            )
            st.markdown(_header_html, unsafe_allow_html=True)
        else:
            st.markdown(f"### {_weekday_letter(_target_day)} {_short_date(_target_day)}  {_total_text}")
            if _lead == 0:
                st.caption("täna · salvestatud ametlik korje-eelne prognoos")
            else:
                _confidence = "kõrgem kindlus" if _lead <= 3 else "keskmine kindlus"
                st.caption(f"{_lead} p ette · {_confidence}")

        _table_rows = []
        for _r in _rows:
            try:
                _cb = None if _r.get("cb_forecast") is None else float(_r.get("cb_forecast"))
            except (TypeError, ValueError):
                _cb = None
            _basis = str(_r.get("basis") or "")
            # Hoia tabel mobiilis loetav: näita Alus veerus põhiosa, championid on
            # snapshotis alles ning neid ei ole vaja igal real korrata.
            _basis_short = _basis.split("; champion=", 1)[0].strip()
            _table_rows.append({
                "Intervall": _r.get("interval_days"),
                "Põld": int(_r.get("field_no")),
                "A+B+C": _r.get("abc_forecast"),
                "C/B": _cb,
                "XL": _r.get("xl_forecast"),
                "Kokku": _r.get("total_forecast"),
                "Alus": _basis_short,
            })
        _ro_df = pd.DataFrame(_table_rows)
        st.dataframe(
            _ro_df.style.format({
                "A+B+C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "C/B": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                "XL": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Kokku": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Intervall": lambda v: "—" if pd.isna(v) else f"{int(v)} p",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Sama snapshoti salvestatud A+B+C selgitus. Seda EI arvutata lehe
        # avamisel uuesti; näitame ainult viimase päris operatiivringi seletust.
        _ro_exp = _load_json_setting(f"forecast_explain_{_target_day.isoformat()}", {})
        _ro_exp_rows = list((_ro_exp or {}).get("rows") or []) if isinstance(_ro_exp, dict) else []
        _ro_exp_by_field = {}
        for _er in _ro_exp_rows:
            try:
                _ro_exp_by_field[int(_er.get("field_no"))] = _er
            except (TypeError, ValueError):
                pass

        _snapshot_fields = {int(_r.get("field_no")) for _r in _rows if _r.get("field_no") is not None}
        _explain_matches = bool(_snapshot_fields) and set(_ro_exp_by_field.keys()) == _snapshot_fields
        if _explain_matches:
            # Kaitse stale selgituse vastu: A+B+C peab vastama samale salvestatud snapshotile.
            for _r in _rows:
                try:
                    _f = int(_r.get("field_no"))
                    _abc_saved = float(_r.get("abc_forecast"))
                    _abc_exp = float(_ro_exp_by_field[_f].get("abc_forecast"))
                    if abs(_abc_saved - _abc_exp) > 0.06:
                        _explain_matches = False
                        break
                except (TypeError, ValueError, KeyError):
                    _explain_matches = False
                    break

        if _explain_matches:
            _explain_rows = []
            _input_lines = []
            for _f in sorted(_snapshot_fields):
                _er = _ro_exp_by_field[_f]
                _effects = dict(_er.get("effects") or {})
                _exrow = {
                    "Põld": _f,
                    "Mudelibaas": _er.get("baseline"),
                    "Temperatuur": _effects.get("Temperatuur", 0.0),
                    "Radiatsioon": _effects.get("Radiatsioon", 0.0),
                    "Sademed": _effects.get("Sademed", 0.0),
                    "Niiskus": _effects.get("Niiskus", 0.0),
                    "ET0": _effects.get("ET0", 0.0),
                    "Tuul": _effects.get("Tuul", 0.0),
                    "Intervall": _effects.get("Intervall", 0.0),
                    "Hooaeg": _effects.get("Hooaeg", 0.0),
                    "Põlluefekt": _effects.get("Põlluefekt", 0.0),
                    "Biokoormus": _effects.get("Biokoormus", 0.0),
                    "A+B+C": _er.get("prediction"),
                }
                try:
                    if abs(float(_effects.get("Muu", 0.0) or 0.0)) >= 0.01:
                        _exrow["Muu"] = _effects.get("Muu", 0.0)
                except (TypeError, ValueError):
                    pass
                _explain_rows.append(_exrow)

                _wxr = dict(_er.get("wx") or {})
                try:
                    _input_lines.append(
                        f"põld {_f}: Tmin {float(_wxr.get('Tmin kesk')):.1f} °C (min {float(_wxr.get('Tmin min')):.1f}) · "
                        f"Tmax {float(_wxr.get('Tmax kesk')):.1f} °C (max {float(_wxr.get('Tmax max')):.1f}) · "
                        f"sooje öid ≥16 °C {int(float(_wxr.get('Soojad ööd 16+')))}/{int(float(_wxr.get('Intervall p')))} · "
                        f"päev {float(_wxr.get('Päevapikkus')):.1f} h ({float(_wxr.get('Päevapikkus Δ7p')):+.1f} h/7p) · "
                        f"rad {float(_wxr.get('Radiatsioon Σ')):.1f} MJ/m² ({float(_wxr.get('Radiatsioon/p')):.1f}/p) · "
                        f"sade {float(_wxr.get('Sademed Σ')):.1f} mm · RH {float(_wxr.get('Niiskus kesk')):.0f}% · "
                        f"ET0 {float(_wxr.get('ET0 Σ')):.1f} mm · intervall {int(float(_wxr.get('Intervall p')))} p"
                    )
                except (TypeError, ValueError):
                    pass

            if _explain_rows:
                st.caption("A+B+C selgitus · +/− = teguri panus kastides võrreldes mudeli neutraalse treeningtasemega")
                _explain_df = pd.DataFrame(_explain_rows)
                _effect_cols = [c for c in _explain_df.columns if c not in {"Põld", "Mudelibaas", "A+B+C"}]
                _fmt_exp = {
                    "Mudelibaas": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                    "A+B+C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                }
                for _c in _effect_cols:
                    _fmt_exp[_c] = lambda v: "—" if pd.isna(v) else f"{float(v):+.1f}"
                st.dataframe(_explain_df.style.format(_fmt_exp), use_container_width=True, hide_index=True)
                if _input_lines:
                    st.caption("Sisendid · " + "  |  ".join(_input_lines))
        else:
            if _target_day == TODAY:
                st.caption(
                    "A+B+C tegurite selgitus puudub selle lukustatud korje-eelse snapshoti juurest. "
                    "Kui järgmine operatiivring annab sama snapshoti, lisatakse selgitus automaatselt; "
                    "eri prognoosi selgitust vana numbri külge ei näidata."
                )
            else:
                st.caption("A+B+C tegurite selgitus salvestub järgmise päris operatiivse prognoosiringiga.")

        _ro_shown += 1

    if _ro_shown == 0:
        st.info("Salvestatud ametlikku prognoosi veel pole. Järgmine päris sisendimuutus käivitab operatiivse prognoosiringi.")
    else:
        try:
            _ro_last_at = str(db.get_app_setting("forecast_last_checked_at", "") or "")
            if _ro_last_at:
                _ro_dt = datetime.fromisoformat(_ro_last_at).astimezone(ZoneInfo("Europe/Tallinn"))
                st.caption(f"Viimane operatiivne prognoosiring: {_ro_dt.strftime('%d.%m %H:%M')}. Tegurite detail uueneb järgmise päris arvutusringiga.")
        except Exception:
            pass

if page == "Mootori tähelepanekud":
    st.subheader("Mootori tähelepanekud")

    st.markdown("#### Mis on uut?")
    _disc_raw = db.get_app_setting("discovery_history_json", "")
    try:
        _disc_hist = json.loads(_disc_raw) if _disc_raw else []
        if not isinstance(_disc_hist, list):
            _disc_hist = []
    except Exception:
        _disc_hist = []

    if _disc_hist:
        _recent = list(reversed(_disc_hist[-5:]))
        _recent_rows = []
        for _x in _recent:
            try:
                _dt = datetime.fromisoformat(str(_x.get("detected_at"))).astimezone(ZoneInfo("Europe/Tallinn"))
                _when = f"{_weekday_letter(_dt.date())} {_dt.strftime('%d.%m')} {_dt.strftime('%H:%M')}"
            except Exception:
                _when = str(_x.get("detected_at") or "—")
            try:
                _imp = f"{float(_x.get('improvement')):+.2f}"
            except Exception:
                _imp = "—"
            _recent_rows.append({
                "Aeg": _when,
                "Tähelepanek": str(_x.get("name") or "—"),
                "Paranemine": _imp,
            })
        st.dataframe(pd.DataFrame(_recent_rows), use_container_width=True, hide_index=True)
        st.caption("Siin on kuni 5 viimast olulist Jäljeotsija muutust. Täielik diagnostika jääb allapoole.")
    else:
        st.caption("Uusi olulisi tähelepanekuid pole veel.")

    with st.expander("🧰 Pausil tööriistad"):
        st.markdown("**Autonoomne ideegeneraator — PAUSIL**")
        st.caption(
            "Loob ise uusi tunnuseid ja koostoimeid ning testib neid walk-forward meetodil. "
            "Praegu on see CPU säästmiseks ja vähese õppimisandmestiku tõttu pausil. "
            "Selle rolli katab hetkel suunatud lai kihiline otsing. "
            "Tööriista tasub uuesti kaaluda siis, kui täielikke korjepäevi ja õppimisnäiteid on oluliselt rohkem."
        )
        st.caption("Olek on ainult informatiivne — selle ploki avamine ei käivita ühtegi arvutust.")

    _lr_at = db.get_app_setting("layered_research_last_at", "")
    _lr_sec = db.get_app_setting("layered_research_last_duration_s", "")
    _lr_n = db.get_app_setting("layered_research_last_candidates", "")
    if _lr_at:
        try:
            _lr_dt = datetime.fromisoformat(_lr_at).astimezone(ZoneInfo("Europe/Tallinn"))
            _lr_age = max(0, int((datetime.now(ZoneInfo("Europe/Tallinn")) - _lr_dt).total_seconds() // 3600))
            _lr_msg = f"Viimane lai uuring {_lr_dt.strftime('%d.%m %H:%M')} · {_lr_age} h tagasi"
        except Exception:
            _lr_msg = f"Viimane lai uuring {_lr_at}"
        if _lr_sec:
            _lr_msg += f" · kestus {_lr_sec} s"
        if _lr_n:
            _lr_msg += f" · raportis {_lr_n} kandidaati"
        st.info("🔬 " + _lr_msg)
    else:
        st.info("🔬 Laia kihilist uuringut pole veel käsitsi käivitatud.")
    st.caption(
        "Mootori tähelepanekute avamine ei arvuta prognoosi ega Jäljeotsijat uuesti. "
        "Uue täieliku korjepäeva järel saab Jäljeotsija ühe ringi ning lai kihiline otsing "
        "võib saada sama päeva ühisest uurimispotist ühe automaatse piiratud vooru. "
        "Kui potis on aega alles, saab laia otsingut käsitsi samal andmeseisul edasi jätkata; "
        "juba testitud kombinatsioone ei korrata. Ilmauuendus üksi uurimist ei käivita."
    )

    with st.expander("⚙️ Tehniline koormus"):
        _lc_sec = db.get_app_setting("light_cycle_last_duration_s", "")
        _lc_total_sec = db.get_app_setting("light_cycle_last_total_duration_s", "")
        _lc_at = db.get_app_setting("light_cycle_last_at", "")
        _lc_reason = db.get_app_setting("light_cycle_last_reason", "")
        _lc_abc = db.get_app_setting("light_cycle_last_abc_candidates", "")
        _lc_cb = db.get_app_setting("light_cycle_last_cb_candidates", "")
        _lc_rows = db.get_app_setting("light_cycle_last_training_rows", "")
        _lc_cache_mode = db.get_app_setting("light_cycle_last_cache_mode", "")
        _lr_sec_tech = db.get_app_setting("layered_research_last_duration_s", "")
        _lr_n_tech = db.get_app_setting("layered_research_last_candidates", "")
        _lr_run_n_tech = db.get_app_setting("layered_research_last_run_candidates", "")
        _lr_budget_tech = db.get_app_setting("layered_research_last_budget_s", "")
        _lr_avg_candidate_tech = db.get_app_setting("layered_research_last_candidate_avg_s", "")
        _fc_sec = db.get_app_setting("forecast_cycle_last_duration_s", "")
        _fc_reason = db.get_app_setting("forecast_cycle_last_reason", "")
        _fc_at = db.get_app_setting("forecast_last_checked_at", "")

        if _lc_sec:
            st.write(f"**Viimane kerge/Jäljeotsija ring:** {_lc_sec} s")
            if _lc_reason:
                st.caption(f"Käivitus: {_lc_reason}")
            if _lc_at:
                try:
                    _lc_dt = datetime.fromisoformat(_lc_at).astimezone(ZoneInfo("Europe/Tallinn"))
                    st.caption(f"Aeg: {_lc_dt.strftime('%d.%m %H:%M:%S')}")
                except Exception:
                    st.caption(f"Aeg: {_lc_at}")
            st.caption(
                f"Õppimisridu: {_lc_rows or '—'} · "
                f"A+B+C kandidaate: {_lc_abc or '—'} · "
                f"C/B kandidaate: {_lc_cb or '—'}"
            )
            if _lc_cache_mode:
                st.caption(f"Walk-forward töörežiim: {_lc_cache_mode}")
            if _lc_total_sec:
                st.caption(
                    f"Kogu sama arvutustsükkel {_lc_total_sec} s · Jäljeotsija näit ei sisalda enam laia uuringu aega."
                )
        else:
            st.caption("Päris Jäljeotsija ringi kestust pole veel mõõdetud.")

        if _fc_sec:
            st.write(f"**Viimane operatiivne prognoosiring:** {_fc_sec} s")
            if _fc_reason:
                st.caption(f"Käivitus: {_fc_reason}")
            if _fc_at:
                try:
                    _fc_dt = datetime.fromisoformat(_fc_at).astimezone(ZoneInfo("Europe/Tallinn"))
                    st.caption(f"Aeg: {_fc_dt.strftime('%d.%m %H:%M:%S')} · see ring ei käivita Jäljeotsijat")
                except Exception:
                    pass

        if _lr_sec_tech:
            st.write(f"**Viimane lai uurimisvoor:** {_lr_sec_tech} s")
            st.caption(
                f"Selles voorus lõpetatud katseid: {_lr_run_n_tech or '—'} · "
                f"selle andmeseisu uuritud kombinatsioone: {_lr_n_tech or '—'} · "
                f"keskmine katse: {_lr_avg_candidate_tech or '—'} s · "
                f"ajabudžett: {_lr_budget_tech or '—'} s"
            )
        else:
            st.caption("Laia kihilise uuringu kestust pole veel salvestatud.")

        st.write(
            f"**Tänane ühine uurimispott:** {_cpu_used_today_s:.1f}/{CPU_RESEARCH_DAILY_BUDGET_S:.0f} s "
            f"· alles {_cpu_remaining_s():.1f} s"
        )
        st.caption(
            f"Jäljeotsija arvestus {_cpu_light_today_s:.1f} s · lai uuring {_cpu_layered_today_s:.1f} s. "
            "Kui pott on täis, uut Jäljeotsijat ega laia uuringut sel päeval ei käivitata. "
            "Ilm ja operatiivne 9 päeva prognoos sellesse potti ei kuulu."
        )
        _last_skip = db.get_app_setting("layered_research_last_skip_reason", "")
        if _last_skip:
            st.caption(f"Viimane CPU-kaitse: {_last_skip}")

        st.caption(
            "Need näidud mõõdavad KurgiMootori enda arvutusringide wall-clock kestust, "
            "mitte Streamlit Cloudi konto täpset CPU-protsenti ega ametlikku päevakvooti."
        )

    _obs_ready = (
        "training_rows" in locals()
        and bool(training_rows)
        and "trace_df" in locals()
        and "champion_name" in locals()
    )

    if not _obs_ready:
        st.caption(
            "Leht näitab viimase päris arvutusringi salvestatud tähelepanekuid. "
            "Selle lehe avamine ei käivita prognoosi ega Jäljeotsijat uuesti."
        )

        _obs_raw = db.get_app_setting("motor_observation_snapshot_json", "")
        try:
            _obs = json.loads(_obs_raw) if _obs_raw else {}
            if not isinstance(_obs, dict):
                _obs = {}
        except Exception:
            _obs = {}

        if not _obs:
            st.info("Täielikku salvestatud tähelepanekute snapshot'i veel pole. See tekib järgmise päris Jäljeotsija või laia uurimisringiga.")
        else:
            _saved_at = str(_obs.get("saved_at") or "")
            if _saved_at:
                try:
                    _sdt = datetime.fromisoformat(_saved_at).astimezone(ZoneInfo("Europe/Tallinn"))
                    st.caption(f"Seis salvestatud {_sdt.strftime('%d.%m %H:%M')}")
                except Exception:
                    pass

            st.caption("Õppimisaudit: põllu identiteet ja korjeintervall on mudelis aktiivsed; täpne ~3 h/põld kasvuaeg on Jäljeotsija diagnostika.")

            _field_watch_saved = _obs.get("field_state_watch") or []
            if _field_watch_saved:
                with st.expander("⚠️ Põllu seisundi võimalik muutus"):
                    st.caption("See ei diagnoosi haigust ega muuda prognoosi automaatselt. Märge tekib korduva kõrvalekalde korral.")
                    st.dataframe(pd.DataFrame(_field_watch_saved), use_container_width=True, hide_index=True)

            st.markdown("### 📝 Viimase korje järel")
            _j = _obs.get("journal") or {}
            _jday = _obs.get("last_complete_harvest")
            if _jday:
                try:
                    _jd = date.fromisoformat(str(_jday))
                    st.caption(f"{_weekday_letter(_jd)} {_short_date(_jd)}")
                except Exception:
                    pass
            if _j.get("actual_total") is not None:
                _msg = f"Tegelik päevasaak **{float(_j['actual_total']):.1f} kasti**"
                if _j.get("forecast_total") is not None:
                    _diff = float(_j['actual_total']) - float(_j['forecast_total'])
                    _msg += f" · enne korjet salvestatud prognoos **{float(_j['forecast_total']):.1f}** · vahe **{_diff:+.1f}**"
                st.write(_msg)
                if _j.get("forecast_champion"):
                    st.caption(f"Selle snapshoti prognoosi juhtis: {_j['forecast_champion']}")
            else:
                st.caption("Viimase korjepäeva päevavõrdlust pole snapshotis.")

            _champ = str(_obs.get("champion_name") or "Baasmudel")
            _cmae = _obs.get("champion_mae")
            _cstats = _obs.get("champion_stats") or {}
            st.markdown("### ✅ Praegu usaldan")
            if _cstats:
                try:
                    st.success(
                        f"**{_champ}** · kinnitus-MAE {float(_cstats.get('Katse MAE')):.2f} vs baas "
                        f"{float(_cstats.get('Baas MAE')):.2f} · paranemine {float(_cstats.get('Paranemine')):+.2f}."
                    )
                except Exception:
                    st.success(f"**{_champ}** on viimane salvestatud kinnitatud champion.")
            else:
                suffix = "" if _cmae is None else f" · MAE {float(_cmae):.2f}"
                st.info(f"**{_champ}** jääb championiks{suffix}.")

            _trace_df_saved = pd.DataFrame(_obs.get("trace") or [])
            _weather_names = set(_obs.get("weather_groups") or [])
            _bio_names = set(_obs.get("bio_groups") or [])
            _memory_names = set(_obs.get("memory_groups") or [])
            _screen_names = set(_obs.get("screening_groups") or [])

            if not _trace_df_saved.empty:
                with st.expander("Mida Jäljeotsija viimati kaalus"):
                    _show_cols = [c for c in ["Jälg", "Baas MAE", "Katse MAE", "Paranemine", "Võidab ridu %", "Halvim pool", "Stabiilne"] if c in _trace_df_saved.columns]
                    st.dataframe(_trace_df_saved[_show_cols].sort_values("Paranemine", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("### 🧬 Üks huvitav kihiline idee")
            _layer_df_saved = pd.DataFrame(_obs.get("layered") or [])
            if not _layer_df_saved.empty:
                _layer_df_saved = _layer_df_saved.sort_values("Paranemine", ascending=False)
                _idea = _layer_df_saved.iloc[0]
                try:
                    st.info(
                        f"**{_idea.get('Mudel', 'Kihiline kandidaat')}** · MAE {float(_idea.get('Katse MAE')):.2f} vs baas "
                        f"{float(_idea.get('Baas MAE')):.2f} · Δ {float(_idea.get('Paranemine')):+.2f}."
                    )
                except Exception:
                    st.info(f"**{_idea.get('Mudel', 'Kihiline kandidaat')}**")
                if _idea.get("Koosseis"):
                    st.caption("Koosseis: " + str(_idea.get("Koosseis")))
                _best_ch_state = _load_json_setting("layered_best_challenger_json", {})
                _idea_parts_key = str(_idea.get("Koosseis") or "")
                _best_parts_key = " | ".join(map(str, (_best_ch_state or {}).get("parts") or [])) if isinstance(_best_ch_state, dict) else ""
                if _best_ch_state and _idea_parts_key and _idea_parts_key == _best_parts_key:
                    if bool(_best_ch_state.get("confirmed")):
                        st.caption("Kinnitus: ✅ kandidaat läbis eraldi hilisema kinnitustesti.")
                    else:
                        _conf = _best_ch_state.get("confirmation_stats")
                        if isinstance(_conf, dict) and _conf:
                            st.caption(
                                "Staatus: avastuses tugev, kuid hilisem kinnitustest ei läbinud veel kõiki lävendeid. "
                                "Championiks ta seetõttu ei lähe."
                            )
                        else:
                            st.caption(
                                "Staatus: avastustulemus. Championiks saab kandidaat alles eraldi hilisema "
                                "kinnitustesti järel; kinnituseks võib praegu olla veel liiga vähe andmeid."
                            )
                else:
                    st.caption("Avastustulemus — championiks saab alles eraldi hilisema kinnitustesti järel.")
                with st.expander("Näita salvestatud kihilisi tulemusi"):
                    st.dataframe(_layer_df_saved, use_container_width=True, hide_index=True)
            else:
                st.caption("Salvestatud kihilist kandidaati veel pole.")

            with st.expander("🔎 Viimane salvestatud ideeradar"):
                _auto_df_saved = pd.DataFrame(_obs.get("autonomous") or [])
                if not _auto_df_saved.empty:
                    _counts = _obs.get("autonomous_category_counts") or {}
                    _n = int(_obs.get("autonomous_candidate_count") or len(_auto_df_saved))
                    st.caption(f"Viimases salvestatud radaris oli {_n} ideed. Ideegeneraator on praegu pausil ja need ei lähe automaatselt championiks.")
                    _auto_show = _auto_df_saved.head(10)
                    st.dataframe(_auto_show.drop(columns=["_Veerg"], errors="ignore"), use_container_width=True, hide_index=True)
                else:
                    st.caption("Vabade ideede radari salvestatud tulemusi pole või viimane ring ei käivitanud seda osa.")

            st.markdown("### 👀 Hoian silma peal")
            if not _trace_df_saved.empty and "Paranemine" in _trace_df_saved.columns:
                _watch_saved = _trace_df_saved[(pd.to_numeric(_trace_df_saved["Paranemine"], errors="coerce") > 0)]
                if "Stabiilne" in _watch_saved.columns:
                    _watch_saved = _watch_saved[_watch_saved["Stabiilne"] != True]
                _watch_saved = _watch_saved.sort_values("Paranemine", ascending=False).head(5)
            else:
                _watch_saved = pd.DataFrame()
            if _watch_saved.empty:
                st.caption("Praegu pole salvestatud jälge, mis parandaks baasi, kuid jääks napilt stabiilsuslävendi alla.")
            else:
                for _, _r in _watch_saved.iterrows():
                    st.write(f"• **{_r.get('Jälg','—')}** — MAE muutus {float(_r.get('Paranemine')):+.2f}; võitis {float(_r.get('Võidab ridu %',0)):.0f}% testiridadest.")

            with st.expander("⛔ Praegu ei kasuta"):
                if not _trace_df_saved.empty and "Paranemine" in _trace_df_saved.columns:
                    _rej = _trace_df_saved[_trace_df_saved["Jälg"].isin(_screen_names)] if _screen_names and "Jälg" in _trace_df_saved.columns else _trace_df_saved
                    _rej = _rej[pd.to_numeric(_rej["Paranemine"], errors="coerce") <= 0].sort_values("Paranemine", ascending=False).head(5)
                else:
                    _rej = pd.DataFrame()
                if _rej.empty:
                    st.caption("Praegu pole salvestatud lubatud kandidaate, mis oleksid baasist selgelt halvemad.")
                else:
                    for _, _r in _rej.iterrows():
                        st.write(f"• **{_r.get('Jälg','—')}** — ei parandanud baasmudelit (MAE muutus {float(_r.get('Paranemine')):+.2f}).")

                if not _trace_df_saved.empty and _memory_names and "Jälg" in _trace_df_saved.columns:
                    _mem = _trace_df_saved[_trace_df_saved["Jälg"].isin(_memory_names)].copy()
                    if not _mem.empty:
                        _mem = _mem.sort_values("Paranemine", ascending=False)
                        _bm = _mem.iloc[0]
                        st.divider()
                        st.markdown("#### 🧪 Ainult uurimiseks")
                        st.write(
                            f"Tugevaim saagimälu diagnostiline signaal on **{_bm.get('Jälg','—')}** "
                            f"(MAE muutus {float(_bm.get('Paranemine')):+.2f}; võitis {float(_bm.get('Võidab ridu %',0)):.0f}% ridadest). "
                            "Seda ei kasutata A+B+C prognoosi ankruna."
                        )

            st.caption("Tähelepanekud pärinevad viimasest salvestatud walk-forward ringist. Leht ise ei õpeta ega muuda mootorit.")
    else:
        st.caption("Õppimisaudit: põllu identiteet ja korjeintervall on mudelis aktiivsed; täpne ~3 h/põld kasvuaeg on praegu ainult Jäljeotsija diagnostika.")
        if "field_state_watch" in locals() and field_state_watch:
            with st.expander("⚠️ Põllu seisundi võimalik muutus"):
                st.caption("See ei diagnoosi haigust ega muuda prognoosi automaatselt. Märge tekib, kui sama põld jääb kahel viimasel ausal testil ootusest selgelt allapoole.")
                st.dataframe(pd.DataFrame(field_state_watch), use_container_width=True, hide_index=True)
        # ---------------------------------------------------------------------
        # Mootori päevik: mida viimane täielik korjepäev muutis?
        # See on ainult diagnostika. Siit EI lähe ükski väärtus prognoosi sisendiks.
        # ---------------------------------------------------------------------
        st.markdown("### 📝 Viimase korje järel")

        _journal_day = last_complete_harvest
        _journal_rows = harvest_by_day.get(_journal_day.isoformat(), []) if _journal_day else []
        _journal_expected = {
            int(r.get("field_no")) for r in _journal_rows if r.get("field_no") is not None
        }

        _journal_actual_total = None
        _journal_actual_abc = None
        if _journal_rows:
            try:
                _journal_actual_total = float(sum(float(r.get("total")) for r in _journal_rows))
                _journal_actual_abc = float(sum(
                    float(r.get("a", 0) or 0)
                    + float(r.get("b", 0) or 0)
                    + float(r.get("c", 0) or 0)
                    for r in _journal_rows
                ))
            except (TypeError, ValueError):
                _journal_actual_total = None
                _journal_actual_abc = None

        # Leia selle korjepäeva viimane täielik prognoosisnapshot, mis oli olemas
        # hiljemalt korjepäeval. Sama päeva lead=0 snapshot on enne esimese tegeliku
        # korje sisestamist lukustatud ja sobib seetõttu ausaks päevavõrdluseks.
        _journal_snapshot = None
        if _journal_day and _journal_expected and "forecast_history_rows" in locals():
            _by_fdate = _snapshot_batch_map(
                forecast_history_rows, _journal_day, max_forecast_date=_journal_day
            )

            _complete_snaps = []
            for _fdate, _rows_for_date in _by_fdate.items():
                if set(_rows_for_date.keys()) != _journal_expected:
                    continue
                try:
                    _total_fc = float(sum(float(_rows_for_date[f]["total_forecast"]) for f in _journal_expected))
                    _abc_fc = float(sum(float(_rows_for_date[f]["abc_forecast"]) for f in _journal_expected))
                except (TypeError, ValueError, KeyError):
                    continue
                _basis_values = [
                    str(_rows_for_date[f].get("basis") or "") for f in _journal_expected
                ]
                _complete_snaps.append((_fdate, _total_fc, _abc_fc, _basis_values))

            if _complete_snaps:
                _complete_snaps.sort(key=lambda x: x[0])
                _journal_snapshot = _complete_snaps[-1]

        if _journal_day is None:
            st.caption("Veel pole täielikku korjepäeva, mille järel mootori reaktsiooni hinnata.")
        elif _journal_snapshot is None or _journal_actual_total is None:
            st.caption(
                f"{_weekday_letter(_journal_day)} {_short_date(_journal_day)} on täielik korjepäev, "
                "aga selle päeva ausat täielikku prognoosisnapshot'i ei ole veel võrdluseks olemas."
            )
        else:
            _j_fdate, _j_fc_total, _j_fc_abc, _j_basis = _journal_snapshot
            _j_diff = _journal_actual_total - _j_fc_total
            _j_abs_pct = abs(_j_diff) / _journal_actual_total * 100.0 if _journal_actual_total > 0 else None
            if abs(_j_diff) < 0.05:
                _j_direction = "tabas praktiliselt täpselt"
            elif _j_diff > 0:
                _j_direction = f"alahindas {_j_diff:.1f} kasti"
            else:
                _j_direction = f"ülehindas {abs(_j_diff):.1f} kasti"

            st.info(
                f"**{_weekday_letter(_journal_day)} {_short_date(_journal_day)}:** "
                f"prognoos {_j_fc_total:.1f} → tegelik {_journal_actual_total:.1f} kasti. "
                f"Mootor {_j_direction}"
                + (f" ({_j_abs_pct:.1f}% tegelikust)." if _j_abs_pct is not None else ".")
            )

            # Mis champion selle snapshoti tegemise ajal prognoosi juhtis?
            _forecast_champion = None
            for _basis in _j_basis:
                _m = re.search(r"(?:^|;\s*)champion=([^;]+)", _basis)
                if _m:
                    _forecast_champion = _m.group(1).strip()
                    break

            if _forecast_champion:
                if _forecast_champion == champion_name:
                    st.write(f"**Champion jäi samaks:** {champion_name}. Üks päev ei põhjustanud mudelivahetust.")
                else:
                    st.write(
                        f"**Champion muutus:** prognoosi tegemisel oli {_forecast_champion}, "
                        f"praegu on {champion_name}."
                    )
            else:
                st.write(f"**Praegune champion:** {champion_name}.")

            # Mõõda, millise operatiivse kandidaadi ajalooline tõendus muutus viimase
            # testipäeva lisandumisel kõige rohkem. Walk-forward ennustused vanematele
            # päevadele ei muutu, seega saab seda võrrelda ilma mudelit uuesti treenimata.
            _journal_changes = []
            if (
                "candidate_predictions" in locals()
                and "operational_candidate_groups" in locals()
                and _journal_day in set(dates)
            ):
                _date_arr = np.asarray(dates, dtype=object)
                _base_arr = np.asarray(predictions, dtype=float)
                _target_arr = np.asarray(y, dtype=float)

                for _name in operational_candidate_groups.keys():
                    _cand = candidate_predictions.get(_name)
                    if _cand is None:
                        continue
                    _cand_arr = np.asarray(_cand, dtype=float)
                    _valid = np.isfinite(_base_arr) & np.isfinite(_cand_arr) & np.isfinite(_target_arr)
                    _prev_idx = np.where(_valid & (_date_arr < _journal_day))[0]
                    _now_idx = np.where(_valid & (_date_arr <= _journal_day))[0]
                    _day_idx = np.where(_valid & (_date_arr == _journal_day))[0]
                    if len(_prev_idx) < 1 or len(_now_idx) < 1 or len(_day_idx) < 1:
                        continue

                    _prev_imp = float(
                        np.mean(np.abs(_base_arr[_prev_idx] - _target_arr[_prev_idx]))
                        - np.mean(np.abs(_cand_arr[_prev_idx] - _target_arr[_prev_idx]))
                    )
                    _now_imp = float(
                        np.mean(np.abs(_base_arr[_now_idx] - _target_arr[_now_idx]))
                        - np.mean(np.abs(_cand_arr[_now_idx] - _target_arr[_now_idx]))
                    )
                    _day_imp = float(
                        np.mean(np.abs(_base_arr[_day_idx] - _target_arr[_day_idx]))
                        - np.mean(np.abs(_cand_arr[_day_idx] - _target_arr[_day_idx]))
                    )
                    _journal_changes.append({
                        "name": _name,
                        "delta": _now_imp - _prev_imp,
                        "day_imp": _day_imp,
                        "now_imp": _now_imp,
                    })

            if _journal_changes:
                _strongest = max(_journal_changes, key=lambda r: abs(r["delta"]))
                if abs(_strongest["delta"]) < 0.005:
                    st.write("**Kandidaatide pilt sisuliselt ei muutunud.** Viimane päev ei nihutanud ühtegi jälge märgatavalt.")
                elif _strongest["delta"] > 0:
                    st.write(
                        f"**Kõige rohkem tugevnes:** {_strongest['name']} "
                        f"(ajalooline MAE-eelis muutus {_strongest['delta']:+.2f} kasti; "
                        f"viimasel päeval oli selle jälje eelis baasi ees {_strongest['day_imp']:+.2f})."
                    )
                else:
                    st.write(
                        f"**Kõige rohkem nõrgenes:** {_strongest['name']} "
                        f"(ajalooline MAE-eelis muutus {_strongest['delta']:+.2f} kasti; "
                        f"viimasel päeval oli selle jälje eelis baasi ees {_strongest['day_imp']:+.2f})."
                    )

                _current_watch = sorted(
                    [r for r in _journal_changes if r["now_imp"] > 0],
                    key=lambda r: r["now_imp"],
                    reverse=True,
                )
                if _current_watch:
                    _w = _current_watch[0]
                    st.caption(
                        f"Praegu tugevaim operatiivne jälg selle lihtsa MAE-võrdluse järgi: "
                        f"{_w['name']} (+{_w['now_imp']:.2f} kasti vs baas). "
                        "See ei tee sellest automaatselt championit; avastus- ja kinnitusetapp jäävad endiselt nõutuks."
                    )
            else:
                st.caption(
                    "Viimase päeva mõju kandidaatide tõendusjõule ei saanud veel eraldi mõõta "
                    "(päev ei ole walk-forward testiosas või võrdlusridu on liiga vähe)."
                )

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


        st.info(
            "🧬 **V6.5 kihiline BIO aktiivne:** üksiktunnused on radar, mitte lõplikud mudelid. "
            "Mootor testib bioloogiliselt lubatud kihtide kombinatsioone; championiks saab "
            "ainult kihiline mudel pärast eraldi kinnitust."
        )

        # V6.5: kasutajale üks selge huvitav kihiline idee.
        st.markdown("### 💡 Üks huvitav kihiline idee")
        if "layered_trace_df" in locals() and not layered_trace_df.empty:
            _idea = layered_trace_df.iloc[0]
            _idea_regime = _idea.get("Režiim Δ")
            _idea_trend = _idea.get("Trend %")
            _idea_over = _idea.get("Ülereageerimine")

            _idea_text = (
                f"**{_idea['Mudel']}** — avastuses MAE {_idea['Katse MAE']:.2f} "
                f"vs baas {_idea['Baas MAE']:.2f} (Δ {_idea['Paranemine']:+.2f}); "
                f"võitis {_idea['Võidab ridu %']:.0f}% ridadest."
            )
            st.info(_idea_text)
            st.caption("Koosseis: " + str(_idea["Koosseis"]))

            _quality_bits = []
            if pd.notna(_idea_trend):
                _quality_bits.append(f"trendi tabamine {_idea_trend:.0f}%")
            if pd.notna(_idea_regime):
                _quality_bits.append(f"ilmamuutuse Δ {_idea_regime:+.2f}")
            if pd.notna(_idea_over):
                _quality_bits.append(f"ülereageerimine {_idea_over:.2f}")
            if _quality_bits:
                st.caption(" · ".join(_quality_bits))

            if bool(_idea.get("Stabiilne", False)):
                _live_best_ch = _load_json_setting("layered_best_challenger_json", {})
                _live_idea_key = str(_idea.get("Koosseis") or "")
                _live_best_key = " | ".join(map(str, (_live_best_ch or {}).get("parts") or [])) if isinstance(_live_best_ch, dict) else ""
                _live_conf = _live_best_ch.get("confirmation_stats") if isinstance(_live_best_ch, dict) and _live_idea_key == _live_best_key else None
                if isinstance(_live_conf, dict) and _live_conf:
                    if bool(_live_conf.get("Stabiilne")):
                        st.success("Avastuses stabiilne ✅ · kinnituses stabiilne ✅. Kui see on parim kinnitatud kandidaat, võib mootor selle championiks valida.")
                    else:
                        st.caption("Avastuses stabiilne ✅ · kinnituses ei läbinud veel kõiki lävendeid. Championiks ta praegu ei lähe.")
                else:
                    st.success(
                        "See kombinatsioon on avastusperioodil stabiilne kandidaat, "
                        "kuid championiks saamiseks peab ta läbima eraldi kinnituse."
                    )
            else:
                st.caption(
                    "See on praegu uurimisidee, mitte champion. Uute korjetega kontrollitakse seda uuesti."
                )
        else:
            st.caption(
                "Kihiline otsing ei leidnud praegusest väikesest ja ühetaolisest andmestikust "
                "veel piisavalt hinnatavat vähemalt kahe kihi kombinatsiooni."
            )

        with st.expander("🔎 Viimane salvestatud ideeradar"):
            st.caption("Autonoomne ideegeneraator on praegu pausil. Allpool on viimane salvestatud radar; selle avamine ei käivita uut otsingut.")

            if "autonomous_trace_df" not in locals() or autonomous_trace_df.empty:
                st.caption("Ideegeneraator ei saanud praegu piisava andmekvaliteediga uusi kandidaate testida.")
            else:
                st.caption(
                    "Allolev osa on ideede radar. Ükski siin eraldi kõrgele jõudev tunnus "
                    "ei saa enam A+B+C championiks; tema väärtus on anda kihilisele otsingule materjali."
                )
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
                    f"ilmamälu **{autonomous_category_counts.get('Ilmamälu', 0)}**",
                    f"hooaeg × ilm **{autonomous_category_counts.get('Hooaeg × ilm', 0)}**",
                    f"bioloogilist koormust **{autonomous_category_counts.get('Bioloogiline koormus', 0)}**",
                    f"BIO füsioloogiat **{autonomous_category_counts.get('BIO füsioloogia', 0)}**",
                    f"2. ringi kombinatsioone **{autonomous_category_counts.get('Teise ringi kombinatsioonid', 0)}**",
                ]
                st.info(" · ".join(_space_parts))

                # Cache'ist taastatud vanematel tulemustel võis päevade nimekiri puududa.
                # Sel juhul tuleta päriselt kasutatud plokkide päevade arv tulemustabeli ridade
                # ja tänase teststruktuuri põhjal, mitte ära kuva eksitavat 0/0.
                _shown_discovery_days = len(_auto_discovery_days) if "_auto_discovery_days" in locals() else 0
                _shown_confirm_days = len(_auto_confirm_days) if "_auto_confirm_days" in locals() else 0
                if _shown_discovery_days == 0 or _shown_confirm_days == 0:
                    _valid_days_for_auto = sorted(set(dates[np.where(np.isfinite(champion_pred) & np.isfinite(y))[0]]))
                    if len(_valid_days_for_auto) >= 4:
                        _shown_confirm_days = max(2, int(round(len(_valid_days_for_auto) * 0.30)))
                        _shown_confirm_days = min(_shown_confirm_days, len(_valid_days_for_auto) - 2)
                    else:
                        _shown_confirm_days = 1 if len(_valid_days_for_auto) >= 2 else 0
                    _shown_discovery_days = max(0, len(_valid_days_for_auto) - _shown_confirm_days)

                st.caption(
                    f"1. ring ja 2. ring valitakse ainult vanemas avastusplokis "
                    f"({_fmt(_shown_discovery_days, 0)} testipäeva). "
                    f"Hilisemad {_fmt(_shown_confirm_days, 0)} testipäeva on eraldi kinnitusplokk. "
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
                        f"Tugevaim eraldi radarisignaal: **{_best['Idee']}**. "
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

        with st.expander("⛔ Praegu ei kasuta"):
            _rejected = _operational_trace[
                _operational_trace["Paranemine"] <= 0
            ].head(5)

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
