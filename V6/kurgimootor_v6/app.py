from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import db
from core import WeatherService

TODAY = date.today()
st.set_page_config(page_title="KurgiMootor V6.3", page_icon="🥒", layout="wide")


def _n(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")




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
try:
    WeatherService().auto_refresh_if_needed(TODAY)
except Exception as exc:
    db.set_app_setting("weather_last_error", f"Automaatne ilmauuendus: {exc}")

st.title("KurgiMootor V6.3")
st.caption("Saagi ennustamise tööriist. Avaleht on töövoog, mitte ilmarakendus.")

tabs = st.tabs(["Täna", "Korjed", "Ilm", "Prognoos", "Mootori tähelepanekud"])

with tabs[0]:
    st.subheader("Täna")
    today_rows = db.get_harvest_for_day(TODAY)
    harvest_history_for_plan = db.get_harvest_history()
    today_planned_fields = _planned_fields_for_day(TODAY, today_rows, harvest_history_for_plan)
    _render_day_block(_short_date(TODAY), today_rows, planned_fields=today_planned_fields)

with tabs[1]:
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
    with st.form("manual_harvest_form"):
        c1, c2, c3 = st.columns(3)
        entry_date = c1.date_input("Kuupäev", value=TODAY, key="manual_harvest_date")
        entry_field = c2.selectbox(
            "Põld",
            list(range(1, 15)),
            index=max(0, min(13, default_field - 1)),
            key=f"manual_harvest_field_{form_version}",
        )
        entry_order = c3.selectbox(
            "Järjekord",
            [1, 2, 3],
            index=max(0, min(2, default_order - 1)),
            key=f"manual_harvest_order_{form_version}",
        )
        q1, q2, q3, q4 = st.columns(4)
        entry_a = q1.number_input("A", 0.0, step=0.1, format="%.1f", key=f"manual_a_{form_version}")
        entry_b = q2.number_input("B", 0.0, step=0.1, format="%.1f", key=f"manual_b_{form_version}")
        entry_c = q3.number_input("C", 0.0, step=0.1, format="%.1f", key=f"manual_c_{form_version}")
        entry_xl = q4.number_input("XL", 0.0, step=0.1, format="%.1f", key=f"manual_xl_{form_version}")
        total_preview = entry_a + entry_b + entry_c + entry_xl
        cb_preview = entry_c / entry_b if entry_b > 0 else None
        preview_text = f"Kokku: {_fmt(total_preview)}"
        if cb_preview is not None:
            preview_text += f" · C/B: {_fmt(cb_preview, 2)}"
        st.caption(preview_text)

        if st.form_submit_button("Salvesta korje"):
            if total_preview <= 0:
                st.warning("Korje kogus on 0. Sisesta vähemalt üks kogus.")
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

                # Järgmise rea loogika: põld edasi, 14 järel 1; järjekord 1→2→3→1.
                st.session_state["next_harvest_field"] = 1 if entry_field >= 14 else entry_field + 1
                st.session_state["next_harvest_order"] = 1 if entry_order >= 3 else entry_order + 1

                # Uus vormiversioon sunnib Streamliti looma värsked väljad.
                # Nii avaneb järgmine rida päriselt järgmise põllu/järjekorraga ja A/B/C/XL = 0.
                st.session_state["harvest_form_version"] = form_version + 1
                st.session_state["harvest_saved_message"] = (
                    f"Salvestatud: {entry_date} · põld {entry_field} · kokku {_fmt(total_preview)}"
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
                day_label = _short_date(day_date)
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

with tabs[2]:
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
        if result.get("error"):
            st.warning(f"Uuendus tehti osaliselt: {result['error']}")
        else:
            st.success("Ilmaandmed uuendatud.")
        st.rerun()

    history_default_start = date(TODAY.year, 7, 1)
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

    st.caption(f"Kuvatakse {len(measured_rows)} mõõdetud päeva. Kõik salvestatud ilmaandmed jäävad mootorile kasutada sõltumata valitud vaatest.")
    measured_display = [{
        "Kuupäev": r["weather_date"],
        "Min °C": r.get("temp_min_c"),
        "Max °C": r.get("temp_max_c"),
        "Tuul m/s": r.get("wind_avg_ms"),
        "Niiskus %": r.get("humidity_avg_pct"),
        "Sademed mm": r.get("precipitation_mm"),
        "ET0 mm": r.get("et0_mm"),
        "Radiatsioon MJ/m²": r.get("radiation_mj_m2"),
        "Kontroll": r.get("check_message"),
        "Olek": "🟢 Kontrollitud" if r.get("checked") else "🔴 Puudulik",
    } for r in measured_rows]
    st.dataframe(pd.DataFrame(measured_display), use_container_width=True, hide_index=True)

    st.subheader("9 päeva prognoos")
    forecast_display = [{
        "Kuupäev": r["weather_date"],
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

with tabs[3]:
    st.subheader("Prognoos")
    st.markdown("#### Andmete valmisolek")
    st.caption(
        "Kontrollime, kas ajaloolistest korjetest saab moodustada päris õppimisnäited: "
        "sama põllu kahe järjestikuse korje vahel peab olema täielik mõõdetud ilm. "
        "Tänane pooleliolev korjepäev ei blokeeri õppimist."
    )

    readiness_start = date(TODAY.year, 7, 1)
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
        for field_name in ("a", "b", "c", "xl", "total"):
            if row.get(field_name) is None:
                harvest_problems.append(f"{day_str} põld {field_no}: {field_name.upper()} puudub")

        if harvest_day is not None and field_no_int is not None:
            parsed_rows.append((field_no_int, harvest_day, row))

    missing_fields = [f for f in range(1, 15) if f not in represented_fields]

    # Ilma baasnõue: 01.07 kuni viimase täieliku korjepäevani peab mõõdetud ilm olema 100% täielik.
    weather_missing = []
    weather_rows = []
    weather_by_day = {}
    required_weather = (
        "temp_min_c", "temp_max_c", "wind_avg_ms", "radiation_mj_m2",
        "humidity_avg_pct", "precipitation_mm", "et0_mm",
    )
    if last_complete_harvest and last_complete_harvest >= readiness_start:
        weather_missing = db.get_incomplete_measured_dates(readiness_start, last_complete_harvest)
        weather_rows = db.get_weather_rows(readiness_start, last_complete_harvest)
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
    if last_complete_harvest:
        for field_no, harvest_day, row in parsed_rows:
            if harvest_day <= last_complete_harvest:
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

    weather_ready = bool(last_complete_harvest) and not weather_missing
    harvest_ready = bool(last_complete_harvest) and not harvest_problems and not missing_fields
    sample_ready = bool(usable_samples) and not incomplete_samples
    training_ready = weather_ready and harvest_ready and sample_ready

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Õppimisnäiteid", len(usable_samples))
    m2.metric("Ilmaauguga näiteid", len(incomplete_samples))
    m3.metric("Põlde esindatud", f"{len(represented_fields)}/14")
    m4.metric("Õppe piir", last_complete_harvest.strftime("%d.%m") if last_complete_harvest else "—")

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
        if last_complete_harvest is None:
            st.error("🔴 Täielikku 3/3 korjepäeva ei leitud.")
        elif weather_ready:
            days_count = (last_complete_harvest - readiness_start).days + 1
            st.success(
                f"🟢 Mõõdetud ilm 01.07–{last_complete_harvest.strftime('%d.%m')} on täielik: "
                f"{days_count}/{days_count} päeva."
            )
        else:
            expected = (last_complete_harvest - readiness_start).days + 1
            ok_count = max(0, expected - len(weather_missing))
            st.error(
                f"🔴 Mõõdetud ilm 01.07–{last_complete_harvest.strftime('%d.%m')}: "
                f"{ok_count}/{expected} päeva valmis."
            )
            if weather_missing:
                missing_text = ", ".join(d.strftime("%d.%m") for d in weather_missing[:20])
                if len(weather_missing) > 20:
                    missing_text += f" … +{len(weather_missing) - 20}"
                st.caption(f"Puudulikud päevad: {missing_text}")

    with c2:
        st.markdown("##### Korjed")
        if harvest_ready:
            st.success("🟢 Korjeread korras ja kõik 14 põldu on ajaloos esindatud.")
        else:
            if missing_fields:
                st.error("🔴 Ajaloost puuduvad põllud: " + ", ".join(map(str, missing_fields)))
            if harvest_problems:
                st.error(f"🔴 Korjeandmetes leiti {len(harvest_problems)} probleemset välja/rida.")
                with st.expander("Näita korjeandmete probleeme"):
                    for problem in harvest_problems[:100]:
                        st.write("• " + problem)

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
        for _row in sorted(_rows, key=lambda r: str(r.get("harvest_date") or "")):
            try:
                _d = date.fromisoformat(str(_row.get("harvest_date")))
                _a = float(_row.get("a")); _b = float(_row.get("b")); _c = float(_row.get("c"))
            except (TypeError, ValueError):
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

        daily_mean_t = [(_n(w.get("temp_min_c")) + _n(w.get("temp_max_c"))) / 2 for w in window_weather]
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
            mt = [(_n(w.get("temp_min_c")) + _n(w.get("temp_max_c"))) / 2 for w in tail]
            return {
                f"T viim{n}": sum(mt) / len(mt),
                f"Rad viim{n}": sum(_n(w.get("radiation_mj_m2")) for w in tail),
                f"Sade viim{n}": sum(_n(w.get("precipitation_mm")) for w in tail),
                f"ET0 viim{n}": sum(_n(w.get("et0_mm")) for w in tail),
                f"Niiskus viim{n}": sum(_n(w.get("humidity_avg_pct")) for w in tail) / len(tail),
            }

        tail1 = _tail_weather(1)
        tail2 = _tail_weather(2)
        tail3 = _tail_weather(3)

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
            "Radiatsioon Σ": sum(rad),
            "Radiatsioon/p": sum(rad) / len(rad),
            "Sademed Σ": sum(rain),
            "Niiskus kesk": sum(hum) / len(hum),
            "ET0 Σ": sum(et0),
            "Tuul kesk": sum(wind) / len(wind),
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
            "T kesk", "Radiatsioon Σ", "Radiatsioon/p", "Sademed Σ",
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

        st.markdown("##### A+B+C põhimudel + eraldi XL-komponent")
        st.caption(
            "Põhimudel ennustab ainult A+B+C saaki, mis on taime tootmise objektiivsem osa. "
            "Baasmudel kasutab ilma, korjeintervalli, hooaja faasi ja põllu identiteeti; eelmine saak ei ole kohustuslik sisend. "
            "XL prognoositakse eraldi mürasema korjejäägi komponendina. Mõlemat hinnatakse ajaliselt ausa walk-forward testiga."
        )

        # Baasmudel on teadlikult puhas bioloogiline mudel: ilm + kasvuaeg + põld + hooaja faas.
        # Eelmise korje saak EI ole baasmudeli kohustuslik sisend; Jäljeotsija võib selle
        # eraldi kandidaadina sisse lubada ainult siis, kui aus walk-forward test tõestab kasu.
        base_cont_cols = [
            "Intervall p", "Hooajapäev", "T kesk", "Radiatsioon Σ", "Radiatsioon/p",
            "Sademed Σ", "Niiskus kesk", "ET0 Σ", "Tuul kesk",
        ]
        model_df = training_df.copy().sort_values(["Kuupäev", "Põld"]).reset_index(drop=True)
        model_df["Hooajapäev"] = pd.to_datetime(model_df["Kuupäev"]).map(
            lambda d: (d.date() - date(2026, 7, 1)).days
        )
        fields = model_df["Põld"].astype(int).to_numpy()
        dates = pd.to_datetime(model_df["Kuupäev"]).dt.date.to_numpy()
        X_base = model_df[base_cont_cols].astype(float).to_numpy()

        y_abc = pd.to_numeric(model_df["ABC saak"], errors="coerce").to_numpy(dtype=float)
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

        def _ridge_walk_predict(target, extra_arrays, train_idx, test_idx, alpha=10.0, floor_zero=True):
            Xtr, Xte, _, _, _ = _build_ridge_design(train_idx, test_idx, extra_arrays)
            penalty = np.eye(Xtr.shape[1]) * alpha
            penalty[0, 0] = 0.0
            beta = np.linalg.pinv(Xtr.T @ Xtr + penalty) @ Xtr.T @ target[train_idx]
            values = Xte @ beta
            return np.maximum(values, 0.0) if floor_zero else values

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
            abc_predictions[test_idx] = _ridge_walk_predict(y_abc, [], train_idx, test_idx)
            # XL-mudel saab kasutada varasemaid XL-e ning eelmise korje ABC/kogusaaki.
            xl_predictions[test_idx] = _ridge_walk_predict(
                y_xl, [raw_xl1, raw_xl2, raw_prev_abc, raw_prev_total], train_idx, test_idx
            )

        # C/B on eraldi kvaliteedimudel. Õpime log(C/B), et prognoos jääks alati positiivseks.
        cb_predictions = np.full(len(model_df), np.nan, dtype=float)
        for test_day in sorted(set(dates)):
            test_idx = np.where(dates == test_day)[0]
            train_idx = np.where((dates < test_day) & np.isfinite(log_y_cb))[0]
            if len(train_idx) < min_train_rows:
                continue
            log_pred = _ridge_walk_predict(log_y_cb, [raw_cb1], train_idx, test_idx, floor_zero=False)
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
            "Viimase 1 päeva ilm": ["T viim1", "Rad viim1", "Sade viim1", "ET0 viim1", "Niiskus viim1"],
            "Viimase 2 päeva ilm": ["T viim2", "Rad viim2", "Sade viim2", "ET0 viim2", "Niiskus viim2"],
            "Viimase 3 päeva ilm": ["T viim3", "Rad viim3", "Sade viim3", "ET0 viim3", "Niiskus viim3"],
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
                preds[test_idx] = _ridge_walk_predict(y, extra_arrays, train_idx, test_idx)
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
        # C/B kvaliteedimootor — eraldi champion, ei mõjuta A+B+C championit
        # -------------------------------------------------------------------------
        st.markdown("##### C/B kvaliteedimootor")
        st.caption(
            "C/B prognoositakse eraldi. Baasmudel kasutab korjeintervalli, ilma ja eelmist usaldusväärset C/B suhet. "
            "Jäljeotsija võib lisada ainult ajaliselt stabiilselt kasulikke tunnuseid."
        )

        valid_cb_base = np.isfinite(cb_predictions) & np.isfinite(y_cb)
        cb_base_mae = float(np.mean(np.abs(cb_predictions[valid_cb_base] - y_cb[valid_cb_base]))) if valid_cb_base.any() else None

        cb_candidate_groups = {
            "C/B mälu 2 korjet": ["C/B -2"],
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
                extra_arrays = [raw_cb1] + [raw_extra[:, j] for j in range(raw_extra.shape[1])]
                log_pred = _ridge_walk_predict(log_y_cb, extra_arrays, train_idx, test_idx, alpha=alpha, floor_zero=False)
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

        def _fit_full_generic(target, extra_arrays, alpha=10.0):
            idx = np.where(np.isfinite(target))[0]
            Xtr, _, means, scales, fills = _build_ridge_design(idx, idx, extra_arrays)
            penalty = np.eye(Xtr.shape[1]) * alpha
            penalty[0, 0] = 0.0
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

        champion_extra_arrays = [
            pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float) for c in champion_cols
        ]
        full_abc_base_model = _fit_full_generic(y_abc, [])
        full_abc_model = _fit_full_generic(y_abc, champion_extra_arrays)
        biological_load_feature_names = {
            c for cols in biological_load_candidate_groups.values() for c in cols
        }
        champion_uses_biological_load = any(c in biological_load_feature_names for c in champion_cols)
        full_xl_model = _fit_full_generic(y_xl, [raw_xl1, raw_xl2, raw_prev_abc, raw_prev_total])
        cb_champion_extra_arrays = [
            pd.to_numeric(model_df[c], errors="coerce").to_numpy(dtype=float) for c in cb_champion_cols
        ]
        full_cb_model = _fit_full_generic(log_y_cb, [raw_cb1] + cb_champion_extra_arrays)

        # -------------------------------------------------------------------------
        # 9 päeva ette: A+B+C + eraldi XL
        # -------------------------------------------------------------------------
        st.markdown("##### 9 päeva saagiprognoos")
        st.caption(
            f"A+B+C kasutab tänast champion-mootorit: {champion_name}. Baasmudel põhineb ilmal, intervallil, "
            f"hooaja faasil ja põllu identiteedil. Tõestatud normaliseeritud bioloogiline koormus võib baasi korrigeerida, "
            f"kuid toores eelmine saak, saagitrend ja muud korjeajaloo mälutunnused ei saa prognoosi ankurdada. "
            f"C/B kasutab eraldi championit: {cb_champion_name}. XL lisatakse eraldi korjejäägi komponendina. "
            "Korjevahemiku möödunud päevadel kasutatakse mõõdetud ilma ja tulevastel päevadel 9 päeva ilmaprognoosi."
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
            "temp_min_c", "temp_max_c", "wind_avg_ms", "radiation_mj_m2",
            "humidity_avg_pct", "precipitation_mm", "et0_mm",
        ]

        def _nearest_weather_value(day_value, feature):
            candidates = []
            for delta in range(1, 4):
                for dd in (day_value - timedelta(days=delta), day_value + timedelta(days=delta)):
                    row = all_weather_by_day.get(dd.isoformat())
                    if row and row.get(feature) is not None:
                        try:
                            candidates.append(float(row.get(feature)))
                        except (TypeError, ValueError):
                            pass
                if candidates:
                    break
            return float(np.mean(candidates)) if candidates else None

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
            mean_t = [(r["temp_min_c"] + r["temp_max_c"]) / 2 for r in rows]
            wx = {
                "Intervall p": (target_day - previous_day).days,
                "Hooajapäev": (target_day - date(2026, 7, 1)).days,
                "T kesk": float(np.mean(mean_t)),
                "Radiatsioon Σ": float(np.sum([r["radiation_mj_m2"] for r in rows])),
                "Radiatsioon/p": float(np.mean([r["radiation_mj_m2"] for r in rows])),
                "Sademed Σ": float(np.sum([r["precipitation_mm"] for r in rows])),
                "Niiskus kesk": float(np.mean([r["humidity_avg_pct"] for r in rows])),
                "ET0 Σ": float(np.sum([r["et0_mm"] for r in rows])),
                "Tuul kesk": float(np.mean([r["wind_avg_ms"] for r in rows])),
            }
            for n in (1, 2, 3):
                tail = rows[-min(n, len(rows)):]
                tvals = [(r["temp_min_c"] + r["temp_max_c"]) / 2 for r in tail]
                wx[f"T viim{n}"] = float(np.mean(tvals))
                wx[f"Rad viim{n}"] = float(np.sum([r["radiation_mj_m2"] for r in tail]))
                wx[f"Sade viim{n}"] = float(np.sum([r["precipitation_mm"] for r in tail]))
                wx[f"ET0 viim{n}"] = float(np.sum([r["et0_mm"] for r in tail]))
                wx[f"Niiskus viim{n}"] = float(np.mean([r["humidity_avg_pct"] for r in tail]))
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
                if k.startswith(("T viim", "Rad viim", "Sade viim", "ET0 viim", "Niiskus viim")):
                    values[k] = v
            return [values.get(c) for c in champion_cols]

        def _cb_champion_feature_values(state, wx):
            values = {
                "C/B -2": state.get("cb_prev"),
                "Eelmine2 ABC": state.get("abc_prev"),
                "ABC trend": (state.get("abc") - state.get("abc_prev")) if state.get("abc") is not None and state.get("abc_prev") is not None else None,
                "XL -1": state.get("xl"),
                "XL -2": state.get("xl_prev"),
                "XL osakaal -1": (state.get("xl") / state.get("total")) if state.get("xl") is not None and state.get("total") not in (None, 0) else None,
                "XL osakaal -2": (state.get("xl_prev") / state.get("total_prev")) if state.get("xl_prev") is not None and state.get("total_prev") not in (None, 0) else None,
            }
            for k, v in wx.items():
                if k.startswith(("T viim", "Rad viim", "Sade viim", "ET0 viim", "Niiskus viim")):
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
                abc_pred = _predict_full_generic(full_abc_base_model, field_no, base_values, [])
                abc_mode = "weather-first fallback"
            else:
                abc_pred = _predict_full_generic(
                    full_abc_model, field_no, base_values,
                    _champion_feature_values(state, wx),
                )
                abc_mode = champion_name
            xl_pred = _predict_full_generic(
                full_xl_model, field_no, base_values,
                [state.get("xl"), state.get("xl_prev"), state.get("abc"), state.get("total")],
            )
            cb_log_pred = _predict_full_generic(
                full_cb_model, field_no, base_values,
                [state.get("cb")] + _cb_champion_feature_values(state, wx),
                floor_zero=False,
            )
            cb_pred = float(np.exp(np.clip(cb_log_pred, np.log(0.10), np.log(10.0))))
            return {
                "abc": abc_pred, "xl": xl_pred, "cb": cb_pred, "total": abc_pred + xl_pred,
                "interval": wx["Intervall p"], "estimated_days": estimated_days or set(), "abc_mode": abc_mode,
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
            if d <= TODAY:
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
        today_plan = _planned_fields_for_day(TODAY, today_rows_live, harvest_rows)
        today_actual = {int(r.get("field_no")): r for r in today_rows_live if r.get("field_no") is not None}

        internal_today = []
        for f in today_plan:
            actual = today_actual.get(int(f))
            if actual:
                try:
                    old = field_state.get(int(f))
                    # harvest_rows sisaldab tavaliselt juba tänast salvestatud rida; ära lisa
                    # sama tegelikku korjet koormusajalukku teist korda.
                    if old and old.get("date") == TODAY and old.get("source") == "tegelik":
                        continue
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
            if not prev:
                continue
            nowcast = _predict_one(int(f), prev, TODAY)
            if nowcast:
                field_state[int(f)] = {
                    "date": TODAY, "abc": nowcast["abc"], "abc_prev": prev.get("abc"),
                    "xl": nowcast["xl"], "xl_prev": prev.get("xl"),
                    "cb": nowcast["cb"], "cb_prev": prev.get("cb"),
                    "total": nowcast["total"], "total_prev": prev.get("total"),
                    "load_index": None, "overload": None, "load2_index": None,
                    "peak": None, "peak_prev": prev.get("peak"),
                    "source": "prognoos",
                }
                internal_today.append((int(f), nowcast["total"]))

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
        MODEL_VERSION = "v6.3-weather-load-champion-v5"
        forecast_payloads = []
        for target_day, rows_day in forecast_days:
            for row in rows_day:
                if row.get("A+B+C") is None or row.get("XL") is None or row.get("Kokku") is None:
                    continue
                forecast_payloads.append({
                    "forecast_date": TODAY.isoformat(),
                    "target_date": target_day.isoformat(),
                    "field_no": int(row["Põld"]),
                    "lead_days": (target_day - TODAY).days,
                    "abc_forecast": float(row["A+B+C"]),
                    "cb_forecast": float(row["C/B"]) if row.get("C/B") is not None else None,
                    "xl_forecast": float(row["XL"]),
                    "total_forecast": float(row["Kokku"]),
                    "interval_days": row.get("Intervall"),
                    "basis": f"{row.get('Alus') or ''}; champion={champion_name}; cb_champion={cb_champion_name}",
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

        if internal_today:
            st.caption("Tänase poolelioleva korje sisemine tööprognoos: " + ", ".join(f"põld {f} ≈ {p:.1f}" for f,p in internal_today) + ". Tegelik kirje asendab selle automaatselt.")
        if any_weather_imputation:
            st.warning("⚠️ Puuduva mõõdetud ilma väärtusi täideti ajutiselt lähimate päevade keskmisega: " + ", ".join(sorted(d.strftime("%d.%m") for d in any_weather_imputation)) + ".")

        for target_day, rows_day in forecast_days:
            vals = [r["Kokku"] for r in rows_day if r["Kokku"] is not None]
            total_day = sum(vals) if len(vals) == 3 else None
            total_text = f"{_fmt(total_day)} kasti" if total_day is not None else "prognoos puudulik"
            lead = (target_day - TODAY).days
            if lead >= 6:
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
            st.dataframe(day_df.style.format({
                "A+B+C": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "C/B": lambda v: "—" if pd.isna(v) else f"{float(v):.2f}",
                "XL": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Kokku": lambda v: "—" if pd.isna(v) else f"{float(v):.1f}",
                "Intervall": lambda v: "—" if pd.isna(v) else f"{int(v)} p",
            }), use_container_width=True, hide_index=True)

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

with tabs[4]:
    st.subheader("Mootori tähelepanekud")
    st.info("Mootor hakkab hiljem mustreid kuvama, kuid ei muuda mudelit ise.")
