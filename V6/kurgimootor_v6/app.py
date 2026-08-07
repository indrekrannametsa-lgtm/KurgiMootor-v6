from datetime import date, timedelta

import pandas as pd
import streamlit as st

import db
from core import WeatherService

TODAY = date.today()
st.set_page_config(page_title="KurgiMootor V6.2", page_icon="🥒", layout="wide")


def _n(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


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
    def _format_field_value(value, digits=1):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        try:
            return _fmt(float(value), digits)
        except (TypeError, ValueError):
            return str(value)

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

st.title("KurgiMootor V6.2")
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

    # Järgmise korje vaikimisi valikud. Pärast salvestust liiguvad need automaatselt edasi.
    # form_version annab pärast salvestust vormiväljadele uued võtmed, et Streamlit
    # võtaks päriselt kasutusele uued vaikimisi väärtused.
    default_field = int(st.session_state.get("next_harvest_field", 1))
    default_order = int(st.session_state.get("next_harvest_order", 1))
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
            previous_day = items[idx - 1][0]
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

    st.markdown("##### Mida mudel hiljem sellest kasutab")
    st.write(
        "Iga õppimisnäite siht on konkreetse põllu järgmine saak. Sisenditesse saab sellest ilmavahemikust "
        "arvutada näiteks temperatuuri, radiatsiooni, sademete, õhuniiskuse ja ET0 summad/keskmised ning "
        "korjeintervalli. Eelmise korje saak võib olla üks lisatunnus, kuid ei ole prognoosi põhialus."
    )

    if last_complete_harvest and latest_harvest and latest_harvest > last_complete_harvest:
        st.caption(
            "Uuem pooleliolev korjepäev jääb õppimisest ajutiselt välja, kuni päeva korjeplokk on täielik."
        )

    st.divider()
    st.info("Järgmine samm pärast rohelist valmisolekut: ehitada nendest õppimisnäidetest esimene põllupõhine kogusaagi mudel.")

with tabs[4]:
    st.subheader("Mootori tähelepanekud")
    st.info("Mootor hakkab hiljem mustreid kuvama, kuid ei muuda mudelit ise.")
