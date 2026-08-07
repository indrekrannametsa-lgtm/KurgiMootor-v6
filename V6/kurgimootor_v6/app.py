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


def _daily_summary(rows):
    count = len(rows)
    a = sum(_n(r.get("a")) for r in rows)
    b = sum(_n(r.get("b")) for r in rows)
    c = sum(_n(r.get("c")) for r in rows)
    xl = sum(_n(r.get("xl")) for r in rows)
    total = sum(_n(r.get("total")) for r in rows)
    cb = c / b if b > 0 else None
    avg = total / count if count else 0.0
    return {
        "count": count,
        "a": a,
        "b": b,
        "c": c,
        "xl": xl,
        "total": total,
        "cb": cb,
        "avg": avg,
    }


def _field_table(rows):
    table = []
    for r in rows:
        b = _n(r.get("b"))
        c = _n(r.get("c"))
        table.append({
            "Põld": r.get("field_no"),
            "Järjekord": r.get("harvest_order"),
            "Intervall": r.get("interval_days"),
            "A": _n(r.get("a")),
            "B": b,
            "C": c,
            "XL": _n(r.get("xl")),
            "C/B": round(c / b, 2) if b > 0 else None,
            "Kokku": _n(r.get("total")),
        })
    return pd.DataFrame(table)


def _render_day_block(day_label, rows, show_quality=False):
    s = _daily_summary(rows)
    status = "3/3 · valmis" if s["count"] >= 3 else f"{s['count']}/3 · sisestatud"
    st.markdown(f"### {day_label} — päev kokku {_fmt(s['total'])}")
    st.caption(status)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("A / B", f"{_fmt(s['a'])} / {_fmt(s['b'])}")
    m2.metric("C / XL", f"{_fmt(s['c'])} / {_fmt(s['xl'])}")
    m3.metric("C/B", "—" if s["cb"] is None else _fmt(s["cb"], 2))
    m4.metric("Keskmine / põld", _fmt(s["avg"], 2))

    df = _field_table(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

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
    if today_rows:
        _render_day_block(TODAY.strftime("%d.%m.%Y"), today_rows)
    else:
        st.info("Tänaseid korjeid pole veel sisestatud. Ava ülevalt „Korjed“ ja sisesta põldude korjed.")

with tabs[1]:
    st.subheader("Korjed")
    st.caption("Andmebaasis hoitakse iga põllu korje eraldi. Äpis vaatame saaki eelkõige päevade kaupa.")

    # Järgmise korje vaikimisi valikud. Pärast salvestust liiguvad need automaatselt edasi.
    default_field = int(st.session_state.get("next_harvest_field", 1))
    default_order = int(st.session_state.get("next_harvest_order", 1))

    st.markdown("#### Lisa või paranda korje")
    with st.form("manual_harvest_form"):
        c1, c2, c3 = st.columns(3)
        entry_date = c1.date_input("Kuupäev", value=TODAY, key="manual_harvest_date")
        entry_field = c2.selectbox(
            "Põld",
            list(range(1, 15)),
            index=max(0, min(13, default_field - 1)),
            key="manual_harvest_field",
        )
        entry_order = c3.selectbox(
            "Järjekord",
            [1, 2, 3],
            index=max(0, min(2, default_order - 1)),
            key="manual_harvest_order",
        )
        q1, q2, q3, q4 = st.columns(4)
        entry_a = q1.number_input("A", 0.0, step=0.1, key="manual_a")
        entry_b = q2.number_input("B", 0.0, step=0.1, key="manual_b")
        entry_c = q3.number_input("C", 0.0, step=0.1, key="manual_c")
        entry_xl = q4.number_input("XL", 0.0, step=0.1, key="manual_xl")
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

                # Nullime koguseväljad ja eemaldame selectboxi hetkeseisu, et järgmise reruniga
                # võetaks ülal arvutatud uued vaikimisi väärtused.
                for key in ["manual_harvest_field", "manual_harvest_order", "manual_a", "manual_b", "manual_c", "manual_xl"]:
                    if key in st.session_state:
                        del st.session_state[key]

                st.success(f"Salvestatud: {entry_date} · põld {entry_field} · kokku {_fmt(total_preview)}")
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
                day_label = date.fromisoformat(day_str).strftime("%d.%m.%Y")
            except ValueError:
                day_label = day_str
            _render_day_block(day_label, day_rows, show_quality=True)
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
            st.dataframe(df, use_container_width=True, hide_index=True)
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
    st.info("Saagiprognoosi mootor ühendatakse pärast täieliku ilmaandmestiku kontrolli.")

with tabs[4]:
    st.subheader("Mootori tähelepanekud")
    st.info("Mootor hakkab hiljem mustreid kuvama, kuid ei muuda mudelit ise.")
