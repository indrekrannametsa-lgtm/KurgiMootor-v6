from datetime import date, timedelta

import pandas as pd
import streamlit as st

import db
from core import WeatherService

TODAY = date.today()
st.set_page_config(page_title="KurgiMootor V6.2", page_icon="🥒", layout="wide")

try:
    db.ensure_default_plan(TODAY)
except Exception as exc:
    st.error(f"Andmebaasi viga: {exc}")
    st.stop()

# Ilm kontrollitakse automaatselt üks kord päevas. Viga ei takista korjete kasutamist.
try:
    WeatherService().auto_refresh_if_needed(TODAY)
except Exception as exc:
    db.set_app_setting("weather_last_error", f"Automaatne ilmauuendus: {exc}")

st.title("KurgiMootor V6.2")
st.caption("Saagi ennustamise tööriist. Avaleht on töövoog, mitte ilmarakendus.")

tabs = st.tabs(["Täna", "Korjed", "Ilm", "Prognoos", "Mootori tähelepanekud"])

with tabs[0]:
    st.subheader("Täna korjatavad põllud")
    plan = db.get_plan_for_day(TODAY)
    harvested = {int(r["field_id"]): r for r in db.get_harvest_for_day(TODAY)}
    ids = [int(r["field_id"]) for r in plan]
    for row in plan:
        field_id = int(row["field_id"])
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown(f"### {row['field_name']}")
        c2.metric("Korjeintervall", f"{int(row['interval_days'])} p")
        if field_id in harvested:
            c3.success("Salvestatud")
        elif c3.button("Eemalda", key=f"remove_{field_id}"):
            db.remove_plan_field(TODAY, field_id)
            st.rerun()
    remaining = [f for f in db.get_all_fields() if int(f["id"]) not in ids]
    if remaining:
        labels = {f["name"]: int(f["id"]) for f in remaining}
        selected = st.selectbox("Lisa tänasesse plaani", list(labels))
        if st.button("Lisa põld"):
            db.add_plan_field(TODAY, labels[selected])
            st.rerun()

    if plan:
        st.divider()
        st.subheader("Sisesta tänased korjed")
        with st.form("harvest_form"):
            payload = {}
            for row in plan:
                field_id = int(row["field_id"])
                old = harvested.get(field_id, {})
                st.markdown(f"**{row['field_name']} — {int(row['interval_days'])} päeva**")
                cols = st.columns(4)
                payload[field_id] = (
                    int(row["interval_days"]),
                    cols[0].number_input("A", 0.0, step=0.1, value=float(old.get("a", 0)), key=f"a{field_id}"),
                    cols[1].number_input("B", 0.0, step=0.1, value=float(old.get("b", 0)), key=f"b{field_id}"),
                    cols[2].number_input("C", 0.0, step=0.1, value=float(old.get("c", 0)), key=f"c{field_id}"),
                    cols[3].number_input("XL", 0.0, step=0.1, value=float(old.get("xl", 0)), key=f"xl{field_id}"),
                )
            if st.form_submit_button("Salvesta korjed"):
                count = 0
                for order_no, (field_id, values) in enumerate(payload.items(), start=1):
                    interval, a, b, c, xl = values
                    if a + b + c + xl > 0:
                        db.save_harvest(TODAY, field_id, interval, a, b, c, xl, harvest_order=order_no)
                        count += 1
                st.success(f"Salvestatud {count} põllu korje.") if count else st.warning("Kõik kogused olid nullid.")
                st.rerun()

with tabs[1]:
    st.subheader("Korjed")
    st.caption("Korjeandmed salvestatakse Supabase'i harvests tabelisse. Üks rida = ühe põllu üks korje.")

    st.markdown("#### Lisa või paranda korje")
    with st.form("manual_harvest_form"):
        c1, c2, c3 = st.columns(3)
        entry_date = c1.date_input("Kuupäev", value=TODAY, key="manual_harvest_date")
        entry_field = c2.selectbox("Põld", list(range(1, 15)), key="manual_harvest_field")
        entry_order = c3.selectbox("Järjekord", [1, 2, 3], key="manual_harvest_order")
        q1, q2, q3, q4 = st.columns(4)
        entry_a = q1.number_input("A", 0.0, step=0.1, key="manual_a")
        entry_b = q2.number_input("B", 0.0, step=0.1, key="manual_b")
        entry_c = q3.number_input("C", 0.0, step=0.1, key="manual_c")
        entry_xl = q4.number_input("XL", 0.0, step=0.1, key="manual_xl")
        total_preview = entry_a + entry_b + entry_c + entry_xl
        st.caption(f"Kokku: {total_preview:.1f}")
        if st.form_submit_button("Salvesta korje"):
            if total_preview <= 0:
                st.warning("Korje kogus on 0. Sisesta vähemalt üks kogus.")
            else:
                db.save_harvest(entry_date, entry_field, 0, entry_a, entry_b, entry_c, entry_xl, harvest_order=entry_order)
                st.success(f"Salvestatud: {entry_date} · põld {entry_field} · kokku {total_preview:.1f}")
                st.rerun()

    st.divider()
    st.markdown("#### Korjeajalugu")
    rows = db.get_harvest_history()
    if rows:
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

    # Ilmaajalugu on äpis vabalt vaadeldav. Vaikimisi näitame kogu selle hooaja
    # mõõdetud ajalugu alates 1. juulist; see valik mõjutab ainult vaadet, mitte
    # mootorile salvestatud andmeid.
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
