from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import db
from core import WeatherService

TODAY = datetime.now(ZoneInfo("Europe/Tallinn")).date()
st.set_page_config(page_title="KurgiMootor V6.1", page_icon="🥒", layout="wide")

try:
    db.ensure_default_plan(TODAY)
except Exception as exc:
    st.error(f"Andmebaasi viga: {exc}")
    st.stop()

st.title("KurgiMootor V6.1")
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
                for field_id, values in payload.items():
                    interval, a, b, c, xl = values
                    if a + b + c + xl > 0:
                        db.save_harvest(TODAY, field_id, interval, a, b, c, xl)
                        count += 1
                st.success(f"Salvestatud {count} põllu korje.") if count else st.warning("Kõik kogused olid nullid.")
                st.rerun()

with tabs[1]:
    st.subheader("Korjeajalugu")
    rows = db.get_harvest_history()
    if rows:
        df = pd.DataFrame(rows)
        df.columns = ["Kuupäev", "Põld", "Intervall", "A", "B", "C", "XL", "Kokku"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Korjeid pole veel sisestatud.")

with tabs[2]:
    st.subheader("Ilmaklots")
    st.caption("Mõõdetud temperatuur ja tuul Häädemeestelt, globaalradiatsioon Pärnust.")
    counts = db.get_weather_counts()
    a, b, c = st.columns(3)
    a.metric("Mõõdetud päevi", counts["measured"])
    b.metric("Rohelisi päevi", counts["checked"])
    c.metric("Prognoosipäevi", counts["forecast"])
    st.caption(f"Viimane uuendus: {db.get_app_setting('weather_last_refresh_at', '—')}")
    error = db.get_app_setting("weather_last_error", "")
    if error:
        st.error(error)
    test_col, refresh_col = st.columns(2)
    if test_col.button("Testi ilmaallikaid"):
        with st.spinner("Kontrollin Häädemeeste, Pärnu ja prognoosi allikaid..."):
            try:
                test = WeatherService().test_sources(TODAY)
                if test["ok"]:
                    st.success("Kõik neli vajalikku ilmaallikat vastasid.")
                else:
                    st.warning("Vähemalt üks ilmaallikas ei tagastanud andmeid.")
                st.json(test)
            except Exception as exc:
                st.error(f"Ilmaallikate test ebaõnnestus: {exc}")

    if refresh_col.button("Uuenda ilm kohe"):
        with st.spinner("Laen mõõteandmeid ja 9 päeva prognoosi..."):
            result = WeatherService().safe_refresh_all(TODAY)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("Ilmaandmed uuendatud.")
            st.rerun()

    rows = db.get_weather_rows(TODAY - timedelta(days=30), TODAY + timedelta(days=8))
    measured_rows = [r for r in rows if r.get("data_kind") == "measured"][-14:][::-1]
    forecast_rows = [r for r in rows if r.get("data_kind") == "forecast"]

    st.subheader("Viimased mõõdetud päevad")
    measured_display = [{
        "Kuupäev": r["weather_date"],
        "Min °C": r.get("temp_min_c"),
        "Max °C": r.get("temp_max_c"),
        "Tuul m/s": r.get("wind_avg_ms"),
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
