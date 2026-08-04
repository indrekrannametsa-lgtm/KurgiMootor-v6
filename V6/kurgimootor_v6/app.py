from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from core import BASE_INTERVALS, KurgiDB, WeatherService

DB_PATH = Path(__file__).with_name("kurgimootor_v6.db")
db = KurgiDB(DB_PATH)
TODAY_DATE = date.today()
TODAY = TODAY_DATE.isoformat()
db.ensure_default_plan(TODAY)

st.set_page_config(page_title="KurgiMootor V6", page_icon="🥒", layout="wide")

# At most one automatic update attempt per calendar day.
weather_service = WeatherService(db)
auto_weather_result = weather_service.safe_refresh_all(TODAY_DATE)

st.title("KurgiMootor V6")
st.caption("Saagi ennustamise tööriist. Avaleht on töövoog, mitte ilmarakendus.")

tabs = st.tabs(["Täna", "Korjed", "Ilm", "Prognoos", "Mootori tähelepanekud"])

with tabs[0]:
    st.subheader("Täna korjatavad põllud")
    current = db.plan_for(TODAY)
    current_ids = [r["field_id"] for r in current]

    for idx, row in enumerate(current):
        saved = db.harvest_for(TODAY, row["field_id"])
        interval = saved["interval_days"] if saved else db.interval_days(
            row["field_id"], TODAY, BASE_INTERVALS[idx % 3]
        )
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown(f'### {row["name"]}')
        c2.metric("Korjeintervall", f"{interval} p")
        if saved:
            c3.success("Salvestatud")
        elif c3.button("Eemalda", key=f'remove_{row["field_id"]}'):
            db.replace_plan(TODAY, [x for x in current_ids if x != row["field_id"]])
            st.rerun()

    remaining = [f for f in db.all_fields() if f["id"] not in current_ids]
    if remaining:
        labels = {f["name"]: f["id"] for f in remaining}
        selected = st.selectbox("Lisa tänasesse plaani", list(labels))
        if st.button("Lisa põld"):
            db.replace_plan(TODAY, current_ids + [labels[selected]])
            st.rerun()

    if not current:
        st.info("Täna ei korjata. Lisa põld ainult siis, kui otsustate siiski korjata.")

    if current:
        st.divider()
        st.subheader("Sisesta tänased korjed")
        with st.form("harvest_form"):
            payload = {}
            for idx, row in enumerate(current):
                existing = db.harvest_for(TODAY, row["field_id"])
                interval = existing["interval_days"] if existing else db.interval_days(
                    row["field_id"], TODAY, BASE_INTERVALS[idx % 3]
                )
                st.markdown(f'**{row["name"]} — {interval} päeva**')
                cols = st.columns(4)
                payload[row["field_id"]] = (
                    interval,
                    cols[0].number_input("A", 0.0, step=0.1, value=float(existing["a"]) if existing else 0.0, key=f'a{row["field_id"]}'),
                    cols[1].number_input("B", 0.0, step=0.1, value=float(existing["b"]) if existing else 0.0, key=f'b{row["field_id"]}'),
                    cols[2].number_input("C", 0.0, step=0.1, value=float(existing["c"]) if existing else 0.0, key=f'c{row["field_id"]}'),
                    cols[3].number_input("XL", 0.0, step=0.1, value=float(existing["xl"]) if existing else 0.0, key=f'xl{row["field_id"]}'),
                )
            if st.form_submit_button("Salvesta korjed"):
                saved_count = 0
                for field_id, values in payload.items():
                    interval, a, b, c, xl = values
                    if a + b + c + xl > 0:
                        db.save_harvest(TODAY, field_id, interval, a, b, c, xl)
                        saved_count += 1
                if saved_count:
                    st.success(f"Salvestatud {saved_count} põllu korje.")
                else:
                    st.warning("Ühtegi korjet ei salvestatud, sest kõik kogused olid nullid.")
                st.rerun()

with tabs[1]:
    st.subheader("Korjeajalugu")
    st.caption("Ajalugu aitab vaadata ja õppida, kuid selle puudumine ei blokeeri prognoosi.")
    rows = db.harvest_rows()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df.columns = ["Kuupäev", "Põld", "Intervall", "A", "B", "C", "XL", "Kokku"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Korjeid pole veel sisestatud.")

with tabs[2]:
    st.subheader("Ilmaklots")
    st.caption("Ilm on mudeli sisend. Avalehel seda ei näidata.")

    measured, checked, forecast = db.weather_status()
    a, b, c = st.columns(3)
    a.metric("Mõõdetud päevi", measured)
    b.metric("Rohelisi päevi", checked)
    c.metric("Prognoosipäevi", forecast)

    last_at = db.get_setting("weather_last_refresh_at", "—")
    last_error = db.get_setting("weather_last_error", "")
    st.caption(f"Viimane automaatne uuendus: {last_at}")
    if last_error:
        st.error(f"Automaatne ilmateade: {last_error}")

    if st.button("Uuenda ilm kohe"):
        with st.spinner("Kontrollin mõõteandmeid ja 9 päeva prognoosi..."):
            result = weather_service.safe_refresh_all(TODAY_DATE, force=True)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("Ilmaandmed uuendatud.")
            st.rerun()

    season_start = date(TODAY_DATE.year, 7, 1)
    forecast_end = TODAY_DATE + timedelta(days=8)
    weather = db.weather_rows(season_start.isoformat(), forecast_end.isoformat())

    measured_display = []
    forecast_display = []
    for raw in weather:
        row = dict(raw)
        item = {
            "Kuupäev": row["weather_date"],
            "Temp °C": row["t_avg"],
            "Tuul m/s": row["wind_avg"],
            "Radiatsioon MJ/m²": row["radiation"],
            "Kontroll": row["check_message"],
        }
        if row["data_kind"] == "forecast":
            item["Olek"] = "🔵 Prognoos" if row["checked"] else "🔴 Vigane prognoos"
            forecast_display.append(item)
        else:
            item["Olek"] = "🟢 Kontrollitud" if row["checked"] else "🔴 Puudulik"
            measured_display.append(item)

    st.markdown("#### Viimased mõõdetud päevad")
    if measured_display:
        # Uusim päev üleval; näitame korraga viimaseid 14 päeva.
        latest_measured = list(reversed(measured_display))[:14]
        st.dataframe(
            pd.DataFrame(latest_measured),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Mõõdetud ilmaandmeid pole veel salvestatud.")

    st.markdown("#### 9 päeva prognoos")
    if forecast_display:
        st.dataframe(
            pd.DataFrame(forecast_display),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Prognoosi pole veel salvestatud. Vajuta „Uuenda ilm kohe“.")

    st.caption(
        "Mõõdetud: temperatuur ja tuul Häädemeestelt, globaalradiatsioon Pärnust. "
        "Prognoos: farmi piirkonna 9 päeva mudelprognoos."
    )

with tabs[3]:
    st.subheader("Prognoos")
    st.warning("Kontrollitud saagiprognoosi mudel ei ole veel ühendatud. V6 ei väljasta näidisprognoosi.")
    for idx, row in enumerate(db.plan_for(TODAY)):
        saved = db.harvest_for(TODAY, row["field_id"])
        interval = saved["interval_days"] if saved else db.interval_days(row["field_id"], TODAY, BASE_INTERVALS[idx % 3])
        st.write(f'• {row["name"]} — intervall {interval} päeva')

with tabs[4]:
    st.subheader("Mootori tähelepanekud")
    st.caption("Mootor otsib hiljem mustreid ja pakub ideid, kuid ei muuda mudelit ise.")
    if len(db.harvest_rows()) < 20:
        st.info("Tähelepanekute jaoks on veel vähe andmeid.")
