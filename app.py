from __future__ import annotations

from datetime import date
from typing import Dict, List

import streamlit as st

from db import (
    DatabaseError,
    add_plan_field,
    ensure_default_plan,
    get_all_fields,
    get_harvest_for_day,
    get_plan_for_day,
    get_used_interval,
    remove_plan_field,
    save_harvest,
    get_weather_for_day,
    get_weather_status,
    save_weather,
)

st.set_page_config(
    page_title="KurgiMootor V6.1",
    page_icon="🥒",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 1.4rem;}
    .field-row {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .field-name {font-size: 1.1rem; font-weight: 700;}
    .interval {font-size: 1.55rem; font-weight: 800;}
    .muted {opacity: .65;}
    </style>
    """,
    unsafe_allow_html=True,
)

TODAY = date.today()


def optional_number(container, label: str, current, **kwargs):
    value = None if current is None else float(current)
    return container.number_input(label, value=value, **kwargs)


def rerun() -> None:
    st.rerun()


def show_database_error(exc: Exception) -> None:
    st.error(
        "Ühendus ühise andmebaasiga ebaõnnestus. "
        "Kontrolli Streamliti Secrets seadeid ja Supabase'i tabelite olemasolu."
    )
    st.code(str(exc))


st.title("KurgiMootor V6.1")
st.caption("Puhas tööversioon: põllud, tänane korjeplaan, intervallid ja korjete sisestamine.")

try:
    ensure_default_plan(TODAY)
    plan = get_plan_for_day(TODAY)
    all_fields = get_all_fields()
except DatabaseError as exc:
    show_database_error(exc)
    st.stop()

tab_today, tab_weather, tab_history = st.tabs(["Täna", "Ilmaandmed", "Korjeajalugu"])

with tab_today:
    st.subheader("Täna korjatavad põllud")
    st.caption("Vaikimisi 3 põldu. Võid lisada, eemaldada või jätta päeva tühjaks.")

    if not plan:
        st.info("Täna ei ole ühtegi põldu korjeplaanis.")

    for row in plan:
        field_id = int(row["field_id"])
        field_name = str(row["field_name"])
        interval = int(row["interval_days"])

        col1, col2, col3 = st.columns([4, 2, 1.4])
        with col1:
            st.markdown(
                f'<div class="field-row"><div class="field-name">{field_name}</div>'
                '<div class="muted">Tänases korjeplaanis</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="field-row"><div class="interval">{interval} p</div>'
                '<div class="muted">korjeintervall</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            if st.button("Eemalda", key=f"remove_{field_id}", use_container_width=True):
                try:
                    remove_plan_field(TODAY, field_id)
                    rerun()
                except DatabaseError as exc:
                    show_database_error(exc)

    planned_ids = {int(row["field_id"]) for row in plan}
    available = [f for f in all_fields if int(f["id"]) not in planned_ids]

    if available:
        label_to_id = {str(f["name"]): int(f["id"]) for f in available}
        col_add1, col_add2 = st.columns([3, 1])
        with col_add1:
            chosen_name = st.selectbox(
                "Lisa tänasesse plaani",
                options=list(label_to_id.keys()),
                label_visibility="collapsed",
            )
        with col_add2:
            if st.button("Lisa põld", use_container_width=True):
                try:
                    add_plan_field(TODAY, label_to_id[chosen_name])
                    rerun()
                except DatabaseError as exc:
                    show_database_error(exc)

    st.divider()
    st.subheader("Sisesta tänased korjed")

    if plan:
        existing_by_field: Dict[int, dict] = {}
        try:
            for row in get_harvest_for_day(TODAY):
                existing_by_field[int(row["field_id"])] = row
        except DatabaseError as exc:
            show_database_error(exc)

        with st.form("harvest_form", clear_on_submit=False):
            values: Dict[int, List[float]] = {}

            for row in plan:
                field_id = int(row["field_id"])
                field_name = str(row["field_name"])
                existing = existing_by_field.get(field_id, {})

                st.markdown(f"#### {field_name}")
                c1, c2, c3, c4 = st.columns(4)
                values[field_id] = [
                    c1.number_input(
                        "A",
                        min_value=0.0,
                        step=0.1,
                        value=float(existing.get("a", 0.0)),
                        key=f"a_{field_id}",
                    ),
                    c2.number_input(
                        "B",
                        min_value=0.0,
                        step=0.1,
                        value=float(existing.get("b", 0.0)),
                        key=f"b_{field_id}",
                    ),
                    c3.number_input(
                        "C",
                        min_value=0.0,
                        step=0.1,
                        value=float(existing.get("c", 0.0)),
                        key=f"c_{field_id}",
                    ),
                    c4.number_input(
                        "XL",
                        min_value=0.0,
                        step=0.1,
                        value=float(existing.get("xl", 0.0)),
                        key=f"xl_{field_id}",
                    ),
                ]

            submitted = st.form_submit_button("Salvesta korjed", use_container_width=True)

        if submitted:
            try:
                saved = 0
                for row in plan:
                    field_id = int(row["field_id"])
                    a, b, c_value, xl = values[field_id]
                    if (a + b + c_value + xl) <= 0:
                        continue

                    used_interval = get_used_interval(TODAY, field_id)
                    save_harvest(
                        harvest_date=TODAY,
                        field_id=field_id,
                        interval_days=used_interval,
                        a=a,
                        b=b,
                        c=c_value,
                        xl=xl,
                    )
                    saved += 1

                if saved:
                    st.success(f"Salvestatud {saved} põllu korje.")
                else:
                    st.warning("Ühtegi positiivse kogusega korjet ei olnud salvestada.")
            except DatabaseError as exc:
                show_database_error(exc)
    else:
        st.info("Korjete sisestamiseks lisa tänasesse plaani vähemalt üks põld.")


with tab_weather:
    st.subheader("Ilmaandmed")
    st.caption(
        "Häädemeeste jaam on põhiallikas. Globaalradiatsioon võetakse Pärnu jaamast. "
        "Korjeandmed ei ole prognoosi avamise tingimus."
    )

    weather_day = st.date_input("Kuupäev", value=TODAY, key="weather_day")

    try:
        existing_weather = get_weather_for_day(weather_day) or {}
        status = get_weather_status(weather_day)
    except DatabaseError as exc:
        show_database_error(exc)
        existing_weather = {}
        status = {"is_complete": False, "missing_fields": []}

    if bool(status.get("is_complete")):
        st.success("Ilmaandmed on prognoosi jaoks täielikud.")
    else:
        missing = status.get("missing_fields") or []
        if missing:
            st.warning("Prognoosi jaoks puudub: " + ", ".join(str(x) for x in missing))
        else:
            st.warning("Selle kuupäeva ilmaandmed pole veel sisestatud.")

    with st.form("weather_form", clear_on_submit=False):
        st.markdown("#### Häädemeeste ilmajaam")
        c1, c2, c3 = st.columns(3)
        temp_avg = optional_number(c1, "Keskmine temperatuur, °C", existing_weather.get("temp_avg_c"), step=0.1)
        temp_min = optional_number(c2, "Miinimumtemperatuur, °C", existing_weather.get("temp_min_c"), step=0.1)
        temp_max = optional_number(c3, "Maksimumtemperatuur, °C", existing_weather.get("temp_max_c"), step=0.1)

        c1, c2, c3 = st.columns(3)
        humidity_avg = optional_number(c1, "Keskmine õhuniiskus, %", existing_weather.get("humidity_avg_pct"), min_value=0.0, max_value=100.0, step=0.1)
        humidity_min = optional_number(c2, "Minimaalne õhuniiskus, %", existing_weather.get("humidity_min_pct"), min_value=0.0, max_value=100.0, step=0.1)
        humidity_max = optional_number(c3, "Maksimaalne õhuniiskus, %", existing_weather.get("humidity_max_pct"), min_value=0.0, max_value=100.0, step=0.1)

        c1, c2 = st.columns(2)
        precipitation = optional_number(c1, "Sademed, mm", existing_weather.get("precipitation_mm"), min_value=0.0, step=0.1)
        dewpoint = optional_number(c2, "Keskmine kastepunkt, °C", existing_weather.get("dewpoint_avg_c"), step=0.1)

        st.markdown("#### Tuul")
        c1, c2, c3, c4 = st.columns(4)
        wind_avg = optional_number(c1, "Keskmine, m/s", existing_weather.get("wind_avg_ms"), min_value=0.0, step=0.1)
        wind_max = optional_number(c2, "Maksimum, m/s", existing_weather.get("wind_max_ms"), min_value=0.0, step=0.1)
        wind_gust = optional_number(c3, "Maks. puhang, m/s", existing_weather.get("wind_gust_ms"), min_value=0.0, step=0.1)
        wind_direction = optional_number(c4, "Suund, °", existing_weather.get("wind_direction_deg"), min_value=0.0, max_value=360.0, step=1.0)

        st.markdown("#### Muud näitajad")
        c1, c2 = st.columns(2)
        pressure = optional_number(c1, "Keskmine õhurõhk, hPa", existing_weather.get("pressure_avg_hpa"), min_value=0.0, step=0.1)
        sunshine = optional_number(c2, "Päikesepaiste kestus, h", existing_weather.get("sunshine_hours"), min_value=0.0, step=0.1)

        st.markdown("#### Pärnu ilmajaam")
        radiation = optional_number(st, "Globaalradiatsioon, MJ/m²", existing_weather.get("radiation_mj_m2"), min_value=0.0, step=0.01)
        notes = st.text_area("Märkused", value=str(existing_weather.get("notes") or ""))

        weather_submitted = st.form_submit_button("Salvesta ilmaandmed", use_container_width=True)

    if weather_submitted:
        try:
            save_weather(
                weather_day,
                {
                    "source_station": "Häädemeeste",
                    "temp_avg_c": temp_avg,
                    "temp_min_c": temp_min,
                    "temp_max_c": temp_max,
                    "humidity_avg_pct": humidity_avg,
                    "humidity_min_pct": humidity_min,
                    "humidity_max_pct": humidity_max,
                    "precipitation_mm": precipitation,
                    "wind_avg_ms": wind_avg,
                    "wind_max_ms": wind_max,
                    "wind_gust_ms": wind_gust,
                    "wind_direction_deg": wind_direction,
                    "pressure_avg_hpa": pressure,
                    "dewpoint_avg_c": dewpoint,
                    "sunshine_hours": sunshine,
                    "radiation_mj_m2": radiation,
                    "radiation_station": "Pärnu",
                    "notes": notes.strip() or None,
                },
            )
            st.success("Ilmaandmed salvestatud.")
            rerun()
        except DatabaseError as exc:
            show_database_error(exc)


with tab_history:
    st.subheader("Korjeajalugu")
    st.caption("Ajalugu on vaatamiseks ja mudeli õppimiseks. See ei blokeeri prognoosi.")

    try:
        from db import get_harvest_history

        history = get_harvest_history(limit=300)
        if not history:
            st.info("Korjeid pole veel salvestatud.")
        else:
            table = []
            for row in history:
                table.append(
                    {
                        "Kuupäev": row["harvest_date"],
                        "Põld": row["field_name"],
                        "Intervall": row["interval_days"],
                        "A": row["a"],
                        "B": row["b"],
                        "C": row["c"],
                        "XL": row["xl"],
                        "Kokku": row["total"],
                    }
                )
            st.dataframe(table, use_container_width=True, hide_index=True)
    except DatabaseError as exc:
        show_database_error(exc)
