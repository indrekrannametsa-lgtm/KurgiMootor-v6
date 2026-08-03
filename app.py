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

tab_today, tab_history = st.tabs(["Täna", "Korjeajalugu"])

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
