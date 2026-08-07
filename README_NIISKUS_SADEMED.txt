KurgiMootor V6.1 – täielik ilmaandmestik alates 01.07.2026

Lisatud Pärnu ametlikest kliimaandmetest:
- ööpäeva keskmine suhteline õhuniiskus: DRH08
- ööpäeva sademete summa: DPREC

Juba olemas:
- õhutemperatuur: TA (tunnid -> päeva min/max)
- tuul: WS10M (tunnid -> päeva keskmine)
- globaalradiatsioon: DRQS

OLULINE JÄRJEKORD:
1. Käivita Supabase SQL Editoris supabase_weather_patch.sql.
2. Alles siis asenda GitHubi harus weather-fix-v6-1 kaustas V6/kurgimootor_v6 failid app.py, core.py ja db.py.
3. Streamlitis Reboot.
4. Esimesel käivitusel loeb automaatika uuesti kõik 01.07.2026 alates olevad mõõdetud päevad, millel niiskus või sademed puuduvad.

Prognoosis lisatakse samuti õhuniiskuse päeva keskmine ja sademete summa Open-Meteost.
