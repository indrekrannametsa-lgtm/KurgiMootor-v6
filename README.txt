KurgiMootor V6.1 – kontrollitud ilmaosa

1. Laadi GitHubi kogu kaust V6/kurgimootor_v6 või asenda selles failid:
   - app.py
   - core.py
   - db.py
   - requirements.txt

2. Käivita Supabase SQL Editoris fail supabase_weather_patch.sql.
   Skript loob või täiendab weather_daily ja app_settings tabelid ning tagab,
   et weather_date sobib upsert-võtmeks.

3. Streamlit rakenduses ava „Ilm“ ja vajuta esmalt „Testi ilmaallikaid“.
   See ei kirjuta veel andmebaasi. Eduka testi korral peavad kõik neli loendurit
   olema suuremad kui 0 ja ok peab olema true.

4. Alles seejärel vajuta „Uuenda ilm kohe“.

Oluline loogika:
- weather_daily sisaldab ühte rida kuupäeva kohta.
- Tänane/tulevane rida on forecast.
- Kui päev on möödunud, asendatakse sama kuupäeva prognoos measured-reaga.
