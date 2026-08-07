KurgiMootor V6.1 – ET0 lisandus

Lisatud weather_daily.et0_mm.
ET0 arvutatakse FAO-56 Penman-Monteith päevase valemiga temperatuurist,
10 m tuulest (teisendatakse 2 m kõrgusele), globaalradiatsioonist ja
keskmisest suhtelisest õhuniiskusest. Sademed jäävad eraldi sisendiks.

Enne koodi paigaldamist käivita Supabase SQL Editoris supabase_weather_patch.sql.
Seejärel laadi V6/kurgimootor_v6 failid weather-fix-v6-1 harusse.
Automaatika loeb 01.07 alates uuesti päevad, mille ET0 on puudu.
