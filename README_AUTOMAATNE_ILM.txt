KurgiMootor V6.1 – automaatne ilma uuendus

Asenda GitHubi harus weather-fix-v6-1 kaustas V6/kurgimootor_v6 need failid:
- app.py
- core.py
- db.py

requirements.txt ei muutunud, kuid on paketis kaasas.

Tööloogika:
- äpi avamisel kontrollitakse ilma automaatselt;
- automaatne uuendus toimub kõige rohkem üks kord kohaliku kalendripäeva jooksul;
- mõõdetud andmed laaditakse Pärnu jaamast kuni üle-eilseni;
- uuesti kirjutatakse ainult puuduvad või puudulikud mõõdetud päevad;
- 9 päeva prognoos uuendatakse kord päevas;
- mõõdetud andmete või prognoosi viga ei katkesta teise osa uuendamist;
- API vea korral ei hakka iga Streamliti rerun sama päringut kordama;
- nupp „Uuenda ilm kohe“ jääb käsitsi korduskatseks alles;
- Ilm lehel kuvatakse viimase uuenduse aeg, tulemus ja võimalik viga.

Supabase SQL-i ei ole vaja uuesti käivitada, kui app_settings tabel ja weather_daily veerud on juba olemas.
