# KurgiMootor V6.1

Puhas uus alus. Selles versioonis on ainult:

- Supabase kui ainus andmebaas
- 14 põldu
- tänane muudetav korjeplaan
- automaatne korjeintervall
- A, B, C ja XL sisestamine põllu kaupa
- korjeajalugu
- Häädemeeste ilmaandmed koos tuulega
- Pärnu globaalradiatsioon

Selles versioonis **ei ole** SQLite'i, prognoosi ega AI-mootorit. Ilmaandmete salvestamine ja täielikkuse kontroll on olemas.

## 1. Supabase'i skeem

Ava Supabase'is **SQL Editor**, kopeeri sinna kogu faili `supabase_schema.sql` sisu ja vajuta **Run**.

Skeem:

- loob vajalikud tabelid;
- lisab Põld 1 kuni Põld 14;
- loob päeva plaani ja intervallide funktsioonid;
- lubab tühja korjepäeva;
- salvestab iga korje juurde kasutatud intervalli.

## 2. Streamlit Secrets

Streamlit Community Cloudis:

**Manage app → Settings → Secrets**

Lisa:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
secret_key = "sb_secret_YOUR_SECRET_KEY"
```

Supabase'i uut secret key'd kasuta ainult Streamliti serverisaladuses. Ära lisa seda GitHubi.

## 3. GitHub

Repo juurkaustas või sinu praeguses rakenduse kaustas peavad olema:

- `app.py`
- `db.py`
- `requirements.txt`
- `supabase_schema.sql`

Main file path jääb sinu praeguse kaustastruktuuri korral:

```text
V6/kurgimootor_v6/app.py
```

## Tööloogika

- Uuel päeval loob süsteem vaikimisi kolm kõige kauem korjamata põldu.
- Kasutaja võib põlde lisada või eemaldada.
- Kui kõik põllud eemaldatakse, jääb päev tühjaks ega täitu uuesti automaatselt.
- Intervall saadakse viimase tegeliku korje kuupäevast.
- Kui varasemat korjet pole, kasutatakse baasjaotust 5–5–4.
- Korje salvestatakse põllu kaupa koos A, B, C, XL ja kasutatud intervalliga.


## Ilmaandmete loogika

- Põhiallikas on Häädemeeste ilmajaam.
- Salvestatakse temperatuur, õhuniiskus, sademed, tuul, õhurõhk, kastepunkt ja päikesepaiste kestus.
- Globaalradiatsioon salvestatakse Pärnu jaamast.
- Prognoosi jaoks loetakse kohustuslikuks keskmine/min/max temperatuur, keskmine õhuniiskus, sademed, keskmine tuul, maksimaalne puhang ja globaalradiatsioon.
- Korjeandmed on mudeli õppimiseks ega blokeeri prognoosi.
