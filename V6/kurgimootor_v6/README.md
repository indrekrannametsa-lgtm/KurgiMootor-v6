# KurgiMootor V6

Puhas uus algus. V5 koodi ega vana ilmaloogikat ei ole üle võetud.

## Praegu töötab

- 14 püsivat põldu
- tänase korjeplaani automaatne 3 põllu algvalik
- põllu lisamine ja eemaldamine tänasest plaanist
- korjeintervalli kuvamine
- baasintervall 5–5–4, kui varasem korje puudub
- korjete sisestamine põllu kaupa: A, B, C, XL
- ühine SQLite andmebaas
- korjeajalugu
- automaatne ilmaklots:
  - mõõdetud temperatuur ja tuul Häädemeeste jaamast
  - mõõdetud globaalradiatsioon Pärnu jaamast
  - tänasest alates 9 päeva prognoosi farmi piirkonnale
  - roheline ainult siis, kui kõik kolm mõõdetud sisendit on olemas ja kontrolli läbinud
  - puudulik päev jääb punaseks
  - prognoosipäevad on sinised
- prognoos ei väljasta saagiprognoosi enne valideeritud mudeli ühendamist
- mootori tähelepanekute eraldi ala

## Käivitamine

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Kaks kasutajat

Käivita rakendus ühes arvutis või serveris. Mõlemad kasutajad avavad sama aadressi ning kasutavad sama `kurgimootor_v6.db` andmebaasi.

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Kontroll

```bash
python test_core.py
python -m unittest -v test_weather.py
```

## Oluline

Ilma uuendatakse rakenduse avamisel kõige rohkem üks kord päevas. Nupp **Uuenda ilm kohe** sunnib uue päringu. Võrgu- või API-vea korral jäävad olemasolevad andmed alles ja rakendus kuvab vea; puuduvat mõõtepäeva ei märgita roheliseks.
