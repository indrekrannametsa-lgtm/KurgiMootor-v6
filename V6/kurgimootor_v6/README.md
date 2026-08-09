# KurgiMootor V6.4

KurgiMootor on 14 põlluga avamaa kurgikasvatuse saagiprognoosi tööriist.  
Rakenduse põhieesmärk on prognoosida järgmisi korjeid põldude kaupa ning anda päevane koondprognoos.

## Põhimõtted

- Kõik 14 põldu hoitakse mudelis eraldi.
- Korjed sisestatakse põllu kaupa: A, B, C ja XL.
- A+B+C on põhiline kasvupotentsiaali prognoos.
- XL prognoositakse eraldi komponendina.
- C/B prognoositakse eraldi kvaliteedimudeliga.
- A+B+C baasmudel on weather-first:
  - ilm
  - korjeintervall
  - hooaja faas
  - põllu identiteet
- Toores eelmine saak ei ole järgmise korje ankur.
- Korjeajalugu kasutatakse õppimiseks ja seoste avastamiseks.
- Lisatunnus võib mudelisse pääseda ainult ajaliselt ausa walk-forward testi kaudu.

## Andmebaas

Rakendus kasutab Supabase'i.

Põhiandmed:

- korjed põldude kaupa
- mõõdetud ja prognoositud ilm
- rakenduse seaded
- saagiprognooside snapshotid (`yield_forecasts`)

Prognoosisnapshotid võimaldavad hiljem võrrelda varem tehtud prognoosi tegeliku korjega.

## Ilm

Mõõdetud ilm pärineb Pärnu jaamast.

Mudelis kasutatakse:

- Tmin
- Tmax
- keskmine tuul
- globaalradiatsioon
- õhuniiskus
- sademed
- ET0

ET0 arvutab KurgiMootor ise.

Tuleviku ilmaprognoos tuleb Open-Meteost ja ulatub 9 päeva ette.

Puuduva mõõdetud väärtuse korral võib mudeli töövaade kasutada ajutiselt kuni kolme varasema päeva keskmist. Andmebaasi ametlikku mõõterida selle tõttu ei muudeta.

## Prognoos

KurgiMootor teeb:

- tänase tööprognoosi
- 1–9 päeva saagiprognoosi
- põllupõhise A+B+C prognoosi
- eraldi XL prognoosi
- eraldi C/B prognoosi
- päevase koondprognoosi

Prognooside ajalugu salvestatakse `yield_forecasts` tabelisse.

Avalehel kuvatakse ka:

**MOOTORI TÄPSUS · 3P**

See mõõdab viimaste kuni kolme täieliku korjepäeva operatiivset prognoositäpsust. Tänast pooleliolevat päeva ei kasutata.

## Walk-forward test

Mudeli hindamine on ajaliselt aus.

Iga testitava korje puhul treenitakse mudel ainult andmetel, mis olid enne seda korjet olemas. Tuleviku infot ei tohi testirea tunnustesse sattuda.

Põhiline veamõõdik on MAE.

Madalam MAE on parem.

## Champion-mudel

A+B+C, XL ja C/B võivad kasutada eraldi champion-mudelit.

Champion valitakse walk-forward tulemuste põhjal. Uus kandidaat peab näitama piisavat ja ajaliselt stabiilset paranemist.

A+B+C weather-first baas jääb kasutusse seni, kuni mõni lubatud kandidaat tõestab parema tulemuse.

## Jäljeotsija

Jäljeotsija testib võimalikke lisatunnuseid, näiteks:

- viimase 1–3 päeva ilm
- temperatuurikäitumine
- tuule ja kuivuse koostoimed
- päevapikkus
- normaliseeritud bioloogiline koormus

Toored saagimälu tunnused võivad olla diagnostikas nähtavad, kuid ei saa A+B+C prognoosi ankurdada.

## Mootori enda avastused

KurgiMootor genereerib ka ise uusi tunnuseideid.

Avastusruum sisaldab muu hulgas:

- mittelineaarseid teisendusi
- koostoimeid
- suhtarve
- ajamuutusi
- temperatuurilävesid
- teise ringi kombinatsioone

Ideede genereerimine võib olla lai, kuid kasutuselevõtt on range.

Mootori enda leitud idee ei lähe automaatselt prognoosimudelisse.

Avastusmootor kasutab kahte ajaplokki:

1. vanem avastusplokk ideede leidmiseks;
2. hilisem kinnitusplokk tulemuse kontrollimiseks.

Kinnitusplokki ei kasutata idee valimiseks.

Mediaanist kõrvalekalde tüüpi tunnused arvutatakse ainult varasemate kuupäevade põhjal.

## Mootori tähelepanekud

Menüü **Mootori tähelepanekud** kuvab:

- praegu usaldatavad championid
- mootori enda avastused
- avastusruumi mahu
- kinnitatud ja uurimisjärgus leiud
- kandidaadid, mida hoitakse silma peal
- praegu kõrvale jäetud kandidaadid
- diagnostilised saagimälu signaalid

Tähelepanekute leht ei muuda mudelit ise.

## Hooaja faas

Hooaja faasi lähtepunkt on jooksva aasta **15. juuni**.

```python
SEASON_START = date(TODAY.year, 6, 15)
```

## Avastusmootori cache

Avastusmootori rasked 1. ja 2. ringi walk-forward katsed salvestatakse Streamliti sessioon-cache'i.

Kui õppimisandmestik ja champion ei muutu, kasutatakse olemasolevat avastustulemust.

Uus korje, muutunud ilm või muutunud champion muudab cache-võtit ja avastusmootor arvutatakse uuesti.

Prognoos ise ei tule sellest cache'ist.

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

Rakendus vajab Streamliti secrets-seadistuses Supabase'i ühenduse andmeid.

## Testid

V6.4 testid:

```bash
python -m unittest -v test_core.py
python -m unittest -v test_weather.py
python -m unittest -v test_model_rules.py
```

`test_model_rules.py` kaitseb muu hulgas järgmisi arhitektuurireegleid:

- A+B+C weather-first baas
- XL/C/B ilma automaatse toore eelkorje ankruta
- mediaanitunnused ainult minevikust
- eraldi avastus- ja kinnitusplokk
- 3P täpsus ei kasuta tänast päeva
- ilma fallback vaatab ainult varasematele päevadele
- prognoosisnapshotide võtmeväljad

## Oluline

- Ilmaandmete viga ei tohi korjeandmeid rikkuda.
- Mõõdetud ilma andmebaasirida ei muudeta hinnanguliseks ainult mudeli fallback'i tõttu.
- Prognoositud eelkorjet ei toideta bioloogilise koormuse sisendina järgmisse A+B+C prognoosi.
- Mootori enda avastused on kandidaadid, mitte automaatsed otsused.
- Töötavat weather-first prognoosiloogikat ei muudeta pelgalt ajaloolise sobivuse põhjal.
