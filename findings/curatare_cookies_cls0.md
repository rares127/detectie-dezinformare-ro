# Curățare cookies cls0 — raport

**Scope:** DOAR tratament boilerplate cookie (Pas A = aruncare banner pur,
Pas B = curățare prefix banner + păstrare conținut real).

**NU s-au aplicat** încă: etichete vorbitor, titluri concatenate, filtrare
lungime, deduplicare. Acelea vin în pași separați.

## 1. Rezumat operații

| Operație | Număr propoziții |
|---|---|
| Input brut | 6,047 |
| Banner pur aruncat | 59 |
| Prefix curățat → conținut păstrat | 44 |
| Prefix curățat → prea scurt după (aruncat) | 9 |
| **Output** | **5,979** |

**Retenție:** 98.88%

## 2. Verificare bug-fix

Pattern vechi (buggy): `coo?kie` — matcha doar 0-1 `o` suplimentar.
Pattern nou: `co+kie` — prinde `cookie`, `coookie`, `cooookie` etc.

În raportul anterior (filtrare_corpus_cls0.md), Pasul 1a raporta
**0 propoziții modificate**. Acum avem **53 modificate**
(44 păstrate + 9 prea scurte).

## 3. Breakdown pe sursă

| Sursă | Input | Output | Retenție |
|---|---|---|---|
| digi24.ro | 3,297 | 3,229 | 97.94% |
| g4media.ro | 2,750 | 2,750 | 100.00% |

> **Verificare ipoteză:** întreg efectul ar trebui să fie pe digi24
> (g4media nu are cookie banners în scraping).

## 4. Exemple — banner pur aruncat

- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.

## 5. Exemple — prefix curățat (ÎNAINTE / DUPĂ)

> Verifică vizual că bucata rămasă e conținut real, nu fragment fără sens.

**Sursă:** digi24.ro (39w → 20w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media „Pentru a susține libertatea presei, am construit un oraș slavic, numit Voina, care înseamnă război în rusă”, a spus Mukka.
- DUPĂ:   Pentru a susține libertatea presei, am construit un oraș slavic, numit Voina, care înseamnă război în rusă”, a spus Mukka.

**Sursă:** digi24.ro (40w → 21w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media ISW spune că forțele rusești au continuat bombardamentele intense, atacurile cu rachete și atacurile aeriene de-a lungul întregii
- DUPĂ:   ISW spune că forțele rusești au continuat bombardamentele intense, atacurile cu rachete și atacurile aeriene de-a lungul întregii linii de front.

**Sursă:** digi24.ro (52w → 33w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media Londra notează că acum comanda SGF ar urma să fie preluată de generalul-colonel Serghei Surovikin, deoarece SGF continuă să joace
- DUPĂ:   Londra notează că acum comanda SGF ar urma să fie preluată de generalul-colonel Serghei Surovikin, deoarece SGF continuă să joace un rol central în ofensiva pe care armata rusă o desfășoară în Donbas.

**Sursă:** digi24.ro (41w → 22w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media „Astăzi am participat la o reuniune extraordinară a liderilor UE privind Ucraina, foarte utilă şi extrem de necesară, convocată d
- DUPĂ:   Astăzi am participat la o reuniune extraordinară a liderilor UE privind Ucraina, foarte utilă şi extrem de necesară, convocată de preşedintele Costa.

**Sursă:** digi24.ro (39w → 20w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media În apropierea localității Robotîne, aflată sub ocupație rusă, la sud de linia frontului din regiunea Zaporojie, se dau lupte grel
- DUPĂ:   În apropierea localității Robotîne, aflată sub ocupație rusă, la sud de linia frontului din regiunea Zaporojie, se dau lupte grele.

## 6. Pași următori

1. **Verifică Secțiunea 5** — conținutul curățat e jurnalistic valid?
   Dacă da, corpusul intermediar e bun.
2. Rulează scriptul de audit din nou pe `propozitii_cls0_no_cookies.parquet`
   pentru a confirma că nu mai există pattern-uri cookie reziduale.
3. Abia apoi decidem care e următoarea regulă de aplicat:
   etichete vorbitor, titluri concatenate, filtrare lungime, deduplicare.