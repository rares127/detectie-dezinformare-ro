# Curățare cookies cls0 v3 — raport

**Versiune:** v3 (rezolvă bug-ul banner dublu prin curățare iterativă)

**Scope:** DOAR tratament boilerplate cookie, cu detecție robustă
indiferent de poziția banner-ului, și aplicare iterativă a curățării
pentru cazurile cu banner-e consecutive în aceeași propoziție.

## 1. Rezumat operații

| Operație | Număr propoziții |
|---|---|
| Input brut | 6,047 |
| Nemodificate | 5,933 |
| Banner pur aruncat (scenariul 1) | 71 |
| Curățat prefix banner (scenariul 2) | 41 |
| Curățat sufix banner (scenariul 3, NOU) | 2 |
| Aruncat — prea scurt după curățare | 0 |
| **Output** | **5,976** |

**Retenție:** 98.83%

## 2. Comparație v1 → v2

| Scenariu | v1 | v2 |
|---|---|---|
| Banner la început (scenariul 1+2) | Tratat | Tratat |
| Banner la mijloc/final (scenariul 3) | **NETRATAT** | **TRATAT** |
| Banner pur variații minore | Parțial | Complet |

## 3. Breakdown pe sursă

| Sursă | Input | Output | Retenție |
|---|---|---|---|
| digi24.ro | 3,297 | 3,226 | 97.85% |
| g4media.ro | 2,750 | 2,750 | 100.00% |

> Toate modificările ar trebui concentrate pe digi24.ro. G4Media = 100%.

## 4. Exemple — banner pur aruncat

- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *19w*: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
- [digi24.ro] *11w*: Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.

## 5. Exemple — prefix curățat (banner la început)

**Sursă:** digi24.ro (39w → 20w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media „Pentru a susține libertatea presei, am construit un oraș slavic, numit Voina, care înseamnă război în rusă”, a spus Mukka.
- DUPĂ:   „Pentru a susține libertatea presei, am construit un oraș slavic, numit Voina, care înseamnă război în rusă”, a spus Mukka.

**Sursă:** digi24.ro (40w → 21w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media ISW spune că forțele rusești au continuat bombardamentele intense, atacurile cu rachete și atacurile aeriene de-a lungul întregii
- DUPĂ:   ISW spune că forțele rusești au continuat bombardamentele intense, atacurile cu rachete și atacurile aeriene de-a lungul întregii linii de front.

**Sursă:** digi24.ro (52w → 33w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media Londra notează că acum comanda SGF ar urma să fie preluată de generalul-colonel Serghei Surovikin, deoarece SGF continuă să joace
- DUPĂ:   Londra notează că acum comanda SGF ar urma să fie preluată de generalul-colonel Serghei Surovikin, deoarece SGF continuă să joace un rol central în ofensiva pe care armata rusă o desfășoară în Donbas.

**Sursă:** digi24.ro (41w → 22w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media „Astăzi am participat la o reuniune extraordinară a liderilor UE privind Ucraina, foarte utilă şi extrem de necesară, convocată d
- DUPĂ:   „Astăzi am participat la o reuniune extraordinară a liderilor UE privind Ucraina, foarte utilă şi extrem de necesară, convocată de preşedintele Costa.

**Sursă:** digi24.ro (39w → 20w)
- ÎNAINTE: Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media În apropierea localității Robotîne, aflată sub ocupație rusă, la sud de linia frontului din regiunea Zaporojie, se dau lupte grel
- DUPĂ:   În apropierea localității Robotîne, aflată sub ocupație rusă, la sud de linia frontului din regiunea Zaporojie, se dau lupte grele.

## 6. Exemple — sufix curățat (banner la mijloc/final, NOU în v2)

> Cazul critic: propoziții lungi cu conținut real + banner agățat la coadă.
> Verifică că bucata păstrată e propoziție jurnalistică validă.

**Sursă:** digi24.ro (48w → 37w)
- ÎNAINTE: «Panțir-S1», care acoperea obiective importante și poziții de apărare aeriană inamice împotriva dronelor și rachetelor la altitudini joase și medii, a încetat și el să mai existe”, se arată într-un raport al Forțelor Speciale, potrivit Ukrinform . Setarile tale privind cookie-urile nu permit afisare
- DUPĂ:   «Panțir-S1», care acoperea obiective importante și poziții de apărare aeriană inamice împotriva dronelor și rachetelor la altitudini joase și medii, a încetat și el să mai existe”, se arată într-un raport al Forțelor Speciale, potrivit Ukrinform .

**Sursă:** digi24.ro (61w → 50w)
- ÎNAINTE: Într-o înregistrare video publicată pe Twitter, se observă un tanc Challenger - cel mai modern astfel de blindat din dotarea trupelor Kievului - complet distrus în apropiere de Robotîne, chiar în regiunea unde forțele ucrainene au anunțat zilele trecute că au reușit străpungerea primei linii de apăr
- DUPĂ:   Într-o înregistrare video publicată pe Twitter, se observă un tanc Challenger - cel mai modern astfel de blindat din dotarea trupelor Kievului - complet distrus în apropiere de Robotîne, chiar în regiunea unde forțele ucrainene au anunțat zilele trecute că au reușit străpungerea primei linii de apăr

## 7. Verificare finală

După rularea acestui script, rulează din nou `investigare_cookies_ramase.py`.
Rezultat așteptat: **0 matches** pe toate pattern-urile (A, B, C, D).