# Audit Cleaning — stopfals_raw_v1.csv

**Data**: 2026-04-19

## Sumar

| Metric | Valoare |
|---|---|
| Articole raw | 238 |
| Articole eliminate | 32 |
| Articole curate | 206 |
| Cu stire_citata | 96 |
| Fără stire_citata | 110 |

## Distribuție per an (clean)

- **2022**: 58 articole
- **2023**: 66 articole
- **2024**: 82 articole

## Articole eliminate (32)

| An | Cuvinte | Motiv | Titlu |
|---|---|---|---|
| 2022 | 46 | sub_200_cuvinte | CRONICA DEZINFORMĂRII (16-28 februarie 2022) |
| 2022 | 66 | sub_200_cuvinte | CRONICA DEZINFORMĂRII (1-15 martie 2022) |
| 2022 | 65 | sub_200_cuvinte | CRONICA DEZINFORMĂRII (16-28 martie 2022) |
| 2022 | 90 | sub_200_cuvinte | CELE MAI RĂSPÂNDITE FALSURI DESPRE RĂZBOIUL RUSIEI ÎMPOTRIVA UCRAINEI |
| 2022 | 89 | sub_200_cuvinte | FALSURI DESPRE MOLDOVA ÎN CONTEXTUL INVAZIEI MILITARE RUSEȘTI ÎN UCRAI |
| 2022 | 73 | sub_200_cuvinte | FALSURI DESPRE/DIN REGIUNEA TRANSNISTREANĂ |
| 2022 | 104 | sub_200_cuvinte | FALS: Autoritățile europene ar impune autoritățile moldovenești să leg |
| 2023 | 120 | sub_200_cuvinte | Radio Moldova: Adevărul despre falsul că Republica Moldova se înarmeaz |
| 2023 | 121 | sub_200_cuvinte | INFORMAȚIA OBIECTIVĂ VS DEZINFORMAREA: CARE-I DIFERENȚA? |
| 2023 | 98 | sub_200_cuvinte | CELE MAI RĂSPÂNDITE FALSURI ÎN CONTEXTUL CRIZEI ENERGETICE DIN R. MOLD |
| 2023 | 75 | sub_200_cuvinte | TANCURILE NATO PE... AUTOSTRADA PROPAGANDEI |
| 2023 | 92 | sub_200_cuvinte | DEZMINȚIREA ȘTIRILOR FALSE DESPRE SUVERANITATEA REPUBLICII MOLDOVA |
| 2023 | 75 | sub_200_cuvinte | CUM NE INFORMĂM CORECT? |
| 2023 | 71 | sub_200_cuvinte | Cum dezinformarea afectează societatea? |
| 2023 | 84 | sub_200_cuvinte | GÂNDIREA CRITICĂ – „SCUTUL” ÎMPOTRIVA DEZINFORMĂRII |
| 2023 | 144 | sub_200_cuvinte | Deepfake cu folosirea imaginii Maiei Sandu și a unei bănci comerciale |
| 2023 | 150 | sub_200_cuvinte | Escrocherie pe social media cu folosirea imaginii lui Ion Chicu și pro |
| 2024 | 135 | sub_200_cuvinte | În Chișinău au apărut afișe FALSE care pretind că Legiunea Franceză re |
| 2024 | 115 | sub_200_cuvinte | FALS: Fanii români au scandat „Putin” și au afișat drapelul așa-numite |
| 2024 | 115 | sub_200_cuvinte | FALS: R. Moldova intenționează să introducă regim de vize cu Rusia |
| 2024 | 162 | sub_200_cuvinte | Speculații și manipulări la tema podului peste Prut ce urmează a fi co |
| 2024 | 192 | sub_200_cuvinte | FALS: PAS a anunțat interzicerea Bisericii Ortodoxe Ruse din Moldova |
| 2024 | 124 | sub_200_cuvinte | CE ESTE INFORMAȚIA OBIECTIVĂ ȘI CUM O DEOSEBIM DE DEZINFORMARE? |
| 2024 | 122 | sub_200_cuvinte | DEZINFORMAREA - AMENINȚARE LA ADRESA SECURITĂȚII NAȚIONALE ȘI A SIGURA |
| 2024 | 127 | sub_200_cuvinte | FALS: Linella oferă carduri promoționale de 6.000 de lei persoanelor c |
| 2024 | 107 | sub_200_cuvinte | Ilan Șor: „Acuș o să vă arăt...” |
| 2024 | 54 | sub_200_cuvinte | Bogdan Țîrdea, bucătarul-șef al propagandei Kremlinului în Moldova, în |
| 2024 | 64 | sub_200_cuvinte | Urmașii mătușii din Etulia |
| 2024 | 59 | sub_200_cuvinte | Canal 5 și pudra în cantități industriale |
| 2024 | 55 | sub_200_cuvinte | Primiți cu pluguȘORul? |
| 2022 | 417 | titlu_compilatie | Au fost desemnați câștigătorii primului test de verificare a „anticorp |
| 2022 | 243 | titlu_compilatie | Anticorpi la fals: Cod roșu de… propagandă la Tiraspol. De ce regiunea |

## Note

- Articolele cu `stire_citata` gol nu sunt eliminate — stopfals.md
  nu are întotdeauna citat izolat în `<em>`, dar `text_curat` conține
  oricum narațiunea falsă integrată în corpul articolului.
- Pragul `nr_cuvinte >= 200` elimină cronicile scurte și anunțurile.
- Input-ul corect pentru clasificator: `text_curat` (corpul articolului),
  NU `stire_citata` (care e mai degrabă util pentru modulul granular).
