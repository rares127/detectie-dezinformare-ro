# Curățare test set extern cls0 — cookie banners și meta-elemente

Diagnostic benchmark v4 a expus că AUC 0.98 pe Test A min era artefact: propozițiile 'min' pe cls0 erau cookie banners HotNews/Pro TV, nu conținut articol. Acest script elimină poluarea pentru re-rulare benchmark v4.

## Rezumat numeric

- Propoziții înainte: **2,181**
- Sufix cookie tăiat (concatenate): **105**
- Banner pur aruncat: **14**
- Comentarii aruncate: **6**
- Sub 7 cuvinte (post-curățare): **95**
- Peste 54 cuvinte: **0**
- Propoziții după: **2,066** (retenție 94.7%)

## Breakdown per sursă

| Sursă | Înainte | După | Δ | % păstrat |
|---|---:|---:|---:|---:|
| hotnews.ro | 870 | 764 | -106 | 87.82% |
| libertatea.ro | 121 | 112 | -9 | 92.56% |
| stirileprotv.ro | 134 | 134 | +0 | 100.0% |
| stopfals.md | 100 | 100 | +0 | 100.0% |
| veridica.ro | 956 | 956 | +0 | 100.0% |

## Articole afectate

- Articole dispărute complet: **0** (toate propozițiile au fost eliminate — suspect)
- Articole cu pierderi mari (>50% propoziții): **0**

## Exemple: tăiere sufix cookie (concatenate)

**1.** `ext_001` · libertatea.ro

- Original: `Urmărește pe Libertatea LIVETEXT cu cele mai noi informații despre războiul din Ucraina Loghează-te în contul tău pentru a adăuga comentarii și a te alătura dialogului.`
- Curățat: `Urmărește pe Libertatea LIVETEXT cu cele mai noi informații despre războiul din Ucraina`

**2.** `ext_003` · hotnews.ro

- Original: `HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.`
- Curățat: ``

**3.** `ext_003` · hotnews.ro

- Original: `Continuarea navigării implică acceptarea Termenilor și Condițiilor.`
- Curățat: ``

**4.** `ext_004` · hotnews.ro

- Original: `HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.`
- Curățat: ``

**5.** `ext_004` · hotnews.ro

- Original: `Continuarea navigării implică acceptarea Termenilor și Condițiilor.`
- Curățat: ``

**6.** `ext_005` · hotnews.ro

- Original: `Now is the time when it is important to reap the harvest… pic.twitter.com/JeDdKXkaQH HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs.`
- Curățat: `Now is the time when it is important to reap the harvest… pic.twitter.com/JeDdKXkaQH`

**7.** `ext_005` · hotnews.ro

- Original: `Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.`
- Curățat: ``

**8.** `ext_005` · hotnews.ro

- Original: `Continuarea navigării implică acceptarea Termenilor și Condițiilor.`
- Curățat: ``

**9.** `ext_006` · hotnews.ro

- Original: `HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.`
- Curățat: ``

**10.** `ext_006` · hotnews.ro

- Original: `Continuarea navigării implică acceptarea Termenilor și Condițiilor.`
- Curățat: ``

## Exemple: banner pur aruncat

**1.** `ext_004` · hotnews.ro: *Urmărește ultimele evoluții din a 373-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**2.** `ext_006` · hotnews.ro: *Urmărește ultimele evoluții din a 373-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**3.** `ext_009` · hotnews.ro: *Urmărește ultimele evoluții din a 518-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**4.** `ext_012` · hotnews.ro: *Urmărește ultimele evoluții din a 518-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**5.** `ext_013` · hotnews.ro: *Urmărește ultimele evoluții din a 663-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**6.** `ext_016` · hotnews.ro: *Urmărește ultimele evoluții din a 373-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**7.** `ext_021` · hotnews.ro: *Urmărește ultimele evoluții din a 337-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**8.** `ext_022` · hotnews.ro: *Urmărește ultimele evoluții din a 323-a zi a războiului din Ucraina LIVETEXT pe HOTNEWS.RO.*
**9.** `ext_032` · hotnews.ro: *Evenimentele de marți, ziua 1035 a războiului, au fost LIVE aici*
**10.** `ext_037` · hotnews.ro: *Informațiile de joi, ziua 1100 a agresiunii ruse, au fost LIVE aici pe HotNews.ro*

## Exemple: comentarii aruncate

**1.** `ext_018` · libertatea.ro: *clona.mea                                          26.02.2023, 21:51 Deci *** promite să recupereze Crimeea de la ruși.*
**2.** `ext_018` · libertatea.ro: *Acest comentariu a fost moderat pentru: limbaj vulgar sau jignitor dinamo_forever_77                                          26.02.2023, 22:24 ...stai linistit ca daca era asta in Romania toti erau pana acum  plecati din tara prin Grecia si alte state care nu returneaza nici macar  infractorii cons*
**3.** `ext_018` · libertatea.ro: *dante69                                          26.02.2023, 22:36 Istoric vorbind Crimea e a Ucrainei cum e și Bucovina de Nord.*
**4.** `ext_033` · libertatea.ro: *Lucaluca                                          22.05.2024, 15:26 NConteaza                                          22.05.2024, 15:40 zorrozabal                                          22.05.2024, 16:12*
**5.** `ext_034` · libertatea.ro: *Abiatar                                          29.05.2024, 11:11 floris                                          29.05.2024, 11:30 Babatia asta din„nortwest”-care mi-a furat nicnameul de acum vreo 8 ani(„cuviosul pafnutie”) mai adaugându-și un„sfântul”-imparte ea certificate de românism p-aci!Cond*
**6.** `ext_034` · libertatea.ro: *BONY                                          29.05.2024, 11:49 Nu-i nicio surpriza,era de asteptat ca Zelenski sa nu tina cont de conditiile puse de Occident,intersul lui este sa largeasca hora in care a intrat.*

## Următorul pas

Re-rulare `benchmark_v4.py` folosind `subset_benchmark_v3_curat.parquet` în loc de `subset_benchmark_v3.parquet`. Embeddings-urile corpusurilor (cls0 + cls1) rămân valide — cache hit instant. Doar embeddings-urile test set se recalculează (<5 sec pe MPS).

---

*Modul 3 · Pasul A2.6 · Curățare test set extern pre-re-rulare*