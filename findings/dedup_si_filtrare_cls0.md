# Deduplicare + Filtrare lungime cls0 — raport

**Pașii 4+5 din pipeline-ul de preprocessing.**
Aceștia produc `propozitii_cls0_corpus.parquet` —
corpusul final gata de embeddings.

## 1. Pasul 4 — Deduplicare

| Metrică | Valoare |
|---|---|
| Input (după filtru rezidual) | 5,915 |
| Duplicate eliminate | 98 |
| Output după deduplicare | 5,817 |

### Breakdown per sursă

| Sursă | Înainte | După | Eliminate |
|---|---|---|---|
| digi24.ro | 3,205 | 3,122 | 83 |
| g4media.ro | 2,710 | 2,695 | 15 |

### Top duplicate eliminate

> Propoziții care apăreau de mai multe ori — confirmare că deduplicarea
> a prins artefactele repetate (ex. aceeași fotografie de presă, aceleași
> fraze standard preluate de la agenții).

- **13 apariții** (23w) [digi24.ro]: Jurnalistul care i-a luat ultimul interviu lui Mircea Lucescu povestește culisele întâlnirii Răzvan Lucescu, discurs emoţionant în biserică înainte de înmormântarea tatălui său:
- **10 apariții** (8w) [digi24.ro]: Sanatoriul din Sergheevca, regiunea Odesa, bombardat de ruși.
- **9 apariții** (15w) [digi24.ro]: A fost emis RO-Alert pentru nordul județului Tulcea „S-a simțit rău și a pierdut kilograme”.
- **8 apariții** (11w) [digi24.ro]: UE analizează relația cu Ungaria, după dezvăluirile privind legăturile cu Rusia.
- **8 apariții** (19w) [digi24.ro]: Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- **5 apariții** (33w) [digi24.ro]: „A fost un luptător, a învins, a pierdut” „Ghost Murmur”, instrumentul CIA care a urmărit bătăile inimii aviatorului american dispărut în Iran UE analizează relația cu Ungaria, după dezvăluirile privi
- **5 apariții** (11w) [digi24.ro]: Putin a vizitat pentru prima oară militari ruşi răniţi în Ucraina.
- **4 apariții** (20w) [digi24.ro]: Un român și un moldovean produc drone interceptoare care pot distruge Shahed-urile lansate de Kremlin Atacuri rusești la granița României.
- **4 apariții** (15w) [digi24.ro]: Siegfrid Mureșan: „Să stăm departe de acest virus” „S-a simțit rău și a pierdut kilograme”.
- **4 apariții** (106w) [digi24.ro]: „A fost un luptător, a învins, a pierdut” „Ghost Murmur”, instrumentul CIA care a urmărit bătăile inimii aviatorului american dispărut în Iran Scumpiri în lanț în București: biletul STB ar putea ajung

## 2. Pasul 5 — Filtrare lungime

**Praguri calculate pe corpusul deduplicat:**
[p5=7w, p95=54w]

| Metrică | Valoare |
|---|---|
| Input (după deduplicare) | 5,817 |
| Eliminate < 7w (sub p5) | 249 |
| Eliminate > 54w (peste p95) | 278 |
| **Output final** | **5,290** |
| Retenție față de input pasul 5 | 90.94% |

### Distribuție lungime după filtrare

| min | p25 | mediană | medie | p75 | max |
|---|---|---|---|---|---|
| 7 | 16 | 23 | 24.85 | 33 | 54 |

## 3. Sumar final pipeline preprocessing

| Pas | Fișier | Propoziții |
|---|---|---|
| Segmentare (Stanza) | propozitii_cls0_raw.parquet | 6,047 |
| Curățare cookies v3 | propozitii_cls0_no_cookies.parquet | 5,976 |
| Filtru rezidual | propozitii_cls0_filtrat.parquet | 5,915 |
| Deduplicare | — | 5,817 |
| Filtrare lungime [7w, 54w] | **propozitii_cls0_corpus.parquet** | **5,290** |

**Retenție totală față de raw:** 87.48%

## 4. Pasul următor

Corpusul `propozitii_cls0_corpus.parquet` e gata pentru:
**Benchmark model embeddings** — comparație XLM-RoBERTa mean-pooled
vs sentence-transformers multilingv pe acest corpus.