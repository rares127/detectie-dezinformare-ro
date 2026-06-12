# Explorare corpus cls0 — raport cantitativ

**Segmenter folosit:** `stanza`  
**Input:** articolele cls0 (Digi24 + G4Media) din dataset_licenta_complet.csv  
**Coloană segmentată:** `stire_citata`

## 1. Volum propoziții

- Total propoziții (brut): **6,047**
- Unice prin hash exact: **5,885** (97.32%)
- Unice prin hash normalizat (lowercase+fără punctuație): **5,877** (97.19%)
- Rată duplicare exactă: **2.68%**
- Rată duplicare normalizată (near-duplicate): **2.81%**

## 2. Distribuție lungime (cuvinte/propoziție)

| Statistică | Valoare |
|---|---|
| min | 1 |
| max | 142 |
| medie | 25.76 |
| mediana | 23.0 |
| std | 15.58 |

### Percentile

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| 3 | 6 | 9 | 15 | 23 | 34 | 45 | 54 | 78 |

## 3. Candidate la filtrare

- sub 3 cuvinte: **33**
- sub 5 cuvinte: **138** (pragul propus inițial în handoff)
- peste 40 cuvinte: **888** (pragul propus inițial în handoff)
- peste 60 cuvinte: **187**

> **Notă metodologică.** Decizia finală privind pragurile de filtrare
> se ia pe baza percentilelor reale (de exemplu [p5, p95]), nu pe
> numere rotunde propuse a priori în handoff.

## 4. Breakdown pe sursă

| Sursă | Articole | Propoziții | Prop/articol | Lungime mediană | Unice (normalizat) |
|---|---|---|---|---|---|
| digi24.ro | 389 | 3,297 | 8.48 | 22 | 3,148 |
| g4media.ro | 348 | 2,750 | 7.9 | 24 | 2,739 |

> **Semnal pentru stylistic fingerprint.** Dacă cele două surse au
> distribuții foarte diferite de lungime sau de propoziții-per-articol,
> embeddings-urile din modulul 3 pot prinde stilul sursei în loc de
> conținutul factual. Acest risc se verifică efectiv în pasul 3
> (benchmark model embeddings).

## 5. Suprapunere propoziții între surse

- Surse: digi24.ro, g4media.ro
- Propoziții comune (hash normalizat): **10**
- Procent din sursa mai mică: **0.37%**

## 6. Top 10 propoziții duplicate

1. **59 apariții** (11 cuvinte): Setarile tale privind cookie-urile nu permit afisarea continutul din aceasta sectiune.
2. **13 apariții** (23 cuvinte): Jurnalistul care i-a luat ultimul interviu lui Mircea Lucescu povestește culisele întâlnirii Răzvan Lucescu, discurs emoţionant în biserică înainte de înmormântarea tatălui său:
3. **10 apariții** (8 cuvinte): Sanatoriul din Sergheevca, regiunea Odesa, bombardat de ruși.
4. **9 apariții** (15 cuvinte): A fost emis RO-Alert pentru nordul județului Tulcea „S-a simțit rău și a pierdut kilograme”.
5. **8 apariții** (11 cuvinte): UE analizează relația cu Ungaria, după dezvăluirile privind legăturile cu Rusia.
6. **7 apariții** (19 cuvinte): Poti actualiza setarile modulelor coookie direct din browser sau de aici – e nevoie sa accepti cookie-urile social media
7. **6 apariții** (21 cuvinte): Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
8. **5 apariții** (33 cuvinte): „A fost un luptător, a învins, a pierdut” „Ghost Murmur”, instrumentul CIA care a urmărit bătăile inimii aviatorului american dispărut în Iran UE analizează relația cu Ungaria, după dezvăluirile privi...
9. **5 apariții** (11 cuvinte): Putin a vizitat pentru prima oară militari ruşi răniţi în Ucraina.
10. **4 apariții** (1 cuvinte): Zelenski:

## 7. Recomandări pentru pasul următor

1. **Filtrare lungime:** folosește [p5, p95] reale în loc de [5, 40] propuse a priori.
2. **Deduplicare:** folosește hash normalizat, nu exact — prinde și near-duplicatele banale.
3. **Echilibrare surse:** dacă Digi24/G4Media au volume foarte diferite de propoziții,
   ia în considerare subsampling pentru sursa dominantă înainte de embeddings.
4. **Investigație duplicate cross-source:** dacă procentul de propoziții comune între
   Digi24 și G4Media e mare, sunt probabil formulări de agenție (Agerpres/Reuters)
   preluate identic — pot fi păstrate, dar contribuie ca ancoră factuală unică.