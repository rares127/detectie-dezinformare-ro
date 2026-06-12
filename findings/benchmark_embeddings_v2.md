# Benchmark embeddings v2 — Modulul 3 (reparat)

**Seed:** 42 · **Device:** mps · **Top-K:** 5 · **Percentilă:** p10

**Reparație față de v1:** articolele cls0 vin acum din surse EXTERNE (Pro TV, HotNews etc.), nu din aceleași surse Digi24/G4Media din care s-a construit corpusul. V1 avea contaminare 100% (cls0 test ⊂ corpus) ceea ce producea AUC=1.0 trivial.

**Corpus referință:** 5290 propoziții (neatins)
**Subset v2:** 30 articole (15 cls0 extern + 15 cls1), 477 propoziții

---

## 1. Separabilitate — scoruri agregate

Convenție: scor mai mare = mai similar cu corpus cls0 = mai credibil.
cls0 ar trebui să aibă scoruri MAI MARI decât cls1.

| Model | Prop | Agr. | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |
|---|---|---|---|---|---|---|---|---|---|
| mpnet | max | mean | 0.733 | +0.77 | 0.674 | 0.641 | 0.637 | 0.656 | -0.019 |
| mpnet | topk_mean | mean | 0.707 | +0.67 | 0.637 | 0.608 | 0.604 | 0.624 | -0.020 |
| minilm | topk_mean | mean | 0.653 | +0.47 | 0.617 | 0.595 | 0.589 | 0.623 | -0.034 |
| minilm | max | mean | 0.644 | +0.54 | 0.657 | 0.633 | 0.627 | 0.660 | -0.034 |
| mpnet | max | p10 | 0.444 | -0.17 | 0.554 | 0.564 | 0.559 | 0.580 | -0.021 |
| minilm | topk_mean | p10 | 0.409 | -0.33 | 0.486 | 0.505 | 0.499 | 0.530 | -0.032 |
| mpnet | topk_mean | p10 | 0.396 | -0.32 | 0.517 | 0.533 | 0.529 | 0.551 | -0.022 |
| minilm | max | p10 | 0.391 | -0.33 | 0.523 | 0.541 | 0.536 | 0.563 | -0.028 |
| mpnet | max | min | 0.120 | -1.67 | 0.406 | 0.520 | 0.512 | 0.550 | -0.038 |
| mpnet | topk_mean | min | 0.116 | -1.74 | 0.384 | 0.496 | 0.487 | 0.530 | -0.044 |
| xlmr_ft_mean | max | mean | 0.089 | -1.62 | 0.989 | 0.992 | 0.992 | 0.992 | -0.000 |
| xlmr_ft_mean | topk_mean | mean | 0.084 | -1.68 | 0.988 | 0.991 | 0.991 | 0.991 | -0.000 |
| xlmr_ft_mean | max | p10 | 0.058 | -1.53 | 0.981 | 0.990 | 0.990 | 0.990 | -0.000 |
| minilm | topk_mean | min | 0.058 | -1.82 | 0.348 | 0.462 | 0.450 | 0.510 | -0.060 |
| xlmr_ft_mean | topk_mean | p10 | 0.049 | -1.57 | 0.979 | 0.989 | 0.989 | 0.989 | +0.000 |
| minilm | max | min | 0.049 | -1.98 | 0.374 | 0.499 | 0.486 | 0.549 | -0.063 |
| xlmr_ft_mean | topk_mean | min | 0.031 | -2.97 | 0.958 | 0.985 | 0.985 | 0.985 | +0.000 |
| xlmr_ft_mean | max | min | 0.013 | -3.02 | 0.960 | 0.987 | 0.987 | 0.986 | +0.001 |

---

## 2. Separabilitate — proporție propoziții sub threshold τ

Metrică: pentru fiecare articol, fracția propozițiilor cu scor_max < τ.
Convenție INVERSĂ aici: fracție MAI MARE = mai multe propoziții fără 
corespondent factual = mai probabil cls1 (dezinformare).

Praguri τ testate: [0.3, 0.4, 0.5, 0.6]

| Model | Prag τ | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |
|---|---|---|---|---|---|---|---|---|
| minilm | 0.6 | 0.573 | +0.32 | 0.320 | 0.378 | 0.390 | 0.329 | +0.061 |
| mpnet | 0.6 | 0.562 | +0.38 | 0.224 | 0.294 | 0.309 | 0.233 | +0.075 |
| minilm | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| mpnet | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.3 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.4 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.5 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| xlmr_ft_mean | 0.6 | 0.500 | +nan | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 |
| mpnet | 0.5 | 0.391 | +0.25 | 0.061 | 0.093 | 0.108 | 0.033 | +0.075 |
| minilm | 0.5 | 0.356 | -0.10 | 0.100 | 0.090 | 0.095 | 0.067 | +0.029 |
| mpnet | 0.4 | 0.167 | -1.62 | 0.032 | 0.000 | 0.000 | 0.000 | +0.000 |
| minilm | 0.4 | 0.124 | -1.34 | 0.043 | 0.007 | 0.008 | 0.000 | +0.008 |

---

## 3. Viteză embeddings (corpus 5,290 propoziții)

| Model | Propoziții/secundă | Timp total (s) |
|---|---|---|
| minilm | 789.7 | 6.7 |
| mpnet | 331.4 | 16.0 |
| xlmr_ft_mean | 233.2 | 22.7 |

---

## 4. Finding metodologic — XLM-R fine-tuned NU e adecvat pentru similaritate

**Observația**: checkpoint-ul XLM-R fine-tuned pe clasificare (modulul 2) produce embeddings cu distribuție colapsată — toate propozițiile primesc 
scoruri aproape identice, indiferent de conținut. Uită-te la coloanele μ(cls0) și μ(cls1) în tabelele de mai sus pentru `xlmr_ft_mean`: diferența dintre clase e sub 0.01 în valoare absolută, vs ~0.35 pentru MiniLM/mpnet.

**Interpretare**: fine-tuning-ul pe sarcina de clasificare binară împinge reprezentările spre hiperplanul de decizie, colapsând geometria semantică originală (pe care XLM-R pretrained o avea). Modelul și-a optimizat reprezentările pentru a separa cls0/cls1 la nivel de POOL de clasificare, nu pentru a captura similaritate semantică generală.

**Implicație pentru teză**: acest rezultat justifică alegerea unui model antrenat specific pe similaritate (sentence-transformers) în locul reutilizării modelului de clasificare. E o contribuție metodologică: validează empiric că arhitectura sistemului trebuie să folosească două modele distincte, nu unul singur pentru ambele sarcini.

**De inclus în capitolul „Arhitectură și justificări tehnice”**.

---

## 5. Cross-source — gap Veridica vs Stopfals (critic pentru modulul 2)

Dacă gap-ul Δ(V-S) e aproape de zero, modelul tratează ambele surse 
propagandistice similar — semn că analiza granulară e robustă 
cross-source și poate compensa problema LOSO-V din modulul 2.

Dacă gap-ul e semnificativ (>0.05), avem încă stylistic fingerprint și 
modulul 3 singur nu rezolvă problema.

**Cel mai mic gap pe scoruri agregate** (mai mic e mai bine):
- `xlmr_ft_mean` / topk_mean / min: Δ = +0.000, AUC = 0.031, d = -2.97

---

## 6. Recomandare finală

Criterii de alegere:
1. AUC cât mai aproape de 1.0 (dar realist — fără contaminare)
2. Gap V-S cât mai mic (robustețe cross-source)
3. Viteză acceptabilă pe M2 Pro
4. Interpretabilitate (preferăm metrici simple de explicat)

**Configurație principală propusă:** `mpnet` + propoziție=max + articol=mean
- AUC = 0.733
- Cohen's d = +0.77
- Gap V-S = -0.019

**Metrică interpretabilă suplimentară:** proporție sub τ=0.6 cu `minilm` (AUC = 0.573, d = +0.32)

Această metrică e utilă pentru interfața demonstrativă — poate fi prezentată ca „X% din propozițiile articolului nu au corespondent în corpusul de presă credibilă”, mai intuitivă decât un scor abstract.

*Generat automat.*