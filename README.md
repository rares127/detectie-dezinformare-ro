# Sistem de Detectie Automata si Explicabila a Dezinformarii Pro-Ruse in Presa Romaneasca

> **Lucrare de licenta** · Facultatea de Informatica Iasi · Universitatea Alexandru Ioan Cuza (UAIC) · 2025-2026
> Autor: Rares Ungureanu

[![Model on HF](https://img.shields.io/badge/🤗_Model-xlmr--dezinformare--ro-yellow)](https://huggingface.co/rares127/xlmr-dezinformare-ro)
[![Dataset on HF](https://img.shields.io/badge/🤗_Dataset-dezinformare--ro-blue)](https://huggingface.co/datasets/rares127/dezinformare-ro)

Pipeline NLP end-to-end care clasifica articole din presa romaneasca ca
dezinformare pro-rusa (`label=1`) sau stiri credibile (`label=0`), cu
explicabilitate la nivel de propozitie si interfata web demonstrativa.

## Arhitectura (5 module)

| Modul | Rol | Tehnologie |
|---|---|---|
| 1. Preprocessing | Segmentare propozitii, curatare | Stanza (ro) |
| 2. Clasificare globala | Verdict la nivel de articol | XLM-RoBERTa fine-tuned |
| 3. Analiza granulara | Similaritate semantica per propozitie vs. corpusuri de referinta | paraphrase-multilingual-mpnet-base-v2 |
| 4. Explainability | LIME (validat empiric doar pe cls0) + diagnostic IG/DeepLift/GradShap | lime, captum |
| 5. Interfata web | Demo cu vizualizare colorata per propozitie | FastAPI + Vanilla JS |

**Decizia finala vine de la Modulul 3** (F1 = 0.9454, LOSO-V drop doar 7.7pp),
nu de la clasificatorul global (LOSO-V drop 70.65pp — stylistic fingerprint,
documentat ca limitare in lucrare). Threshold productie: **-0.0073**
(calibrat CV 5-fold).

## Structura proiectului

```
app/                  Modulul 5 — aplicatia FastAPI (backend + frontend)
├── config.py         toate constantele/hyperparametrii centralizati
├── core/             wrappere inferenta (modulele 2, 3, 4) + preprocessing
├── routes/           /api/predict, /api/explain_lime, /api/health
├── schemas/          modele Pydantic request/response
├── static/, templates/   frontend (HTML + Tailwind CDN + Vanilla JS)
src/
├── scraping/         scrapere + curatare per sursa (Veridica, Stopfals, G4Media, Digi24)
├── training/         Modul 2: split, antrenare, evaluare, LOSO, diagnostic XAI
└── modul3/           Modul 3: pipeline complet
    ├── corpus/       constructia si auditul corpusurilor de referinta
    ├── benchmark/    benchmark modele embeddings + subseturi de evaluare
    ├── calibrare/    calibrarea threshold-ului (CV 5-fold)
    └── loso/         evaluare leave-one-source-out pentru modulul 3
data/processed/       corpusuri parquet + split-uri train/val/test (incluse, mici)
data/final/           dataset complet anottat, gata de utilizare
findings/             rezultate empirice (JSON + MD + vizualizari HTML)
tests/                teste de integrare API (pytest, modele reale)
scripts/              utilitare (ex. scoate_diacritice.py)
```

**Conventie:** toate scripturile se ruleaza din radacina proiectului
(path-urile relative `data/...`, `findings/...` se rezolva fata de CWD).

## Ce NU este in repository (dimensiune)

| Artefact | Dimensiune | Cum se obtine |
|---|---|---|
| `models/` (XLM-R fine-tuned, LOSO) | ~26 GB | descarca de pe HF: `huggingface-cli download rares127/xlmr-dezinformare-ro --local-dir models/xlmr_baseline_v2/final` sau re-antrenare: `python src/training/02_train_xlmr_baseline_v2.py` (~4 min pe M2 Pro / T4) |
| `data/raw/` (CSV-uri scraping) | ~550 MB | re-rulare scrapere din `src/scraping/` |
| `data/processed/embeddings_cache/` | ~45 MB | regenerat automat la primul startup al aplicatiei |

Corpusurile de referinta si split-urile finale (necesare antrenarii si
aplicatiei) SUNT incluse — sunt mici (parquet < 1 MB fiecare).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import stanza; stanza.download('ro')"

# Antreneaza modelul baseline (necesar pentru aplicatie)
python src/training/02_train_xlmr_baseline_v2.py \
    --data_dir data/processed --output_dir models/xlmr_baseline_v2

# Porneste aplicatia (http://127.0.0.1:8000)
./run_app.sh

# Teste de integrare (folosesc modelele reale)
pytest tests/test_health_endpoint.py tests/test_predict_endpoint.py tests/test_explain_endpoint.py -v
```

## Dataset

`data/final/dataset_licenta_complet.csv` — 1.781 articole adnotate manual (5.7 MB).

| Coloana | Descriere |
|---|---|
| `id` | Identificator unic per articol (prefix sursa + numar) |
| `url` | URL original al articolului |
| `titlu`, `text_curat` | Titlul si corpul articolului dupa curatare |
| `data`, `an`, `luna` | Data publicarii |
| `sursa_site` | Sursa: `veridica.ro`, `stopfals.md`, `g4media.ro`, `digi24.ro` |
| `label` / `label_numeric` | `dezinformare_pro_rusa` / `stire_credibila` → 1 / 0 |
| `stire_citata` | Textul sursei citate (ex. RIA Novosti) — util pentru analiza "reported speech trap" |
| `naratiuni_false`, `obiective_propaganda` | Adnotari editorial Veridica/Stopfals (cls1 only) |
| `nr_cuvinte_v4` | Numar de cuvinte dupa curatare finala |
| `calitate_extractie` | Flag calitate scraping (excelenta / buna / medie) |
| `hash_continut` | SHA1 al textului curat — permite detectarea duplicatelor |

**Distributie:** 746 dezinformare (cls1) / 735 credibile (cls0) · Perioada: 2022-2026.  
**Surse cls1:** Veridica.ro, Stopfals.md · **Surse cls0:** G4Media.ro, Digi24.ro

## Rezultate cheie

- **Dataset:** 1.483 articole (746 dezinformare / 737 credibile), 2022-2026,
  surse: Veridica.ro, Stopfals.md (cls1) / G4Media.ro, Digi24.ro (cls0)
- **Modul 2 (IID):** macro-F1 100%, accuracy 99.07% — dar LOSO-V recall 29.35%
  (stylistic fingerprint, prima expunere prin LOSO pe fake news romanesc)
- **Modul 3:** AUC 0.969, F1 0.9454 ± 0.0397, LOSO-V AUC 0.892 (~9x mai robust
  cross-source decat modulul 2)
- **XAI:** comparatie 4-way (LIME, IG, DeepLift, GradientShap) pe transformer
  saturat; finding original D<C — narațiune distribuita cross-token

Detaliile empirice complete sunt in `findings/`.