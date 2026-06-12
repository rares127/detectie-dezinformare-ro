# Modul 5 — Aplicație Web (FastAPI)

Interfața web demonstrativă pentru sistemul de detecție a dezinformării
pro-ruse din lucrarea de licență.

## Cerințe

- Python 3.11 sau 3.12
- macOS cu chip M1/M2 (pentru MPS) — sau Linux cu CUDA — sau CPU (lent)
- ~3 GB RAM disponibili pentru modele
- Modulele 2 și 3 antrenate (vezi `models/xlmr_baseline_v2/final/` și
  `data/processed/propozitii_cls{0,1}_corpus*.parquet`)

## Instalare

```bash
# Din rădăcina proiectului
pip install -r requirements_app.txt

# Descarcă modelul Stanza ro (~700MB) — o singură dată
python -c "import stanza; stanza.download('ro')"
```

## Rulare locală

```bash
# Opțiunea 1: script convenience
./run_app.sh

# Opțiunea 2: direct
python -m app.main

# Opțiunea 3: uvicorn explicit (dev)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Aplicația va fi disponibilă la http://127.0.0.1:8000.

## Endpoint-uri

| Metodă | Cale                  | Descriere |
|--------|------------------------|-----------|
| GET    | `/`                    | Single-page UI (Tailwind + Vanilla JS) |
| POST   | `/api/predict`         | Clasificare globală + similaritate semantică |
| POST   | `/api/explain_lime`    | Explicație LIME (DOAR pentru predicții cls0) |
| GET    | `/api/health`          | Status modele încărcate |
| GET    | `/docs`                | Swagger UI (auto-generat) |

## Decizii arhitecturale

### Strategia diferențiată pe clasă (centrală)

Conform findings_xai_l4.md (capitolul 4 al tezei):

- **Pe cls0** (știri credibile): faith_auc LIME = +0.169 → cuvinte cu impact
  cauzal validat empiric. Afișăm colorarea cuvintelor.
- **Pe cls1** (propagandă): faith_auc LIME ≈ 0 sau NEGATIV → ștergerea cuvintelor
  nu schimbă predicția (sau o crește). Colorarea ar fi misleading. NU afișăm.
- **În schimb**, pe cls1 afișăm propozițiile similare din corpusul cls1
  (modul 3, similaritate semantică) — explicabilitatea principală.

Această diferență e validată empiric și transformă o limitare metodologică
într-un feature: aplicația face dezvăluire onestă a propriei limite de
explicabilitate.

### Edge case INCERT

Dacă articolul nu conține nicio propoziție cu lungime între 7 și 54 cuvinte,
modulul 3 nu poate calcula scorul. În acest caz:
- Decizia afișată = "Incert"
- Se afișează doar scorul baseline (modul 2) ca informație
- Nu se afișează propoziții similare sau LIME

### Lazy LIME

LIME se inițializează la primul request `/explain_lime`, nu la startup.
Dacă utilizatorul testează doar articole propagandistice, LIME nu se
încarcă niciodată — economisim memorie.

## Performanță estimată (MacBook M2 Pro)

| Operație               | Timp |
|------------------------|------|
| Startup (loading modele) | 15-20 s |
| `/predict` (modul 2 + 3) | 1-5 s |
| `/explain_lime` (LIME)   | 10-30 s |
| `/health`                | < 10 ms |

## Teste

```bash
# Toate testele de integrare
pytest tests/

# Test individual
pytest tests/test_predict_endpoint.py -v
```

## Limitări cunoscute

1. **Trunchiere XLM-R**: textele > 256 tokens (~1500 caractere) sunt trunchiate
   pentru clasificarea globală. Modul 3 (per-propoziție) nu e afectat. UI-ul
   afișează un warning.

2. **Stanza pe CPU**: pipeline-ul Stanza rulează pe CPU pentru consistență cu
   antrenarea. Pe articole foarte lungi (>10k caractere), segmentarea poate
   adăuga 1-2 secunde la latency.

3. **Hot-reload dezactivat**: din cauza modelelor mari (~2.5GB total), opțiunea
   `--reload` ar reîncărca tot la fiecare schimbare de cod. Pentru dev frontend
   rapid, restartează serverul manual.

## Versionare modele

Verifică modelul curent prin `/api/health`:

```bash
curl http://127.0.0.1:8000/api/health | jq
```

Versiunea producție (validată în teză):
- `modul2_classifier`: `xlmr_baseline_v2`
- `modul3_encoder`: `paraphrase-multilingual-mpnet-base-v2`
- `threshold_modul3`: `-0.0073` (calibrat CV 5-fold, F1 = 0.9454)
