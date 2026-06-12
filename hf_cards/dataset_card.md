---
language:
  - ro
license: mit
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
pretty_name: Dezinformare Pro-Rusa in Presa Romaneasca
size_categories:
  - 1K<n<10K
tags:
  - disinformation
  - fake-news
  - romanian
  - ukraine
  - news
  - propaganda
---

# Dezinformare Pro-Rusă în Presa Românească

Dataset de 1.483 articole din presa românească, anotate pentru detecția
dezinformării pro-ruse în contextul războiului din Ucraina (2022–2026).

Construit și utilizat în lucrarea de licență: **Sistem de Detecție Automată și
Explicabilă a Dezinformării Pro-Ruse în Presa Românească** · Facultatea de
Informatică · 2025-2026.

## Statistici

| | Valoare |
|---|---|
| Total articole | **1.483** |
| Dezinformare pro-rusă (cls1) | **746** (50.3%) |
| Știri credibile (cls0) | **737** (49.7%) |
| Perioada acoperită | 2022 – 2026 |
| Limbă | Română |

### Distribuție pe surse

| Sursă | Articole | Clasă |
|---|---|---|
| veridica.ro | 661 | cls1 — dezinformare |
| stopfals.md | 85 | cls1 — dezinformare |
| digi24.ro | 389 | cls0 — credibil |
| g4media.ro | 348 | cls0 — credibil |

### Distribuție temporală

| An | Articole |
|---|---|
| 2022 | 417 |
| 2023 | 342 |
| 2024 | 314 |
| 2025 | 336 |
| 2026 | 74 |

## Structura fișierului CSV

| Coloană | Tip | Descriere |
|---|---|---|
| `id` | string | Identificator unic (prefix sursă + număr, ex. `vrd_0134`) |
| `url` | string | URL original al articolului |
| `titlu` | string | Titlul articolului |
| `text_curat` | string | Corpul articolului după curățare (fără cookies, boilerplate) |
| `data` | string | Data publicării (YYYY-MM-DD) |
| `an` | string | Anul publicării |
| `luna` | string | Luna publicării (YYYY-MM) |
| `sursa_site` | string | Domeniul sursei |
| `sectiune` | string | Secțiunea editorială a articolului |
| `label` | string | `dezinformare_pro_rusa` sau `stire_credibila` |
| `label_numeric` | int | 1 (dezinformare) sau 0 (credibil) |
| `stire_citata` | string | Textul sursei citate (ex. RIA Novosti) — relevant pentru analiza *reported speech* |
| `naratiuni_false` | string | Narațiunile false identificate (doar cls1, din editorial Veridica/Stopfals) |
| `obiective_propaganda` | string | Obiectivele propagandistice (doar cls1) |
| `nr_cuvinte_v4` | int | Număr de cuvinte după curățare finală |
| `nr_cuvinte_truncat` | int | Număr de cuvinte după trunchiere la 256 tokens XLM-R |
| `calitate_extractie` | string | Calitatea scrapingului: `excelenta` / `buna` / `medie` |
| `hash_continut` | string | SHA1 al `text_curat` — deduplicare |

## Surse și colectare

- **Cls1 (dezinformare):** articole de fact-checking de pe [Veridica.ro](https://veridica.ro)
  și [Stopfals.md](https://stopfals.md) — surse editoriale specializate în
  dezminteirea dezinformării pro-ruse. Articolele conțin narațiunile false
  identificate editorial.
- **Cls0 (credibil):** articole de știri de pe [G4Media.ro](https://g4media.ro)
  și [Digi24.ro](https://digi24.ro) — surse de presă mainstream românească.

Colectarea prin scraping propriu (BeautifulSoup/Requests), cu curățare manuală
și automată (deduplicare SHA1, filtrare cookie banners, validare calitate extracție).

## Utilizare

```python
import pandas as pd

df = pd.read_csv("dataset_licenta_complet.csv")
print(df["label"].value_counts())
# dezinformare_pro_rusa    746
# stire_credibila          737

# Split train/val/test folosit in lucrare
# (disponibil si ca fisiere separate in repo-ul GitHub)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.15, stratify=df["sursa_site"], random_state=42)
```

## Model asociat

Modelul fine-tuned pe acest dataset: [`rares127/xlmr-dezinformare-ro`](https://huggingface.co/rares127/xlmr-dezinformare-ro)

## Cod sursă

Pipeline complet (antrenare, evaluare LOSO, interfață web FastAPI):
[![GitHub](https://img.shields.io/badge/GitHub-detectie--dezinformare--ro-black?logo=github)](https://github.com/rares127/detectie-dezinformare-ro)

## Notă etică

Articolele de dezinformare sunt citate **doar în scop de cercetare** pentru
detecție automată. Narațiunile false conținute în `text_curat` și `stire_citata`
nu reprezintă poziția autorilor.

## Citare

```bibtex
@thesis{ungureanu2026dezinformare,
  title   = {Sistem de Detec{\c{t}}ie Automat{\u{a}} {\c{s}}i Explicabil{\u{a}}
             a Dezinform{\u{a}}rii Pro-Ruse \^{i}n Presa Rom\^{a}neasc{\u{a}}},
  author  = {Ungureanu, Rares},
  year    = {2026},
  school  = {Facultatea de Informatica}
}
```
