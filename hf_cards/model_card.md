---
language:
  - ro
license: mit
base_model: xlm-roberta-base
tags:
  - text-classification
  - disinformation
  - fake-news
  - romanian
  - ukraine
pipeline_tag: text-classification
datasets:
  - rares127/dezinformare-ro
metrics:
  - f1
  - accuracy
model-index:
  - name: xlmr-dezinformare-ro
    results:
      - task:
          type: text-classification
          name: Text Classification
        dataset:
          name: Dezinformare Pro-Rusa in Presa Romaneasca
          type: rares127/dezinformare-ro
        metrics:
          - type: f1
            name: Macro F1 (IID test set)
            value: 1.0
          - type: accuracy
            name: Accuracy (IID test set)
            value: 0.9907
---

# xlmr-dezinformare-ro

XLM-RoBERTa fine-tuned pentru detecția dezinformării pro-ruse în presa românească,
în contextul războiului din Ucraina (2022–2026).

Parte din lucrarea de licență: **Sistem de Detecție Automată și Explicabilă a
Dezinformării Pro-Ruse în Presa Românească** · Facultatea de Informatică · 2025-2026.

> **⚠️ Notă importantă despre utilizare:** Modelul singur (IID F1=1.0) suferă de
> *stylistic fingerprint* cross-sursă (LOSO-V recall 29.35% — drop de 70.65pp).
> În sistemul final, decizia vine de la **Modulul 3** (similaritate semantică,
> F1=0.9454, LOSO-V drop doar 7.7pp). Utilizați acest model doar ca parte a
> pipeline-ului complet, disponibil pe GitHub.

## Utilizare rapidă

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="rares127/xlmr-dezinformare-ro"
)

text = "Articolul de analizat în limba română..."
result = classifier(text)
# Exemplu output:
# [{'label': 'stire_credibila',       'score': 0.987}]
# [{'label': 'dezinformare_pro_rusa', 'score': 0.994}]
```

> **Limită:** modelul procesează maxim 256 tokens (~1500 caractere). Articolele
> mai lungi sunt trunchiate automat de tokenizer.

## Etichete

| Label | ID | Sursă de antrenare |
|---|---|---|
| `stire_credibila` | 0 | G4Media.ro, Digi24.ro |
| `dezinformare_pro_rusa` | 1 | Veridica.ro, Stopfals.md |

## Detalii antrenare

| Parametru | Valoare |
|---|---|
| Model de bază | `xlm-roberta-base` |
| Task | Clasificare binară (sequence classification) |
| Loss | CrossEntropyLoss cu ponderi de clasă (WeightedTrainer) |
| Optimizer | AdamW |
| EarlyStopping | patience=2 pe validation F1 |
| Split | 70% train / 15% val / 15% test (stratificat pe sursă) |
| Hardware | Apple M2 Pro / Google Colab T4 (~4 min) |

## Performanță

### IID (test set in-distribution)

| Metrică | Valoare |
|---|---|
| Macro F1 | **1.000** |
| Accuracy | **99.07%** |
| Precision cls1 | 100% |
| Recall cls1 | 100% |

### LOSO-V (Leave-One-Source-Out pe Veridica.ro)

| Metrică | Valoare |
|---|---|
| Recall cls1 | **29.35%** |
| Drop față de IID | **−70.65 pp** |

**Interpretare:** Performanța IID excepțională este parțial artefact de *stylistic
fingerprint* — modelul a învățat stilul editorial al surselor, nu exclusiv conținutul
dezinformativ. Evaluarea LOSO expune această limitare (prima documentare a fenomenului
pe fake news românesc).

## Dataset

Antrenat pe [`rares127/dezinformare-ro`](https://huggingface.co/datasets/rares127/dezinformare-ro):
1.483 articole (746 dezinformare / 737 credibile), perioada 2022–2026.

## Sistem complet (recomandat)

Modelul este integrat într-un pipeline de 5 module cu interfață web FastAPI.

[![GitHub](https://img.shields.io/badge/GitHub-detectie--dezinformare--ro-black?logo=github)](https://github.com/rares127/detectie-dezinformare-ro)

Codul sursă complet, scripturile de antrenare și evaluare LOSO sunt disponibile pe GitHub.

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
